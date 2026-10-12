"""File d'attente des transcriptions, traitée par un thread de fond.

Les tâches sont traitées une par une (un gros modèle occupe tout le GPU).
L'état de chaque tâche est enregistré dans data/jobs/<id>/job.json, ce qui
permet de retrouver l'historique après un redémarrage.
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from . import pipeline
from .backends import Backend, Cancelled, get_backend
from .config import Settings

log = logging.getLogger(__name__)

BackendFactory = Callable[[str, Settings], Backend]


@dataclass
class Job:
    id: str
    filename: str
    backend: str
    language: str | None
    multilingual: bool
    status: str = "queued"  # queued | running | done | error | cancelled
    progress: float = 0.0
    message: str = "En attente"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    duration: float | None = None
    detected_language: str | None = None
    language_probability: float | None = None
    languages: dict[str, float] | None = None
    error: str | None = None
    cancel_requested: bool = False


class JobManager:
    def __init__(self, settings: Settings, backend_factory: BackendFactory = get_backend):
        self.settings = settings
        self.backend_factory = backend_factory
        self.root = settings.data_dir / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.queue: queue.Queue[str] = queue.Queue()
        self._load_existing()
        self.worker = threading.Thread(target=self._loop, daemon=True, name="transcriber-worker")
        self.worker.start()

    # --- persistance -----------------------------------------------------
    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _save(self, job: Job) -> None:
        path = self.job_dir(job.id) / "job.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(job), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _load_existing(self) -> None:
        for meta in sorted(self.root.glob("*/job.json")):
            try:
                job = Job(**json.loads(meta.read_text(encoding="utf-8")))
            except Exception:
                log.warning("Tâche illisible ignorée : %s", meta)
                continue
            if job.status in ("queued", "running"):
                # Interrompue par un redémarrage : on la remet dans la file.
                job.status, job.progress, job.message = "queued", 0.0, "En attente (reprise)"
                self.queue.put(job.id)
            self.jobs[job.id] = job
            self._save(job)

    # --- API publique ----------------------------------------------------
    def submit(self, upload_path: Path, filename: str, backend: str, language: str | None, multilingual: bool) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            filename=filename,
            backend=backend,
            language=language or None,
            multilingual=multilingual,
        )
        d = self.job_dir(job.id)
        d.mkdir(parents=True)
        shutil.move(str(upload_path), d / ("input" + Path(filename).suffix.lower()))
        with self.lock:
            self.jobs[job.id] = job
            self._save(job)
        self.queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> Job | None:
        job = self.jobs.get(job_id)
        if job and job.status in ("queued", "running"):
            with self.lock:
                job.cancel_requested = True
                if job.status == "queued":
                    job.status, job.message, job.finished_at = "cancelled", "Annulée", time.time()
                self._save(job)
        return job

    def delete(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job or job.status == "running":
            return False
        with self.lock:
            job.cancel_requested = True
            self.jobs.pop(job_id, None)
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
        return True

    def wait_idle(self, timeout: float = 30) -> None:
        """Attend que la file soit vide (utile pour les tests)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.queue.unfinished_tasks == 0:
                return
            time.sleep(0.05)
        raise TimeoutError("La file de transcription n'est pas vide.")

    # --- worker ----------------------------------------------------------
    def _loop(self) -> None:
        while True:
            job_id = self.queue.get()
            try:
                job = self.jobs.get(job_id)
                if job and job.status == "queued" and not job.cancel_requested:
                    self._process(job)
            finally:
                self.queue.task_done()

    def _process(self, job: Job) -> None:
        d = self.job_dir(job.id)
        src = next(d.glob("input*"), None)
        last_save = 0.0

        def progress(fraction: float, message: str | None) -> None:
            nonlocal last_save
            job.progress = round(max(job.progress, min(fraction, 1.0)), 4)
            if message:
                job.message = message
            if time.time() - last_save > 2:  # limite les écritures disque
                last_save = time.time()
                self._save(job)

        with self.lock:
            job.status, job.started_at, job.message = "running", time.time(), "Chargement du modèle…"
            self._save(job)
        try:
            if src is None:
                raise FileNotFoundError("Fichier source introuvable.")
            backend = self.backend_factory(job.backend, self.settings)
            transcript = pipeline.run(
                src,
                d,
                backend,
                language=job.language,
                multilingual=job.multilingual,
                max_duration=self.settings.max_duration,
                progress=progress,
                is_cancelled=lambda: job.cancel_requested,
            )
            job.status, job.progress = "done", 1.0
            job.duration = transcript.duration
            job.detected_language = transcript.language
            job.language_probability = transcript.language_probability
            job.languages = transcript.languages
            job.message = "Terminé"
        except Cancelled:
            job.status, job.message = "cancelled", "Annulée"
        except Exception as exc:
            log.error("Échec de la tâche %s :\n%s", job.id, traceback.format_exc())
            job.status, job.error, job.message = "error", str(exc), "Erreur"
        finally:
            job.finished_at = time.time()
            with self.lock:
                if job.id in self.jobs:
                    self._save(job)
