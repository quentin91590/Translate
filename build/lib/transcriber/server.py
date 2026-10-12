"""API HTTP + interface web."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .backends import BACKENDS
from .config import Settings, get_settings
from .formats import FORMATS
from .jobs import BackendFactory, JobManager

STATIC = Path(__file__).parent / "static"
MEDIA_TYPES = {"txt": "text/plain", "srt": "application/x-subrip", "vtt": "text/vtt", "json": "application/json"}


def _public(job) -> dict:
    d = asdict(job)
    d.pop("cancel_requested", None)
    return d


def create_app(settings: Settings | None = None, backend_factory: BackendFactory | None = None) -> FastAPI:
    settings = settings or get_settings()
    manager = JobManager(settings, backend_factory) if backend_factory else JobManager(settings)
    app = FastAPI(title="Transcripteur audio", version="1.0.0")
    app.state.manager = manager

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/config")
    def config() -> dict:
        return {
            "backends": BACKENDS,
            "default_backend": settings.default_backend,
            "max_duration": settings.max_duration,
            "formats": FORMATS,
        }

    @app.post("/api/jobs", status_code=201)
    async def create_job(
        file: UploadFile = File(...),
        backend: str = Form(None),
        language: str = Form(""),
        multilingual: bool = Form(False),
    ) -> dict:
        backend = backend or settings.default_backend
        if backend not in BACKENDS:
            raise HTTPException(400, f"Moteur inconnu : {backend}")
        tmp_dir = settings.data_dir / "uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        # Écriture par blocs : les fichiers de 3 h peuvent peser plusieurs centaines de Mo.
        with tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False) as out:
            while chunk := await file.read(4 * 1024 * 1024):
                out.write(chunk)
        filename = Path(file.filename or "audio").name
        job = manager.submit(Path(out.name), filename, backend, language.strip().lower() or None, multilingual)
        return _public(job)

    @app.get("/api/jobs")
    def list_jobs() -> list[dict]:
        return [_public(j) for j in manager.list()]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = manager.get(job_id)
        if not job:
            raise HTTPException(404, "Tâche introuvable")
        return _public(job)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict:
        job = manager.cancel(job_id)
        if not job:
            raise HTTPException(404, "Tâche introuvable")
        return _public(job)

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str) -> None:
        if not manager.delete(job_id):
            raise HTTPException(409, "Tâche introuvable ou en cours (annulez-la d'abord)")

    @app.get("/api/jobs/{job_id}/transcript.{fmt}")
    def download(job_id: str, fmt: str) -> FileResponse:
        job = manager.get(job_id)
        if fmt not in FORMATS or not job:
            raise HTTPException(404, "Introuvable")
        path = manager.job_dir(job_id) / f"transcript.{fmt}"
        if job.status != "done" or not path.exists():
            raise HTTPException(409, "Transcription pas encore disponible")
        name = f"{Path(job.filename).stem}.{fmt}"
        return FileResponse(path, media_type=MEDIA_TYPES[fmt] + "; charset=utf-8", filename=name)

    return app


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Serveur web de transcription audio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
