"""Moteur Whisper large-v3 via faster-whisper (CTranslate2).

Gère nativement l'audio long (fenêtre glissante de 30 s), filtre les silences
avec Silero VAD et détecte la langue parmi 99 langues.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..models import Segment, Transcript
from .base import Backend, CancelFn, Cancelled, ProgressFn

log = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import ctranslate2

        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:  # pragma: no cover - dépend de l'installation
        return "cpu"


class WhisperBackend(Backend):
    name = "whisper"

    def __init__(self, model_name: str = "large-v3", device: str = "auto", compute_type: str = "auto"):
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        self.model_name = model_name
        self.device = _resolve_device(device)
        if compute_type == "auto":
            compute_type = "float16" if self.device == "cuda" else "int8"
        log.info("Chargement de Whisper %s sur %s (%s)…", model_name, self.device, compute_type)
        self.model = WhisperModel(model_name, device=self.device, compute_type=compute_type)
        # Sur GPU, l'inférence par lots est plusieurs fois plus rapide pour les longs fichiers.
        self.batched = BatchedInferencePipeline(self.model) if self.device == "cuda" else None

    def transcribe(
        self,
        wav_path: Path,
        duration: float,
        language: str | None,
        progress: ProgressFn,
        is_cancelled: CancelFn,
        multilingual: bool = False,
    ) -> Transcript:
        progress(0.0, "Détection de la langue…")
        options = dict(
            language=language,
            task="transcribe",
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            # Évite les boucles de répétition sur les enregistrements de plusieurs heures.
            condition_on_previous_text=False,
            multilingual=multilingual,
            # Détection de langue sur plusieurs fenêtres de 30 s : plus fiable
            # si l'enregistrement commence par de la musique ou du silence.
            language_detection_segments=4,
        )
        if self.batched is not None:
            segments_iter, info = self.batched.transcribe(str(wav_path), batch_size=16, **options)
        else:
            segments_iter, info = self.model.transcribe(str(wav_path), **options)

        total = info.duration or duration or 1.0
        lang_msg = f"Langue détectée : {info.language} ({info.language_probability:.0%})"
        progress(0.0, lang_msg)

        segments: list[Segment] = []
        for seg in segments_iter:
            if is_cancelled():
                raise Cancelled()
            segments.append(Segment(start=seg.start, end=seg.end, text=seg.text.strip()))
            progress(min(seg.end / total, 0.999), lang_msg)

        return Transcript(
            language=info.language,
            language_probability=round(info.language_probability, 4),
            duration=total,
            backend=self.name,
            model=self.model_name,
            segments=segments,
        )
