"""Chaîne complète : fichier source → WAV 16 kHz → transcription → exports."""

from __future__ import annotations

from pathlib import Path

from . import audio, formats
from .backends import Backend
from .backends.base import CancelFn, ProgressFn
from .models import Transcript


def run(
    src: Path,
    workdir: Path,
    backend: Backend,
    language: str | None = None,
    multilingual: bool = False,
    max_duration: float | None = None,
    progress: ProgressFn = lambda f, m: None,
    is_cancelled: CancelFn = lambda: False,
) -> Transcript:
    workdir.mkdir(parents=True, exist_ok=True)
    progress(0.0, "Analyse du fichier…")
    duration = audio.probe_duration(src)
    if max_duration and duration > max_duration:
        raise audio.AudioError(
            f"Fichier trop long ({duration / 3600:.2f} h, maximum {max_duration / 3600:.2f} h)."
        )
    progress(0.0, "Conversion en WAV 16 kHz…")
    wav = audio.to_wav16k(src, workdir / "audio16k.wav")
    try:
        transcript = backend.transcribe(wav, duration, language or None, progress, is_cancelled, multilingual)
    finally:
        wav.unlink(missing_ok=True)
    for fmt in formats.FORMATS:
        (workdir / f"transcript.{fmt}").write_text(formats.render(transcript, fmt), encoding="utf-8")
    progress(1.0, "Terminé")
    return transcript
