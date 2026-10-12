"""Registre des moteurs de transcription, chargés à la demande puis gardés en mémoire."""

from __future__ import annotations

import threading

from ..config import Settings
from .base import Backend, Cancelled

BACKENDS = {
    "whisper": "Whisper large-v3 (OpenAI, MIT) — 99 langues, via faster-whisper",
    "qwen": "Qwen3-ASR 1.7B (Alibaba, Apache-2.0) — 52 langues/dialectes, GPU recommandé",
}

_cache: dict[tuple, Backend] = {}
_lock = threading.Lock()


def _cache_key(name: str, settings: Settings) -> tuple:
    """Tout ce qui change le modèle chargé : un autre réglage = un autre modèle."""
    if name == "whisper":
        return (name, settings.whisper_model, settings.device, settings.compute_type)
    return (name, settings.qwen_model, settings.device, settings.chunk_seconds)


def get_backend(name: str, settings: Settings) -> Backend:
    if name not in BACKENDS:
        raise ValueError(f"Moteur inconnu : {name}. Choix possibles : {', '.join(BACKENDS)}")
    key = _cache_key(name, settings)
    with _lock:
        if key not in _cache:
            # Un seul modèle lourd en mémoire à la fois (VRAM limitée).
            _cache.clear()
            if name == "whisper":
                from .whisper import WhisperBackend

                _cache[key] = WhisperBackend(settings.whisper_model, settings.device, settings.compute_type)
            else:
                from .qwen import QwenBackend

                _cache[key] = QwenBackend(settings.qwen_model, settings.device, settings.chunk_seconds)
        return _cache[key]


__all__ = ["BACKENDS", "Backend", "Cancelled", "get_backend"]
