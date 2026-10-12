"""Configuration lue depuis les variables d'environnement."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # Dossier où sont stockés les fichiers envoyés et les transcriptions.
    data_dir: Path = field(default_factory=lambda: Path(_env("TRANSCRIBER_DATA_DIR", "data")))
    # Moteur par défaut : "whisper" (faster-whisper) ou "qwen" (Qwen3-ASR).
    default_backend: str = field(default_factory=lambda: _env("TRANSCRIBER_BACKEND", "whisper"))
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "large-v3"))
    qwen_model: str = field(default_factory=lambda: _env("QWEN_MODEL", "Qwen/Qwen3-ASR-1.7B"))
    # "auto", "cuda" ou "cpu".
    device: str = field(default_factory=lambda: _env("TRANSCRIBER_DEVICE", "auto"))
    # Type de calcul CTranslate2 ("auto" = float16 sur GPU, int8 sur CPU).
    compute_type: str = field(default_factory=lambda: _env("WHISPER_COMPUTE_TYPE", "auto"))
    # Durée maximale acceptée (secondes). 3h30 par défaut.
    max_duration: float = field(default_factory=lambda: float(_env("TRANSCRIBER_MAX_DURATION", str(3.5 * 3600))))
    # Taille des morceaux pour les moteurs qui ne gèrent pas l'audio long nativement.
    chunk_seconds: float = field(default_factory=lambda: float(_env("TRANSCRIBER_CHUNK_SECONDS", "240")))


def get_settings() -> Settings:
    return Settings()
