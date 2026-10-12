from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from ..models import Transcript

# progress(fraction entre 0 et 1, message optionnel)
ProgressFn = Callable[[float, str | None], None]
CancelFn = Callable[[], bool]


class Cancelled(Exception):
    pass


class Backend(ABC):
    name: str
    model_name: str

    @abstractmethod
    def transcribe(
        self,
        wav_path: Path,
        duration: float,
        language: str | None,
        progress: ProgressFn,
        is_cancelled: CancelFn,
        multilingual: bool = False,
    ) -> Transcript:
        """Transcrit un WAV mono 16 kHz. `language=None` = détection automatique."""
