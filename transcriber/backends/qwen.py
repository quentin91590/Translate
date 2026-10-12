"""Moteur Qwen3-ASR (Alibaba Qwen, Apache-2.0) — 52 langues et dialectes.

Le modèle est appliqué morceau par morceau (~4 min, coupés dans les silences),
ce qui permet de traiter des enregistrements de plusieurs heures et de détecter
la langue de chaque morceau (utile pour l'audio multilingue).
Les horodatages des sous-titres sont estimés phrase par phrase à l'intérieur
de chaque morceau.
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path

from .. import audio
from ..models import Segment, Transcript
from .base import Backend, CancelFn, Cancelled, ProgressFn

log = logging.getLogger(__name__)

# Qwen3-ASR utilise des noms de langue en anglais ; on les relie aux codes ISO 639-1.
LANG_NAMES = {
    "ar": "Arabic", "cs": "Czech", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fa": "Persian", "fi": "Finnish", "fil": "Filipino",
    "fr": "French", "hi": "Hindi", "hu": "Hungarian", "id": "Indonesian", "it": "Italian",
    "ja": "Japanese", "ko": "Korean", "mk": "Macedonian", "ms": "Malay", "nl": "Dutch",
    "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sv": "Swedish",
    "th": "Thai", "tr": "Turkish", "vi": "Vietnamese", "yue": "Cantonese", "zh": "Chinese",
}
NAME_TO_CODE = {v.lower(): k for k, v in LANG_NAMES.items()}

_SENTENCE_RE = re.compile(r"(?<=[.!?。！？…])\s+")


def split_timed(text: str, start: float, end: float, max_chars: int = 160) -> list[tuple[float, float, str]]:
    """Découpe le texte d'un morceau en phrases avec horodatages proportionnels."""
    pieces: list[str] = []
    for sentence in _SENTENCE_RE.split(text.strip()):
        sentence = sentence.strip()
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)
    total = sum(len(p) for p in pieces) or 1
    out, t = [], start
    for p in pieces:
        dt = (end - start) * len(p) / total
        out.append((t, t + dt, p))
        t += dt
    return out


class QwenBackend(Backend):
    name = "qwen"

    def __init__(self, model_name: str = "Qwen/Qwen3-ASR-1.7B", device: str = "auto", chunk_seconds: float = 240):
        import torch
        from qwen_asr import Qwen3ASRModel

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device == "cuda":
            device = "cuda:0"
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        log.info("Chargement de %s sur %s…", model_name, device)
        self.model_name = model_name
        self.chunk_seconds = chunk_seconds
        self.model = Qwen3ASRModel.from_pretrained(
            model_name, dtype=dtype, device_map=device, max_inference_batch_size=8, max_new_tokens=4096
        )

    def transcribe(
        self,
        wav_path: Path,
        duration: float,
        language: str | None,
        progress: ProgressFn,
        is_cancelled: CancelFn,
        multilingual: bool = False,
    ) -> Transcript:
        progress(0.0, "Recherche des silences pour le découpage…")
        silences = audio.detect_silences(wav_path)
        chunks = audio.plan_chunks(duration, silences, self.chunk_seconds)
        forced = LANG_NAMES.get(language, language) if language else None

        segments: list[Segment] = []
        with tempfile.TemporaryDirectory() as tmp:
            for i, (start, end) in enumerate(chunks):
                if is_cancelled():
                    raise Cancelled()
                progress(start / duration, f"Morceau {i + 1}/{len(chunks)}")
                piece = audio.extract_segment(wav_path, Path(tmp) / f"{i}.wav", start, end)
                result = self.model.transcribe(audio=str(piece), language=forced)[0]
                lang_name = (getattr(result, "language", None) or "").strip()
                lang = NAME_TO_CODE.get(lang_name.lower(), lang_name or None)
                for s, e, text in split_timed(result.text or "", start, end):
                    segments.append(Segment(start=s, end=e, text=text, language=lang))

        # Langue principale = celle qui couvre le plus de temps de parole.
        t = Transcript(None, None, duration, self.name, self.model_name, segments)
        if language:
            t.language, t.language_probability = language, 1.0
        elif t.languages:
            t.language, t.language_probability = next(iter(t.languages.items()))
        return t
