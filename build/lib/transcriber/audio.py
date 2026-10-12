"""Outils audio basés sur ffmpeg : normalisation, durée, découpage aux silences."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

SAMPLE_RATE = 16_000


class AudioError(RuntimeError):
    pass


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise AudioError(f"'{binary}' est introuvable : installez ffmpeg.")
    return path


def probe_duration(path: Path) -> float:
    """Durée du fichier en secondes (via ffprobe)."""
    out = subprocess.run(
        [_require("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise AudioError(f"Fichier audio illisible : {out.stderr.strip()}")
    try:
        return float(json.loads(out.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise AudioError("Impossible de déterminer la durée du fichier.") from exc


def to_wav16k(src: Path, dst: Path) -> Path:
    """Convertit n'importe quel format (mp3, m4a, mp4, ogg…) en WAV mono 16 kHz."""
    out = subprocess.run(
        [_require("ffmpeg"), "-nostdin", "-y", "-v", "error", "-i", str(src),
         "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst)],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise AudioError(f"Échec de la conversion audio : {out.stderr.strip()}")
    return dst


_SILENCE_RE = re.compile(r"silence_(start|end): (-?[\d.]+)")


def detect_silences(path: Path, noise_db: int = -35, min_silence: float = 0.4) -> list[tuple[float, float]]:
    """Renvoie les intervalles de silence (début, fin) détectés par ffmpeg."""
    out = subprocess.run(
        [_require("ffmpeg"), "-nostdin", "-v", "info", "-i", str(path),
         "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    silences: list[tuple[float, float]] = []
    start: float | None = None
    for kind, value in _SILENCE_RE.findall(out.stderr):
        if kind == "start":
            start = max(0.0, float(value))
        elif start is not None:
            silences.append((start, float(value)))
            start = None
    return silences


def plan_chunks(
    duration: float,
    silences: list[tuple[float, float]],
    target: float,
    tolerance: float = 0.25,
) -> list[tuple[float, float]]:
    """Découpe [0, duration] en morceaux d'environ `target` secondes.

    Chaque coupe est placée au milieu du silence le plus proche de la cible,
    dans une fenêtre de ±`tolerance`·target, pour ne pas couper un mot.
    Sans silence disponible, on coupe à la durée cible.
    """
    if duration <= target * (1 + tolerance):
        return [(0.0, duration)]
    mids = [(a + b) / 2 for a, b in silences]
    chunks: list[tuple[float, float]] = []
    pos = 0.0
    while duration - pos > target * (1 + tolerance):
        ideal = pos + target
        lo, hi = ideal - target * tolerance, ideal + target * tolerance
        candidates = [m for m in mids if lo <= m <= hi]
        cut = min(candidates, key=lambda m: abs(m - ideal)) if candidates else ideal
        chunks.append((pos, cut))
        pos = cut
    chunks.append((pos, duration))
    return chunks


def extract_segment(src: Path, dst: Path, start: float, end: float) -> Path:
    out = subprocess.run(
        [_require("ffmpeg"), "-nostdin", "-y", "-v", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
         "-i", str(src), "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst)],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise AudioError(f"Échec du découpage audio : {out.stderr.strip()}")
    return dst
