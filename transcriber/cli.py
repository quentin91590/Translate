"""Transcription en ligne de commande.

Exemple : python -m transcriber.cli interview.mp3 -o sorties/
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import pipeline
from .backends import BACKENDS, get_backend
from .config import get_settings


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    p = argparse.ArgumentParser(description="Transcrit un fichier audio/vidéo en texte (TXT, SRT, VTT, JSON).")
    p.add_argument("input", type=Path, help="fichier audio ou vidéo (mp3, wav, m4a, mp4, ogg, flac…)")
    p.add_argument("-o", "--output", type=Path, help="dossier de sortie (défaut : à côté du fichier)")
    p.add_argument("-b", "--backend", choices=list(BACKENDS), default=settings.default_backend)
    p.add_argument("-l", "--language", help="code langue (fr, en…) ; détection automatique si absent")
    p.add_argument("--multilingual", action="store_true", help="l'audio mélange plusieurs langues")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.input.exists():
        p.error(f"fichier introuvable : {args.input}")
    out_dir = args.output or args.input.parent / f"{args.input.stem}_transcription"

    t0 = time.time()

    def progress(fraction: float, message: str | None) -> None:
        bar = "#" * int(fraction * 30)
        sys.stderr.write(f"\r[{bar:<30}] {fraction:6.1%}  {message or ''}"[:120].ljust(120))
        sys.stderr.flush()

    backend = get_backend(args.backend, settings)
    transcript = pipeline.run(
        args.input, out_dir, backend, args.language, args.multilingual, settings.max_duration, progress
    )
    sys.stderr.write("\n")
    elapsed = time.time() - t0
    print(f"Langue : {transcript.language} | durée audio : {transcript.duration / 60:.1f} min "
          f"| temps de calcul : {elapsed / 60:.1f} min")
    print(f"Fichiers écrits dans : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
