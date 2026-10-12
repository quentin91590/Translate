"""Export des transcriptions : TXT, SRT, VTT, JSON."""

from __future__ import annotations

import json

from .models import Transcript

FORMATS = ("txt", "srt", "vtt", "json")


def _ts(seconds: float, sep: str) -> str:
    ms = max(0, round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_txt(t: Transcript) -> str:
    return t.text + "\n"


def to_srt(t: Transcript) -> str:
    blocks = []
    for i, seg in enumerate((s for s in t.segments if s.text.strip()), start=1):
        blocks.append(f"{i}\n{_ts(seg.start, ',')} --> {_ts(seg.end, ',')}\n{seg.text.strip()}\n")
    return "\n".join(blocks)


def to_vtt(t: Transcript) -> str:
    lines = ["WEBVTT", ""]
    for seg in t.segments:
        if seg.text.strip():
            lines += [f"{_ts(seg.start, '.')} --> {_ts(seg.end, '.')}", seg.text.strip(), ""]
    return "\n".join(lines)


def to_json(t: Transcript) -> str:
    return json.dumps(t.to_dict(), ensure_ascii=False, indent=2)


def render(t: Transcript, fmt: str) -> str:
    return {"txt": to_txt, "srt": to_srt, "vtt": to_vtt, "json": to_json}[fmt](t)
