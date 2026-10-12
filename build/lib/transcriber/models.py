"""Structures de données communes à tous les moteurs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Segment:
    start: float
    end: float
    text: str
    language: str | None = None


@dataclass
class Transcript:
    language: str | None
    language_probability: float | None
    duration: float
    backend: str
    model: str
    segments: list[Segment] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def languages(self) -> dict[str, float]:
        """Part du temps de parole par langue (utile pour l'audio multilingue)."""
        totals: dict[str, float] = {}
        for s in self.segments:
            lang = s.language or self.language
            if lang:
                totals[lang] = totals.get(lang, 0.0) + max(0.0, s.end - s.start)
        total = sum(totals.values()) or 1.0
        return {k: round(v / total, 3) for k, v in sorted(totals.items(), key=lambda kv: -kv[1])}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["text"] = self.text
        d["languages"] = self.languages
        return d
