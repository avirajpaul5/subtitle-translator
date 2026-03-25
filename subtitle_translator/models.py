from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Cue:
    """A subtitle cue with timing and text lines."""

    index: Optional[int]
    start: str
    end: str
    text_lines: List[str]
    identifier: Optional[str] = None
    settings: Optional[str] = None

    @property
    def text(self) -> str:
        return "\n".join(self.text_lines)

    def with_text(self, text: str) -> "Cue":
        return Cue(
            index=self.index,
            start=self.start,
            end=self.end,
            text_lines=text.splitlines() or [""],
            identifier=self.identifier,
            settings=self.settings,
        )


@dataclass
class SubtitleDocument:
    format: str  # srt | vtt
    cues: List[Cue] = field(default_factory=list)
    header_lines: List[str] = field(default_factory=list)
