from __future__ import annotations

import re
from typing import List

from subtitle_translator.models import SubtitleDocument

_SPEAKER_RE = re.compile(r"^\s*([A-Z][A-Z\s\-\.\']{1,30}):", re.MULTILINE)

# Single words that are common English or stage-direction labels, not character names
_SKIP_WORDS = {
    "A", "AN", "THE", "AND", "OR", "BUT", "NOT", "NO", "YES", "OK", "OH",
    "AH", "HM", "UP", "DOWN", "IN", "OUT", "ON", "OFF", "MR", "MRS", "DR",
    "MS", "NOTE", "INT", "EXT", "CUT", "FADE", "END", "SCENE", "ACT",
    "SUBTITLES", "TRANSLATED", "CHAPTER",
}


def detect_speaker_names(document: SubtitleDocument) -> List[str]:
    """Return sorted list of unique speaker names found in subtitle cues.

    Scans each cue for lines matching 'SPEAKER NAME: ...' (all-caps) and
    collects the names. Common English words and stage-direction labels are
    filtered out.
    """
    names: set[str] = set()
    for cue in document.cues:
        for match in _SPEAKER_RE.finditer(cue.text):
            name = match.group(1).strip()
            # Must be at least 2 chars and not a known non-name label
            if len(name) >= 2 and name not in _SKIP_WORDS:
                names.add(name)
    return sorted(names)
