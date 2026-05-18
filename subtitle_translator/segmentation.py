from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from subtitle_translator.models import Cue

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[।.!?。！？])\s+")


@dataclass
class MergedChunk:
    cue_indices: List[int]
    text: str


def merge_short_cues(cues: List[Cue], min_chars: int = 60) -> List[MergedChunk]:
    chunks: List[MergedChunk] = []
    pending_indices: List[int] = []
    pending_text_parts: List[str] = []

    for idx, cue in enumerate(cues):
        cue_text = cue.text.strip()
        pending_indices.append(idx)
        pending_text_parts.append(cue_text)

        current_len = len(" ".join(pending_text_parts))
        if current_len >= min_chars:
            chunks.append(MergedChunk(cue_indices=pending_indices[:], text="\n".join(pending_text_parts)))
            pending_indices.clear()
            pending_text_parts.clear()

    if pending_indices:
        chunks.append(MergedChunk(cue_indices=pending_indices[:], text="\n".join(pending_text_parts)))

    return chunks


def split_translated_chunk(translated_text: str, original_count: int) -> List[str]:
    lines = [line for line in translated_text.splitlines() if line.strip()]
    if not lines:
        return [""] * original_count

    if len(lines) == original_count:
        return lines

    flat = translated_text.replace("\n", " ").strip()
    rough_parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(flat) if part.strip()]
    if rough_parts and len(rough_parts) >= original_count:
        return _fit_count(rough_parts, original_count)

    return _fit_count(lines, original_count)


def _fit_count(parts: List[str], count: int) -> List[str]:
    if len(parts) == count:
        return parts
    if len(parts) < count:
        return parts + [""] * (count - len(parts))

    bucket_size = len(parts) / count
    out: List[str] = []
    for i in range(count):
        start = int(round(i * bucket_size))
        end = int(round((i + 1) * bucket_size))
        segment = " ".join(parts[start:end]).strip()
        out.append(segment)
    return out
