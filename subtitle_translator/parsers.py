from __future__ import annotations

import re
from typing import List

from subtitle_translator.models import Cue, SubtitleDocument

TIMING_RE = re.compile(r"^(?P<start>[^\s]+)\s+-->\s+(?P<end>[^\s]+)(?P<settings>.*)$")


class SubtitleParseError(ValueError):
    pass


def decode_subtitle_bytes(data: bytes) -> str:
    """Decode subtitle file bytes, tolerating common non-UTF-8 encodings.

    Real-world .srt files frequently arrive in cp1252 or latin-1 (Windows tooling)
    rather than UTF-8. We try a small priority list; latin-1 never raises so it is
    the guaranteed fallback.
    """
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16")
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def parse_subtitle(content: str, file_ext: str) -> SubtitleDocument:
    ext = file_ext.lower().strip(".")
    if ext == "srt":
        return parse_srt(content)
    if ext == "vtt":
        return parse_vtt(content)
    raise SubtitleParseError(f"Unsupported subtitle format: {file_ext}")


def parse_srt(content: str) -> SubtitleDocument:
    blocks = re.split(r"\n\s*\n", content.lstrip("﻿").strip().replace("\r\n", "\n"))
    cues: List[Cue] = []
    warnings: List[str] = []

    for block_num, block in enumerate(blocks, start=1):
        lines = block.split("\n")
        if not any(line.strip() for line in lines):
            continue

        idx = None
        pointer = 0
        if lines[0].strip().isdigit():
            idx = int(lines[0].strip())
            pointer = 1

        if pointer >= len(lines):
            warnings.append(f"Block {block_num}: missing timing line, skipped")
            continue

        timing_match = TIMING_RE.match(lines[pointer].strip())
        if not timing_match:
            warnings.append(
                f"Block {block_num}: invalid timing line {lines[pointer]!r}, skipped"
            )
            continue
        pointer += 1
        text_lines = lines[pointer:] or [""]

        cues.append(
            Cue(
                index=idx,
                start=timing_match.group("start"),
                end=timing_match.group("end"),
                text_lines=text_lines,
                settings=timing_match.group("settings").strip() or None,
            )
        )

    if not cues:
        raise SubtitleParseError("No cues found in SRT file")

    return SubtitleDocument(format="srt", cues=cues, warnings=warnings)


def parse_vtt(content: str) -> SubtitleDocument:
    normalized = content.lstrip("﻿").replace("\r\n", "\n")
    lines = normalized.split("\n")
    if not lines or not lines[0].strip().startswith("WEBVTT"):
        raise SubtitleParseError("VTT content must start with WEBVTT")

    header_lines: List[str] = [lines[0].rstrip("\n")]
    pointer = 1
    while pointer < len(lines) and lines[pointer].strip() != "":
        header_lines.append(lines[pointer])
        pointer += 1

    body = "\n".join(lines[pointer:]).strip()
    blocks = re.split(r"\n\s*\n", body) if body else []

    cues: List[Cue] = []
    warnings: List[str] = []

    for block_num, block in enumerate(blocks, start=1):
        block_lines = block.split("\n")
        if not block_lines:
            continue
        if block_lines[0].startswith("NOTE"):
            continue

        identifier = None
        timing_line_idx = 0
        timing_match = TIMING_RE.match(block_lines[0].strip())
        if not timing_match and len(block_lines) > 1:
            identifier = block_lines[0]
            timing_line_idx = 1
            timing_match = TIMING_RE.match(block_lines[1].strip())

        if not timing_match:
            warnings.append(f"Block {block_num}: missing timing line, skipped")
            continue

        text_lines = block_lines[timing_line_idx + 1 :] or [""]
        cues.append(
            Cue(
                index=None,
                identifier=identifier,
                start=timing_match.group("start"),
                end=timing_match.group("end"),
                settings=timing_match.group("settings").strip() or None,
                text_lines=text_lines,
            )
        )

    if not cues:
        raise SubtitleParseError("No cues found in VTT file")

    return SubtitleDocument(
        format="vtt", cues=cues, header_lines=header_lines, warnings=warnings
    )


def serialize_subtitle(document: SubtitleDocument) -> str:
    if document.format == "srt":
        return _serialize_srt(document)
    if document.format == "vtt":
        return _serialize_vtt(document)
    raise ValueError(f"Unsupported subtitle format: {document.format}")


def _serialize_srt(document: SubtitleDocument) -> str:
    blocks: List[str] = []
    for i, cue in enumerate(document.cues, start=1):
        idx = cue.index if cue.index is not None else i
        timing = f"{cue.start} --> {cue.end}"
        if cue.settings:
            timing = f"{timing} {cue.settings}"
        block = "\n".join([str(idx), timing, *cue.text_lines])
        blocks.append(block)
    return "\n\n".join(blocks).strip() + "\n"


def _serialize_vtt(document: SubtitleDocument) -> str:
    blocks: List[str] = []
    for cue in document.cues:
        lines: List[str] = []
        if cue.identifier:
            lines.append(cue.identifier)
        timing = f"{cue.start} --> {cue.end}"
        if cue.settings:
            timing = f"{timing} {cue.settings}"
        lines.append(timing)
        lines.extend(cue.text_lines)
        blocks.append("\n".join(lines))

    header = "\n".join(document.header_lines) if document.header_lines else "WEBVTT"
    return f"{header}\n\n" + "\n\n".join(blocks).strip() + "\n"
