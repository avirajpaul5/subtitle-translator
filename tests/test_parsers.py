from __future__ import annotations

from pathlib import Path

import pytest

from subtitle_translator.parsers import (
    SubtitleParseError,
    decode_subtitle_bytes,
    parse_srt,
    parse_subtitle,
    serialize_subtitle,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_srt_roundtrip():
    content = _read("sample.srt")
    doc = parse_subtitle(content, ".srt")
    assert len(doc.cues) == 6
    assert doc.warnings == []
    assert doc.cues[2].settings == "X1:200 X2:600"
    assert doc.cues[1].text_lines == ["This line has <i>italic</i> text", "spanning two lines."]
    assert serialize_subtitle(doc).strip() == content.strip()


def test_vtt_roundtrip():
    content = _read("sample.vtt")
    doc = parse_subtitle(content, ".vtt")
    assert len(doc.cues) == 4
    assert doc.warnings == []
    assert doc.header_lines[0].startswith("WEBVTT")
    assert doc.cues[1].identifier == "intro"
    assert doc.cues[2].settings == "align:start"
    rendered = serialize_subtitle(doc)
    # Header preserved
    assert rendered.startswith("WEBVTT\nKind: captions\nLanguage: en\n")
    # NOTE block intentionally dropped on re-serialize but cues preserved
    assert "Hello and welcome to Dhaka." in rendered
    assert "The end." in rendered


def test_decode_handles_cp1252():
    raw = (FIXTURES / "sample_cp1252.srt").read_bytes()
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")
    text = decode_subtitle_bytes(raw)
    assert "That’s the end." in text
    doc = parse_srt(text)
    assert len(doc.cues) == 6


def test_decode_handles_utf8_bom():
    raw = _read("sample.srt").encode("utf-8-sig")
    assert raw[:3] == b"\xef\xbb\xbf"
    decoded = decode_subtitle_bytes(raw)
    assert not decoded.startswith("﻿")
    doc = parse_srt(decoded)
    assert len(doc.cues) == 6


def test_parse_srt_skips_bad_block_keeps_good():
    bad = (
        "1\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "Good cue one.\n"
        "\n"
        "2\n"
        "this is not a timing line\n"
        "Broken cue body.\n"
        "\n"
        "3\n"
        "00:00:05,000 --> 00:00:06,000\n"
        "Good cue two.\n"
    )
    doc = parse_srt(bad)
    assert [c.index for c in doc.cues] == [1, 3]
    assert doc.warnings, "expected a warning about the skipped block"


def test_parse_srt_empty_raises():
    with pytest.raises(SubtitleParseError):
        parse_srt("")
