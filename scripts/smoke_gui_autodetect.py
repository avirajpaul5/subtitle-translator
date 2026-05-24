"""Phase 5 smoke: exercise the file-open detection path without launching Qt.

Mirrors what gui.MainWindow._on_open does after parsing: run
detect_speaker_names + detect_preserve_spans, merge into a glossary blob,
and verify the resulting do_not_translate list is sensible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subtitle_translator.auto_dnt import detect_preserve_spans
from subtitle_translator.parsers import decode_subtitle_bytes, parse_subtitle
from subtitle_translator.speaker_detection import detect_speaker_names


def main(srt_path: str) -> int:
    p = Path(srt_path)
    print(f"Loading {p.name}")
    text = decode_subtitle_bytes(p.read_bytes())
    doc = parse_subtitle(text, p.suffix)
    print(f"  parsed {len(doc.cues)} cues")

    speakers = detect_speaker_names(doc)
    print(f"  speakers ({len(speakers)}): {speakers[:10]}{' …' if len(speakers) > 10 else ''}")

    auto_terms = detect_preserve_spans(doc)
    print(f"  auto-detected ({len(auto_terms)}): {auto_terms[:15]}{' …' if len(auto_terms) > 15 else ''}")

    merged = list(dict.fromkeys([*speakers, *auto_terms]))
    glossary = {"glossary": {}, "do_not_translate": merged}
    print(f"\nFinal do_not_translate ({len(merged)} terms):")
    print(json.dumps(glossary, ensure_ascii=False, indent=2)[:1200])

    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: smoke_gui_autodetect.py <path-to-srt>")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
