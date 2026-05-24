"""Phase 2 smoke test: run the full pipeline with a do_not_translate list
against the real IndicTrans2 model and verify protected terms appear verbatim
in the output for Hindi, Bengali, Tamil.

Uses generic synthetic terms (placeholder personal names, a place name, and a
non-naturalized foreign phrase) so the test isn't tied to any specific film.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subtitle_translator.glossary import GlossaryConfig
from subtitle_translator.models import Cue, SubtitleDocument
from subtitle_translator.pipeline import TranslationSettings, translate_document
from subtitle_translator.translators.indictrans2 import IndicTrans2Translator

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "indictrans2-en-indic"


def _cue(i, lines):
    return Cue(index=i, start=f"00:00:0{i},000", end=f"00:00:0{i+1},000",
               text_lines=lines)


# Generic test inputs exercising the patterns we care about: a personal name,
# multiple names in one cue, a place name, and a non-naturalized foreign
# phrase. None tied to a specific movie.
CUES_AND_EXPECTED = [
    (_cue(1, ["I met Alice in Tokyo."]),                   ["Alice", "Tokyo"]),
    (_cue(2, ["Bob said bonjour at the station."]),        ["Bob", "bonjour"]),
    (_cue(3, ["Welcome to Berlin, madame."]),              ["Berlin"]),
    (_cue(4, ["Carol told Dave to meet at Yokohama."]),    ["Carol", "Dave", "Yokohama"]),
]
CUES = [c for c, _ in CUES_AND_EXPECTED]
PROTECTED = sorted({term for _, terms in CUES_AND_EXPECTED for term in terms})

TARGETS = ["hi", "bn", "ta"]


def main() -> int:
    print(f"Loading model from {MODEL_PATH}...")
    translator = IndicTrans2Translator(str(MODEL_PATH), device="mps")
    doc = SubtitleDocument(format="srt", cues=CUES, header_lines=[])
    glossary = GlossaryConfig(glossary_map={}, do_not_translate=PROTECTED)

    all_pass = True
    for tgt in TARGETS:
        print(f"\n=== en → {tgt} ===")
        settings = TranslationSettings(source_lang="en", target_lang=tgt)
        out = translate_document(doc, translator, settings, glossary)
        text = "\n".join(c.text for c in out.cues)
        print(text)

        for cue, expected_terms in CUES_AND_EXPECTED:
            translated = out.cues[cue.index - 1].text
            for term in expected_terms:
                survived = term in translated
                status = "OK" if survived else "MISS"
                print(f"  [{status}] {term!r} in cue {cue.index}")
                if not survived:
                    all_pass = False

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
