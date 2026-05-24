"""End-to-end smoke that exercises the file-open auto-detection AND the
sentinel preservation path through the real IndicTrans2 model.

Uses generic synthetic cues exercising every corruption pattern the
production evaluation surfaced:
  - Stage directions in (CAPS): must translate, not preserve
  - Speaker labels (SPEAKER:): must extract, not preserve
  - Common nouns capitalized (Detective, Doctor): must translate
  - Pronouns spaCy mis-tags (US): must not preserve
  - Real proper names (Alice, Tokyo): must preserve verbatim
  - Foreign loans (bonjour): must preserve verbatim
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subtitle_translator.auto_dnt import detect_preserve_spans
from subtitle_translator.glossary import GlossaryConfig
from subtitle_translator.models import Cue, SubtitleDocument
from subtitle_translator.pipeline import TranslationSettings, translate_document
from subtitle_translator.translators.indictrans2 import IndicTrans2Translator

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "indictrans2-en-indic"


def _cue(i, lines):
    return Cue(index=i, start=f"00:00:0{i},000", end=f"00:00:0{i+1},000",
               text_lines=lines)


CUES_AND_EXPECTED_PRESERVE = [
    # Pure stage directions — nothing to preserve
    (_cue(1, ["(BELL TOLLING)"]),                                    []),
    (_cue(2, ["(CROWD CLAMORING)"]),                                 []),
    # Speaker label + dialogue with a real name
    (_cue(3, ["POIROT: Alice arrived at Tokyo."]),                   ["Alice", "Tokyo"]),
    # Common-noun capitalization (false-positive risk)
    (_cue(4, ["Detective Smith examined the body."]),                ["Smith"]),
    # Pronoun mis-tag risk
    (_cue(5, ["US, we should talk to Carol about it."]),             ["Carol"]),
    # Foreign loan + proper name
    (_cue(6, ["She whispered bonjour to Bob in Berlin."]),           ["bonjour", "Bob", "Berlin"]),
]
CUES = [c for c, _ in CUES_AND_EXPECTED_PRESERVE]
TARGETS = ["hi", "bn", "ta"]


def main() -> int:
    print(f"Loading model from {MODEL_PATH}...")
    translator = IndicTrans2Translator(str(MODEL_PATH), device="mps")
    doc = SubtitleDocument(format="srt", cues=CUES, header_lines=[])

    auto = detect_preserve_spans(doc)
    print(f"\nAuto-detected preserve list ({len(auto)}): {auto}")

    # Words from the evaluation that must NEVER appear in the preserve list.
    must_not_appear = {"BELL", "TOLLING", "CROWD", "CLAMORING", "MEN", "HORN",
                       "CAMERA", "POIROT", "Detective", "US", "Huh"}
    leaked = must_not_appear & set(auto)
    if leaked:
        print(f"  LEAKED into preserve: {leaked}")
        return 1
    print("  no stage-direction / speaker / common-word leaks ✓")

    glossary = GlossaryConfig(glossary_map={}, do_not_translate=auto)

    all_pass = True
    for tgt in TARGETS:
        print(f"\n=== en → {tgt} ===")
        settings = TranslationSettings(source_lang="en", target_lang=tgt)
        out = translate_document(doc, translator, settings, glossary)
        text = "\n".join(c.text for c in out.cues)
        print(text)

        # Validate no sentinel debris reached the output.
        debris_markers = ["ZZID", "ZZIT", "ID0Z", "ID1Z", "ID2Z", "ID3Z",
                          "জেড", "जेड"]
        for marker in debris_markers:
            if marker in text:
                print(f"  [DEBRIS] '{marker}' leaked into output")
                all_pass = False

        for cue, expected in CUES_AND_EXPECTED_PRESERVE:
            translated = out.cues[cue.index - 1].text
            for term in expected:
                if term in translated:
                    print(f"  [OK]   {term!r} preserved in cue {cue.index}")
                else:
                    print(f"  [MISS] {term!r} NOT in cue {cue.index}: {translated!r}")
                    all_pass = False

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
