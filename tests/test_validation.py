"""Tests for the post-translation validation pass.

Uses synthetic strings and the echo backend — never invokes a real model.
"""
from __future__ import annotations

from subtitle_translator.glossary import GlossaryConfig
from subtitle_translator.models import Cue, SubtitleDocument
from subtitle_translator.pipeline import TranslationSettings, translate_document
from subtitle_translator.translators.echo import EchoTranslator
from subtitle_translator.validation import (
    flag_grammar_issues,
    has_corruption,
    validate_translation,
)


def _doc(*lines: str, **kw) -> SubtitleDocument:
    cues = [
        Cue(index=i + 1, start="00:00:00,000", end="00:00:01,000",
            text_lines=[line])
        for i, line in enumerate(lines)
    ]
    return SubtitleDocument(format="srt", cues=cues, header_lines=[], **kw)


# ---------------------------------------------------------------------------
# Corruption detection
# ---------------------------------------------------------------------------

def test_corruption_detects_sentinel_debris():
    assert has_corruption("She met ZZID3ZZ in Paris.")
    assert has_corruption("She met ID7ZZ at the door.")
    assert has_corruption("MENID4ZZ shouted.")


def test_corruption_detects_unclosed_paren():
    assert has_corruption("(BELL TOLLING")
    assert not has_corruption("(BELL TOLLING)")


def test_corruption_detects_bengali_letter_spellout():
    assert has_corruption("আজ জেড. জেড. লেগেছে।", target_lang="bn")
    assert has_corruption("সে আই. ডি. দিয়ে এসেছে।", target_lang="bn")
    assert has_corruption("357জেড পরে।", target_lang="bn")
    # Real V3 evaluation: model produces `( আইএন জেডজেড187জেডজেড )`
    # — IN + ZZZZ187ZZZZ in Bengali letters. Both halves trigger.
    assert has_corruption("( আইএন জেডজেড187জেডজেড )", target_lang="bn")


def test_corruption_detects_id_stripped_form():
    """`Monsieur 210ZZ` — model stripped ZZID prefix entirely. The trailing
    `<num>ZZ` is still recognisable debris."""
    assert has_corruption("Monsieur 210ZZ capitaine")
    assert has_corruption("Result: 42ZZ done.")
    # But normal text with numbers next to letters should not fire.
    assert not has_corruption("She bought 5 apples for $200.")


def test_corruption_detects_invalid_bengali_vowel_cluster():
    """Two consecutive Bengali vowel signs on one consonant — invalid.
    Real V3 evaluation example: `চিোকার` (চ + ি + ো + কার)."""
    assert has_corruption("চিোকার", target_lang="bn")
    assert has_corruption("আমি চিোকার বললাম।", target_lang="bn")
    # Single vowel sign per consonant is fine.
    assert not has_corruption("চিকার", target_lang="bn")
    assert not has_corruption("চোকার", target_lang="bn")


def test_corruption_detects_hindi_letter_spellout():
    assert has_corruption("वह जेड. जेड. कहा।", target_lang="hi")
    assert has_corruption("कोई आई. डी. पास नहीं।", target_lang="hi")


def test_corruption_clean_text_passes():
    assert not has_corruption("মেরি ডেবেনহাম স্টেশনে এসেছিলেন।", target_lang="bn")
    assert not has_corruption("She walked into the room.")


# ---------------------------------------------------------------------------
# Grammar flagging
# ---------------------------------------------------------------------------

def test_grammar_repeated_word_universal():
    assert "repeated_word" in flag_grammar_issues("The horn horn rang.")
    assert "repeated_word" not in flag_grammar_issues("The horn rang.")


def test_grammar_english_word_before_indic():
    # English token immediately before Bengali script — likely untranslated.
    assert "likely_untranslated_word" in flag_grammar_issues(
        "Doctor এসেছেন।", target_lang="bn"
    )


def test_grammar_bengali_subject_verb_mismatch():
    # "আমি একটি X ছিল" should be "ছিলাম" — common model error.
    assert "subject_verb_mismatch" in flag_grammar_issues(
        "আমি একটি ডাক্তার ছিল", target_lang="bn"
    )
    # Real V3 evaluation line 1054: multi-word noun phrase between
    # "একটি" and "ছিল".
    assert "subject_verb_mismatch" in flag_grammar_issues(
        "আমি একটি ভাল সপ্তাহ ছিল", target_lang="bn"
    )
    # Correct form (ছিলাম) should not fire.
    assert "subject_verb_mismatch" not in flag_grammar_issues(
        "আমি একটি ডাক্তার ছিলাম", target_lang="bn"
    )


def test_grammar_clean_lines_unflagged():
    assert flag_grammar_issues("সে দরজা খুলল।", target_lang="bn") == []
    assert flag_grammar_issues("She opened the door.") == []


# ---------------------------------------------------------------------------
# validate_translation entry point
# ---------------------------------------------------------------------------

def test_validate_translation_returns_one_issue_per_dirty_cue():
    orig = ["A clean line.", "Another clean line.", "Yet another."]
    trans = [
        "MENID3ZZ এসেছেন।",                  # corruption
        "একদম পরিষ্কার একটি বাক্য।",          # clean
        "horn horn went off।",               # repeated_word
    ]
    nums = [1, 2, 3]
    issues = validate_translation(orig, trans, nums, target_lang="bn")
    assert len(issues) == 2
    assert issues[0].cue_number == 1
    assert "sentinel_debris" in issues[0].issues
    assert issues[1].cue_number == 3
    assert "repeated_word" in issues[1].issues


def test_validate_translation_unknown_lang_falls_back_to_universal():
    """No language-specific patterns means only universal patterns fire."""
    orig = ["One.", "Two."]
    trans = ["clean line", "MENID3ZZ debris"]
    issues = validate_translation(orig, trans, [1, 2], target_lang="zz")
    assert len(issues) == 1
    assert issues[0].cue_number == 2


# ---------------------------------------------------------------------------
# Validation surfaces in SubtitleDocument.warnings
# ---------------------------------------------------------------------------

def test_pipeline_appends_validation_issues_to_warnings():
    """A cue whose translation contains debris must end up in warnings.
    Index 99999 is intentionally far beyond any sentinel restore_terms could
    have assigned, so the orphan slips past restoration and validation
    catches it."""
    doc = _doc("MENID99999ZZ entered the building.")
    out = translate_document(
        document=doc,
        translator=EchoTranslator(),
        settings=TranslationSettings(target_lang="bn"),
        glossary=GlossaryConfig(glossary_map={}, do_not_translate=[]),
    )
    assert any("cue 1" in w and "sentinel_debris" in w for w in out.warnings)


def test_pipeline_no_warnings_for_clean_input():
    doc = _doc("Welcome to the city.", "She walked home.")
    out = translate_document(
        document=doc,
        translator=EchoTranslator(),
        settings=TranslationSettings(target_lang="bn"),
        glossary=GlossaryConfig(glossary_map={}, do_not_translate=[]),
    )
    assert out.warnings == []


def test_pipeline_preserves_pre_existing_warnings():
    doc = _doc("Hello world.", warnings=["parser caveat: trailing blank line"])
    out = translate_document(
        document=doc,
        translator=EchoTranslator(),
        settings=TranslationSettings(target_lang="bn"),
        glossary=GlossaryConfig(glossary_map={}, do_not_translate=[]),
    )
    assert "parser caveat: trailing blank line" in out.warnings
