"""Tests for the per-target-language defaults merge.

Uses the echo backend — no model needed.
"""
from __future__ import annotations

from subtitle_translator.defaults import (
    PER_LANG_GLOSSARY,
    UNIVERSAL_DNT,
    get_default_glossary_for,
    merge_with_defaults,
)
from subtitle_translator.glossary import GlossaryConfig
from subtitle_translator.models import Cue, SubtitleDocument
from subtitle_translator.pipeline import TranslationSettings, translate_document
from subtitle_translator.translators.echo import EchoTranslator


def _doc(*lines: str) -> SubtitleDocument:
    cues = [
        Cue(index=i + 1, start="00:00:00,000", end="00:00:01,000",
            text_lines=[line])
        for i, line in enumerate(lines)
    ]
    return SubtitleDocument(format="srt", cues=cues, header_lines=[])


# ---------------------------------------------------------------------------
# Per-language glossary defaults
# ---------------------------------------------------------------------------

def test_bengali_defaults_populated():
    bn = get_default_glossary_for("bn")
    # A few high-confidence anchors — full coverage isn't worth pinning.
    assert bn.get("doctor") == "ডাক্তার"
    assert bn.get("priest") == "পুরোহিত"
    assert bn.get("english") == "ইংরেজি"


def test_unknown_target_lang_returns_empty_glossary():
    assert get_default_glossary_for("zz") == {}


def test_universal_dnt_contains_foreign_phrases():
    # Sample assertions — full list documented in defaults.py.
    assert "Monsieur" in UNIVERSAL_DNT
    assert "mon ami" in UNIVERSAL_DNT
    assert "Habibi" in UNIVERSAL_DNT


# ---------------------------------------------------------------------------
# merge_with_defaults
# ---------------------------------------------------------------------------

def test_merge_user_glossary_overrides_default():
    merged_map, _ = merge_with_defaults(
        user_map={"doctor": "চিকিৎসক"},  # custom Bengali for doctor
        user_dnt=[],
        target_lang="bn",
    )
    assert merged_map["doctor"] == "চিকিৎসক"  # user wins
    assert merged_map["priest"] == "পুরোহিত"   # default carries through


def test_merge_dnt_unions_without_duplicates():
    _, merged_dnt = merge_with_defaults(
        user_map={},
        user_dnt=["Monsieur", "Custom-Term"],  # Monsieur already in universal
        target_lang="bn",
    )
    # Case-insensitive dedup, but order preserved.
    assert merged_dnt.count("Monsieur") == 1
    assert "Custom-Term" in merged_dnt


def test_merge_isolated_per_language():
    """Bengali defaults must NOT leak into a Hindi translation request."""
    bn_map, _ = merge_with_defaults({}, [], target_lang="bn")
    hi_map, _ = merge_with_defaults({}, [], target_lang="hi")
    # Bengali map has entries; Hindi map only carries the empty default.
    assert "doctor" in bn_map
    # Hindi has no per-lang entries yet.
    assert hi_map == {}


# ---------------------------------------------------------------------------
# Pipeline-level integration: defaults applied to actual translation output
# ---------------------------------------------------------------------------

def test_pipeline_applies_bengali_default_post_translation():
    """EchoTranslator returns 'doctor' verbatim; the default glossary
    rewrite must turn it into ডাক্তার."""
    doc = _doc("The doctor arrived early.")
    out = translate_document(
        document=doc,
        translator=EchoTranslator(),
        settings=TranslationSettings(target_lang="bn"),
        glossary=GlossaryConfig(glossary_map={}, do_not_translate=[]),
    )
    text = out.cues[0].text
    assert "ডাক্তার" in text
    assert "doctor" not in text.lower()


def test_pipeline_user_glossary_overrides_default():
    doc = _doc("The doctor arrived.")
    out = translate_document(
        document=doc,
        translator=EchoTranslator(),
        settings=TranslationSettings(target_lang="bn"),
        glossary=GlossaryConfig(
            glossary_map={"doctor": "চিকিৎসক"},  # user override
            do_not_translate=[],
        ),
    )
    assert "চিকিৎসক" in out.cues[0].text
    assert "ডাক্তার" not in out.cues[0].text


def test_pipeline_universal_dnt_protects_foreign_phrase():
    """`Monsieur` should reach the output unchanged via the universal DNT."""
    doc = _doc("Monsieur Poirot arrived.")
    out = translate_document(
        document=doc,
        translator=EchoTranslator(),
        settings=TranslationSettings(target_lang="bn"),
        glossary=GlossaryConfig(glossary_map={}, do_not_translate=[]),
    )
    # EchoTranslator round-trips, so the sentinel must restore to "Monsieur".
    assert "Monsieur" in out.cues[0].text


def test_pipeline_non_bengali_target_skips_bengali_defaults():
    """Hindi target must NOT pick up Bengali default substitutions."""
    doc = _doc("The doctor arrived.")
    out = translate_document(
        document=doc,
        translator=EchoTranslator(),
        settings=TranslationSettings(target_lang="hi"),
        glossary=GlossaryConfig(glossary_map={}, do_not_translate=[]),
    )
    # Echo backend → English passes through; no Bengali default applies.
    assert "ডাক্তার" not in out.cues[0].text
    assert "doctor" in out.cues[0].text.lower()
