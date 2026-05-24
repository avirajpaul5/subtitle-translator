"""Calibration tests for the auto-detect preservation module.

Tests are property-based and use synthetic fixtures inline so they stay
independent of any specific movie or external SRT file.
"""
from __future__ import annotations

from pathlib import Path

from subtitle_translator.auto_dnt import (
    DEFAULT_ZIPF_THRESHOLD,
    detect_preserve_spans,
)
from subtitle_translator.models import Cue, SubtitleDocument
from subtitle_translator.parsers import parse_subtitle
from subtitle_translator.speaker_detection import _SKIP_WORDS

FIXTURES = Path(__file__).parent / "fixtures"


def _doc(*lines: str) -> SubtitleDocument:
    cues = [
        Cue(index=i + 1, start="00:00:00,000", end="00:00:01,000",
            text_lines=[line])
        for i, line in enumerate(lines)
    ]
    return SubtitleDocument(format="srt", cues=cues, header_lines=[])


def test_detects_proper_names_and_places():
    doc = _doc(
        "I met Alice in Tokyo.",
        "Bob said hello at the station.",
        "Welcome to Berlin.",
    )
    detected = set(detect_preserve_spans(doc))
    assert {"Alice", "Bob", "Tokyo", "Berlin"} <= detected


def test_detects_non_naturalized_foreign_words():
    doc = _doc(
        "She whispered bonjour.",
        "He asked for oeufs at the cafe.",
    )
    detected = set(detect_preserve_spans(doc))
    assert {"bonjour", "oeufs"} <= detected


def test_excludes_common_english_words():
    doc = _doc(
        "The cat sat on the mat and stared at the door.",
        "She walked quickly to the station and bought a ticket.",
    )
    detected = set(detect_preserve_spans(doc))
    common = {"the", "cat", "sat", "on", "mat", "and", "stared",
              "at", "door", "she", "walked", "quickly", "to",
              "bought", "a", "ticket"}
    # Compare lower-case so capitalised sentence starts don't slip through.
    detected_lower = {d.lower() for d in detected}
    assert detected_lower.isdisjoint(common)


def test_excludes_skip_words():
    """Words in the speaker-detection skip list never appear in output."""
    # Construct sentences using each skip word in a position where it might
    # otherwise be flagged.
    skip_sample = " ".join(w.lower() for w in list(_SKIP_WORDS)[:10])
    doc = _doc(f"Speaker says {skip_sample} loudly.")
    detected = {d.upper() for d in detect_preserve_spans(doc)}
    assert detected.isdisjoint(_SKIP_WORDS)


def test_multi_word_entities_kept_as_phrase():
    doc = _doc("She visited New York City last summer.")
    detected = detect_preserve_spans(doc)
    assert "New York City" in detected
    # The constituent tokens shouldn't *also* be returned as standalone
    # entries when they're already covered by the multi-word span.
    assert "New" not in detected
    assert "York" not in detected
    assert "City" not in detected


def test_output_sorted_longest_first():
    """`protect_terms` relies on length-descending order so multi-word
    phrases get substituted before their substrings."""
    doc = _doc(
        "Tokyo is busy.",
        "She lives in New York City.",
        "He met Alice.",
    )
    detected = detect_preserve_spans(doc)
    lengths = [len(t) for t in detected]
    assert lengths == sorted(lengths, reverse=True)


def test_output_is_deterministic():
    doc = _doc(
        "Alice met Bob in Tokyo.",
        "He whispered bonjour.",
    )
    a = detect_preserve_spans(doc)
    b = detect_preserve_spans(doc)
    assert a == b


def test_zipf_threshold_separates_common_from_rare():
    """Verify the calibrated default threshold puts common English words
    above and clearly non-naturalized foreign loans below.

    Borderline cases (monsieur, café, etc.) are intentionally NOT tested —
    those are dictionary-entered loanwords whose status is a design call.
    """
    from wordfreq import zipf_frequency
    common = ["said", "station", "welcome", "people", "walked"]
    rare = ["bonjour", "oeufs", "schadenfreude"]
    for w in common:
        assert zipf_frequency(w, "en") >= DEFAULT_ZIPF_THRESHOLD, w
    for w in rare:
        assert zipf_frequency(w, "en") < DEFAULT_ZIPF_THRESHOLD, w


# --- Sanity pass against the in-repo sample SRT --------------------------
# Property-based: never assert specific terms (the fixture's content may
# evolve), only structural invariants.

def test_sample_srt_runs_cleanly():
    text = (FIXTURES / "sample.srt").read_text(encoding="utf-8")
    doc = parse_subtitle(text, ".srt")
    detected = detect_preserve_spans(doc)
    assert isinstance(detected, list)
    assert len(detected) >= 1
    # Every detected token must be protectable (no empty strings, no
    # skip-words, no pure-punctuation).
    for term in detected:
        assert term.strip() == term
        assert term.upper() not in _SKIP_WORDS
        assert any(c.isalpha() for c in term)
