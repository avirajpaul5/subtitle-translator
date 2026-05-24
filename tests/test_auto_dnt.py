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


# --- Regression tests against patterns from production evaluation -------
# Each case below was a real false positive observed in a Bengali translation
# of an English SRT — sound-effect words leaking into the preserve list,
# pronouns mis-tagged as places, common nouns capitalized at sentence start.
# All assertions are about behaviour, never specific film content.

def test_excludes_parenthesized_stage_directions():
    """Sound-effect / stage-direction content inside (…) should never go
    into the preserve list — those words must be translated semantically."""
    doc = _doc(
        "(BELL TOLLING)",
        "(CROWD CLAMORING)",
        "(MEN SHOUTING)",
        "(HORN HONKING)",
        "(CAMERA CLICKING)",
        "(DOOR SLAMS)",
    )
    detected = set(detect_preserve_spans(doc))
    leaked = {"BELL", "CROWD", "MEN", "HORN", "CAMERA", "DOOR",
              "TOLLING", "CLAMORING", "SHOUTING", "HONKING", "CLICKING", "SLAMS"}
    assert detected.isdisjoint(leaked)


def test_excludes_bracketed_stage_directions():
    """[ALL CAPS] style brackets are also stage directions."""
    doc = _doc("[MUSIC PLAYING] She walked away.")
    detected = set(detect_preserve_spans(doc))
    assert "MUSIC" not in detected
    assert "PLAYING" not in detected


def test_excludes_speaker_labels():
    """ALL-CAPS SPEAKER: prefixes are handled by the pipeline separately;
    they must not contribute to the preserve list."""
    doc = _doc(
        "POIROT: Good morning.",
        "CHIEF INSPECTOR: The case is closed.",
    )
    detected = set(detect_preserve_spans(doc))
    assert "POIROT" not in detected
    assert "CHIEF" not in detected
    assert "INSPECTOR" not in detected


def test_excludes_common_nouns_capitalized():
    """Common nouns that get PROPN-tagged due to capitalization (titles or
    sentence start) shouldn't slip into preserve list."""
    doc = _doc(
        "Detective arrived early.",
        "The Doctor examined the patient.",
        "Priest walked into the church.",
        "Huh, that is strange.",
    )
    detected_lower = {d.lower() for d in detect_preserve_spans(doc)}
    common_capitalised = {"detective", "doctor", "priest", "huh"}
    assert detected_lower.isdisjoint(common_capitalised)


def test_excludes_pronouns_misframed_as_entities():
    """spaCy parses standalone 'US' as GPE=United States; preserve-list
    must filter those out via the pronoun blocklist."""
    doc = _doc(
        "US, we should talk.",
        "He went home. She stayed.",
        "It was a long day.",
    )
    detected = set(detect_preserve_spans(doc))
    pronouns = {"US", "He", "She", "It", "Him", "Her", "Them"}
    assert detected.isdisjoint(pronouns)


def test_keeps_real_names_alongside_filtering():
    """The aggressive filtering above must not regress: real names in cues
    that also contain stage directions / pronouns should still be detected."""
    doc = _doc(
        "(BELL TOLLING)",
        "POIROT: Mary Debenham just arrived at Yokohama.",
        "US, we should ask Alice about it.",
    )
    detected = set(detect_preserve_spans(doc))
    assert "Mary Debenham" in detected
    assert "Yokohama" in detected
    assert "Alice" in detected


def test_html_formatting_tags_dont_leak():
    """Italic/bold tags get stripped before NER — the closing tag fragment
    must not produce a fake preserve entry."""
    doc = _doc("She said <i>au revoir</i> at the door.")
    detected = set(detect_preserve_spans(doc))
    leaked = {"i", "au revoir</i", "italic</i"}
    # The wrapping tag never becomes an entry; the foreign phrase is fine
    # though wordfreq may or may not catch it as a unit.
    assert all(t not in leaked for t in detected)


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
