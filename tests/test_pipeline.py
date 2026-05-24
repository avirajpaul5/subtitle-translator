from __future__ import annotations

from pathlib import Path

from subtitle_translator.glossary import (
    GlossaryConfig,
    protect_terms,
    restore_terms,
)
from subtitle_translator.parsers import parse_subtitle
from subtitle_translator.pipeline import TranslationSettings, translate_document
from subtitle_translator.translators.echo import EchoTranslator

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str):
    return parse_subtitle((FIXTURES / name).read_text(encoding="utf-8"), Path(name).suffix)


def test_translate_document_echo_preserves_structure():
    doc = _load("sample.srt")
    translated = translate_document(
        document=doc,
        translator=EchoTranslator(),
        settings=TranslationSettings(),
        glossary=GlossaryConfig(glossary_map={}, do_not_translate=[]),
    )
    assert len(translated.cues) == len(doc.cues)
    for original, out in zip(doc.cues, translated.cues):
        assert out.index == original.index
        assert out.start == original.start
        assert out.end == original.end
        assert out.settings == original.settings


def test_translate_document_applies_glossary():
    doc = _load("sample.srt")
    glossary = GlossaryConfig(
        glossary_map={"Dhaka": "DHK_TOKEN"},
        do_not_translate=[],
    )
    translated = translate_document(
        document=doc,
        translator=EchoTranslator(),
        settings=TranslationSettings(),
        glossary=glossary,
    )
    combined = "\n".join(c.text for c in translated.cues)
    assert "DHK_TOKEN" in combined
    assert "Dhaka" not in combined


def test_translate_document_protects_do_not_translate():
    doc = _load("sample.srt")
    glossary = GlossaryConfig(
        glossary_map={},
        do_not_translate=["OpenAI", "Python"],
    )
    translated = translate_document(
        document=doc,
        translator=EchoTranslator(),
        settings=TranslationSettings(),
        glossary=glossary,
    )
    combined = "\n".join(c.text for c in translated.cues)
    assert "OpenAI" in combined
    assert "Python" in combined


# --- Sentinel restoration regression tests -------------------------------
# Each input below corresponds to a corruption mode observed in a real
# Bengali translation. The expected output is "the original term substituted
# cleanly back in place, with no sentinel debris left behind".

def _round_trip(terms, model_output):
    """Helper: produce a replacements map for `terms`, then run
    restore_terms on `model_output` to see what would be shown to the user."""
    _, replacements = protect_terms(["dummy"], terms)
    return restore_terms([model_output], replacements)[0]


def test_restore_clean_sentinel():
    assert _round_trip(["Poirot"], "I met ZZID0ZZ in Paris.") \
        == "I met Poirot in Paris."


def test_restore_id_stripped_form():
    """Real V3 evaluation: `Monsieur 210ZZ` — model dropped 'ZZID' entirely.
    We still recognise the trailing `<num>ZZ` and substitute back."""
    # Force the sentinel for the second term to be ZZID1ZZ ("Bonsoir").
    out = _round_trip(["Monsieur", "Bonsoir"], "Monsieur 1ZZ capitaine")
    assert "Bonsoir" in out
    assert "1ZZ" not in out


def test_restore_id_stripped_orphan_stripped():
    """An ID-stripped sentinel with an index we never assigned must be
    removed rather than left as `999ZZ` in the output."""
    out = _round_trip(["Alice"], "She met 999ZZ at the door.")
    assert "999ZZ" not in out
    assert "999" not in out


def test_restore_eaten_trailing_z():
    """Real evaluation case: `BELLID114Z` — only one trailing Z survived."""
    assert _round_trip(["Bell"], "(ZZID0Z TOLLING)") \
        .startswith("(") and "Bell" in _round_trip(["Bell"], "(ZZID0Z TOLLING)")


def test_restore_d_to_t_letter_swap():
    """Model occasionally outputs ZZIT0ZZ instead of ZZID0ZZ."""
    assert _round_trip(["Poirot"], "He met ZZIT0ZZ.") == "He met Poirot."


def test_restore_prefix_fused_single_word():
    """Real evaluation case: `MENID209ZZ` — model re-injected the source
    word and fused it to the sentinel."""
    assert _round_trip(["Suspect"], "SuspectID0ZZ arrived.") \
        == "Suspect arrived."


def test_restore_prefix_fused_multiword_last_word():
    """Real evaluation case: `Mary DebenhamID155ZZ` for term `Mary Debenham`.
    The 'Mary ' is already in the output, so we must not duplicate it —
    only the fused last word should be emitted."""
    out = _round_trip(["Mary Debenham"], "Welcome Mary DebenhamID0ZZ.")
    assert "Mary Debenham" in out
    assert "Mary Mary" not in out


def test_restore_bengali_letter_spellout():
    """Real evaluation case: `( জেড. জেড. আই. ডি. 357জেড.` — Bengali decoder
    spelled the sentinel out letter-by-letter."""
    out = _round_trip(["Poirot"], "( জেড. জেড. আই. ডি. 0 জেড. জেড. বললেন)")
    assert "Poirot" in out
    assert "জেড" not in out
    assert "আই" not in out
    assert "ডি" not in out


def test_restore_hindi_letter_spellout():
    out = _round_trip(["Poirot"], "वह जेड. जेड. आई. डी. 0 जेड. जेड. से मिले")
    assert "Poirot" in out
    assert "जेड" not in out


def test_restore_orphan_id_with_no_match():
    """Sentinel for an index we never assigned must NOT survive into the
    user-facing output — it gets stripped."""
    out = _round_trip(["Alice"], "She met ZZID0ZZ and ZZID7ZZ.")
    assert "Alice" in out
    assert "ZZID7ZZ" not in out
    assert "ID7" not in out


def test_restore_doesnt_eat_real_english_words():
    """Words like 'kid3' or 'Sid8' would superficially match an unguarded
    regex. With leading-or-trailing Z required, they shouldn't fire."""
    out = _round_trip(["Carol"], "The kid3 ran past Sid8 to ZZID0ZZ.")
    assert "kid3" in out
    assert "Sid8" in out
    assert "Carol" in out


def test_protect_pads_sentinels_with_spaces():
    """The model is much less likely to fuse a sentinel with its neighbours
    when there's whitespace on both sides — verify the substitution always
    has a space (or punctuation) bounding it."""
    protected, _ = protect_terms(
        ["I met Poirot,Bouc here."],
        ["Poirot", "Bouc"],
    )
    out = protected[0]
    # Both sentinels should be present, each surrounded by whitespace.
    assert "ZZID0ZZ" in out
    assert "ZZID1ZZ" in out
    # The pre-existing comma-attachment shouldn't fuse the sentinel.
    assert "ZZID0ZZ,ZZID1ZZ" not in out and "ZZID1ZZ,ZZID0ZZ" not in out


def test_speaker_label_not_word_substituted_by_default_glossary():
    """Real V3 regression: `CHIEF INSPECTOR:` became `CHIEF পরিদর্শক:`
    because the Bengali default glossary contains `inspector → পরিদর্শক`
    and the old pipeline did word-level substitution on speaker labels.

    Speaker labels must only be replaced wholesale (full-name match)."""
    fixture = FIXTURES / "sample.srt"
    raw = fixture.read_text(encoding="utf-8")
    # Inject a cue with a multi-word speaker label.
    raw_with_label = raw + (
        "\n7\n00:00:20,000 --> 00:00:22,000\n"
        "CHIEF INSPECTOR: Make way please.\n"
    )
    from subtitle_translator.parsers import parse_subtitle
    doc = parse_subtitle(raw_with_label, ".srt")
    translated = translate_document(
        document=doc,
        translator=EchoTranslator(),
        settings=TranslationSettings(target_lang="bn"),
        glossary=GlossaryConfig(glossary_map={}, do_not_translate=[]),
    )
    # The label "CHIEF INSPECTOR:" must remain intact — neither "CHIEF"
    # nor "INSPECTOR" should be partially translated.
    last = translated.cues[-1].text
    assert "CHIEF INSPECTOR" in last
    assert "পরিদর্শক:" not in last  # word-level substitution would produce this


def test_speaker_label_translated_only_on_full_name_match():
    """If the user explicitly maps `POIROT` → `পয়রট`, the label still
    gets translated wholesale; word-level glossary entries are ignored."""
    fixture = FIXTURES / "sample.srt"
    raw = fixture.read_text(encoding="utf-8")
    raw_with_label = raw + (
        "\n7\n00:00:20,000 --> 00:00:22,000\n"
        "POIROT: Greetings.\n"
    )
    from subtitle_translator.parsers import parse_subtitle
    doc = parse_subtitle(raw_with_label, ".srt")
    translated = translate_document(
        document=doc,
        translator=EchoTranslator(),
        settings=TranslationSettings(target_lang="bn"),
        glossary=GlossaryConfig(
            glossary_map={"POIROT": "পয়রট"},
            do_not_translate=[],
        ),
    )
    assert "পয়রট:" in translated.cues[-1].text


def test_translate_document_vtt_roundtrip_structure():
    doc = _load("sample.vtt")
    translated = translate_document(
        document=doc,
        translator=EchoTranslator(),
        settings=TranslationSettings(),
        glossary=GlossaryConfig(glossary_map={}, do_not_translate=[]),
    )
    assert translated.format == "vtt"
    assert len(translated.cues) == len(doc.cues)
    assert translated.header_lines == doc.header_lines
    assert translated.cues[1].identifier == "intro"
