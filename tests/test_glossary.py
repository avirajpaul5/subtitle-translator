from __future__ import annotations

import pytest

from subtitle_translator.glossary import (
    ProtectedTermAlignmentError,
    protect_terms,
    restore_terms,
    validate_restored_terms,
)


def test_protect_terms_registers_only_terms_present_in_source():
    protected, replacements = protect_terms(
        ["Hello world."],
        ["Mademoiselle", "Monsieur"],
    )

    assert protected == ["Hello world."]
    assert replacements == {}
    # With no active replacement, a hallucinated marker remains visible for
    # the validation pass instead of becoming a plausible protected name.
    assert restore_terms(["ID0ZZ"], replacements) == ["ID0ZZ"]


def test_protect_terms_assigns_equal_length_terms_deterministically():
    first = protect_terms(["Zulu and Alfa"], ["Zulu", "Alfa"])
    second = protect_terms(["Zulu and Alfa"], ["Alfa", "Zulu"])

    assert first == second
    assert first[1] == {
        "ZZID0ZZ": "Alfa",
        "ZZID1ZZ": "Zulu",
    }


def test_protect_terms_restores_each_source_spelling_exactly():
    source = "MONSIEUR met Monsieur and monsieur."

    protected, replacements = protect_terms([source], ["Monsieur"])

    assert replacements == {
        "ZZID0ZZ": "MONSIEUR",
        "ZZID1ZZ": "Monsieur",
        "ZZID2ZZ": "monsieur",
    }
    assert restore_terms(protected, replacements) == [source]


def test_restore_terms_can_preserve_document_spacing():
    assert restore_terms(
        ["left\tZZID0ZZ  right"],
        {"ZZID0ZZ": "Monsieur"},
        normalize_spacing=False,
    ) == ["left\tMonsieur  right"]


def test_validation_does_not_count_a_short_term_inside_larger_words():
    protected, replacements = protect_terms(
        ["Japan said Ja. Dada replied Da."],
        ["Ja", "Da"],
    )
    restored = restore_terms(protected, replacements)[0]

    validate_restored_terms(protected[0], restored, replacements)
    assert restored == "Japan said Ja. Dada replied Da."


def test_validation_rejects_reordered_protected_terms():
    protected, replacements = protect_terms(["Ja then Da."], ["Ja", "Da"])

    with pytest.raises(ProtectedTermAlignmentError, match="reordered"):
        validate_restored_terms(protected[0], "Da then Ja.", replacements)
