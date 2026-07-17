from __future__ import annotations

import re

import pytest

from subtitle_translator.document_pipeline import (
    DocumentTranslationInterruptedError,
    DocumentTranslationSettings,
    translate_text_document,
)
from subtitle_translator.documents import parse_document, serialize_document
from subtitle_translator.glossary import GlossaryConfig
from subtitle_translator.translators.base import BaseTranslator
from subtitle_translator.translators.echo import EchoTranslator


class _RecordingTranslator(BaseTranslator):
    def __init__(self, fail_on_call: int | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_on_call = fail_on_call

    def translate_batch(self, texts, source_lang: str, target_lang: str):
        materialized = list(texts)
        self.calls.append(materialized)
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("rate limit")
        return [
            re.sub(r"(ZZID9\d{3}ZZ)\s*", r"\1 bn:", text)
            if "ZZID9001ZZ" in text
            else f"bn:{text}"
            for text in materialized
        ]


class _WarningTranslator(_RecordingTranslator):
    def __init__(self, warning: str) -> None:
        super().__init__()
        self.warnings = [warning]


class _CrossBlockSentinelTranslator(BaseTranslator):
    def translate_batch(self, texts, source_lang: str, target_lang: str):
        return [text if "ZZID0ZZ" in text else "ID0ZZ" for text in texts]


class _DuplicatingSentinelTranslator(BaseTranslator):
    def translate_batch(self, texts, source_lang: str, target_lang: str):
        return [f"{text} ID0ZZ" for text in texts]


class _TokenBudgetTranslator(_RecordingTranslator):
    def accepts_input(self, text: str, source_lang: str, target_lang: str) -> bool:
        return len(text) <= 50


class _DroppingLayoutTranslator(BaseTranslator):
    def translate_batch(self, texts, source_lang: str, target_lang: str):
        return [re.sub(r"\s*ZZWS\d+ZZ\s*", " ", text) for text in texts]


class _ReorderingLayoutTranslator(BaseTranslator):
    def translate_batch(self, texts, source_lang: str, target_lang: str):
        return [
            text.replace("ZZWS0ZZ", "ZZWSTEMPZZ")
            .replace("ZZWS1ZZ", "ZZWS0ZZ")
            .replace("ZZWSTEMPZZ", "ZZWS1ZZ")
            for text in texts
        ]


def test_txt_document_translation_preserves_blank_line_structure():
    document = parse_document("First paragraph.\n\nSecond paragraph.\n", ".txt")
    translated = translate_text_document(
        document,
        _RecordingTranslator(),
        DocumentTranslationSettings(),
        GlossaryConfig({}, []),
    )

    assert serialize_document(translated) == (
        "bn:First paragraph.\n\nbn:Second paragraph.\n"
    )


def test_blank_lines_do_not_break_semantic_context_windows():
    source = "# Guide\n\nFirst paragraph.\n\nSecond paragraph.\n"
    translator = _RecordingTranslator()

    translated = translate_text_document(
        parse_document(source, ".md"),
        translator,
        DocumentTranslationSettings(context_window_blocks=6),
        GlossaryConfig({}, []),
    )

    assert len(translator.calls) == 1
    assert len(translator.calls[0]) == 1
    assert translator.calls[0][0].count("ZZID9") == 3
    assert serialize_document(translated) == (
        "# bn:Guide\n\nbn:First paragraph.\n\nbn:Second paragraph.\n"
    )


@pytest.mark.parametrize(
    "source, extension",
    [
        ("Column A    Column B\nKeep\t\taligned\n", ".txt"),
        ("Use `git` and  keep    these\t\tgaps.\n", ".md"),
    ],
)
def test_document_echo_round_trip_preserves_horizontal_layout(source, extension):
    translated = translate_text_document(
        parse_document(source, extension),
        EchoTranslator(),
        DocumentTranslationSettings(),
        GlossaryConfig({}, []),
    )

    assert serialize_document(translated) == source


def test_document_rejects_dropped_horizontal_layout_marker():
    with pytest.raises(DocumentTranslationInterruptedError, match="layout marker"):
        translate_text_document(
            parse_document("Column A    Column B\n", ".txt"),
            _DroppingLayoutTranslator(),
            DocumentTranslationSettings(),
            GlossaryConfig({}, []),
        )


def test_document_rejects_reordered_horizontal_layout_markers():
    with pytest.raises(DocumentTranslationInterruptedError, match="reordered"):
        translate_text_document(
            parse_document("Column A    Column B\tColumn C\n", ".txt"),
            _ReorderingLayoutTranslator(),
            DocumentTranslationSettings(),
            GlossaryConfig({}, []),
        )


def test_markdown_document_translation_restores_inline_code_and_url():
    source = (
        "# Guide\n\n"
        "Use `git` and read [the docs](https://example.com/guide).\n"
        "This is **important**, _emphasized_, and \\*escaped\\*.  \n"
    )
    document = parse_document(source, ".md")
    translated = translate_text_document(
        document,
        EchoTranslator(),
        DocumentTranslationSettings(target_lang="hi"),
        GlossaryConfig({}, []),
    )

    assert serialize_document(translated) == source


def test_document_pipeline_splits_long_blocks_below_context_budget():
    document = parse_document(" ".join(["sentence"] * 80), ".txt")
    translator = _RecordingTranslator()
    translated = translate_text_document(
        document,
        translator,
        DocumentTranslationSettings(
            context_window_chars=140,
            context_window_blocks=2,
        ),
        GlossaryConfig({}, []),
    )

    assert serialize_document(translated).startswith("bn:")
    assert all(len(payload) <= 140 for call in translator.calls for payload in call)


def test_document_pipeline_checkpoints_and_resumes_by_window(tmp_path):
    document = parse_document("One.\n\nTwo.\n\nThree.", ".txt")
    checkpoint = tmp_path / "document.json"
    settings = DocumentTranslationSettings(
        chunk_size=1,
        context_window_chars=100,
        context_window_blocks=1,
    )

    with pytest.raises(DocumentTranslationInterruptedError) as raised:
        translate_text_document(
            document,
            _RecordingTranslator(fail_on_call=2),
            settings,
            GlossaryConfig({}, []),
            checkpoint_path=checkpoint,
        )

    assert checkpoint.exists()
    assert serialize_document(raised.value.partial_document).startswith("bn:One.")

    translator = _RecordingTranslator()
    translated = translate_text_document(
        document,
        translator,
        settings,
        GlossaryConfig({}, []),
        checkpoint_path=checkpoint,
    )
    assert translator.calls == [["Two."], ["Three."]]
    assert serialize_document(translated) == "bn:One.\n\nbn:Two.\n\nbn:Three."


def test_document_pipeline_surfaces_provider_warnings():
    translated = translate_text_document(
        parse_document("Hello.", ".txt"),
        _WarningTranslator("FALLBACK USED: local backend handled this batch."),
        DocumentTranslationSettings(),
        GlossaryConfig({}, []),
    )

    assert translated.warnings == (
        "FALLBACK USED: local backend handled this batch.",
    )


def test_document_pipeline_flags_suspect_blocks_for_review():
    translated = translate_text_document(
        parse_document("MENID99999ZZ entered.", ".txt"),
        EchoTranslator(),
        DocumentTranslationSettings(target_lang="bn"),
        GlossaryConfig({}, []),
    )

    assert any(
        warning == "block txt:b000001: sentinel_debris"
        for warning in translated.warnings
    )


def test_markdown_protected_content_wins_over_glossary():
    translated = translate_text_document(
        parse_document("Use `doctor`, then call doctor.\n", ".md"),
        EchoTranslator(),
        DocumentTranslationSettings(),
        GlossaryConfig({"doctor": "ডাক্তার"}, []),
    )

    assert serialize_document(translated) == "Use `doctor`, then call ডাক্তার.\n"
    assert not any(
        "glossary_term_untranslated:doctor" in warning
        for warning in translated.warnings
    )


def test_document_sentinel_restoration_is_scoped_to_source_piece():
    translated = translate_text_document(
        parse_document("Monsieur said.\n\nHello.", ".txt"),
        _CrossBlockSentinelTranslator(),
        DocumentTranslationSettings(context_window_blocks=1),
        GlossaryConfig({}, ["Monsieur"]),
    )

    assert serialize_document(translated) == "Monsieur said.\n\nID0ZZ"
    assert any(
        "block txt:b000003: sentinel_debris" == warning
        for warning in translated.warnings
    )


def test_document_preserves_each_protected_term_spelling_and_case():
    source = "MONSIEUR met Monsieur and monsieur.\n"

    translated = translate_text_document(
        parse_document(source, ".txt"),
        EchoTranslator(),
        DocumentTranslationSettings(),
        GlossaryConfig({}, ["Monsieur"]),
    )

    assert serialize_document(translated) == source


def test_document_short_protected_terms_do_not_collide_with_larger_words():
    source = "Japan said Ja. Dada replied Da.\n"

    translated = translate_text_document(
        parse_document(source, ".txt"),
        EchoTranslator(),
        DocumentTranslationSettings(),
        GlossaryConfig({}, ["Ja", "Da"]),
    )

    assert serialize_document(translated) == source


def test_document_checkpoint_identity_includes_source_format(tmp_path):
    checkpoint = tmp_path / "shared.json"
    settings = DocumentTranslationSettings(context_window_blocks=1)
    translate_text_document(
        parse_document("# Title", ".txt"),
        _RecordingTranslator(),
        settings,
        GlossaryConfig({}, []),
        checkpoint_path=checkpoint,
    )

    markdown_translator = _RecordingTranslator()
    translated = translate_text_document(
        parse_document("# Title", ".md"),
        markdown_translator,
        settings,
        GlossaryConfig({}, []),
        checkpoint_path=checkpoint,
    )

    assert markdown_translator.calls == [["Title"]]
    assert serialize_document(translated) == "# bn:Title"


def test_document_rejects_extra_active_sentinel_copy():
    with pytest.raises(DocumentTranslationInterruptedError, match="restored 2 time"):
        translate_text_document(
            parse_document("Monsieur said.", ".txt"),
            _DuplicatingSentinelTranslator(),
            DocumentTranslationSettings(),
            GlossaryConfig({}, ["Monsieur"]),
        )


def test_document_pipeline_recursively_splits_to_exact_provider_budget():
    translator = _TokenBudgetTranslator()
    translated = translate_text_document(
        parse_document("word " * 50, ".txt"),
        translator,
        DocumentTranslationSettings(context_window_chars=500),
        GlossaryConfig({}, []),
    )

    assert serialize_document(translated).startswith("bn:")
    assert translator.calls
    assert all(len(payload) <= 50 for call in translator.calls for payload in call)
