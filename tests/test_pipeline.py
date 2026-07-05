from __future__ import annotations

from pathlib import Path

import pytest

from subtitle_translator.glossary import GlossaryConfig
from subtitle_translator.models import Cue, SubtitleDocument
from subtitle_translator.parsers import parse_subtitle
from subtitle_translator.pipeline import (
    TranslationInterruptedError,
    TranslationSettings,
    translate_document,
)
from subtitle_translator.translators.base import BaseTranslator
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


class _WarningTranslator(BaseTranslator):
    warnings = ["Sarvam API failed for a batch; used indictrans2 fallback."]

    def translate_batch(self, texts, source_lang: str, target_lang: str):
        return list(texts)


def test_translate_document_appends_translator_warnings():
    doc = _load("sample.srt")
    translated = translate_document(
        document=doc,
        translator=_WarningTranslator(),
        settings=TranslationSettings(),
        glossary=GlossaryConfig(glossary_map={}, do_not_translate=[]),
    )
    assert "Sarvam API failed for a batch" in translated.warnings[-1]


def test_translate_document_progress_reports_provider_used():
    doc = _load("sample.srt")
    messages: list[str] = []

    translate_document(
        document=doc,
        translator=EchoTranslator(),
        settings=TranslationSettings(chunk_size=99),
        glossary=GlossaryConfig(glossary_map={}, do_not_translate=[]),
        progress_cb=lambda value, message: messages.append(message),
    )

    assert any("Translating 1/1 with echo" in message for message in messages)
    assert any("Translated 1/1 chunks with echo" in message for message in messages)


class _RecordingTranslator(BaseTranslator):
    def __init__(self, fail_on_call: int | None = None) -> None:
        self.inputs: list[str] = []
        self.calls = 0
        self.fail_on_call = fail_on_call

    @property
    def display_name(self) -> str:
        return "recording"

    def translate_batch(self, texts, source_lang: str, target_lang: str):
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise RuntimeError("rate limit")
        materialized = list(texts)
        self.inputs.extend(materialized)
        return [f"bn:{text}" for text in materialized]


class _ExplodingTranslator(BaseTranslator):
    def translate_batch(self, texts, source_lang: str, target_lang: str):
        raise AssertionError("translator should not be called")


def _mini_doc(*texts: str) -> SubtitleDocument:
    return SubtitleDocument(
        format="srt",
        cues=[
            Cue(
                index=i,
                start=f"00:00:0{i},000",
                end=f"00:00:0{i + 1},000",
                text_lines=[text],
            )
            for i, text in enumerate(texts, start=1)
        ],
    )


def test_translate_document_skips_protected_only_text():
    doc = _mini_doc("Monsieur")

    translated = translate_document(
        document=doc,
        translator=_ExplodingTranslator(),
        settings=TranslationSettings(chunk_size=1),
        glossary=GlossaryConfig(glossary_map={}, do_not_translate=["Monsieur"]),
    )

    assert translated.cues[0].text == "Monsieur"


def test_translate_document_deduplicates_model_payloads():
    doc = _mini_doc("Hello there.", "Hello there.")
    translator = _RecordingTranslator()

    translated = translate_document(
        document=doc,
        translator=translator,
        settings=TranslationSettings(chunk_size=2),
        glossary=GlossaryConfig(glossary_map={}, do_not_translate=[]),
    )

    assert translator.inputs == ["Hello there."]
    assert [cue.text for cue in translated.cues] == ["bn:Hello there.", "bn:Hello there."]


def test_translate_document_saves_checkpoint_and_resumes(tmp_path):
    doc = _mini_doc("First line.", "Second line.", "Third line.")
    checkpoint = tmp_path / "movie.checkpoint.json"

    with pytest.raises(TranslationInterruptedError) as raised:
        translate_document(
            document=doc,
            translator=_RecordingTranslator(fail_on_call=2),
            settings=TranslationSettings(chunk_size=1),
            glossary=GlossaryConfig(glossary_map={}, do_not_translate=[]),
            checkpoint_path=checkpoint,
        )

    assert checkpoint.exists()
    assert "1/3 chunks" in str(raised.value)
    assert raised.value.partial_document.cues[0].text == "bn:First line."
    assert raised.value.partial_document.cues[1].text == "Second line."

    resume_translator = _RecordingTranslator()
    translated = translate_document(
        document=doc,
        translator=resume_translator,
        settings=TranslationSettings(chunk_size=1),
        glossary=GlossaryConfig(glossary_map={}, do_not_translate=[]),
        checkpoint_path=checkpoint,
    )

    assert resume_translator.inputs == ["Second line.", "Third line."]
    assert [cue.text for cue in translated.cues] == [
        "bn:First line.",
        "bn:Second line.",
        "bn:Third line.",
    ]
