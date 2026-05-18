from __future__ import annotations

from pathlib import Path

from subtitle_translator.glossary import GlossaryConfig
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
