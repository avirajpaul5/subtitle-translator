from __future__ import annotations

import pytest

from subtitle_translator.credentials import get_sarvam_api_key
from subtitle_translator.translators.base import BaseTranslator
from subtitle_translator.translators.factory import TranslatorInitError, build_translator
from subtitle_translator.translators.fallback import FallbackTranslator
from subtitle_translator.translators.sarvam_api import (
    SarvamApiError,
    SarvamApiTranslator,
    to_sarvam_language_code,
)


def test_sarvam_language_code_mapping():
    assert to_sarvam_language_code("en") == "en-IN"
    assert to_sarvam_language_code("bn") == "bn-IN"
    assert to_sarvam_language_code("hi-IN") == "hi-IN"
    assert to_sarvam_language_code("auto") == "auto"
    with pytest.raises(SarvamApiError):
        to_sarvam_language_code("fr")


def test_sarvam_translator_sends_expected_payload():
    payloads: list[dict] = []

    def transport(payload: dict) -> dict:
        payloads.append(payload)
        return {"translated_text": f"bn:{payload['input']}"}

    translator = SarvamApiTranslator(
        api_key="test-key",
        model="mayura:v1",
        mode="classic-colloquial",
        max_workers=1,
        transport=transport,
    )

    out = translator.translate_batch(["Hello there"], source_lang="en", target_lang="bn")

    assert out == ["bn:Hello there"]
    assert payloads == [
        {
            "input": "Hello there",
            "source_language_code": "en-IN",
            "target_language_code": "bn-IN",
            "model": "mayura:v1",
            "numerals_format": "international",
            "mode": "classic-colloquial",
        }
    ]


def test_sarvam_translate_model_forces_formal_and_omits_mode_payload():
    payloads: list[dict] = []

    def transport(payload: dict) -> dict:
        payloads.append(payload)
        return {"translated_text": "translated"}

    translator = SarvamApiTranslator(
        api_key="test-key",
        model="sarvam-translate:v1",
        mode="modern-colloquial",
        transport=transport,
    )

    assert translator.mode == "formal"
    assert translator.translate_batch(["Hello"], "en", "bn") == ["translated"]
    assert "mode" not in payloads[0]


def test_sarvam_translator_splits_long_text_under_model_limit():
    payloads: list[dict] = []

    def transport(payload: dict) -> dict:
        payloads.append(payload)
        return {"translated_text": payload["input"].upper()}

    translator = SarvamApiTranslator(
        api_key="test-key",
        model="mayura:v1",
        max_workers=1,
        transport=transport,
    )
    long_text = "A" * 900 + ". " + "B" * 900 + "."

    out = translator.translate_batch([long_text], "en", "bn")

    assert len(payloads) == 2
    assert all(len(payload["input"]) <= 1000 for payload in payloads)
    assert out == [long_text]


def test_sarvam_translator_rejects_missing_translated_text():
    translator = SarvamApiTranslator(
        api_key="test-key",
        transport=lambda payload: {"unexpected": "shape"},
    )

    with pytest.raises(SarvamApiError, match="translated_text"):
        translator.translate_batch(["Hello"], "en", "bn")


def test_get_sarvam_api_key_prefers_explicit_then_env(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", " env-key ")

    assert get_sarvam_api_key(" explicit-key ", use_keyring=False) == "explicit-key"
    assert get_sarvam_api_key(use_keyring=False) == "env-key"


def test_factory_requires_sarvam_api_key(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)

    with pytest.raises(TranslatorInitError, match="Sarvam API key"):
        build_translator(
            "sarvam-api",
            model_path=None,
            sarvam_use_keyring=False,
        )


def test_factory_builds_sarvam_with_explicit_key():
    translator = build_translator(
        "sarvam-api",
        model_path=None,
        sarvam_api_key="test-key",
        sarvam_use_keyring=False,
        sarvam_fallback_backend=None,
    )

    assert isinstance(translator, SarvamApiTranslator)


class _FailingTranslator(BaseTranslator):
    def translate_batch(self, texts, source_lang: str, target_lang: str):
        raise RuntimeError("primary exploded")


class _UpperTranslator(BaseTranslator):
    def translate_batch(self, texts, source_lang: str, target_lang: str):
        return [text.upper() for text in texts]


def test_fallback_translator_uses_fallback_and_records_warning():
    translator = FallbackTranslator(
        _FailingTranslator(),
        lambda: _UpperTranslator(),
        primary_name="Sarvam API",
        fallback_name="echo",
    )

    assert translator.translate_batch(["hello"], "en", "bn") == ["HELLO"]
    assert translator.warnings
    assert "Sarvam API failed" in translator.warnings[0]
    assert "echo fallback" in translator.warnings[0]
