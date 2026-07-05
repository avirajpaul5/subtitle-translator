from __future__ import annotations

import io
import ssl
import urllib.error
from email.message import Message

import pytest

from subtitle_translator.credentials import get_sarvam_api_key
from subtitle_translator.translators.base import BaseTranslator
from subtitle_translator.translators.factory import TranslatorInitError, build_translator
from subtitle_translator.translators.fallback import FallbackTranslationError, FallbackTranslator
from subtitle_translator.translators.sarvam_api import (
    SarvamApiError,
    SarvamApiTranslator,
    SarvamRateLimitError,
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
        return {"translated_text": f"bn:{payload['input']}", "request_id": "req-1"}

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
    assert translator.attempted_request_count == 1
    assert translator.successful_request_count == 1
    assert translator.attempted_input_chars == len("Hello there")
    assert translator.successful_input_chars == len("Hello there")
    assert translator.request_ids == ["req-1"]
    assert "1 successful/1 attempted" in translator.usage_summary
    assert "input chars" in translator.usage_summary
    assert "req-1" in translator.usage_summary


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
    assert translator.attempted_request_count == 1
    assert translator.successful_request_count == 0
    assert translator.attempted_input_chars == len("Hello")


def test_sarvam_translator_passes_ssl_context_to_urlopen(monkeypatch):
    seen: dict[str, object] = {}
    ssl_context = ssl.create_default_context()

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"translated_text": "translated", "request_id": "req-ssl"}'

    def fake_urlopen(request, *, timeout, context):
        seen["request"] = request
        seen["timeout"] = timeout
        seen["context"] = context
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    translator = SarvamApiTranslator(
        api_key="test-key",
        max_workers=1,
        ssl_context=ssl_context,
    )

    assert translator.translate_batch(["Hello"], "en", "bn") == ["translated"]
    assert seen["context"] is ssl_context
    assert translator.request_ids == ["req-ssl"]


def test_sarvam_certificate_error_is_actionable(monkeypatch):
    def fake_urlopen(request, *, timeout, context):
        raise urllib.error.URLError(
            ssl.SSLCertVerificationError("certificate verify failed")
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    translator = SarvamApiTranslator(
        api_key="test-key",
        retries=0,
        max_workers=1,
    )

    with pytest.raises(SarvamApiError, match="Could not validate Sarvam API TLS certificate"):
        translator.translate_batch(["Hello"], "en", "bn")

    assert translator.attempted_request_count == 1
    assert translator.successful_request_count == 0


def test_sarvam_rate_limit_error_is_actionable(monkeypatch):
    headers = Message()
    headers["Retry-After"] = "12"

    def fake_urlopen(request, *, timeout, context):
        raise urllib.error.HTTPError(
            request.full_url,
            429,
            "Too Many Requests",
            headers,
            io.BytesIO(
                b'{"error":{"code":"rate_limit_exceeded_error","message":"Rate limit exceeded"}}'
            ),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    translator = SarvamApiTranslator(
        api_key="test-key",
        retries=0,
        max_workers=1,
        min_request_interval_seconds=0,
    )

    with pytest.raises(SarvamRateLimitError, match="Retry after about 12 seconds"):
        translator.translate_batch(["Hello"], "en", "bn")

    assert translator.attempted_request_count == 1
    assert translator.successful_request_count == 0


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
    assert translator.display_name == "Sarvam API (mayura:v1, classic-colloquial)"


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
    assert "FALLBACK USED" in translator.warnings[0]
    assert "Sarvam API failed" in translator.warnings[0]
    assert "echo fallback" in translator.warnings[0]
    assert translator.fallback_count == 1
    assert translator.last_used_name == "UpperTranslator fallback"
    assert "used UpperTranslator fallback for 1 batch" in translator.usage_summary


def test_fallback_translator_reports_when_fallback_not_used():
    translator = FallbackTranslator(
        _UpperTranslator(),
        lambda: _FailingTranslator(),
        primary_name="Sarvam API",
        fallback_name="echo",
    )

    assert translator.translate_batch(["hello"], "en", "bn") == ["HELLO"]
    assert translator.fallback_count == 0
    assert translator.usage_summary == "UpperTranslator; fallback not used"


def test_fallback_translator_reports_primary_usage_in_warning():
    class _SarvamLikeFailure(BaseTranslator):
        attempted_request_count = 1
        successful_request_count = 0

        @property
        def display_name(self) -> str:
            return "Sarvam API (mayura:v1)"

        @property
        def usage_summary(self) -> str:
            return "Sarvam API (mayura:v1); Sarvam API responses: 0 successful/1 attempted"

        def translate_batch(self, texts, source_lang: str, target_lang: str):
            raise RuntimeError("HTTP 403")

    translator = FallbackTranslator(
        _SarvamLikeFailure(),
        lambda: _UpperTranslator(),
        primary_name="Sarvam API",
        fallback_name="indictrans2",
    )

    assert translator.translate_batch(["hello"], "en", "bn") == ["HELLO"]
    assert "0 successful/1 attempted" in translator.warnings[0]
    assert "0 successful/1 attempted" in translator.usage_summary


def test_fallback_translator_combines_primary_and_fallback_failures():
    translator = FallbackTranslator(
        _FailingTranslator(),
        lambda: _FailingTranslator(),
        primary_name="Sarvam API",
        fallback_name="indictrans2",
    )

    with pytest.raises(FallbackTranslationError, match="fallback also failed"):
        translator.translate_batch(["hello"], "en", "bn")
