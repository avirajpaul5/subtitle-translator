from __future__ import annotations

from subtitle_translator.credentials import get_sarvam_api_key
from subtitle_translator.translators.base import BaseTranslator
from subtitle_translator.translators.echo import EchoTranslator
from subtitle_translator.translators.fallback import FallbackTranslator
from subtitle_translator.translators.indictrans2 import (
    IndicTrans2InputChecker,
    IndicTrans2Translator,
    _local_model_stamp,
)
from subtitle_translator.translators.nllb import NLLBTranslator
from subtitle_translator.translators.sarvam_api import SarvamApiTranslator


class TranslatorInitError(RuntimeError):
    pass


def build_translator(
    backend: str,
    model_path: str | None,
    device: str = "cpu",
    *,
    sarvam_api_key: str | None = None,
    sarvam_model: str = "mayura:v1",
    sarvam_mode: str = "classic-colloquial",
    sarvam_fallback_backend: str | None = None,
    sarvam_use_keyring: bool = True,
) -> BaseTranslator:
    backend = backend.lower()
    if backend == "echo":
        return EchoTranslator()

    if backend == "indictrans2":
        if not model_path:
            raise TranslatorInitError("IndicTrans2 model path is required.")
        try:
            return IndicTrans2Translator(model_path=model_path, device=device)
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            raise TranslatorInitError(f"Failed to initialize IndicTrans2: {exc}") from exc

    if backend == "nllb":
        if not model_path:
            raise TranslatorInitError("NLLB model path is required.")
        try:
            return NLLBTranslator(model_path=model_path)
        except Exception as exc:  # pragma: no cover
            raise TranslatorInitError(f"Failed to initialize NLLB: {exc}") from exc

    if backend in {"sarvam", "sarvam-api", "sarvam_api"}:
        api_key = get_sarvam_api_key(sarvam_api_key, use_keyring=sarvam_use_keyring)
        if not api_key:
            raise TranslatorInitError(
                "Sarvam API key is required. Enter a key, save one in the OS keychain, "
                "or set SARVAM_API_KEY."
            )
        try:
            primary = SarvamApiTranslator(
                api_key=api_key,
                model=sarvam_model,
                mode=sarvam_mode,
            )
        except Exception as exc:
            raise TranslatorInitError(f"Failed to initialize Sarvam API: {exc}") from exc

        if sarvam_fallback_backend:
            fallback_backend = sarvam_fallback_backend.lower()
            fallback_checker: IndicTrans2InputChecker | None = None

            def _fallback_factory() -> BaseTranslator:
                return build_translator(
                    fallback_backend,
                    model_path=model_path,
                    device=device,
                    sarvam_use_keyring=False,
                )

            def _fallback_accepts_input(
                text: str,
                source_lang: str,
                target_lang: str,
            ) -> bool:
                nonlocal fallback_checker
                if fallback_backend != "indictrans2":
                    return False
                if fallback_checker is None:
                    if not model_path:
                        return False
                    fallback_checker = IndicTrans2InputChecker(model_path)
                return fallback_checker.accepts_input(
                    text,
                    source_lang,
                    target_lang,
                )

            return FallbackTranslator(
                primary,
                _fallback_factory,
                primary_name="Sarvam API",
                fallback_name=fallback_backend,
                fallback_max_input_chars=(
                    500 if fallback_backend == "indictrans2" else None
                ),
                fallback_checkpoint_fingerprint=(
                    f"{fallback_backend}|model={_local_model_stamp(model_path or '')}"
                ),
                fallback_accepts_input=_fallback_accepts_input,
            )

        return primary

    raise TranslatorInitError(f"Unsupported backend: {backend}")
