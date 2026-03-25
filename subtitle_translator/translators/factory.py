from __future__ import annotations

from subtitle_translator.translators.base import BaseTranslator
from subtitle_translator.translators.echo import EchoTranslator
from subtitle_translator.translators.indictrans2 import IndicTrans2Translator
from subtitle_translator.translators.nllb import NLLBTranslator


class TranslatorInitError(RuntimeError):
    pass


def build_translator(backend: str, model_path: str | None, device: str = "cpu") -> BaseTranslator:
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

    raise TranslatorInitError(f"Unsupported backend: {backend}")
