from __future__ import annotations

from typing import Callable, Iterable, List

from subtitle_translator.translators.base import BaseTranslator


class FallbackTranslator(BaseTranslator):
    """Use a fallback translator when the primary backend fails for a batch."""

    def __init__(
        self,
        primary: BaseTranslator,
        fallback_factory: Callable[[], BaseTranslator],
        *,
        primary_name: str,
        fallback_name: str,
    ) -> None:
        self.primary = primary
        self._fallback_factory = fallback_factory
        self._fallback: BaseTranslator | None = None
        self.primary_name = primary_name
        self.fallback_name = fallback_name
        self.warnings: list[str] = []

    def translate_batch(self, texts: Iterable[str], source_lang: str, target_lang: str) -> List[str]:
        materialized = list(texts)
        try:
            return self.primary.translate_batch(materialized, source_lang, target_lang)
        except Exception as exc:
            message = (
                f"{self.primary_name} failed for a batch; used {self.fallback_name} fallback. "
                f"Reason: {_safe_reason(exc)}"
            )
            self.warnings.append(message)
            if self._fallback is None:
                self._fallback = self._fallback_factory()
            return self._fallback.translate_batch(materialized, source_lang, target_lang)


def _safe_reason(exc: Exception) -> str:
    text = str(exc).replace("\n", " ").strip()
    if len(text) > 220:
        text = text[:217] + "..."
    return text or exc.__class__.__name__
