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
        self.fallback_count = 0
        self._last_used_name = primary.display_name

    def translate_batch(self, texts: Iterable[str], source_lang: str, target_lang: str) -> List[str]:
        materialized = list(texts)
        self._last_used_name = self.primary.display_name
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
            self.fallback_count += 1
            self._last_used_name = f"{self._fallback.display_name} fallback"
            return self._fallback.translate_batch(materialized, source_lang, target_lang)

    @property
    def display_name(self) -> str:
        return f"{self.primary.display_name} (fallback: {self.fallback_name})"

    @property
    def last_used_name(self) -> str:
        return self._last_used_name

    @property
    def usage_summary(self) -> str:
        if self.fallback_count == 0:
            return f"{self.primary.display_name}; fallback not used"

        fallback_name = (
            self._fallback.display_name if self._fallback is not None else self.fallback_name
        )
        batches = "batch" if self.fallback_count == 1 else "batches"
        return (
            f"{self.primary.display_name}; used {fallback_name} fallback for "
            f"{self.fallback_count} {batches}"
        )


def _safe_reason(exc: Exception) -> str:
    text = str(exc).replace("\n", " ").strip()
    if len(text) > 220:
        text = text[:217] + "..."
    return text or exc.__class__.__name__
