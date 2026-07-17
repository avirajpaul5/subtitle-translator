from __future__ import annotations

from typing import Callable, Iterable, List

from subtitle_translator.translators.base import BaseTranslator


class FallbackTranslationError(RuntimeError):
    pass


class FallbackTranslator(BaseTranslator):
    """Use a fallback translator when the primary backend fails for a batch."""

    def __init__(
        self,
        primary: BaseTranslator,
        fallback_factory: Callable[[], BaseTranslator],
        *,
        primary_name: str,
        fallback_name: str,
        fallback_max_input_chars: int | None = None,
        fallback_checkpoint_fingerprint: str | None = None,
        fallback_accepts_input: Callable[[str, str, str], bool] | None = None,
    ) -> None:
        self.primary = primary
        self._fallback_factory = fallback_factory
        self._fallback: BaseTranslator | None = None
        self.primary_name = primary_name
        self.fallback_name = fallback_name
        self.fallback_max_input_chars = fallback_max_input_chars
        self.fallback_checkpoint_fingerprint = (
            fallback_checkpoint_fingerprint or fallback_name
        )
        self._fallback_accepts_input = fallback_accepts_input
        self.warnings: list[str] = []
        self.fallback_count = 0
        self._last_used_name = primary.display_name

    def translate_batch(self, texts: Iterable[str], source_lang: str, target_lang: str) -> List[str]:
        materialized = list(texts)
        self._last_used_name = self.primary.display_name
        try:
            return self.primary.translate_batch(materialized, source_lang, target_lang)
        except Exception as exc:
            primary_reason = _safe_reason(exc)
            try:
                if self._fallback is None:
                    self._fallback = self._fallback_factory()
                translated = self._fallback.translate_batch(materialized, source_lang, target_lang)
            except Exception as fallback_exc:
                raise FallbackTranslationError(
                    f"{self.primary_name} failed and {self.fallback_name} fallback also failed. "
                    f"{self.primary_name} reason: {primary_reason}. "
                    f"{self.fallback_name} reason: {_safe_reason(fallback_exc)}"
                ) from fallback_exc

            self.fallback_count += 1
            self._last_used_name = f"{self._fallback.display_name} fallback"
            self.warnings.append(
                (
                    f"FALLBACK USED: {self.primary_name} failed for a batch; "
                    f"that batch was translated with {self.fallback_name} fallback. "
                    f"Reason: {primary_reason}"
                )
                + _primary_usage_note(self.primary)
            )
            return translated

    @property
    def display_name(self) -> str:
        return f"{self.primary.display_name} (fallback: {self.fallback_name})"

    @property
    def checkpoint_fingerprint(self) -> str:
        return (
            f"primary={self.primary.checkpoint_fingerprint}|"
            f"fallback={self.fallback_checkpoint_fingerprint}|"
            f"fallback_cap={self.fallback_max_input_chars}"
        )

    @property
    def pipeline_chunk_size(self) -> int | None:
        value = getattr(self.primary, "pipeline_chunk_size", None)
        return int(value) if isinstance(value, int) and value > 0 else None

    @property
    def max_input_chars(self) -> int | None:
        limits = [
            value
            for value in (
                getattr(self.primary, "max_input_chars", None),
                self.fallback_max_input_chars,
            )
            if isinstance(value, int) and value > 0
        ]
        return min(limits) if limits else None

    def accepts_input(self, text: str, source_lang: str, target_lang: str) -> bool:
        """Accept a payload only when both possible execution paths do.

        A fallback batch must be safe to send to either backend. Character
        caps provide an inexpensive first check, while each translator's
        tokenizer-aware check remains authoritative.
        """

        max_chars = self.max_input_chars
        if isinstance(max_chars, int) and len(text) > max_chars:
            return False
        if not self.primary.accepts_input(text, source_lang, target_lang):
            return False
        if self._fallback_accepts_input is not None:
            return self._fallback_accepts_input(text, source_lang, target_lang)
        if self._fallback is not None:
            return self._fallback.accepts_input(text, source_lang, target_lang)
        # Do not instantiate an expensive fallback model just to plan a
        # primary-provider request. Without an exact capability callback, we
        # cannot safely claim that both paths accept this payload.
        return False

    @property
    def last_used_name(self) -> str:
        return self._last_used_name

    @property
    def usage_summary(self) -> str:
        primary_summary = self.primary.usage_summary
        if self.fallback_count == 0:
            return f"{primary_summary}; fallback not used"

        fallback_name = (
            self._fallback.display_name if self._fallback is not None else self.fallback_name
        )
        batches = "batch" if self.fallback_count == 1 else "batches"
        return (
            f"{primary_summary}; used {fallback_name} fallback for "
            f"{self.fallback_count} {batches}"
        )


def _safe_reason(exc: Exception) -> str:
    text = str(exc).replace("\n", " ").strip()
    if len(text) > 220:
        text = text[:217] + "..."
    return text or exc.__class__.__name__


def _primary_usage_note(primary: BaseTranslator) -> str:
    attempted = getattr(primary, "attempted_request_count", None)
    successful = getattr(primary, "successful_request_count", None)
    if not isinstance(attempted, int) or not isinstance(successful, int):
        return ""

    if attempted == 0:
        return " No Sarvam API request was sent before fallback."

    return (
        f" Sarvam API responses before fallback: "
        f"{successful} successful/{attempted} attempted."
    )
