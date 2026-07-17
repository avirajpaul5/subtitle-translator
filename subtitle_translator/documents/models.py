from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Mapping


@dataclass(frozen=True)
class InlineSpan:
    """A source-text span with an explicit translation policy.

    Markdown adapters use non-translatable spans for inline code, URLs, and raw
    HTML.  Offsets are relative to ``TranslationBlock.source_text``.
    """

    kind: str
    text: str
    start: int
    end: int
    translatable: bool = True

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("Inline span offsets must be ordered and non-negative.")
        if self.end - self.start != len(self.text):
            raise ValueError("Inline span offsets must match the span text length.")


@dataclass(frozen=True)
class TranslationBlock:
    """One stable, independently addressable piece of a source document.

    ``source_text`` is immutable.  Setting a translation returns a new block,
    which keeps checkpoints and comparisons from accidentally losing the
    original text.
    """

    block_id: str
    kind: str
    source_text: str
    target_text: str | None = None
    translatable: bool = True
    prefix: str = ""
    suffix: str = ""
    spans: tuple[InlineSpan, ...] = ()
    path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.block_id:
            raise ValueError("Translation blocks require a stable block_id.")
        if self.target_text is not None and not self.translatable:
            raise ValueError(f"Block {self.block_id} is structural and cannot be translated.")

        if self.spans:
            cursor = 0
            for span in self.spans:
                if span.start != cursor:
                    raise ValueError(
                        f"Inline spans for {self.block_id} must cover source_text contiguously."
                    )
                if self.source_text[span.start : span.end] != span.text:
                    raise ValueError(
                        f"Inline span text does not match source_text in {self.block_id}."
                    )
                cursor = span.end
            if cursor != len(self.source_text):
                raise ValueError(
                    f"Inline spans for {self.block_id} must cover all of source_text."
                )

        if self.target_text is not None:
            self._validate_protected_content(self.target_text)

    @property
    def text(self) -> str:
        """Return translated text when present, otherwise the immutable source."""

        return self.target_text if self.target_text is not None else self.source_text

    @property
    def protected_spans(self) -> tuple[InlineSpan, ...]:
        return tuple(span for span in self.spans if not span.translatable)

    def with_target_text(self, target_text: str | None) -> "TranslationBlock":
        return replace(self, target_text=target_text)

    def _validate_protected_content(self, target_text: str) -> None:
        required = Counter(
            span.text for span in self.spans if not span.translatable and span.text
        )
        missing = [
            protected
            for protected, count in required.items()
            if target_text.count(protected) < count
        ]
        if missing:
            formatted = ", ".join(repr(value) for value in missing)
            raise ValueError(
                f"Translation for {self.block_id} dropped protected content: {formatted}"
            )
        duplicated = [
            protected
            for protected, count in required.items()
            if target_text.count(protected) > count
        ]
        if duplicated:
            formatted = ", ".join(repr(value) for value in duplicated)
            raise ValueError(
                f"Translation for {self.block_id} duplicated protected content: {formatted}"
            )

        cursor = 0
        for span in self.protected_spans:
            position = target_text.find(span.text, cursor)
            if position < 0:
                raise ValueError(
                    f"Translation for {self.block_id} dropped protected content "
                    f"or changed its order: {span.text!r}"
                )
            cursor = position + len(span.text)


@dataclass(frozen=True)
class TranslationDocument:
    """Normalized document IR shared by text-bearing format adapters."""

    format: str
    source_hash: str
    blocks: tuple[TranslationBlock, ...]
    newline: str = "\n"
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.format not in {"txt", "md"}:
            raise ValueError(f"Unsupported normalized document format: {self.format}")
        if self.newline not in {"\n", "\r\n", "\r"}:
            raise ValueError("newline must be LF, CRLF, or CR.")
        ids = [block.block_id for block in self.blocks]
        if len(ids) != len(set(ids)):
            raise ValueError("Translation document block IDs must be unique.")

    @property
    def translatable_blocks(self) -> tuple[TranslationBlock, ...]:
        return tuple(block for block in self.blocks if block.translatable)

    def with_translations(
        self, translations: Mapping[str, str | None]
    ) -> "TranslationDocument":
        known_ids = {block.block_id for block in self.blocks}
        unknown_ids = sorted(set(translations) - known_ids)
        if unknown_ids:
            raise KeyError(f"Unknown document block IDs: {', '.join(unknown_ids)}")

        blocks = tuple(
            block.with_target_text(translations[block.block_id])
            if block.block_id in translations
            else block
            for block in self.blocks
        )
        return replace(self, blocks=blocks)
