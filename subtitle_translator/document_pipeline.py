from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

from subtitle_translator.contextual import (
    ContextSegment,
    ContextWindow,
    build_context_windows,
    fit_context_windows,
    translate_context_batch,
)
from subtitle_translator.defaults import merge_with_defaults
from subtitle_translator.documents import TranslationBlock, TranslationDocument
from subtitle_translator.glossary import (
    GlossaryConfig,
    apply_glossary_overrides,
    restore_terms,
    validate_restored_terms,
)
from subtitle_translator.translators.base import BaseTranslator
from subtitle_translator.validation import validate_translation


_CHECKPOINT_VERSION = 2
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?।。！？])\s+")
_LAYOUT_SENTINEL_RE = re.compile(r"ZZWS\d+ZZ")


@dataclass(frozen=True)
class DocumentTranslationSettings:
    source_lang: str = "en"
    target_lang: str = "bn"
    chunk_size: int = 8
    context_window_chars: int = 700
    context_window_blocks: int = 6


class DocumentTranslationInterruptedError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        partial_document: TranslationDocument,
        checkpoint_path: Path | None,
        original_exception: Exception,
    ) -> None:
        super().__init__(message)
        self.partial_document = partial_document
        self.checkpoint_path = checkpoint_path
        self.original_exception = original_exception


@dataclass(frozen=True)
class _PreparedBlock:
    block: TranslationBlock
    piece_keys: tuple[str, ...]
    replacements_by_key: dict[str, dict[str, str]]
    layout_replacements_by_key: dict[str, dict[str, str]]
    protected_source_by_key: dict[str, str]


class _ReplacementRegistry:
    def __init__(self) -> None:
        self._sentinel_by_text: dict[str, str] = {}
        self.replacements: dict[str, str] = {}

    def token_for(self, text: str) -> str:
        existing = self._sentinel_by_text.get(text)
        if existing is not None:
            return existing
        sentinel = f"ZZID{len(self._sentinel_by_text)}ZZ"
        self._sentinel_by_text[text] = sentinel
        self.replacements[sentinel] = text
        return sentinel


class _LayoutRegistry:
    """Opaque markers for horizontal layout that must round-trip byte-for-byte."""

    def __init__(self) -> None:
        self._sentinel_by_text: dict[str, str] = {}
        self.replacements: dict[str, str] = {}

    def token_for(self, text: str) -> str:
        existing = self._sentinel_by_text.get(text)
        if existing is not None:
            return existing
        sentinel = f"ZZWS{len(self._sentinel_by_text)}ZZ"
        self._sentinel_by_text[text] = sentinel
        self.replacements[sentinel] = text
        return sentinel


def translate_text_document(
    document: TranslationDocument,
    translator: BaseTranslator,
    settings: DocumentTranslationSettings,
    glossary: GlossaryConfig,
    progress_cb: Callable[[float, str], None] | None = None,
    checkpoint_path: str | Path | None = None,
    resume_from_checkpoint: bool = True,
) -> TranslationDocument:
    """Translate a TXT/Markdown IR without passing structural blocks to the model."""

    merged_map, merged_dnt = merge_with_defaults(
        glossary.glossary_map,
        glossary.do_not_translate,
        settings.target_lang,
    )
    glossary = GlossaryConfig(merged_map, merged_dnt)
    char_limit = _effective_context_char_limit(translator, settings.context_window_chars)
    accepts_input = _memoized_input_acceptor(
        translator,
        settings.source_lang,
        settings.target_lang,
    )
    prepared, segments = _prepare_document(
        document,
        glossary.do_not_translate,
        piece_char_limit=max(1, char_limit - 24),
        accepts_input=accepts_input,
    )
    windows = _plan_document_windows(
        document,
        segments,
        max_chars=char_limit,
        max_segments=settings.context_window_blocks,
    )
    windows = fit_context_windows(
        windows,
        accepts_input,
    )
    prepared_by_id = {item.block.block_id: item for item in prepared}

    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    identity = _checkpoint_identity(document, settings, glossary, translator)
    piece_translations: dict[str, str] = {}
    completed_window_indices: set[int] = set()
    alignment_warnings: list[str] = []
    chunk_size = _effective_pipeline_chunk_size(translator, settings.chunk_size)
    total_windows = len(windows)
    total_chunks = max(1, (total_windows + chunk_size - 1) // chunk_size)

    if checkpoint is not None and resume_from_checkpoint:
        loaded = _load_checkpoint(checkpoint, identity, total_windows)
        if loaded is not None:
            piece_translations, completed_window_indices, alignment_warnings = loaded
            if progress_cb and completed_window_indices:
                progress_cb(
                    0.0,
                    f"Resuming with {len(completed_window_indices)}/{total_windows} context windows complete.",
                )

    for chunk_idx, offset in enumerate(range(0, total_windows, chunk_size)):
        batch_indices = list(range(offset, min(offset + chunk_size, total_windows)))
        if batch_indices and all(index in completed_window_indices for index in batch_indices):
            if progress_cb:
                progress_cb(
                    (chunk_idx + 1) / total_chunks,
                    f"Restored {chunk_idx + 1}/{total_chunks} chunks from checkpoint.",
                )
            continue

        batch = [windows[index] for index in batch_indices]
        translations_before_batch = dict(piece_translations)
        try:
            if progress_cb:
                progress_cb(
                    chunk_idx / total_chunks,
                    f"Translating document chunk {chunk_idx + 1}/{total_chunks} with {translator.display_name}...",
                )
            result = translate_context_batch(
                batch,
                translator,
                source_lang=settings.source_lang,
                target_lang=settings.target_lang,
            )
            piece_translations.update(result.translations)
            if result.alignment_fallback_keys:
                block_ids = sorted(
                    {key.rsplit(":p", 1)[0] for key in result.alignment_fallback_keys}
                )
                alignment_warnings.append(
                    "Context alignment was not preserved for document block(s) "
                    + ", ".join(block_ids)
                    + "; those pieces were safely retried one at a time."
                )
            # Materialize before marking the window complete. This verifies
            # that every protected Markdown span can be restored, preventing
            # an invalid completed checkpoint from becoming permanently
            # unresumable.
            _materialize_document(
                document,
                prepared_by_id,
                piece_translations,
                glossary.glossary_map,
                glossary.do_not_translate,
                alignment_warnings,
                target_lang=settings.target_lang,
            )
        except Exception as exc:
            piece_translations = translations_before_batch
            _extend_unique(
                alignment_warnings,
                [str(value) for value in getattr(translator, "warnings", [])],
            )
            partial = _materialize_document(
                document,
                prepared_by_id,
                piece_translations,
                glossary.glossary_map,
                glossary.do_not_translate,
                alignment_warnings,
                target_lang=settings.target_lang,
            )
            checkpoint_warning = _save_checkpoint(
                checkpoint,
                identity,
                total_windows,
                piece_translations,
                completed_window_indices,
                alignment_warnings,
            )
            message = (
                f"Document translation interrupted after {len(completed_window_indices)}/"
                f"{total_windows} context windows."
            )
            if checkpoint is not None:
                if checkpoint_warning is None:
                    message += f" Checkpoint saved to {checkpoint}."
                else:
                    message += f" {checkpoint_warning}"
            raise DocumentTranslationInterruptedError(
                f"{message} Original error: {exc}",
                partial_document=partial,
                checkpoint_path=checkpoint,
                original_exception=exc,
            ) from exc

        _extend_unique(
            alignment_warnings,
            [str(value) for value in getattr(translator, "warnings", [])],
        )
        completed_window_indices.update(batch_indices)
        _save_checkpoint(
            checkpoint,
            identity,
            total_windows,
            piece_translations,
            completed_window_indices,
            alignment_warnings,
        )
        if progress_cb:
            progress_cb(
                (chunk_idx + 1) / total_chunks,
                f"Translated document chunk {chunk_idx + 1}/{total_chunks} with {translator.last_used_name}",
            )

    _extend_unique(
        alignment_warnings,
        [str(value) for value in getattr(translator, "warnings", [])],
    )
    return _materialize_document(
        document,
        prepared_by_id,
        piece_translations,
        glossary.glossary_map,
        glossary.do_not_translate,
        alignment_warnings,
        target_lang=settings.target_lang,
    )


def _prepare_document(
    document: TranslationDocument,
    do_not_translate: list[str],
    *,
    piece_char_limit: int,
    accepts_input: Callable[[str], bool],
) -> tuple[list[_PreparedBlock], dict[str, list[ContextSegment]]]:
    prepared: list[_PreparedBlock] = []
    segments_by_block: dict[str, list[ContextSegment]] = {}

    for block in document.translatable_blocks:
        registry = _ReplacementRegistry()
        layout_registry = _LayoutRegistry()
        masked = _mask_protected_spans(block, registry, layout_registry)
        masked = _mask_terms(masked, do_not_translate, registry)
        pieces = _split_long_text(masked, piece_char_limit)
        pieces = _split_pieces_to_fit(
            pieces,
            accepts_input,
        )
        context_segments = [
            ContextSegment(key=f"{block.block_id}:p{index:04d}", text=piece)
            for index, piece in enumerate(pieces, start=1)
        ]
        segments_by_block[block.block_id] = context_segments
        replacements_by_key = {
            segment.key: {
                sentinel: original
                for sentinel, original in registry.replacements.items()
                if sentinel in segment.text
            }
            for segment in context_segments
        }
        layout_replacements_by_key = {
            segment.key: {
                sentinel: original
                for sentinel, original in layout_registry.replacements.items()
                if sentinel in segment.text
            }
            for segment in context_segments
        }
        prepared.append(
            _PreparedBlock(
                block=block,
                piece_keys=tuple(segment.key for segment in context_segments),
                replacements_by_key=replacements_by_key,
                layout_replacements_by_key=layout_replacements_by_key,
                protected_source_by_key={
                    segment.key: segment.text for segment in context_segments
                },
            )
        )
    return prepared, segments_by_block


def _mask_protected_spans(
    block: TranslationBlock,
    registry: _ReplacementRegistry,
    layout_registry: _LayoutRegistry,
) -> str:
    if not block.spans:
        return block.source_text
    parts: list[str] = []
    for span in block.spans:
        if span.translatable:
            parts.append(_mask_horizontal_layout(span.text, layout_registry))
        else:
            parts.append(f" {registry.token_for(span.text)} ")
    return "".join(parts).strip()


def _mask_terms(
    text: str,
    terms: Iterable[str],
    registry: _ReplacementRegistry,
) -> str:
    output = text
    for term in sorted(
        {term for term in terms if term},
        key=lambda value: (-len(value), value.casefold(), value),
    ):
        pattern = re.compile(rf"\b{re.escape(term)}\b", flags=re.IGNORECASE)
        if not pattern.search(output):
            continue
        output = pattern.sub(
            lambda match: registry.token_for(match.group(0)),
            output,
        )
    return output


def _mask_horizontal_layout(text: str, registry: _LayoutRegistry) -> str:
    """Protect tabs and repeated spaces instead of asking MT to reproduce them."""

    return re.sub(
        r"(?:[ \t]*\t[ \t]*| {2,})",
        lambda match: f" {registry.token_for(match.group(0))} ",
        text,
    )


def _split_long_text(text: str, limit: int) -> list[str]:
    if not text or len(text) <= limit:
        return [text]
    sentences = [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(text) if part.strip()]
    if len(sentences) <= 1:
        return _hard_split(text, limit)

    pieces: list[str] = []
    pending = ""
    for sentence in sentences:
        candidate = f"{pending} {sentence}".strip() if pending else sentence
        if len(candidate) <= limit:
            pending = candidate
            continue
        if pending:
            pieces.append(pending)
        if len(sentence) <= limit:
            pending = sentence
        else:
            pieces.extend(_hard_split(sentence, limit))
            pending = ""
    if pending:
        pieces.append(pending)
    return pieces


def _hard_split(text: str, limit: int) -> list[str]:
    pieces: list[str] = []
    remaining = text.strip()
    while remaining:
        if len(remaining) <= limit:
            pieces.append(remaining)
            break
        split_at = remaining.rfind(" ", 0, limit + 1)
        if split_at < max(1, limit // 2):
            split_at = limit
        pieces.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    return pieces


def _split_pieces_to_fit(
    pieces: Iterable[str],
    accepts_input: Callable[[str], bool],
) -> list[str]:
    fitted: list[str] = []

    def add(piece: str) -> None:
        if accepts_input(piece):
            fitted.append(piece)
            return
        if len(piece) <= 1:
            raise ValueError("A document character exceeds the provider token budget.")
        smaller = _hard_split(piece, max(1, len(piece) // 2))
        if len(smaller) == 1 and smaller[0] == piece:
            midpoint = max(1, len(piece) // 2)
            smaller = [piece[:midpoint], piece[midpoint:]]
        for candidate in smaller:
            if candidate:
                add(candidate)

    for piece in pieces:
        add(piece)
    return fitted


def _plan_document_windows(
    document: TranslationDocument,
    segments_by_block: dict[str, list[ContextSegment]],
    *,
    max_chars: int,
    max_segments: int,
) -> list[ContextWindow]:
    windows: list[ContextWindow] = []
    semantic_group: list[ContextSegment] = []

    def flush() -> None:
        if semantic_group:
            windows.extend(
                build_context_windows(
                    semantic_group,
                    max_chars=max_chars,
                    max_segments=max_segments,
                )
            )
            semantic_group.clear()

    for block in document.blocks:
        if block.kind == "heading":
            flush()
        if not block.translatable:
            # Blank lines are presentation structure, not semantic boundaries.
            # Keeping them out of the payload while allowing adjacent prose to
            # share a window gives the model paragraph-level context without
            # risking any change to the original line layout.
            if block.kind != "separator":
                flush()
            continue
        semantic_group.extend(segments_by_block.get(block.block_id, []))
    flush()
    return windows


def _materialize_document(
    document: TranslationDocument,
    prepared_by_id: dict[str, _PreparedBlock],
    piece_translations: dict[str, str],
    glossary_map: dict[str, str],
    protected_terms: Iterable[str],
    warnings: list[str],
    *,
    target_lang: str,
) -> TranslationDocument:
    translated_blocks: dict[str, str] = {}
    for block_id, prepared in prepared_by_id.items():
        if not prepared.piece_keys or not all(
            key in piece_translations for key in prepared.piece_keys
        ):
            continue
        restored_pieces: list[str] = []
        for key in prepared.piece_keys:
            piece = apply_glossary_overrides(
                [piece_translations[key].strip()], glossary_map
            )[0]
            piece = _restore_horizontal_layout(
                piece,
                prepared.protected_source_by_key[key],
                prepared.layout_replacements_by_key.get(key, {}),
            )
            piece = restore_terms(
                [piece],
                prepared.replacements_by_key.get(key, {}),
                normalize_spacing=False,
            )[0]
            validate_restored_terms(
                prepared.protected_source_by_key[key],
                piece,
                prepared.replacements_by_key.get(key, {}),
            )
            restored_pieces.append(piece)
        restored = " ".join(restored_pieces).strip()
        restored = _restore_protected_spacing(restored, prepared.block)
        translated_blocks[block_id] = restored

    translated = document.with_translations(translated_blocks)
    completed_blocks = [
        block
        for block in translated.translatable_blocks
        if block.target_text is not None
    ]
    issues = validate_translation(
        original_texts=[
            "".join(span.text for span in block.spans if span.translatable)
            for block in completed_blocks
        ],
        translated_texts=[
            _without_protected_spans(block.target_text or "", block)
            for block in completed_blocks
        ],
        cue_numbers=[None] * len(completed_blocks),
        target_lang=target_lang,
        glossary_terms=glossary_map.keys(),
        protected_terms=protected_terms,
    )
    validation_warnings = [
        f"block {completed_blocks[issue.cue_index].block_id}: {', '.join(issue.issues)}"
        for issue in issues
    ]
    combined_warnings: list[str] = []
    _extend_unique(
        combined_warnings,
        [*document.warnings, *warnings, *validation_warnings],
    )
    return replace(translated, warnings=tuple(combined_warnings))


def _without_protected_spans(text: str, block: TranslationBlock) -> str:
    """Remove exact Markdown syntax/code/URL spans from review heuristics."""

    if not block.protected_spans:
        return text
    parts: list[str] = []
    cursor = 0
    for span in block.protected_spans:
        position = text.find(span.text, cursor)
        if position < 0:
            # The document integrity contract validates this before review;
            # retain the text here defensively if a custom block bypassed it.
            return text
        parts.append(text[cursor:position])
        cursor = position + len(span.text)
    parts.append(text[cursor:])
    return "".join(parts)


def _restore_horizontal_layout(
    text: str,
    protected_source: str,
    replacements: dict[str, str],
) -> str:
    """Restore layout sentinels exactly and reject any structural drift."""

    expected_order = _LAYOUT_SENTINEL_RE.findall(protected_source)
    actual_order = _LAYOUT_SENTINEL_RE.findall(text)
    if actual_order != expected_order:
        raise ValueError(
            "Horizontal layout markers were missing, duplicated, or reordered. "
            "The translation was not accepted."
        )
    for sentinel, original in replacements.items():
        expected = protected_source.count(sentinel)
        pattern = re.compile(rf"[ \t]*{re.escape(sentinel)}[ \t]*")
        actual = len(pattern.findall(text))
        if actual != expected:
            raise ValueError(
                f"Horizontal layout marker {sentinel!r} restored {actual} time(s); "
                f"expected {expected}. The translation was not accepted."
            )
        text = pattern.sub(lambda _match, value=original: value, text)
    return text


def _checkpoint_identity(
    document: TranslationDocument,
    settings: DocumentTranslationSettings,
    glossary: GlossaryConfig,
    translator: BaseTranslator,
) -> dict:
    return {
        "source_hash": document.source_hash,
        "format": document.format,
        "settings": asdict(settings),
        "glossary_hash": _stable_hash(
            {
                "glossary_map": glossary.glossary_map,
                "do_not_translate": glossary.do_not_translate,
            }
        ),
        "provider": translator.checkpoint_fingerprint,
    }


def _load_checkpoint(
    path: Path,
    identity: dict,
    total_windows: int,
) -> tuple[dict[str, str], set[int], list[str]] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        data.get("version") != _CHECKPOINT_VERSION
        or data.get("identity") != identity
        or data.get("total_windows") != total_windows
    ):
        return None
    pieces = data.get("piece_translations")
    completed = data.get("completed_window_indices")
    warnings = data.get("warnings", [])
    if not isinstance(pieces, dict) or not isinstance(completed, list):
        return None
    return (
        {str(key): str(value) for key, value in pieces.items()},
        {int(value) for value in completed},
        [str(value) for value in warnings] if isinstance(warnings, list) else [],
    )


def _save_checkpoint(
    path: Path | None,
    identity: dict,
    total_windows: int,
    pieces: dict[str, str],
    completed: set[int],
    warnings: list[str],
) -> str | None:
    if path is None:
        return None
    payload = {
        "version": _CHECKPOINT_VERSION,
        "identity": identity,
        "total_windows": total_windows,
        "piece_translations": pieces,
        "completed_window_indices": sorted(completed),
        "warnings": warnings,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)
    except OSError as exc:
        return f"Could not save checkpoint: {exc}"
    return None


def _effective_context_char_limit(translator: BaseTranslator, requested: int) -> int:
    limit = max(1, int(requested))
    provider_limit = getattr(translator, "max_input_chars", None)
    if not isinstance(provider_limit, int):
        provider = getattr(translator, "primary", translator)
        provider_limit = getattr(provider, "max_input_chars", None)
    if isinstance(provider_limit, int) and provider_limit > 0:
        return min(limit, max(1, int(provider_limit * 0.9)))
    return limit


def _effective_pipeline_chunk_size(translator: BaseTranslator, requested: int) -> int:
    size = max(1, int(requested))
    preferred = getattr(translator, "pipeline_chunk_size", None)
    if isinstance(preferred, int) and preferred > 0:
        return min(size, preferred)
    return size


def _stable_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _memoized_input_acceptor(
    translator: BaseTranslator,
    source_lang: str,
    target_lang: str,
) -> Callable[[str], bool]:
    cache: dict[str, bool] = {}

    def accepts(payload: str) -> bool:
        if payload not in cache:
            cache[payload] = translator.accepts_input(
                payload,
                source_lang,
                target_lang,
            )
        return cache[payload]

    return accepts


def _restore_protected_spacing(text: str, block: TranslationBlock) -> str:
    """Reapply the source's exact horizontal spacing around protected spans."""

    cursor = 0
    for span in block.protected_spans:
        position = text.find(span.text, cursor)
        if position < 0:
            continue

        source_left = span.start
        while source_left > 0 and block.source_text[source_left - 1] in " \t":
            source_left -= 1
        desired_left = block.source_text[source_left : span.start]
        target_left = position
        while target_left > 0 and text[target_left - 1] in " \t":
            target_left -= 1
        text = text[:target_left] + desired_left + text[position:]
        position = target_left + len(desired_left)

        end = position + len(span.text)
        source_right = span.end
        while (
            source_right < len(block.source_text)
            and block.source_text[source_right] in " \t"
        ):
            source_right += 1
        desired_right = block.source_text[span.end : source_right]
        target_right = end
        while target_right < len(text) and text[target_right] in " \t":
            target_right += 1
        text = text[:end] + desired_right + text[target_right:]

        cursor = end + len(desired_right)
    return text


def _extend_unique(destination: list[str], values: Iterable[str]) -> None:
    for value in values:
        if value and value not in destination:
            destination.append(value)
