from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List

from subtitle_translator.contextual import (
    ContextSegment,
    ContextWindow,
    build_context_windows,
    fit_context_windows,
    translate_context_batch,
)
from subtitle_translator.defaults import merge_with_defaults
from subtitle_translator.formatter import subtitle_line_break
from subtitle_translator.glossary import (
    GlossaryConfig,
    apply_glossary_overrides,
    protect_terms,
    restore_terms,
    validate_restored_terms,
)
from subtitle_translator.models import Cue, SubtitleDocument
from subtitle_translator.translators.base import BaseTranslator
from subtitle_translator.validation import validate_translation

# Matches lines that are entirely a parenthetical stage direction in ALL CAPS,
# e.g. "(BELL TOLLING)" or "(SPEAKS FRENCH)" — including multi-word with spaces/hyphens.
_STAGE_DIRECTION_RE = re.compile(r"^\s*\(([A-Z][A-Z\s\-']+)\)\s*$")

# Matches an ALL-CAPS speaker label at the very start of a cue, e.g. "POIROT: "
# or "CHIEF INSPECTOR: ".  We extract this before translation so the model
# never has to pass an arbitrary token through — labels are re-attached after.
_SPEAKER_LABEL_RE = re.compile(r"^([A-Z][A-Z\s\-\.\']{0,30}):\s*")
_WRAPPING_TAG_RE = re.compile(
    r"^\s*(<(?P<tag>[A-Za-z][\w:-]*)(?:\s+[^>]*)?>)(?P<body>.*)(</(?P=tag)>)\s*$",
    re.DOTALL,
)
_SENTINEL_TOKEN_RE = re.compile(r"\bZZID\d+ZZ\b")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_ENTITY_RE = re.compile(r"&[A-Za-z][A-Za-z0-9]+;")
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)

_CHECKPOINT_VERSION = 2


@dataclass
class TranslationSettings:
    source_lang: str = "en"
    target_lang: str = "bn"
    chunk_size: int = 12
    context_window_chars: int = 700
    context_window_cues: int = 8
    max_line_length: int = 42
    max_lines: int = 2


@dataclass(frozen=True)
class _PreparedCue:
    key: str
    protected_text: str
    speaker_name: str | None
    wrapping_markup: tuple[str, str] | None
    is_stage_direction: bool
    preserve_verbatim: str | None
    replacements: dict[str, str]


class TranslationInterruptedError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        partial_document: SubtitleDocument,
        checkpoint_path: Path | None,
        original_exception: Exception,
    ) -> None:
        super().__init__(message)
        self.partial_document = partial_document
        self.checkpoint_path = checkpoint_path
        self.original_exception = original_exception


def translate_document(
    document: SubtitleDocument,
    translator: BaseTranslator,
    settings: TranslationSettings,
    glossary: GlossaryConfig,
    progress_cb: Callable[[float, str], None] | None = None,
    checkpoint_path: str | Path | None = None,
    resume_from_checkpoint: bool = True,
) -> SubtitleDocument:
    # Merge built-in defaults with the user-supplied glossary. The per-target-
    # language map covers common English nouns/verbs the model leaves
    # untranslated; the universal DNT list covers foreign-language phrases
    # ("Monsieur", "Señor", "Habibi") that are intentionally preserved.
    merged_map, merged_dnt = merge_with_defaults(
        glossary.glossary_map, glossary.do_not_translate, settings.target_lang
    )
    glossary = GlossaryConfig(glossary_map=merged_map, do_not_translate=merged_dnt)

    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    checkpoint_identity = _checkpoint_identity(document, settings, glossary, translator)

    # Preprocess each cue before packing context windows. This preserves every
    # cue's speaker label, stage-direction semantics, and wrapping markup; the
    # previous merge-first flow only handled those features on the first cue in
    # a merged group.
    prepared_cues = _prepare_cues(document.cues, glossary)
    context_segments = [
        ContextSegment(key=prepared.key, text=prepared.protected_text)
        for prepared in prepared_cues
    ]
    input_acceptance_cache: dict[str, bool] = {}

    def accepts_input(payload: str) -> bool:
        if payload not in input_acceptance_cache:
            input_acceptance_cache[payload] = translator.accepts_input(
                payload,
                settings.source_lang,
                settings.target_lang,
            )
        return input_acceptance_cache[payload]

    context_windows = build_context_windows(
        context_segments,
        max_chars=_effective_context_char_limit(translator, settings.context_window_chars),
        max_segments=settings.context_window_cues,
    )
    context_windows = fit_context_windows(
        context_windows,
        accepts_input,
    )
    prepared_by_key = {prepared.key: prepared for prepared in prepared_cues}
    translated_cues: List[Cue] = [cue for cue in document.cues]
    completed_chunk_indices: set[int] = set()
    alignment_warnings: list[str] = []
    chunk_size = _effective_pipeline_chunk_size(translator, settings.chunk_size)

    total = len(context_windows)
    total_chunks = max(1, (total + chunk_size - 1) // chunk_size)
    if checkpoint is not None and resume_from_checkpoint:
        loaded = _load_checkpoint(checkpoint, checkpoint_identity, len(document.cues), total_chunks)
        if loaded is not None:
            translated_cues, completed_chunk_indices, alignment_warnings = loaded
            if progress_cb and completed_chunk_indices:
                progress_cb(
                    0.0,
                    (
                        f"Resuming from checkpoint with "
                        f"{len(completed_chunk_indices)}/{total_chunks} chunks complete."
                    ),
                )

    for chunk_idx, offset in enumerate(range(0, total, chunk_size)):
        if chunk_idx in completed_chunk_indices:
            if progress_cb:
                progress_cb(
                    (chunk_idx + 1) / total_chunks,
                    f"Restored {chunk_idx + 1}/{total_chunks} chunks from checkpoint.",
                )
            continue

        batch = context_windows[offset : offset + chunk_size]

        try:
            if progress_cb:
                progress_cb(
                    chunk_idx / total_chunks,
                    f"Translating {chunk_idx + 1}/{total_chunks} with {translator.display_name}...",
                )

            model_windows: list[ContextWindow] = []
            translated_by_key: dict[str, str] = {}
            for window in batch:
                if any(
                    prepared_by_key[segment.key].preserve_verbatim is None
                    and _needs_model_translation(segment.text)
                    for segment in window.segments
                ):
                    model_windows.append(window)
                else:
                    translated_by_key.update(
                        {segment.key: segment.text for segment in window.segments}
                    )

            result = translate_context_batch(
                model_windows,
                translator,
                source_lang=settings.source_lang,
                target_lang=settings.target_lang,
            )
            translated_by_key.update(result.translations)
            if result.alignment_fallback_keys:
                cue_labels = [
                    str(document.cues[int(key)].index or int(key) + 1)
                    for key in result.alignment_fallback_keys
                ]
                alignment_warnings.append(
                    "Context alignment was not preserved for cue(s) "
                    + ", ".join(cue_labels)
                    + "; those cues were safely retried one at a time."
                )

            batch_prepared = [
                prepared_by_key[segment.key]
                for window in batch
                for segment in window.segments
            ]
            translated_batch = [
                translated_by_key[prepared.key] for prepared in batch_prepared
            ]
            translated_batch = apply_glossary_overrides(
                translated_batch, glossary.glossary_map
            )
            translated_batch = [
                restore_terms([text], prepared.replacements)[0]
                for prepared, text in zip(batch_prepared, translated_batch)
            ]
            for prepared, restored in zip(batch_prepared, translated_batch):
                validate_restored_terms(
                    prepared.protected_text,
                    restored,
                    prepared.replacements,
                )
            translated_batch = _restore_stage_direction_parens(
                translated_batch,
                [prepared.is_stage_direction for prepared in batch_prepared],
            )
            translated_batch = [
                prepared.preserve_verbatim
                if prepared.preserve_verbatim is not None
                else text
                for prepared, text in zip(batch_prepared, translated_batch)
            ]
            translated_batch = [
                "{}: {}".format(
                    _translate_speaker_label(prepared.speaker_name, glossary.glossary_map),
                    text,
                )
                if prepared.speaker_name
                else text
                for prepared, text in zip(batch_prepared, translated_batch)
            ]
            translated_batch = _restore_wrapping_markup(
                translated_batch,
                [prepared.wrapping_markup for prepared in batch_prepared],
            )

            for prepared, translated in zip(batch_prepared, translated_batch):
                cue_idx = int(prepared.key)
                formatted = subtitle_line_break(
                    translated,
                    max_line_length=settings.max_line_length,
                    max_lines=settings.max_lines,
                )
                translated_cues[cue_idx] = translated_cues[cue_idx].with_text(formatted)
        except Exception as exc:
            _extend_unique(
                alignment_warnings,
                [str(value) for value in getattr(translator, "warnings", [])],
            )
            message = (
                f"Translation interrupted after {len(completed_chunk_indices)}/"
                f"{total_chunks} chunks. Progress can be resumed with the same "
                "file, settings, glossary, and provider."
            )
            checkpoint_warning = _save_checkpoint(
                checkpoint,
                identity=checkpoint_identity,
                document=document,
                translated_cues=translated_cues,
                completed_chunk_indices=completed_chunk_indices,
                total_chunks=total_chunks,
                pipeline_warnings=alignment_warnings,
            )
            if checkpoint is not None and checkpoint_warning is None:
                message += f" Checkpoint saved to {checkpoint}."
            elif checkpoint_warning is not None:
                message += f" {checkpoint_warning}"
            partial_document = _build_partial_document(
                document,
                translated_cues,
                translator,
                warning=message,
                pipeline_warnings=alignment_warnings,
            )
            raise TranslationInterruptedError(
                f"{message} Original error: {exc}",
                partial_document=partial_document,
                checkpoint_path=checkpoint,
                original_exception=exc,
            ) from exc

        _extend_unique(
            alignment_warnings,
            [str(value) for value in getattr(translator, "warnings", [])],
        )
        completed_chunk_indices.add(chunk_idx)
        _save_checkpoint(
            checkpoint,
            identity=checkpoint_identity,
            document=document,
            translated_cues=translated_cues,
            completed_chunk_indices=completed_chunk_indices,
            total_chunks=total_chunks,
            pipeline_warnings=alignment_warnings,
        )

        if progress_cb:
            progress_cb(
                (chunk_idx + 1) / total_chunks,
                (
                    f"Translated {chunk_idx + 1}/{total_chunks} chunks with "
                    f"{translator.last_used_name}"
                ),
            )

    # Post-translation validation: flag (don't auto-fix) sentinel debris and
    # grammar patterns indicating poor translations. Surfaces in the
    # SubtitleDocument's `warnings` list — the GUI shows it in the status bar.
    issues = validate_translation(
        original_texts=[c.text for c in document.cues],
        translated_texts=[c.text for c in translated_cues],
        cue_numbers=[c.index for c in document.cues],
        target_lang=settings.target_lang,
        glossary_terms=glossary.glossary_map.keys(),
        protected_terms=glossary.do_not_translate,
    )
    warnings: list[str] = []
    _extend_unique(
        warnings,
        [
            *document.warnings,
            *alignment_warnings,
            *[issue.formatted() for issue in issues],
            *[str(value) for value in getattr(translator, "warnings", [])],
        ],
    )

    return SubtitleDocument(
        format=document.format,
        cues=translated_cues,
        header_lines=document.header_lines,
        warnings=warnings,
    )


def make_translation_checkpoint_path(
    source_name: str,
    source_content: str | bytes,
    directory: str | Path = ".translation-checkpoints",
) -> Path:
    source_bytes = (
        source_content.encode("utf-8")
        if isinstance(source_content, str)
        else source_content
    )
    digest = hashlib.sha256(source_bytes).hexdigest()[:16]
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(source_name).name).strip("._")
    safe_name = safe_name[:80] or "subtitle"
    return Path(directory) / f"{safe_name}.{digest}.json"


def _prepare_cues(
    cues: List[Cue],
    glossary: GlossaryConfig,
) -> list[_PreparedCue]:
    model_texts, wrapping_markup = _extract_wrapping_markup([cue.text for cue in cues])
    extracted = [_extract_speaker_label(text) for text in model_texts]
    speaker_names = [speaker for speaker, _ in extracted]
    body_texts = [body for _, body in extracted]
    normalised_texts, stage_flags = _normalise_stage_directions(body_texts)
    protected_with_replacements = [
        protect_terms([text], glossary.do_not_translate)
        for text in normalised_texts
    ]
    protected_texts = [protected[0][0] for protected in protected_with_replacements]
    replacements = [protected[1] for protected in protected_with_replacements]
    prepared = [
        _PreparedCue(
            key=str(index),
            protected_text=protected,
            speaker_name=speaker,
            wrapping_markup=wrapper,
            is_stage_direction=is_stage,
            preserve_verbatim=(body_texts[index] if is_stage else None),
            replacements=replacements[index],
        )
        for index, (protected, speaker, wrapper, is_stage) in enumerate(
            zip(protected_texts, speaker_names, wrapping_markup, stage_flags)
        )
    ]
    return prepared


def _effective_context_char_limit(
    translator: BaseTranslator,
    requested_chars: int,
) -> int:
    requested = max(1, int(requested_chars))
    provider_limit = getattr(translator, "max_input_chars", None)
    if not isinstance(provider_limit, int):
        provider = getattr(translator, "primary", translator)
        provider_limit = getattr(provider, "max_input_chars", None)
    if isinstance(provider_limit, int) and provider_limit > 0:
        # Leave headroom for boundary markers and provider-side normalization.
        return min(requested, max(1, int(provider_limit * 0.9)))
    return requested


def _effective_pipeline_chunk_size(translator: BaseTranslator, requested_chunk_size: int) -> int:
    requested = max(1, requested_chunk_size)
    preferred = getattr(translator, "pipeline_chunk_size", None)
    if isinstance(preferred, int) and preferred > 0:
        return min(requested, preferred)
    return requested


def _needs_model_translation(text: str) -> bool:
    compact = _SENTINEL_TOKEN_RE.sub("", text)
    compact = _HTML_TAG_RE.sub("", compact)
    compact = _HTML_ENTITY_RE.sub("", compact)
    return bool(_LETTER_RE.search(compact))


def _extract_wrapping_markup(texts: List[str]) -> tuple[List[str], List[tuple[str, str] | None]]:
    stripped: List[str] = []
    wrappers: List[tuple[str, str] | None] = []
    for text in texts:
        match = _WRAPPING_TAG_RE.match(text)
        if match:
            stripped.append(match.group("body").strip())
            wrappers.append((match.group(1), match.group(4)))
        else:
            stripped.append(text)
            wrappers.append(None)
    return stripped, wrappers


def _restore_wrapping_markup(
    texts: List[str],
    wrappers: List[tuple[str, str] | None],
) -> List[str]:
    restored: List[str] = []
    for text, wrapper in zip(texts, wrappers):
        if wrapper is None:
            restored.append(text)
            continue
        start, end = wrapper
        restored.append(f"{start}{text.strip()}{end}" if text.strip() else f"{start}{end}")
    return restored


def _checkpoint_identity(
    document: SubtitleDocument,
    settings: TranslationSettings,
    glossary: GlossaryConfig,
    translator: BaseTranslator,
) -> dict:
    return {
        "source_hash": _document_hash(document),
        "settings": asdict(settings),
        "glossary_hash": _glossary_hash(glossary),
        "provider": translator.checkpoint_fingerprint,
    }


def _document_hash(document: SubtitleDocument) -> str:
    payload = {
        "format": document.format,
        "header_lines": document.header_lines,
        "cues": [
            {
                "index": cue.index,
                "start": cue.start,
                "end": cue.end,
                "settings": cue.settings,
                "identifier": cue.identifier,
                "text_lines": cue.text_lines,
            }
            for cue in document.cues
        ],
    }
    return _stable_hash(payload)


def _glossary_hash(glossary: GlossaryConfig) -> str:
    return _stable_hash(
        {
            "glossary_map": glossary.glossary_map,
            "do_not_translate": glossary.do_not_translate,
        }
    )


def _stable_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_checkpoint(
    checkpoint_path: Path,
    identity: dict,
    cue_count: int,
    total_chunks: int,
) -> tuple[List[Cue], set[int], list[str]] | None:
    if not checkpoint_path.exists():
        return None

    try:
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if data.get("version") != _CHECKPOINT_VERSION:
        return None
    if data.get("identity") != identity:
        return None
    if data.get("total_chunks") != total_chunks:
        return None

    cue_payloads = data.get("translated_cues")
    completed = data.get("completed_chunk_indices")
    stored_warnings = data.get("pipeline_warnings", [])
    if not isinstance(cue_payloads, list) or len(cue_payloads) != cue_count:
        return None
    if not isinstance(completed, list):
        return None
    if not isinstance(stored_warnings, list):
        return None

    cues: List[Cue] = []
    try:
        for payload in cue_payloads:
            cues.append(_cue_from_checkpoint_payload(payload))
        completed_indices = {int(idx) for idx in completed}
    except (TypeError, ValueError, KeyError):
        return None

    return cues, completed_indices, [str(value) for value in stored_warnings]


def _save_checkpoint(
    checkpoint_path: Path | None,
    *,
    identity: dict,
    document: SubtitleDocument,
    translated_cues: List[Cue],
    completed_chunk_indices: set[int],
    total_chunks: int,
    pipeline_warnings: list[str],
) -> str | None:
    if checkpoint_path is None:
        return None

    payload = {
        "version": _CHECKPOINT_VERSION,
        "identity": identity,
        "format": document.format,
        "header_lines": document.header_lines,
        "total_chunks": total_chunks,
        "completed_chunk_indices": sorted(completed_chunk_indices),
        "pipeline_warnings": pipeline_warnings,
        "translated_cues": [_cue_to_checkpoint_payload(cue) for cue in translated_cues],
    }

    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(checkpoint_path)
    except OSError as exc:
        return f"Could not save checkpoint: {exc}"
    return None


def _cue_to_checkpoint_payload(cue: Cue) -> dict:
    return {
        "index": cue.index,
        "start": cue.start,
        "end": cue.end,
        "text_lines": cue.text_lines,
        "identifier": cue.identifier,
        "settings": cue.settings,
    }


def _cue_from_checkpoint_payload(payload: dict) -> Cue:
    text_lines = payload["text_lines"]
    if not isinstance(text_lines, list):
        raise ValueError("Invalid checkpoint cue text_lines")
    return Cue(
        index=payload.get("index"),
        start=str(payload["start"]),
        end=str(payload["end"]),
        text_lines=[str(line) for line in text_lines],
        identifier=payload.get("identifier"),
        settings=payload.get("settings"),
    )


def _build_partial_document(
    document: SubtitleDocument,
    translated_cues: List[Cue],
    translator: BaseTranslator,
    *,
    warning: str,
    pipeline_warnings: list[str],
) -> SubtitleDocument:
    warnings: list[str] = []
    _extend_unique(
        warnings,
        [
            *document.warnings,
            *pipeline_warnings,
            warning,
            *[str(value) for value in getattr(translator, "warnings", [])],
        ],
    )
    return SubtitleDocument(
        format=document.format,
        cues=translated_cues,
        header_lines=document.header_lines,
        warnings=warnings,
    )


def _extract_speaker_label(text: str):
    """Return (speaker_name, body) if cue starts with ALL-CAPS SPEAKER:, else (None, text)."""
    m = _SPEAKER_LABEL_RE.match(text)
    if m:
        return m.group(1).strip(), text[m.end():]
    return None, text


def _translate_speaker_label(name: str, glossary_map: dict) -> str:
    """Look up the *full* speaker label as a single key (case-insensitive).
    Avoids word-by-word substitution that would translate component words
    like "INSPECTOR" inside "CHIEF INSPECTOR" via a generic default glossary."""
    target = name.casefold()
    for k, v in glossary_map.items():
        if k.casefold() == target:
            return v
    return name


def _normalise_stage_directions(texts: List[str]):
    """Lower-case ALL-CAPS directions when they are included as model context.

    The original text is restored verbatim after translation. Returns
    ``(normalised_texts, flags)`` so callers can identify these context-only
    units without exposing model output for them.
    """
    normalised: List[str] = []
    flags: List[bool] = []
    for text in texts:
        m = _STAGE_DIRECTION_RE.match(text)
        if m:
            normalised.append(f"({m.group(1).lower()})")
            flags.append(True)
        else:
            normalised.append(text)
            flags.append(False)
    return normalised, flags


def _restore_stage_direction_parens(texts: List[str], flags: List[bool]) -> List[str]:
    """Ensure translated stage directions are wrapped in parentheses."""
    out: List[str] = []
    for text, is_stage in zip(texts, flags):
        if is_stage:
            t = text.strip()
            if t and not (t.startswith("(") and t.endswith(")")):
                t = f"({t})"
            out.append(t)
        else:
            out.append(text)
    return out


def _extend_unique(destination: list[str], values: List[str]) -> None:
    for value in values:
        if value and value not in destination:
            destination.append(value)
