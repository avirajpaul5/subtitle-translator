from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List

from subtitle_translator.defaults import merge_with_defaults
from subtitle_translator.formatter import subtitle_line_break
from subtitle_translator.glossary import (
    GlossaryConfig,
    apply_glossary_overrides,
    protect_terms,
    restore_terms,
)
from subtitle_translator.models import Cue, SubtitleDocument
from subtitle_translator.segmentation import merge_short_cues, split_translated_chunk
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

_CHECKPOINT_VERSION = 1


@dataclass
class TranslationSettings:
    source_lang: str = "en"
    target_lang: str = "bn"
    chunk_size: int = 12
    merge_min_chars: int = 0
    max_line_length: int = 42
    max_lines: int = 2


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

    chunks = merge_short_cues(document.cues, min_chars=settings.merge_min_chars)
    translated_cues: List[Cue] = [cue for cue in document.cues]
    completed_chunk_indices: set[int] = set()
    translation_cache: dict[str, str] = {}
    chunk_size = _effective_pipeline_chunk_size(translator, settings.chunk_size)

    total = len(chunks)
    total_chunks = max(1, (total + chunk_size - 1) // chunk_size)
    if checkpoint is not None and resume_from_checkpoint:
        loaded = _load_checkpoint(checkpoint, checkpoint_identity, len(document.cues), total_chunks)
        if loaded is not None:
            translated_cues, completed_chunk_indices = loaded
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

        batch = chunks[offset : offset + chunk_size]
        batch_texts = [item.text for item in batch]

        try:
            if progress_cb:
                progress_cb(
                    chunk_idx / total_chunks,
                    f"Translating {chunk_idx + 1}/{total_chunks} with {translator.display_name}...",
                )

            # Strip speaker labels (e.g. "POIROT: ") before sending to the model.
            # The model cannot reliably preserve arbitrary token formats, so we
            # never expose labels to it — we reattach them after translation.
            extracted = [_extract_speaker_label(t) for t in batch_texts]
            speaker_names = [s for s, _ in extracted]
            body_texts = [b for _, b in extracted]

            # Normalise ALL-CAPS stage directions so the model translates them
            # semantically rather than phonetically transliterating them.
            # "(BELL TOLLING)" → "(bell tolling)"; we track which were normalised
            # so we can wrap the translation back in parentheses if needed.
            normalised_texts, stage_flags = _normalise_stage_directions(body_texts)
            model_texts, wrapping_markup = _extract_wrapping_markup(normalised_texts)

            # Protect do-not-translate terms with bracket-free sentinels
            # (ZZID{n}ZZ). Angle-bracket formats like <dnt>...</dnt> and <IDn>
            # get garbled by the model — see scripts/spike_dnt.py for evidence.
            # All-alphabetic sentinels are passed through verbatim.
            protected_texts, replacements = protect_terms(
                model_texts, glossary.do_not_translate
            )
            translated_batch = _translate_protected_texts(
                protected_texts,
                translator,
                settings,
                translation_cache,
            )
            translated_batch = restore_terms(translated_batch, replacements)
            translated_batch = apply_glossary_overrides(translated_batch, glossary.glossary_map)
            translated_batch = _restore_wrapping_markup(translated_batch, wrapping_markup)

            # Re-wrap stage-direction translations in parentheses when the model
            # stripped them (it sometimes drops surrounding punctuation).
            translated_batch = _restore_stage_direction_parens(translated_batch, stage_flags)

            # Reattach speaker labels. Use full-name glossary lookup only —
            # NOT word-level substitution. Word-level was rewriting
            # "CHIEF INSPECTOR" into "CHIEF পরিদর্শক" because the default
            # Bengali glossary contains "inspector". User can still get
            # "POIROT" → "পয়রট" by adding the whole label as a glossary key.
            translated_batch = [
                "{}: {}".format(_translate_speaker_label(name, glossary.glossary_map), text)
                if name else text
                for name, text in zip(speaker_names, translated_batch)
            ]

            for merged, translated in zip(batch, translated_batch):
                split_texts = split_translated_chunk(translated, len(merged.cue_indices))
                for cue_idx, cue_text in zip(merged.cue_indices, split_texts):
                    formatted = subtitle_line_break(
                        cue_text,
                        max_line_length=settings.max_line_length,
                        max_lines=settings.max_lines,
                    )
                    translated_cues[cue_idx] = translated_cues[cue_idx].with_text(formatted)
        except Exception as exc:
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
            )
            raise TranslationInterruptedError(
                f"{message} Original error: {exc}",
                partial_document=partial_document,
                checkpoint_path=checkpoint,
                original_exception=exc,
            ) from exc

        completed_chunk_indices.add(chunk_idx)
        _save_checkpoint(
            checkpoint,
            identity=checkpoint_identity,
            document=document,
            translated_cues=translated_cues,
            completed_chunk_indices=completed_chunk_indices,
            total_chunks=total_chunks,
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
    )
    warnings = list(document.warnings) + [issue.formatted() for issue in issues]
    warnings.extend(str(w) for w in getattr(translator, "warnings", []))

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


def _translate_protected_texts(
    protected_texts: list[str],
    translator: BaseTranslator,
    settings: TranslationSettings,
    translation_cache: dict[str, str],
) -> list[str]:
    translated: list[str | None] = [None] * len(protected_texts)
    pending_texts: list[str] = []
    pending_positions_by_text: dict[str, list[int]] = {}

    for pos, text in enumerate(protected_texts):
        if not _needs_model_translation(text):
            translated[pos] = text
            continue
        cached = translation_cache.get(text)
        if cached is not None:
            translated[pos] = cached
            continue
        if text in pending_positions_by_text:
            pending_positions_by_text[text].append(pos)
        else:
            pending_positions_by_text[text] = [pos]
            pending_texts.append(text)

    if pending_texts:
        pending_translated = translator.translate_batch(
            pending_texts,
            source_lang=settings.source_lang,
            target_lang=settings.target_lang,
        )
        if len(pending_translated) != len(pending_texts):
            raise RuntimeError(
                "Translator returned a different number of lines than requested."
            )
        for original, out in zip(pending_texts, pending_translated):
            translation_cache[original] = out
            for pos in pending_positions_by_text[original]:
                translated[pos] = out

    return [text or "" for text in translated]


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
        "provider": translator.display_name,
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
) -> tuple[List[Cue], set[int]] | None:
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
    if not isinstance(cue_payloads, list) or len(cue_payloads) != cue_count:
        return None
    if not isinstance(completed, list):
        return None

    cues: List[Cue] = []
    try:
        for payload in cue_payloads:
            cues.append(_cue_from_checkpoint_payload(payload))
        completed_indices = {int(idx) for idx in completed}
    except (TypeError, ValueError, KeyError):
        return None

    return cues, completed_indices


def _save_checkpoint(
    checkpoint_path: Path | None,
    *,
    identity: dict,
    document: SubtitleDocument,
    translated_cues: List[Cue],
    completed_chunk_indices: set[int],
    total_chunks: int,
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
) -> SubtitleDocument:
    warnings = list(document.warnings)
    warnings.append(warning)
    warnings.extend(str(w) for w in getattr(translator, "warnings", []))
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
    """Lower-case ALL-CAPS parenthetical stage directions for better translation.

    Returns (normalised_texts, flags) where flags[i] is True when text[i] was
    a stage direction so the caller can re-wrap after translation.
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
