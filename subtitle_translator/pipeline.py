from __future__ import annotations

import re
from dataclasses import dataclass
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
from subtitle_translator.validation import validate_translation

# Matches lines that are entirely a parenthetical stage direction in ALL CAPS,
# e.g. "(BELL TOLLING)" or "(SPEAKS FRENCH)" — including multi-word with spaces/hyphens.
_STAGE_DIRECTION_RE = re.compile(r"^\s*\(([A-Z][A-Z\s\-']+)\)\s*$")

# Matches an ALL-CAPS speaker label at the very start of a cue, e.g. "POIROT: "
# or "CHIEF INSPECTOR: ".  We extract this before translation so the model
# never has to pass an arbitrary token through — labels are re-attached after.
_SPEAKER_LABEL_RE = re.compile(r"^([A-Z][A-Z\s\-\.\']{0,30}):\s*")

from subtitle_translator.translators.base import BaseTranslator


@dataclass
class TranslationSettings:
    source_lang: str = "en"
    target_lang: str = "bn"
    chunk_size: int = 12
    merge_min_chars: int = 0
    max_line_length: int = 42
    max_lines: int = 2


def translate_document(
    document: SubtitleDocument,
    translator: BaseTranslator,
    settings: TranslationSettings,
    glossary: GlossaryConfig,
    progress_cb: Callable[[float, str], None] | None = None,
) -> SubtitleDocument:
    # Merge built-in defaults with the user-supplied glossary. The per-target-
    # language map covers common English nouns/verbs the model leaves
    # untranslated; the universal DNT list covers foreign-language phrases
    # ("Monsieur", "Señor", "Habibi") that are intentionally preserved.
    merged_map, merged_dnt = merge_with_defaults(
        glossary.glossary_map, glossary.do_not_translate, settings.target_lang
    )
    glossary = GlossaryConfig(glossary_map=merged_map, do_not_translate=merged_dnt)

    chunks = merge_short_cues(document.cues, min_chars=settings.merge_min_chars)
    translated_cues: List[Cue] = [cue for cue in document.cues]

    total = len(chunks)
    total_chunks = max(1, (total + settings.chunk_size - 1) // settings.chunk_size)
    for chunk_idx, offset in enumerate(range(0, total, settings.chunk_size)):
        batch = chunks[offset : offset + settings.chunk_size]
        batch_texts = [item.text for item in batch]

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

        # Protect do-not-translate terms with bracket-free sentinels
        # (ZZID{n}ZZ). Angle-bracket formats like <dnt>...</dnt> and <IDn>
        # get garbled by the model — see scripts/spike_dnt.py for evidence.
        # All-alphabetic sentinels are passed through verbatim.
        protected_texts, replacements = protect_terms(
            normalised_texts, glossary.do_not_translate
        )
        translated_batch = translator.translate_batch(
            protected_texts,
            source_lang=settings.source_lang,
            target_lang=settings.target_lang,
        )
        translated_batch = restore_terms(translated_batch, replacements)
        translated_batch = apply_glossary_overrides(translated_batch, glossary.glossary_map)

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
