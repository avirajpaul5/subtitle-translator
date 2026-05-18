from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List

from subtitle_translator.formatter import subtitle_line_break
from subtitle_translator.glossary import (
    GlossaryConfig,
    apply_glossary_overrides,
    protect_terms,
    restore_terms,
)
from subtitle_translator.models import Cue, SubtitleDocument
from subtitle_translator.segmentation import merge_short_cues, split_translated_chunk

# Matches lines that are entirely a parenthetical stage direction in ALL CAPS,
# e.g. "(BELL TOLLING)" or "(SPEAKS FRENCH)" — including multi-word with spaces/hyphens.
_STAGE_DIRECTION_RE = re.compile(r"^\s*\(([A-Z][A-Z\s\-']+)\)\s*$")
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
    chunks = merge_short_cues(document.cues, min_chars=settings.merge_min_chars)
    translated_cues: List[Cue] = [cue for cue in document.cues]

    all_terms = list(glossary.do_not_translate) + list(glossary.glossary_map.keys())

    total = len(chunks)
    for offset in range(0, total, settings.chunk_size):
        batch = chunks[offset : offset + settings.chunk_size]
        batch_texts = [item.text for item in batch]

        # Normalise ALL-CAPS stage directions so the model translates them
        # semantically rather than phonetically transliterating them.
        # "(BELL TOLLING)" → "(bell tolling)"; we track which were normalised
        # so we can wrap the translation back in parentheses if needed.
        normalised_texts, stage_flags = _normalise_stage_directions(batch_texts)

        protected_texts, replacements = protect_terms(normalised_texts, all_terms)
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
            progress = min(1.0, (offset + len(batch)) / max(total, 1))
            progress_cb(progress, f"Translated {offset + len(batch)}/{total} merged chunks")

    return SubtitleDocument(format=document.format, cues=translated_cues, header_lines=document.header_lines)


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
