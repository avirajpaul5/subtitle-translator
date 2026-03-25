from __future__ import annotations

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
from subtitle_translator.translators.base import BaseTranslator


@dataclass
class TranslationSettings:
    source_lang: str = "en"
    target_lang: str = "bn"
    chunk_size: int = 12
    merge_min_chars: int = 60
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

        protected_texts, replacements = protect_terms(batch_texts, all_terms)
        translated_batch = translator.translate_batch(
            protected_texts,
            source_lang=settings.source_lang,
            target_lang=settings.target_lang,
        )
        translated_batch = restore_terms(translated_batch, replacements)
        translated_batch = apply_glossary_overrides(translated_batch, glossary.glossary_map)

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
