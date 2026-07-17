from __future__ import annotations

import pytest

from subtitle_translator.contextual import (
    ContextAlignmentError,
    ContextSegment,
    build_context_windows,
    decode_context_segments,
    encode_context_segments,
    fit_context_windows,
    translate_context_batch,
)
from subtitle_translator.translators.base import BaseTranslator


class _MarkerAwareTranslator(BaseTranslator):
    def translate_batch(self, texts, source_lang: str, target_lang: str):
        return [text.replace("Hello", "হ্যালো").replace("World", "বিশ্ব") for text in texts]


class _MarkerBreakingTranslator(BaseTranslator):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def translate_batch(self, texts, source_lang: str, target_lang: str):
        materialized = list(texts)
        self.calls.append(materialized)
        if any("ZZID9001ZZ" in text for text in materialized):
            return ["markers disappeared"] * len(materialized)
        return [f"bn:{text}" for text in materialized]


def _segments(*texts: str) -> list[ContextSegment]:
    return [ContextSegment(key=f"cue-{index}", text=text) for index, text in enumerate(texts, 1)]


def test_context_codec_accepts_opaque_indictrans_markers():
    segments = _segments("Hello", "World")
    translated = "ZZID9001ZZ হ্যালো\nZZID9002ZZ বিশ্ব"
    assert decode_context_segments(translated, segments) == ["হ্যালো", "বিশ্ব"]


def test_context_codec_still_decodes_legacy_hash_markers():
    segments = _segments("Hello", "World")
    translated = "##000001 ## হ্যালো\n## 000002 ## বিশ্ব"
    assert decode_context_segments(translated, segments) == ["হ্যালো", "বিশ্ব"]


@pytest.mark.parametrize(
    "translated",
    [
        "ZZID9001ZZ one",  # missing
        "ZZID9002ZZ two\nZZID9001ZZ one",  # reordered
        "ZZID9001ZZ one\nZZID9001ZZ duplicate",  # duplicated
        "intro\nZZID9001ZZ one\nZZID9002ZZ two",  # unexpected prefix
        "ZZID9001ZZ one\nZZID9002ZZ",  # empty non-empty source
    ],
)
def test_context_codec_rejects_unsafe_alignment(translated: str):
    with pytest.raises(ContextAlignmentError):
        decode_context_segments(translated, _segments("one", "two"))


def test_context_windows_respect_encoded_size_and_segment_cap():
    windows = build_context_windows(
        _segments("a" * 20, "b" * 20, "c" * 20),
        max_chars=55,
        max_segments=2,
    )
    assert [len(window.segments) for window in windows] == [1, 1, 1]
    assert all(len(window.encoded_text) <= 55 for window in windows)


def test_source_text_resembling_a_context_marker_is_kept_marker_free():
    windows = build_context_windows(
        _segments("ID9001ZZ is literal source text", "second"),
        max_chars=500,
        max_segments=8,
    )

    assert [[segment.text for segment in window.segments] for window in windows] == [
        ["ID9001ZZ is literal source text"],
        ["second"],
    ]


def test_context_translation_keeps_exact_segment_mapping():
    segments = _segments("Hello", "World")
    window = build_context_windows(segments, max_chars=200, max_segments=4)[0]
    result = translate_context_batch(
        [window],
        _MarkerAwareTranslator(),
        source_lang="en",
        target_lang="bn",
    )
    assert result.translations == {"cue-1": "হ্যালো", "cue-2": "বিশ্ব"}
    assert result.alignment_fallback_keys == ()
    assert encode_context_segments(segments).startswith("ZZID9001ZZ")


def test_context_translation_retries_only_broken_window_as_atomic_segments():
    translator = _MarkerBreakingTranslator()
    segments = _segments("one", "two")
    window = build_context_windows(segments, max_chars=200, max_segments=4)[0]

    result = translate_context_batch(
        [window],
        translator,
        source_lang="en",
        target_lang="bn",
    )

    assert result.translations == {"cue-1": "bn:one", "cue-2": "bn:two"}
    assert result.alignment_fallback_keys == ("cue-1", "cue-2")
    assert len(translator.calls) == 2


def test_single_segment_window_avoids_marker_overhead():
    translator = _MarkerBreakingTranslator()
    window = build_context_windows(_segments("one"), max_chars=200, max_segments=4)[0]

    result = translate_context_batch(
        [window],
        translator,
        source_lang="en",
        target_lang="bn",
    )

    assert translator.calls == [["one"]]
    assert result.translations == {"cue-1": "bn:one"}
    assert result.alignment_fallback_keys == ()


def test_token_budget_refines_a_multi_segment_window_without_losing_order():
    window = build_context_windows(
        _segments("one", "two", "three", "four"),
        max_chars=500,
        max_segments=8,
    )[0]

    fitted = fit_context_windows(
        [window],
        lambda payload: payload.count("ZZID9") <= 2,
    )

    assert [[segment.text for segment in item.segments] for item in fitted] == [
        ["one", "two"],
        ["three", "four"],
    ]
