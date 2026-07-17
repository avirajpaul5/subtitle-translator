from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from subtitle_translator.translators.base import BaseTranslator


# The local IndicTrans2 checkpoint drops hash-only markers, but preserves its
# opaque ``ZZID…ZZ`` token family. Context markers live in a reserved 9xxx
# range so they cannot collide with the per-unit protected-content IDs, which
# begin at zero. The legacy hash form remains decodable for compatibility with
# synthetic/custom providers.
_CONTEXT_MARKER_BASE = 9000
_MAX_CONTEXT_SEGMENTS = 999
_MARKER_RE = re.compile(
    r"(?:[Zz]*[Ii][Dd]\s*(?P<opaque>9\d{3})\s*[Zz]+)"
    r"|"
    r"(?:(?:#\s*){2}0*(?P<legacy>\d{1,6})\s*(?:#\s*){2})"
)


class ContextAlignmentError(RuntimeError):
    """A contextual translation did not preserve every segment boundary."""


@dataclass(frozen=True)
class ContextSegment:
    key: str
    text: str


@dataclass(frozen=True)
class ContextWindow:
    segments: tuple[ContextSegment, ...]

    @property
    def encoded_text(self) -> str:
        return encode_context_segments(self.segments)


@dataclass(frozen=True)
class ContextBatchResult:
    translations: dict[str, str]
    alignment_fallback_keys: tuple[str, ...] = ()


def encode_context_segments(segments: Sequence[ContextSegment]) -> str:
    """Encode adjacent segments into one model input with stable local IDs."""

    if len(segments) > _MAX_CONTEXT_SEGMENTS:
        raise ValueError(
            f"A context window supports at most {_MAX_CONTEXT_SEGMENTS} segments."
        )
    return "\n".join(
        f"ZZID{_CONTEXT_MARKER_BASE + position}ZZ {segment.text.strip()}"
        for position, segment in enumerate(segments, start=1)
    )


def decode_context_segments(
    translated_text: str,
    segments: Sequence[ContextSegment],
) -> list[str]:
    """Decode and strictly validate a contextual model response.

    Every marker must occur exactly once and in the original order. A missing
    translation for a non-empty source is also rejected. Callers can then retry
    just that window one segment at a time instead of accepting shifted or empty
    output.
    """

    matches = list(_MARKER_RE.finditer(translated_text))
    expected_positions = list(range(1, len(segments) + 1))
    actual_positions = [
        (
            int(match.group("opaque")) - _CONTEXT_MARKER_BASE
            if match.group("opaque") is not None
            else int(match.group("legacy"))
        )
        for match in matches
    ]
    if actual_positions != expected_positions:
        raise ContextAlignmentError(
            "Context markers were missing, duplicated, or reordered "
            f"(expected {expected_positions}, received {actual_positions})."
        )

    if matches and translated_text[: matches[0].start()].strip():
        raise ContextAlignmentError("Unexpected translated text appeared before the first marker.")

    decoded: list[str] = []
    for index, (match, segment) in enumerate(zip(matches, segments)):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(translated_text)
        value = translated_text[match.end() : end].strip()
        if segment.text.strip() and not value:
            raise ContextAlignmentError(
                f"Context segment {segment.key!r} returned an empty translation."
            )
        decoded.append(value)
    return decoded


def build_context_windows(
    segments: Iterable[ContextSegment],
    *,
    max_chars: int,
    max_segments: int,
) -> list[ContextWindow]:
    """Pack consecutive segments without exceeding the requested safe window."""

    safe_chars = max(1, int(max_chars))
    safe_segments = max(1, int(max_segments))
    windows: list[ContextWindow] = []
    pending: list[ContextSegment] = []

    for segment in segments:
        if _MARKER_RE.search(segment.text):
            # A literal source token that resembles our boundary must never be
            # interpreted as alignment metadata. Keep that unit marker-free;
            # validation can then surface any genuine sentinel-like source
            # text without risking silent deletion or shifted output.
            if pending:
                windows.append(ContextWindow(tuple(pending)))
                pending = []
            windows.append(ContextWindow((segment,)))
            continue
        candidate = [*pending, segment]
        candidate_text = encode_context_segments(candidate)
        if pending and (
            len(candidate) > safe_segments or len(candidate_text) > safe_chars
        ):
            windows.append(ContextWindow(tuple(pending)))
            pending = [segment]
        else:
            pending = candidate

    if pending:
        windows.append(ContextWindow(tuple(pending)))
    return windows


def fit_context_windows(
    windows: Sequence[ContextWindow],
    accepts_input: Callable[[str], bool],
) -> list[ContextWindow]:
    """Split multi-segment windows until a provider's exact budget accepts them."""

    fitted: list[ContextWindow] = []

    def add(window: ContextWindow) -> None:
        payload = (
            window.segments[0].text
            if len(window.segments) == 1
            else window.encoded_text
        )
        if accepts_input(payload):
            fitted.append(window)
            return
        if len(window.segments) == 1:
            raise ValueError(
                f"Translation segment {window.segments[0].key!r} exceeds the provider input budget."
            )
        midpoint = len(window.segments) // 2
        add(ContextWindow(window.segments[:midpoint]))
        add(ContextWindow(window.segments[midpoint:]))

    for window in windows:
        add(window)
    return fitted


def translate_context_batch(
    windows: Sequence[ContextWindow],
    translator: BaseTranslator,
    *,
    source_lang: str,
    target_lang: str,
) -> ContextBatchResult:
    """Translate contextual windows, retrying only invalid windows atomically."""

    if not windows:
        return ContextBatchResult(translations={})

    translated_windows = translator.translate_batch(
        [
            window.segments[0].text
            if len(window.segments) == 1
            else window.encoded_text
            for window in windows
        ],
        source_lang=source_lang,
        target_lang=target_lang,
    )
    if len(translated_windows) != len(windows):
        raise RuntimeError("Translator returned a different number of context windows than requested.")

    translations: dict[str, str] = {}
    fallback_keys: list[str] = []
    for window, translated_window in zip(windows, translated_windows):
        if len(window.segments) == 1:
            segment = window.segments[0]
            cleaned = translated_window.strip()
            if segment.text.strip() and not cleaned:
                raise RuntimeError(
                    f"Translator returned an empty translation for segment {segment.key!r}."
                )
            translations[segment.key] = cleaned
            continue

        try:
            decoded = decode_context_segments(translated_window, window.segments)
        except ContextAlignmentError:
            # Marker fidelity is a capability, not an assumption. Dedicated MT
            # models sometimes alter or drop delimiters, so fall back to exact
            # one-input/one-output alignment for only the affected window.
            decoded = translator.translate_batch(
                [segment.text for segment in window.segments],
                source_lang=source_lang,
                target_lang=target_lang,
            )
            if len(decoded) != len(window.segments):
                raise RuntimeError(
                    "Translator returned a different number of fallback segments than requested."
                )
            fallback_keys.extend(segment.key for segment in window.segments)

        for segment, value in zip(window.segments, decoded):
            cleaned = value.strip()
            if segment.text.strip() and not cleaned:
                raise RuntimeError(
                    f"Translator returned an empty translation for segment {segment.key!r}."
                )
            translations[segment.key] = cleaned

    return ContextBatchResult(
        translations=translations,
        alignment_fallback_keys=tuple(fallback_keys),
    )
