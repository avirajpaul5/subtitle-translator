from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable

from subtitle_translator.documents.models import (
    InlineSpan,
    TranslationBlock,
    TranslationDocument,
)


class DocumentParseError(ValueError):
    pass


@dataclass(frozen=True)
class _PhysicalLine:
    body: str
    ending: str

    @property
    def raw(self) -> str:
        return self.body + self.ending

    @property
    def is_blank(self) -> bool:
        return not self.body.strip()


@dataclass(frozen=True)
class _ProtectedInterval:
    start: int
    end: int
    kind: str
    priority: int


class _BlockBuilder:
    def __init__(self, format_name: str) -> None:
        self.format_name = format_name
        self.blocks: list[TranslationBlock] = []

    def add(
        self,
        *,
        kind: str,
        source_text: str,
        translatable: bool,
        prefix: str = "",
        suffix: str = "",
        spans: tuple[InlineSpan, ...] | None = None,
        path: tuple[str, ...] = (),
    ) -> None:
        block_id = f"{self.format_name}:b{len(self.blocks) + 1:06d}"
        if spans is None:
            spans = _plain_spans(source_text) if translatable else _protected_block_spans(
                source_text, kind
            )
        self.blocks.append(
            TranslationBlock(
                block_id=block_id,
                kind=kind,
                source_text=source_text,
                translatable=translatable,
                prefix=prefix,
                suffix=suffix,
                spans=spans,
                path=path or (f"block-{len(self.blocks) + 1}",),
            )
        )


_FENCE_OPEN_RE = re.compile(r"^(?P<indent> {0,3})(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
_ATX_HEADING_RE = re.compile(
    r"^(?P<prefix> {0,3}(?P<marks>#{1,6})[ \t]+)(?P<body>.*)$"
)
_ATX_CLOSING_RE = re.compile(r"^(?P<body>.*?)(?P<closing>[ \t]+#+[ \t]*)$")
_SETEXT_RE = re.compile(r"^ {0,3}(?P<marks>=+|-+)[ \t]*$")
_LIST_ITEM_RE = re.compile(
    r"^(?P<prefix> {0,3}(?:[*+-]|\d{1,9}[.)])[ \t]+)(?P<body>.*)$"
)
_TASK_MARKER_RE = re.compile(r"^(?P<marker>\[[ xX]\][ \t]+)(?P<body>.*)$")
_BLOCKQUOTE_RE = re.compile(
    r"^(?P<prefix> {0,3}(?:>[ \t]?)+)(?P<body>.*)$"
)
_LINK_DEFINITION_RE = re.compile(
    r"^ {0,3}\[(?P<label>[^\]\n]+)\]:[ \t]*\S+"
)
_HTML_BLOCK_START_RE = re.compile(
    r"^ {0,3}(?:<!--|<\?|<![A-Z]|</?[A-Za-z][A-Za-z0-9-]*(?:[ \t/>]|$))"
)
_HTML_RAW_CONTAINER_RE = re.compile(
    r"^ {0,3}<(?P<tag>script|pre|style|textarea)(?:[ \t>]|$)", re.IGNORECASE
)
_THEMATIC_BREAK_RE = re.compile(
    r"^ {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$"
)

_INLINE_CODE_RE = re.compile(
    r"(?<!`)(?P<ticks>`+)(?!`)(?P<body>.*?)(?<!`)(?P=ticks)(?!`)",
    re.DOTALL,
)
_AUTOLINK_RE = re.compile(r"<(?:https?://|mailto:)[^<>\s]+>", re.IGNORECASE)
_INLINE_HTML_RE = re.compile(
    r"<!--.*?-->|<\?.*?\?>|<![A-Z][^>]*>|"
    r"</?[A-Za-z][A-Za-z0-9-]*(?:[ \t][^>\n]*|/?)>",
    re.DOTALL,
)
_LINK_DESTINATION_RE = re.compile(
    r"!?\[[^\]\n]*\]\([ \t]*(?P<destination><[^>\n]+>|[^)\s]+)",
)
_IMAGE_SIGIL_RE = re.compile(r"!(?=\[)")
_REFERENCE_LINK_RE = re.compile(
    r"!?\[(?P<label>[^\]\n]*)\][ \t]*\[(?P<identifier>[^\]\n]*)\]"
)
_FOOTNOTE_REFERENCE_RE = re.compile(r"\[(?P<identifier>\^[^\]\n]+)\]")
_BRACKET_LABEL_RE = re.compile(
    r"(?<!\\)\[(?P<identifier>[^\]\n]+)\](?![ \t]*(?:\[|\())"
)
_HARD_LINE_BREAK_RE = re.compile(r"\\(?=$)")
_BARE_URL_RE = re.compile(r"\b(?:https?://|mailto:)[^\s<>()\[\]{}]+", re.IGNORECASE)
_MARKDOWN_DELIMITER_RE = re.compile(
    r"\\[!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~]|[*_~]{1,3}|[\[\]()]"
)


def parse_document(content: str, file_ext: str) -> TranslationDocument:
    """Parse TXT or conservative CommonMark into the normalized document IR.

    The function operates on decoded text.  Callers remain responsible for byte
    decoding and for encoding serialized target text as UTF-8.
    """

    if not isinstance(content, str):
        raise TypeError("Document content must be decoded text.")

    ext = _normalize_extension(file_ext)
    parser: Callable[[str], TranslationDocument]
    if ext == "txt":
        parser = _parse_plain_text
    elif ext in {"md", "markdown"}:
        parser = _parse_markdown
    else:
        raise DocumentParseError(f"Unsupported document format: {file_ext}")
    return parser(content)


def serialize_document(document: TranslationDocument) -> str:
    """Serialize a normalized document, preserving source structure exactly.

    When a block has target text, only that block's text payload is replaced;
    format prefixes, suffixes, blank lines, code, and raw structural blocks are
    emitted from the source representation.
    """

    if not isinstance(document, TranslationDocument):
        raise TypeError("serialize_document expects a TranslationDocument.")

    rendered: list[str] = []
    for block in document.blocks:
        body = block.source_text
        if block.translatable and block.target_text is not None:
            body = _normalize_newlines(block.target_text, document.newline)
        rendered.extend((block.prefix, body, block.suffix))
    return "".join(rendered)


def _parse_plain_text(content: str) -> TranslationDocument:
    builder = _BlockBuilder("txt")
    lines = _split_physical_lines(content)
    separators: list[_PhysicalLine] = []

    def flush_separators() -> None:
        if not separators:
            return
        builder.add(
            kind="separator",
            source_text="".join(line.raw for line in separators),
            translatable=False,
        )
        separators.clear()

    for line in lines:
        if line.is_blank:
            separators.append(line)
        else:
            flush_separators()
            leading, body, trailing = _split_edge_whitespace(line.body)
            builder.add(
                kind="paragraph",
                source_text=body,
                translatable=bool(body),
                prefix=leading,
                suffix=trailing + line.ending,
            )

    flush_separators()
    return _build_document("txt", content, builder)


def _parse_markdown(content: str) -> TranslationDocument:
    builder = _BlockBuilder("md")
    lines = _split_physical_lines(content)
    reference_labels = _collect_reference_labels(lines)
    index = 0

    while index < len(lines):
        line = lines[index]

        if line.is_blank:
            end = index + 1
            while end < len(lines) and lines[end].is_blank:
                end += 1
            builder.add(
                kind="separator",
                source_text="".join(item.raw for item in lines[index:end]),
                translatable=False,
            )
            index = end
            continue

        fence = _FENCE_OPEN_RE.match(line.body)
        if fence:
            end = _find_fence_end(lines, index, fence.group("fence"))
            builder.add(
                kind="code_block",
                source_text="".join(item.raw for item in lines[index:end]),
                translatable=False,
                path=("code-block", str(index)),
            )
            index = end
            continue

        raw_container = _HTML_RAW_CONTAINER_RE.match(line.body)
        if raw_container:
            end = _find_raw_html_container_end(lines, index, raw_container.group("tag"))
            builder.add(
                kind="raw_html",
                source_text="".join(item.raw for item in lines[index:end]),
                translatable=False,
                path=("raw-html", str(index)),
            )
            index = end
            continue

        if _HTML_BLOCK_START_RE.match(line.body):
            end = index + 1
            while end < len(lines) and not lines[end].is_blank:
                end += 1
            builder.add(
                kind="raw_html",
                source_text="".join(item.raw for item in lines[index:end]),
                translatable=False,
                path=("raw-html", str(index)),
            )
            index = end
            continue

        if line.body.startswith(("    ", "\t")):
            end = index + 1
            while end < len(lines) and lines[end].body.startswith(("    ", "\t")):
                end += 1
            builder.add(
                kind="code_block",
                source_text="".join(item.raw for item in lines[index:end]),
                translatable=False,
                path=("indented-code", str(index)),
            )
            index = end
            continue

        if _LINK_DEFINITION_RE.match(line.body) or _THEMATIC_BREAK_RE.match(line.body):
            builder.add(
                kind="link_definition" if _LINK_DEFINITION_RE.match(line.body) else "thematic_break",
                source_text=line.raw,
                translatable=False,
            )
            index += 1
            continue

        if index + 1 < len(lines) and _SETEXT_RE.match(lines[index + 1].body):
            underline = lines[index + 1]
            level = "1" if underline.body.lstrip().startswith("=") else "2"
            leading, body, trailing = _split_edge_whitespace(line.body)
            builder.add(
                kind="heading",
                source_text=body,
                translatable=bool(body),
                prefix=leading,
                suffix=trailing + line.ending + underline.raw,
                spans=_inline_spans(body, reference_labels),
                path=("heading", level, str(index)),
            )
            index += 2
            continue

        heading = _ATX_HEADING_RE.match(line.body)
        if heading:
            body = heading.group("body")
            closing = ""
            closing_match = _ATX_CLOSING_RE.match(body)
            if closing_match:
                body = closing_match.group("body")
                closing = closing_match.group("closing")
            leading, body, trailing = _split_edge_whitespace(body)
            level = str(len(heading.group("marks")))
            builder.add(
                kind="heading",
                source_text=body,
                translatable=bool(body),
                prefix=heading.group("prefix") + leading,
                suffix=trailing + closing + line.ending,
                spans=_inline_spans(body, reference_labels),
                path=("heading", level, str(index)),
            )
            index += 1
            continue

        list_item = _LIST_ITEM_RE.match(line.body)
        if list_item:
            prefix = list_item.group("prefix")
            body = list_item.group("body")
            task = _TASK_MARKER_RE.match(body)
            if task:
                prefix += task.group("marker")
                body = task.group("body")
            nested_prefix, body = _consume_nested_container_prefix(body)
            prefix += nested_prefix
            leading, body, trailing = _split_edge_whitespace(body)
            builder.add(
                kind="list_item",
                source_text=body,
                translatable=bool(body),
                prefix=prefix + leading,
                suffix=trailing + line.ending,
                spans=_inline_spans(body, reference_labels),
                path=("list-item", str(index)),
            )
            index += 1
            continue

        quote = _BLOCKQUOTE_RE.match(line.body)
        if quote:
            body = quote.group("body")
            nested_prefix, body = _consume_nested_container_prefix(body)
            leading, body, trailing = _split_edge_whitespace(body)
            builder.add(
                kind="blockquote",
                source_text=body,
                translatable=bool(body),
                prefix=quote.group("prefix") + nested_prefix + leading,
                suffix=trailing + line.ending,
                spans=_inline_spans(body, reference_labels),
                path=("blockquote", str(index)),
            )
            index += 1
            continue

        leading, body, trailing = _split_edge_whitespace(line.body)
        builder.add(
            kind="paragraph",
            source_text=body,
            translatable=bool(body),
            prefix=leading,
            suffix=trailing + line.ending,
            spans=_inline_spans(body, reference_labels),
            path=("paragraph", str(index)),
        )
        index += 1

    return _build_document("md", content, builder)


def _consume_nested_container_prefix(text: str) -> tuple[str, str]:
    """Peel nested quote/list/task markers out of model-visible text."""

    prefix = ""
    remaining = text
    while remaining:
        quote = _BLOCKQUOTE_RE.match(remaining)
        if quote:
            prefix += quote.group("prefix")
            remaining = quote.group("body")
            continue
        list_item = _LIST_ITEM_RE.match(remaining)
        if list_item:
            prefix += list_item.group("prefix")
            remaining = list_item.group("body")
            task = _TASK_MARKER_RE.match(remaining)
            if task:
                prefix += task.group("marker")
                remaining = task.group("body")
            continue
        break
    return prefix, remaining


def _find_fence_end(
    lines: list[_PhysicalLine], start: int, opening_fence: str
) -> int:
    marker = opening_fence[0]
    minimum_length = len(opening_fence)
    closing = re.compile(
        rf"^ {{0,3}}{re.escape(marker)}{{{minimum_length},}}[ \t]*$"
    )
    for index in range(start + 1, len(lines)):
        if closing.match(lines[index].body):
            return index + 1
    return len(lines)


def _find_raw_html_container_end(
    lines: list[_PhysicalLine], start: int, tag: str
) -> int:
    closing = re.compile(rf"</{re.escape(tag)}\s*>", re.IGNORECASE)
    for index in range(start, len(lines)):
        if closing.search(lines[index].body):
            return index + 1
    return len(lines)


def _inline_spans(
    text: str, reference_labels: frozenset[str] = frozenset()
) -> tuple[InlineSpan, ...]:
    if not text:
        return ()

    candidates: list[_ProtectedInterval] = []
    for match in _INLINE_CODE_RE.finditer(text):
        candidates.append(_ProtectedInterval(match.start(), match.end(), "inline_code", 0))
    for match in _AUTOLINK_RE.finditer(text):
        candidates.append(_ProtectedInterval(match.start(), match.end(), "url", 1))
    for match in _INLINE_HTML_RE.finditer(text):
        candidates.append(_ProtectedInterval(match.start(), match.end(), "raw_html", 2))
    for match in _LINK_DESTINATION_RE.finditer(text):
        start, end = match.span("destination")
        candidates.append(_ProtectedInterval(start, end, "url", 1))
    for match in _IMAGE_SIGIL_RE.finditer(text):
        candidates.append(
            _ProtectedInterval(match.start(), match.end(), "markdown_syntax", 3)
        )
    for match in _REFERENCE_LINK_RE.finditer(text):
        group = "identifier" if match.group("identifier") else "label"
        start, end = match.span(group)
        if start < end:
            candidates.append(
                _ProtectedInterval(start, end, "reference_identifier", 3)
            )
    for match in _FOOTNOTE_REFERENCE_RE.finditer(text):
        start, end = match.span("identifier")
        candidates.append(
            _ProtectedInterval(start, end, "reference_identifier", 3)
        )
    if reference_labels:
        for match in _BRACKET_LABEL_RE.finditer(text):
            identifier = match.group("identifier")
            if _normalize_reference_label(identifier) not in reference_labels:
                continue
            start, end = match.span("identifier")
            candidates.append(
                _ProtectedInterval(start, end, "reference_identifier", 3)
            )
    for match in _BARE_URL_RE.finditer(text):
        end = match.end()
        while end > match.start() and text[end - 1] in ".,;:!?":
            end -= 1
        if end > match.start():
            candidates.append(_ProtectedInterval(match.start(), end, "url", 1))
    for match in _HARD_LINE_BREAK_RE.finditer(text):
        candidates.append(
            _ProtectedInterval(match.start(), match.end(), "markdown_syntax", 3)
        )
    for match in _MARKDOWN_DELIMITER_RE.finditer(text):
        candidates.append(
            _ProtectedInterval(
                match.start(),
                match.end(),
                "markdown_syntax",
                3,
            )
        )

    accepted: list[_ProtectedInterval] = []
    for candidate in sorted(
        candidates, key=lambda item: (item.start, item.priority, -(item.end - item.start))
    ):
        if candidate.start == candidate.end:
            continue
        if any(
            candidate.start < existing.end and existing.start < candidate.end
            for existing in accepted
        ):
            continue
        accepted.append(candidate)
    accepted.sort(key=lambda item: item.start)

    spans: list[InlineSpan] = []
    cursor = 0
    for interval in accepted:
        if interval.start > cursor:
            spans.append(
                InlineSpan(
                    kind="text",
                    text=text[cursor : interval.start],
                    start=cursor,
                    end=interval.start,
                    translatable=True,
                )
            )
        spans.append(
            InlineSpan(
                kind=interval.kind,
                text=text[interval.start : interval.end],
                start=interval.start,
                end=interval.end,
                translatable=False,
            )
        )
        cursor = interval.end
    if cursor < len(text):
        spans.append(
            InlineSpan(
                kind="text",
                text=text[cursor:],
                start=cursor,
                end=len(text),
                translatable=True,
            )
        )
    return tuple(spans) if spans else _plain_spans(text)


def _collect_reference_labels(lines: list[_PhysicalLine]) -> frozenset[str]:
    """Collect definition labels used to recognize shortcut references safely."""

    labels = {
        _normalize_reference_label(match.group("label"))
        for line in lines
        if (match := _LINK_DEFINITION_RE.match(line.body))
    }
    return frozenset(label for label in labels if label)


def _normalize_reference_label(label: str) -> str:
    """Apply the useful subset of CommonMark reference-label normalization."""

    return re.sub(r"[ \t\r\n]+", " ", label.strip()).casefold()


def _plain_spans(text: str) -> tuple[InlineSpan, ...]:
    if not text:
        return ()
    return (
        InlineSpan(
            kind="text",
            text=text,
            start=0,
            end=len(text),
            translatable=True,
        ),
    )


def _protected_block_spans(text: str, kind: str) -> tuple[InlineSpan, ...]:
    if not text:
        return ()
    return (
        InlineSpan(
            kind=kind,
            text=text,
            start=0,
            end=len(text),
            translatable=False,
        ),
    )


def _split_edge_whitespace(text: str) -> tuple[str, str, str]:
    """Separate layout whitespace so translation cannot silently trim it."""

    leading_match = re.match(r"^[ \t]*", text)
    leading = leading_match.group(0) if leading_match else ""
    remainder = text[len(leading) :]
    trailing_match = re.search(r"[ \t]*$", remainder)
    trailing = trailing_match.group(0) if trailing_match else ""
    body = remainder[: len(remainder) - len(trailing)] if trailing else remainder
    return leading, body, trailing


def _split_physical_lines(content: str) -> list[_PhysicalLine]:
    lines: list[_PhysicalLine] = []
    position = 0
    for match in re.finditer(r"\r\n|\r|\n", content):
        lines.append(_PhysicalLine(content[position : match.start()], match.group(0)))
        position = match.end()
    if position < len(content):
        lines.append(_PhysicalLine(content[position:], ""))
    return lines


def _detect_newline(content: str) -> str:
    matches = list(re.finditer(r"\r\n|\r|\n", content))
    if not matches:
        return "\n"
    counts: dict[str, int] = {}
    first_position: dict[str, int] = {}
    for match in matches:
        value = match.group(0)
        counts[value] = counts.get(value, 0) + 1
        first_position.setdefault(value, match.start())
    return max(counts, key=lambda value: (counts[value], -first_position[value]))


def _normalize_newlines(text: str, newline: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.replace("\n", newline)


def _build_document(
    format_name: str, content: str, builder: _BlockBuilder
) -> TranslationDocument:
    source_hash = hashlib.sha256(
        content.encode("utf-8", errors="surrogatepass")
    ).hexdigest()
    return TranslationDocument(
        format=format_name,
        source_hash=source_hash,
        blocks=tuple(builder.blocks),
        newline=_detect_newline(content),
    )


def _normalize_extension(file_ext: str) -> str:
    if not isinstance(file_ext, str):
        raise TypeError("file_ext must be a string.")
    value = file_ext.strip().lower()
    if "." in value:
        value = value.rsplit(".", 1)[-1]
    return value.lstrip(".")
