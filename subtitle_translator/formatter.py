from __future__ import annotations

from textwrap import wrap


def subtitle_line_break(text: str, max_line_length: int = 42, max_lines: int = 2) -> str:
    """Wrap text for subtitle readability while preserving existing paragraph breaks."""
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    if not paragraphs:
        return ""

    wrapped_paragraphs = []
    for p in paragraphs:
        lines = wrap(p, width=max_line_length, break_long_words=False, break_on_hyphens=False)
        if not lines:
            lines = [p]
        if len(lines) > max_lines:
            lines = lines[: max_lines - 1] + [" ".join(lines[max_lines - 1 :])]
        wrapped_paragraphs.append("\n".join(lines))

    return "\n".join(wrapped_paragraphs)
