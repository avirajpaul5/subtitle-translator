from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from subtitle_translator.documents import (
    DocumentParseError,
    parse_document,
    serialize_document,
)


def test_txt_exact_roundtrip_and_deterministic_immutable_blocks():
    source = "\r\nFirst line\r\nwrapped line\r\n \r\nSecond line\n\nThird line\r"

    document = parse_document(source, ".txt")
    reparsed = parse_document(source, "notes.txt")

    assert serialize_document(document) == source
    assert document.source_hash == reparsed.source_hash
    assert [block.block_id for block in document.blocks] == [
        block.block_id for block in reparsed.blocks
    ]
    assert document.newline == "\r\n"
    assert [block.kind for block in document.blocks] == [
        "separator",
        "paragraph",
        "paragraph",
        "separator",
        "paragraph",
        "separator",
        "paragraph",
    ]
    assert [block.source_text for block in document.translatable_blocks[:2]] == [
        "First line",
        "wrapped line",
    ]

    with pytest.raises(FrozenInstanceError):
        document.translatable_blocks[0].source_text = "changed"  # type: ignore[misc]


def test_txt_translation_preserves_structural_newlines_and_blank_lines():
    source = "First line\r\nwrapped line\r\n\r\nSecond line\r\n"
    document = parse_document(source, "txt")
    first, wrapped, second = document.translatable_blocks

    translated = document.with_translations(
        {
            first.block_id: "প্রথম\nঅনুচ্ছেদ",
            wrapped.block_id: "মোড়ানো লাইন",
            second.block_id: "দ্বিতীয় অনুচ্ছেদ",
        }
    )

    assert first.source_text == "First line"
    assert wrapped.source_text == "wrapped line"
    assert first.target_text is None
    assert serialize_document(translated) == (
        "প্রথম\r\nঅনুচ্ছেদ\r\nমোড়ানো লাইন\r\n\r\nদ্বিতীয় অনুচ্ছেদ\r\n"
    )


def test_empty_and_whitespace_only_txt_roundtrip():
    assert serialize_document(parse_document("", ".txt")) == ""
    assert serialize_document(parse_document("  \n\t\n", ".txt")) == "  \n\t\n"
    assert parse_document("  \n\t\n", ".txt").translatable_blocks == ()


def test_markdown_exact_roundtrip_preserves_structural_blocks():
    source = (
        "## Heading ##\r\n"
        "\r\n"
        "- [x] Use `git` with [the docs](https://example.com/guide).\r\n"
        "\r\n"
        "```python\r\n"
        "print('not translated')\r\n"
        "```\r\n"
        "\r\n"
        "<div class=\"note\">\r\n"
        "Raw HTML stays untouched.\r\n"
        "</div>\r\n"
    )

    document = parse_document(source, ".md")

    assert serialize_document(document) == source
    assert [block.kind for block in document.blocks] == [
        "heading",
        "separator",
        "list_item",
        "separator",
        "code_block",
        "separator",
        "raw_html",
    ]
    heading, list_item = document.translatable_blocks
    assert heading.prefix == "## "
    assert heading.source_text == "Heading"
    assert heading.suffix == " ##\r\n"
    assert list_item.prefix == "- [x] "
    assert {span.kind for span in list_item.protected_spans} == {
        "inline_code",
        "markdown_syntax",
        "url",
    }
    assert all(
        not block.translatable
        for block in document.blocks
        if block.kind in {"code_block", "raw_html"}
    )


def test_markdown_translation_preserves_prefixes_and_inline_protected_content():
    source = (
        "# Heading\n\n"
        "- Use `git` with [the docs](https://example.com/guide).\n"
    )
    document = parse_document(source, "README.markdown")
    heading, list_item = document.translatable_blocks

    translated = document.with_translations(
        {
            heading.block_id: "শিরোনাম",
            list_item.block_id: "`git` দিয়ে [নথি](https://example.com/guide) পড়ুন।",
        }
    )

    assert document.format == "md"
    assert serialize_document(translated) == (
        "# শিরোনাম\n\n"
        "- `git` দিয়ে [নথি](https://example.com/guide) পড়ুন।\n"
    )


def test_markdown_translation_rejects_dropped_inline_code_url_or_html():
    source = "Press <kbd>Ctrl+C</kbd>, run `quit`, then visit https://example.com.\n"
    document = parse_document(source, ".md")
    block = document.translatable_blocks[0]

    protected = {(span.kind, span.text) for span in block.protected_spans}
    assert ("raw_html", "<kbd>") in protected
    assert ("raw_html", "</kbd>") in protected
    assert ("inline_code", "`quit`") in protected
    assert ("url", "https://example.com") in protected

    with pytest.raises(ValueError, match="dropped protected content"):
        block.with_target_text("সবকিছু বাদ দেওয়া হয়েছে।")


def test_markdown_image_and_reference_identifiers_survive_translation_exactly():
    source = (
        "![System diagram][architecture]\n"
        "Read [the guide][docs], [quick notes], and note[^caveat].\n"
        "\n"
        "[architecture]: https://example.com/architecture.png\n"
        "[docs]: https://example.com/docs\n"
        "[quick notes]: https://example.com/notes\n"
        "[^caveat]: This definition stays structural.\n"
    )
    document = parse_document(source, ".md")
    image, paragraph = document.translatable_blocks

    assert ("markdown_syntax", "!") in {
        (span.kind, span.text) for span in image.protected_spans
    }
    assert ("reference_identifier", "architecture") in {
        (span.kind, span.text) for span in image.protected_spans
    }
    assert {
        span.text
        for span in paragraph.protected_spans
        if span.kind == "reference_identifier"
    } == {"docs", "quick notes", "^caveat"}

    translated = document.with_translations(
        {
            image.block_id: "![সিস্টেম চিত্র][architecture]",
            paragraph.block_id: (
                "[নির্দেশিকাটি][docs] এবং [quick notes] পড়ুন, "
                "তারপর টীকাটি দেখুন[^caveat]।"
            ),
        }
    )

    assert serialize_document(translated) == (
        "![সিস্টেম চিত্র][architecture]\n"
        "[নির্দেশিকাটি][docs] এবং [quick notes] পড়ুন, তারপর টীকাটি দেখুন[^caveat]।\n"
        "\n"
        "[architecture]: https://example.com/architecture.png\n"
        "[docs]: https://example.com/docs\n"
        "[quick notes]: https://example.com/notes\n"
        "[^caveat]: This definition stays structural.\n"
    )
    assert all(not block.translatable for block in document.blocks[3:])


def test_markdown_image_or_reference_identifier_cannot_be_changed_by_translation():
    document = parse_document(
        "![Diagram][asset]\nRead [the guide][docs] and note[^one].\n",
        ".md",
    )
    image, paragraph = document.translatable_blocks

    with pytest.raises(ValueError, match="dropped protected content"):
        image.with_target_text("[চিত্র][asset]")
    with pytest.raises(ValueError, match="dropped protected content"):
        image.with_target_text("![চিত্র][সম্পদ]")
    with pytest.raises(ValueError, match="dropped protected content"):
        paragraph.with_target_text("[নির্দেশিকা][নথি] এবং টীকা[^one] পড়ুন।")
    with pytest.raises(ValueError, match="dropped protected content"):
        paragraph.with_target_text("[নির্দেশিকা][docs] এবং টীকা[^এক] পড়ুন।")


def test_markdown_setext_heading_list_and_blockquote_prefixes_roundtrip():
    source = (
        "Document title\n"
        "==============\n"
        "\n"
        "12) Numbered item\n"
        "> Quoted text\n"
        "\n"
        "[ref]: https://example.com\n"
    )

    document = parse_document(source, ".md")

    assert serialize_document(document) == source
    assert [block.kind for block in document.blocks] == [
        "heading",
        "separator",
        "list_item",
        "blockquote",
        "separator",
        "link_definition",
    ]
    assert document.blocks[0].source_text == "Document title"
    assert document.blocks[2].prefix == "12) "
    assert document.blocks[3].prefix == "> "
    assert not document.blocks[-1].translatable


def test_markdown_nested_containers_and_hard_break_are_structural():
    source = (
        "> > Nested quote\n"
        "> - [x] Nested task\n"
        "- > Nested list quote\n"
        "first line\\\n"
    )
    document = parse_document(source, ".md")
    nested_quote, nested_task, list_quote, hard_break = document.translatable_blocks

    assert serialize_document(document) == source
    assert nested_quote.prefix == "> > "
    assert nested_task.prefix == "> - [x] "
    assert list_quote.prefix == "- > "
    assert hard_break.source_text == "first line\\"
    assert ("markdown_syntax", "\\") in {
        (span.kind, span.text) for span in hard_break.protected_spans
    }

    with pytest.raises(ValueError, match="dropped protected content"):
        hard_break.with_target_text("প্রথম লাইন")


def test_markdown_structural_blocks_cannot_receive_target_text():
    document = parse_document("```\ncode\n```\n", ".md")

    with pytest.raises(ValueError, match="structural"):
        document.blocks[0].with_target_text("translated")


def test_document_with_translations_rejects_unknown_block_id():
    document = parse_document("Hello", ".txt")

    with pytest.raises(KeyError, match="Unknown document block IDs"):
        document.with_translations({"txt:missing": "হ্যালো"})


def test_document_adapter_rejects_unsupported_or_non_text_input():
    with pytest.raises(DocumentParseError, match="Unsupported document format"):
        parse_document("hello", ".docx")
    with pytest.raises(TypeError, match="decoded text"):
        parse_document(b"hello", ".txt")  # type: ignore[arg-type]
