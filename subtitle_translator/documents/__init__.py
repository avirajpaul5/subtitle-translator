"""Generic document translation IR and format adapters.

Phase 1A supports plain text (``.txt``) and a conservative CommonMark subset
(``.md`` / ``.markdown``).  Subtitle parsing remains in
``subtitle_translator.parsers``.
"""

from subtitle_translator.documents.adapters import (
    DocumentParseError,
    parse_document,
    serialize_document,
)
from subtitle_translator.documents.models import (
    InlineSpan,
    TranslationBlock,
    TranslationDocument,
)

__all__ = [
    "DocumentParseError",
    "InlineSpan",
    "TranslationBlock",
    "TranslationDocument",
    "parse_document",
    "serialize_document",
]
