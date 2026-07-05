from __future__ import annotations

from typing import Iterable, List

from subtitle_translator.translators.base import BaseTranslator


class EchoTranslator(BaseTranslator):
    """Test mode translator that leaves text unchanged."""

    @property
    def display_name(self) -> str:
        return "echo"

    def translate_batch(self, texts: Iterable[str], source_lang: str, target_lang: str) -> List[str]:
        return list(texts)
