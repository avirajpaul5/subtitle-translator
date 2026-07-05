from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List


class BaseTranslator(ABC):
    @property
    def display_name(self) -> str:
        return self.__class__.__name__.lstrip("_")

    @property
    def last_used_name(self) -> str:
        return self.display_name

    @property
    def usage_summary(self) -> str:
        return self.last_used_name

    @abstractmethod
    def translate_batch(self, texts: Iterable[str], source_lang: str, target_lang: str) -> List[str]:
        raise NotImplementedError
