from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List


class BaseTranslator(ABC):
    @abstractmethod
    def translate_batch(self, texts: Iterable[str], source_lang: str, target_lang: str) -> List[str]:
        raise NotImplementedError
