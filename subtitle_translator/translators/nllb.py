from __future__ import annotations

from typing import Iterable, List

from subtitle_translator.translators.base import BaseTranslator


class NLLBTranslator(BaseTranslator):
    """Alternative local translator backend for easy swapping."""

    def __init__(self, model_path: str, source_lang: str = "eng_Latn", target_lang: str = "ben_Beng") -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
        self.source_lang = source_lang
        self.target_lang = target_lang

    def translate_batch(self, texts: Iterable[str], source_lang: str, target_lang: str) -> List[str]:
        tokenized = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            src_lang=self.source_lang,
        )
        generated = self.model.generate(
            **tokenized,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.target_lang),
            max_new_tokens=256,
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
