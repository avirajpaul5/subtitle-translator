from __future__ import annotations

from typing import Iterable, List

from subtitle_translator.translators.base import BaseTranslator
from subtitle_translator.translators.indictrans2 import (
    _generation_hit_limit,
    _local_model_stamp,
    _sequence_length,
)


class NLLBInputTooLongError(ValueError):
    pass


class NLLBOutputTooLongError(RuntimeError):
    pass


class NLLBTranslator(BaseTranslator):
    """Alternative local translator backend for easy swapping."""

    def __init__(
        self,
        model_path: str,
        source_lang: str = "eng_Latn",
        target_lang: str = "ben_Beng",
        max_new_tokens: int = 256,
    ) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
        self.model.eval()
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_new_tokens = max_new_tokens
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        model_limit = getattr(
            getattr(self.model, "config", None), "max_position_embeddings", None
        )
        finite_limits = [
            int(value)
            for value in (tokenizer_limit, model_limit)
            if isinstance(value, int) and 0 < value < 1_000_000
        ]
        self.max_source_tokens = min(finite_limits) if finite_limits else 1024
        self.max_input_chars = 500
        self._model_stamp = _local_model_stamp(model_path)

    @property
    def display_name(self) -> str:
        return f"NLLB ({self.model_path})"

    @property
    def checkpoint_fingerprint(self) -> str:
        return (
            f"{self.display_name}|model={self._model_stamp}|"
            f"source_tokens={self.max_source_tokens}|max_new_tokens={self.max_new_tokens}"
        )

    def accepts_input(self, text: str, source_lang: str, target_lang: str) -> bool:
        encoded = self._tokenize([text], source_lang, padding=False)
        length = _sequence_length(encoded.get("input_ids"))
        return length is None or length <= self.max_source_tokens

    def translate_batch(self, texts: Iterable[str], source_lang: str, target_lang: str) -> List[str]:
        tokenized = self._tokenize(list(texts), source_lang, padding=True)
        source_tokens = _sequence_length(tokenized.get("input_ids"))
        if source_tokens is not None and source_tokens > self.max_source_tokens:
            raise NLLBInputTooLongError(
                f"NLLB input requires {source_tokens} tokens but this model supports "
                f"{self.max_source_tokens}. Input was not truncated."
            )
        target_tag = _lang_to_nllb_tag(target_lang, self.target_lang)
        generated = self.model.generate(
            **tokenized,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(target_tag),
            max_new_tokens=self.max_new_tokens,
        )
        if _generation_hit_limit(
            generated,
            max_length=self.max_new_tokens,
            eos_token_id=getattr(getattr(self.model, "config", None), "eos_token_id", None),
            pad_token_id=getattr(getattr(self.model, "config", None), "pad_token_id", None),
        ):
            raise NLLBOutputTooLongError(
                "NLLB output reached the generation limit without an end token. "
                "The incomplete translation was not accepted."
            )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def _tokenize(self, texts: list[str], source_lang: str, *, padding: bool):
        source_tag = _lang_to_nllb_tag(source_lang, self.source_lang)
        self.tokenizer.src_lang = source_tag
        return self.tokenizer(
            texts,
            return_tensors="pt" if padding else None,
            padding=padding,
            truncation=False,
        )


def _lang_to_nllb_tag(lang_code: str, fallback: str) -> str:
    return {
        "en": "eng_Latn",
        "bn": "ben_Beng",
    }.get(lang_code, lang_code if "_" in lang_code else fallback)
