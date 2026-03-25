from __future__ import annotations

from typing import Iterable, List

from subtitle_translator.translators.base import BaseTranslator


class IndicTrans2Translator(BaseTranslator):
    """
    Local IndicTrans2 translator wrapper.

    Requires a locally available HuggingFace model directory and uses local_files_only=True
    to avoid cloud calls during inference.
    """

    def __init__(self, model_path: str, device: str = "cpu", max_new_tokens: int = 256) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

        device_index = -1 if device == "cpu" else 0
        self.pipe = pipeline(
            "translation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device_index,
            max_new_tokens=max_new_tokens,
        )

    def translate_batch(self, texts: Iterable[str], source_lang: str, target_lang: str) -> List[str]:
        src_tag = _lang_to_indic_tag(source_lang)
        tgt_tag = _lang_to_indic_tag(target_lang)

        prefixed = [f"{src_tag} {tgt_tag} {text}" for text in texts]
        outputs = self.pipe(prefixed)
        return [x["translation_text"].strip() for x in outputs]


def _lang_to_indic_tag(lang_code: str) -> str:
    mapping = {
        "en": "eng_Latn",
        "bn": "ben_Beng",
        "hi": "hin_Deva",
    }
    return mapping.get(lang_code, lang_code)
