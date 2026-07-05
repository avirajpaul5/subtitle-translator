from __future__ import annotations

from typing import Iterable, List, Optional

from subtitle_translator.translators.base import BaseTranslator

# IndicTrans2 always outputs Devanagari internally; transliterate to the target script.
# Devanagari-script languages need no conversion.
_SCRIPT_MAP = {
    "ben_Beng": "BENGALI",    # Bengali
    "asm_Beng": "BENGALI",    # Assamese (uses Bengali script)
    "mni_Beng": "BENGALI",    # Meitei/Manipuri (Bengali script variant)
    "guj_Gujr": "GUJARATI",   # Gujarati
    "kan_Knda": "KANNADA",    # Kannada
    "mal_Mlym": "MALAYALAM",  # Malayalam
    "ory_Orya": "ORIYA",      # Odia
    "pan_Guru": "GURMUKHI",   # Punjabi
    "tam_Taml": "TAMIL",      # Tamil
    "tel_Telu": "TELUGU",     # Telugu
}


class IndicTrans2Translator(BaseTranslator):
    """
    Local IndicTrans2 translator wrapper.

    Requires a locally available HuggingFace model directory and uses local_files_only=True
    to avoid cloud calls during inference.
    """

    def __init__(self, model_path: str, device: str = "cpu", max_new_tokens: int = 256) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        self.model.eval()
        if device != "cpu":
            self.model = self.model.to(device)
        self.torch = torch

    @property
    def display_name(self) -> str:
        return f"IndicTrans2 ({self.model_path})"

    def translate_batch(self, texts: Iterable[str], source_lang: str, target_lang: str) -> List[str]:
        import torch

        src_tag = _lang_to_indic_tag(source_lang)
        tgt_tag = _lang_to_indic_tag(target_lang)

        # Tokenizer's _src_tokenize expects "src_lang tgt_lang <text>"
        prefixed = [f"{src_tag} {tgt_tag} {text}" for text in texts]

        inputs = self.tokenizer(
            prefixed,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=self.max_new_tokens,
                num_beams=5,
                num_return_sequences=1,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )

        decoded = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        results = [t.strip() for t in decoded]

        # Transliterate from Devanagari to the target script when needed
        target_script = _SCRIPT_MAP.get(tgt_tag)
        if target_script:
            results = [_transliterate_deva(t, target_script) for t in results]

        return results


def _transliterate_deva(text: str, target_script: str) -> str:
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
        return transliterate(text, sanscript.DEVANAGARI, getattr(sanscript, target_script))
    except Exception:
        return text


def _lang_to_indic_tag(lang_code: str) -> str:
    mapping = {
        "en": "eng_Latn",
        "bn": "ben_Beng",
        "hi": "hin_Deva",
        "as": "asm_Beng",
        "gu": "guj_Gujr",
        "kn": "kan_Knda",
        "ml": "mal_Mlym",
        "mr": "mar_Deva",
        "ne": "npi_Deva",
        "or": "ory_Orya",
        "pa": "pan_Guru",
        "ta": "tam_Taml",
        "te": "tel_Telu",
        "ur": "urd_Arab",
    }
    return mapping.get(lang_code, lang_code)
