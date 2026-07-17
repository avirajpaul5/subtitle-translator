from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List

from subtitle_translator.translators.base import BaseTranslator


class IndicTransInputTooLongError(ValueError):
    pass


class IndicTransOutputTooLongError(RuntimeError):
    pass


class IndicTrans2InputChecker:
    """Tokenizer-only capability check for a local IndicTrans2 model.

    Sarvam fallback planning uses this instead of loading the full local model
    weights before a fallback is actually needed.
    """

    def __init__(self, model_path: str) -> None:
        from IndicTransToolkit import IndicProcessor
        from transformers import AutoTokenizer

        self.processor = IndicProcessor(inference=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.max_source_tokens = _max_source_positions_from_config(model_path)

    def accepts_input(self, text: str, source_lang: str, target_lang: str) -> bool:
        return _indictrans_input_fits(
            self.processor,
            self.tokenizer,
            text,
            source_lang,
            target_lang,
            self.max_source_tokens,
        )


class IndicTrans2Translator(BaseTranslator):
    """
    Local IndicTrans2 translator wrapper.

    Requires a locally available HuggingFace model directory and uses local_files_only=True
    to avoid cloud calls during inference.
    """

    def __init__(self, model_path: str, device: str = "cpu", max_new_tokens: int = 256) -> None:
        import torch
        from IndicTransToolkit import IndicProcessor
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.processor = IndicProcessor(inference=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        self.model.eval()
        if device != "cpu":
            self.model = self.model.to(device)
        self.torch = torch
        self.max_source_tokens = max(
            1,
            int(getattr(getattr(self.model, "config", None), "max_source_positions", 256)),
        )
        # A conservative character capability used by the shared context
        # planner. The exact token check below remains authoritative because
        # character-to-token ratios vary with source content.
        self.max_input_chars = 500
        self._model_stamp = _local_model_stamp(model_path)

    @property
    def display_name(self) -> str:
        return f"IndicTrans2 ({self.model_path})"

    @property
    def checkpoint_fingerprint(self) -> str:
        return (
            f"{self.display_name}|model={self._model_stamp}|"
            f"source_tokens={self.max_source_tokens}|max_length={self.max_new_tokens}|"
            "beams=5|processor=IndicProcessor"
        )

    def accepts_input(self, text: str, source_lang: str, target_lang: str) -> bool:
        return _indictrans_input_fits(
            self.processor,
            self.tokenizer,
            text,
            source_lang,
            target_lang,
            self.max_source_tokens,
        )

    def translate_batch(self, texts: Iterable[str], source_lang: str, target_lang: str) -> List[str]:
        src_tag = _lang_to_indic_tag(source_lang)
        tgt_tag = _lang_to_indic_tag(target_lang)

        batch = self.processor.preprocess_batch(
            list(texts),
            src_lang=src_tag,
            tgt_lang=tgt_tag,
        )

        inputs = self.tokenizer(
            batch,
            truncation=False,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )
        source_tokens = _sequence_length(inputs.get("input_ids"))
        if source_tokens is not None and source_tokens > self.max_source_tokens:
            raise IndicTransInputTooLongError(
                "IndicTrans2 input requires "
                f"{source_tokens} tokens but this local model supports "
                f"{self.max_source_tokens}. Reduce the context-window character "
                "budget or split the source block; input was not truncated."
            )
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=self.max_new_tokens,
                num_beams=5,
                num_return_sequences=1,
            )

        if _generation_hit_limit(
            generated_tokens,
            max_length=self.max_new_tokens,
            eos_token_id=getattr(getattr(self.model, "config", None), "eos_token_id", None),
            pad_token_id=getattr(getattr(self.model, "config", None), "pad_token_id", None),
        ):
            raise IndicTransOutputTooLongError(
                "IndicTrans2 output reached the generation limit without an end token. "
                "The incomplete translation was not accepted; reduce the context window."
            )

        decoded = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        results = self.processor.postprocess_batch(decoded, lang=tgt_tag)
        return [text.strip() for text in results]


def _lang_to_indic_tag(lang_code: str) -> str:
    mapping = {
        "en": "eng_Latn",
        "as": "asm_Beng",
        "bn": "ben_Beng",
        "brx": "brx_Deva",
        "doi": "doi_Deva",
        "gom": "gom_Deva",
        "gu": "guj_Gujr",
        "hi": "hin_Deva",
        "kn": "kan_Knda",
        "kok": "gom_Deva",
        "ks": "kas_Arab",
        "mai": "mai_Deva",
        "ml": "mal_Mlym",
        "mni": "mni_Beng",
        "mr": "mar_Deva",
        "ne": "npi_Deva",
        "or": "ory_Orya",
        "pa": "pan_Guru",
        "sa": "san_Deva",
        "sat": "sat_Olck",
        "sd": "snd_Arab",
        "ta": "tam_Taml",
        "te": "tel_Telu",
        "ur": "urd_Arab",
    }
    return mapping.get(lang_code, lang_code)


def _indictrans_input_fits(
    processor,
    tokenizer,
    text: str,
    source_lang: str,
    target_lang: str,
    max_source_tokens: int,
) -> bool:
    prepared = processor.preprocess_batch(
        [text],
        src_lang=_lang_to_indic_tag(source_lang),
        tgt_lang=_lang_to_indic_tag(target_lang),
    )
    encoded = tokenizer(
        prepared,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )
    length = _sequence_length(encoded.get("input_ids"))
    return length is None or length <= max_source_tokens


def _max_source_positions_from_config(model_path: str) -> int:
    config_path = Path(model_path).expanduser() / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        return max(1, int(payload.get("max_source_positions", 256)))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 256


def _sequence_length(input_ids) -> int | None:
    shape = getattr(input_ids, "shape", None)
    if shape is not None and len(shape) >= 1:
        try:
            return int(shape[-1])
        except (TypeError, ValueError):
            pass
    try:
        materialized = list(input_ids)
    except TypeError:
        return None
    if not materialized:
        return 0
    first = materialized[0]
    try:
        return len(first)
    except TypeError:
        return len(materialized)


def _generation_hit_limit(
    generated_tokens,
    *,
    max_length: int,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> bool:
    try:
        rows = generated_tokens.tolist()
    except AttributeError:
        rows = generated_tokens
    if not isinstance(rows, (list, tuple)):
        return False
    for row in rows:
        if not isinstance(row, (list, tuple)):
            continue
        tokens = list(row)
        if pad_token_id is not None:
            while tokens and tokens[-1] == pad_token_id:
                tokens.pop()
        if len(tokens) < max_length:
            continue
        if eos_token_id is None or not tokens or tokens[-1] != eos_token_id:
            return True
    return False


def _local_model_stamp(model_path: str) -> str:
    path = Path(model_path).expanduser()
    if not path.is_dir():
        return str(path)
    records: list[tuple[str, int, int]] = []
    patterns = (
        "*.json",
        "*.model",
        "*.safetensors",
        "*.bin",
    )
    seen: set[Path] = set()
    try:
        for pattern in patterns:
            for candidate in path.glob(pattern):
                if candidate in seen or not candidate.is_file():
                    continue
                seen.add(candidate)
                stat = candidate.stat()
                records.append((candidate.name, stat.st_size, stat.st_mtime_ns))
    except OSError:
        return str(path)
    payload = json.dumps(sorted(records), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]
