from __future__ import annotations

import sys
import types
from contextlib import nullcontext

import pytest

from subtitle_translator.translators.indictrans2 import (
    IndicTransInputTooLongError,
    IndicTrans2Translator,
    IndicTransOutputTooLongError,
    _lang_to_indic_tag,
)


def _install_fake_runtime(
    monkeypatch,
    *,
    sequence_length: int = 8,
    generated_sequence: list[int] | None = None,
):
    calls: dict[str, object] = {
        "model_devices": [],
        "tensor_devices": [],
    }

    class FakeTensor:
        shape = (2, sequence_length)

        def to(self, device):
            calls["tensor_devices"].append(device)
            return self

    class FakeProcessor:
        def __init__(self, *, inference):
            calls["processor_inference"] = inference

        def preprocess_batch(self, texts, *, src_lang, tgt_lang):
            calls["preprocess"] = {
                "texts": texts,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
            }
            return [f"prepared:{text}" for text in texts]

        def postprocess_batch(self, texts, *, lang):
            calls["postprocess"] = {"texts": texts, "lang": lang}
            return [f" restored:{text} " for text in texts]

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            calls["tokenize"] = {"texts": texts, "kwargs": kwargs}
            return {
                "input_ids": FakeTensor(),
                "attention_mask": FakeTensor(),
            }

        def batch_decode(self, generated_tokens, **kwargs):
            calls["decode"] = {
                "generated_tokens": generated_tokens,
                "kwargs": kwargs,
            }
            return ["decoded-one", "decoded-two"]

    class FakeModel:
        config = types.SimpleNamespace(
            max_source_positions=256,
            eos_token_id=2,
            pad_token_id=1,
        )

        def eval(self):
            calls["model_eval"] = True

        def to(self, device):
            calls["model_devices"].append(device)
            return self

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return (
                [generated_sequence]
                if generated_sequence is not None
                else ["generated-token-ids"]
            )

    tokenizer = FakeTokenizer()
    model = FakeModel()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            calls["tokenizer_load"] = {"model_path": model_path, "kwargs": kwargs}
            return tokenizer

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            calls["model_load"] = {"model_path": model_path, "kwargs": kwargs}
            return model

    torch_module = types.ModuleType("torch")
    torch_module.no_grad = nullcontext

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoTokenizer = FakeAutoTokenizer
    transformers_module.AutoModelForSeq2SeqLM = FakeAutoModel

    toolkit_module = types.ModuleType("IndicTransToolkit")
    toolkit_module.IndicProcessor = FakeProcessor

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "IndicTransToolkit", toolkit_module)
    return calls


def test_indictrans2_loads_offline_model_and_official_processor(monkeypatch):
    calls = _install_fake_runtime(monkeypatch)

    translator = IndicTrans2Translator("/models/indictrans2", device="cuda")

    expected_load = {
        "model_path": "/models/indictrans2",
        "kwargs": {"local_files_only": True, "trust_remote_code": True},
    }
    assert calls["processor_inference"] is True
    assert calls["tokenizer_load"] == expected_load
    assert calls["model_load"] == expected_load
    assert calls["model_eval"] is True
    assert calls["model_devices"] == ["cuda"]
    assert translator.display_name == "IndicTrans2 (/models/indictrans2)"


def test_indictrans2_uses_official_preprocess_generate_postprocess_flow(monkeypatch):
    calls = _install_fake_runtime(monkeypatch)
    translator = IndicTrans2Translator(
        "/models/indictrans2",
        device="cuda",
        max_new_tokens=96,
    )

    output = translator.translate_batch(
        (text for text in ["First sentence.", "Second sentence."]),
        source_lang="en",
        target_lang="bn",
    )

    assert calls["preprocess"] == {
        "texts": ["First sentence.", "Second sentence."],
        "src_lang": "eng_Latn",
        "tgt_lang": "ben_Beng",
    }
    assert calls["tokenize"] == {
        "texts": ["prepared:First sentence.", "prepared:Second sentence."],
        "kwargs": {
            "truncation": False,
            "padding": "longest",
            "return_tensors": "pt",
            "return_attention_mask": True,
        },
    }
    assert calls["tensor_devices"] == ["cuda", "cuda"]

    generation = calls["generate"]
    assert generation["num_beams"] == 5
    assert generation["num_return_sequences"] == 1
    assert generation["max_length"] == 96
    assert "repetition_penalty" not in generation
    assert "no_repeat_ngram_size" not in generation

    assert calls["decode"] == {
        "generated_tokens": ["generated-token-ids"],
        "kwargs": {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": True,
        },
    }
    assert calls["postprocess"] == {
        "texts": ["decoded-one", "decoded-two"],
        "lang": "ben_Beng",
    }
    assert output == ["restored:decoded-one", "restored:decoded-two"]


def test_indictrans2_rejects_over_budget_input_without_truncating(monkeypatch):
    _install_fake_runtime(monkeypatch, sequence_length=257)
    translator = IndicTrans2Translator("/models/indictrans2")

    with pytest.raises(IndicTransInputTooLongError, match="input requires 257 tokens"):
        translator.translate_batch(["A deliberately long input."], "en", "bn")


def test_indictrans2_rejects_generation_cut_off_at_limit(monkeypatch):
    _install_fake_runtime(
        monkeypatch,
        generated_sequence=list(range(12)),
    )
    translator = IndicTrans2Translator("/models/indictrans2", max_new_tokens=12)

    with pytest.raises(IndicTransOutputTooLongError, match="reached the generation limit"):
        translator.translate_batch(["A source input."], "en", "bn")


def test_indictrans2_maps_the_full_ui_indic_language_set():
    expected = {
        "as": "asm_Beng",
        "bn": "ben_Beng",
        "brx": "brx_Deva",
        "doi": "doi_Deva",
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

    assert {code: _lang_to_indic_tag(code) for code in expected} == expected
