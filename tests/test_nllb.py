from __future__ import annotations

import sys
import types

import pytest

from subtitle_translator.translators.nllb import (
    NLLBInputTooLongError,
    NLLBOutputTooLongError,
    NLLBTranslator,
)


def _install_fake_transformers(
    monkeypatch,
    *,
    sequence_length: int = 8,
    generated: list[int] | None = None,
):
    calls: dict[str, object] = {}

    class FakeIds:
        shape = (1, sequence_length)

    class FakeTokenizer:
        model_max_length = 16
        src_lang = None

        def __call__(self, texts, **kwargs):
            calls["tokenize"] = {
                "texts": texts,
                "kwargs": kwargs,
                "src_lang": self.src_lang,
            }
            return {"input_ids": FakeIds()}

        def convert_tokens_to_ids(self, value):
            calls["target_tag"] = value
            return 42

        def batch_decode(self, values, **kwargs):
            calls["decode"] = kwargs
            return ["translated"]

    class FakeModel:
        config = types.SimpleNamespace(
            max_position_embeddings=32,
            eos_token_id=2,
            pad_token_id=1,
        )

        def eval(self):
            calls["eval"] = True

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return [generated if generated is not None else [4, 2]]

    tokenizer = FakeTokenizer()
    model = FakeModel()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["tokenizer_load"] = (path, kwargs)
            return tokenizer

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["model_load"] = (path, kwargs)
            return model

    module = types.ModuleType("transformers")
    module.AutoTokenizer = FakeAutoTokenizer
    module.AutoModelForSeq2SeqLM = FakeAutoModel
    monkeypatch.setitem(sys.modules, "transformers", module)
    return calls


def test_nllb_uses_runtime_language_tags_without_truncation(monkeypatch):
    calls = _install_fake_transformers(monkeypatch)
    translator = NLLBTranslator("/models/nllb")

    assert translator.translate_batch(["Hello"], "en", "bn") == ["translated"]
    assert calls["tokenizer_load"] == ("/models/nllb", {"local_files_only": True})
    assert calls["model_load"] == ("/models/nllb", {"local_files_only": True})
    assert calls["eval"] is True
    assert calls["tokenize"] == {
        "texts": ["Hello"],
        "kwargs": {
            "return_tensors": "pt",
            "padding": True,
            "truncation": False,
        },
        "src_lang": "eng_Latn",
    }
    assert calls["target_tag"] == "ben_Beng"
    assert calls["generate"]["forced_bos_token_id"] == 42


def test_nllb_rejects_over_budget_input(monkeypatch):
    _install_fake_transformers(monkeypatch, sequence_length=17)
    translator = NLLBTranslator("/models/nllb")

    assert not translator.accepts_input("long", "en", "bn")
    with pytest.raises(NLLBInputTooLongError, match="Input was not truncated"):
        translator.translate_batch(["long"], "en", "bn")


def test_nllb_rejects_output_cut_off_at_generation_cap(monkeypatch):
    _install_fake_transformers(monkeypatch, generated=list(range(6)))
    translator = NLLBTranslator("/models/nllb", max_new_tokens=6)

    with pytest.raises(NLLBOutputTooLongError, match="reached the generation limit"):
        translator.translate_batch(["Hello"], "en", "bn")
