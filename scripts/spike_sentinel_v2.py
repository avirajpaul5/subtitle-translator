"""Search for a more robust sentinel format.

Tries variants against the same 4 sentences x 3 targets and measures
sentinel-token survival (the prefix surviving so we can substitute back).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from subtitle_translator.translators.indictrans2 import _SCRIPT_MAP, _transliterate_deva

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "indictrans2-en-indic"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

TARGETS = [("eng_Latn", "hin_Deva"), ("eng_Latn", "ben_Beng"), ("eng_Latn", "tam_Taml")]

# Generic templates with in-place sentinel slots. Not tied to any specific film.
SAMPLES = [
    ("I met {S0} in Tokyo.", ["{S0}"]),
    ("{S0} said {S1} at the station.", ["{S0}", "{S1}"]),
    ("Welcome to {S0}, madame.", ["{S0}"]),
    ("{S0} told {S1} to meet at {S2}.", ["{S0}", "{S1}", "{S2}"]),
]

# Candidate sentinel families. {n} is replaced with index per sample.
SENTINEL_FAMILIES = {
    "ZZID":     "ZZID{n}ZZ",
    "XQX":      "XQX{n}XQX",
    "NNXn":     "NN{n}NN",
    "MARKER":   "MARKER{n}MARKER",
    "XKQ":      "XKQ{n}XKQ",
    "DNX":      "DNX{n}DNX",
}


def load_model():
    print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_PATH), local_files_only=True, trust_remote_code=True)
    model.eval()
    if DEVICE != "cpu":
        model = model.to(DEVICE)
    return tok, model


def generate(tok, model, prefixed):
    inputs = tok(prefixed, truncation=True, padding="longest",
                 return_tensors="pt", return_attention_mask=True)
    if DEVICE != "cpu":
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, use_cache=True, min_length=0,
                             max_length=256, num_beams=5, num_return_sequences=1)
    return [s.strip() for s in tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)]


def script_convert(text, tgt):
    script = _SCRIPT_MAP.get(tgt)
    return _transliterate_deva(text, script) if script else text


def survived(template, sentinel_count, output):
    # Check that every numbered sentinel (n=0..count-1) appears verbatim.
    placed_count = 0
    for n in range(sentinel_count):
        marker = template.format(n=n)
        if marker in output:
            placed_count += 1
    return placed_count, sentinel_count


def main():
    tok, model = load_model()
    print()

    for fam_name, template in SENTINEL_FAMILIES.items():
        total_s = 0
        total_t = 0
        print(f"\n=== Family {fam_name}  ({template}) ===")
        for src, tgt in TARGETS:
            # Build inputs by substituting slots
            inputs_only = []
            sentinel_counts = []
            for sentence, slots in SAMPLES:
                vals = {f"S{i}": template.format(n=i) for i in range(len(slots))}
                rendered = sentence.format(**vals)
                inputs_only.append(rendered)
                sentinel_counts.append(len(slots))
            prefixed = [f"{src} {tgt} {t}" for t in inputs_only]
            decoded = generate(tok, model, prefixed)
            converted = [script_convert(d, tgt) for d in decoded]

            lang_s, lang_t = 0, 0
            for inp, out, sc in zip(inputs_only, converted, sentinel_counts):
                s, t = survived(template, sc, out)
                lang_s += s
                lang_t += t
            print(f"  {tgt}: {lang_s}/{lang_t}")
            total_s += lang_s
            total_t += lang_t

        rate = total_s / total_t if total_t else 0
        print(f"  TOTAL {fam_name}: {total_s}/{total_t} = {rate:.0%}")


if __name__ == "__main__":
    main()
