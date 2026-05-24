"""Phase 1 spike v3: find a placeholder format the model passes through verbatim,
AND evaluate whether raw transliteration is good enough that we don't need one.

Strategies tested:
  RAW       — no protection.
  DNT       — <dnt>term</dnt> (claimed by IndicTrans2 docs).
  IDPLACE   — <IDn> (IndicTransToolkit-style).
  ZZID      — ZZIDnZZ (bracket-free, all-alpha sentinel).
  QQNUM     — QQ9990QQ-style sentinels (pure alnum).

Also runs the existing _transliterate_deva script-conversion so output
matches what the GUI/pipeline actually produces, not raw Devanagari.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from subtitle_translator.translators.indictrans2 import _SCRIPT_MAP, _transliterate_deva

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "indictrans2-en-indic"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Generic test inputs — placeholder names + non-naturalized foreign phrase.
# Not tied to any specific film.
SAMPLES: List[Tuple[str, List[str]]] = [
    ("I met Alice in Tokyo.", ["Alice", "Tokyo"]),
    ("Bob said bonjour at the station.", ["Bob", "bonjour"]),
    ("Welcome to Berlin, madame.", ["Berlin"]),
    ("Carol told Dave to meet at Yokohama.", ["Carol", "Dave", "Yokohama"]),
]
TARGETS = [
    ("eng_Latn", "hin_Deva"),
    ("eng_Latn", "ben_Beng"),
    ("eng_Latn", "tam_Taml"),
]


def load_model():
    print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
    tok = AutoTokenizer.from_pretrained(
        str(MODEL_PATH), local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(MODEL_PATH), local_files_only=True, trust_remote_code=True
    )
    model.eval()
    if DEVICE != "cpu":
        model = model.to(DEVICE)
    return tok, model


def generate(tok, model, prefixed_texts):
    inputs = tok(prefixed_texts, truncation=True, padding="longest",
                 return_tensors="pt", return_attention_mask=True)
    if DEVICE != "cpu":
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, use_cache=True, min_length=0,
                             max_length=256, num_beams=5, num_return_sequences=1)
    return [s.strip() for s in tok.batch_decode(
        out, skip_special_tokens=True, clean_up_tokenization_spaces=True)]


def script_convert(text, tgt):
    script = _SCRIPT_MAP.get(tgt)
    return _transliterate_deva(text, script) if script else text


# --- Strategies ---

def prep_raw(s, t): return s, {}
def restore_raw(o, m): return o

def prep_dnt(s, terms):
    out = s
    for t in sorted(set(terms), key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(t)}\b", f"<dnt>{t}</dnt>", out)
    return out, {}
def restore_dnt(o, m):
    return re.sub(r"</?dnt>|Â/dnt\S*|w/dnt\S*|·/dnt\S*|̃/dnt\S*", "", o)

def prep_idplace(s, terms):
    mapping = {}
    out = s
    for i, t in enumerate(sorted(set(terms), key=len, reverse=True)):
        token = f"<ID{i}>"
        mapping[token] = t
        out = re.sub(rf"\b{re.escape(t)}\b", token, out)
    return out, mapping
def restore_idplace(o, m):
    # Match <ID0 with whatever the model appended (>, ′, etc.)
    for k, v in m.items():
        n = re.search(r"\d+", k).group(0)
        o = re.sub(rf"<\s*ID\s*{n}\s*\S{{0,3}}?", v, o, flags=re.IGNORECASE)
    return o

def prep_zzid(s, terms):
    mapping = {}
    out = s
    for i, t in enumerate(sorted(set(terms), key=len, reverse=True)):
        token = f"ZZID{i}ZZ"
        mapping[token] = t
        out = re.sub(rf"\b{re.escape(t)}\b", token, out)
    return out, mapping
def restore_zzid(o, m):
    for k, v in m.items():
        n = re.search(r"\d+", k).group(0)
        # Allow the model to mangle the trailing ZZ but keep ZZIDn
        o = re.sub(rf"ZZID{n}Z*", v, o, flags=re.IGNORECASE)
    return o

def prep_qqnum(s, terms):
    mapping = {}
    out = s
    for i, t in enumerate(sorted(set(terms), key=len, reverse=True)):
        token = f"QQ999{i}QQ"
        mapping[token] = t
        out = re.sub(rf"\b{re.escape(t)}\b", token, out)
    return out, mapping
def restore_qqnum(o, m):
    for k, v in m.items():
        n = re.search(r"\d+", k).group(0)
        o = re.sub(rf"QQ{n}Q*", v, o, flags=re.IGNORECASE)
    return o


STRATEGIES = [
    ("RAW", prep_raw, restore_raw),
    ("DNT", prep_dnt, restore_dnt),
    ("IDPLACE", prep_idplace, restore_idplace),
    ("ZZID", prep_zzid, restore_zzid),
    ("QQNUM", prep_qqnum, restore_qqnum),
]


def run_strategy(tok, model, name, prep, restore, src, tgt):
    prepped = [prep(s, terms) for s, terms in SAMPLES]
    inputs_only = [p[0] for p in prepped]
    mappings = [p[1] for p in prepped]
    prefixed = [f"{src} {tgt} {t}" for t in inputs_only]
    decoded = generate(tok, model, prefixed)
    converted = [script_convert(d, tgt) for d in decoded]
    restored = [restore(o, m) for o, m in zip(converted, mappings)]

    survived = 0
    total = 0
    for (sentence, terms), prep_in, raw_out, final_out in zip(
        SAMPLES, inputs_only, converted, restored
    ):
        print(f"  [{name}] in : {sentence}")
        if prep_in != sentence:
            print(f"          prep: {prep_in}")
        print(f"          raw : {raw_out}")
        if final_out != raw_out:
            print(f"          fin : {final_out}")
        for t in terms:
            total += 1
            if t in final_out:
                survived += 1
    return survived, total


def main() -> int:
    tok, model = load_model()
    results: dict = {}
    for src, tgt in TARGETS:
        print(f"\n=== {src} → {tgt} ===")
        for name, prep, restore in STRATEGIES:
            s, t = run_strategy(tok, model, name, prep, restore, src, tgt)
            results.setdefault(name, []).append((tgt, s, t))
            print(f"  [{name}] survival: {s}/{t}")

    print("\n=== SUMMARY (Roman-form survival rate) ===")
    best = None
    for name, rows in results.items():
        total_s = sum(s for _, s, _ in rows)
        total_t = sum(t for _, _, t in rows)
        rate = total_s / total_t if total_t else 0
        print(f"  {name}: {total_s}/{total_t} = {rate:.0%}")
        if best is None or rate > best[1]:
            best = (name, rate)

    if best and best[1] >= 0.9:
        print(f"\nPASS: strategy {best[0]} preserved {best[1]:.0%} of terms")
        return 0
    print(f"\nFAIL: best strategy {best[0]} only preserved {best[1]:.0%}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
