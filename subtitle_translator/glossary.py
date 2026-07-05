from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass
class GlossaryConfig:
    glossary_map: Dict[str, str]
    do_not_translate: List[str]


def load_glossary_json(raw_text: str | None) -> GlossaryConfig:
    if not raw_text:
        return GlossaryConfig(glossary_map={}, do_not_translate=[])

    data = json.loads(raw_text)
    glossary_map = data.get("glossary", {}) if isinstance(data, dict) else {}
    dnt = data.get("do_not_translate", []) if isinstance(data, dict) else []

    if not isinstance(glossary_map, dict):
        raise ValueError("'glossary' must be a JSON object")
    if not isinstance(dnt, list):
        raise ValueError("'do_not_translate' must be a JSON array")

    return GlossaryConfig(
        glossary_map={str(k): str(v) for k, v in glossary_map.items()},
        do_not_translate=[str(x) for x in dnt],
    )


# Bracket-free, all-alpha sentinels (ZZID{n}ZZ) survive IndicTrans2 verbatim.
# Tried `<dnt>...</dnt>` and `<IDn>`: both have their brackets stripped or
# garbled by the model (Â/dntÂ, w/dntw, <ID0′, etc.) — see scripts/spike_dnt.py.
# Pure alphabetic sequences are treated as opaque foreign tokens and passed
# through untouched.
_SENTINEL_PREFIX = "ZZID"
_SENTINEL_SUFFIX = "ZZ"


def _sentinel(n: int) -> str:
    return f"{_SENTINEL_PREFIX}{n}{_SENTINEL_SUFFIX}"


def protect_terms(texts: Iterable[str], terms: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Replace each term with a ZZID{n}ZZ sentinel.

    Returns the substituted texts plus a replacements map (sentinel → original)
    that `restore_terms` uses to put the originals back. The same term across
    different texts gets the same sentinel ID so the map stays compact.
    """
    if not terms:
        return list(texts), {}

    ordered = sorted({t for t in terms if t}, key=len, reverse=True)
    term_to_sentinel: Dict[str, str] = {
        term: _sentinel(i) for i, term in enumerate(ordered)
    }

    protected: List[str] = []
    for text in texts:
        out = text
        for term, sentinel in term_to_sentinel.items():
            # Pad with spaces so the model is less likely to fuse the
            # sentinel with neighbouring tokens (it was producing things
            # like `MENID209ZZ` and `USID134ZZHuh` without the padding).
            out = re.sub(
                rf"\b{re.escape(term)}\b", f" {sentinel} ", out, flags=re.IGNORECASE
            )
        # Collapse the runs of spaces we just introduced.
        out = re.sub(r"[ \t]{2,}", " ", out).strip()
        # Don't strip space before punctuation we care about.
        out = re.sub(r"\s+([,.;:!?])", r"\1", out)
        protected.append(out)

    replacements = {sentinel: term for term, sentinel in term_to_sentinel.items()}
    return protected, replacements


# Restoration handles every corruption mode observed in production output:
#   1. Clean: `ZZID3ZZ`
#   2. Mangled Z's: `ZZID3Z`, `ID3ZZ`, `Z*ID3Z*`
#   3. Letter swaps: `ZZIT3ZZ` (D→T), `ZSID3ZZ` (Z→S)
#   4. Prefix-fused with model's re-injected source: `MENID3ZZ` (single-word
#      saved terms only — eating the prefix for multi-word terms duplicates)
#   5. Suffix-fused: `ID134ZZHuh` — we leave Huh alone (suffix is often
#      meaningful next-word text the model failed to put a space before)
#   6. Bengali letter spell-out: `জেড. জেড. আই. ডি. 3 জেড. জেড.`
#   7. Hindi letter spell-out: `जेड. जेड. आई. डी. 3 जेड. जेड.`
#
# Space-padding in protect_terms makes fusion rare to begin with, so the
# regex below is intentionally conservative: it requires *some* sentinel
# residue (leading or trailing Z, or the explicit ID/IT marker) to fire.

# Main Latin matcher. Requires either leading or trailing Z(s) so it can't
# false-positive on stray English words like "kid3" or "Sid8".
_SENTINEL_RE = re.compile(
    r"[Zz]+[A-Za-z]{0,12}?[Ii][DdTt](\d+)[Zz]*"   # leading Zs required
    r"|"
    r"[A-Za-z]{0,12}?[Ii][DdTt](\d+)[Zz]+"        # trailing Zs required
)

# Bengali letter-by-letter spelling (Z=জেড, I=আই, D=ডি).
_SENTINEL_RE_BN = re.compile(
    r"(?:জেড\s*\.?\s*)*\s*আই\s*\.?\s*ডি\s*\.?\s*(\d+)"
    r"(?:\s*জেড\s*\.?\s*)*"
)

# Hindi letter-by-letter spelling.
_SENTINEL_RE_HI = re.compile(
    r"(?:जेड\s*\.?\s*)*\s*आई\s*\.?\s*डी\s*\.?\s*(\d+)"
    r"(?:\s*जेड\s*\.?\s*)*"
)

# When the model strips "ID" entirely, just the digits + Z's survive
# (observed: `Monsieur 210ZZ`). Match conservatively: 1–4 digits surrounded
# by non-alphanumerics, followed by 2+ Z's. We try it as a lossy *restore*
# (look up the index) before the orphan sweep.
_SENTINEL_ID_STRIPPED_RE = re.compile(
    r"(?<![A-Za-z\d])(\d{1,4})Z{2,}(?![A-Za-z\d])"
)

# Last-resort cleanup: any orphan sentinel-like debris becomes empty so it
# never reaches the viewer.
_ORPHAN_RE = re.compile(
    r"\b[Zz]+[Ii][DdTt]\d+[Zz]*\b"          # canonical with surrounding Z's
    r"|\b[Ii][DdTt]\d+[Zz]+\b"              # trailing Z form
    r"|(?<![A-Za-z\d])\d{1,4}Z{2,}(?![A-Za-z\d])"  # ID-stripped leftover
)


def _lookup(idx_str: str, replacements: Dict[str, str]) -> str | None:
    try:
        return replacements.get(_sentinel(int(idx_str)))
    except (ValueError, TypeError):
        return None


def restore_terms(texts: Iterable[str], replacements: Dict[str, str]) -> List[str]:
    """Substitute sentinels back, handling letter-fusion and target-script
    transliteration of the sentinel itself."""
    if not replacements:
        return [t.strip() for t in texts]

    def _sub_latin(match: re.Match) -> str:
        idx = match.group(1) or match.group(2)
        saved = _lookup(idx, replacements)
        if saved is None:
            return match.group(0)
        # If the model fused a re-injected source word as prefix that also
        # happens to be the LAST word of a multi-word saved term, the text
        # before us already contains the preceding words — only emit the
        # last word so we don't duplicate.
        full = match.group(0)
        prefix_m = re.match(r"[Zz]*([A-Za-z]+?)[Ii][DdTt]\d+", full)
        prefix = prefix_m.group(1) if prefix_m else ""
        saved_words = saved.split()
        if (
            prefix and len(saved_words) > 1
            and prefix.lower() == saved_words[-1].lower()
        ):
            return " " + saved_words[-1] + " "
        return " " + saved + " "

    def _sub_indic(match: re.Match) -> str:
        saved = _lookup(match.group(1), replacements)
        return " " + (saved or "") + " "

    result: List[str] = []
    for text in texts:
        # Pass 1: Bengali/Hindi letter-by-letter forms first (most specific).
        for indic_re in (_SENTINEL_RE_BN, _SENTINEL_RE_HI):
            text = indic_re.sub(_sub_indic, text)
        # Pass 2: main Latin sentinel matcher (handles prefix-fused too).
        text = _SENTINEL_RE.sub(_sub_latin, text)
        # Pass 3: model occasionally strips "ID" entirely leaving just
        # `<num>ZZ`. Look it up like a real sentinel before the orphan sweep
        # wipes it; otherwise valid restores would be lost as garbage.
        text = _SENTINEL_ID_STRIPPED_RE.sub(_sub_indic, text)
        # Pass 4: nuke any leftover sentinel-like debris (indices we never
        # assigned, or shapes the above passes couldn't restore).
        text = _ORPHAN_RE.sub("", text)
        # Tidy spacing.
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\s+([,.;:!?।])", r"\1", text)
        result.append(text.strip())
    return result


def apply_glossary_overrides(texts: Iterable[str], glossary_map: Dict[str, str]) -> List[str]:
    if not glossary_map:
        return list(texts)

    items = sorted(glossary_map.items(), key=lambda kv: len(kv[0]), reverse=True)
    out: List[str] = []
    for text in texts:
        transformed = text
        for source, target in items:
            pattern = re.compile(rf"\b{re.escape(source)}\b", re.IGNORECASE)
            transformed = pattern.sub(target, transformed)
        out.append(transformed)
    return out
