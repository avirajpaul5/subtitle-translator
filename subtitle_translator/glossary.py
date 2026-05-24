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
            out = re.sub(rf"\b{re.escape(term)}\b", sentinel, out, flags=re.IGNORECASE)
        protected.append(out)

    replacements = {sentinel: term for term, sentinel in term_to_sentinel.items()}
    return protected, replacements


# Tolerant matcher: in addition to the canonical ZZID{n}ZZ, accept:
#  - extra/missing trailing Z's
#  - single-letter swaps in the prefix (the model occasionally turns D → T,
#    producing forms like ZZIT4ZZ)
# Z[ZS] handles ZS where the model swaps in S; I[DT] handles D↔T.
_SENTINEL_RE = re.compile(r"Z[ZS]I[DT](\d+)Z*", re.IGNORECASE)


def restore_terms(texts: Iterable[str], replacements: Dict[str, str]) -> List[str]:
    """Substitute ZZID{n}ZZ sentinels back to their original terms."""
    if not replacements:
        return [t.strip() for t in texts]

    result: List[str] = []
    for text in texts:
        def _sub(match: re.Match) -> str:
            key = _sentinel(int(match.group(1)))
            return replacements.get(key, match.group(0))
        result.append(_SENTINEL_RE.sub(_sub, text).strip())
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
