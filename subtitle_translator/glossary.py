from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass
class GlossaryConfig:
    glossary_map: Dict[str, str]
    do_not_translate: List[str]


TOKEN_PREFIX = "__DNT_"
TOKEN_SUFFIX = "__"


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


def protect_terms(texts: Iterable[str], terms: List[str]) -> Tuple[List[str], Dict[str, str]]:
    replacements: Dict[str, str] = {}
    protected_texts = list(texts)

    ordered_terms = sorted({term for term in terms if term}, key=len, reverse=True)
    for idx, term in enumerate(ordered_terms):
        token = f"{TOKEN_PREFIX}{idx}{TOKEN_SUFFIX}"
        replacements[token] = term
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        protected_texts = [pattern.sub(token, text) for text in protected_texts]

    return protected_texts, replacements


def restore_terms(texts: Iterable[str], replacements: Dict[str, str]) -> List[str]:
    restored = []
    for text in texts:
        updated = text
        for token, original in replacements.items():
            updated = updated.replace(token, original)
        restored.append(updated)
    return restored


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
