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


# Matches an existing <dnt>...</dnt> span so we can skip re-wrapping its content.
_EXISTING_DNT_RE = re.compile(r'<dnt>.*?</dnt>', re.DOTALL)


def protect_terms(texts: Iterable[str], terms: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Wrap each term in <dnt>...</dnt> tags.

    IndicTrans2 is trained with IndicTransToolkit which uses this exact tag
    format for entity protection — the model knows to pass tag contents through
    unchanged.  We split on existing <dnt> spans before applying each pattern
    to prevent double-wrapping when a shorter term is a substring of a longer
    one that was already protected.
    """
    if not terms:
        return list(texts), {}

    protected = list(texts)
    ordered = sorted({t for t in terms if t}, key=len, reverse=True)

    for term in ordered:
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        new_protected = []
        for text in protected:
            # Split into [plain, dnt-span, plain, dnt-span, ...] parts so we
            # only apply the pattern to the plain segments.
            segments = _EXISTING_DNT_RE.split(text)
            tags = _EXISTING_DNT_RE.findall(text)
            result = []
            for i, seg in enumerate(segments):
                result.append(pattern.sub(r'<dnt>\g<0></dnt>', seg))
                if i < len(tags):
                    result.append(tags[i])
            new_protected.append(''.join(result))
        protected = new_protected

    # Return an empty replacements dict for API compatibility; callers pass it
    # to restore_terms which no longer needs it with the <dnt> approach.
    return protected, {}


def restore_terms(texts: Iterable[str], replacements: Dict[str, str]) -> List[str]:
    """Strip <dnt> tags, preserving their contents.

    If the model preserved a <dnt>TERM</dnt> span verbatim the content is
    returned as-is.  If the model translated the content inside the tags that
    translation is returned (usually still acceptable).  Broken tag fragments
    are stripped so they never appear in viewer-facing output.
    """
    result = []
    for text in texts:
        t = re.sub(r'<dnt>(.*?)</dnt>', r'\1', text, flags=re.DOTALL)
        t = re.sub(r'</?dnt>', '', t).strip()
        result.append(t)
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
