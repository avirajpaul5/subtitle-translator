"""Auto-detection of terms to preserve during translation.

Three detection layers, all feeding the same `do_not_translate` list:
  Layer 1a (NER):       spaCy entity recognition catches multi-token proper
                        nouns (PERSON, GPE, LOC, ORG, WORK_OF_ART, FAC,
                        PRODUCT, NORP, EVENT) — including phrase spans.
  Layer 1b (POS=PROPN): catches single-token proper nouns the NER missed
                        (less common names, unusual place names).
  Layer 2 (frequency):  tokens with corpus zipf-frequency below threshold
                        are treated as non-naturalized in English.

A token is preserved if any layer flags it. spaCy is loaded lazily so the
import stays cheap when auto-detection is unused.
"""
from __future__ import annotations

import re
from typing import List, Set

from subtitle_translator.models import SubtitleDocument
from subtitle_translator.speaker_detection import _SKIP_WORDS

# Entity types worth preserving as-is during translation.
_PRESERVE_LABELS = {
    "PERSON", "GPE", "LOC", "ORG", "WORK_OF_ART", "FAC", "PRODUCT",
    "NORP",  # nationalities/religious-political groups (often proper nouns)
    "EVENT",
}

# Tokens below this zipf score are considered non-naturalized in English.
# Calibrated so common words ("said"=6.0, "station"=5.0) stay above and
# foreign loans ("bonjour"=2.8, "oeufs"=1.2) fall below.
DEFAULT_ZIPF_THRESHOLD = 3.0

# Strip punctuation around tokens before frequency lookup.
_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)

# Subtitle formatting tags (<i>, <b>, <u>, <font ...>, etc.) — strip before
# NER so the tokenizer doesn't fuse them with surrounding words.
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*/?>")

# Leading articles to strip from multi-word entity spans
# (so "the Bay of Bengal" becomes "Bay of Bengal").
_LEADING_ARTICLES = {"the", "a", "an"}


def _load_spacy():
    """Lazy-load spaCy with the small English NER model. Cached on the function."""
    if _load_spacy._nlp is None:
        import spacy
        _load_spacy._nlp = spacy.load(
            "en_core_web_sm",
            # Disable components we don't need; keeps inference fast.
            disable=["lemmatizer", "textcat"],
        )
    return _load_spacy._nlp
_load_spacy._nlp = None  # type: ignore[attr-defined]


def _is_protectable(token: str) -> bool:
    """Filter out anything that shouldn't go into the do-not-translate list."""
    if len(token) < 2:
        return False
    if token.upper() in _SKIP_WORDS:
        return False
    if not any(c.isalpha() for c in token):
        return False
    return True


def detect_preserve_spans(
    document: SubtitleDocument,
    zipf_threshold: float = DEFAULT_ZIPF_THRESHOLD,
) -> List[str]:
    """Return a deduplicated, length-sorted list of terms to preserve.

    Length-sorted (longest first) so that downstream `protect_terms` substitutes
    multi-word phrases before their constituent tokens.
    """
    nlp = _load_spacy()
    from wordfreq import zipf_frequency

    full_text = "\n".join(cue.text for cue in document.cues)
    # Strip subtitle formatting tags before NER; otherwise spaCy tokenises
    # things like "italic</i>" as a single token.
    full_text = _HTML_TAG_RE.sub(" ", full_text)
    doc = nlp(full_text)

    found: Set[str] = set()

    # Layer 1a — NER: keep multi-word entity surface forms verbatim.
    entity_token_ids: Set[int] = set()
    for ent in doc.ents:
        if ent.label_ not in _PRESERVE_LABELS:
            continue
        # Drop a leading article ("the Bay of Bengal" → "Bay of Bengal").
        tokens = list(ent)
        if tokens and tokens[0].text.lower() in _LEADING_ARTICLES:
            tokens = tokens[1:]
        if not tokens:
            continue
        span = " ".join(t.text for t in tokens).strip()
        if _is_protectable(span):
            found.add(span)
            entity_token_ids.update(t.i for t in tokens)

    # Layer 1b + 2 — per-token: catch PROPN-tagged tokens that NER missed,
    # plus any low-frequency tokens treated as non-naturalized.
    for token in doc:
        if token.i in entity_token_ids:
            continue
        if token.is_space or token.is_punct or token.like_num:
            continue
        word = _PUNCT_RE.sub("", token.text)
        if not _is_protectable(word):
            continue

        is_propn = token.pos_ == "PROPN"
        is_rare = zipf_frequency(word.lower(), "en") < zipf_threshold
        if is_propn or is_rare:
            found.add(word)

    return sorted(found, key=len, reverse=True)
