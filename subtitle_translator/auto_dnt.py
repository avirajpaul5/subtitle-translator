"""Auto-detection of terms to preserve during translation.

Three detection layers, all feeding the same `do_not_translate` list:
  Layer 1a (NER):       spaCy entity recognition catches multi-token proper
                        nouns (PERSON, GPE, LOC, ORG, WORK_OF_ART, FAC,
                        PRODUCT, NORP, EVENT) — including phrase spans.
  Layer 1b (POS=PROPN): catches single-token proper nouns the NER missed
                        (less common names, unusual place names). Filtered
                        by frequency to avoid sentence-start capitalization
                        of common nouns like "Detective" or "Doctor".
  Layer 2 (frequency):  tokens with corpus zipf-frequency below threshold
                        are treated as non-naturalized in English.

Before any layer runs, the text is cleaned to remove sources of NER noise:
parenthesized stage directions, ALL-CAPS speaker labels, and HTML formatting
tags. This avoids preserving sound-effect words like BELL, HORN, CAMERA.

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

# PROPN tokens *without* an entity label only get preserved if their zipf
# frequency is below this. Keeps genuine but NER-missed names (Carol≈3.9)
# while excluding sentence-start common nouns (Doctor=4.9, Detective=4.2,
# Huh=4.3) that spaCy POS-tags as PROPN purely due to capitalization.
PROPN_NOENTITY_MAX_ZIPF = 4.0

# Strip punctuation around tokens before frequency lookup.
_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)

# Subtitle formatting tags (<i>, <b>, <u>, <font ...>, etc.) — strip before
# NER so the tokenizer doesn't fuse them with surrounding words.
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*/?>")

# Parenthesized content is conventionally untranslated stage direction or
# sound effect — strip before NER to avoid flagging BELL/HORN/MEN/CAMERA.
_PARENS_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]")

# ALL-CAPS speaker labels ("POIROT: …") — pipeline handles them separately,
# and feeding them to NER produces spurious PROPN tags for common dialogue
# words on subsequent lines. Mirrors the regex in pipeline._extract_speaker_label.
_SPEAKER_LABEL_RE = re.compile(r"^[A-Z][A-Z\s\-\.\']{0,30}:\s*", re.MULTILINE)

# Leading articles to strip from multi-word entity spans
# (so "the Bay of Bengal" becomes "Bay of Bengal").
_LEADING_ARTICLES = {"the", "a", "an"}

# Pronouns and other short interjections sometimes mis-tagged as PROPN or
# entity (spaCy parses bare "US" as GPE=United States). They should never
# survive into the preserve list.
_NEVER_PRESERVE = {
    "I", "ME", "MY", "MINE", "MYSELF",
    "WE", "US", "OUR", "OURS", "OURSELVES",
    "YOU", "YOUR", "YOURS", "YOURSELF", "YOURSELVES",
    "HE", "HIM", "HIS", "HIMSELF",
    "SHE", "HER", "HERS", "HERSELF",
    "IT", "ITS", "ITSELF",
    "THEY", "THEM", "THEIR", "THEIRS", "THEMSELVES",
    "THIS", "THAT", "THESE", "THOSE",
    "WHAT", "WHICH", "WHO", "WHOM", "WHOSE",
    "HUH", "UM", "UH", "ER", "AH", "OH", "OW",
}


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
    if token.upper() in _NEVER_PRESERVE:
        return False
    if not any(c.isalpha() for c in token):
        return False
    return True


def _clean_for_ner(text: str) -> str:
    """Remove noise that confuses NER: HTML tags, parenthesized stage
    directions / sound effects, and ALL-CAPS speaker labels. Replace each
    with a single space so surrounding tokens stay separated."""
    text = _HTML_TAG_RE.sub(" ", text)
    text = _SPEAKER_LABEL_RE.sub(" ", text)
    text = _PARENS_RE.sub(" ", text)
    return text


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
    full_text = _clean_for_ner(full_text)
    doc = nlp(full_text)

    found: Set[str] = set()

    # Layer 1a — NER: keep multi-word entity surface forms verbatim.
    # Entity-labelled tokens are trusted; no frequency check applied so
    # genuine names with high zipf (e.g. Mary=4.78) still preserve.
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
        # If every token in the (post-article) span is in the never-preserve
        # set, drop the whole entity (handles `US, we should talk.` where
        # spaCy mis-tags `US` as GPE=United States).
        if all(t.text.upper() in _NEVER_PRESERVE for t in tokens):
            continue
        span = " ".join(t.text for t in tokens).strip()
        if _is_protectable(span):
            found.add(span)
            entity_token_ids.update(t.i for t in tokens)

    # Layer 1b + 2 — per-token: catch PROPN-tagged tokens NER missed
    # (filtered by frequency to skip sentence-start common nouns) plus
    # any low-frequency tokens treated as non-naturalized.
    for token in doc:
        if token.i in entity_token_ids:
            continue
        if token.is_space or token.is_punct or token.like_num:
            continue
        word = _PUNCT_RE.sub("", token.text)
        if not _is_protectable(word):
            continue

        freq = zipf_frequency(word.lower(), "en")
        is_propn_likely_name = (
            token.pos_ == "PROPN" and freq < PROPN_NOENTITY_MAX_ZIPF
        )
        is_rare = freq < zipf_threshold
        if is_propn_likely_name or is_rare:
            found.add(word)

    return sorted(found, key=len, reverse=True)
