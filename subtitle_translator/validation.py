"""Post-translation validation. Flags suspect lines for human review.

Two classes of checks, both per-target-language:

* **Corruption** — sentinel debris (`ZZIDnZZ` leftovers, Bengali letter-
  spell-out `জেড. আই. ডি.`, etc.) that escaped `restore_terms`. These are
  data-integrity bugs.
* **Grammar** — language-specific patterns that usually indicate a poor
  translation (subject/verb mismatch, repeated word, mid-line English/Indic
  mixing, unclosed parens).

Neither check auto-corrects: corruption indicates a pipeline bug to fix at
the source, and grammar issues are context-dependent. Output is a list of
`ValidationIssue` records the caller can surface in the UI, log, or write
into `SubtitleDocument.warnings`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Pattern, Sequence, Tuple

# ---------------------------------------------------------------------------
# Corruption patterns
# ---------------------------------------------------------------------------

# Patterns that indicate sentinel debris in ANY target language.
# Three Latin variants catch every observed leak mode:
#   - prefix-fused with the model's re-injected source: `MENID4ZZ`
#   - clean canonical form: `ZZID3ZZ`
#   - ID-prefix-stripped, just number + Z's: `Monsieur 210ZZ`
# Requires AT LEAST one Z somewhere — keeps the pattern from false-firing
# on real English like `kid3` or `Sid8`. No `\b` anchors (Python's `\b`
# breaks on Indic combining marks).
_UNIVERSAL_CORRUPTION: List[Tuple[Pattern[str], str]] = [
    (re.compile(r"[Zz]+[A-Za-z]{0,12}?[Ii][Dd]\d+[Zz]*"),         "sentinel_debris"),
    (re.compile(r"[A-Za-z]{0,12}?[Ii][Dd]\d+[Zz]+"),               "sentinel_debris"),
    (re.compile(r"(?<![A-Za-z\d])\d{1,4}Z{2,}(?![A-Za-z\d])"),     "sentinel_debris"),
    (re.compile(r"\([^)]*$"),                                      "unclosed_parenthesis"),
]

# Bengali-specific sentinel transliteration (জেড = Z, আই = I, ডি = D).
# Bengali combining marks (Mc category) confuse \b — match without anchors.
_BENGALI_CORRUPTION: List[Tuple[Pattern[str], str]] = [
    (re.compile(r"জেড\s*\.?\s*জেড"),         "bengali_sentinel_spellout"),
    (re.compile(r"আই\s*\.?\s*ডি"),           "bengali_sentinel_spellout"),
    (re.compile(r"\d{1,4}\s*জেড"),           "bengali_sentinel_spellout"),
    (re.compile(r"আইএন\s*জেড"),              "bengali_sentinel_spellout"),
    # Two consecutive Bengali dependent vowel signs on the same consonant.
    # Always invalid in Bengali (e.g. `চিোকার` — ি + ো on চ). Indicates
    # a model decoder bug; we can flag but not fix.
    (re.compile(r"[া-ৌ]{2,}"),     "invalid_bengali_vowel_cluster"),
]

# Hindi-specific sentinel transliteration (जेड = Z, आई = I, डी = D).
_HINDI_CORRUPTION: List[Tuple[Pattern[str], str]] = [
    (re.compile(r"जेड\s*\.?\s*जेड"),         "hindi_sentinel_spellout"),
    (re.compile(r"आई\s*\.?\s*डी"),           "hindi_sentinel_spellout"),
    (re.compile(r"\d{1,4}\s*जेड"),           "hindi_sentinel_spellout"),
]

CORRUPTION_BY_LANG: Dict[str, List[Tuple[Pattern[str], str]]] = {
    "bn": _UNIVERSAL_CORRUPTION + _BENGALI_CORRUPTION,
    "hi": _UNIVERSAL_CORRUPTION + _HINDI_CORRUPTION,
    "mr": _UNIVERSAL_CORRUPTION + _HINDI_CORRUPTION,  # Marathi uses Devanagari
    "ne": _UNIVERSAL_CORRUPTION + _HINDI_CORRUPTION,  # Nepali uses Devanagari
}


def _corruption_patterns(target_lang: str) -> List[Tuple[Pattern[str], str]]:
    return CORRUPTION_BY_LANG.get(target_lang, _UNIVERSAL_CORRUPTION)


def has_corruption(text: str, target_lang: str = "") -> bool:
    """True if `text` matches any sentinel-debris / unclosed-paren pattern."""
    return bool(_corruption_labels(text, target_lang))


def _corruption_labels(text: str, target_lang: str) -> List[str]:
    labels: List[str] = []
    for pattern, label in _corruption_patterns(target_lang):
        if pattern.search(text) and label not in labels:
            labels.append(label)
    return labels


# ---------------------------------------------------------------------------
# Grammar patterns
# ---------------------------------------------------------------------------

# A grammar flag is purely heuristic — used to highlight lines for human
# review, never to auto-correct. Patterns are intentionally narrow to avoid
# false-positive fatigue.

# Unicode ranges for common Indic scripts the model emits.
_INDIC_SCRIPT_RANGE = (
    r"ऀ-ॿ"   # Devanagari (Hindi, Marathi, Nepali, Sanskrit)
    r"ঀ-৿"   # Bengali / Assamese
    r"਀-੿"   # Gurmukhi (Punjabi)
    r"઀-૿"   # Gujarati
    r"଀-୿"   # Odia
    r"஀-௿"   # Tamil
    r"ఀ-౿"   # Telugu
    r"ಀ-೿"   # Kannada
    r"ഀ-ൿ"   # Malayalam
)

# Patterns that apply for any target language (script-agnostic).
_UNIVERSAL_GRAMMAR: List[Tuple[Pattern[str], str]] = [
    # English word doubled at a word boundary ("horn horn", "the the").
    # Latin-only — Indic combining marks break Python's \b/\w semantics.
    (re.compile(r"\b([A-Za-z]{3,})\s+\1\b", re.IGNORECASE),     "repeated_word"),
    # Three-letter-or-more English word right before Indic script (likely
    # an untranslated English token mid-sentence).
    (re.compile(rf"\b[A-Za-z]{{3,}}\s+[{_INDIC_SCRIPT_RANGE}]"), "likely_untranslated_word"),
]

# Bengali word token: any letter (Lo) plus combining marks (Mc, Mn).
_BENGALI_WORD = r"[ঀ-৿]+"

_BENGALI_GRAMMAR: List[Tuple[Pattern[str], str]] = [
    # "আমি একটি X[ Y]* ছিল" — first-person subject with 3rd-person past tense
    # verb (should be ছিলাম, not ছিল). Allow 1–3 intervening words between
    # "একটি" and "ছিল" so "আমি একটি ভাল সপ্তাহ ছিল" (line 1054 in eval) fires.
    (re.compile(rf"আমি\s+একটি(?:\s+{_BENGALI_WORD}){{1,3}}\s+ছিল(?:\s|[.!?।]|$)"),
        "subject_verb_mismatch"),
]

GRAMMAR_BY_LANG: Dict[str, List[Tuple[Pattern[str], str]]] = {
    "bn": _UNIVERSAL_GRAMMAR + _BENGALI_GRAMMAR,
}


def _grammar_patterns(target_lang: str) -> List[Tuple[Pattern[str], str]]:
    return GRAMMAR_BY_LANG.get(target_lang, _UNIVERSAL_GRAMMAR)


def flag_grammar_issues(
    text: str,
    target_lang: str = "",
    protected_terms: Iterable[str] = (),
) -> List[str]:
    # Intentional foreign phrases, titles, names, brands, and acronyms are not
    # untranslated-word failures. Hide the exact protected phrases before the
    # heuristic grammar scan so they do not create review noise.
    scan_text = text
    for term in sorted(
        {term for term in protected_terms if term},
        key=lambda value: (-len(value), value.casefold(), value),
    ):
        scan_text = re.sub(
            rf"\b{re.escape(term)}\b",
            "",
            scan_text,
            flags=re.IGNORECASE,
        )
    labels: List[str] = []
    for pattern, label in _grammar_patterns(target_lang):
        if pattern.search(scan_text) and label not in labels:
            labels.append(label)
    return labels


def flag_glossary_coverage(
    original: str,
    translated: str,
    glossary_terms: Iterable[str],
    protected_terms: Iterable[str] = (),
) -> List[str]:
    """Flag must-translate glossary terms that remain unchanged in output."""

    protected = {term.casefold() for term in protected_terms if term}
    flags: List[str] = []
    for term in sorted(
        {term for term in glossary_terms if term},
        key=lambda value: (-len(value), value.casefold(), value),
    ):
        if term.casefold() in protected:
            continue
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        if pattern.search(original) and pattern.search(translated):
            flags.append(f"glossary_term_untranslated:{term}")
    return flags


# ---------------------------------------------------------------------------
# Top-level validation entry point
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    cue_index: int           # 0-based position in the cue list
    cue_number: Optional[int]  # The cue's own .index field (1-based, may be None)
    original: str
    translated: str
    issues: List[str] = field(default_factory=list)

    def formatted(self) -> str:
        label = f"cue {self.cue_number}" if self.cue_number is not None else f"cue #{self.cue_index + 1}"
        return f"{label}: {', '.join(self.issues)}"


def validate_translation(
    original_texts: Sequence[str],
    translated_texts: Sequence[str],
    cue_numbers: Sequence[Optional[int]],
    target_lang: str = "",
    glossary_terms: Iterable[str] = (),
    protected_terms: Iterable[str] = (),
) -> List[ValidationIssue]:
    """Run every check against each cue. Returns one record per cue with at
    least one issue; cues that are clean don't appear in the list."""
    out: List[ValidationIssue] = []
    for i, (orig, trans, num) in enumerate(
        zip(original_texts, translated_texts, cue_numbers)
    ):
        labels = _corruption_labels(trans, target_lang)
        labels.extend(
            l
            for l in flag_grammar_issues(
                trans,
                target_lang,
                protected_terms=protected_terms,
            )
            if l not in labels
        )
        labels.extend(
            label
            for label in flag_glossary_coverage(
                orig,
                trans,
                glossary_terms,
                protected_terms,
            )
            if label not in labels
        )
        if labels:
            out.append(ValidationIssue(
                cue_index=i,
                cue_number=num,
                original=orig,
                translated=trans,
                issues=labels,
            ))
    return out
