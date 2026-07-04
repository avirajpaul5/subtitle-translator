"""Default glossaries and preserve-lists applied at translation time.

Two layers of defaults, both merged with whatever the user supplies via the
GUI/Streamlit glossary editor:

1. **Per-target-language glossary** (`PER_LANG_GLOSSARY`) — English words that
   the model leaves untranslated and should be replaced with target-language
   equivalents. Only the entries for the active target language are applied,
   so Bengali defaults never leak into Hindi/Tamil output.

2. **Universal `do_not_translate` list** (`UNIVERSAL_DNT`) — foreign-language
   phrases that are intentionally preserved across all films and all target
   languages (French "mon ami", Spanish "Señor", German "Herr", etc.).

The user's glossary always wins on collisions: their `{key: value}` overrides
the same `key` in the defaults, and their `do_not_translate` entries are
unioned with the universal list.
"""
from __future__ import annotations

from typing import Dict, List


# ---------------------------------------------------------------------------
# Per-target-language glossaries — English → target-script translation that
# the model consistently fails to do. Organised by lang code (ISO 639-1).
# ---------------------------------------------------------------------------

_BENGALI: Dict[str, str] = {
    # Language names
    "english": "ইংরেজি",
    "french": "ফরাসি",
    "german": "জার্মান",
    "spanish": "স্পেনিশ",
    "italian": "ইতালিয়ান",
    "russian": "রুশ",
    "arabic": "আরবি",
    "hindi": "হিন্দি",
    "chinese": "চীনা",
    "japanese": "জাপানি",
    # Nationalities
    "american": "আমেরিকান",
    "americans": "আমেরিকানরা",
    "british": "ব্রিটিশ",
    "indian": "ভারতীয়",
    # Professions / titles
    "doctor": "ডাক্তার",
    "waiter": "ওয়েটার",
    "waitress": "ওয়েট্রেস",
    "teacher": "শিক্ষক",
    "priest": "পুরোহিত",
    "rabbi": "রাব্বি",
    "imam": "ইমাম",
    "governess": "গভর্নেস",
    "cartographer": "মানচিত্রকার",
    "engineer": "প্রকৌশলী",
    "soldier": "সৈনিক",
    "detective": "গোয়েন্দা",
    "inspector": "পরিদর্শক",
    "colonel": "কর্নেল",
    "captain": "ক্যাপ্টেন",
    "sergeant": "সার্জেন্ট",
    "nurse": "নার্স",
    "lawyer": "আইনজীবী",
    "judge": "বিচারক",
    "secretary": "সচিব",
    # Everyday concrete nouns
    "luggage": "লাগেজ",
    "table": "টেবিল",
    "chair": "চেয়ার",
    "door": "দরজা",
    "window": "জানালা",
    "letter": "চিঠি",
    "money": "টাকা",
    "gun": "বন্দুক",
    "knife": "ছুরি",
    "train": "ট্রেন",
    "car": "গাড়ি",
    "phone": "ফোন",
    "hotel": "হোটেল",
    "room": "কামরা",
    "ticket": "টিকিট",
    "passport": "পাসপোর্ট",
    # Place names the user previously had
    "Dhaka": "ঢাকা",
    "Bangladesh": "বাংলাদেশ",
}

PER_LANG_GLOSSARY: Dict[str, Dict[str, str]] = {
    "bn": _BENGALI,
    # Add other languages (hi, ta, te, etc.) as their corpora are curated.
}


# ---------------------------------------------------------------------------
# Universal do-not-translate — foreign-language phrases that are intentionally
# preserved across every film. Apply for any target language.
# ---------------------------------------------------------------------------

UNIVERSAL_DNT: List[str] = [
    # French
    "Monsieur", "Mademoiselle", "Madame",
    "mon ami", "Allons-y", "Au revoir", "Bonsoir", "merci",
    # Spanish
    "Señor", "Señorita", "Señora", "gracias",
    # German
    "Ja", "Nein", "Herr", "Frau", "Danke",
    # Italian
    "Ciao", "Bella", "Amore", "prego",
    # Arabic / Urdu
    "Habibi", "Walah", "Inshallah",
    # Russian
    "Da", "Nyet", "Tovarishch",
    # Latin
    "et cetera", "etc",
]


# ---------------------------------------------------------------------------
# Default glossary blob shown in the GUI/Streamlit editor.
# Kept minimal so the user can see what they're starting from; the built-in
# per-language map and DNT list are *also* merged at translation time.
# ---------------------------------------------------------------------------

DEFAULT_GLOSSARY = {
    "glossary": {},
    "do_not_translate": [],
}


def get_default_glossary_for(target_lang: str) -> Dict[str, str]:
    """Return the per-target-language defaults; empty dict for unknown langs."""
    return dict(PER_LANG_GLOSSARY.get(target_lang, {}))


def merge_with_defaults(
    user_map: Dict[str, str],
    user_dnt: List[str],
    target_lang: str,
) -> tuple[Dict[str, str], List[str]]:
    """Combine per-language built-in glossary + universal DNT with what the
    user supplied. The user's entries override on key collision; DNT lists
    are unioned (case preserved as-given)."""
    merged_map: Dict[str, str] = {**get_default_glossary_for(target_lang), **user_map}

    seen_lower: set[str] = set()
    merged_dnt: List[str] = []
    for term in [*UNIVERSAL_DNT, *user_dnt]:
        if term.lower() not in seen_lower:
            seen_lower.add(term.lower())
            merged_dnt.append(term)
    return merged_map, merged_dnt
