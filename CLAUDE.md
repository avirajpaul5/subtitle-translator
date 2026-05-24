# Subtitle Translator — Project Guide for Claude Code

## What This Project Does

Translates English `.srt` / `.vtt` subtitle files to Bengali using **IndicTrans 1B** (local, offline).
The pipeline lives in `subtitle_translator/` and is exposed via two frontends: `app.py` (Streamlit) and `gui.py` (PySide6).

---

## Project Structure

```
.
├── app.py                          # Streamlit frontend
├── gui.py                          # PySide6 native frontend
├── subtitle_translator/
│   ├── pipeline.py                 # Core translation orchestrator — touch this most often
│   ├── translators/
│   │   ├── indictrans2.py          # IndicTrans 1B wrapper — token corruption issues live here
│   │   ├── nllb.py                 # Alternative backend
│   │   ├── echo.py                 # Test/echo mode
│   │   └── factory.py              # Backend selector
│   ├── glossary.py                 # protect_terms(), restore_terms(), apply_glossary_overrides()
│   ├── parsers.py                  # SRT/VTT parse + serialize
│   ├── segmentation.py             # Cue merging + re-splitting
│   ├── formatter.py                # Line wrapping
│   └── defaults.py                 # Default glossary
├── examples/
│   └── sample_glossary.json        # Glossary schema reference
├── scripts/
│   └── download_models.py          # Model downloader
└── tests/                          # pytest suite (uses echo backend, no model required)
```

---

## Known Issues & What to Do About Them

### 1. Token Corruption — CHECK FIRST on any translation output

**What it looks like:**
- `ID\d+ZZ` mixed into Bengali text (e.g. `Detective23ZZ`)
- `জেডজেড` repeated sequences (Bengali fallback for token marker `Z`)
- Numbers floating mid-sentence (`Au revoir 332 capitaine`)
- Garbage transliterations (`বিউইক`, `বাইউক`) at end of lines

**Where to look:** `subtitle_translator/translators/indictrans2.py` → `translate_batch()`

**Root cause:** IndicTrans internal token IDs leaking through detokenization. The tokenizer and the model may be mismatched, or `skip_special_tokens=True` is not being respected.

**How to detect programmatically:**
```python
import re

CORRUPTION_PATTERNS = [
    r'ID\d+ZZ',                      # Primary artifact
    r'জেড{2,}',       # জেড repeated (Bengali Z fallback)
    r'\d{2,}জেড',                    # Number + জেড
    r'[A-Za-z]+\d{2,}[A-Z]{2}',     # WordNNZZ pattern
]

def has_corruption(text: str) -> bool:
    return any(re.search(p, text) for p in CORRUPTION_PATTERNS)
```

**Add this check inside `translate_batch()` before returning.** Flag or log corrupted lines rather than silently passing them through.

---

### 2. Untranslated English Words — ADD TO GLOSSARY

These English words consistently slip through IndicTrans untranslated. They are **universal across all films** — not specific to any one movie.

Expand `examples/sample_glossary.json` (and `subtitle_translator/defaults.py`) with:

```json
{
  "glossary": {
    "language_names": {
      "english": "ইংরেজি",
      "french": "ফরাসি",
      "german": "জার্মান",
      "spanish": "স্পেনিশ",
      "italian": "ইতালিয়ান",
      "russian": "রুশ",
      "arabic": "আরবি",
      "hindi": "হিন্দি",
      "chinese": "চীনা",
      "japanese": "জাপানি"
    },
    "nationalities": {
      "american": "আমেরিকান",
      "americans": "আমেরিকানরা",
      "british": "ব্রিটিশ",
      "french": "ফরাসি",
      "german": "জার্মান",
      "russian": "রুশ",
      "italian": "ইতালিয়ান",
      "spanish": "স্পেনিশ",
      "indian": "ভারতীয়",
      "chinese": "চীনা"
    },
    "professions": {
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
      "secretary": "সচিব"
    },
    "common_nouns": {
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
      "passport": "পাসপোর্ট"
    },
    "common_verbs_in_dialogue": {
      "said": "বলেছেন",
      "asked": "জিজ্ঞাসা করেছেন",
      "replied": "উত্তর দিয়েছেন",
      "continued": "অব্যাহত রাখলেন",
      "whispered": "ফিসফিস করলেন",
      "shouted": "চিৎকার করলেন"
    }
  },
  "do_not_translate": [
    "Monsieur", "Mademoiselle", "Madame",
    "mon ami", "Allons-y", "Au revoir", "Bonsoir", "merci",
    "Señor", "Señorita", "Señora", "gracias",
    "Ja", "Nein", "Herr", "Danke",
    "Ciao", "Bella", "Amore", "prego",
    "Habibi", "Walah", "Inshallah",
    "Da", "Nyet", "Tovarishch"
  ]
}
```

**Important:** The glossary is applied in `pipeline.py` via `apply_glossary_overrides()`. It runs AFTER translation as a find-and-replace. Words must match at word boundaries (the existing regex handles this).

---

### 3. Grammar Issues — PATTERN-BASED DETECTION

These patterns appear across all films. Add a post-processing validation step in `pipeline.py`:

```python
GRAMMAR_FLAGS = [
    # Subject-verb mismatch: "আমি [thing] ছিল" (I was a [thing])
    (r'আমি\s+একটি\s+\w+\s+ছিল', "subject_verb_mismatch"),

    # Repeated word (e.g. "হর্ন হর্নিং" = horn horning)
    (r'(\b\S+\b)\s+\1\w*', "repeated_word"),

    # English word immediately followed by Bengali (mixed line)
    (r'[A-Za-z]{3,}\s+[ঀ-৿]', "likely_untranslated_word"),

    # Hanging open parenthesis (incomplete stage direction)
    (r'\([^)]*$', "unclosed_parenthesis"),
]

def flag_grammar_issues(text: str) -> list[str]:
    flags = []
    for pattern, label in GRAMMAR_FLAGS:
        if re.search(pattern, text):
            flags.append(label)
    return flags
```

Flag lines with issues for human review. Do NOT auto-correct grammar — just surface them.

---

## Translation Quality Rules

These rules define what "correct" output looks like. Apply them when reviewing or validating translated output.

### PRESERVE (Do Not Translate)

| Category | Rule | Examples |
|----------|------|---------|
| Character names | Always preserve as-is | `Poirot`, `John`, `Mary` |
| Place names | Preserve or use Bengali transliteration | `London`, `Paris`, `Istanbul` |
| Foreign language phrases | Preserve — they are intentional | `mon ami`, `Allons-y`, `Señor` |
| Foreign titles | Preserve | `Mademoiselle`, `Monsieur`, `Señor` |
| ALL CAPS sound effects | Preserve | `(BELL TOLLING)`, `(SQUELCH)` |
| Brand names | Always preserve | `Rolex`, `Rolls Royce` |
| Acronyms | Preserve | `FBI`, `CIA`, `UNESCO` |

**Why foreign phrases are preserved:** In English films with multilingual characters, untranslated foreign phrases are intentional — they signal character background. Bengali-speaking audiences watching an English film do not expect those phrases translated.

**Why ALL CAPS stage directions are preserved:** Standard convention. Bengali-speaking audiences recognise `(BELL TOLLING)` as a sound marker without needing translation.

### TRANSLATE (Must Become Bengali)

| Category | Examples | Reason |
|----------|---------|--------|
| Common professions | `doctor`, `waiter`, `teacher` | Audience expects native word |
| Religious figures | `priest`, `rabbi`, `imam` | Bengali equivalents exist |
| Everyday objects | `table`, `chair`, `door`, `luggage` | No reason to leave English |
| Language names | `english`, `french`, `german` | Bengali equivalents exist |
| Nationalities | `american`, `british`, `german` | Bengali equivalents exist |
| Common verbs in dialogue | `said`, `asked`, `replied` | Core to meaning |
| Descriptors | `good`, `bad`, `quick` | Core to meaning |

### ACCEPTABLE AS-IS (Do Not Flag)

| Pattern | Why Acceptable |
|---------|---------------|
| Hesitation markers (`um`, `uh`, `er`) | Represent speech patterns, not vocabulary |
| Sentence fragments | Reflect natural dialogue rhythm |
| Ellipsis mid-sentence | Reflects character hesitation |
| Informal registers mixed with formal Bengali | Reflects character personality |
| Gist-correct but not word-for-word | As long as meaning carries over, it is a successful translation |

---

## Validation Pipeline (Run on Every Output File)

Run these checks in order after every translation job:

```
1. Token corruption scan    → Flag lines matching CORRUPTION_PATTERNS
2. Glossary coverage check  → Flag English words in must_translate list
3. Grammar pattern check    → Flag lines matching GRAMMAR_FLAGS
4. Parenthesis balance      → Flag lines with unclosed ( or )
5. Output flagged lines     → Human reviews ONLY flagged lines
```

Implementing this in `pipeline.py` at the end of `translate_document()` is the right place:

```python
# At the end of translate_document(), after translating all cues:
validation_report = []
for i, cue in enumerate(translated_cues):
    issues = []
    if has_corruption(cue.text):
        issues.append("token_corruption")
    issues += flag_grammar_issues(cue.text)
    if issues:
        validation_report.append({
            "cue_index": i,
            "original": document.cues[i].text,
            "translated": cue.text,
            "issues": issues,
        })

# Log or return validation_report alongside the translated document
```

---

## What NOT to Auto-Fix

Do not attempt to auto-correct the following — surface them for human review instead:

- Grammar errors (word order, conjugation) — context-dependent
- Ambiguous English phrases — meaning unclear without watching the scene
- Partially translated lines — may need full retranslation
- Stage directions that are narrative descriptions (not ALL CAPS) — need judgment call
- Any line where corruption removed words entirely — original meaning may be unrecoverable without re-running the model

---

## Test Suite

Tests are in `tests/`. They use the `echo` backend — no model weights needed.

```bash
pytest -q
```

When adding a validation feature, add a test in `tests/test_pipeline.py` using `EchoTranslator`.

**Important:** Do not add tests that require model weights. Keep the test suite model-free.

---

## Glossary File Schema

```json
{
  "glossary": {
    "English term": "বাংলা টার্ম"
  },
  "do_not_translate": ["TermToPreserve"]
}
```

- `glossary` entries are applied as **post-translation find-and-replace** (case-insensitive, word boundary matched)
- `do_not_translate` entries are **protected before translation** using placeholder tokens and restored after
- Both lists are additive — the default glossary in `defaults.py` is always merged with any user-supplied glossary

---

## Common Mistakes to Avoid

- **Do not translate proper nouns** — names, places, brands stay as-is even if IndicTrans tries to transliterate them
- **Do not translate ALL CAPS sound markers** — `(BELL TOLLING)` is correct, even without Bengali translation
- **Do not flag foreign phrases as errors** — `mon ami`, `Allons-y` etc. are intentionally preserved
- **Do not assume token corruption means bad translation** — the underlying semantic translation may be correct; only the encoding failed
- **Do not merge cue timing** — the parsed timing must be preserved exactly; only `.text_lines` should change
- **Do not run model inference in tests** — always use `EchoTranslator` for testing pipeline logic

---

## Performance Expectations (IndicTrans 1B on CPU)

| File Size | Estimated Time |
|-----------|---------------|
| Short clip (< 200 cues) | 2–5 min |
| Feature film (800–1200 cues) | 20–40 min |
| Long film (1500+ cues) | 45–90 min |

GPU cuts this by ~10x. Model loading itself takes ~30–60 seconds on first run.

---

## Decisions Already Made (Do Not Revisit Without Good Reason)

| Decision | Reason |
|----------|--------|
| `local_files_only=True` in model loading | Enforce offline-only inference; no accidental cloud calls |
| Glossary applied POST-translation | Pre-translation injection confuses the model's context window |
| `do_not_translate` terms protected PRE-translation | Prevents model from mangling protected terms |
| Cues merged before translation, re-split after | Gives the model more context per inference call; improves quality |
| Echo backend for all tests | Keeps test suite fast and model-weight-free |
| Bengali punctuation: `।` not `.` | Correct Bengali sentence-ending punctuation |
