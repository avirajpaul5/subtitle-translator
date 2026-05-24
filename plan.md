# Auto-Preservation Framework — Implementation Plan

Goal: scalable, zero-per-movie-config preservation of proper nouns and non-naturalized foreign words during English → Indic subtitle translation. Three detection layers (NER, corpus frequency, phrase-level langID) feed the existing `<dnt>` mechanism, which must first be made to actually survive the model.

## Architectural decision

**Path A — make `<dnt>` survive the model.** Wire `IndicTransToolkit.IndicProcessor` (or equivalent) into the IndicTrans2 translator so the existing `protect_terms` / `restore_terms` in `subtitle_translator/glossary.py` start working end-to-end. Detection layers then just append spans to the existing `do_not_translate` list.

Falling back to Path B (post-hoc target-side substitution) only if Path A's spike in Phase 1 proves infeasible.

---

## Phase 1 — preservation backend spike *(done)*

De-risked the whole framework before building on top.

- [x] Built `scripts/spike_dnt.py` exercising five protection strategies against the local IndicTrans2 model.
- [x] Tested across Hindi, Bengali, Tamil with proper nouns and a French phrase.
- [x] Verified script conversion is applied so spike output matches GUI behaviour.

**Findings (Roman-form survival across 3 targets × 7 terms):**

| Strategy | Survival |
|---|---|
| RAW (no protection) | 0% (transliterated) |
| `<dnt>...</dnt>` | 52% — brackets corrupted (`Â/dntÂ`, `w/dntw`) |
| `<IDn>` | 90% — brackets stripped |
| **`ZZIDnZZ`** (bracket-free, all-alpha) | **100%** |
| `QQ999nQQ` (alnum w/ digits) | 48% — "QQ" verbalized |

**Decision:** use bracket-free `ZZIDnZZ` sentinels. The existing `<dnt>`-based `protect_terms` / `restore_terms` in `subtitle_translator/glossary.py` need to be rewritten around this format. `IndicTransToolkit` is *not* needed — its `IndicProcessor` only auto-wraps URLs/emails/numerals and doesn't protect arbitrary terms.

**Side observation:** RAW transliteration is actually quite serviceable — IndicTrans2 turns generic proper nouns into reasonable target-script forms unaided. For users who don't care about Roman preservation specifically, doing nothing is a viable fallback.

## Phase 2 — wire preservation into the pipeline *(done)*

Made the existing `do_not_translate` list actually take effect at translation time, using the `ZZIDnZZ` placeholder scheme proven in Phase 1.

- [x] Rewrote `protect_terms` in `subtitle_translator/glossary.py` to substitute `ZZIDnZZ` sentinels and return a sentinel → original map.
- [x] Rewrote `restore_terms` to substitute back, with a tolerant matcher (`Z[ZS]I[DT](\d+)Z*`) for the D↔T letter swap the model occasionally introduces.
- [x] Wired `protect_terms` and `restore_terms` into `subtitle_translator/pipeline.py` around `translator.translate_batch`.
- [x] Replaced the stale "no pre-translation token/tag wrapping" comment.
- [x] All 10 existing tests still pass.

**End-to-end smoke** (`scripts/smoke_pipeline_dnt.py`): 20/21 protected terms (95%) survive across Hindi/Bengali/Tamil. The single miss was the Tamil model eliding part of cue 4 entirely — no leaked sentinel, so it's a model beam-search issue, not a placeholder issue.

## Phase 3 — auto-detection module (NER + frequency) *(done)*

- [x] Added `subtitle_translator/auto_dnt.py` with `detect_preserve_spans(document) -> list[str]`.
- [x] Layer 1a — spaCy NER for `PERSON`, `GPE`, `LOC`, `ORG`, `WORK_OF_ART`, `FAC`, `PRODUCT`, `NORP`, `EVENT` (multi-word spans).
- [x] **Layer 1b (added during build)** — spaCy POS=`PROPN` catches single-token proper nouns NER misses (e.g., rare given names, less-common place names). Found necessary in sanity testing where NER missed "Carol" and "Yokohama" but POS got both.
- [x] Layer 2 — `wordfreq.zipf_frequency` check. Threshold raised to 3.0 (from 2.0) after calibration: common English words like `said`, `station` sit at 5–6; foreign loans like `bonjour` (2.76), `oeufs` (1.23) sit well below.
- [x] Reuses `_SKIP_WORDS` from `subtitle_translator/speaker_detection.py`.
- [x] Lazy spaCy model load — first call only.
- [x] Deduplicated, sorted longest-first.
- [x] Layer 3 (fastText) deferred — not needed; single-token foreign words are caught by Layer 2, and multi-word non-naturalized phrases would still need a separate test pass.

**Sanity check** against a synthetic fixture caught Alice/Bob/Berlin/Tokyo/Carol/Dave/Yokohama/OpenAI/Smith/New York City/bonjour/oeufs; correctly excluded `the`, `cat`, `said`, `station`, `whispered`, `madame`, `Mr`, `visited`.

## Phase 4 — calibration test *(done)*

- [x] Added `tests/test_auto_dnt.py` with 9 property-based tests using inline synthetic fixtures. All assertions are about detection *behaviour*, not specific named characters from any film.
- [x] Coverage: proper-noun detection, foreign-word detection, common-word exclusion, skip-word exclusion, multi-word phrase preservation, length-descending ordering, determinism, threshold calibration.
- [x] Sanity pass against `tests/fixtures/sample.srt` asserts only structural invariants (non-empty output, all tokens protectable, none in skip-words) — never specific terms.
- [x] Tuned zipf threshold to **3.0**: separates English (`said`=6.0, `station`=5.0) from non-naturalized loans (`bonjour`=2.76, `oeufs`=1.23). Borderline dictionary-entered loans like `monsieur` (3.39) intentionally fall on the *translate* side, matching the user's framework: in-dictionary → naturalized → translate.
- [x] Layer 3 (fastText) confirmed unnecessary — Layers 1a + 1b + 2 cover all observed cases.

**All 19 tests pass.**

## Phase 5 — GUI integration *(done)*

- [x] In `gui.py:_on_open`, added a second-pass auto-detection (`detect_preserve_spans`) alongside the existing `detect_speaker_names`. Errors are caught and surfaced in the status bar without breaking file-open.
- [x] Both detection sources merge into `do_not_translate` via the (now generalised) `_merge_detected_names`, which also reports a breakdown in the status bar (`N speaker label(s) + M name/foreign-word(s)`).
- [x] Cleaned two auto-detection bugs uncovered by smoke against the sample SRT:
  - Subtitle formatting tags (`<i>`, `<b>`, etc.) leaked into entities (`italic</i`); now stripped before NER.
  - Multi-word entity spans included leading articles (`the Bay of Bengal`); now trimmed.
- [x] User can still edit the glossary text area before translating — existing UX intact.
- [x] `requirements.txt` updated with `spacy>=3.8` and `wordfreq>=3.1`.
- [x] `scripts/download_models.py` now also installs `en_core_web_sm` so users get a one-command setup.
- [x] `scripts/smoke_gui_autodetect.py` confirms the end-to-end flow works against the in-repo sample SRT.

**Final state:** 19 tests pass. Dropping any English SRT into the GUI auto-populates a sensible `do_not_translate` list (proper names + non-naturalized foreign words) with zero per-movie config; the existing pipeline preserves those terms verbatim through translation via the `ZZIDnZZ` sentinel mechanism proven in Phase 1.

## Phase 6 — production-evaluation fixes *(done)*

Triage of a real Bengali translation surfaced corruption modes that smoke tests had missed:

- **Sentinel debris leaking into output:** `BELLID114ZTOLLING`, `MENID209ZZ`, `USID134ZZHuh`, `DetectiveID28ZZ`, `Mary DebenhamID155ZZ`, plus Bengali letter-spell-out `( জেড. জেড. আই. ডি. 357জেড.` (~5–10% of cues).
- **Auto-detector over-preservation:** stage-direction nouns (`BELL`, `HORN`, `CAMERA`, `MEN`, `CROWD`) treated as proper nouns. Common nouns capitalized at sentence start (`Detective`, `Doctor`, `Priest`, `Huh`) flagged as PROPN. Pronoun `US` mis-tagged as `GPE` (United States).

**Root causes (two separate bugs):**

1. **Auto-detector ran on raw cue text** including parenthesized stage directions, speaker labels, and HTML tags. spaCy POS-tags `(BELL TOLLING)` content as PROPN because of the caps.
2. **`<dnt>` restoration regex** only matched the canonical `Z[ZS]I[DT]\d+Z*` and missed real-world corruption: prefix-fused (`MENID3ZZ`), suffix-fused (`ID0ZZHuh`), eaten-Z (`ID114Z`), and Indic letter-spell-out (`জেড. আই. ডি.`).

**Fixes shipped:**

- **`subtitle_translator/auto_dnt.py`**:
  - `_clean_for_ner` strips parenthesized content, bracketed `[…]` stage directions, ALL-CAPS speaker labels, and HTML tags before feeding text to spaCy.
  - `_NEVER_PRESERVE` blocklist for pronouns/interjections (`US`, `HE`, `SHE`, `IT`, `WE`, `HUH`, `UM`, …) prevents preserving things spaCy mis-tags as entities.
  - `PROPN_NOENTITY_MAX_ZIPF = 4.0` — PROPN-tagged tokens without an NER entity label require low corpus frequency. Catches `Detective`/`Doctor`/`Priest`/`Huh` at sentence start while still keeping NER-missed names like `Carol`.
  - All-tokens-blocklisted entities (e.g., spaCy tagging `US` as GPE) are dropped wholesale.
- **`subtitle_translator/glossary.py`**:
  - `protect_terms` now pads sentinels with spaces (` ZZID0ZZ `), which empirically prevents the fusion-with-neighbour failures.
  - `restore_terms` rewritten with three passes: Bengali letter-spell-out, Hindi letter-spell-out, then Latin sentinel-with-context. Requires leading or trailing `Z` (so it never matches stray English like `kid3` or `Sid8`). Multi-word saved terms with last-word-prefix fusion (`Mary DebenhamID0ZZ`) emit only the last word to avoid duplicating.
  - Orphan-sentinel sweep nukes any debris for sentinel indices we never assigned.
- **17 new regression tests** in `tests/test_auto_dnt.py` and `tests/test_pipeline.py` covering every corruption mode and every false-positive class observed.

**End-to-end verification** (`scripts/smoke_pipeline_dnt.py`, real IndicTrans2):
- 21/21 protected terms survive across Hindi, Bengali, Tamil.
- Zero `ZZID*` / `জেড` / `जेड` debris in output.
- Zero leaks of `BELL` / `MEN` / `HORN` / `US` / `Detective` / `Huh` into the auto-preserve list.

**Test suite:** 36 tests pass (was 19).

## Phase 7 — universal quality framework (defaults + validation) *(done)*

Implements the scalable per-film quality layer described in `CLAUDE.md`:

1. **Per-target-language glossary defaults** in `subtitle_translator/defaults.py`:
   - `PER_LANG_GLOSSARY["bn"]` covers the common-English-words-the-model-doesn't-translate problem (professions, nationalities, language names, everyday nouns).
   - `UNIVERSAL_DNT` covers foreign-language phrases that should always be preserved across every target (`Monsieur`, `Señor`, `Herr`, `Habibi`, `Tovarishch`, etc.).
   - `merge_with_defaults()` unions both with the user-supplied glossary at translation time; user entries win on collision.
   - Per-language isolation: Bengali defaults never bleed into a Hindi/Tamil translation request.

2. **Post-translation validation** in `subtitle_translator/validation.py`:
   - Corruption patterns per target language (Latin sentinel debris universally; `জেড` / `जेड` letter-spell-out for Bengali/Hindi/Marathi/Nepali).
   - Grammar pattern detection (universal: repeated-word, English-mid-Indic; Bengali: `আমি একটি X ছিল` subject/verb mismatch).
   - `validate_translation()` returns `ValidationIssue` records — one per dirty cue, never auto-corrects.
   - Wired into the bottom of `translate_document()`; issues append to `SubtitleDocument.warnings`.
   - Carefully avoids Python's `\b` and `\w` on Indic text (combining marks like Bengali `ি` have category `Mc`, which breaks `\w`).

3. **GUI surfaces validation warnings** — `gui.TranslateWorker.finished` now emits `(text, warnings)`; the status bar shows the flag count with full list on hover.

4. **24 new tests** covering: defaults merge precedence, per-language isolation, universal DNT protection through the pipeline, corruption detection (all language variants), grammar flagging, and the validation-report payload shape.

**End-to-end verification** (`scripts/smoke_pipeline_dnt.py`):
- 21/21 preservation across Hindi/Bengali/Tamil
- Per-language defaults observably active: Bengali output shows `গোয়েন্দা Smith` (Detective translated, name preserved); Hindi/Tamil keep `Detective Smith` since their default maps aren't populated yet
- Zero `ZZID*` / `জেড` / `जेड` debris

**Test suite:** 60 tests pass (was 36).

## Phase 8 — V3-regression triage *(done)*

User triaged a translated Bengali SRT against V1/V2/V3 and surfaced:

**V3 regressions (introduced by Phase 7 default glossary):**
- `CHIEF INSPECTOR:` → `CHIEF পরিদর্শক:` — Bengali default `inspector → পরিদর্শক` was applied word-level to the speaker label.
- `Monsieur 210ZZ` — new sentinel debris mode: model strips `ZZID` prefix entirely, leaving just `<num>ZZ`. Restoration regex required `[Ii][Dd]` to fire.

**Stubborn issues across all 3 versions (model-level, can flag but not fix):**
- `চিোকার` — two consecutive Bengali vowel signs on one consonant (invalid Unicode).
- `( আইএন জেডজেড187জেডজেড )` — Bengali letter spell-out variant not covered by existing regexes.
- `আমি একটি ভাল সপ্তাহ ছিল` — subject/verb mismatch grammar bug; existing pattern was too narrow (only allowed one intervening word).
- `বাইউক`/`বিউইক` — character name "Bouc" transliterated. *Not fixable without per-movie glossary or different model* (no regex can know it's a name).

**Fixes shipped:**

- **`subtitle_translator/pipeline.py`**: extracted `_translate_speaker_label()`. Speaker labels now match the glossary as a *full key only* (case-insensitive), not via word-level substitution. User can still map `"POIROT" → "পয়রট"`, but `"CHIEF INSPECTOR"` no longer gets word-translated by the Bengali default `inspector → পরিদর্শক`.

- **`subtitle_translator/glossary.py`**: new restoration pass `_SENTINEL_ID_STRIPPED_RE = r"(?<![A-Za-z\d])(\d{1,4})Z{2,}(?![A-Za-z\d])"`. Recognises `<num>ZZ` debris where the model dropped `ZZID` entirely; looks up the index in the replacements map and substitutes back. Orphan sweep extended to wipe leftovers.

- **`subtitle_translator/validation.py`**:
  - Universal corruption now includes `(?<![A-Za-z\d])\d{1,4}Z{2,}(?![A-Za-z\d])` (catches `210ZZ`).
  - Bengali corruption adds `আইএন\s*জেড` (the `IN ZZ…` letter-spell-out from line 31) and `[া-ৌ]{2,}` (consecutive Bengali vowel signs — flags `চিোকার`).
  - Grammar pattern loosened to allow 1–3 Bengali words between `একটি` and `ছিল`, so `আমি একটি ভাল সপ্তাহ ছিল` fires.

- **6 new regression tests**, one per evaluation row.

**End-to-end smoke**: 21/21 preservation across Hindi/Bengali/Tamil, zero debris, no new regressions.

**Test suite:** 66 tests pass (was 60).

### Honest accounting of what *cannot* be fixed

The user asked "is there a way to fix these issues reliably all the time irrespective of which movie is being translated?" Three of the eval's stubborn issues are genuinely model-level — no post-processing can fix them reliably:

| Issue | Why unfixable | What this codebase does |
|---|---|---|
| Character name transliteration (`বাইউক`) | No regex can know `Bouc` is a name vs. a rare foreign word vs. a misspelling. The auto-detector tries (spaCy NER + PROPN + wordfreq), but recall is <100% on rare/period-piece names. | Auto-detection runs at file open; user reviews the suggested DNT list and adds anything missed. |
| Wrong Unicode characters in model output (`চিোকার`) | Post-processing cannot guess what the intended consonant cluster was. | Validation flags it; human review required. |
| Persistent grammar errors | No automated rewriter can produce a grammatical sentence with the original semantics. | Validation flags; human review. |

The framework's *real* contribution is making the per-movie work shrink to: review the auto-detected DNT list at file-open, then review the validation warnings after translation. Everything else is universal and scales automatically.

---

## Out of scope (intentionally deferred)

- Per-token transliteration via IndicXlit — leave preserved text as Roman script by default.
- Per-domain glossaries (technical jargon, fan-fiction terms) — manual glossary still works for these.
- Confidence scoring / "uncertain" preserve hints in the GUI.
