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

---

## Out of scope (intentionally deferred)

- Per-token transliteration via IndicXlit — leave preserved text as Roman script by default.
- Per-domain glossaries (technical jargon, fan-fiction terms) — manual glossary still works for these.
- Confidence scoring / "uncertain" preserve hints in the GUI.
