# IndicSub

IndicSub is a local-first translation workspace for English subtitles and text documents. The Streamlit app accepts `.srt`, `.vtt`, `.txt`, and `.md`; the native PySide6 app remains focused on subtitles. Both translation paths group adjacent units for model context while preserving the exact cue or document structure needed to rebuild the output.

Local IndicTrans2 inference is the default. Sarvam remains an optional, explicitly selected API backend for users who provide their own key.

## Features

- Drag-and-drop upload for subtitles (`.srt`, `.vtt`) and documents (`.txt`, `.md`) in the browser workspace
- Preserves subtitle timing and cue order, or document paragraphs, blank lines, headings, lists, blockquotes, and Markdown syntax
- Keeps fenced/indented code, raw HTML blocks, link definitions, inline code, URLs, and other protected Markdown spans out of model translation
- Input and translated side-by-side previews
- Provider-aware route pickers, checkpoint-type readiness checks, and visible effective context caps before a run starts
- Durable review output with provider/fallback state, partial/completed status, and every validation warning
- Default language pair: `en → bn` (Bengali); supports all IndicTrans2 Indic targets
- **Auto speaker-name detection**: scans cues for `SPEAKER: ...` patterns and auto-adds detected names to the do-not-translate list
- **Speaker label extraction**: ALL-CAPS speaker labels (e.g. `POIROT:`) are stripped before translation and re-attached after, preventing the model from mangling them; glossary mappings are applied to labels too
- **Stage-direction preservation**: ALL-CAPS sound and stage markers such as `(BELL TOLLING)` are restored verbatim
- **Glossary protection via sentinels**: protected terms are hidden behind model-resistant placeholders during inference and restored after translation
- **Post-translation glossary overrides**: case-insensitive regex replacements applied after translation for reliable term substitution
- **Context-aware translation with exact alignment**: consecutive cues or document blocks share a bounded model input, with numbered boundary markers decoded and validated before output is accepted
- **Safe alignment fallback**: if a model drops, duplicates, or reorders a boundary marker, only that affected window is retried one unit at a time; shifted or empty output is never silently accepted
- **No silent model truncation**: local backends plan against their tokenizer limit after preprocessing; oversized document pieces are recursively split, and inputs or outputs that still hit a hard model cap are rejected instead of exported incomplete
- **Exact protected-content contract**: do-not-translate terms, links, code, HTML, and Markdown delimiters must return in the original count and order before a window can be checkpointed
- **Official IndicTrans2 processing**: local inference uses `IndicProcessor.preprocess_batch()` and `postprocess_batch()` around tokenization/generation
- Optional Sarvam API backend with password entry, environment/keychain lookup, model/mode selection, retries, request/input-size reporting, rate-limit backoff, resumable checkpoints, and opt-in local fallback
- Glossary JSON upload and in-UI editor
- Adjustable model batch size, context-character budget, and maximum units per context window
- Subtitle-friendly line wrapping and max line length controls
- ETA and progress display during translation, including the active provider/model
- Resumable checkpoints for interrupted translation jobs
- Download translated output in the same supported format as input
- Echo/test mode to validate parsing + writing pipeline without model inference
- Translator abstraction with pluggable backends (`indictrans2`, `sarvam-api`, `nllb`, `echo`)

## Project structure

```text
.
├── app.py                          # Streamlit browser GUI
├── gui.py                          # PySide6 native desktop GUI
├── requirements.txt
├── README.md
├── examples/
│   └── sample_glossary.json
├── scripts/
│   └── download_models.py          # HuggingFace model fetcher
└── subtitle_translator/
    ├── __init__.py
    ├── defaults.py                 # Default glossary entries
    ├── models.py
    ├── parsers.py
    ├── glossary.py                 # Glossary protection + override logic
    ├── contextual.py               # Exact context-window encoding + alignment fallback
    ├── document_pipeline.py        # TXT/Markdown translation orchestration
    ├── documents/
    │   ├── models.py               # Format-neutral document IR
    │   └── adapters.py             # TXT + conservative Markdown adapters
    ├── segmentation.py             # Legacy segmentation helpers
    ├── formatter.py
    ├── pipeline.py                 # Subtitle translation orchestration
    ├── speaker_detection.py        # Auto speaker-name scanner
    ├── credentials.py              # Env/keychain API key lookup
    └── translators/
        ├── __init__.py
        ├── base.py
        ├── echo.py
        ├── fallback.py
        ├── indictrans2.py
        ├── nllb.py
        ├── sarvam_api.py
        └── factory.py
```

## Setup

**Requires Python 3.10 or newer.** The macOS system Python (`/usr/bin/python3`,
currently 3.9) is too old — PyTorch no longer publishes arm64 wheels for 3.9,
so pip silently falls back to the x86_64 build and crashes at import on Apple
Silicon with `incompatible architecture (have 'x86_64', need 'arm64')`.

On macOS, install Python 3.12 from <https://www.python.org/downloads/macos/>
(the "macOS 64-bit universal2 installer"), then:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Linux / other platforms with a recent Python already on `PATH`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Two front-ends are available. Use the browser workspace for subtitles or text documents; use the native GUI for a subtitle-only desktop workflow.

Native desktop GUI (PySide6):

```bash
python gui.py
```

Browser GUI (Streamlit):

```bash
streamlit run app.py
```

Both share the same glossary and translator backends. Subtitle files use the cue-aware subtitle pipeline; `.txt` and `.md` use a document IR that keeps translatable text separate from structural and protected content.

## Tests

```bash
pytest -q
```

The test suite uses the `echo` backend, so no model weights are required.

## File encoding

Uploaded text-based files are decoded using a fallback chain: UTF-8 with BOM, UTF-8, UTF-16, cp1252, then latin-1. Common Windows-encoded subtitle and text files load without manual conversion.

## Model setup (offline)

Place a local model folder and set **Local model path** in the app sidebar.

Default expected path:

```text
./models/indictrans2-en-indic
```

A helper script fetches weights from HuggingFace into the expected location. The downloader currently defaults to AI4Bharat's distilled 200M model (`ai4bharat/indictrans2-en-indic-dist-200M`); the full 1B model is opt-in:

```bash
python scripts/download_models.py                 # IndicTrans2 distilled 200M (~800 MB)
python scripts/download_models.py --model 1B      # IndicTrans2 full 1B (~4.5 GB)
python scripts/download_models.py --model nllb    # NLLB-200 distilled 600M (~2.5 GB)
```

The `models/` directory is gitignored. The distilled model uses the default IndicTrans2 path; NLLB defaults to `./models/nllb-200-distilled-600M`. The full 1B and NLLB downloads use separate folders so model files cannot mix; copy the path printed by the helper into **Local model path**. Both interfaces inspect `config.json` before enabling translation and reject an NLLB checkpoint selected for IndicTrans2 (or the reverse). The same check is required when local IndicTrans is enabled as Sarvam's backup.

## Sarvam API setup (optional)

Choose the `sarvam-api` backend when you want to use Sarvam's hosted translation models instead of local inference.

API keys are resolved in this order:

1. The password field in the UI for the current run.
2. The `SARVAM_API_KEY` environment variable.
3. A key saved in the OS keychain.

Keys are never written to project files. To save a key, enter it in the UI and enable **Save key in OS keychain**. If the optional keychain backend is unavailable on the machine, the app will continue using the key for the current run and show a warning.

Sarvam options:

- `mayura:v1` is the subtitle default because it supports colloquial modes. Its UI routes are limited to English, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, and Telugu.
- `sarvam-translate:v1` is available for formal translation and exposes the wider language set supported by the workspace.
- **Use local IndicTrans backup if Sarvam fails** is off by default. Leave it off when you want Sarvam errors to stop the job so you can fix the key, model, language, or account issue. Enable it only when you explicitly want backup output.
- Enabling the local backup limits the route picker to the intersection both providers can handle: English to an Indic target supported by the selected Sarvam model. The UI shows the backup checkpoint identity and blocks the run until that checkpoint is ready.
- When fallback is enabled and used, the output document includes a `FALLBACK USED` warning. Completion messages also report Sarvam API attempts, successful responses, and the last Sarvam `request_id` when available.
- Sarvam requests are sent one at a time with a short delay and longer retry backoff on HTTP 429 rate limits. If a job is interrupted, rerun with the same file, settings, glossary, and provider to resume from `.translation-checkpoints/` instead of resending completed chunks.
- The pipeline reduces unnecessary Sarvam input by removing outer subtitle markup from model inputs, skipping model calls for protected-only units, and respecting the provider's effective character cap. The provider summary reports sent and successful input characters.

## Glossary JSON format

Use `examples/sample_glossary.json` or paste directly into the glossary editor in the app.

```json
{
  "glossary": {"English term": "বাংলা টার্ম"},
  "do_not_translate": ["BrandName", "PersonName"]
}
```

Speaker names detected automatically from the subtitle file are merged into `do_not_translate` at runtime — you don't need to list them manually.

## Translation pipeline

1. **Structure-aware parsing** — subtitle cues become individually addressable units; TXT/Markdown becomes a document IR whose translatable blocks are separate from format syntax and protected spans.
2. **Per-unit preparation** — speaker labels, wrapping markup, ALL-CAPS stage directions, do-not-translate terms, inline code, and URLs are protected before grouping, so every unit retains its own metadata.
3. **Bounded context planning** — adjacent units are packed in source order up to a character budget and unit-count limit, then refined against a local model's exact post-processor token count. Headings start new document context sections. Blank lines remain structural output, but adjacent paragraphs can still share a translation window for continuity.
4. **Exact boundary encoding** — multi-unit windows use numbered markers. The decoder requires every marker exactly once and in the original order, and rejects empty translations for non-empty source units.
5. **Inference** — the selected backend translates each window. Local IndicTrans2 uses the official `IndicProcessor` pre-processing → tokenizer/model generation → `IndicProcessor` post-processing flow with beam search.
6. **Alignment fallback** — an invalid multi-unit response is discarded and only that window is retried as one input per cue/block. This replaces heuristic punctuation-based re-splitting.
7. **Overrides and restoration** — glossary overrides run while protected sentinels are still hidden; protected content, layout, and source-unit metadata are then restored and validated in their exact original count and order.
8. **Format-specific output** — subtitle text is line-wrapped without changing timing; document text is serialized back into the preserved TXT/Markdown structure.

## Design notes

- **Offline inference**: `local_files_only=True` is enforced on local model loading. Missing weights produce a clear error at startup.
- **IndicTrans2 preference**: default backend is `indictrans2`; `sarvam-api`, `nllb`, and `echo` are swappable alternatives.
- **Sarvam API security**: API keys are accepted from masked UI input, `SARVAM_API_KEY`, or the OS keychain; they are not stored in repo files.
- **Sentinel protection**: glossary terms are wrapped in `ZZID{n}ZZ` placeholders before inference and restored afterward.
- **Context is bounded, not guessed**: character and unit limits keep requests within the selected backend's usable window while still giving the model adjacent text.
- **Alignment is fail-safe**: output is only mapped back when the exact marker contract validates; otherwise the affected context window takes the safe one-unit fallback path and emits a warning.
- **Checkpoint identity includes the model**: local model-file metadata and generation limits, or hosted model/mode settings, are fingerprinted so replacing a model cannot silently reuse stale output.
- **Markdown scope**: the first document adapter intentionally supports a conservative CommonMark subset. `.docx`, `.pdf`, OCR, and rich-layout reconstruction are not supported yet.

## Sample commands

Echo/test mode (safe parsing + serialization check, no model needed):

```bash
streamlit run app.py
# In Translation provider, choose "Echo · structure test only"
```

Syntax verification:

```bash
python -m compileall app.py gui.py subtitle_translator
```
