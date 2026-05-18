# Local Offline Subtitle Translator (Streamlit)

A desktop-style local web app for translating existing English subtitle files (`.srt` / `.vtt`) to Bengali while preserving timestamps and output format.

## What this MVP does

- Drag-and-drop upload for `.srt` and `.vtt`
- Preserves cue timing and order
- Input and translated previews
- Default language controls: `en -> bn`
- Glossary JSON upload/edit in UI
- Do-not-translate support
- Batch translation with adjustable chunk size
- Merges short cues for better context, then re-splits into original cue count
- Subtitle-friendly line wrapping and max line length controls
- Progress and clear status/error messages
- Download translated subtitle in same format as input
- Echo/test mode to validate parsing + writing pipeline without model inference
- Translator abstraction with pluggable backends (`indictrans2`, `nllb`, `echo`)

## Project structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── examples/
│   └── sample_glossary.json
└── subtitle_translator/
    ├── __init__.py
    ├── models.py
    ├── parsers.py
    ├── glossary.py
    ├── segmentation.py
    ├── formatter.py
    ├── pipeline.py
    └── translators/
        ├── __init__.py
        ├── base.py
        ├── echo.py
        ├── indictrans2.py
        ├── nllb.py
        └── factory.py
```

## Assumptions and design notes

1. **Offline inference**: the app enforces local model usage (`local_files_only=True`) for model loading. If model weights are missing locally, initialization fails with a clear error.
2. **IndicTrans2 preference**: default backend is `indictrans2`; `nllb` is provided as a swappable backend.
3. **Glossary strategy**:
   - Protected terms (do-not-translate + glossary keys) are tokenized before translation, restored after translation.
   - Glossary replacements are then applied as post-processing.
4. **Cue re-splitting**: merged cue translation is split back to original cue count using line and punctuation heuristics. This is robust for MVP but can be improved with alignment in future.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Two front-ends are available — pick whichever you prefer.

Native desktop GUI (PySide6):

```bash
python gui.py
```

Browser GUI (Streamlit):

```bash
streamlit run app.py
```

Both share the same parser, pipeline, glossary, and translator backends.

## Tests

```bash
pytest -q
```

The test suite uses the `echo` backend, so no model weights are required.

## File encoding

Subtitle files are decoded using a fallback chain: UTF-8 with BOM, UTF-8, UTF-16,
cp1252, then latin-1. This means common Windows-encoded `.srt` files load
without manual conversion.

## Model setup (offline)

Place a local model folder and set **Local model path** in the app sidebar.

Example expected path (default in UI):

```text
./models/indictrans2-en-indic
```

If you want full offline runtime, pre-download model artifacts ahead of time and copy them into this path.

A helper script is included to fetch a model from HuggingFace into the expected location:

```bash
python scripts/download_models.py                 # IndicTrans2 distilled 200M (~800 MB)
python scripts/download_models.py --model 1B      # IndicTrans2 full 1B (~4.5 GB)
python scripts/download_models.py --model nllb    # NLLB-200 distilled 600M (~2.5 GB)
```

The `models/` directory is gitignored. The default destination matches the
default "Local model path" in both GUIs, so no extra configuration is needed
after download.

## Sample glossary JSON

Use `examples/sample_glossary.json` or paste directly into the glossary editor in app.

Schema:

```json
{
  "glossary": {"English term": "বাংলা টার্ম"},
  "do_not_translate": ["BrandName", "PersonName"]
}
```

## Sample commands

Echo/test mode flow (safe parsing/serialization check):

```bash
streamlit run app.py
# In UI: enable "Echo/test mode"
```

Quick syntax verification:

```bash
python -m compileall app.py subtitle_translator
```

## Next practical improvements

- Better sentence-to-cue alignment using token-level alignment.
- Dedicated Bengali punctuation normalization.
- Optional GPU device selection and model warmup diagnostics.
- Unit tests for parser edge-cases and chunk split logic.
