# Local Offline Subtitle Translator

A desktop-style local web app for translating English subtitle files (`.srt` / `.vtt`) to Indian languages while preserving timestamps and output format. Runs entirely offline — no API keys or internet connection needed after model download.

## Features

- Drag-and-drop upload for `.srt` and `.vtt`
- Preserves cue timing, order, and output format
- Input and translated side-by-side previews
- Default language pair: `en → bn` (Bengali); supports all IndicTrans2 Indic targets
- **Auto speaker-name detection**: scans cues for `SPEAKER: ...` patterns and auto-adds detected names to the do-not-translate list
- **Speaker label extraction**: ALL-CAPS speaker labels (e.g. `POIROT:`) are stripped before translation and re-attached after, preventing the model from mangling them; glossary mappings are applied to labels too
- **Stage direction normalization**: `(BELL TOLLING)` is lowercased before translation so the model translates semantically rather than transliterating
- **Glossary protection via `<dnt>` tags**: IndicTrans2-compatible entity protection keeps named terms verbatim through inference
- **Post-translation glossary overrides**: case-insensitive regex replacements applied after translation for reliable term substitution
- Glossary JSON upload and in-UI editor
- Batch translation with adjustable chunk size
- Merges short cues for better context, then re-splits into original cue count
- Subtitle-friendly line wrapping and max line length controls
- ETA and progress display during translation
- Download translated subtitle in same format as input
- Echo/test mode to validate parsing + writing pipeline without model inference
- Translator abstraction with pluggable backends (`indictrans2`, `nllb`, `echo`)

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
    ├── segmentation.py
    ├── formatter.py
    ├── pipeline.py                 # Translation orchestration
    ├── speaker_detection.py        # Auto speaker-name scanner
    └── translators/
        ├── __init__.py
        ├── base.py
        ├── echo.py
        ├── indictrans2.py
        ├── nllb.py
        └── factory.py
```

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

Subtitle files are decoded using a fallback chain: UTF-8 with BOM, UTF-8, UTF-16, cp1252, then latin-1. Common Windows-encoded `.srt` files load without manual conversion.

## Model setup (offline)

Place a local model folder and set **Local model path** in the app sidebar.

Default expected path:

```text
./models/indictrans2-en-indic
```

A helper script fetches weights from HuggingFace into the expected location:

```bash
python scripts/download_models.py                 # IndicTrans2 distilled 200M (~800 MB)
python scripts/download_models.py --model 1B      # IndicTrans2 full 1B (~4.5 GB)
python scripts/download_models.py --model nllb    # NLLB-200 distilled 600M (~2.5 GB)
```

The `models/` directory is gitignored. The default destination matches the default "Local model path" in both GUIs, so no extra configuration is needed after download.

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

1. **Cue merging** — short cues are merged for context, then re-split back to the original cue count after translation.
2. **Speaker label extraction** — `SPEAKER: body` is split into `(label, body)` before the model sees the text; the label is reattached (with any glossary mapping applied) after.
3. **Stage direction normalization** — `(ALL CAPS)` parentheticals are lowercased so the model translates them rather than phonetically copying them.
4. **Inference** — batches sent to the backend with repetition penalties to suppress looping artifacts.
5. **Glossary overrides** — case-insensitive regex replacements applied to model output.
6. **Line wrapping** — translated text formatted to subtitle-safe line lengths.

## Design notes

- **Offline inference**: `local_files_only=True` is enforced on model loading. Missing weights produce a clear error at startup.
- **IndicTrans2 preference**: default backend is `indictrans2`; `nllb` is a swappable alternative.
- **`<dnt>` tag protection**: glossary terms are wrapped in `<dnt>…</dnt>` before inference. IndicTrans2 is trained with IndicTransToolkit which recognises this format and passes tag contents through unchanged.
- **Cue re-splitting**: merged chunk translation is split back by line and punctuation heuristics — robust for MVP, improvable with token-level alignment.

## Sample commands

Echo/test mode (safe parsing + serialization check, no model needed):

```bash
streamlit run app.py
# In UI: enable "Echo/test mode"
```

Syntax verification:

```bash
python -m compileall app.py subtitle_translator
```
