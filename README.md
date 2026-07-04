# Subtitle Translator

A desktop-style app for translating English subtitle files (`.srt` / `.vtt`) to Indian languages while preserving timestamps and output format. It defaults to local offline models, with an optional Sarvam API backend for users who provide their own API key.

## Features

- Drag-and-drop upload for `.srt` and `.vtt`
- Preserves cue timing, order, and output format
- Input and translated side-by-side previews
- Default language pair: `en → bn` (Bengali); supports all IndicTrans2 Indic targets
- **Auto speaker-name detection**: scans cues for `SPEAKER: ...` patterns and auto-adds detected names to the do-not-translate list
- **Speaker label extraction**: ALL-CAPS speaker labels (e.g. `POIROT:`) are stripped before translation and re-attached after, preventing the model from mangling them; glossary mappings are applied to labels too
- **Stage direction normalization**: `(BELL TOLLING)` is lowercased before translation so the model translates semantically rather than transliterating
- **Glossary protection via sentinels**: protected terms are hidden behind model-resistant placeholders during inference and restored after translation
- **Post-translation glossary overrides**: case-insensitive regex replacements applied after translation for reliable term substitution
- Optional Sarvam API backend with password entry, environment/keychain lookup, model/mode selection, retries, and local fallback
- Glossary JSON upload and in-UI editor
- Batch translation with adjustable chunk size
- Merges short cues for better context, then re-splits into original cue count
- Subtitle-friendly line wrapping and max line length controls
- ETA and progress display during translation
- Download translated subtitle in same format as input
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
    ├── segmentation.py
    ├── formatter.py
    ├── pipeline.py                 # Translation orchestration
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

## Sarvam API setup (optional)

Choose the `sarvam-api` backend when you want to use Sarvam's hosted translation models instead of local inference.

API keys are resolved in this order:

1. The password field in the UI for the current run.
2. The `SARVAM_API_KEY` environment variable.
3. A key saved in the OS keychain.

Keys are never written to project files. To save a key, enter it in the UI and enable **Save key in OS keychain**. If the optional keychain backend is unavailable on the machine, the app will continue using the key for the current run and show a warning.

Sarvam options:

- `mayura:v1` is the default because it supports colloquial modes that usually fit subtitle dialogue better.
- `sarvam-translate:v1` is available for formal translation and supports a wider set of Indian languages.
- **Fallback to local IndicTrans** loads the local model only if a Sarvam batch fails; when fallback is used, the output document includes a warning.

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
5. **Protected-term restore** — do-not-translate sentinels are restored to their original text.
6. **Glossary overrides** — case-insensitive regex replacements applied to model output.
7. **Line wrapping** — translated text formatted to subtitle-safe line lengths.

## Design notes

- **Offline inference**: `local_files_only=True` is enforced on local model loading. Missing weights produce a clear error at startup.
- **IndicTrans2 preference**: default backend is `indictrans2`; `sarvam-api`, `nllb`, and `echo` are swappable alternatives.
- **Sarvam API security**: API keys are accepted from masked UI input, `SARVAM_API_KEY`, or the OS keychain; they are not stored in repo files.
- **Sentinel protection**: glossary terms are wrapped in `ZZID{n}ZZ` placeholders before inference and restored afterward.
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
