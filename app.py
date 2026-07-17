from __future__ import annotations

import base64
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Iterable

import streamlit as st

from subtitle_translator.credentials import (
    CredentialStorageError,
    save_sarvam_api_key,
)
from subtitle_translator.defaults import DEFAULT_GLOSSARY
from subtitle_translator.document_pipeline import (
    DocumentTranslationInterruptedError,
    DocumentTranslationSettings,
    translate_text_document,
)
from subtitle_translator.documents import (
    DocumentParseError,
    parse_document,
    serialize_document,
)
from subtitle_translator.glossary import GlossaryConfig, load_glossary_json
from subtitle_translator.parsers import (
    SubtitleParseError,
    decode_subtitle_bytes,
    parse_subtitle,
    serialize_subtitle,
)
from subtitle_translator.pipeline import (
    TranslationInterruptedError,
    TranslationSettings,
    make_translation_checkpoint_path,
    translate_document,
)
from subtitle_translator.translators.factory import TranslatorInitError, build_translator
from subtitle_translator.translators.fallback import FallbackTranslationError
from subtitle_translator.translators.sarvam_api import SarvamApiError


def _init_state() -> None:
    defaults = {
        "translated_text": "",
        "last_file_name": "",
        "active_job_signature": "",
        "result_output_name": "",
        "result_mime": "text/plain",
        "result_provider": "",
        "result_warnings": [],
        "result_is_partial": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _clear_result() -> None:
    st.session_state.translated_text = ""
    st.session_state.result_output_name = ""
    st.session_state.result_mime = "text/plain"
    st.session_state.result_provider = ""
    st.session_state.result_warnings = []
    st.session_state.result_is_partial = False


def _store_result(
    output_text: str,
    *,
    output_name: str,
    mime: str,
    provider: str,
    warnings: Iterable[str] = (),
    partial: bool = False,
) -> None:
    st.session_state.translated_text = output_text
    st.session_state.result_output_name = output_name
    st.session_state.result_mime = mime
    st.session_state.result_provider = provider
    st.session_state.result_warnings = list(warnings)
    st.session_state.result_is_partial = partial


def _fallback_was_used(translator) -> bool:
    return int(getattr(translator, "fallback_count", 0) or 0) > 0


LANGUAGE_OPTIONS = {
    "en": "English · English",
    "as": "Assamese · অসমীয়া",
    "bn": "Bengali · বাংলা",
    "brx": "Bodo · बर'",
    "doi": "Dogri · डोगरी",
    "gu": "Gujarati · ગુજરાતી",
    "hi": "Hindi · हिन्दी",
    "kn": "Kannada · ಕನ್ನಡ",
    "kok": "Konkani · कोंकणी",
    "ks": "Kashmiri · کٲشُر",
    "mai": "Maithili · मैथिली",
    "ml": "Malayalam · മലയാളം",
    "mni": "Manipuri (Meitei)",
    "mr": "Marathi · मराठी",
    "ne": "Nepali · नेपाली",
    "or": "Odia · ଓଡ଼ିଆ",
    "pa": "Punjabi · ਪੰਜਾਬੀ",
    "sa": "Sanskrit · संस्कृतम्",
    "sat": "Santali · ᱥᱟᱱᱛᱟᱲᱤ",
    "sd": "Sindhi · سنڌي",
    "ta": "Tamil · தமிழ்",
    "te": "Telugu · తెలుగు",
    "ur": "Urdu · اردو",
}

# Mayura supports a narrower route set than Sarvam Translate. Keep these as
# internal language codes; ``SarvamApiTranslator`` maps Odia's ``or`` to the
# provider's ``od-IN`` code at request time.
SARVAM_MAYURA_CODES = (
    "en",
    "bn",
    "gu",
    "hi",
    "kn",
    "ml",
    "mr",
    "or",
    "pa",
    "ta",
    "te",
)

PROVIDER_OPTIONS = {
    "indictrans2": "IndicTrans2 · local/offline",
    "sarvam-api": "Sarvam API · hosted",
    "nllb": "NLLB · local/offline",
    "echo": "Echo · structure test only",
}

SUBTITLE_EXTENSIONS = {".srt", ".vtt"}
DOCUMENT_EXTENSIONS = {".txt", ".md"}


def _local_model_identity(model_path: str) -> str:
    config_path = Path(model_path).expanduser() / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return Path(model_path).name or model_path
    if not isinstance(config, dict):
        return Path(model_path).name or model_path
    identity = config.get("name_or_path") or config.get("_name_or_path")
    return str(identity) if identity else (Path(model_path).name or model_path)


def _local_model_status(
    model_path: str,
    expected_backend: str,
) -> tuple[bool, str, str]:
    """Inspect a local checkpoint before enabling an inference action."""

    path = Path(model_path).expanduser()
    identity = path.name or model_path
    backend_label = "IndicTrans2" if expected_backend == "indictrans2" else "NLLB"
    if not path.is_dir():
        return False, identity, f"Local model directory was not found: {model_path}"

    config_path = path / "config.json"
    if not config_path.is_file():
        return (
            False,
            identity,
            f"{backend_label} model is not ready: {config_path} is missing.",
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        return False, identity, f"Could not read local model config: {exc}"
    if not isinstance(config, dict):
        return False, identity, "Local model config.json must contain a JSON object."

    configured_identity = config.get("name_or_path") or config.get("_name_or_path")
    if configured_identity:
        identity = str(configured_identity)
    architectures = config.get("architectures")
    if isinstance(architectures, list):
        architecture_text = " ".join(str(item) for item in architectures)
    else:
        architecture_text = str(architectures or "")
    evidence = " ".join(
        (
            identity,
            str(config.get("model_type") or ""),
            architecture_text,
        )
    ).lower()
    expected_marker = "indictrans" if expected_backend == "indictrans2" else "nllb"
    if expected_marker not in evidence:
        return (
            False,
            identity,
            f"Model type mismatch: {identity} is not an identifiable {backend_label} checkpoint.",
        )
    return True, identity, f"{backend_label} checkpoint is ready: {identity}"


def _estimated_context_windows(
    texts: Iterable[str],
    *,
    max_chars: int,
    max_units: int,
) -> int:
    """Estimate packed windows without invoking a provider or altering source text."""

    char_limit = max(1, int(max_chars))
    unit_limit = max(1, int(max_units))
    windows = 0
    used_chars = 0
    used_units = 0

    for text in texts:
        length = max(1, len(text.strip()))
        piece_count = max(1, math.ceil(length / char_limit))
        remaining = length
        for _ in range(piece_count):
            piece_length = min(char_limit, remaining)
            separator_cost = 1 if used_units else 0
            if used_units and (
                used_units >= unit_limit
                or used_chars + separator_cost + piece_length > char_limit
            ):
                windows += 1
                used_chars = 0
                used_units = 0
                separator_cost = 0
            used_chars += separator_cost + piece_length
            used_units += 1
            remaining = max(0, remaining - piece_length)

    return windows + (1 if used_units else 0)


def _estimated_document_windows(document, *, max_chars: int, max_blocks: int) -> int:
    windows = 0
    semantic_group: list[str] = []

    def flush() -> None:
        nonlocal windows
        if semantic_group:
            windows += _estimated_context_windows(
                semantic_group,
                max_chars=max_chars,
                max_units=max_blocks,
            )
            semantic_group.clear()

    for block in document.blocks:
        if block.kind == "heading":
            flush()
        if not block.translatable:
            flush()
            continue
        semantic_group.append(block.source_text)
    flush()
    return windows


def _job_signature(source_bytes: bytes, settings: dict, glossary_raw: str) -> str:
    digest = hashlib.sha256()
    digest.update(source_bytes)
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    digest.update(glossary_raw.encode("utf-8"))
    return digest.hexdigest()


def _mime_type(ext: str) -> str:
    return {
        ".srt": "application/x-subrip",
        ".vtt": "text/vtt",
        ".md": "text/markdown",
        ".txt": "text/plain",
    }.get(ext, "text/plain")


def _effective_plan_char_limit(
    *,
    backend: str,
    sarvam_model: str,
    sarvam_fallback_enabled: bool,
    requested: int,
) -> int:
    provider_cap: int | None = None
    if backend in {"indictrans2", "nllb"} or (
        backend == "sarvam-api" and sarvam_fallback_enabled
    ):
        provider_cap = 500
    elif backend == "sarvam-api":
        provider_cap = 1000 if sarvam_model == "mayura:v1" else 2000
    if provider_cap is None:
        return requested
    return min(requested, max(1, int(provider_cap * 0.9)))


HELP_TEXT = {
    "backend": (
        "Choose the translation engine. IndicTrans2 is offline and private, Sarvam API can "
        "be faster and more natural with a valid key, echo is for testing, and NLLB is an "
        "alternative local backend."
    ),
    "model_path": (
        "Directory containing local model weights. A wrong path stops local translation; "
        "larger IndicTrans models usually improve quality but load more slowly."
    ),
    "sarvam_api_key": (
        "Optional hosted translation key. Leave blank to use SARVAM_API_KEY or a saved "
        "keychain value. Keys entered here are only used for the current run unless saved."
    ),
    "sarvam_save_key": (
        "Save the Sarvam key to the OS keychain so you do not have to paste it again. "
        "Leave off on shared machines."
    ),
    "sarvam_model": (
        "Mayura is tuned for colloquial dialogue and usually fits subtitles better. "
        "Sarvam Translate is more formal and may be better for literal/documentary lines."
    ),
    "sarvam_mode": (
        "Controls tone for Mayura. Classic colloquial is a safe subtitle default; modern "
        "colloquial can feel more casual; formal may sound cleaner but less conversational."
    ),
    "sarvam_fallback": (
        "When on, failed Sarvam batches fall back to local IndicTrans instead of stopping. "
        "This can save a run, but mixed-provider output should be reviewed."
    ),
    "source_lang": (
        "Language of the input text. If this is wrong, translation quality "
        "drops sharply because the model interprets the source incorrectly."
    ),
    "target_lang": (
        "Language for the translated output. Bengali remains the default."
    ),
    "chunk_size": (
        "Number of context windows translated per pipeline batch. Larger values can be faster, "
        "but use more memory and make failures affect more text at once."
    ),
    "context_window_chars": (
        "Character budget for one semantic context window. The pipeline maps every cue or "
        "document block back exactly and safely retries individual units if alignment fails."
    ),
    "context_window_units": (
        "Maximum cues or document blocks grouped into one model request. More units add context; "
        "the character budget still prevents oversized provider requests."
    ),
    "max_line_length": (
        "Preferred characters per subtitle line. Lower values are easier to read on small "
        "screens, but can create more line breaks."
    ),
    "max_lines": (
        "Maximum lines per cue after wrapping. Two is subtitle-friendly; more lines preserve "
        "longer text but can cover too much of the video."
    ),
    "source_file": (
        "Drop or upload a subtitle (.srt/.vtt) or text document (.txt/.md). You can drop "
        "supported source files anywhere on the page."
    ),
    "glossary_file": (
        "Optional JSON with glossary replacements and do-not-translate terms. These merge "
        "with the built-in defaults for the current run."
    ),
    "glossary_json": (
        "Edit glossary overrides and protected terms before running. Invalid JSON will stop "
        "translation so the app does not run with a broken glossary."
    ),
}


APP_CSS = """
<style>
:root {
    --indicsub-background: #fafafa;
    --indicsub-foreground: #09090b;
    --indicsub-card: #ffffff;
    --indicsub-muted: #f4f4f5;
    --indicsub-muted-foreground: #71717a;
    --indicsub-border: #e4e4e7;
    --indicsub-primary: #18181b;
    --indicsub-primary-foreground: #fafafa;
    --indicsub-ring: #a1a1aa;
    --indicsub-radius: 10px;
}

.stApp {
    background: var(--indicsub-background);
    color: var(--indicsub-foreground);
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2.5rem;
}

[data-testid="stSidebar"] {
    background: var(--indicsub-card);
    border-right: 1px solid var(--indicsub-border);
}

header[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stToolbar"],
#MainMenu,
footer {
    visibility: hidden;
    height: 0;
}

[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
[data-testid="stExpandSidebarButton"] {
    align-items: center;
    display: inline-flex !important;
    justify-content: center;
    cursor: pointer;
    opacity: 1 !important;
    visibility: visible !important;
}

[data-testid="stSidebarHeader"] {
    align-items: center;
    min-height: 2.75rem;
    visibility: visible !important;
}

[data-testid="stSidebarCollapseButton"] button,
[data-testid="collapsedControl"] button,
[data-testid="stExpandSidebarButton"] {
    align-items: center !important;
    background: var(--indicsub-card) !important;
    border: 1px solid var(--indicsub-border) !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 2px rgba(24, 24, 27, 0.06);
    color: var(--indicsub-foreground) !important;
    cursor: pointer !important;
    display: inline-flex !important;
    height: 2rem !important;
    justify-content: center !important;
    min-height: 2rem !important;
    padding: 0 !important;
    width: 2rem !important;
}

[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="collapsedControl"] button:hover,
[data-testid="stExpandSidebarButton"]:hover {
    background: var(--indicsub-muted) !important;
    border-color: var(--indicsub-ring) !important;
}

[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
[data-testid="collapsedControl"] [data-testid="stIconMaterial"],
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {
    color: transparent !important;
    display: inline-flex !important;
    font-size: 0 !important;
    height: 1rem !important;
    justify-content: center;
    line-height: 0 !important;
    position: relative;
    width: 1rem !important;
}

[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"]::before,
[data-testid="collapsedControl"] [data-testid="stIconMaterial"]::before,
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"]::before {
    color: var(--indicsub-foreground);
    font-family: ui-sans-serif, system-ui, sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    line-height: 1;
    position: absolute;
}

[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"]::before {
    content: "<";
}

[data-testid="collapsedControl"] [data-testid="stIconMaterial"]::before,
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"]::before {
    content: ">";
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: var(--indicsub-foreground);
    letter-spacing: 0;
}

.app-hero {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1.25rem;
    border: 1px solid var(--indicsub-border);
    border-radius: var(--indicsub-radius);
    background: var(--indicsub-card);
    padding: 1.25rem 1.35rem;
    margin-bottom: 1rem;
}

.app-eyebrow,
.section-kicker {
    margin: 0 0 0.35rem;
    color: var(--indicsub-muted-foreground);
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.app-hero h1,
.section-heading h3 {
    margin: 0;
    color: var(--indicsub-foreground);
    letter-spacing: 0;
}

.app-hero h1 {
    font-size: 2rem;
    line-height: 1.1;
}

.app-hero p,
.section-heading p {
    color: var(--indicsub-muted-foreground);
}

.app-hero p {
    margin: 0.45rem 0 0;
    font-size: 0.95rem;
}

.hero-badges,
.status-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: flex-end;
}

.hero-badge,
.status-pill {
    border: 1px solid var(--indicsub-border);
    border-radius: 999px;
    background: var(--indicsub-muted);
    color: var(--indicsub-foreground);
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.35rem 0.65rem;
    white-space: nowrap;
}

.section-heading {
    margin: 0.2rem 0 0.9rem;
}

.section-heading h3 {
    font-size: 1rem;
}

.section-heading p {
    margin: 0.25rem 0 0;
    font-size: 0.86rem;
}

.status-strip {
    justify-content: flex-start;
    margin-bottom: 1rem;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--indicsub-border);
    border-radius: var(--indicsub-radius);
    background: var(--indicsub-card);
    box-shadow: 0 1px 2px rgba(24, 24, 27, 0.04);
}

label,
[data-testid="stWidgetLabel"] p {
    color: var(--indicsub-foreground);
    font-size: 0.84rem;
    font-weight: 600;
}

[data-testid="stWidgetLabel"] {
    align-items: center !important;
    display: inline-flex !important;
    gap: 0.35rem !important;
}

[data-testid="stTooltipIcon"],
[data-testid="stTooltipHoverTarget"] {
    align-items: center !important;
    color: var(--indicsub-muted-foreground) !important;
    display: inline-flex !important;
    flex: 0 0 auto;
    justify-content: center !important;
    line-height: 1;
    opacity: 1 !important;
    visibility: visible !important;
}

[data-testid="stTooltipIcon"] button,
[data-testid="stTooltipHoverTarget"] button {
    align-items: center !important;
    background: transparent !important;
    border: 0 !important;
    border-radius: 999px !important;
    box-shadow: none !important;
    color: var(--indicsub-muted-foreground) !important;
    cursor: help !important;
    display: inline-flex !important;
    height: 1rem !important;
    justify-content: center !important;
    margin: 0 !important;
    min-height: 1rem !important;
    min-width: 1rem !important;
    padding: 0 !important;
    width: 1rem !important;
}

[data-testid="stTooltipIcon"] button:hover,
[data-testid="stTooltipHoverTarget"] button:hover,
[data-testid="stTooltipIcon"] button:focus,
[data-testid="stTooltipHoverTarget"] button:focus {
    background: var(--indicsub-muted) !important;
    color: var(--indicsub-foreground) !important;
}

[data-testid="stTooltipIcon"] svg,
[data-testid="stTooltipHoverTarget"] svg {
    color: currentColor !important;
    fill: currentColor !important;
    height: 0.95rem !important;
    width: 0.95rem !important;
}

div[data-baseweb="tooltip"],
div[data-baseweb="popover"] [role="tooltip"] {
    background: var(--indicsub-foreground) !important;
    border: 1px solid var(--indicsub-foreground) !important;
    border-radius: 8px !important;
    box-shadow: 0 8px 24px rgba(24, 24, 27, 0.18) !important;
    color: var(--indicsub-primary-foreground) !important;
    font-size: 0.8rem !important;
    line-height: 1.4 !important;
    max-width: 19rem !important;
}

div[data-baseweb="tooltip"] *,
div[data-baseweb="popover"] [role="tooltip"] * {
    color: var(--indicsub-primary-foreground) !important;
}

div[data-baseweb="select"] > div,
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextArea"] textarea {
    border-color: var(--indicsub-border);
    border-radius: calc(var(--indicsub-radius) - 2px);
    background: var(--indicsub-card);
    color: var(--indicsub-foreground) !important;
    -webkit-text-fill-color: var(--indicsub-foreground) !important;
    opacity: 1;
}

div[data-testid="stTextInput"] > div,
div[data-testid="stTextInput"] [data-baseweb="input"] {
    align-items: center;
    border-color: var(--indicsub-border) !important;
    border-radius: calc(var(--indicsub-radius) - 2px) !important;
    background: var(--indicsub-card) !important;
    min-height: 2.5rem;
    overflow: hidden;
}

div[data-testid="stTextInput"] input {
    border: 0 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    min-height: 2.5rem;
    outline: 0 !important;
    padding: 0.55rem 0.75rem !important;
}

div[data-testid="stTextInput"] label button,
div[data-testid="stTextInput"] label button:hover,
div[data-testid="stTextInput"] label button:focus {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    color: var(--indicsub-muted-foreground) !important;
    min-height: auto;
    padding: 0 !important;
}

div[data-testid="stTextInputRootElement"] [data-baseweb="base-input"] {
    align-items: stretch;
    background: var(--indicsub-card) !important;
    border: 0 !important;
    box-shadow: none !important;
    min-height: 2.5rem;
}

div[data-testid="stTextInputRootElement"] button,
div[data-testid="stTextInputRootElement"] button:hover,
div[data-testid="stTextInputRootElement"] button:focus {
    align-items: center;
    align-self: stretch;
    background: var(--indicsub-card) !important;
    border: 0 !important;
    border-left: 1px solid var(--indicsub-border) !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    color: var(--indicsub-muted-foreground) !important;
    cursor: pointer;
    display: inline-flex;
    justify-content: center;
    margin: 0 !important;
    min-height: 2.5rem;
    min-width: 2.75rem;
    padding: 0 0.8rem !important;
}

div[data-testid="stTextInputRootElement"] [data-baseweb="base-input"] > button:last-child {
    border-radius: 0 calc(var(--indicsub-radius) - 2px) calc(var(--indicsub-radius) - 2px) 0 !important;
}

div[data-testid="stTextInputRootElement"] button::before,
div[data-testid="stTextInputRootElement"] button::after {
    display: none !important;
}

div[data-testid="stTextInputRootElement"] button svg {
    color: var(--indicsub-muted-foreground) !important;
    fill: currentColor !important;
}

div[data-testid="stTextInputRootElement"] button:hover svg {
    color: var(--indicsub-foreground) !important;
}

div[data-baseweb="select"],
div[data-baseweb="select"] * {
    cursor: pointer !important;
}

div[data-testid="stCheckbox"] label[data-baseweb="checkbox"] {
    align-items: center;
    cursor: pointer;
    gap: 0.65rem;
}

div[data-testid="stCheckbox"] label[data-baseweb="checkbox"] > span:first-child {
    background: var(--indicsub-card) !important;
    border: 1px solid #d4d4d8 !important;
    border-radius: 6px !important;
    box-shadow: 0 1px 1px rgba(24, 24, 27, 0.04);
    box-sizing: border-box;
    height: 1rem !important;
    min-height: 1rem !important;
    min-width: 1rem !important;
    position: relative;
    width: 1rem !important;
}

div[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:hover > span:first-child {
    border-color: var(--indicsub-ring) !important;
}

div[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input:checked) > span:first-child,
div[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input[aria-checked="true"]) > span:first-child {
    background: var(--indicsub-primary) !important;
    border-color: var(--indicsub-primary) !important;
}

div[data-testid="stCheckbox"] label[data-baseweb="checkbox"] > span:first-child::after {
    border: solid var(--indicsub-primary-foreground);
    border-width: 0 2px 2px 0;
    content: "";
    height: 0.5rem;
    left: 0.3rem;
    opacity: 0;
    position: absolute;
    top: 0.12rem;
    transform: rotate(45deg);
    width: 0.28rem;
}

div[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input:checked) > span:first-child::after,
div[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input[aria-checked="true"]) > span:first-child::after {
    opacity: 1;
}

div[data-testid="stCheckbox"] label[data-baseweb="checkbox"] input {
    accent-color: var(--indicsub-primary);
}

div[data-testid="stSlider"] [data-baseweb="slider"] {
    color: var(--indicsub-primary);
}

div[data-testid="stSlider"] [data-baseweb="slider"] [style*="height: 0.25rem"] {
    background: var(--indicsub-primary) !important;
    border-radius: 999px !important;
    height: 0.35rem !important;
}

div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
    background: var(--indicsub-muted) !important;
    border-radius: 999px !important;
}

div[data-testid="stSlider"] [role="slider"] {
    background: var(--indicsub-card) !important;
    border: 2px solid var(--indicsub-primary) !important;
    border-radius: 999px !important;
    box-shadow: 0 1px 3px rgba(24, 24, 27, 0.16);
    height: 1rem !important;
    outline: none !important;
    width: 1rem !important;
}

div[data-testid="stSlider"] [role="slider"]:focus {
    box-shadow: 0 0 0 3px rgba(24, 24, 27, 0.12);
}

div[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
    color: var(--indicsub-foreground) !important;
}

div[data-testid="stSlider"] [data-testid="stSliderTickBar"],
div[data-testid="stSlider"] [data-baseweb="tick-bar"],
div[data-testid="stSlider"] [data-baseweb="tickbar"] {
    display: none !important;
    pointer-events: none !important;
}

div[data-testid="stTextArea"] textarea {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.86rem;
}

div[data-baseweb="select"] span,
div[data-baseweb="select"] svg {
    color: var(--indicsub-foreground) !important;
    fill: var(--indicsub-foreground) !important;
}

div[data-testid="stTextArea"] textarea:disabled,
div[data-testid="stTextInput"] input:disabled {
    color: var(--indicsub-muted-foreground) !important;
    -webkit-text-fill-color: var(--indicsub-muted-foreground) !important;
}

div[data-testid="stTextArea"] textarea::placeholder,
div[data-testid="stTextInput"] input::placeholder {
    color: var(--indicsub-muted-foreground) !important;
    -webkit-text-fill-color: var(--indicsub-muted-foreground) !important;
}

div[data-testid="stFileUploader"] section {
    border-color: var(--indicsub-border);
    border-radius: calc(var(--indicsub-radius) - 2px);
    background: var(--indicsub-muted);
}

div[data-testid="stFileUploader"] section,
div[data-testid="stFileUploader"] section * {
    color: var(--indicsub-muted-foreground) !important;
}

div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    background: var(--indicsub-primary) !important;
    border-color: var(--indicsub-primary) !important;
}

div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] *,
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] :is(span, p) {
    color: var(--indicsub-primary-foreground) !important;
    -webkit-text-fill-color: var(--indicsub-primary-foreground) !important;
}

div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileSize"],
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileSize"] * {
    color: rgba(250, 250, 250, 0.72) !important;
    -webkit-text-fill-color: rgba(250, 250, 250, 0.72) !important;
}

div[data-testid="stFileUploader"] button,
div[data-testid="stFileUploader"] button * {
    background: var(--indicsub-card) !important;
    color: var(--indicsub-foreground) !important;
    -webkit-text-fill-color: var(--indicsub-foreground) !important;
}

#indicsub-page-drop-overlay {
    align-items: center;
    backdrop-filter: blur(4px);
    background: rgba(250, 250, 250, 0.82);
    border: 2px dashed var(--indicsub-primary);
    border-radius: var(--indicsub-radius);
    color: var(--indicsub-foreground);
    display: none;
    font-size: 1.05rem;
    font-weight: 700;
    inset: 1rem;
    justify-content: center;
    pointer-events: none;
    position: fixed;
    text-align: center;
    z-index: 999999;
}

div[data-testid="stButton"] > button,
div[data-testid="stDownloadButton"] > button {
    min-height: 2.35rem;
    border: 1px solid var(--indicsub-border);
    border-radius: calc(var(--indicsub-radius) - 2px);
    background: var(--indicsub-card);
    color: var(--indicsub-foreground);
    font-weight: 600;
    box-shadow: 0 1px 1px rgba(24, 24, 27, 0.04);
}

div[data-testid="stButton"] > button:hover,
div[data-testid="stDownloadButton"] > button:hover {
    border-color: var(--indicsub-ring);
    background: var(--indicsub-muted);
    color: var(--indicsub-foreground);
}

div[data-testid="stButton"] > button[kind="primary"] {
    border-color: var(--indicsub-primary);
    background: var(--indicsub-primary);
    color: var(--indicsub-primary-foreground);
}

div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #27272a;
    color: var(--indicsub-primary-foreground);
}

[data-testid="stProgress"] > div > div > div > div {
    background: var(--indicsub-primary);
}

[data-testid="stAlert"] {
    border-radius: var(--indicsub-radius);
    border-color: var(--indicsub-border);
}

hr {
    border-color: var(--indicsub-border);
}

@media (max-width: 760px) {
    .app-hero {
        align-items: flex-start;
        flex-direction: column;
    }

    .hero-badges {
        justify-content: flex-start;
    }
}
</style>
"""


def _inject_theme() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)


def _install_page_drop_target() -> None:
    script = """
(() => {
    const appWindow = window;
    const doc = appWindow.document;

    if (appWindow.__indicSubPageDropTargetInstalled) {
        return;
    }
    appWindow.__indicSubPageDropTargetInstalled = true;
    doc.documentElement.setAttribute("data-indicsub-drop-installed", "true");

    const sourceExtensions = [".srt", ".vtt", ".txt", ".md"];

    function ensureOverlay() {
        let overlay = doc.getElementById("indicsub-page-drop-overlay");
        if (!overlay) {
            overlay = doc.createElement("div");
            overlay.id = "indicsub-page-drop-overlay";
            overlay.textContent = "Drop a subtitle or document anywhere to upload";
            doc.body.appendChild(overlay);
        }
        return overlay;
    }

    function isInsideNativeUploader(event) {
        const target = event.target;
        return Boolean(
            target instanceof appWindow.Element
            && target.closest('div[data-testid="stFileUploader"]')
        );
    }

    function hasDraggedFiles(event) {
        const items = Array.from(event.dataTransfer?.items || []);
        return items.some((item) => item.kind === "file");
    }

    function getSourceInput() {
        const uploaders = Array.from(
            doc.querySelectorAll('div[data-testid="stFileUploader"]')
        );
        const sourceUploader = uploaders.find((uploader) => {
            let node = uploader;
            for (let depth = 0; node && depth < 8; depth += 1) {
                const text = node.innerText?.toLowerCase() || "";
                if (text.includes("source file")
                    || text.includes("subtitle or document")) {
                    return true;
                }
                node = node.parentElement;
            }
            return false;
        }) || uploaders[0];
        return sourceUploader?.querySelector('input[type="file"]') || null;
    }

    function sourceFiles(fileList) {
        return Array.from(fileList || []).filter((file) =>
            sourceExtensions.some((extension) =>
                file.name.toLowerCase().endsWith(extension)
            )
        );
    }

    function showOverlay() {
        ensureOverlay().style.display = "flex";
    }

    function hideOverlay() {
        ensureOverlay().style.display = "none";
    }

    doc.addEventListener("dragenter", (event) => {
        if (!isInsideNativeUploader(event) && hasDraggedFiles(event)) {
            showOverlay();
        }
    }, true);

    doc.addEventListener("dragover", (event) => {
        if (!isInsideNativeUploader(event) && hasDraggedFiles(event)) {
            event.preventDefault();
            event.dataTransfer.dropEffect = "copy";
            showOverlay();
        }
    }, true);

    doc.addEventListener("dragleave", (event) => {
        if (event.clientX <= 0 || event.clientY <= 0
            || event.clientX >= appWindow.innerWidth
            || event.clientY >= appWindow.innerHeight) {
            hideOverlay();
        }
    }, true);

    doc.addEventListener("drop", (event) => {
        if (isInsideNativeUploader(event)) {
            hideOverlay();
            return;
        }

        const files = sourceFiles(event.dataTransfer?.files);
        if (!files.length) {
            hideOverlay();
            return;
        }

        const input = getSourceInput();
        if (!input) {
            hideOverlay();
            return;
        }

        event.preventDefault();
        const transfer = new appWindow.DataTransfer();
        files.slice(0, 1).forEach((file) => transfer.items.add(file));
        input.files = transfer.files;
        input.dispatchEvent(new Event("input", { bubbles: true }));
        input.dispatchEvent(new Event("change", { bubbles: true }));
        hideOverlay();
    }, true);
})();
"""
    encoded_script = base64.b64encode(script.encode("utf-8")).decode("ascii")
    st.html(
        f"""
        <script>
            window.__indicSubPageDropTargetBootstrapSeen = true;
            Function(atob("{encoded_script}"))();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def _section_heading(title: str, description: str | None = None) -> None:
    description_html = f"<p>{description}</p>" if description else ""
    st.markdown(
        f"""
        <div class="section-heading">
            <h3>{title}</h3>
            {description_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _status_strip(*items: str) -> None:
    pills = "".join(
        f'<span class="status-pill">{html.escape(str(item))}</span>'
        for item in items
        if item
    )
    if pills:
        st.markdown(f'<div class="status-strip">{pills}</div>', unsafe_allow_html=True)


st.set_page_config(page_title="IndicSub", layout="wide", initial_sidebar_state="expanded")
_init_state()
_inject_theme()
_install_page_drop_target()

st.markdown(
    """
    <section class="app-hero">
        <div>
            <p class="app-eyebrow">Indic translation workspace</p>
            <h1>IndicSub</h1>
            <p>Context-aware translation for subtitles and documents, with exact structure mapping, protected terms, and resumable output.</p>
        </div>
        <div class="hero-badges">
            <span class="hero-badge">.srt / .vtt / .txt / .md</span>
            <span class="hero-badge">Local-first</span>
            <span class="hero-badge">Structure-safe</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    _section_heading(
        "Source file",
        "Start with a timed subtitle or a plain-text/Markdown document.",
    )
    uploaded_file = st.file_uploader(
        "Subtitle or document",
        type=["srt", "vtt", "txt", "md"],
        help=HELP_TEXT["source_file"],
    )
    st.caption(
        "Subtitle timing stays fixed. TXT paragraph spacing and conservative Markdown "
        "structure (headings, lists, links, code, and HTML) are preserved."
    )

uploaded_ext = Path(uploaded_file.name).suffix.lower() if uploaded_file else ""
source_kind = (
    "subtitle"
    if uploaded_ext in SUBTITLE_EXTENSIONS
    else "document"
    if uploaded_ext in DOCUMENT_EXTENSIONS
    else None
)

sarvam_api_key = ""
sarvam_model = "mayura:v1"
sarvam_mode = "classic-colloquial"
sarvam_fallback_enabled = False
sarvam_save_key = False
max_line_length = 42
max_lines = 2

with st.sidebar:
    st.markdown("## Translation setup")
    with st.container(border=True):
        _section_heading("Provider", "Local by default; hosted translation is opt-in.")
        backend = st.selectbox(
            "Translation provider",
            list(PROVIDER_OPTIONS),
            index=0,
            format_func=lambda value: PROVIDER_OPTIONS[value],
            help=HELP_TEXT["backend"],
        )
        if backend in {"indictrans2", "nllb"}:
            st.caption("Offline mode: source text stays on this machine.")
        elif backend == "echo":
            st.caption("Structure test only: output text is not translated.")

    if backend == "sarvam-api":
        with st.container(border=True):
            _section_heading("Sarvam API", "Hosted provider and fallback policy.")
            sarvam_api_key = st.text_input(
                "Sarvam API key",
                value="",
                type="password",
                help=HELP_TEXT["sarvam_api_key"],
            )
            sarvam_save_key = st.checkbox(
                "Save key in OS keychain",
                value=False,
                help=HELP_TEXT["sarvam_save_key"],
            )
            sarvam_model = st.selectbox(
                "Sarvam model",
                ["mayura:v1", "sarvam-translate:v1"],
                index=1 if source_kind == "document" else 0,
                help=HELP_TEXT["sarvam_model"],
            )
            mode_options = ["classic-colloquial", "modern-colloquial", "formal"]
            sarvam_mode = st.selectbox(
                "Translation mode",
                mode_options,
                index=0,
                disabled=sarvam_model == "sarvam-translate:v1",
                help=HELP_TEXT["sarvam_mode"],
            )
            sarvam_fallback_enabled = st.checkbox(
                "Use local IndicTrans backup",
                value=False,
                help=HELP_TEXT["sarvam_fallback"],
            )
            if sarvam_fallback_enabled:
                st.warning(
                    "Local backup is ON. A failed API window may come from IndicTrans; "
                    "mixed-provider output will be flagged in Review."
                )
            else:
                st.caption(
                    "Strict provider mode: an API failure stops and checkpoints the run."
                )

    with st.container(border=True):
        _section_heading(
            "Languages",
            "Only routes supported by the selected provider are shown.",
        )
        indic_codes = [code for code in LANGUAGE_OPTIONS if code != "en"]
        if backend == "indictrans2":
            source_options = ["en"]
            target_options = indic_codes
            language_profile = "en_indic"
        elif backend == "nllb":
            source_options = ["en"]
            target_options = ["bn"]
            language_profile = "nllb_en_bn"
        elif backend == "sarvam-api":
            primary_codes = (
                list(SARVAM_MAYURA_CODES)
                if sarvam_model == "mayura:v1"
                else list(LANGUAGE_OPTIONS)
            )
            if sarvam_fallback_enabled:
                # The backup checkpoint is English→Indic only, so the UI exposes
                # the intersection of the primary and backup route sets.
                source_options = ["en"]
                target_options = [
                    code for code in primary_codes if code != "en" and code in indic_codes
                ]
                language_profile = f"sarvam_{sarvam_model}_local_backup"
            else:
                source_options = primary_codes
                target_options = primary_codes
                language_profile = f"sarvam_{sarvam_model}_strict"
        else:
            source_options = list(LANGUAGE_OPTIONS)
            target_options = list(LANGUAGE_OPTIONS)
            language_profile = "echo_bidirectional"

        source_lang = st.selectbox(
            "Source language",
            source_options,
            index=source_options.index("en") if "en" in source_options else 0,
            format_func=lambda value: LANGUAGE_OPTIONS[value],
            help=HELP_TEXT["source_lang"],
            key=f"source_language_{language_profile}",
        )
        target_lang = st.selectbox(
            "Target language",
            target_options,
            index=target_options.index("bn") if "bn" in target_options else 0,
            format_func=lambda value: LANGUAGE_OPTIONS[value],
            help=HELP_TEXT["target_lang"],
            key=f"target_language_{language_profile}",
        )
        if source_lang == target_lang:
            st.warning("Choose different source and target languages before translating.")

    with st.container(border=True):
        unit_name = (
            "cues"
            if source_kind == "subtitle"
            else "blocks"
            if source_kind == "document"
            else "units"
        )
        _section_heading(
            "Context",
            f"Group related {unit_name} without losing their boundaries.",
        )
        context_slider_max = (
            1800
            if backend == "sarvam-api" and sarvam_model == "sarvam-translate:v1"
            else 1800
            if backend == "echo"
            else 900
        )
        context_window_chars = st.slider(
            "Context character budget",
            200,
            context_slider_max,
            700,
            step=50,
            help=HELP_TEXT["context_window_chars"],
            key=(
                f"context_char_budget_{backend}_"
                f"{sarvam_model if backend == 'sarvam-api' else 'local'}"
            ),
        )
        context_window_units = st.slider(
            f"Maximum {unit_name} per context window",
            1,
            16,
            8 if source_kind != "document" else 6,
            help=HELP_TEXT["context_window_units"],
            key=f"context_units_{source_kind or 'generic'}",
        )
        st.caption(
            "If a model changes boundary markers, only that window is retried one unit at a time."
        )

    with st.expander("Advanced settings", expanded=False):
        model_profile = "nllb" if backend == "nllb" else "indictrans2"
        model_path = st.text_input(
            "Local model path",
            value=(
                "./models/nllb-200-distilled-600M"
                if model_profile == "nllb"
                else "./models/indictrans2-en-indic"
            ),
            help=HELP_TEXT["model_path"],
            key=f"local_model_path_{model_profile}",
            disabled=(
                backend == "echo"
                or (backend == "sarvam-api" and not sarvam_fallback_enabled)
            ),
        )
        chunk_size = st.slider(
            "Pipeline batch size",
            1,
            64,
            12,
            help=HELP_TEXT["chunk_size"],
        )
        if source_kind == "subtitle":
            max_line_length = st.slider(
                "Subtitle line length",
                20,
                60,
                42,
                help=HELP_TEXT["max_line_length"],
            )
            max_lines = st.slider(
                "Maximum lines per cue",
                1,
                4,
                2,
                help=HELP_TEXT["max_lines"],
            )
        elif source_kind == "document":
            st.caption("Document wrapping is preserved by its TXT/Markdown adapter.")

with st.expander("Glossary & protected terms · optional", expanded=False):
    glossary_upload_col, glossary_editor_col = st.columns([0.34, 0.66], gap="large")
    with glossary_upload_col:
        uploaded_glossary = st.file_uploader(
            "Glossary JSON",
            type=["json"],
            help=HELP_TEXT["glossary_file"],
        )
        st.caption(
            "Glossary replacements run after translation. Protected terms are masked before "
            "translation and restored exactly."
        )
    with glossary_editor_col:
        if uploaded_glossary:
            glossary_seed = uploaded_glossary.getvalue().decode("utf-8")
            glossary_upload_signature = hashlib.sha256(
                uploaded_glossary.getvalue()
            ).hexdigest()
            if (
                st.session_state.get("glossary_upload_signature")
                != glossary_upload_signature
            ):
                st.session_state.glossary_json_editor = glossary_seed
                st.session_state.glossary_upload_signature = glossary_upload_signature
        elif "glossary_json_editor" not in st.session_state:
            st.session_state.glossary_json_editor = json.dumps(
                DEFAULT_GLOSSARY,
                ensure_ascii=False,
                indent=2,
            )
        glossary_raw = st.text_area(
            "Glossary JSON editor",
            height=260,
            help=HELP_TEXT["glossary_json"],
            key="glossary_json_editor",
        )

if not uploaded_file:
    with st.container(border=True):
        _section_heading(
            "Workspace",
            "Upload once, inspect the plan, translate, then review and export.",
        )
        st.info(
            "Supported now: SubRip (.srt), WebVTT (.vtt), plain text (.txt), and "
            "conservative Markdown (.md)."
        )
else:
    try:
        source_bytes = uploaded_file.getvalue()
        content = decode_subtitle_bytes(source_bytes)
        ext = uploaded_ext
        if source_kind == "subtitle":
            document = parse_subtitle(content, ext)
            unit_count = len(document.cues)
            translatable_chars = sum(len(cue.text) for cue in document.cues)
            effective_chars = _effective_plan_char_limit(
                backend=backend,
                sarvam_model=sarvam_model,
                sarvam_fallback_enabled=sarvam_fallback_enabled,
                requested=context_window_chars,
            )
            context_windows = _estimated_context_windows(
                (cue.text for cue in document.cues),
                max_chars=effective_chars,
                max_units=context_window_units,
            )
            unit_label = "cues"
            format_label = ext[1:].upper() + " subtitle"
            structure_note = (
                "Cue timings, indices, speaker labels, markup, and ALL-CAPS sound "
                "cues stay mapped to their source cue."
            )
        elif source_kind == "document":
            document = parse_document(content, ext)
            unit_count = len(document.translatable_blocks)
            translatable_chars = sum(
                len(block.source_text) for block in document.translatable_blocks
            )
            effective_chars = _effective_plan_char_limit(
                backend=backend,
                sarvam_model=sarvam_model,
                sarvam_fallback_enabled=sarvam_fallback_enabled,
                requested=context_window_chars,
            )
            context_windows = _estimated_document_windows(
                document,
                max_chars=effective_chars,
                max_blocks=context_window_units,
            )
            unit_label = "translatable blocks"
            format_label = (
                "Markdown document" if ext == ".md" else "Plain-text document"
            )
            protected_count = sum(
                len(block.protected_spans) for block in document.translatable_blocks
            )
            structure_note = (
                f"{len(document.blocks) - unit_count} structural blocks and {protected_count} "
                "inline code/link/HTML spans will bypass translation and be restored in place."
            )
        else:
            raise DocumentParseError(f"Unsupported source format: {ext or 'unknown'}")

        planned_char_limit = _effective_plan_char_limit(
            backend=backend,
            sarvam_model=sarvam_model,
            sarvam_fallback_enabled=sarvam_fallback_enabled,
            requested=context_window_chars,
        )
        requires_local_model = backend in {"indictrans2", "nllb"} or (
            backend == "sarvam-api" and sarvam_fallback_enabled
        )
        expected_local_backend = "nllb" if backend == "nllb" else "indictrans2"
        if requires_local_model:
            local_model_ready, local_model_identity, local_model_message = (
                _local_model_status(model_path, expected_local_backend)
            )
        else:
            local_model_ready = True
            local_model_identity = _local_model_identity(model_path)
            local_model_message = "Local model is not used by this provider configuration."
        provider_identity = (
            f"IndicTrans2 · {local_model_identity} · offline"
            if backend == "indictrans2"
            else f"NLLB · {local_model_identity} · offline"
            if backend == "nllb"
            else (
                f"Sarvam API · {sarvam_model} · backup IndicTrans2 · "
                f"{local_model_identity}"
            )
            if backend == "sarvam-api" and sarvam_fallback_enabled
            else f"Sarvam API · {sarvam_model}"
            if backend == "sarvam-api"
            else "Echo · no model inference"
        )
        fallback_label = (
            ("Local backup ready" if local_model_ready else "Local backup not ready")
            if backend == "sarvam-api" and sarvam_fallback_enabled
            else "Fallback off"
        )
        effective_batch = 1 if backend == "sarvam-api" else chunk_size

        signature_settings = {
            "source_name": uploaded_file.name,
            "source_ext": ext,
            "backend": backend,
            "model_path": model_path,
            "sarvam_model": sarvam_model,
            "sarvam_mode": sarvam_mode,
            "sarvam_fallback": sarvam_fallback_enabled,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "chunk_size": chunk_size,
            "context_window_chars": context_window_chars,
            "context_window_units": context_window_units,
            "max_line_length": max_line_length,
            "max_lines": max_lines,
        }
        current_job_signature = _job_signature(
            source_bytes,
            signature_settings,
            glossary_raw,
        )
        if st.session_state.active_job_signature != current_job_signature:
            _clear_result()
            st.session_state.active_job_signature = current_job_signature
        st.session_state.last_file_name = uploaded_file.name

        parse_warnings = list(document.warnings)
        if parse_warnings:
            st.warning(
                f"Parser reported {len(parse_warnings)} note(s). Open Review for details."
            )

        _status_strip(
            uploaded_file.name,
            format_label,
            f"{LANGUAGE_OPTIONS[source_lang]} → {LANGUAGE_OPTIONS[target_lang]}",
            provider_identity,
            fallback_label,
        )

        with st.container(border=True):
            _section_heading(
                "Translation plan",
                "A preflight view of what will be grouped, preserved, and sent to the provider.",
            )
            metric_cols = st.columns(4, gap="medium")
            metric_cols[0].metric("Format", format_label)
            metric_cols[1].metric("Translation units", f"{unit_count:,} {unit_label}")
            metric_cols[2].metric("Text to translate", f"{translatable_chars:,} chars")
            metric_cols[3].metric("Context windows", f"≈ {context_windows:,}")
            st.caption(structure_note)
            _status_strip(
                f"Up to {planned_char_limit} chars / window",
                f"Up to {context_window_units} {unit_label.split()[-1]} / window",
                f"{effective_batch} window{'s' if effective_batch != 1 else ''} / pipeline batch",
                "Checkpoint resume on",
            )
            if planned_char_limit < context_window_chars:
                st.info(
                    f"The provider limit reduces the requested {context_window_chars}-character "
                    f"window to {planned_char_limit} characters with safety headroom."
                )
            if backend == "indictrans2" and "dist-200M" in local_model_identity:
                st.info(
                    "Quality profile: the installed model is the faster distilled 200M "
                    "checkpoint. For maximum local quality, install the full 1B checkpoint "
                    "with `python scripts/download_models.py --model 1B`, then select its "
                    "printed model path. It needs more RAM and runs more slowly."
                )
            if requires_local_model:
                readiness_message = (
                    f"Local backup · {local_model_message}"
                    if backend == "sarvam-api"
                    else local_model_message
                )
                if local_model_ready:
                    st.success(readiness_message)
                else:
                    st.error(readiness_message)
                    if backend == "sarvam-api":
                        st.caption(
                            "Disable local backup to run strict Sarvam mode, or select a valid "
                            "IndicTrans2 checkpoint before translating."
                        )

        with st.container(border=True):
            action_col, detail_col = st.columns([0.28, 0.72], gap="large")
            with action_col:
                translate_clicked = st.button(
                    "Translate file",
                    type="primary",
                    use_container_width=True,
                    disabled=(
                        source_lang == target_lang
                        or unit_count == 0
                        or (requires_local_model and not local_model_ready)
                    ),
                )
            with detail_col:
                _section_heading(
                    "Ready to translate" if local_model_ready else "Model setup required",
                    (
                        "The source is translated in semantic windows, then restored to exact "
                        "file structure."
                        if local_model_ready
                        else local_model_message
                    ),
                )

        translator = None
        if translate_clicked:
            # A new attempt owns the output area even if provider or glossary
            # initialization fails; never leave an earlier result looking current.
            _clear_result()
            try:
                try:
                    glossary_cfg: GlossaryConfig = load_glossary_json(glossary_raw)
                except Exception as exc:
                    st.error(f"Invalid glossary JSON: {exc}")
                    st.stop()

                if backend == "sarvam-api" and sarvam_save_key and sarvam_api_key:
                    try:
                        save_sarvam_api_key(sarvam_api_key)
                    except CredentialStorageError as exc:
                        st.warning(str(exc))

                with st.spinner("Initializing translation provider..."):
                    translator = build_translator(
                        backend,
                        model_path=model_path,
                        sarvam_api_key=sarvam_api_key,
                        sarvam_model=sarvam_model,
                        sarvam_mode=sarvam_mode,
                        sarvam_fallback_backend=(
                            "indictrans2"
                            if backend == "sarvam-api" and sarvam_fallback_enabled
                            else None
                        ),
                    )
                    st.info(f"Active provider: {translator.display_name}")

                progress = st.progress(0.0, text="Preparing context windows...")
                checkpoint_path = make_translation_checkpoint_path(
                    uploaded_file.name,
                    source_bytes,
                )

                def on_progress(value: float, message: str) -> None:
                    progress.progress(value, text=message)

                if source_kind == "subtitle":
                    translated_document = translate_document(
                        document=document,
                        translator=translator,
                        settings=TranslationSettings(
                            source_lang=source_lang,
                            target_lang=target_lang,
                            chunk_size=chunk_size,
                            context_window_chars=context_window_chars,
                            context_window_cues=context_window_units,
                            max_line_length=max_line_length,
                            max_lines=max_lines,
                        ),
                        glossary=glossary_cfg,
                        progress_cb=on_progress,
                        checkpoint_path=checkpoint_path,
                    )
                    output_text = serialize_subtitle(translated_document)
                else:
                    translated_document = translate_text_document(
                        document=document,
                        translator=translator,
                        settings=DocumentTranslationSettings(
                            source_lang=source_lang,
                            target_lang=target_lang,
                            chunk_size=chunk_size,
                            context_window_chars=context_window_chars,
                            context_window_blocks=context_window_units,
                        ),
                        glossary=glossary_cfg,
                        progress_cb=on_progress,
                        checkpoint_path=checkpoint_path,
                    )
                    output_text = serialize_document(translated_document)

                result_warnings = list(translated_document.warnings)
                if _fallback_was_used(translator):
                    result_warnings.append(
                        "Local fallback was used for at least one provider batch. Review "
                        "mixed-provider phrasing."
                    )
                output_name = f"{Path(uploaded_file.name).stem}.translated{ext}"
                _store_result(
                    output_text,
                    output_name=output_name,
                    mime=_mime_type(ext),
                    provider=translator.usage_summary,
                    warnings=result_warnings,
                )
                st.success("Translation complete. Review the full output before export.")
                if _fallback_was_used(translator):
                    st.warning("Local fallback was used and has been added to Review notes.")

            except TranslationInterruptedError as exc:
                partial_text = serialize_subtitle(exc.partial_document)
                _store_result(
                    partial_text,
                    output_name=f"{Path(uploaded_file.name).stem}.partial{ext}",
                    mime=_mime_type(ext),
                    provider=translator.usage_summary if translator else provider_identity,
                    warnings=exc.partial_document.warnings,
                    partial=True,
                )
                st.error(f"Translation interrupted: {exc}")
                if exc.checkpoint_path is not None:
                    st.info(f"Checkpoint: {exc.checkpoint_path}")
                    st.info("Run the same job again to resume instead of starting over.")
            except DocumentTranslationInterruptedError as exc:
                partial_text = serialize_document(exc.partial_document)
                _store_result(
                    partial_text,
                    output_name=f"{Path(uploaded_file.name).stem}.partial{ext}",
                    mime=_mime_type(ext),
                    provider=translator.usage_summary if translator else provider_identity,
                    warnings=exc.partial_document.warnings,
                    partial=True,
                )
                st.error(f"Document translation interrupted: {exc}")
                if exc.checkpoint_path is not None:
                    st.info(f"Checkpoint: {exc.checkpoint_path}")
                    st.info("Run the same job again to resume instead of starting over.")
            except TranslatorInitError as exc:
                st.error(f"Translator initialization error: {exc}")
            except SarvamApiError as exc:
                st.error(f"Sarvam API error: {exc}")
                st.info(
                    "With fallback off, the run stops instead of silently changing providers. "
                    "Fix the API setup or explicitly enable local backup."
                )
            except FallbackTranslationError as exc:
                st.error(f"Primary and fallback translation failed: {exc}")
            except Exception as exc:
                st.error(f"Unexpected translation error: {exc}")

        with st.container(border=True):
            _section_heading(
                "Review & export",
                "Inspect the complete source and output—not a shortened six-unit preview.",
            )
            compare_tab, output_tab, review_tab = st.tabs(
                ["Side-by-side", "Output & export", "Review notes"]
            )
            with compare_tab:
                source_review_col, translated_review_col = st.columns(2, gap="large")
                with source_review_col:
                    st.text_area(
                        "Complete source",
                        value=content,
                        height=440,
                        disabled=True,
                    )
                with translated_review_col:
                    st.text_area(
                        "Complete translation",
                        value=(
                            st.session_state.translated_text
                            or "Run translation to populate the complete output."
                        ),
                        height=440,
                        disabled=True,
                    )
            with output_tab:
                if st.session_state.translated_text:
                    st.text_area(
                        "Translated file contents",
                        value=st.session_state.translated_text,
                        height=440,
                        disabled=True,
                    )
                    download_label = (
                        "Download partial output"
                        if st.session_state.result_is_partial
                        else "Download translated file"
                    )
                    st.download_button(
                        label=download_label,
                        data=st.session_state.translated_text.encode("utf-8"),
                        file_name=st.session_state.result_output_name,
                        mime=st.session_state.result_mime,
                        use_container_width=True,
                    )
                else:
                    st.info("Run translation to populate the full output and download file.")
            with review_tab:
                combined_warnings = list(
                    dict.fromkeys([*parse_warnings, *st.session_state.result_warnings])
                )
                if st.session_state.result_provider:
                    _status_strip(
                        f"Provider used: {st.session_state.result_provider}",
                        (
                            "Partial output"
                            if st.session_state.result_is_partial
                            else "Complete output"
                        ),
                    )
                if combined_warnings:
                    st.warning(f"{len(combined_warnings)} item(s) need review.")
                    st.code(
                        "\n".join(
                            f"{index}. {warning}"
                            for index, warning in enumerate(combined_warnings, 1)
                        )
                    )
                elif st.session_state.translated_text:
                    st.success(
                        "No parser, alignment, corruption, glossary, or grammar flags "
                        "were reported."
                    )
                else:
                    st.info("Quality and alignment flags will appear here after translation.")

    except (SubtitleParseError, DocumentParseError) as exc:
        st.error(f"Parsing error: {exc}")
    except UnicodeError as exc:
        st.error(f"Could not decode the uploaded text file: {exc}")
    except Exception as exc:
        st.error(f"Unexpected workspace error: {exc}")
