from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from subtitle_translator.credentials import (
    CredentialStorageError,
    save_sarvam_api_key,
)
from subtitle_translator.defaults import DEFAULT_GLOSSARY
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
    if "translated_text" not in st.session_state:
        st.session_state.translated_text = ""
    if "last_file_name" not in st.session_state:
        st.session_state.last_file_name = ""


def _fallback_was_used(translator) -> bool:
    return int(getattr(translator, "fallback_count", 0) or 0) > 0


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

header[data-testid="stHeader"],
[data-testid="stToolbar"],
#MainMenu,
footer {
    visibility: hidden;
    height: 0;
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

div[data-baseweb="select"],
div[data-baseweb="select"] * {
    cursor: pointer !important;
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

div[data-testid="stFileUploader"] button,
div[data-testid="stFileUploader"] button * {
    background: var(--indicsub-card) !important;
    color: var(--indicsub-foreground) !important;
    -webkit-text-fill-color: var(--indicsub-foreground) !important;
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
    pills = "".join(f'<span class="status-pill">{item}</span>' for item in items if item)
    if pills:
        st.markdown(f'<div class="status-strip">{pills}</div>', unsafe_allow_html=True)


st.set_page_config(page_title="IndicSub", layout="wide", initial_sidebar_state="expanded")
_init_state()
_inject_theme()

st.markdown(
    """
    <section class="app-hero">
        <div>
            <p class="app-eyebrow">Subtitle translation studio</p>
            <h1>IndicSub</h1>
            <p>English subtitle files to Bengali with protected terms, glossary overrides, and resumable output.</p>
        </div>
        <div class="hero-badges">
            <span class="hero-badge">.srt / .vtt</span>
            <span class="hero-badge">Local-first</span>
            <span class="hero-badge">Bengali output</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Controls")
    with st.container(border=True):
        _section_heading("Provider", "Translation backend and model source.")
        backend = st.selectbox("Backend", ["indictrans2", "sarvam-api", "echo", "nllb"], index=0)
        model_path = st.text_input(
            "Local model path",
            value="./models/indictrans2-en-indic",
            help="Path to local model directory. Used by local backends and optional Sarvam fallback.",
        )

    sarvam_api_key = ""
    sarvam_model = "mayura:v1"
    sarvam_mode = "classic-colloquial"
    sarvam_fallback_enabled = False
    sarvam_save_key = False
    if backend == "sarvam-api":
        with st.container(border=True):
            _section_heading("Sarvam API", "Hosted model settings.")
            sarvam_api_key = st.text_input(
                "Sarvam API key",
                value="",
                type="password",
                help="Leave blank to use SARVAM_API_KEY or a key saved in the OS keychain.",
            )
            sarvam_save_key = st.checkbox("Save key in OS keychain", value=False)
            sarvam_model = st.selectbox(
                "Sarvam model",
                ["mayura:v1", "sarvam-translate:v1"],
                index=0,
                help="Mayura supports colloquial modes; Sarvam Translate supports formal translation.",
            )
            mode_options = ["classic-colloquial", "modern-colloquial", "formal"]
            sarvam_mode = st.selectbox(
                "Sarvam mode",
                mode_options,
                index=0,
                disabled=sarvam_model == "sarvam-translate:v1",
                help="Sarvam Translate always uses formal mode.",
            )
            sarvam_fallback_enabled = st.checkbox(
                "Use local IndicTrans backup if Sarvam fails",
                value=False,
                help=(
                    "Leave off to stop and show the Sarvam error. Enable only when "
                    "you want a backup output; fallback usage is flagged in the result."
                ),
            )
            if sarvam_fallback_enabled:
                st.warning(
                    "Backup fallback is ON. If Sarvam fails, the output may come from "
                    "local IndicTrans and will be flagged for review."
                )

    with st.container(border=True):
        _section_heading("Languages", "Source and target locale.")
        source_lang = st.selectbox("Source language", ["en", "hi", "bn"], index=0)
        target_lang = st.selectbox("Target language", ["bn", "hi", "en"], index=0)

    with st.container(border=True):
        _section_heading("Formatting", "Batching and subtitle line limits.")
        chunk_size = st.slider("Batch chunk size", 1, 64, 12)
        merge_min_chars = st.slider("Merge cues below chars", 10, 200, 60)
        max_line_length = st.slider("Max line length", 20, 60, 42)
        max_lines = st.slider("Max lines per cue", 1, 4, 2)

    with st.container(border=True):
        _section_heading("Run Mode")
        echo_mode = st.checkbox("Echo/test mode (skip real translation)", value=False)

input_col, glossary_col = st.columns([0.9, 1.1], gap="large")
with input_col:
    with st.container(border=True):
        _section_heading("Source Subtitle", "Upload the subtitle file to process.")
        uploaded_file = st.file_uploader("Subtitle file", type=["srt", "vtt"])

with glossary_col:
    with st.container(border=True):
        _section_heading("Glossary & Protected Terms", "Overrides and do-not-translate entries.")
        uploaded_glossary = st.file_uploader("Glossary JSON", type=["json"])
        if uploaded_glossary:
            glossary_raw = uploaded_glossary.getvalue().decode("utf-8")
        else:
            glossary_raw = json.dumps(DEFAULT_GLOSSARY, ensure_ascii=False, indent=2)

        glossary_raw = st.text_area("Glossary JSON", value=glossary_raw, height=220)

if uploaded_file:
    try:
        ext = Path(uploaded_file.name).suffix
        content = decode_subtitle_bytes(uploaded_file.getvalue())
        document = parse_subtitle(content, ext)

        if uploaded_file.name != st.session_state.last_file_name:
            st.session_state.translated_text = ""
            st.session_state.last_file_name = uploaded_file.name

        if document.warnings:
            st.warning(
                "Parser skipped {n} malformed block(s):\n- ".format(n=len(document.warnings))
                + "\n- ".join(document.warnings)
            )

        _status_strip(
            uploaded_file.name,
            f"{len(document.cues)} cues",
            f"{source_lang} -> {target_lang}",
            "Echo mode" if echo_mode else backend,
        )

        with st.container(border=True):
            _section_heading("Subtitle Preview", "Original cues and the latest translated preview.")
            col1, col2 = st.columns(2, gap="large")
            with col1:
                preview = "\n\n".join([cue.text for cue in document.cues[:6]])
                st.text_area("Original", value=preview, height=260, disabled=True)

            with col2:
                st.text_area(
                    "Translated",
                    value=st.session_state.translated_text or "Run translation to see preview",
                    height=260,
                    disabled=True,
                )

        with st.container(border=True):
            action_col, detail_col = st.columns([0.28, 0.72], gap="large")
            with action_col:
                translate_clicked = st.button(
                    "Translate",
                    type="primary",
                    use_container_width=True,
                )
            with detail_col:
                _status_strip(
                    f"Chunk {chunk_size}",
                    f"Merge under {merge_min_chars}",
                    f"{max_line_length} chars",
                    f"{max_lines} lines",
                )

        if translate_clicked:
            try:
                glossary_cfg: GlossaryConfig = load_glossary_json(glossary_raw)
            except Exception as exc:
                st.error(f"Invalid glossary JSON: {exc}")
                st.stop()

            with st.spinner("Initializing translator..."):
                selected_backend = "echo" if echo_mode else backend
                if selected_backend == "sarvam-api" and sarvam_save_key and sarvam_api_key:
                    try:
                        save_sarvam_api_key(sarvam_api_key)
                    except CredentialStorageError as exc:
                        st.warning(str(exc))

                translator = build_translator(
                    selected_backend,
                    model_path=model_path,
                    sarvam_api_key=sarvam_api_key,
                    sarvam_model=sarvam_model,
                    sarvam_mode=sarvam_mode,
                    sarvam_fallback_backend=(
                        "indictrans2"
                        if selected_backend == "sarvam-api" and sarvam_fallback_enabled
                        else None
                    ),
                )
                st.info(f"Translation provider: {translator.display_name}")

            settings = TranslationSettings(
                source_lang=source_lang,
                target_lang=target_lang,
                chunk_size=chunk_size,
                merge_min_chars=merge_min_chars,
                max_line_length=max_line_length,
                max_lines=max_lines,
            )

            progress = st.progress(0.0, text="Starting...")
            checkpoint_path = make_translation_checkpoint_path(
                uploaded_file.name,
                uploaded_file.getvalue(),
            )

            def on_progress(value: float, message: str) -> None:
                progress.progress(value, text=message)

            translated_doc = translate_document(
                document=document,
                translator=translator,
                settings=settings,
                glossary=glossary_cfg,
                progress_cb=on_progress,
                checkpoint_path=checkpoint_path,
            )
            output_text = serialize_subtitle(translated_doc)
            st.session_state.translated_text = "\n\n".join([cue.text for cue in translated_doc.cues[:6]])

            output_name = f"{Path(uploaded_file.name).stem}.translated{ext}"
            st.success("Translation complete.")
            st.info(f"Provider used: {translator.usage_summary}")
            if _fallback_was_used(translator):
                st.warning(
                    "Fallback was used. This output was not fully produced by Sarvam; "
                    "review the notes below before trusting the file."
                )
            if translated_doc.warnings:
                st.warning(
                    "Review notes:\n- " + "\n- ".join(translated_doc.warnings[:10])
                    + ("\n- ..." if len(translated_doc.warnings) > 10 else "")
                )
            st.download_button(
                label="Download translated subtitle",
                data=output_text.encode("utf-8"),
                file_name=output_name,
                mime="text/plain",
                use_container_width=True,
            )

    except SubtitleParseError as exc:
        st.error(f"Parsing error: {exc}")
    except TranslatorInitError as exc:
        st.error(f"Translator initialization error: {exc}")
    except TranslationInterruptedError as exc:
        partial_text = serialize_subtitle(exc.partial_document)
        st.session_state.translated_text = "\n\n".join(
            [cue.text for cue in exc.partial_document.cues[:6]]
        )
        st.error(f"Translation interrupted: {exc}")
        if exc.checkpoint_path is not None:
            st.info(f"Progress checkpoint saved at: {exc.checkpoint_path}")
            st.info("Rerun with the same file, settings, glossary, and provider to resume.")
        st.download_button(
            label="Download partial translated subtitle",
            data=partial_text.encode("utf-8"),
            file_name=f"{Path(uploaded_file.name).stem}.partial{ext}",
            mime="text/plain",
            use_container_width=True,
        )
    except SarvamApiError as exc:
        st.error(f"Sarvam API error: {exc}")
        st.info(
            "Backup fallback is off, so translation stopped. Fix the Sarvam key/model/"
            "language issue, or enable the local IndicTrans backup and rerun."
        )
    except FallbackTranslationError as exc:
        st.error(f"Translation failed: {exc}")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
else:
    with st.container(border=True):
        _section_heading("Workspace")
        st.info("Upload a subtitle file to begin.")
