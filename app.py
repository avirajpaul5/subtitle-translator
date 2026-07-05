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
from subtitle_translator.pipeline import TranslationSettings, translate_document
from subtitle_translator.translators.factory import TranslatorInitError, build_translator


def _init_state() -> None:
    if "translated_text" not in st.session_state:
        st.session_state.translated_text = ""
    if "last_file_name" not in st.session_state:
        st.session_state.last_file_name = ""


st.set_page_config(page_title="Subtitle Translator", layout="wide")
_init_state()

st.title("📝 Subtitle Translator")
st.caption("Translate .srt/.vtt English subtitles to Bengali using local models or Sarvam API.")

with st.sidebar:
    st.header("Settings")
    backend = st.selectbox("Backend", ["indictrans2", "sarvam-api", "echo", "nllb"], index=0)
    model_path = st.text_input(
        "Local model path",
        value="./models/indictrans2-en-indic",
        help="Path to local model directory. Used by local backends and optional Sarvam fallback.",
    )

    sarvam_api_key = ""
    sarvam_model = "mayura:v1"
    sarvam_mode = "classic-colloquial"
    sarvam_fallback_enabled = True
    sarvam_save_key = False
    if backend == "sarvam-api":
        st.subheader("Sarvam API")
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
            "Fallback to local IndicTrans on Sarvam errors",
            value=True,
            help="Fallback is loaded only if the Sarvam request fails.",
        )

    source_lang = st.selectbox("Source language", ["en", "hi", "bn"], index=0)
    target_lang = st.selectbox("Target language", ["bn", "hi", "en"], index=0)

    st.subheader("Batch & Formatting")
    chunk_size = st.slider("Batch chunk size", 1, 64, 12)
    merge_min_chars = st.slider("Merge cues below chars", 10, 200, 60)
    max_line_length = st.slider("Max line length", 20, 60, 42)
    max_lines = st.slider("Max lines per cue", 1, 4, 2)

    st.subheader("Mode")
    echo_mode = st.checkbox("Echo/test mode (skip real translation)", value=False)

st.subheader("1) Upload subtitle")
uploaded_file = st.file_uploader("Drop a .srt or .vtt file", type=["srt", "vtt"])

st.subheader("2) Glossary & protected terms")
uploaded_glossary = st.file_uploader("Upload glossary JSON (optional)", type=["json"])
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

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Input preview")
            preview = "\n\n".join([cue.text for cue in document.cues[:6]])
            st.text_area("Original", value=preview, height=250, disabled=True)

        with col2:
            st.markdown("### Output preview")
            st.text_area(
                "Translated",
                value=st.session_state.translated_text or "Run translation to see preview",
                height=250,
                disabled=True,
            )

        if st.button("Translate", type="primary"):
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

            settings = TranslationSettings(
                source_lang=source_lang,
                target_lang=target_lang,
                chunk_size=chunk_size,
                merge_min_chars=merge_min_chars,
                max_line_length=max_line_length,
                max_lines=max_lines,
            )

            progress = st.progress(0.0, text="Starting...")

            def on_progress(value: float, message: str) -> None:
                progress.progress(value, text=message)

            translated_doc = translate_document(
                document=document,
                translator=translator,
                settings=settings,
                glossary=glossary_cfg,
                progress_cb=on_progress,
            )
            output_text = serialize_subtitle(translated_doc)
            st.session_state.translated_text = "\n\n".join([cue.text for cue in translated_doc.cues[:6]])

            output_name = f"{Path(uploaded_file.name).stem}.translated{ext}"
            st.success("Translation complete.")
            st.download_button(
                label="Download translated subtitle",
                data=output_text.encode("utf-8"),
                file_name=output_name,
                mime="text/plain",
            )

    except SubtitleParseError as exc:
        st.error(f"Parsing error: {exc}")
    except TranslatorInitError as exc:
        st.error(f"Translator initialization error: {exc}")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
else:
    st.info("Upload a subtitle file to begin.")
