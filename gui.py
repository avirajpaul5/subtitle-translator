"""Native desktop GUI for IndicSub (PySide6).

Run with: python gui.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from subtitle_translator.credentials import (
    CredentialStorageError,
    save_sarvam_api_key,
)
from subtitle_translator.defaults import DEFAULT_GLOSSARY
from subtitle_translator.glossary import GlossaryConfig, load_glossary_json
from subtitle_translator.models import SubtitleDocument
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
from subtitle_translator.speaker_detection import detect_speaker_names
from subtitle_translator.translators.factory import TranslatorInitError, build_translator
from subtitle_translator.translators.fallback import FallbackTranslationError
from subtitle_translator.translators.sarvam_api import SarvamApiError

# Lazy import — auto_dnt pulls spaCy which is heavy. We import on demand.
def _load_auto_dnt():
    from subtitle_translator.auto_dnt import detect_preserve_spans
    return detect_preserve_spans

SUBTITLE_FILTER = "Subtitle files (*.srt *.vtt);;SRT (*.srt);;WebVTT (*.vtt);;All files (*)"

LANGUAGE_CODES = [
    "en",
    "as",
    "bn",
    "brx",
    "doi",
    "gu",
    "hi",
    "kn",
    "kok",
    "ks",
    "mai",
    "ml",
    "mni",
    "mr",
    "ne",
    "or",
    "pa",
    "sa",
    "sat",
    "sd",
    "ta",
    "te",
    "ur",
]
INDIC_LANGUAGE_CODES = [code for code in LANGUAGE_CODES if code != "en"]
SARVAM_MAYURA_CODES = [
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
]
LOCAL_MODEL_DEFAULTS = {
    "indictrans2": "./models/indictrans2-en-indic",
    "nllb": "./models/nllb-200-distilled-600M",
}


def _local_model_status(
    model_path: str,
    expected_backend: str,
) -> tuple[bool, str, str]:
    """Return readiness, checkpoint identity, and a user-facing status."""

    path = Path(model_path).expanduser()
    identity = path.name or model_path
    backend_label = "IndicTrans2" if expected_backend == "indictrans2" else "NLLB"
    if not path.is_dir():
        return False, identity, f"Local model directory was not found: {model_path}"

    config_path = path / "config.json"
    if not config_path.is_file():
        return False, identity, f"{backend_label} model config is missing: {config_path}"
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
    architecture_text = (
        " ".join(str(item) for item in architectures)
        if isinstance(architectures, list)
        else str(architectures or "")
    )
    evidence = " ".join(
        (identity, str(config.get("model_type") or ""), architecture_text)
    ).lower()
    expected_marker = "indictrans" if expected_backend == "indictrans2" else "nllb"
    if expected_marker not in evidence:
        return (
            False,
            identity,
            f"Model type mismatch: {identity} is not an identifiable {backend_label} checkpoint.",
        )
    return True, identity, f"{backend_label} checkpoint is ready: {identity}"

APP_STYLESHEET = """
QWidget {
    color: #09090b;
    font-family: "SF Pro Text", "Inter", "Segoe UI", sans-serif;
    font-size: 13px;
}

QMainWindow,
QWidget#AppRoot {
    background: #fafafa;
}

QFrame#HeaderCard,
QFrame#ActionBar {
    background: #ffffff;
    border: 1px solid #e4e4e7;
    border-radius: 10px;
}

QLabel#Eyebrow {
    color: #71717a;
    font-size: 11px;
    font-weight: 700;
}

QLabel#AppTitle {
    color: #09090b;
    font-size: 28px;
    font-weight: 800;
}

QLabel#Subtitle {
    color: #71717a;
    font-size: 13px;
}

QLabel#StatusPill {
    background: #f4f4f5;
    border: 1px solid #e4e4e7;
    border-radius: 999px;
    color: #18181b;
    font-size: 12px;
    font-weight: 600;
    padding: 6px 10px;
}

QGroupBox {
    background: #ffffff;
    border: 1px solid #e4e4e7;
    border-radius: 10px;
    font-weight: 700;
    margin-top: 18px;
    padding: 20px 14px 14px 14px;
}

QGroupBox::title {
    background: #ffffff;
    color: #09090b;
    left: 12px;
    padding: 0 6px;
    subcontrol-origin: margin;
    top: 2px;
}

QLabel {
    color: #3f3f46;
}

QLineEdit,
QPlainTextEdit,
QComboBox,
QSpinBox {
    background: #ffffff;
    border: 1px solid #e4e4e7;
    border-radius: 8px;
    color: #09090b;
    padding: 6px 8px;
    selection-background-color: #18181b;
    selection-color: #fafafa;
}

QPlainTextEdit {
    font-family: "SF Mono", "Menlo", "Consolas", monospace;
    font-size: 12px;
}

QLineEdit:focus,
QPlainTextEdit:focus,
QComboBox:focus,
QSpinBox:focus {
    border-color: #a1a1aa;
}

QComboBox {
    padding-right: 24px;
}

QComboBox::drop-down,
QSpinBox::up-button,
QSpinBox::down-button {
    border: 0;
    width: 22px;
}

QPushButton {
    background: #ffffff;
    border: 1px solid #e4e4e7;
    border-radius: 8px;
    color: #18181b;
    font-weight: 600;
    min-height: 26px;
    padding: 7px 12px;
}

QPushButton:hover {
    background: #f4f4f5;
    border-color: #d4d4d8;
}

QPushButton:pressed {
    background: #e4e4e7;
}

QPushButton:disabled {
    background: #f4f4f5;
    color: #a1a1aa;
}

QPushButton[variant="primary"] {
    background: #18181b;
    border-color: #18181b;
    color: #fafafa;
}

QPushButton[variant="primary"]:hover {
    background: #27272a;
    border-color: #27272a;
}

QPushButton[variant="primary"]:disabled {
    background: #a1a1aa;
    border-color: #a1a1aa;
    color: #fafafa;
}

QCheckBox {
    color: #3f3f46;
    spacing: 8px;
}

QCheckBox::indicator {
    background: #ffffff;
    border: 1px solid #d4d4d8;
    border-radius: 4px;
    height: 14px;
    width: 14px;
}

QCheckBox::indicator:checked {
    background: #18181b;
    border-color: #18181b;
}

QProgressBar {
    background: #f4f4f5;
    border: 1px solid #e4e4e7;
    border-radius: 7px;
    color: transparent;
    height: 13px;
    text-align: center;
}

QProgressBar::chunk {
    background: #18181b;
    border-radius: 6px;
}

QSplitter::handle {
    background: #e4e4e7;
    margin: 10px 3px;
    width: 1px;
}

QStatusBar {
    background: #ffffff;
    border-top: 1px solid #e4e4e7;
    color: #71717a;
}

QToolTip {
    background: #18181b;
    border: 1px solid #27272a;
    border-radius: 6px;
    color: #fafafa;
    padding: 6px;
}
"""


def _format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s"


class TranslateWorker(QObject):
    progress = Signal(float, str)
    model_loaded = Signal(str)   # provider label once translator is ready
    finished = Signal(str, list, str)  # serialized output, validation warnings, provider summary
    interrupted = Signal(str, str, list)  # message, serialized partial output, warnings
    failed = Signal(str)

    def __init__(
        self,
        document: SubtitleDocument,
        backend: str,
        model_path: str,
        sarvam_api_key: str,
        sarvam_model: str,
        sarvam_mode: str,
        sarvam_fallback_backend: str | None,
        checkpoint_path: str | None,
        settings: TranslationSettings,
        glossary: GlossaryConfig,
    ) -> None:
        super().__init__()
        self._document = document
        self._backend = backend
        self._model_path = model_path
        self._sarvam_api_key = sarvam_api_key
        self._sarvam_model = sarvam_model
        self._sarvam_mode = sarvam_mode
        self._sarvam_fallback_backend = sarvam_fallback_backend
        self._checkpoint_path = checkpoint_path
        self._settings = settings
        self._glossary = glossary

    @Slot()
    def run(self) -> None:
        try:
            translator = build_translator(
                self._backend,
                model_path=self._model_path,
                sarvam_api_key=self._sarvam_api_key,
                sarvam_model=self._sarvam_model,
                sarvam_mode=self._sarvam_mode,
                sarvam_fallback_backend=self._sarvam_fallback_backend,
            )
            self.model_loaded.emit(translator.display_name)
            translated = translate_document(
                document=self._document,
                translator=translator,
                settings=self._settings,
                glossary=self._glossary,
                progress_cb=lambda v, m: self.progress.emit(v, m),
                checkpoint_path=self._checkpoint_path,
            )
            self.finished.emit(
                serialize_subtitle(translated),
                list(translated.warnings),
                translator.usage_summary,
            )
        except (SubtitleParseError, TranslatorInitError) as exc:
            self.failed.emit(str(exc))
        except TranslationInterruptedError as exc:
            self.interrupted.emit(
                str(exc),
                serialize_subtitle(exc.partial_document),
                list(exc.partial_document.warnings),
            )
        except SarvamApiError as exc:
            message = f"Sarvam API error: {exc}"
            if self._backend == "sarvam-api" and self._sarvam_fallback_backend is None:
                message += (
                    "\n\nBackup fallback is off, so translation stopped. Fix the "
                    "Sarvam key/model/language issue, or enable local IndicTrans "
                    "backup and rerun."
                )
            self.failed.emit(message)
        except FallbackTranslationError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Unexpected error: {exc}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("IndicSub")
        self.resize(1180, 820)

        self._document: SubtitleDocument | None = None
        self._source_path: Path | None = None
        self._translated_output: str = ""
        self._thread: QThread | None = None
        self._worker: TranslateWorker | None = None
        self._translate_start_time: float | None = None
        self._translation_in_progress = False
        self._active_backend = "indictrans2"
        self._model_paths = dict(LOCAL_MODEL_DEFAULTS)

        self._build_ui()
        self.statusBar().showMessage("Open a .srt or .vtt file to begin.")

    @staticmethod
    def _button(text: str, variant: str = "secondary") -> QPushButton:
        button = QPushButton(text)
        button.setProperty("variant", variant)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    @staticmethod
    def _combo(items: list[str]) -> QComboBox:
        combo = QComboBox()
        combo.addItems(items)
        combo.setCursor(Qt.CursorShape.PointingHandCursor)
        combo.view().setCursor(Qt.CursorShape.PointingHandCursor)
        return combo

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("AppRoot")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 18, 18, 14)
        root.setSpacing(14)

        header = QFrame()
        header.setObjectName("HeaderCard")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 16, 18, 16)
        header_layout.setSpacing(16)

        title_stack = QVBoxLayout()
        title_stack.setSpacing(3)
        eyebrow = QLabel("SUBTITLE TRANSLATION STUDIO")
        eyebrow.setObjectName("Eyebrow")
        title = QLabel("IndicSub")
        title.setObjectName("AppTitle")
        subtitle = QLabel(
            "English subtitle files to Bengali with protected terms, glossary overrides, "
            "and resumable output."
        )
        subtitle.setObjectName("Subtitle")
        subtitle.setWordWrap(True)
        title_stack.addWidget(eyebrow)
        title_stack.addWidget(title)
        title_stack.addWidget(subtitle)

        self._file_label = QLabel("No file loaded")
        self._file_label.setObjectName("StatusPill")
        self._file_label.setAlignment(Qt.AlignCenter)
        self._open_btn = self._button("Open subtitle...")
        self._open_btn.clicked.connect(self._on_open)

        header_layout.addLayout(title_stack, 1)
        header_layout.addWidget(self._file_label)
        header_layout.addWidget(self._open_btn)
        root.addWidget(header)

        settings_glossary_row = QHBoxLayout()
        settings_glossary_row.setSpacing(14)
        settings_glossary_row.addWidget(self._build_settings_group(), 1)
        settings_glossary_row.addWidget(self._build_glossary_group(), 1)
        root.addLayout(settings_glossary_row)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        original_group = QGroupBox("Original Subtitle")
        original_layout = QVBoxLayout(original_group)
        original_layout.setContentsMargins(10, 12, 10, 10)
        self._original_view = QPlainTextEdit()
        self._original_view.setReadOnly(True)
        self._original_view.setPlaceholderText("Original subtitle content will appear here.")
        original_layout.addWidget(self._original_view)

        translated_group = QGroupBox("Translated Subtitle")
        translated_layout = QVBoxLayout(translated_group)
        translated_layout.setContentsMargins(10, 12, 10, 10)
        self._translated_view = QPlainTextEdit()
        self._translated_view.setReadOnly(True)
        self._translated_view.setPlaceholderText("Translated subtitle content will appear here.")
        translated_layout.addWidget(self._translated_view)

        splitter.addWidget(original_group)
        splitter.addWidget(translated_group)
        splitter.setSizes([580, 580])
        root.addWidget(splitter, 1)

        review_group = QGroupBox("Review")
        review_layout = QVBoxLayout(review_group)
        review_layout.setContentsMargins(10, 12, 10, 10)
        review_layout.setSpacing(6)
        self._review_summary = QLabel("No translation run yet.")
        self._review_summary.setWordWrap(True)
        self._review_details = QPlainTextEdit()
        self._review_details.setReadOnly(True)
        self._review_details.setMaximumHeight(92)
        self._review_details.setPlaceholderText(
            "Provider, completion state, fallback use, and flagged cues will appear here."
        )
        review_layout.addWidget(self._review_summary)
        review_layout.addWidget(self._review_details)
        root.addWidget(review_group)

        action_shell = QFrame()
        action_shell.setObjectName("ActionBar")
        action_bar = QHBoxLayout(action_shell)
        action_bar.setContentsMargins(14, 12, 14, 12)
        action_bar.setSpacing(12)
        self._translate_btn = self._button("Translate", "primary")
        self._translate_btn.setEnabled(False)
        self._translate_btn.clicked.connect(self._on_translate)
        self._save_btn = self._button("Save translated...")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        action_bar.addWidget(self._translate_btn)
        action_bar.addWidget(self._progress, 1)
        action_bar.addWidget(self._save_btn)
        root.addWidget(action_shell)

    def _build_settings_group(self) -> QGroupBox:
        group = QGroupBox("Settings")
        form = QFormLayout(group)
        form.setContentsMargins(12, 14, 12, 12)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setHorizontalSpacing(14)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setVerticalSpacing(10)

        self._backend_combo = self._combo(["indictrans2", "sarvam-api", "echo", "nllb"])
        form.addRow("Backend", self._backend_combo)

        model_row = QHBoxLayout()
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.setSpacing(8)
        self._model_path_edit = QLineEdit(LOCAL_MODEL_DEFAULTS["indictrans2"])
        self._model_path_edit.textChanged.connect(self._refresh_model_status)
        self._model_browse_btn = self._button("Browse")
        self._model_browse_btn.clicked.connect(self._on_browse_model)
        model_row.addWidget(self._model_path_edit, 1)
        model_row.addWidget(self._model_browse_btn)
        self._model_container = QWidget()
        self._model_container.setLayout(model_row)
        form.addRow("Model path", self._model_container)

        self._model_status_label = QLabel()
        self._model_status_label.setWordWrap(True)
        form.addRow("Model status", self._model_status_label)

        self._sarvam_key_edit = QLineEdit()
        self._sarvam_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._sarvam_key_edit.setPlaceholderText(
            "Leave blank to use SARVAM_API_KEY or OS keychain"
        )
        form.addRow("Sarvam API key", self._sarvam_key_edit)

        self._save_sarvam_key_check = QCheckBox("Save Sarvam key in OS keychain")
        form.addRow(self._save_sarvam_key_check)

        self._sarvam_model_combo = self._combo(["mayura:v1", "sarvam-translate:v1"])
        self._sarvam_model_combo.currentTextChanged.connect(self._on_sarvam_model_changed)
        form.addRow("Sarvam model", self._sarvam_model_combo)

        self._sarvam_mode_combo = self._combo(
            ["classic-colloquial", "modern-colloquial", "formal"]
        )
        form.addRow("Sarvam mode", self._sarvam_mode_combo)

        self._sarvam_fallback_check = QCheckBox("Use local IndicTrans backup if Sarvam fails")
        self._sarvam_fallback_check.setChecked(False)
        self._sarvam_fallback_check.setToolTip(
            "Leave off to stop and show the Sarvam error. Enable only when you "
            "want a backup output; fallback usage is flagged in the result."
        )
        self._sarvam_fallback_check.toggled.connect(
            lambda _: self._refresh_backend_controls()
        )
        form.addRow(self._sarvam_fallback_check)

        self._source_combo = self._combo(["en"])
        self._target_combo = self._combo(INDIC_LANGUAGE_CODES)
        self._target_combo.setCurrentText("bn")
        self._source_combo.currentTextChanged.connect(self._refresh_translate_button)
        self._target_combo.currentTextChanged.connect(self._refresh_translate_button)
        form.addRow("Source language", self._source_combo)
        form.addRow("Target language", self._target_combo)

        self._chunk_size_spin = self._spin(1, 64, 12)
        self._context_chars_spin = self._spin(100, 2000, 700)
        self._context_chars_spin.setToolTip(
            "Maximum source characters grouped into one context-aware model input."
        )
        self._context_cap_label = QLabel()
        self._context_cap_label.setWordWrap(True)
        self._context_cues_spin = self._spin(1, 32, 8)
        self._context_cues_spin.setToolTip(
            "Maximum subtitle cues translated together while preserving exact cue boundaries."
        )
        self._max_line_spin = self._spin(20, 80, 42)
        self._max_lines_spin = self._spin(1, 4, 2)
        form.addRow("Batch chunk size", self._chunk_size_spin)
        form.addRow("Context window chars", self._context_chars_spin)
        form.addRow("Effective context cap", self._context_cap_label)
        form.addRow("Cues per context window", self._context_cues_spin)
        form.addRow("Max line length", self._max_line_spin)
        form.addRow("Max lines per cue", self._max_lines_spin)

        self._backend_combo.currentTextChanged.connect(self._on_backend_changed)
        self._on_backend_changed(self._backend_combo.currentText())

        return group

    @Slot(str)
    def _on_backend_changed(self, backend: str) -> None:
        previous_backend = self._active_backend
        if previous_backend in {"indictrans2", "nllb", "sarvam-api"}:
            previous_profile = (
                "nllb" if previous_backend == "nllb" else "indictrans2"
            )
            self._model_paths[previous_profile] = self._model_path_edit.text().strip()

        self._active_backend = backend
        model_profile = "nllb" if backend == "nllb" else "indictrans2"
        desired_path = self._model_paths[model_profile]
        if self._model_path_edit.text() != desired_path:
            self._model_path_edit.setText(desired_path)
        self._refresh_backend_controls()

    @Slot(str)
    def _on_sarvam_model_changed(self, model: str) -> None:
        if model == "sarvam-translate:v1":
            self._sarvam_mode_combo.setCurrentText("formal")
        self._refresh_backend_controls()

    def _refresh_backend_controls(self) -> None:
        backend = self._backend_combo.currentText()
        is_sarvam = backend == "sarvam-api"
        has_local_backup = is_sarvam and self._sarvam_fallback_check.isChecked()
        needs_local_model = backend in {"indictrans2", "nllb"} or has_local_backup

        self._sarvam_key_edit.setEnabled(is_sarvam)
        self._save_sarvam_key_check.setEnabled(is_sarvam)
        self._sarvam_model_combo.setEnabled(is_sarvam)
        self._sarvam_fallback_check.setEnabled(is_sarvam)
        self._sarvam_mode_combo.setEnabled(
            is_sarvam and self._sarvam_model_combo.currentText() == "mayura:v1"
        )
        self._model_path_edit.setEnabled(needs_local_model)
        self._model_browse_btn.setEnabled(needs_local_model)

        if backend in {"indictrans2", "nllb"} or has_local_backup:
            context_cap = 450
            cap_note = "local tokenizer limit with safety headroom"
        elif is_sarvam and self._sarvam_model_combo.currentText() == "mayura:v1":
            context_cap = 900
            cap_note = "Mayura provider limit with safety headroom"
        elif is_sarvam:
            context_cap = 1800
            cap_note = "Sarvam Translate limit with safety headroom"
        else:
            context_cap = 2000
            cap_note = "structure-test UI limit"
        self._context_chars_spin.setMaximum(context_cap)
        if self._context_chars_spin.value() > context_cap:
            self._context_chars_spin.setValue(context_cap)
        self._context_chars_spin.setToolTip(
            f"Maximum source characters per context input: {context_cap} ({cap_note})."
        )
        self._context_cap_label.setText(f"{context_cap} characters · {cap_note}")

        current_source = self._source_combo.currentText()
        current_target = self._target_combo.currentText()
        if backend == "indictrans2":
            source_options = ["en"]
            target_options = INDIC_LANGUAGE_CODES
        elif backend == "nllb":
            source_options = ["en"]
            target_options = ["bn"]
        elif backend == "sarvam-api":
            primary_codes = (
                SARVAM_MAYURA_CODES
                if self._sarvam_model_combo.currentText() == "mayura:v1"
                else LANGUAGE_CODES
            )
            if has_local_backup:
                source_options = ["en"]
                target_options = [
                    code
                    for code in primary_codes
                    if code != "en" and code in INDIC_LANGUAGE_CODES
                ]
            else:
                source_options = list(primary_codes)
                target_options = list(primary_codes)
        else:
            source_options = LANGUAGE_CODES
            target_options = LANGUAGE_CODES

        self._replace_combo_options(
            self._source_combo,
            source_options,
            preferred=current_source if current_source in source_options else "en",
        )
        self._replace_combo_options(
            self._target_combo,
            target_options,
            preferred=current_target if current_target in target_options else "bn",
        )
        if self._source_combo.currentText() == self._target_combo.currentText():
            alternative = next(
                (
                    code
                    for code in target_options
                    if code != self._source_combo.currentText()
                ),
                "",
            )
            if alternative:
                self._target_combo.setCurrentText(alternative)
        self._refresh_model_status()

    @staticmethod
    def _replace_combo_options(
        combo: QComboBox,
        options: list[str],
        *,
        preferred: str,
    ) -> None:
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(options)
        if preferred in options:
            combo.setCurrentText(preferred)
        combo.blockSignals(False)

    def _refresh_model_status(self, _text: str | None = None) -> None:
        backend = self._backend_combo.currentText()
        needs_local_model = backend in {"indictrans2", "nllb"} or (
            backend == "sarvam-api" and self._sarvam_fallback_check.isChecked()
        )
        if not needs_local_model:
            self._model_status_label.setText("Not used by the selected provider configuration.")
            self._model_status_label.setStyleSheet("color: #71717a;")
            self._refresh_translate_button()
            return

        expected_backend = "nllb" if backend == "nllb" else "indictrans2"
        ready, identity, message = _local_model_status(
            self._model_path_edit.text().strip(), expected_backend
        )
        backend_label = "NLLB" if expected_backend == "nllb" else "IndicTrans2"
        if ready:
            prefix = "Backup ready" if backend == "sarvam-api" else "Ready"
            status_text = f"{prefix} · {backend_label} · {identity}"
        else:
            prefix = "Backup not ready" if backend == "sarvam-api" else "Not ready"
            status_text = f"{prefix} · {message}"
        self._model_status_label.setText(status_text)
        self._model_status_label.setStyleSheet(
            "color: #3f3f46;" if ready else "color: #52525b; font-weight: 600;"
        )
        self._refresh_translate_button()

    def _refresh_translate_button(self, _value: str | None = None) -> None:
        if not hasattr(self, "_translate_btn"):
            return
        backend = self._backend_combo.currentText()
        model_ready = True
        if backend in {"indictrans2", "nllb"} or (
            backend == "sarvam-api" and self._sarvam_fallback_check.isChecked()
        ):
            expected_backend = "nllb" if backend == "nllb" else "indictrans2"
            model_ready = _local_model_status(
                self._model_path_edit.text().strip(), expected_backend
            )[0]
        self._translate_btn.setEnabled(
            self._document is not None
            and not self._translation_in_progress
            and model_ready
            and self._source_combo.currentText() != self._target_combo.currentText()
        )

    def _build_glossary_group(self) -> QGroupBox:
        group = QGroupBox("Glossary JSON")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 14, 12, 12)
        layout.setSpacing(10)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(8)
        load_btn = self._button("Load JSON...")
        load_btn.clicked.connect(self._on_load_glossary)
        save_btn = self._button("Save JSON...")
        save_btn.clicked.connect(self._on_save_glossary)
        btn_row.addWidget(load_btn)
        btn_row.addWidget(save_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self._glossary_edit = QPlainTextEdit(
            json.dumps(DEFAULT_GLOSSARY, ensure_ascii=False, indent=2)
        )
        layout.addWidget(self._glossary_edit, 1)
        return group

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _set_review(self, summary: str, details: list[str] | tuple[str, ...] = ()) -> None:
        self._review_summary.setText(summary)
        self._review_details.setPlainText("\n".join(str(item) for item in details if item))

    def _selected_provider_label(self) -> str:
        backend = self._backend_combo.currentText()
        if backend == "sarvam-api":
            return f"Sarvam API · {self._sarvam_model_combo.currentText()}"
        if backend == "indictrans2":
            return "IndicTrans2 · local"
        if backend == "nllb":
            return "NLLB · local"
        return "Echo · structure test"

    @Slot()
    def _on_open(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Open subtitle", "", SUBTITLE_FILTER)
        if not path_str:
            return
        path = Path(path_str)
        try:
            text = decode_subtitle_bytes(path.read_bytes())
            document = parse_subtitle(text, path.suffix)
        except SubtitleParseError as exc:
            QMessageBox.critical(self, "Parse error", str(exc))
            return
        except OSError as exc:
            QMessageBox.critical(self, "Read error", str(exc))
            return

        self._document = document
        self._source_path = path
        self._translated_output = ""
        self._translated_view.clear()
        self._set_review(
            "Source loaded · no translation run yet.",
            list(document.warnings),
        )
        self._progress.setValue(0)
        self._save_btn.setEnabled(False)
        self._refresh_translate_button()
        self._file_label.setText(f"{path.name} | {len(document.cues)} cues")
        self._original_view.setPlainText(serialize_subtitle(document))

        # Two-pass detection: speaker labels (regex, cheap) + linguistic
        # detection (spaCy NER + POS + wordfreq). The latter is slow on first
        # use because spaCy lazy-loads its model, so we keep its failures
        # non-fatal — auto-detection is best-effort, not load-bearing.
        detected_names = detect_speaker_names(document)
        auto_terms: list[str] = []
        try:
            detect_preserve_spans = _load_auto_dnt()
            auto_terms = detect_preserve_spans(document)
        except Exception as exc:  # noqa: BLE001 — surface but don't break
            self.statusBar().showMessage(
                f"Auto-detection unavailable: {exc}", 6000
            )

        all_detected = list(dict.fromkeys([*detected_names, *auto_terms]))
        if all_detected:
            self._merge_detected_names(
                all_detected,
                speaker_count=len(detected_names),
                auto_count=len(auto_terms),
            )

        if document.warnings:
            self.statusBar().showMessage(
                f"Loaded with {len(document.warnings)} warning(s). See details on hover.",
                10000,
            )
            self.statusBar().setToolTip("\n".join(document.warnings))
        else:
            self.statusBar().showMessage(f"Loaded {len(document.cues)} cues.", 5000)
            self.statusBar().setToolTip("")

    @Slot()
    def _on_browse_model(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select model directory")
        if path:
            self._model_path_edit.setText(path)

    @Slot()
    def _on_load_glossary(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Load glossary JSON", "", "JSON (*.json);;All files (*)"
        )
        if not path_str:
            return
        try:
            self._glossary_edit.setPlainText(Path(path_str).read_text(encoding="utf-8"))
        except OSError as exc:
            QMessageBox.critical(self, "Read error", str(exc))

    @Slot()
    def _on_save_glossary(self) -> None:
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save glossary JSON", "glossary.json", "JSON (*.json)"
        )
        if not path_str:
            return
        try:
            Path(path_str).write_text(self._glossary_edit.toPlainText(), encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Write error", str(exc))

    def _merge_detected_names(
        self,
        names: list[str],
        speaker_count: int = 0,
        auto_count: int = 0,
    ) -> None:
        try:
            existing = json.loads(self._glossary_edit.toPlainText())
        except Exception:
            existing = {}
        if not isinstance(existing, dict):
            existing = {}
        dnt: list = existing.get("do_not_translate", [])
        if not isinstance(dnt, list):
            dnt = []
        added = [n for n in names if n not in dnt]
        dnt.extend(added)
        existing["do_not_translate"] = dnt
        self._glossary_edit.setPlainText(json.dumps(existing, ensure_ascii=False, indent=2))
        if added:
            if speaker_count and auto_count:
                breakdown = f"{speaker_count} speaker label(s) + {auto_count} name/foreign-word(s)"
            elif speaker_count:
                breakdown = f"{speaker_count} speaker label(s)"
            else:
                breakdown = f"{auto_count} name/foreign-word(s)"
            preview = ", ".join(added[:5]) + (" …" if len(added) > 5 else "")
            self.statusBar().showMessage(
                f"Auto-detected {len(added)} term(s) — {breakdown}: {preview}",
                8000,
            )

    @Slot()
    def _on_translate(self) -> None:
        if self._document is None:
            return

        # A new attempt owns the translated pane even when validation or model
        # initialization fails, so an older result can never look current.
        self._translated_output = ""
        self._translated_view.clear()
        self._save_btn.setEnabled(False)
        self._set_review("Translation attempt started · preflight in progress.")
        try:
            glossary = load_glossary_json(self._glossary_edit.toPlainText())
        except Exception as exc:
            self._set_review("Translation did not start · invalid glossary.", [str(exc)])
            QMessageBox.critical(self, "Invalid glossary JSON", str(exc))
            return

        backend = self._backend_combo.currentText()
        sarvam_api_key = self._sarvam_key_edit.text()
        sarvam_fallback_backend = (
            "indictrans2"
            if backend == "sarvam-api" and self._sarvam_fallback_check.isChecked()
            else None
        )
        source_lang = self._source_combo.currentText()
        target_lang = self._target_combo.currentText()
        if source_lang == target_lang:
            QMessageBox.warning(
                self,
                "Choose a target language",
                "Source and target languages must be different.",
            )
            return
        if (backend == "indictrans2" or sarvam_fallback_backend) and (
            source_lang != "en" or target_lang == "en"
        ):
            QMessageBox.warning(
                self,
                "Unsupported local route",
                "This installed IndicTrans model translates English to Indic languages. "
                "Choose English as the source and an Indic language as the target.",
            )
            return
        if backend == "nllb" and (source_lang, target_lang) != ("en", "bn"):
            QMessageBox.warning(
                self,
                "Unsupported NLLB route",
                "The configured NLLB backend is currently fixed to English → Bengali.",
            )
            return
        if backend == "sarvam-api" and self._sarvam_model_combo.currentText() == "mayura:v1":
            if source_lang not in SARVAM_MAYURA_CODES or target_lang not in SARVAM_MAYURA_CODES:
                QMessageBox.warning(
                    self,
                    "Unsupported Mayura route",
                    "Mayura supports English, Bengali, Gujarati, Hindi, Kannada, "
                    "Malayalam, Marathi, Odia, Punjabi, Tamil, and Telugu. Choose "
                    "Sarvam Translate for the broader language set.",
                )
                return

        if backend in {"indictrans2", "nllb"} or sarvam_fallback_backend:
            expected_backend = "nllb" if backend == "nllb" else "indictrans2"
            model_ready, model_identity, model_message = _local_model_status(
                self._model_path_edit.text().strip(), expected_backend
            )
            if not model_ready:
                role = "backup " if sarvam_fallback_backend else ""
                QMessageBox.critical(
                    self,
                    f"Local {role}model not ready",
                    f"{model_identity}\n\n{model_message}",
                )
                self._refresh_model_status()
                return

        if backend == "sarvam-api" and self._save_sarvam_key_check.isChecked() and sarvam_api_key:
            try:
                save_sarvam_api_key(sarvam_api_key)
            except CredentialStorageError as exc:
                QMessageBox.warning(self, "Could not save Sarvam key", str(exc))

        checkpoint_path = None
        if self._source_path is not None:
            try:
                checkpoint_path = str(
                    make_translation_checkpoint_path(
                        self._source_path.name,
                        self._source_path.read_bytes(),
                    )
                )
            except OSError:
                checkpoint_path = str(
                    make_translation_checkpoint_path(
                        self._source_path.name,
                        serialize_subtitle(self._document).encode("utf-8"),
                    )
                )
        settings = TranslationSettings(
            source_lang=source_lang,
            target_lang=target_lang,
            chunk_size=self._chunk_size_spin.value(),
            context_window_chars=self._context_chars_spin.value(),
            context_window_cues=self._context_cues_spin.value(),
            max_line_length=self._max_line_spin.value(),
            max_lines=self._max_lines_spin.value(),
        )

        self._translation_in_progress = True
        self._refresh_translate_button()
        self._save_btn.setEnabled(False)
        self._progress.setRange(0, 0)   # indeterminate while model loads
        self._translate_start_time = None
        self.statusBar().showMessage("Loading model…")

        self._thread = QThread()
        self._worker = TranslateWorker(
            document=self._document,
            backend=backend,
            model_path=self._model_path_edit.text().strip(),
            sarvam_api_key=sarvam_api_key,
            sarvam_model=self._sarvam_model_combo.currentText(),
            sarvam_mode=self._sarvam_mode_combo.currentText(),
            sarvam_fallback_backend=sarvam_fallback_backend,
            checkpoint_path=checkpoint_path,
            settings=settings,
            glossary=glossary,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.model_loaded.connect(self._on_model_loaded)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_translate_finished)
        self._worker.interrupted.connect(self._on_translate_interrupted)
        self._worker.failed.connect(self._on_translate_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.interrupted.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    @Slot()
    def _on_model_loaded(self, provider: str) -> None:
        self._progress.setRange(0, 100)  # switch to determinate for real progress
        self._progress.setValue(0)
        self._translate_start_time = time.monotonic()
        self._set_review(f"In progress · {provider}")
        self.statusBar().showMessage(f"Translating with {provider}...")

    @Slot(float, str)
    def _on_progress(self, value: float, message: str) -> None:
        self._progress.setValue(int(value * 100))
        if value > 0.01 and self._translate_start_time is not None:
            elapsed = time.monotonic() - self._translate_start_time
            remaining = elapsed / value * (1.0 - value)
            self.statusBar().showMessage(f"{message}  —  ETA {_format_eta(remaining)}")
        else:
            self.statusBar().showMessage(message)

    @Slot(str, list, str)
    def _on_translate_finished(self, output: str, warnings: list, provider_summary: str) -> None:
        self._translated_output = output
        self._translated_view.setPlainText(output)
        self._progress.setValue(100)
        self._save_btn.setEnabled(True)
        self._translation_in_progress = False
        self._refresh_translate_button()
        self._translate_start_time = None
        fallback_used = any("FALLBACK USED" in warning for warning in warnings)
        fallback_note = " · local fallback used" if fallback_used else " · fallback not used"
        issue_note = (
            f"{len(warnings)} review warning(s)"
            if warnings
            else "no issues flagged"
        )
        self._set_review(
            f"Completed · {provider_summary}{fallback_note} · {issue_note}",
            [str(warning) for warning in warnings],
        )
        if warnings:
            preview = warnings[0]
            extra = f" (+{len(warnings) - 1} more)" if len(warnings) > 1 else ""
            self.statusBar().showMessage(
                f"Translation complete via {provider_summary}. "
                f"{len(warnings)} cue(s) flagged for review — "
                f"hover for details: {preview}{extra}",
                15000,
            )
            self.statusBar().setToolTip("\n".join(warnings))
            if fallback_used:
                QMessageBox.warning(
                    self,
                    "Fallback used",
                    "Sarvam failed for at least one batch, so local IndicTrans "
                    "handled fallback output. Review the warning details before "
                    "trusting or saving the file.",
                )
        else:
            self.statusBar().showMessage(
                f"Translation complete via {provider_summary}. No issues flagged.",
                5000,
            )
            self.statusBar().setToolTip("")

    @Slot(str, str, list)
    def _on_translate_interrupted(
        self,
        message: str,
        partial_output: str,
        warnings: list,
    ) -> None:
        self._translated_output = partial_output
        self._translated_view.setPlainText(partial_output)
        self._progress.setRange(0, 100)
        self._save_btn.setEnabled(bool(partial_output))
        self._translation_in_progress = False
        self._refresh_translate_button()
        self._translate_start_time = None
        self._set_review(
            f"Partial output · {self._selected_provider_label()} · "
            "translation interrupted · checkpoint saved.",
            [message, *[str(warning) for warning in warnings]],
        )
        self.statusBar().showMessage(
            "Translation interrupted. Progress checkpoint saved; rerun to resume.",
            15000,
        )
        self.statusBar().setToolTip("\n".join([message, *[str(w) for w in warnings]]))
        QMessageBox.warning(
            self,
            "Translation interrupted",
            f"{message}\n\nPartial output is shown and can be saved. "
            "Rerun with the same file, settings, glossary, and provider to resume.",
        )

    @Slot(str)
    def _on_translate_failed(self, message: str) -> None:
        self._translation_in_progress = False
        self._refresh_translate_button()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._translate_start_time = None
        self._set_review(
            f"Failed · {self._selected_provider_label()} · no completed output accepted.",
            [message],
        )
        self.statusBar().showMessage("Translation failed.", 5000)
        QMessageBox.critical(self, "Translation failed", message)

    @Slot()
    def _on_save(self) -> None:
        if not self._translated_output or self._source_path is None:
            return
        ext = self._source_path.suffix
        default_name = f"{self._source_path.stem}.translated{ext}"
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save translated subtitle", default_name, SUBTITLE_FILTER
        )
        if not path_str:
            return
        try:
            Path(path_str).write_text(self._translated_output, encoding="utf-8")
            self.statusBar().showMessage(f"Saved to {path_str}", 5000)
        except OSError as exc:
            QMessageBox.critical(self, "Write error", str(exc))


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
