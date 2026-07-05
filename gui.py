"""Native desktop GUI for the subtitle translator (PySide6).

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
from subtitle_translator.pipeline import TranslationSettings, translate_document
from subtitle_translator.speaker_detection import detect_speaker_names
from subtitle_translator.translators.factory import TranslatorInitError, build_translator

# Lazy import — auto_dnt pulls spaCy which is heavy. We import on demand.
def _load_auto_dnt():
    from subtitle_translator.auto_dnt import detect_preserve_spans
    return detect_preserve_spans

SUBTITLE_FILTER = "Subtitle files (*.srt *.vtt);;SRT (*.srt);;WebVTT (*.vtt);;All files (*)"


def _format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s"


class TranslateWorker(QObject):
    progress = Signal(float, str)
    model_loaded = Signal()   # fires once the model is in memory, before translation
    finished = Signal(str, list)  # serialized output, validation warnings
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
            self.model_loaded.emit()
            translated = translate_document(
                document=self._document,
                translator=translator,
                settings=self._settings,
                glossary=self._glossary,
                progress_cb=lambda v, m: self.progress.emit(v, m),
            )
            self.finished.emit(serialize_subtitle(translated), list(translated.warnings))
        except (SubtitleParseError, TranslatorInitError) as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Unexpected error: {exc}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Subtitle Translator")
        self.resize(1100, 780)

        self._document: SubtitleDocument | None = None
        self._source_path: Path | None = None
        self._translated_output: str = ""
        self._thread: QThread | None = None
        self._worker: TranslateWorker | None = None
        self._translate_start_time: float | None = None

        self._build_ui()
        self.statusBar().showMessage("Open a .srt or .vtt file to begin.")

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        top_bar = QHBoxLayout()
        self._open_btn = QPushButton("Open subtitle…")
        self._open_btn.clicked.connect(self._on_open)
        self._file_label = QLabel("No file loaded")
        top_bar.addWidget(self._open_btn)
        top_bar.addWidget(self._file_label, 1)
        root.addLayout(top_bar)

        settings_glossary_row = QHBoxLayout()
        settings_glossary_row.addWidget(self._build_settings_group(), 1)
        settings_glossary_row.addWidget(self._build_glossary_group(), 1)
        root.addLayout(settings_glossary_row)

        splitter = QSplitter(Qt.Horizontal)
        self._original_view = QPlainTextEdit()
        self._original_view.setReadOnly(True)
        self._original_view.setPlaceholderText("Original subtitle content will appear here.")
        self._translated_view = QPlainTextEdit()
        self._translated_view.setReadOnly(True)
        self._translated_view.setPlaceholderText("Translated subtitle content will appear here.")
        splitter.addWidget(self._original_view)
        splitter.addWidget(self._translated_view)
        splitter.setSizes([550, 550])
        root.addWidget(splitter, 1)

        action_bar = QHBoxLayout()
        self._translate_btn = QPushButton("Translate")
        self._translate_btn.setEnabled(False)
        self._translate_btn.clicked.connect(self._on_translate)
        self._save_btn = QPushButton("Save translated…")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        action_bar.addWidget(self._translate_btn)
        action_bar.addWidget(self._progress, 1)
        action_bar.addWidget(self._save_btn)
        root.addLayout(action_bar)

    def _build_settings_group(self) -> QGroupBox:
        group = QGroupBox("Settings")
        form = QFormLayout(group)

        self._backend_combo = QComboBox()
        self._backend_combo.addItems(["indictrans2", "sarvam-api", "echo", "nllb"])
        form.addRow("Backend", self._backend_combo)

        model_row = QHBoxLayout()
        self._model_path_edit = QLineEdit("./models/indictrans2-en-indic")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._on_browse_model)
        model_row.addWidget(self._model_path_edit, 1)
        model_row.addWidget(browse_btn)
        model_container = QWidget()
        model_container.setLayout(model_row)
        form.addRow("Model path", model_container)

        self._sarvam_key_edit = QLineEdit()
        self._sarvam_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._sarvam_key_edit.setPlaceholderText(
            "Leave blank to use SARVAM_API_KEY or OS keychain"
        )
        form.addRow("Sarvam API key", self._sarvam_key_edit)

        self._save_sarvam_key_check = QCheckBox("Save Sarvam key in OS keychain")
        form.addRow(self._save_sarvam_key_check)

        self._sarvam_model_combo = QComboBox()
        self._sarvam_model_combo.addItems(["mayura:v1", "sarvam-translate:v1"])
        self._sarvam_model_combo.currentTextChanged.connect(self._on_sarvam_model_changed)
        form.addRow("Sarvam model", self._sarvam_model_combo)

        self._sarvam_mode_combo = QComboBox()
        self._sarvam_mode_combo.addItems(["classic-colloquial", "modern-colloquial", "formal"])
        form.addRow("Sarvam mode", self._sarvam_mode_combo)

        self._sarvam_fallback_check = QCheckBox("Fallback to local IndicTrans on Sarvam errors")
        self._sarvam_fallback_check.setChecked(True)
        form.addRow(self._sarvam_fallback_check)

        self._source_combo = QComboBox()
        self._source_combo.addItems(["en", "hi", "bn"])
        self._target_combo = QComboBox()
        self._target_combo.addItems(["bn", "hi", "en"])
        form.addRow("Source language", self._source_combo)
        form.addRow("Target language", self._target_combo)

        self._chunk_size_spin = self._spin(1, 64, 12)
        self._merge_min_spin = self._spin(0, 200, 0)
        self._max_line_spin = self._spin(20, 80, 42)
        self._max_lines_spin = self._spin(1, 4, 2)
        form.addRow("Batch chunk size", self._chunk_size_spin)
        form.addRow("Merge cues below chars", self._merge_min_spin)
        form.addRow("Max line length", self._max_line_spin)
        form.addRow("Max lines per cue", self._max_lines_spin)

        self._echo_check = QCheckBox("Echo / test mode (skip real translation)")
        form.addRow(self._echo_check)

        return group

    @Slot(str)
    def _on_sarvam_model_changed(self, model: str) -> None:
        is_sarvam_translate = model == "sarvam-translate:v1"
        self._sarvam_mode_combo.setEnabled(not is_sarvam_translate)
        if is_sarvam_translate:
            self._sarvam_mode_combo.setCurrentText("formal")

    def _build_glossary_group(self) -> QGroupBox:
        group = QGroupBox("Glossary JSON")
        layout = QVBoxLayout(group)

        btn_row = QHBoxLayout()
        load_btn = QPushButton("Load JSON…")
        load_btn.clicked.connect(self._on_load_glossary)
        save_btn = QPushButton("Save JSON…")
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
        self._progress.setValue(0)
        self._save_btn.setEnabled(False)
        self._translate_btn.setEnabled(True)
        self._file_label.setText(f"{path.name}  ·  {len(document.cues)} cues")
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
        try:
            glossary = load_glossary_json(self._glossary_edit.toPlainText())
        except Exception as exc:
            QMessageBox.critical(self, "Invalid glossary JSON", str(exc))
            return

        backend = "echo" if self._echo_check.isChecked() else self._backend_combo.currentText()
        sarvam_api_key = self._sarvam_key_edit.text()
        if backend == "sarvam-api" and self._save_sarvam_key_check.isChecked() and sarvam_api_key:
            try:
                save_sarvam_api_key(sarvam_api_key)
            except CredentialStorageError as exc:
                QMessageBox.warning(self, "Could not save Sarvam key", str(exc))

        sarvam_fallback_backend = (
            "indictrans2"
            if backend == "sarvam-api" and self._sarvam_fallback_check.isChecked()
            else None
        )
        settings = TranslationSettings(
            source_lang=self._source_combo.currentText(),
            target_lang=self._target_combo.currentText(),
            chunk_size=self._chunk_size_spin.value(),
            merge_min_chars=self._merge_min_spin.value(),
            max_line_length=self._max_line_spin.value(),
            max_lines=self._max_lines_spin.value(),
        )

        self._translate_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._progress.setRange(0, 0)   # indeterminate while model loads
        self._translate_start_time = None
        self.statusBar().showMessage("Loading model…")

        self._thread = QThread()
        self._worker = TranslateWorker(
            document=self._document,
            backend=backend,
            model_path=self._model_path_edit.text(),
            sarvam_api_key=sarvam_api_key,
            sarvam_model=self._sarvam_model_combo.currentText(),
            sarvam_mode=self._sarvam_mode_combo.currentText(),
            sarvam_fallback_backend=sarvam_fallback_backend,
            settings=settings,
            glossary=glossary,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.model_loaded.connect(self._on_model_loaded)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_translate_finished)
        self._worker.failed.connect(self._on_translate_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    @Slot()
    def _on_model_loaded(self) -> None:
        self._progress.setRange(0, 100)  # switch to determinate for real progress
        self._progress.setValue(0)
        self._translate_start_time = time.monotonic()
        self.statusBar().showMessage("Translating…")

    @Slot(float, str)
    def _on_progress(self, value: float, message: str) -> None:
        self._progress.setValue(int(value * 100))
        if value > 0.01 and self._translate_start_time is not None:
            elapsed = time.monotonic() - self._translate_start_time
            remaining = elapsed / value * (1.0 - value)
            self.statusBar().showMessage(f"{message}  —  ETA {_format_eta(remaining)}")
        else:
            self.statusBar().showMessage(message)

    @Slot(str, list)
    def _on_translate_finished(self, output: str, warnings: list) -> None:
        self._translated_output = output
        self._translated_view.setPlainText(output)
        self._progress.setValue(100)
        self._save_btn.setEnabled(True)
        self._translate_btn.setEnabled(True)
        self._translate_start_time = None
        if warnings:
            preview = warnings[0]
            extra = f" (+{len(warnings) - 1} more)" if len(warnings) > 1 else ""
            self.statusBar().showMessage(
                f"Translation complete. {len(warnings)} cue(s) flagged for review — "
                f"hover for details: {preview}{extra}",
                15000,
            )
            self.statusBar().setToolTip("\n".join(warnings))
        else:
            self.statusBar().showMessage("Translation complete. No issues flagged.", 5000)
            self.statusBar().setToolTip("")

    @Slot(str)
    def _on_translate_failed(self, message: str) -> None:
        self._translate_btn.setEnabled(True)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._translate_start_time = None
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
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
