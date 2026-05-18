"""First-run setup and launcher.

Entry point used by SubtitleTranslator.app after the shell script bootstraps
the venv and installs PySide6.  Handles:
  1. Installing the remaining Python packages (pip install -r requirements.txt)
  2. Downloading a translation model from HuggingFace
  3. Launching the main GUI (gui.py) via os.execv
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).parent
PYTHON = Path(sys.executable)

MODELS = {
    "IndicTrans2 distilled 200M  (~800 MB, recommended)": "dist",
    "IndicTrans2 full 1B  (~4.5 GB)": "1B",
    "NLLB-200 distilled 600M  (~2.5 GB)": "nllb",
}

# Model dirs that indicate a completed download
_MODEL_DIRS = {
    "dist": ROOT / "models" / "indictrans2-en-indic",
    "1B": ROOT / "models" / "indictrans2-en-indic",
    "nllb": ROOT / "models" / "nllb-200-distilled-600M",
}


def _packages_ok() -> bool:
    result = subprocess.run(
        [str(PYTHON), "-c", "import torch; import transformers; import sentencepiece"],
        capture_output=True,
    )
    return result.returncode == 0


def _any_model_ready() -> bool:
    return any(d.exists() and any(d.iterdir()) for d in _MODEL_DIRS.values())


class _CommandWorker(QObject):
    line = Signal(str)
    done = Signal(int)

    def __init__(self, cmd: list[str]) -> None:
        super().__init__()
        self._cmd = cmd

    @Slot()
    def run(self) -> None:
        try:
            proc = subprocess.Popen(
                self._cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout
            for raw in proc.stdout:
                self.line.emit(raw.rstrip())
            proc.wait()
            self.done.emit(proc.returncode)
        except Exception as exc:
            self.line.emit(f"Error: {exc}")
            self.done.emit(1)


class SetupWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Subtitle Translator — Setup")
        self.resize(700, 620)

        self._pkg_ok = _packages_ok()
        self._model_ok = _any_model_ready()
        self._thread: QThread | None = None
        self._worker: _CommandWorker | None = None

        self._build_ui()
        self._refresh()

    # ------------------------------------------------------------------ UI build

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Subtitle Translator — First Time Setup")
        title.setAlignment(Qt.AlignCenter)
        f = title.font()
        f.setPointSize(16)
        f.setBold(True)
        title.setFont(f)
        root.addWidget(title)

        subtitle = QLabel(
            "Complete the two steps below, then open the translator. "
            "You only need to do this once."
        )
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        root.addWidget(self._build_packages_group())
        root.addWidget(self._build_models_group())
        root.addWidget(self._build_log_group(), 1)

        self._launch_btn = QPushButton("Open Subtitle Translator  ▶")
        self._launch_btn.setMinimumHeight(46)
        f = self._launch_btn.font()
        f.setPointSize(14)
        self._launch_btn.setFont(f)
        self._launch_btn.clicked.connect(self._on_launch)
        root.addWidget(self._launch_btn)

    def _build_packages_group(self) -> QGroupBox:
        g = QGroupBox("Step 1 — Python packages")
        lay = QVBoxLayout(g)

        self._pkg_status = QLabel()
        lay.addWidget(self._pkg_status)

        self._pkg_btn = QPushButton("Install packages")
        self._pkg_btn.clicked.connect(self._on_install_packages)
        lay.addWidget(self._pkg_btn)
        return g

    def _build_models_group(self) -> QGroupBox:
        g = QGroupBox("Step 2 — Translation model")
        lay = QVBoxLayout(g)

        self._model_status = QLabel()
        lay.addWidget(self._model_status)

        row = QHBoxLayout()
        row.addWidget(QLabel("Model:"))
        self._model_combo = QComboBox()
        for label in MODELS:
            self._model_combo.addItem(label)
        row.addWidget(self._model_combo, 1)
        lay.addLayout(row)

        self._download_btn = QPushButton("Download model")
        self._download_btn.clicked.connect(self._on_download_model)
        lay.addWidget(self._download_btn)

        skip_row = QHBoxLayout()
        skip_lbl = QLabel("Already have a model folder?")
        self._browse_btn = QPushButton("Browse existing path…")
        self._browse_btn.clicked.connect(self._on_browse_model)
        skip_row.addWidget(skip_lbl)
        skip_row.addWidget(self._browse_btn)
        skip_row.addStretch()
        lay.addLayout(skip_row)
        return g

    def _build_log_group(self) -> QGroupBox:
        g = QGroupBox("Output")
        lay = QVBoxLayout(g)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        lay.addWidget(self._progress)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(1000)
        self._log.setFont(QFont("Menlo", 11))
        self._log.setMinimumHeight(140)
        lay.addWidget(self._log)
        return g

    # ------------------------------------------------------------------ state

    def _refresh(self) -> None:
        if self._pkg_ok:
            self._pkg_status.setText("✓  Python packages are installed.")
            self._pkg_status.setStyleSheet("color: green;")
            self._pkg_btn.setEnabled(False)
        else:
            self._pkg_status.setText("⚠  Python packages are not yet installed.")
            self._pkg_status.setStyleSheet("color: #cc7700;")
            self._pkg_btn.setEnabled(not self._busy())

        if self._model_ok:
            self._model_status.setText("✓  Translation model found.")
            self._model_status.setStyleSheet("color: green;")
            self._download_btn.setEnabled(False)
            self._model_combo.setEnabled(False)
        else:
            self._model_status.setText("⚠  No model downloaded yet.")
            self._model_status.setStyleSheet("color: #cc7700;")
            can = self._pkg_ok and not self._busy()
            self._download_btn.setEnabled(can)
            self._model_combo.setEnabled(can)

        self._browse_btn.setEnabled(not self._busy())
        # Can open the main app once packages are installed (model path is
        # configurable inside the app itself).
        self._launch_btn.setEnabled(self._pkg_ok and not self._busy())

    def _busy(self) -> bool:
        return bool(self._thread and self._thread.isRunning())

    # ------------------------------------------------------------------ actions

    @Slot()
    def _on_install_packages(self) -> None:
        self._run(
            [str(PYTHON), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")],
            on_done=self._after_install,
        )

    def _after_install(self, rc: int) -> None:
        if rc == 0:
            self._pkg_ok = True
            self._log.appendPlainText("\n✓ Packages installed successfully.")
        else:
            self._log.appendPlainText("\n✗ Installation failed — see output above.")
        self._refresh()

    @Slot()
    def _on_download_model(self) -> None:
        key = MODELS[self._model_combo.currentText()]
        self._run(
            [str(PYTHON), str(ROOT / "scripts" / "download_models.py"), "--model", key],
            on_done=self._after_download,
        )

    def _after_download(self, rc: int) -> None:
        if rc == 0:
            self._model_ok = _any_model_ready()
            self._log.appendPlainText("\n✓ Model downloaded successfully.")
        else:
            self._log.appendPlainText("\n✗ Download failed — see output above.")
        self._refresh()

    @Slot()
    def _on_browse_model(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select existing model directory")
        if path and Path(path).exists():
            self._log.appendPlainText(
                f"Path noted: {path}\n"
                "You can set this path in the main app's 'Model path' field."
            )
            self._model_ok = True
            self._refresh()

    @Slot()
    def _on_launch(self) -> None:
        self.hide()
        os.execv(str(PYTHON), [str(PYTHON), str(ROOT / "gui.py")])

    # ------------------------------------------------------------------ worker

    def _run(self, cmd: list[str], *, on_done) -> None:
        if self._busy():
            return
        self._log.clear()
        self._progress.setVisible(True)
        self._refresh()

        self._thread = QThread()
        self._worker = _CommandWorker(cmd)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.line.connect(self._log.appendPlainText)
        self._worker.done.connect(lambda rc: self._on_worker_done(rc, on_done))
        self._worker.done.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_worker_done(self, rc: int, on_done) -> None:
        self._progress.setVisible(False)
        on_done(rc)


def main() -> int:
    app = QApplication(sys.argv)
    win = SetupWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
