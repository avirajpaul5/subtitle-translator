"""Download translator model weights from HuggingFace into ./models/.

Run once on the machine where you'll use the GUI:

    python scripts/download_models.py                 # default: IndicTrans2 distilled 200M
    python scripts/download_models.py --model 1B      # full IndicTrans2 1B (~4.5 GB)
    python scripts/download_models.py --model nllb    # NLLB-200 distilled 600M

The destination path matches the default "Local model path" shown in the GUI,
so no settings change is needed after download. Weights are not committed to
git — the ./models/ directory is gitignored.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODELS = {
    "dist": {
        "repo_id": "ai4bharat/indictrans2-en-indic-dist-200M",
        "local_dir": "models/indictrans2-en-indic",
        "size_hint": "~800 MB",
        "backend": "indictrans2",
    },
    "1B": {
        "repo_id": "ai4bharat/indictrans2-en-indic-1B",
        "local_dir": "models/indictrans2-en-indic",
        "size_hint": "~4.5 GB",
        "backend": "indictrans2",
    },
    "nllb": {
        "repo_id": "facebook/nllb-200-distilled-600M",
        "local_dir": "models/nllb-200-distilled-600M",
        "size_hint": "~2.5 GB",
        "backend": "nllb",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=list(MODELS),
        default="dist",
        help="Which model to download (default: dist = IndicTrans2 distilled 200M).",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Override the destination directory. Defaults to ./models/<model-name>/.",
    )
    args = parser.parse_args()

    spec = MODELS[args.model]
    dest = Path(args.dest or spec["local_dir"]).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is required. Install with: pip install huggingface_hub",
            file=sys.stderr,
        )
        return 1

    print(f"Downloading {spec['repo_id']} ({spec['size_hint']}) into {dest}")
    print("This may take a while on first run. Subsequent runs resume / are cached.")

    snapshot_download(
        repo_id=spec["repo_id"],
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )

    # Also fetch the spaCy NER model used for auto-detection of names &
    # foreign words. Small (~13 MB), idempotent, no-op if already present.
    print()
    print("Ensuring spaCy 'en_core_web_sm' is installed (used for auto-detection)…")
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print("  already installed.")
    except (ImportError, OSError):
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=False,
        )

    print()
    print(f"Done. In the GUI, set:")
    print(f"  Backend    = {spec['backend']}")
    print(f"  Model path = {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
