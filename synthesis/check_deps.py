"""
Quick dependency smoke-check for the UVFP source-filter pipeline.

Run:
  python3 synthesis/check_deps.py

If you don't have conda on PATH, you can use venv:
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  python3 synthesis/check_deps.py
"""

from __future__ import annotations


def main() -> None:
    # Core numeric / data
    import numpy as np  # noqa: F401
    import pandas as pd  # noqa: F401
    import scipy  # noqa: F401

    # Audio + DSP
    import librosa  # noqa: F401
    import soundfile as sf  # noqa: F401

    # Praat features (pip package name: parselmouth / praat-parselmouth)
    import parselmouth  # noqa: F401

    # WORLD vocoder
    import pyworld  # noqa: F401

    # Plotting + progress
    import matplotlib  # noqa: F401
    from tqdm import tqdm  # noqa: F401

    print("OK: all synthesis dependencies imported successfully.")


if __name__ == "__main__":
    main()

