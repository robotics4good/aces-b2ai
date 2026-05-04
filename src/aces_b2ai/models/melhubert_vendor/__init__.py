"""Vendored MelHuBERT source.

Copied from https://github.com/nervjack2/MelHuBERT (ASRU 2023, MIT License).
Only change: bare module imports converted to relative imports so this
sub-package works inside the aces_b2ai Python package.
"""

from .model import MelHuBERTConfig, MelHuBERTModel

__all__ = ["MelHuBERTConfig", "MelHuBERTModel"]
