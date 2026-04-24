from __future__ import annotations

import numpy as np


def as_float2d(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name}: expected 2D array, got shape {a.shape}")
    return a


def as_float1d(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).ravel()
    return a


def assert_positive_length(n: int, name: str) -> None:
    if n <= 0:
        raise ValueError(f"{name}: length must be positive, got {n}")
