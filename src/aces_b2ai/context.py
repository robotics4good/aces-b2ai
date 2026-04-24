from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ClipContext:
    """Aligned clip-level tensors and identifiers for secondary extraction."""

    participant_id: str
    session_id: str
    task_name: str

    mel: np.ndarray | None = None  # (n_mels, T_mel)
    mfcc: np.ndarray | None = None  # (n_mfcc, T_mfcc)
    spectrogram: np.ndarray | None = None  # (n_freq, T_spec)
    pitch_torch: np.ndarray | None = None  # (T_p,) Hz torchaudio
    periodicity_sparc: np.ndarray | None = None  # (T_s,) [0,1]
    pitch_sparc: np.ndarray | None = None  # (T_s,) Hz optional

    static_features: dict[str, float] = field(default_factory=dict)

    age_years: float | None = None
    sex: str | None = None

    # After preprocessing: common length T_ref, aligned series
    pitch_aligned: np.ndarray | None = None
    mel_mean_energy: np.ndarray | None = None  # (T_ref,) mean over mel bins
    mfcc_aligned: np.ndarray | None = None  # (C, T_ref)
    voiced_mask: np.ndarray | None = None  # (T_ref,) bool
    frame_rate_hz: float = 50.0

    extras: dict[str, Any] = field(default_factory=dict)
