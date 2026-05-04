"""Utilities for dataset preprocessing"""

import numpy as np
import torch


def pad_or_truncate(mel_spec, target_length):
    """
    Pad or truncate mel spectrogram to target length along time axis.

    Args:
        mel_spec: np.ndarray of shape [n_mels, time] (typical) or [time, n_mels]
        target_length: Target number of time frames

    Returns:
        np.ndarray of shape [target_length, n_mels] (SSAST format)
    """
    # Detect format: if first dim is likely n_mels (60, 64, 80, 128, 201), transpose
    # Common n_mels values for audio: 60, 64, 80, 128, 201
    if mel_spec.shape[0] in [60, 64, 80, 128, 201]:
        # [n_mels, time] → transpose to [time, n_mels]
        mel_spec = mel_spec.T
    elif mel_spec.shape[0] < mel_spec.shape[1]:
        # Ambiguous case: assume smaller dimension is n_mels
        mel_spec = mel_spec.T

    current_length = mel_spec.shape[0]
    n_mels = mel_spec.shape[1]

    if current_length < target_length:
        # Pad with zeros
        padding = np.zeros((target_length - current_length, n_mels), dtype=mel_spec.dtype)
        mel_spec = np.concatenate([mel_spec, padding], axis=0)
    elif current_length > target_length:
        # Truncate (center crop)
        start_idx = (current_length - target_length) // 2
        mel_spec = mel_spec[start_idx:start_idx + target_length, :]

    return mel_spec


def normalize_mel(mel_spec, mean, std):
    """
    Normalize mel spectrogram using SSAST normalization scheme.

    SSAST uses: (x - mean) / (std * 2)

    Args:
        mel_spec: np.ndarray or torch.Tensor
        mean: Global mean computed from training set
        std: Global std computed from training set

    Returns:
        Normalized mel spectrogram (same type as input)
    """
    return (mel_spec - mean) / (std * 2)


def subsample_time_axis(spec, factor=2):
    """
    Subsample spectrogram along time axis (e.g., 100Hz → 50Hz).

    Args:
        spec: np.ndarray of shape [freq, time]
        factor: Subsampling factor (default 2 for 100Hz → 50Hz)

    Returns:
        Subsampled spectrogram [freq, time // factor]
    """
    return spec[:, ::factor]
