from __future__ import annotations

import numpy as np

from aces_b2ai.config import PipelineConfig, VoicingConfig
from aces_b2ai.context import ClipContext


def _resample_series(y: np.ndarray, n_target: int) -> np.ndarray:
    """Linear interpolation onto `n_target` samples over normalized [0, 1]."""
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.size == 0:
        return np.full(n_target, np.nan)
    if y.size == 1:
        return np.full(n_target, float(y[0]))
    x_old = np.linspace(0.0, 1.0, num=y.size)
    x_new = np.linspace(0.0, 1.0, num=n_target)
    return np.interp(x_new, x_old, y)


def reference_length(ctx: ClipContext) -> int:
    if ctx.mel is not None and ctx.mel.ndim == 2 and ctx.mel.shape[1] > 0:
        return int(ctx.mel.shape[1])
    if ctx.mfcc is not None and ctx.mfcc.ndim == 2 and ctx.mfcc.shape[1] > 0:
        return int(ctx.mfcc.shape[1])
    if ctx.spectrogram is not None and ctx.spectrogram.ndim == 2:
        return int(ctx.spectrogram.shape[1])
    if ctx.pitch_torch is not None:
        return int(np.asarray(ctx.pitch_torch).size)
    return 0


def voiced_mask_from_pitch(
    pitch_hz: np.ndarray,
    cfg: VoicingConfig,
) -> np.ndarray:
    p = np.asarray(pitch_hz, dtype=np.float64).ravel()
    m = (
        np.isfinite(p)
        & (p > cfg.f0_min_hz)
        & (p < cfg.f0_max_hz)
    )
    return m


def voiced_mask_hierarchical(
    pitch_aligned: np.ndarray,
    periodicity_aligned: np.ndarray | None,
    cfg: VoicingConfig,
) -> tuple[np.ndarray, str]:
    """Single voicing mask for the aligned grid (combined doc §23)."""
    if periodicity_aligned is not None and periodicity_aligned.size == pitch_aligned.size:
        per = np.asarray(periodicity_aligned, dtype=np.float64).ravel()
        m = np.isfinite(per) & (per > cfg.periodicity_threshold)
        return m, "sparc_periodicity"
    m = voiced_mask_from_pitch(pitch_aligned, cfg)
    return m, "pitch_bounds"


def prepare_clip_context(ctx: ClipContext, cfg: PipelineConfig) -> ClipContext:
    """Populate `pitch_aligned`, `mfcc_aligned`, `mel_mean_energy`, `voiced_mask`, `frame_rate_hz`."""
    T = reference_length(ctx)
    if T <= 0:
        return ctx

    if ctx.pitch_torch is not None:
        ctx.pitch_aligned = _resample_series(ctx.pitch_torch, T)
    else:
        ctx.pitch_aligned = np.full(T, np.nan)

    per_aligned: np.ndarray | None = None
    if ctx.periodicity_sparc is not None:
        per_aligned = _resample_series(ctx.periodicity_sparc, T)

    if ctx.mel is not None and ctx.mel.ndim == 2:
        ctx.mel_mean_energy = np.mean(ctx.mel, axis=0)
        if ctx.mel_mean_energy.size != T:
            ctx.mel_mean_energy = _resample_series(ctx.mel_mean_energy, T)
    else:
        ctx.mel_mean_energy = None

    if ctx.mfcc is not None and ctx.mfcc.ndim == 2:
        c, tm = ctx.mfcc.shape
        out = np.zeros((c, T))
        for i in range(c):
            out[i] = _resample_series(ctx.mfcc[i], T)
        ctx.mfcc_aligned = out
    else:
        ctx.mfcc_aligned = None

    mask, source = voiced_mask_hierarchical(ctx.pitch_aligned, per_aligned, cfg.voicing)
    ctx.voiced_mask = mask
    ctx.frame_rate_hz = cfg.torchaudio_2d_frame_rate_hz
    ctx.extras["voicing_mask_source"] = source
    return ctx
