from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from aces_b2ai.config import ArticulatorySmoothnessConfig
from aces_b2ai.context import ClipContext
from aces_b2ai.core.base import BaseExtractor, ExtractionResult


def _interp_nans(x: np.ndarray) -> np.ndarray:
    """Fill any non-finite values via linear interpolation.

    In practice the SPARC EMA data has no NaN frames (neural inversion, not
    physical sensors), so this is a defensive guard only.
    """
    finite = np.isfinite(x)
    if finite.all():
        return x
    if not finite.any():
        return x
    xi = np.arange(len(x))
    return np.interp(xi, xi[finite], x[finite])


def _active_mask_velocity(
    signal: np.ndarray,
    cfg: ArticulatorySmoothnessConfig,
) -> np.ndarray:
    """Boolean mask: frames where |velocity| exceeds threshold.

    Uses the same SG parameters as the jerk computation so that the mask
    and the jerk signal share the same smoothing assumptions.
    """
    velocity = savgol_filter(
        signal,
        window_length=cfg.sg_window_frames,
        polyorder=cfg.sg_polyorder,
        deriv=1,
        delta=1.0 / cfg.fs,
    )
    return np.abs(velocity) > cfg.velocity_threshold


def _active_mask_loudness(
    signal: np.ndarray,
    loudness: np.ndarray,
    cfg: ArticulatorySmoothnessConfig,
) -> np.ndarray:
    """Resample loudness to EMA length and threshold at 0.1 z-scored amplitude.

    Loudness is z-scored waveform amplitude at 50 Hz (same grid as EMA, ±1
    frame). A threshold of 0.1 excludes near-silence without being too
    aggressive on quiet speech.
    """
    t_ema = len(signal)
    t_loud = len(loudness)
    if t_loud == t_ema:
        loud_resampled = loudness
    else:
        xi_ema = np.linspace(0, t_loud - 1, t_ema)
        xi_loud = np.arange(t_loud)
        loud_resampled = np.interp(xi_ema, xi_loud, loudness)
    return loud_resampled > 0.1


def _normalized_jerk(
    signal: np.ndarray,
    cfg: ArticulatorySmoothnessConfig,
    active_mask: np.ndarray | None,
) -> float:
    """Compute normalized jerk integral for one EMA channel.

    Returns (1 / T_active) * integral(jerk^2 dt) over active frames, where
    jerk = d³x/dt³ estimated via Savitzky-Golay analytic differentiation.

    Returns NaN if insufficient active frames remain after masking and edge
    trimming.
    """
    dt = 1.0 / cfg.fs

    jerk = savgol_filter(
        signal,
        window_length=cfg.sg_window_frames,
        polyorder=cfg.sg_polyorder,
        deriv=3,
        delta=dt,
    )

    if cfg.trim_edge_frames:
        half_w = cfg.sg_window_frames // 2
        jerk = jerk[half_w : len(jerk) - half_w]
        if active_mask is not None:
            active_mask = active_mask[half_w : len(active_mask) - half_w]

    if active_mask is not None:
        jerk_active = jerk[active_mask]
        T_active = float(np.sum(active_mask)) * dt
    else:
        jerk_active = jerk
        T_active = len(jerk) * dt

    if jerk_active.size < cfg.min_active_frames or T_active <= 0.0:
        return float("nan")

    return float((1.0 / T_active) * np.sum(jerk_active**2) * dt)


class ArticulatorySmoothnessExtractor(BaseExtractor):
    """Normalized jerk of SPARC EMA articulators (Feature Family 2).

    Computes the time-normalized integral of squared third derivative (jerk)
    for three articulator signals extracted from the 12-channel SPARC EMA
    tensor at 50 Hz:

      - TBX (tongue body anterior-posterior, channel 2)
      - TBY (tongue body vertical, channel 3)
      - lip aperture = ULY (channel 9) − LLY (channel 11)

    EMA values are z-scored normalized units (SPARC neural inversion output,
    not raw mm). Jerk is in normalized-units/s³; the normalized integral is
    dimensionless for comparison across utterances of different lengths.

    Outputs: artic_jerk_tbx, artic_jerk_tby, artic_jerk_lip
    """

    name = "articulatory_smoothness"

    def __init__(self, cfg: ArticulatorySmoothnessConfig | None = None) -> None:
        self.cfg = cfg or ArticulatorySmoothnessConfig()

    def extract(self, ctx: ClipContext) -> ExtractionResult:
        feats: dict[str, float] = {}
        warns: list[str] = []
        cfg = self.cfg

        if ctx.ema_sparc is None:
            warns.append("articulatory_smoothness: no ema_sparc on context")
            return ExtractionResult(feats, {"extractor": self.name}, warns)

        ema = np.asarray(ctx.ema_sparc, dtype=np.float64)

        if ema.ndim != 2 or ema.shape[1] != 12:
            warns.append(
                f"articulatory_smoothness: unexpected ema shape {ema.shape}, expected (T, 12)"
            )
            return ExtractionResult(feats, {"extractor": self.name}, warns)

        if ema.shape[0] < cfg.sg_window_frames + 2:
            warns.append(
                f"articulatory_smoothness: ema too short ({ema.shape[0]} frames) "
                f"for window_length={cfg.sg_window_frames}"
            )
            return ExtractionResult(feats, {"extractor": self.name}, warns)

        tbx = _interp_nans(ema[:, cfg.idx_tbx])
        tby = _interp_nans(ema[:, cfg.idx_tby])
        lip = _interp_nans(ema[:, cfg.idx_uly] - ema[:, cfg.idx_lly])

        if cfg.active_mask_source == "velocity":
            # Union of per-channel velocity masks: active if any articulator is moving
            mask_tbx = _active_mask_velocity(tbx, cfg)
            mask_tby = _active_mask_velocity(tby, cfg)
            mask_lip = _active_mask_velocity(lip, cfg)
            shared_mask = mask_tbx | mask_tby | mask_lip
        elif cfg.active_mask_source == "loudness":
            if ctx.loudness_sparc is None:
                warns.append(
                    "articulatory_smoothness: active_mask_source='loudness' but "
                    "loudness_sparc is None; falling back to velocity mask"
                )
                mask_tbx = _active_mask_velocity(tbx, cfg)
                mask_tby = _active_mask_velocity(tby, cfg)
                mask_lip = _active_mask_velocity(lip, cfg)
                shared_mask = mask_tbx | mask_tby | mask_lip
            else:
                loudness = np.asarray(ctx.loudness_sparc, dtype=np.float64).ravel()
                shared_mask = _active_mask_loudness(tbx, loudness, cfg)
        else:
            shared_mask = None

        for key, sig in [("jerk_tbx", tbx), ("jerk_tby", tby), ("jerk_lip", lip)]:
            feats[f"artic_{key}"] = _normalized_jerk(sig, cfg, shared_mask)

        return ExtractionResult(feats, {"extractor": self.name}, warns)
