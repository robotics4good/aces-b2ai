from __future__ import annotations

import numpy as np

from aces_b2ai.context import ClipContext
from aces_b2ai.core.base import BaseExtractor, ExtractionResult


def _voiced_runs(mask: np.ndarray) -> list[int]:
    m = np.asarray(mask, dtype=bool).ravel()
    runs: list[int] = []
    i = 0
    n = m.size
    while i < n:
        if m[i]:
            j = i + 1
            while j < n and m[j]:
                j += 1
            runs.append(j - i)
            i = j
        else:
            i += 1
    return runs


def _histogram_entropy(x: np.ndarray, n_bins: int = 32) -> float:
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    hist, _ = np.histogram(x, bins=n_bins, density=False)
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return float("nan")
    p = p / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-12)))


class PitchSecondaryExtractor(BaseExtractor):
    name = "pitch_secondary"

    def extract(self, ctx: ClipContext) -> ExtractionResult:
        feats: dict[str, float] = {}
        warns: list[str] = []
        p = ctx.pitch_aligned
        if p is None or p.size == 0:
            warns.append("pitch_secondary: no aligned pitch")
            return ExtractionResult({}, {}, warns)

        mask = ctx.voiced_mask
        if mask is None or mask.size != p.size:
            mask = np.isfinite(p) & (p > 1.0)

        vf = float(np.mean(mask)) if mask.size else float("nan")
        feats["pitch_voiced_fraction_aligned"] = vf

        pv = p[mask]
        if pv.size:
            feats["pitch_f0_mean_hz_masked"] = float(np.mean(pv))
            feats["pitch_f0_std_hz_masked"] = float(np.std(pv))
            feats["pitch_f0_entropy_masked"] = _histogram_entropy(pv)
        else:
            feats["pitch_f0_mean_hz_masked"] = float("nan")
            feats["pitch_f0_std_hz_masked"] = float("nan")
            feats["pitch_f0_entropy_masked"] = float("nan")
            warns.append("pitch_secondary: no voiced frames for stats")

        t = np.arange(p.size, dtype=np.float64)
        if np.sum(mask) >= 3:
            tv = t[mask]
            pv2 = p[mask]
            tv = tv - tv.mean()
            denom = float(np.sum(tv**2)) + 1e-12
            slope = float(np.sum(tv * (pv2 - pv2.mean())) / denom)
            feats["pitch_depletion_slope_hz_per_frame"] = slope
        else:
            feats["pitch_depletion_slope_hz_per_frame"] = float("nan")

        runs = _voiced_runs(mask)
        feats["pitch_voiced_run_count"] = float(len(runs))
        if runs:
            feats["pitch_mean_voiced_run_len_frames"] = float(np.mean(runs))
            feats["pitch_max_voiced_run_len_frames"] = float(np.max(runs))
        else:
            feats["pitch_mean_voiced_run_len_frames"] = float("nan")
            feats["pitch_max_voiced_run_len_frames"] = float("nan")

        dp = np.diff(p)
        vm = mask[1:] & mask[:-1]
        if np.any(vm):
            feats["pitch_frame_diff_std_hz"] = float(np.std(dp[vm]))
        else:
            feats["pitch_frame_diff_std_hz"] = float("nan")

        return ExtractionResult(feats, {"extractor": self.name}, warns)
