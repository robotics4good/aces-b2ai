from __future__ import annotations

import numpy as np

from aces_b2ai.context import ClipContext
from aces_b2ai.core.base import BaseExtractor, ExtractionResult


def _acf_at_lag(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x - np.mean(x)
    if lag >= x.size or lag < 0:
        return float("nan")
    num = np.dot(x[:-lag], x[lag:]) if lag > 0 else np.dot(x, x)
    den = float(np.dot(x, x)) + 1e-12
    return float(num / den)


class MfccSecondaryExtractor(BaseExtractor):
    name = "mfcc_secondary"

    def extract(self, ctx: ClipContext) -> ExtractionResult:
        feats: dict[str, float] = {}
        warns: list[str] = []
        m = ctx.mfcc_aligned if ctx.mfcc_aligned is not None else ctx.mfcc
        if m is None or m.ndim != 2 or m.shape[1] < 2:
            warns.append("mfcc_secondary: missing mfcc")
            return ExtractionResult(feats, {"extractor": self.name}, warns)

        d = np.diff(m, axis=1)
        for i in range(min(4, m.shape[0])):
            feats[f"mfcc_delta_coeff{i}_abs_mean"] = (
                float(np.mean(np.abs(d[i]))) if d.size else float("nan")
            )

        T = m.shape[1]
        third = max(1, T // 3)
        early = m[:, :third].mean(axis=1)
        late = m[:, -third:].mean(axis=1)
        drift = float(np.linalg.norm(early[:4] - late[:4]))
        feats["mfcc_segment_drift_l2_c0_3"] = drift

        lags = (1, 2, 5, 10)
        for coeff in range(min(13, m.shape[0])):
            for lag in lags:
                feats[f"mfcc{coeff}_autocorr_lag{lag}"] = _acf_at_lag(m[coeff], lag)

        return ExtractionResult(feats, {"extractor": self.name}, warns)
