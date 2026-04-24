from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from aces_b2ai.context import ClipContext
from aces_b2ai.core.base import BaseExtractor, ExtractionResult


def sample_entropy(x: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
    """Sample entropy via template matching (Chebyshev distance, O(N^2); capped N)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n > 400:
        idx = np.linspace(0, n - 1, num=400).astype(int)
        x = x[idx]
        n = x.size
    if n < m + 10:
        return float("nan")
    sd = float(np.std(x))
    if sd < 1e-12:
        return float("nan")
    r = r_frac * sd
    if r <= 0:
        return float("nan")

    def count_pairs(mlen: int) -> int:
        cnt = 0
        upper = n - mlen + 1
        for i in range(upper):
            for j in range(upper):
                if i == j:
                    continue
                if np.max(np.abs(x[i : i + mlen] - x[j : j + mlen])) < r:
                    cnt += 1
        return cnt

    b = count_pairs(m)
    a = count_pairs(m + 1)
    if b <= 0 or a <= 0:
        return float("nan")
    return float(-np.log(a / b))


def takens_spread(x: np.ndarray, tau: int) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if tau < 1 or x.size <= tau + 2:
        return float("nan")
    a = x[:-tau]
    b = x[tau:]
    return float(np.std(a) * np.std(b))


class DynamicalExtractor(BaseExtractor):
    name = "dynamical"

    def __init__(self, min_voiced_frames: int = 20) -> None:
        self.min_voiced_frames = min_voiced_frames

    def extract(self, ctx: ClipContext) -> ExtractionResult:
        feats: dict[str, float] = {}
        warns: list[str] = []
        p = ctx.pitch_aligned
        mask = ctx.voiced_mask
        if p is None or mask is None or p.size < 4:
            warns.append("dynamical: insufficient pitch/mask")
            return ExtractionResult(feats, {"extractor": self.name}, warns)

        d1 = np.gradient(p)
        vm = mask & np.isfinite(p) & np.isfinite(d1)
        if int(np.sum(vm)) < self.min_voiced_frames:
            warns.append("dynamical: too few voiced frames for phase portrait")
            return ExtractionResult(feats, {"extractor": self.name}, warns)

        pts = np.column_stack([p[vm], d1[vm]])
        try:
            hull = ConvexHull(pts)
            feats["dyn_phase_hull_area"] = float(hull.volume)
        except QhullError:
            feats["dyn_phase_hull_area"] = float("nan")
            warns.append("dynamical: ConvexHull failed")

        n = pts.shape[0]
        t1 = n // 3
        t2 = 2 * n // 3
        early = pts[:t1].mean(axis=0) if t1 > 0 else pts.mean(axis=0)
        late = pts[t2:].mean(axis=0) if t2 < n else pts.mean(axis=0)
        feats["dyn_phase_centroid_drift_l2"] = float(np.linalg.norm(late - early))

        pv = p[vm]
        feats["dyn_pitch_sample_entropy_m2"] = sample_entropy(pv, m=2, r_frac=0.2)
        feats["dyn_takens_spread_tau3"] = takens_spread(pv, tau=3)

        return ExtractionResult(feats, {"extractor": self.name}, warns)
