from __future__ import annotations

import numpy as np

from aces_b2ai.context import ClipContext
from aces_b2ai.core.base import BaseExtractor, ExtractionResult


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size or a.size < 3:
        return float("nan")
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    m = np.corrcoef(a, b)
    return float(m[0, 1])


class CrossmodalExtractor(BaseExtractor):
    name = "crossmodal"

    def extract(self, ctx: ClipContext) -> ExtractionResult:
        feats: dict[str, float] = {}
        warns: list[str] = []
        p = ctx.pitch_aligned
        m = ctx.mfcc_aligned
        mask = ctx.voiced_mask
        if p is None or m is None or m.shape[0] < 1:
            warns.append("crossmodal: need pitch_aligned and mfcc_aligned")
            return ExtractionResult(feats, {"extractor": self.name}, warns)

        c0 = m[0]
        if mask is not None and mask.size == p.size:
            sel = mask & np.isfinite(p) & np.isfinite(c0)
        else:
            sel = np.isfinite(p) & np.isfinite(c0)
        if np.sum(sel) < 5:
            feats["cross_mfcc0_f0_coupling_pearson"] = float("nan")
            warns.append("crossmodal: too few frames for coupling")
        else:
            feats["cross_mfcc0_f0_coupling_pearson"] = _pearson(c0[sel], p[sel])

        return ExtractionResult(feats, {"extractor": self.name}, warns)
