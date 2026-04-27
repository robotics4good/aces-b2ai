from __future__ import annotations

import numpy as np

from aces_b2ai.context import ClipContext
from aces_b2ai.core.base import BaseExtractor, ExtractionResult


def _modulation_band_energy(mel: np.ndarray, sr_frames: float) -> dict[str, float]:
    """Per-bin FFT of time axis; aggregate energy into modulation bands (combined §20)."""
    mel = np.asarray(mel, dtype=np.float64)
    if mel.ndim != 2 or mel.shape[1] < 4:
        return {
            "mel_mod_band_dc_to_1hz_mean": float("nan"),
            "mel_mod_band_1_to_4hz_mean": float("nan"),
            "mel_mod_band_4_to_8hz_mean": float("nan"),
            "mel_mod_band_above_8hz_mean": float("nan"),
        }
    n_mels, T = mel.shape
    tms_power = np.zeros((n_mels, T // 2 + 1))
    win = np.hanning(T)
    mod_freqs = np.fft.rfftfreq(T, d=1.0 / sr_frames)
    for b in range(n_mels):
        sig = mel[b] * win
        fft_mag = np.abs(np.fft.rfft(sig))
        tms_power[b] = fft_mag**2

    def band_mean(lo: float, hi: float | None) -> float:
        if hi is None:
            sel = mod_freqs >= lo
        else:
            sel = (mod_freqs >= lo) & (mod_freqs < hi)
        if not np.any(sel):
            return float("nan")
        return float(np.mean(tms_power[:, sel]))

    return {
        "mel_mod_band_dc_to_1hz_mean": band_mean(0.0, 1.0),
        "mel_mod_band_1_to_4hz_mean": band_mean(1.0, 4.0),
        "mel_mod_band_4_to_8hz_mean": band_mean(4.0, 8.0),
        "mel_mod_band_above_8hz_mean": band_mean(8.0, None),
    }


class MelSecondaryExtractor(BaseExtractor):
    name = "mel_secondary"

    def extract(self, ctx: ClipContext) -> ExtractionResult:
        feats: dict[str, float] = {}
        warns: list[str] = []
        mel = ctx.mel
        sr = float(ctx.frame_rate_hz)

        if mel is None or mel.ndim != 2 or mel.size == 0:
            warns.append("mel_secondary: missing mel tensor for modulation/depletion")
            e = ctx.mel_mean_energy
            if e is not None and e.size >= 3:
                T = e.size
                third = max(1, T // 3)
                early = float(np.mean(e[:third]))
                late = float(np.mean(e[-third:]))
                denom = float(np.mean(np.abs(e))) + 1e-8
                feats["mel_energy_depletion_ratio"] = (early - late) / denom
                feats["mel_phonation_stability_cv"] = (
                    float(np.std(e) / (np.mean(np.abs(e)) + 1e-8))
                )
            return ExtractionResult(feats, {"extractor": self.name}, warns)

        energy_t = np.mean(mel, axis=0)
        T = energy_t.size
        third = max(1, T // 3)
        early = float(np.mean(energy_t[:third]))
        late = float(np.mean(energy_t[-third:]))
        denom = float(np.mean(np.abs(energy_t))) + 1e-8
        feats["mel_energy_depletion_ratio"] = (early - late) / denom
        feats["mel_phonation_stability_cv"] = float(
            np.std(energy_t) / (np.mean(np.abs(energy_t)) + 1e-8)
        )

        d = np.diff(mel, axis=1)
        feats["mel_modulation_abs_mean"] = float(np.mean(np.abs(d))) if d.size else float("nan")

        feats.update(_modulation_band_energy(mel, sr))
        return ExtractionResult(feats, {"extractor": self.name}, warns)
