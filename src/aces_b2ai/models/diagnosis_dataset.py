from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from aces_b2ai.models.feature_bundle import FeatureBundle
from aces_b2ai.models.model_config import DiagnosisModelConfig


class DiagnosisDataset(Dataset):
    """PyTorch Dataset wrapping a list of ``FeatureBundle`` objects.

    Each item is a dict of tensors ready for ``FusionModel.forward()``, plus
    ``"label_binary"`` and ``"label_multiclass"`` entries.

    Parameters
    ----------
    bundles:
        One ``FeatureBundle`` per clip (not per participant). Aggregate across
        clips before constructing if participant-level labels are needed.
    cfg:
        Controls which feature keys are included.  Keys not present in a
        bundle are returned as zero tensors with shape ``(C, 1)`` (series) or
        ``(F, 1)`` (spectrogram).
    """

    def __init__(
        self,
        bundles: list[FeatureBundle],
        cfg: DiagnosisModelConfig,
    ) -> None:
        cfg.validate()
        self.bundles = bundles
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.bundles)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        b = self.bundles[idx]
        item: dict[str, Any] = {}

        # ---- Spectrogram tensors -------------------------------------------
        for key in self.cfg.spectrogram_feature_keys:
            arr = b.spectrograms.get(key)
            if arr is None:
                freq_bins = self.cfg.spectrogram_freq_bins.get(key, 1)
                arr = np.zeros((freq_bins, 1), dtype=np.float32)
            item[key] = torch.from_numpy(arr.astype(np.float32))  # (F, T)

        # ---- Time-series tensors -------------------------------------------
        for key in self.cfg.series_feature_keys:
            arr = b.series.get(key)
            if arr is None:
                n_ch = self.cfg.series_n_channels.get(key, 1)
                arr = np.zeros((n_ch, 1), dtype=np.float32)
            item[key] = torch.from_numpy(arr.astype(np.float32))  # (C, T)

        # ---- Mel for transformer branch ------------------------------------
        if self.cfg.enable_transformer_branch:
            mel_arr = b.spectrograms.get("mel")
            if mel_arr is None:
                mel_arr = np.zeros(
                    (self.cfg.transformer.input_mel_bins, 1), dtype=np.float32
                )
            item["mel"] = torch.from_numpy(mel_arr.astype(np.float32))

            # EMA position channels for SPARC augmentation
            if self.cfg.transformer.sparc_channels > 0:
                sparc_arr = b.series.get("ema_sparc")
                if sparc_arr is None:
                    sparc_arr = np.zeros(
                        (self.cfg.transformer.sparc_channels, 1), dtype=np.float32
                    )
                item["sparc_series"] = torch.from_numpy(sparc_arr.astype(np.float32))

            # Sparse scalar conditioning vector
            if self.cfg.transformer.sparse_keys:
                values = [
                    float(b.scalars.get(k) or 0.0)
                    for k in self.cfg.transformer.sparse_keys
                ]
                item["sparse_vector"] = torch.tensor(values, dtype=torch.float32)

        # ---- Labels --------------------------------------------------------
        item["label_binary"] = torch.tensor(
            b.label_binary if b.label_binary is not None else -1,
            dtype=torch.float32,
        )
        item["label_multiclass"] = torch.tensor(
            b.label_multiclass if b.label_multiclass is not None else -1,
            dtype=torch.long,
        )

        return item


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length tensors to the maximum length in the batch.

    All 2D tensors ``(C, T_i)`` in the batch are right-zero-padded to
    ``(C, T_max)`` before stacking into ``(B, C, T_max)``.
    """
    out: dict[str, Any] = {}
    keys = batch[0].keys()

    for key in keys:
        vals = [item[key] for item in batch]

        if isinstance(vals[0], torch.Tensor) and vals[0].dim() == 2:
            # (C, T_i) → pad to (C, T_max)
            t_max = max(v.size(1) for v in vals)
            padded = []
            for v in vals:
                if v.size(1) < t_max:
                    pad = torch.zeros(v.size(0), t_max - v.size(1), dtype=v.dtype)
                    v = torch.cat([v, pad], dim=1)
                padded.append(v)
            out[key] = torch.stack(padded)  # (B, C, T_max)
        elif isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals)    # (B, ...) — fixed-width tensors incl. sparse_vector
        else:
            out[key] = vals

    return out
