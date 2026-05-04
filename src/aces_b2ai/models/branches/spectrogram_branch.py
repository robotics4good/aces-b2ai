from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectrogramBranch(nn.Module):
    """Generic 2D CNN for any (freq_bins × T) time-frequency input.

    Treats the spectrogram as an image: freq is the height axis, time is the
    width axis.  Long recordings are processed in non-overlapping ``patch_frames``
    windows that are mean-pooled to produce one fixed-size embedding.

    Parameters
    ----------
    freq_bins:
        Height of the input (e.g. 60 for mel, 60 for MFCC, 201 for spectrogram).
    embed_dim:
        Output embedding dimensionality.
    patch_frames:
        Number of time frames per patch.  Clips shorter than this are
        zero-padded; clips longer are split and pooled.
    dropout:
        Dropout probability applied before the final linear projection.
    """

    def __init__(
        self,
        freq_bins: int,
        embed_dim: int = 128,
        patch_frames: int = 128,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.patch_frames = patch_frames

        # Two conv blocks, then adaptive pool to a fixed (4×4) spatial grid.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),                      # → (batch, 64*4*4 = 1024)
            nn.Linear(64 * 4 * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, freq_bins, T)  float32
            Un-batched first dimension accepted for single-sample inference.

        Returns
        -------
        (batch, embed_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)   # (1, freq_bins, T)

        B, F, T = x.shape
        P = self.patch_frames

        # Split into patches along time axis.
        patches = []
        for start in range(0, max(T, P), P):
            chunk = x[:, :, start : start + P]          # (B, F, <=P)
            if chunk.size(2) < P:
                pad = torch.zeros(B, F, P - chunk.size(2), device=x.device, dtype=x.dtype)
                chunk = torch.cat([chunk, pad], dim=2)  # right-pad
            patches.append(self.encoder(chunk.unsqueeze(1)))  # (B, embed_dim)

        return torch.stack(patches, dim=1).mean(dim=1)  # (B, embed_dim)
