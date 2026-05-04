from __future__ import annotations

import torch
import torch.nn as nn


class _TemporalBlock(nn.Module):
    """Single dilated causal convolution block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal: trim right later

        self.conv1 = nn.utils.parametrize.register_parametrization if False else nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        # LayerNorm over the channel dimension (applied after transposing).
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )
        self._padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        out = self.conv1(x)[:, :, : x.size(2)]          # causal trim
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = self.act(self.dropout(out))

        out = self.conv2(out)[:, :, : x.size(2)]
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = self.act(self.dropout(out))

        res = x if self.downsample is None else self.downsample(x)
        return self.act(out + res)


class TimeSeriesBranch(nn.Module):
    """Generic 1D Temporal Convolutional Network for multi-channel time series.

    Uses exponentially increasing dilations so the receptive field grows to
    ``kernel_size * (2^n_blocks - 1)`` frames without heavy parameter cost.

    Parameters
    ----------
    n_channels:
        Number of input channels (e.g. 12 for EMA, 40 for PPG, 60 for MFCC).
    embed_dim:
        Output embedding dimensionality after global average pooling.
    n_blocks:
        Number of dilated TCN blocks.  Each block doubles the dilation.
    dropout:
        Dropout probability applied inside each block.
    """

    def __init__(
        self,
        n_channels: int,
        embed_dim: int = 128,
        n_blocks: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        hidden = max(embed_dim // 2, 32)
        kernel_size = 3
        blocks: list[nn.Module] = []
        for i in range(n_blocks):
            in_ch = n_channels if i == 0 else hidden
            out_ch = embed_dim if i == n_blocks - 1 else hidden
            blocks.append(
                _TemporalBlock(in_ch, out_ch, kernel_size, dilation=2**i, dropout=dropout)
            )
        self.network = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, n_channels, T)  float32

        Returns
        -------
        (batch, embed_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)   # (1, n_channels, T)

        out = self.network(x)  # (B, embed_dim, T)
        return out.mean(dim=-1)  # global average pool → (B, embed_dim)
