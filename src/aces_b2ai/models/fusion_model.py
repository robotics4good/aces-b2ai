from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from aces_b2ai.models.branches.spectrogram_branch import SpectrogramBranch
from aces_b2ai.models.branches.timeseries_branch import TimeSeriesBranch
from aces_b2ai.models.branches.transformer_branch import TransformerBranch
from aces_b2ai.models.model_config import DiagnosisModelConfig


class FusionModel(nn.Module):
    """Multi-branch classifier that fuses any combination of enabled branches.

    Branch embeddings are concatenated and passed through a 2-layer MLP,
    then split into two heads:

    - **binary** head: healthy (0) vs. pathological (1)  — BCEWithLogitsLoss
    - **multiclass** head: specific disorder index        — CrossEntropyLoss

    Multi-task loss: ``L = alpha * L_binary + (1 - alpha) * L_multiclass``

    Parameters
    ----------
    cfg:
        Must pass ``cfg.validate()`` before constructing.
    """

    def __init__(self, cfg: DiagnosisModelConfig) -> None:
        cfg.validate()
        super().__init__()
        self.cfg = cfg

        total_embed = 0

        # ---- Spectrogram branches (one per key) ----------------------------
        if cfg.enable_spectrogram_branch and cfg.spectrogram_feature_keys:
            self.spec_branches = nn.ModuleDict(
                {
                    key: SpectrogramBranch(
                        freq_bins=cfg.spectrogram_freq_bins[key],
                        embed_dim=cfg.embed_dim,
                        patch_frames=cfg.patch_frames,
                        dropout=cfg.dropout,
                    )
                    for key in cfg.spectrogram_feature_keys
                }
            )
            total_embed += cfg.embed_dim * len(cfg.spectrogram_feature_keys)
        else:
            self.spec_branches = nn.ModuleDict()

        # ---- Time-series branches (one per key) ----------------------------
        if cfg.enable_series_branch and cfg.series_feature_keys:
            self.series_branches = nn.ModuleDict(
                {
                    key: TimeSeriesBranch(
                        n_channels=cfg.series_n_channels[key],
                        embed_dim=cfg.embed_dim,
                        n_blocks=cfg.tcn_n_blocks,
                        dropout=cfg.dropout,
                    )
                    for key in cfg.series_feature_keys
                }
            )
            total_embed += cfg.embed_dim * len(cfg.series_feature_keys)
        else:
            self.series_branches = nn.ModuleDict()

        # ---- Transformer branch -------------------------------------------
        if cfg.enable_transformer_branch:
            self.transformer_branch: TransformerBranch | None = TransformerBranch(
                cfg.transformer
            )
            if cfg.transformer.ckpt_path:
                self.transformer_branch.load_pretrained(cfg.transformer.ckpt_path)
            total_embed += cfg.transformer.embed_dim
        else:
            self.transformer_branch = None

        # cfg.validate() already ensures at least one branch is enabled,
        # so total_embed == 0 here is only reachable if validate() was bypassed.
        if total_embed == 0:
            raise ValueError(
                "FusionModel has no deep learning branches. "
                "Enable at least one of enable_spectrogram_branch / "
                "enable_series_branch / enable_transformer_branch."
            )

        # ---- Fusion MLP ---------------------------------------------------
        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_embed, 256),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 128),
            nn.GELU(),
        )

        # ---- Classification heads -----------------------------------------
        self.binary_head = nn.Linear(128, 1)
        self.multiclass_head = nn.Linear(128, cfg.n_classes_multiclass)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        batch:
            Dict mapping feature key → tensor.
            Spectrogram keys: ``(B, freq_bins, T)``
            Series keys:      ``(B, n_channels, T)``

        Returns
        -------
        Dict with keys ``"binary"`` (B, 1) and ``"multiclass"`` (B, n_classes).
        Logits (pre-sigmoid / pre-softmax).
        """
        embeds: list[torch.Tensor] = []

        for key, branch in self.spec_branches.items():
            embeds.append(branch(batch[key]))

        for key, branch in self.series_branches.items():
            embeds.append(branch(batch[key]))

        if self.transformer_branch is not None:
            embeds.append(
                self.transformer_branch(
                    mel=batch["mel"],
                    pad_mask=batch.get("mel_pad_mask"),
                    sparc_series=batch.get("sparc_series"),
                    sparse_vector=batch.get("sparse_vector"),
                )
            )

        fused = torch.cat(embeds, dim=-1)
        h = self.fusion_mlp(fused)

        return {
            "binary": self.binary_head(h),
            "multiclass": self.multiclass_head(h),
        }

    def get_param_groups(self, base_lr: float = 1e-4) -> list[dict]:
        """Return optimizer param groups with per-component LR scaling.

        When a transformer branch is present the encoder receives 0.1× the
        base LR to prevent catastrophic forgetting.  All other parameters
        receive the full base LR.
        """
        if self.transformer_branch is None:
            return [{"params": list(self.parameters()), "lr": base_lr}]

        transformer_ids: set[int] = set()
        groups: list[dict] = []
        for g in self.transformer_branch.get_param_groups():
            groups.append({"params": g["params"], "lr": base_lr * g["lr_scale"]})
            transformer_ids.update(id(p) for p in g["params"])

        other_params = [p for p in self.parameters() if id(p) not in transformer_ids]
        if other_params:
            groups.append({"params": other_params, "lr": base_lr})
        return groups

    @staticmethod
    def compute_loss(
        outputs: dict[str, torch.Tensor],
        label_binary: torch.Tensor,
        label_multiclass: torch.Tensor,
        alpha: float = 0.5,
        ignore_index: int = -1,
    ) -> torch.Tensor:
        """Weighted multi-task loss.

        Parameters
        ----------
        outputs:
            Output dict from ``forward()``.
        label_binary:
            (B,) float tensor, 0.0 or 1.0.
        label_multiclass:
            (B,) long tensor.  Use ``ignore_index`` (-1) for samples where the
            multi-class label is unknown.
        alpha:
            Weight for binary loss; ``(1 - alpha)`` for multiclass loss.
        """
        # Binary: BCEWithLogitsLoss on valid samples
        l_bin = F.binary_cross_entropy_with_logits(
            outputs["binary"].squeeze(-1),
            label_binary.float(),
            reduction="mean",
        )

        # Multiclass: cross-entropy, ignoring samples with label == ignore_index
        valid = label_multiclass != ignore_index
        if valid.any():
            l_mc = F.cross_entropy(
                outputs["multiclass"][valid],
                label_multiclass[valid],
                reduction="mean",
            )
        else:
            l_mc = torch.zeros(1, device=outputs["binary"].device)

        return alpha * l_bin + (1.0 - alpha) * l_mc
