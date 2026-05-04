"""MelHuBERT-based TransformerBranch for the FusionModel.

Architecture overview
---------------------
                                          ┌─────────────────────────────┐
  mel (B,F,T)  ──► MelSPARCAugmentation ─►  input_proj (Linear or Id)  │
  sparc (B,S,T)   [stack frames, concat]    └────────────┬────────────┘
                                                         │  (B, T_out, feat_emb_dim)
                                            ┌────────────▼────────────┐
                                            │  MelHuBERTModel encoder  │
                                            │  (frozen or unfrozen)    │
                                            └────────────┬────────────┘
  sparse_vector (B, sparse_dim) ──────────► SparseConditionedLayer    │
                                            (cross-attn: hidden←sparse)│
                                            └────────────┬────────────┘
                                                mean / weighted_sum pool
                                                         │
                                               embedding (B, embed_dim)

SPARC augmentation failure — explained
---------------------------------------
The checkpoint's ``pre_extract_proj`` maps exactly 80 → 768.  Concatenating
SPARC channels changes the input to 80+S, which would crash
``load_state_dict(strict=True)`` because the shape no longer matches.

The fix (two-mode design):
  • sparc_channels == 0  →  backbone built with feat_emb_dim=80 (matches ckpt)
                             input_proj = Identity(); load with strict=True
  • sparc_channels  > 0  →  backbone built with feat_emb_dim=768
                             (= encoder_embed_dim, so MelHuBERTModel sets
                             pre_extract_proj=None — no internal projection)
                             input_proj = Linear(80+S → 768), random init
                             load with strict=False; the two
                             pre_extract_proj.* tensors are skipped;
                             all 12 encoder transformer layers still load
                             correctly (199 / 201 tensors)

Jerk channels: do NOT put raw jerk time-series in sparc_channels.
The 3rd derivative at 50 Hz is dominated by noise.  Use jerk *scalars*
(e.g. "jerk_tbx", "jerk_tby") in TransformerBranchConfig.sparse_keys for
utterance-level conditioning via cross-attention instead.
"""
from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from aces_b2ai.models.model_config import TransformerBranchConfig


class MelSPARCAugmentation(nn.Module):
    """Prepare mel + optional EMA position channels for the encoder.

    Steps
    -----
    1.  Optional frame stacking: pairs of consecutive 40-bin frames are
        concatenated along the feature axis → (B, T//2, 80).  Matches the
        20 ms MelHuBERT pretraining convention.
    2.  Optional SPARC concat: EMA position channels are interpolated to the
        stacked mel frame rate and appended → (B, T_out, 80+S).
        Interpolation (nearest-neighbour) handles both the 50Hz→50Hz no-op
        and the future 50Hz→100Hz case transparently.

    Why not jerk time-series?
        Jerk is the 3rd derivative of EMA position.  Each differentiation
        amplifies high-frequency noise; by the 3rd derivative the SNR at 50 Hz
        is very poor.  EMA *positions* (smooth, slow-varying) are the correct
        signal for frame-level augmentation.  Jerk scalars belong in
        TransformerBranchConfig.sparse_keys for utterance-level conditioning.
    """

    def __init__(self, cfg: "TransformerBranchConfig") -> None:
        super().__init__()
        self.use_frame_stacking = cfg.use_frame_stacking
        self.input_mel_bins = cfg.input_mel_bins
        self.sparc_channels = cfg.sparc_channels

    def forward(
        self,
        mel: torch.Tensor,
        sparc_series: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        mel : (B, F, T)
        sparc_series : (B, S, T_sparc) or None — EMA position channels at
            their native frame rate (typically 50 Hz).

        Returns
        -------
        (B, T_out, F_aug)  where F_aug = melhubert_feat_dim [+ sparc_channels]
        """
        # 1. Transpose to time-first
        mel_t = mel.transpose(1, 2)  # (B, T, F)

        # 2. Frame stacking (only when 40-bin mel is supplied)
        if self.use_frame_stacking and self.input_mel_bins == 40:
            odd = mel_t[:, 0::2, :]
            even = mel_t[:, 1::2, :]
            min_t = min(odd.shape[1], even.shape[1])
            out = torch.cat([odd[:, :min_t, :], even[:, :min_t, :]], dim=-1)
        else:
            out = mel_t  # (B, T, F)

        T_out = out.shape[1]

        # 3. SPARC channel concat with automatic resampling
        if sparc_series is not None and self.sparc_channels > 0:
            # Nearest-neighbour resample along the time axis to match T_out.
            # This is a no-op when sparc is already at 50 Hz and mel is also
            # at 50 Hz after frame stacking; it 2× upsamples when mel is at
            # 100 Hz (10 ms frames, no stacking).
            sparc_resampled = F.interpolate(
                sparc_series.float(),  # (B, S, T_sparc)
                size=T_out,
                mode="nearest",
            )  # → (B, S, T_out)
            sparc_t = sparc_resampled.transpose(1, 2)  # (B, T_out, S)
            out = torch.cat([out, sparc_t], dim=-1)

        return out  # (B, T_out, F_aug)


class SparseConditionedLayer(nn.Module):
    """FiLM conditioning: inject utterance-level scalars into frame representations.

    Uses Feature-wise Linear Modulation (scale + shift) rather than full
    MultiheadAttention.  With kv_len=1 (a single conditioning token), MHA
    reduces to a weighted projection anyway, but costs 4×embed_dim² ≈ 2.36M
    params for embed_dim=768 — far too many for small clinical datasets (<200
    clips).  FiLM achieves the same representational effect with only
    2 × sparse_dim × embed_dim ≈ 10K params.

    Mechanism
    ---------
    scale, shift = Linear(sparse_dim → embed_dim) each
    out = LayerNorm(hidden * (1 + scale) + shift)

    The ``(1 + scale)`` formulation initialises close to identity
    (scale ≈ 0 at init) so the residual stream starts undisturbed.
    """

    def __init__(self, embed_dim: int, sparse_dim: int, n_heads: int = 8) -> None:  # n_heads kept for API compat
        super().__init__()
        self.scale_proj = nn.Linear(sparse_dim, embed_dim)
        self.shift_proj = nn.Linear(sparse_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        # Initialise close to identity: scale≈0, shift≈0
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)

    def forward(
        self, hidden: torch.Tensor, sparse_vec: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden : (B, T, embed_dim)
        sparse_vec : (B, sparse_dim)

        Returns
        -------
        (B, T, embed_dim) — conditioned hidden states
        """
        # Guard: replace any NaN/Inf that survived data loading
        sparse_vec = torch.nan_to_num(sparse_vec, nan=0.0, posinf=5.0, neginf=-5.0)
        scale = self.scale_proj(sparse_vec).unsqueeze(1)   # (B, 1, embed_dim)
        shift = self.shift_proj(sparse_vec).unsqueeze(1)   # (B, 1, embed_dim)
        return self.norm(hidden * (1.0 + scale) + shift)


class TransformerBranch(nn.Module):
    """MelHuBERT backbone + optional SPARC augmentation + sparse conditioning.

    The branch accepts mel spectrograms (and optionally EMA position channels
    and scalar conditioning vectors) and produces a fixed-size embedding per
    utterance.

    Typical usage (fine-tuning from the provided checkpoint)
    ---------------------------------------------------------
    ::

        from aces_b2ai.models.model_config import TransformerBranchConfig
        from aces_b2ai.models.branches import TransformerBranch

        cfg = TransformerBranchConfig(
            ckpt_path="weights/melhubert-20ms-stg2.ckpt",
            freeze_encoder=True,          # unfreeze after warmup steps
            melhubert_feat_dim=80,
            sparc_channels=12,            # include EMA positions
            sparse_keys=["jerk_tbx", "jerk_tby", "f0_mean"],
            sparse_dim=64,
            embed_dim=768,
        )
        branch = TransformerBranch(cfg)
        branch.load_pretrained(cfg.ckpt_path)

    See ``TransformerBranchConfig`` docstring for full explanation of the SPARC
    augmentation failure mode and the two-mode fix.
    """

    def __init__(self, cfg: "TransformerBranchConfig") -> None:
        super().__init__()
        self.cfg = cfg

        self.augmentation = MelSPARCAugmentation(cfg)

        # ------------------------------------------------------------------ #
        # Two-mode input projection (see module docstring for why)            #
        #                                                                      #
        # Mode A — no SPARC (sparc_channels == 0):                            #
        #   backbone feat_emb_dim = 80 → matches checkpoint pre_extract_proj  #
        #   input_proj = Identity (no-op)                                     #
        #   load_state_dict strict=True                                        #
        #                                                                      #
        # Mode B — SPARC augmentation (sparc_channels > 0):                   #
        #   backbone feat_emb_dim = 768 = encoder_embed_dim                   #
        #   → MelHuBERTModel sets pre_extract_proj=None (skips its proj)      #
        #   input_proj = Linear(80+S → 768), random init, trains from scratch #
        #   load_state_dict strict=False; pre_extract_proj.* keys skipped;    #
        #   all 12 transformer encoder layers still load (199/201 tensors)    #
        # ------------------------------------------------------------------ #
        if cfg.sparc_channels == 0:
            self.input_proj: nn.Module = nn.Identity()
            self._backbone_feat_dim = cfg.melhubert_feat_dim
            self._load_strict = True
        else:
            self.input_proj = nn.Linear(
                cfg.melhubert_feat_dim + cfg.sparc_channels,
                cfg.embed_dim,
            )
            self._backbone_feat_dim = cfg.embed_dim   # 768 → no pre_extract_proj
            self._load_strict = False

        self.backbone = self._build_backbone_from_scratch()

        if cfg.sparse_keys:
            self.sparse_cond: SparseConditionedLayer | None = SparseConditionedLayer(
                embed_dim=cfg.embed_dim,
                sparse_dim=cfg.sparse_dim,
                n_heads=max(1, cfg.embed_dim // 64),
            )
        else:
            self.sparse_cond = None

        if cfg.pool == "weighted_sum":
            self.layer_weights: nn.Parameter | None = nn.Parameter(
                torch.ones(cfg.n_encoder_layers) / cfg.n_encoder_layers
            )
        else:
            self.layer_weights = None

        if cfg.freeze_encoder:
            self.freeze_encoder()

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _build_backbone_from_scratch(self):
        from aces_b2ai.models.melhubert_vendor import MelHuBERTConfig, MelHuBERTModel

        raw_cfg = {
            "feat_emb_dim": self._backbone_feat_dim,
            "encoder_layers": self.cfg.n_encoder_layers,
            "encoder_embed_dim": self.cfg.embed_dim,
            "encoder_ffn_embed_dim": self.cfg.embed_dim * 4,
            "encoder_attention_heads": max(1, self.cfg.embed_dim // 64),
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "activation_dropout": 0.1,
            "encoder_layerdrop": 0.0,
            "mask_prob": 0.0,   # masking disabled; fine-tuning always passes mask=False
            "num_cluster": 512,
            "final_dim": 40,
        }
        return MelHuBERTModel(MelHuBERTConfig(raw_cfg))

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def load_pretrained(self, ckpt_path: str) -> None:
        """Load backbone weights from a MelHuBERT ``.ckpt`` file.

        The checkpoint is self-describing: its own architecture config is stored
        under ``all_states["Upstream_Config"]["melhubert"]``.  We always
        reconstruct the backbone from that stored config (full 12-layer
        architecture) — ``cfg.n_encoder_layers`` is NOT applied here.

        PyTorch ≥ 2.6 fix
        ------------------
        The checkpoint stores ``Args`` as ``argparse.Namespace``, which the new
        ``weights_only=True`` default rejects.  We add it to safe globals and
        use ``weights_only=False`` (safe because we own and trust this file).

        SPARC strict mode
        -----------------
        When ``sparc_channels > 0``, we patch ``feat_emb_dim=768`` into the
        stored config before reconstructing, so the backbone is built without
        its own ``pre_extract_proj``.  Loading with ``strict=False`` silently
        skips the two ``pre_extract_proj.*`` checkpoint tensors; all encoder
        layers load correctly.
        """
        from aces_b2ai.models.melhubert_vendor import MelHuBERTConfig, MelHuBERTModel

        torch.serialization.add_safe_globals([argparse.Namespace])
        all_states = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        upstream_cfg = dict(all_states["Upstream_Config"]["melhubert"])

        if self.cfg.sparc_channels > 0:
            # Force backbone to skip pre_extract_proj so SPARC path works
            upstream_cfg["feat_emb_dim"] = self.cfg.embed_dim

        # Capture current device before we replace self.backbone.
        # load_pretrained may be called after model.to(device), so the old
        # backbone is already on the target device.  The new MelHuBERTModel()
        # is always CPU-allocated; we must move it back afterwards.
        try:
            target_device = next(self.backbone.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")

        model_cfg = MelHuBERTConfig(upstream_cfg)
        new_backbone = MelHuBERTModel(model_cfg)

        if self._load_strict:
            missing, unexpected = new_backbone.load_state_dict(
                all_states["model"], strict=True
            )
        else:
            res = new_backbone.load_state_dict(all_states["model"], strict=False)
            missing, unexpected = res.missing_keys, res.unexpected_keys

        if not self._load_strict and missing:
            # Only pre_extract_proj keys are expected to be missing
            unexpected_missing = [k for k in missing if "pre_extract_proj" not in k]
            if unexpected_missing:
                raise RuntimeError(
                    f"Unexpected missing keys when loading checkpoint: {unexpected_missing}"
                )

        # Move to the same device as the rest of the model before assigning
        self.backbone = new_backbone.to(target_device)

        # Update layer_weights size if using weighted_sum pool and
        # the checkpoint has a different layer count than n_encoder_layers
        n_layers = model_cfg.encoder_layers
        if self.layer_weights is not None and len(self.layer_weights) != n_layers:
            self.layer_weights = nn.Parameter(
                torch.ones(n_layers, device=target_device) / n_layers
            )

        if self.cfg.freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        """Freeze all MelHuBERT backbone parameters."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_encoder(self, top_n: int | None = None) -> None:
        """Unfreeze MelHuBERT backbone parameters.

        Parameters
        ----------
        top_n : int | None
            If None, unfreeze all 12 layers (risky with small data).
            If set, unfreeze only the last *top_n* transformer layers plus the
            final layer norm — keeps early layers frozen to preserve low-level
            speech features learned on adult data, while adapting the top layers
            to pediatric / pathological speech.

            Recommended for small datasets (<500 clips): top_n=2 or top_n=3.
        """
        if top_n is None:
            for p in self.backbone.parameters():
                p.requires_grad_(True)
            return

        # Always freeze everything first, then selectively unfreeze
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        encoder = self.backbone.encoder  # TransformerEncoder
        n_layers = len(encoder.layers)
        for layer in encoder.layers[n_layers - top_n:]:
            for p in layer.parameters():
                p.requires_grad_(True)

        # Also unfreeze the final layer norm
        if hasattr(encoder, "layer_norm") and encoder.layer_norm is not None:
            for p in encoder.layer_norm.parameters():
                p.requires_grad_(True)

    def get_param_groups(self) -> list[dict]:
        """Return optimizer param groups with per-component LR scaling.

        Unfrozen encoder params get 0.1× LR to prevent catastrophic forgetting
        of the adult-speech representations.  All other params (FiLM conditioning,
        head) get 1.0×.
        """
        encoder_params = [p for p in self.backbone.parameters() if p.requires_grad]
        encoder_ids = {id(p) for p in encoder_params}
        other_params = [p for p in self.parameters()
                        if p.requires_grad and id(p) not in encoder_ids]
        groups = [{"params": other_params, "lr_scale": 1.0}]
        if encoder_params:
            groups.append({"params": encoder_params, "lr_scale": 0.1})
        return groups

    def forward(
        self,
        mel: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        sparc_series: torch.Tensor | None = None,
        sparse_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        mel : (B, F, T) — mel spectrogram, channels-first
        pad_mask : (B, T) float, 1 = valid frame, 0 = padding.
            None means all frames are valid.
        sparc_series : (B, S, T_sparc) — EMA position channels.
            Will be resampled to match stacked mel length automatically.
            Pass None when sparc_channels == 0.
        sparse_vector : (B, sparse_dim) — aggregated scalar features for
            cross-attention conditioning (e.g. jerk scalars, F0 mean).
            Pass None when sparse_keys is empty.

        Returns
        -------
        (B, embed_dim) utterance embedding
        """
        B, F, T = mel.shape

        # 1. Augment frames + optionally concat SPARC channels
        feat = self.augmentation(mel, sparc_series)   # (B, T_out, F_aug)
        feat = self.input_proj(feat)                  # (B, T_out, embed_dim)
        T_out = feat.shape[1]

        # 2. Build padding mask for backbone (1=valid matches MelHuBERT convention)
        if pad_mask is None:
            bk_pad = torch.ones(B, T_out, device=mel.device)
        elif T_out < T:
            # Frame stacking halved T — downsample pad_mask to match
            bk_pad = pad_mask[:, 0::2][:, :T_out]
        else:
            bk_pad = pad_mask[:, :T_out]

        # 3. Run backbone — no masking during fine-tuning
        out = self.backbone(feat, bk_pad, no_pred=True, get_hidden=True)
        last_hidden = out[0]       # (B, T_out, embed_dim)
        layer_hiddens = out[5]     # list of (B, T_out, embed_dim) per layer

        # 4. Optional sparse conditioning via cross-attention
        if self.sparse_cond is not None and sparse_vector is not None:
            last_hidden = self.sparse_cond(last_hidden, sparse_vector)

        # 5. Pool over time axis
        if self.layer_weights is not None and layer_hiddens:
            w = torch.softmax(self.layer_weights, dim=0)
            all_layers = torch.stack(layer_hiddens, dim=0)        # (L, B, T, D)
            last_hidden = (w[:, None, None, None] * all_layers).sum(0)

        mask_f = bk_pad.unsqueeze(-1).float()                     # (B, T, 1)
        embed = (last_hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        return embed   # (B, embed_dim)
