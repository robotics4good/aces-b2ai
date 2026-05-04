from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TransformerBranchConfig:
    """Configuration for the MelHuBERT-based TransformerBranch.

    Checkpoint
    ----------
    ckpt_path:
        Path to a MelHuBERT ``.ckpt`` file produced by ``pretrain_expert.py``.
        ``None`` means train from scratch using the architecture defaults below.
        The repo checkpoint lives at ``weights/melhubert-20ms-stg2.ckpt``.
    freeze_encoder:
        Freeze all MelHuBERT backbone parameters at construction time.
        Recommended for warmup; call ``branch.unfreeze_encoder()`` after N steps.
    unfreeze_after_steps:
        Informational hint for the training loop — not enforced internally.

    Input alignment
    ---------------
    input_mel_bins:
        Mel bins in the tensor fed to this branch (typically 80).
    melhubert_feat_dim:
        Must equal ``feat_emb_dim`` in the checkpoint config.
        20 ms checkpoints: 80 (two 40-bin frames stacked during pretraining).
    use_frame_stacking:
        When True AND ``input_mel_bins == 40``, pairs of consecutive frames are
        stacked: (B, 40, T) → (B, T//2, 80), matching the 20 ms pretraining
        convention.  When False or bins already 80, frames are used as-is.

    SPARC augmentation
    ------------------
    sparc_channels:
        Number of EMA position channels to concatenate onto each mel frame
        before the input projection.  0 = disabled.

        WHY THIS CAN FAIL (and the fix applied here):
        The checkpoint's ``pre_extract_proj`` maps exactly 80 → 768.  Adding
        SPARC channels changes the input width to 80+S, which would cause a
        shape mismatch when loading that weight.

        Fix: when ``sparc_channels > 0`` the backbone is built with
        ``feat_emb_dim = embed_dim`` (768), which makes ``MelHuBERTModel``
        set ``pre_extract_proj = None`` (dimensions are equal → no projection
        needed inside the backbone).  A separate ``TransformerBranch.input_proj
        = Linear(80+S → 768)`` is added externally and trained from scratch.
        The checkpoint is loaded with ``strict=False``; the two
        ``pre_extract_proj.*`` tensors are silently skipped while all 12
        encoder transformer layers load correctly (199/201 tensors).

        Use ``sparc_channels=12`` to include all 12 EMA position channels.
        Do NOT use raw jerk time-series here — the 3rd derivative amplifies
        noise at 50 Hz.  Use jerk *scalars* in ``sparse_keys`` instead.

    Sparse scalar conditioning
    --------------------------
    sparse_keys:
        List of scalar feature names from ``bundle.scalars`` used for
        utterance-level cross-attention conditioning.  Good candidates:
        ``["jerk_tbx", "jerk_tby", "f0_mean", "hnr_mean"]``.
        Empty list = no conditioning.
    sparse_dim:
        Dimension of the projected sparse vector fed into the cross-attention.

    Architecture
    ------------
    embed_dim:
        Output dimension of this branch (= ``encoder_embed_dim`` in the ckpt).
        768 for the Base checkpoint.
    n_encoder_layers:
        Layers used when building from scratch.  Always ignored when
        ``load_pretrained()`` is called (the checkpoint determines layer count).
    pool:
        How to collapse the time axis after the encoder.
        ``"mean"`` — unweighted mean over valid frames.
        ``"weighted_sum"`` — learnable per-layer scalar weights, then mean.
    """

    ckpt_path: str | None = None
    freeze_encoder: bool = True
    unfreeze_after_steps: int = 1000

    input_mel_bins: int = 80
    melhubert_feat_dim: int = 80
    use_frame_stacking: bool = False

    sparc_channels: int = 0

    sparse_keys: list[str] = field(default_factory=list)
    sparse_dim: int = 64

    embed_dim: int = 768
    n_encoder_layers: int = 2
    pool: str = "mean"


@dataclass
class DiagnosisModelConfig:
    """Controls which feature keys flow into which model branch.

    All ``*_feature_keys`` lists are order-preserving: each key maps to one
    branch instance, and embeddings are concatenated in that order.

    Parameters
    ----------
    scalar_feature_keys:
        SCALAR features routed to the tabular sklearn branch.
        If empty and ``enable_scalar_branch`` is True, all scalars in the
        bundle are used.
    spectrogram_feature_keys:
        SPECTROGRAM feature keys for the 2D CNN branch.
    spectrogram_freq_bins:
        Maps each key in ``spectrogram_feature_keys`` to its freq-bin count.
        Required when ``enable_spectrogram_branch`` is True.
    series_feature_keys:
        TIMESERIES feature keys for the 1D TCN branch.
    series_n_channels:
        Maps each key in ``series_feature_keys`` to its channel count.
        Required when ``enable_series_branch`` is True.
    enable_scalar_branch:
        Enable the sklearn tabular branch (Stage 1).
    enable_spectrogram_branch:
        Enable the 2D CNN branch (Stage 2).
    enable_series_branch:
        Enable the 1D TCN branch (Stage 2).
    embed_dim:
        Output dimensionality of each deep learning branch before fusion.
    patch_frames:
        Number of time frames in each spectrogram patch (2D CNN branch).
    tcn_n_blocks:
        Number of dilated TCN blocks.
    dropout:
        Dropout rate applied in all deep learning branches.
    """

    # --- Feature selection ---------------------------------------------------
    scalar_feature_keys: list[str] = field(default_factory=list)
    spectrogram_feature_keys: list[str] = field(default_factory=list)
    spectrogram_freq_bins: dict[str, int] = field(default_factory=dict)
    series_feature_keys: list[str] = field(default_factory=list)
    series_n_channels: dict[str, int] = field(default_factory=dict)

    # --- Branch toggles ------------------------------------------------------
    enable_scalar_branch: bool = True
    enable_spectrogram_branch: bool = False
    enable_series_branch: bool = False

    # --- Architecture hyper-parameters ---------------------------------------
    embed_dim: int = 128
    patch_frames: int = 128
    tcn_n_blocks: int = 4
    dropout: float = 0.4

    # --- Transformer branch --------------------------------------------------
    enable_transformer_branch: bool = False
    transformer: TransformerBranchConfig = field(
        default_factory=TransformerBranchConfig
    )

    # --- Task sizes (set before building FusionModel) ------------------------
    n_classes_binary: int = 2
    n_classes_multiclass: int = 7

    def validate(self) -> None:
        """Raise ``ValueError`` if the config is inconsistent."""
        if self.enable_spectrogram_branch:
            missing = [k for k in self.spectrogram_feature_keys if k not in self.spectrogram_freq_bins]
            if missing:
                raise ValueError(
                    f"spectrogram_freq_bins missing entries for keys: {missing}"
                )
        if self.enable_series_branch:
            missing = [k for k in self.series_feature_keys if k not in self.series_n_channels]
            if missing:
                raise ValueError(
                    f"series_n_channels missing entries for keys: {missing}"
                )
        if (
            not self.enable_scalar_branch
            and not self.enable_spectrogram_branch
            and not self.enable_series_branch
            and not self.enable_transformer_branch
        ):
            raise ValueError("At least one branch must be enabled.")
