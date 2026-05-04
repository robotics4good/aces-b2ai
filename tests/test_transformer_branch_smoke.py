"""Smoke tests for TransformerBranch and its FusionModel integration.

All tests are CPU-only and use from-scratch mode (no checkpoint file required).
Tests that exercise the real checkpoint skip automatically if the file is absent.

SPARC augmentation failure — what the tests verify
---------------------------------------------------
Pitfall: the checkpoint's pre_extract_proj expects exactly 80-dim input.
Adding sparc_channels changes the input to 80+S, causing a shape mismatch
on load_state_dict(strict=True).

Fix (tested in test_sparc_mode_skips_backbone_proj):
  sparc_channels > 0 → backbone built with feat_emb_dim=768=encoder_embed_dim
  → MelHuBERTModel sets pre_extract_proj=None (dims are equal, no projection)
  → TransformerBranch.input_proj = Linear(80+S → 768), randomly initialised
  → load_pretrained uses strict=False; pre_extract_proj.* keys are silently
     skipped; all encoder transformer layers still load (199/201 tensors).
"""
from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

CKPT_PATH = "weights/melhubert-20ms-stg2.ckpt"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(**kwargs):
    from aces_b2ai.models.model_config import TransformerBranchConfig

    defaults = dict(
        n_encoder_layers=1,
        embed_dim=64,
        melhubert_feat_dim=80,
        freeze_encoder=False,
        pool="mean",
    )
    defaults.update(kwargs)
    return TransformerBranchConfig(**defaults)


def _branch(**kwargs):
    from aces_b2ai.models.branches.transformer_branch import TransformerBranch

    return TransformerBranch(_cfg(**kwargs))


# ===========================================================================
# 1. From-scratch forward shape
# ===========================================================================


def test_from_scratch_forward_shape():
    branch = _branch()
    mel = torch.randn(2, 80, 100)
    out = branch(mel)
    assert out.shape == (2, 64), f"expected (2,64), got {out.shape}"


# ===========================================================================
# 2. Frame stacking dims
# ===========================================================================


def test_frame_stacking_correct_dims():
    branch = _branch(input_mel_bins=40, melhubert_feat_dim=80, use_frame_stacking=True)
    mel = torch.randn(2, 40, 100)
    out = branch(mel)
    assert out.shape == (2, 64), f"expected (2,64), got {out.shape}"


# ===========================================================================
# 3. SPARC augmentation mode — backbone has no pre_extract_proj
# ===========================================================================


def test_sparc_mode_skips_backbone_proj():
    """When sparc_channels > 0, backbone.pre_extract_proj must be None.

    This is the key invariant of the two-mode design that prevents the
    checkpoint shape mismatch described in the module docstring.
    """
    branch = _branch(sparc_channels=12, embed_dim=64)
    assert branch.backbone.pre_extract_proj is None, (
        "SPARC path must build backbone with feat_emb_dim=embed_dim so that "
        "pre_extract_proj is None inside MelHuBERTModel"
    )


def test_sparc_augmentation_forward():
    branch = _branch(sparc_channels=12, embed_dim=64)
    mel = torch.randn(2, 80, 100)
    sparc = torch.randn(2, 12, 100)
    out = branch(mel, sparc_series=sparc)
    assert out.shape == (2, 64), f"expected (2,64), got {out.shape}"


def test_sparc_augmentation_resamples_mismatched_frame_rate():
    """SPARC at 50 Hz, mel at 100 Hz → interpolation must not crash."""
    branch = _branch(sparc_channels=12, embed_dim=64)
    mel = torch.randn(2, 80, 200)   # 100 Hz (200 frames)
    sparc = torch.randn(2, 12, 100)  # 50 Hz (100 frames, half the mel length)
    out = branch(mel, sparc_series=sparc)
    assert out.shape == (2, 64)


# ===========================================================================
# 4. Sparse conditioning (cross-attention)
# ===========================================================================


def test_sparse_conditioning():
    branch = _branch(sparse_keys=["f0", "jerk"], sparse_dim=16, embed_dim=64)
    mel = torch.randn(2, 80, 100)
    sv = torch.randn(2, 16)
    out = branch(mel, sparse_vector=sv)
    assert out.shape == (2, 64)


# ===========================================================================
# 5. Gradient flows through unfrozen encoder
# ===========================================================================


def test_gradient_flows_through_encoder():
    """Encoder layers must accumulate gradients when unfrozen.

    Note: final_proj inside the backbone is unused in no_pred=True mode and
    will have grad=None.  We verify that the *encoder attention layers* receive
    grads, not every parameter in the module.
    """
    branch = _branch(freeze_encoder=False)
    mel = torch.randn(2, 80, 50)
    out = branch(mel)
    out.sum().backward()
    # Only check encoder transformer layer params (not final_proj which is unused)
    encoder_layer_grads = [
        p.grad
        for name, p in branch.backbone.named_parameters()
        if "encoder.layers" in name and p.requires_grad
    ]
    assert len(encoder_layer_grads) > 0, "no encoder layer params found"
    assert all(g is not None for g in encoder_layer_grads), (
        "encoder attention layer params should have grads when unfrozen"
    )


# ===========================================================================
# 6. Frozen encoder — backbone has no grad; input_proj does
# ===========================================================================


def test_frozen_encoder_no_grad():
    # sparc_channels > 0 so we have a trainable input_proj Linear
    branch = _branch(freeze_encoder=True, sparc_channels=12, embed_dim=64)
    mel = torch.randn(2, 80, 50)
    sparc = torch.randn(2, 12, 50)
    out = branch(mel, sparc_series=sparc)
    out.sum().backward()

    encoder_grads = [p.grad for p in branch.backbone.parameters()]
    assert all(g is None for g in encoder_grads), (
        "frozen backbone params must have no grad"
    )
    proj_grads = [p.grad for p in branch.input_proj.parameters() if p.grad is not None]
    assert len(proj_grads) > 0, "input_proj must accumulate grad when encoder is frozen"


# ===========================================================================
# 7. 5-step fine-tuning loop — loss decreases
# ===========================================================================


def test_finetune_loop_loss_decreases():
    import torch.nn as nn

    branch = _branch(freeze_encoder=False, n_encoder_layers=1, embed_dim=32)
    head = nn.Linear(32, 2)
    opt = torch.optim.SGD(
        list(branch.parameters()) + list(head.parameters()), lr=1e-3
    )
    losses = []
    torch.manual_seed(0)
    for _ in range(5):
        mel = torch.randn(4, 80, 30)
        labels = torch.randint(0, 2, (4,))
        opt.zero_grad()
        embed = branch(mel)
        loss = nn.functional.cross_entropy(head(embed), labels)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert all(not torch.isnan(torch.tensor(l)) for l in losses), f"NaN in losses: {losses}"
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


# ===========================================================================
# 8. FusionModel with transformer + TCN combined
# ===========================================================================


def test_fusion_with_transformer_and_tcn():
    from aces_b2ai.models.fusion_model import FusionModel
    from aces_b2ai.models.model_config import DiagnosisModelConfig, TransformerBranchConfig

    cfg = DiagnosisModelConfig(
        enable_scalar_branch=False,
        enable_series_branch=True,
        enable_transformer_branch=True,
        series_feature_keys=["ema_sparc"],
        series_n_channels={"ema_sparc": 12},
        embed_dim=32,
        transformer=TransformerBranchConfig(
            n_encoder_layers=1,
            embed_dim=32,
            melhubert_feat_dim=80,
            freeze_encoder=False,
        ),
        n_classes_multiclass=7,
    )
    model = FusionModel(cfg)
    model.train()
    batch = {
        "mel": torch.randn(3, 80, 50),
        "ema_sparc": torch.randn(3, 12, 50),
    }
    labels_bin = torch.tensor([0.0, 1.0, 0.0])
    labels_mc = torch.tensor([0, 2, -1])

    out = model(batch)
    loss = FusionModel.compute_loss(out, labels_bin, labels_mc)
    loss.backward()

    assert loss.isfinite(), f"loss is not finite: {loss.item()}"
    assert out["binary"].shape == (3, 1)
    assert out["multiclass"].shape == (3, 7)

    # At least one encoder param must have grad (encoder is unfrozen)
    enc_grads = [p.grad for p in model.transformer_branch.backbone.parameters() if p.requires_grad]
    assert any(g is not None for g in enc_grads), "No grad found in transformer encoder"


# ===========================================================================
# 9. Load real checkpoint — clean path (strict=True)
# ===========================================================================


def test_load_real_checkpoint_clean():
    """Load the 1.0 GB checkpoint; verify forward pass shape."""
    if not os.path.exists(CKPT_PATH):
        pytest.skip(f"Checkpoint not found at {CKPT_PATH}")

    from aces_b2ai.models.branches.transformer_branch import TransformerBranch
    from aces_b2ai.models.model_config import TransformerBranchConfig

    cfg = TransformerBranchConfig(
        melhubert_feat_dim=80,
        sparc_channels=0,
        freeze_encoder=True,
        embed_dim=768,
    )
    branch = TransformerBranch(cfg)
    branch.load_pretrained(CKPT_PATH)
    branch.eval()

    mel = torch.randn(2, 80, 200)
    with torch.no_grad():
        out = branch(mel)
    assert out.shape == (2, 768), f"expected (2,768), got {out.shape}"


# ===========================================================================
# 10. Load real checkpoint — SPARC-augmented path (strict=False, 199/201 keys)
# ===========================================================================


def test_load_real_checkpoint_sparc_augmented():
    """SPARC augmentation path: pre_extract_proj.* tensors are skipped on load.

    After loading, the backbone must have pre_extract_proj=None and forward
    must succeed with the combined 80+12=92 → 768 projection.
    """
    if not os.path.exists(CKPT_PATH):
        pytest.skip(f"Checkpoint not found at {CKPT_PATH}")

    from aces_b2ai.models.branches.transformer_branch import TransformerBranch
    from aces_b2ai.models.model_config import TransformerBranchConfig

    cfg = TransformerBranchConfig(
        melhubert_feat_dim=80,
        sparc_channels=12,
        freeze_encoder=True,
        embed_dim=768,
    )
    branch = TransformerBranch(cfg)
    branch.load_pretrained(CKPT_PATH)

    assert branch.backbone.pre_extract_proj is None, (
        "After load_pretrained with sparc_channels>0, pre_extract_proj must be None"
    )
    branch.eval()

    mel = torch.randn(2, 80, 200)
    sparc = torch.randn(2, 12, 200)
    with torch.no_grad():
        out = branch(mel, sparc_series=sparc)
    assert out.shape == (2, 768), f"expected (2,768), got {out.shape}"
