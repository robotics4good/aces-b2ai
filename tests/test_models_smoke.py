"""Smoke tests for the feature-routing + model architecture.

All tests use purely synthetic data (no dataset files required).
PyTorch is required; the suite is skipped automatically if it is not installed.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # skip entire module if torch absent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


def _mel(T: int = 200, bins: int = 60) -> np.ndarray:
    return RNG.random((bins, T)).astype(np.float32)


def _ema(T: int = 200, ch: int = 12) -> np.ndarray:
    """EMA stored as (T, ch) on ClipContext — bundler should transpose."""
    return RNG.random((T, ch)).astype(np.float32)


def _bundle(
    *,
    scalars: bool = True,
    mel: bool = True,
    ema: bool = True,
    label_binary: int = 0,
    label_multiclass: int = 0,
):
    from aces_b2ai.context import ClipContext
    from aces_b2ai.models.feature_bundle import FeatureBundle, FeatureBundler
    from aces_b2ai.core.base import ExtractionResult

    ctx = ClipContext(
        participant_id="test",
        session_id="s1",
        task_name="long-sounds",
        mel=_mel() if mel else None,
        ema_sparc=_ema() if ema else None,
    )
    results = []
    if scalars:
        results.append(
            ExtractionResult(
                features={"jerk_tbx": 1.2, "jerk_tby": 0.8, "hf_noise": 0.05}
            )
        )
    bundler = FeatureBundler()
    return bundler.build(ctx, results, label_binary=label_binary, label_multiclass=label_multiclass)


# ===========================================================================
# 1. FeatureBundler routing
# ===========================================================================


def test_bundler_routes_scalars():
    b = _bundle(scalars=True, mel=False, ema=False)
    assert "jerk_tbx" in b.scalars
    assert "jerk_tby" in b.scalars
    assert b.spectrograms == {}
    assert b.series == {}


def test_bundler_routes_mel_to_spectrograms():
    b = _bundle(mel=True, ema=False)
    assert "mel" in b.spectrograms
    mel = b.spectrograms["mel"]
    assert mel.ndim == 2
    assert mel.shape[0] == 60  # freq_bins


def test_bundler_routes_ema_to_series_and_transposes():
    b = _bundle(mel=False, ema=True)
    assert "ema_sparc" in b.series
    ema = b.series["ema_sparc"]
    assert ema.ndim == 2
    # After transpose: (12, T) — channels first
    assert ema.shape[0] == 12


def test_bundler_missing_tensor_produces_empty_dicts():
    b = _bundle(mel=False, ema=False)
    assert b.spectrograms == {}
    assert b.series == {}


def test_bundler_extra_descriptor_reads_from_extras():
    """An extractor that writes a tensor to ctx.extras is routed correctly."""
    from aces_b2ai.context import ClipContext
    from aces_b2ai.models.feature_bundle import FeatureBundler
    from aces_b2ai.models.feature_descriptor import FeatureDescriptor, FeatureType

    custom_desc = FeatureDescriptor(
        name="my_custom_spec",
        feature_type=FeatureType.SPECTROGRAM,
        shape_hint=(128, None),
        frame_rate_hz=50.0,
    )
    ctx = ClipContext(participant_id="x", session_id="s", task_name="t")
    ctx.extras["my_custom_spec"] = np.ones((128, 50), dtype=np.float32)

    bundler = FeatureBundler(extra_descriptors=[custom_desc])
    b = bundler.build(ctx)
    assert "my_custom_spec" in b.spectrograms
    assert b.spectrograms["my_custom_spec"].shape == (128, 50)


# ===========================================================================
# 2. DiagnosisModelConfig validation
# ===========================================================================


def test_config_validate_raises_on_missing_freq_bins():
    from aces_b2ai.models.model_config import DiagnosisModelConfig

    cfg = DiagnosisModelConfig(
        enable_spectrogram_branch=True,
        spectrogram_feature_keys=["mel"],
        spectrogram_freq_bins={},   # missing "mel"
    )
    with pytest.raises(ValueError, match="spectrogram_freq_bins"):
        cfg.validate()


def test_config_validate_raises_when_no_branch_enabled():
    from aces_b2ai.models.model_config import DiagnosisModelConfig

    cfg = DiagnosisModelConfig(
        enable_scalar_branch=False,
        enable_spectrogram_branch=False,
        enable_series_branch=False,
    )
    with pytest.raises(ValueError, match="At least one branch"):
        cfg.validate()


# ===========================================================================
# 3. SpectrogramBranch forward pass
# ===========================================================================


def test_spectrogram_branch_forward_shape():
    from aces_b2ai.models.branches.spectrogram_branch import SpectrogramBranch

    branch = SpectrogramBranch(freq_bins=60, embed_dim=64, patch_frames=64)
    branch.eval()
    x = torch.randn(4, 60, 200)   # batch=4, 60 bins, 200 frames
    with torch.no_grad():
        out = branch(x)
    assert out.shape == (4, 64)


def test_spectrogram_branch_short_clip_padded():
    """Clips shorter than patch_frames should not crash."""
    from aces_b2ai.models.branches.spectrogram_branch import SpectrogramBranch

    branch = SpectrogramBranch(freq_bins=60, embed_dim=32, patch_frames=128)
    branch.eval()
    x = torch.randn(2, 60, 30)    # only 30 frames
    with torch.no_grad():
        out = branch(x)
    assert out.shape == (2, 32)


def test_spectrogram_branch_different_freq_bins():
    """Branch adapts to any freq_bins value (e.g. 201 for linear spectrogram)."""
    from aces_b2ai.models.branches.spectrogram_branch import SpectrogramBranch

    branch = SpectrogramBranch(freq_bins=201, embed_dim=64, patch_frames=64)
    branch.eval()
    x = torch.randn(2, 201, 100)
    with torch.no_grad():
        out = branch(x)
    assert out.shape == (2, 64)


# ===========================================================================
# 4. TimeSeriesBranch forward pass
# ===========================================================================


def test_timeseries_branch_forward_shape():
    from aces_b2ai.models.branches.timeseries_branch import TimeSeriesBranch

    branch = TimeSeriesBranch(n_channels=12, embed_dim=64)
    branch.eval()
    x = torch.randn(4, 12, 300)   # batch=4, 12-ch EMA, 300 frames
    with torch.no_grad():
        out = branch(x)
    assert out.shape == (4, 64)


def test_timeseries_branch_variable_length():
    """Branch should work for different T lengths (global avg pool)."""
    from aces_b2ai.models.branches.timeseries_branch import TimeSeriesBranch

    branch = TimeSeriesBranch(n_channels=40, embed_dim=32)
    branch.eval()
    for T in [50, 200, 500]:
        x = torch.randn(1, 40, T)
        with torch.no_grad():
            out = branch(x)
        assert out.shape == (1, 32)


# ===========================================================================
# 5. FusionModel end-to-end
# ===========================================================================


def test_fusion_model_spectrogram_only():
    from aces_b2ai.models.fusion_model import FusionModel
    from aces_b2ai.models.model_config import DiagnosisModelConfig

    cfg = DiagnosisModelConfig(
        enable_scalar_branch=False,
        enable_spectrogram_branch=True,
        enable_series_branch=False,
        spectrogram_feature_keys=["mel"],
        spectrogram_freq_bins={"mel": 60},
        embed_dim=64,
        patch_frames=64,
        n_classes_multiclass=7,
    )
    model = FusionModel(cfg)
    model.eval()
    batch = {"mel": torch.randn(4, 60, 200)}
    with torch.no_grad():
        out = model(batch)
    assert out["binary"].shape == (4, 1)
    assert out["multiclass"].shape == (4, 7)


def test_fusion_model_series_only():
    from aces_b2ai.models.fusion_model import FusionModel
    from aces_b2ai.models.model_config import DiagnosisModelConfig

    cfg = DiagnosisModelConfig(
        enable_scalar_branch=False,
        enable_spectrogram_branch=False,
        enable_series_branch=True,
        series_feature_keys=["ema_sparc"],
        series_n_channels={"ema_sparc": 12},
        embed_dim=64,
        n_classes_multiclass=7,
    )
    model = FusionModel(cfg)
    model.eval()
    batch = {"ema_sparc": torch.randn(4, 12, 300)}
    with torch.no_grad():
        out = model(batch)
    assert out["binary"].shape == (4, 1)
    assert out["multiclass"].shape == (4, 7)


def test_fusion_model_multi_branch():
    """Both mel + EMA branches active; fusion output shape correct."""
    from aces_b2ai.models.fusion_model import FusionModel
    from aces_b2ai.models.model_config import DiagnosisModelConfig

    cfg = DiagnosisModelConfig(
        enable_scalar_branch=False,
        enable_spectrogram_branch=True,
        enable_series_branch=True,
        spectrogram_feature_keys=["mel", "mfcc"],
        spectrogram_freq_bins={"mel": 60, "mfcc": 60},
        series_feature_keys=["ema_sparc"],
        series_n_channels={"ema_sparc": 12},
        embed_dim=64,
        patch_frames=64,
        n_classes_multiclass=7,
    )
    model = FusionModel(cfg)
    model.eval()
    batch = {
        "mel": torch.randn(3, 60, 200),
        "mfcc": torch.randn(3, 60, 200),
        "ema_sparc": torch.randn(3, 12, 200),
    }
    with torch.no_grad():
        out = model(batch)
    assert out["binary"].shape == (3, 1)
    assert out["multiclass"].shape == (3, 7)


def test_fusion_model_loss_finite():
    """compute_loss should return a finite scalar."""
    from aces_b2ai.models.fusion_model import FusionModel
    from aces_b2ai.models.model_config import DiagnosisModelConfig

    cfg = DiagnosisModelConfig(
        enable_scalar_branch=False,
        enable_spectrogram_branch=True,
        spectrogram_feature_keys=["mel"],
        spectrogram_freq_bins={"mel": 60},
        embed_dim=32,
        patch_frames=64,
        n_classes_multiclass=5,
    )
    model = FusionModel(cfg)
    batch = {"mel": torch.randn(4, 60, 100)}
    labels_bin = torch.tensor([0.0, 1.0, 0.0, 1.0])
    labels_mc = torch.tensor([0, 2, -1, 4])   # -1 = unknown → ignored
    out = model(batch)
    loss = FusionModel.compute_loss(out, labels_bin, labels_mc)
    assert loss.isfinite()


# ===========================================================================
# 6. DiagnosisDataset + collate_fn
# ===========================================================================


def test_diagnosis_dataset_len_and_keys():
    from aces_b2ai.models.diagnosis_dataset import DiagnosisDataset, collate_fn
    from aces_b2ai.models.feature_bundle import FeatureBundle
    from aces_b2ai.models.model_config import DiagnosisModelConfig

    bundles = [
        FeatureBundle(
            participant_id=str(i),
            task_name="long-sounds",
            spectrograms={"mel": _mel(T=200 + i * 10)},
            series={"ema_sparc": _ema(T=200 + i * 10).T},  # already (12, T)
            label_binary=i % 2,
            label_multiclass=i % 3,
        )
        for i in range(6)
    ]
    cfg = DiagnosisModelConfig(
        enable_spectrogram_branch=True,
        enable_series_branch=True,
        spectrogram_feature_keys=["mel"],
        spectrogram_freq_bins={"mel": 60},
        series_feature_keys=["ema_sparc"],
        series_n_channels={"ema_sparc": 12},
    )
    ds = DiagnosisDataset(bundles, cfg)
    assert len(ds) == 6

    item = ds[0]
    assert "mel" in item
    assert "ema_sparc" in item
    assert item["mel"].shape[0] == 60   # freq_bins
    assert item["ema_sparc"].shape[0] == 12  # n_channels

    loader = torch.utils.data.DataLoader(ds, batch_size=3, collate_fn=collate_fn)
    batch = next(iter(loader))
    assert batch["mel"].shape[0] == 3
    assert batch["mel"].ndim == 3          # (B, F, T_max)
    assert batch["ema_sparc"].ndim == 3    # (B, C, T_max)
