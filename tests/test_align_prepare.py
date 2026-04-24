import numpy as np

from aces_b2ai.align import prepare_clip_context, reference_length, voiced_mask_from_pitch
from aces_b2ai.config import PipelineConfig, VoicingConfig
from aces_b2ai.context import ClipContext


def test_reference_length_prefers_mel():
    T = 50
    mel = np.random.randn(60, T)
    pitch = np.ones(200) * 200.0
    ctx = ClipContext("1", "s1", "task", mel=mel, pitch_torch=pitch)
    assert reference_length(ctx) == T


def test_prepare_resamples_pitch_to_mel_width():
    T = 40
    mel = np.random.randn(60, T) * 0.1
    pitch = np.linspace(200, 180, 120)
    ctx = ClipContext("1", "s1", "t", mel=mel, pitch_torch=pitch)
    ctx = prepare_clip_context(ctx, PipelineConfig())
    assert ctx.pitch_aligned is not None
    assert ctx.pitch_aligned.shape[0] == T
    assert ctx.voiced_mask is not None
    assert ctx.voiced_mask.shape[0] == T


def test_voiced_mask_from_pitch():
    p = np.array([0.0, 100.0, 200.0, 600.0, np.nan])
    m = voiced_mask_from_pitch(p, VoicingConfig())
    assert not bool(m[0]) and bool(m[1]) and bool(m[2]) and not bool(m[3]) and not bool(m[4])


def test_periodicity_overrides_voicing():
    T = 30
    mel = np.zeros((60, T))
    pitch = np.ones(T) * 200.0
    per = np.zeros(T)
    per[5:25] = 0.9
    ctx = ClipContext("1", "s1", "t", mel=mel, pitch_torch=pitch, periodicity_sparc=per)
    ctx = prepare_clip_context(ctx, PipelineConfig())
    assert ctx.extras.get("voicing_mask_source") == "sparc_periodicity"
    assert int(ctx.voiced_mask.sum()) == 20
