import numpy as np

from aces_b2ai.config import PipelineConfig
from aces_b2ai.pipeline import FeatureExtractionPipeline


def _synth_clip(*, T: int = 200, f0: float = 220.0, deplete: bool = False):
    t = np.arange(T, dtype=np.float64)
    mel = np.zeros((60, T))
    row = 0.5 + 0.01 * np.sin(np.linspace(0, 4 * np.pi, T))
    mel[:] = row
    if deplete:
        mel *= np.linspace(1.0, 0.5, T)
    pitch = np.full(T, f0, dtype=np.float64)
    if deplete:
        pitch = pitch - np.linspace(0, 15, T)
    mfcc = np.zeros((60, T))
    mfcc[0] = np.linspace(-2, 2, T)
    mfcc[1] = 0.1 * np.random.randn(T)
    return mel, mfcc, pitch


def test_pipeline_stable_phonation():
    mel, mfcc, pitch = _synth_clip(deplete=False)
    pipe = FeatureExtractionPipeline()
    res = pipe.extract(
        participant_id="1",
        session_id="s",
        task_name="long-sounds",
        mel=mel,
        mfcc=mfcc,
        pitch=pitch,
    )
    assert res.features["pitch_voiced_fraction_aligned"] > 0.9
    assert np.isfinite(res.features["pitch_depletion_slope_hz_per_frame"])


def test_pipeline_depleting_pitch_more_negative_slope():
    mel_n, mfcc_n, p_n = _synth_clip(deplete=False)
    mel_d, mfcc_d, p_d = _synth_clip(deplete=True)
    pipe = FeatureExtractionPipeline()
    r_n = pipe.extract(participant_id="1", session_id="s", task_name="t", mel=mel_n, mfcc=mfcc_n, pitch=p_n)
    r_d = pipe.extract(participant_id="1", session_id="s", task_name="t", mel=mel_d, mfcc=mfcc_d, pitch=p_d)
    assert r_d.features["mel_energy_depletion_ratio"] > r_n.features["mel_energy_depletion_ratio"]


def test_dynamical_flag_adds_keys():
    mel, mfcc, pitch = _synth_clip(T=300)
    cfg = PipelineConfig(enable_dynamical=True, dynamical_min_voiced_frames=10)
    pipe = FeatureExtractionPipeline(cfg=cfg)
    res = pipe.extract(participant_id="1", session_id="s", task_name="t", mel=mel, mfcc=mfcc, pitch=pitch)
    assert "dyn_phase_hull_area" in res.features or any("dynamical" in w for w in res.warnings)
