from pathlib import Path

import pandas as pd
import pytest

from aces_b2ai.stress_labels import load_stress_labels_csv, merge_stress_labels


def test_load_stress_labels_csv(tmp_path: Path):
    p = tmp_path / "stress.csv"
    p.write_text(
        "participant_id,session_id,stress_level,protocol_id,notes\n"
        "1,sessA,0.3,proto1,\n"
    )
    df = load_stress_labels_csv(p)
    assert df.shape[0] == 1
    assert df["stress_level"].iloc[0] == 0.3


def test_load_stress_labels_rejects_out_of_range(tmp_path: Path):
    p = tmp_path / "bad.csv"
    p.write_text("participant_id,session_id,stress_level,protocol_id\n1,s,2.0,x\n")
    with pytest.raises(ValueError):
        load_stress_labels_csv(p)


def test_merge_stress_labels():
    feat = pd.DataFrame({"participant_id": ["1"], "session_id": ["s"], "f": [1.0]})
    stress = pd.DataFrame(
        {"participant_id": ["1"], "session_id": ["s"], "stress_level": [0.5], "protocol_id": ["p"]}
    )
    m = merge_stress_labels(feat, stress)
    assert "stress_level" in m.columns
    assert m["stress_level"].iloc[0] == 0.5
