from aces_b2ai.tasks import (
    aggregate_secondary_by_task,
    filter_rows_by_tasks,
    is_sustained_phonation_task,
    task_theme,
)


def test_sustained_tasks():
    assert is_sustained_phonation_task("long-sounds")
    assert not is_sustained_phonation_task("passage")


def test_filter_rows():
    rows = [
        {"task_name": "long-sounds", "x": 1},
        {"task_name": "passage", "x": 2},
    ]
    f = filter_rows_by_tasks(rows, sustained_phonation_only=True)
    assert len(f) == 1 and f[0]["task_name"] == "long-sounds"


def test_task_theme():
    assert task_theme("long-sounds") == "sustained_phonation_like"
    assert task_theme("passage") == "read_speech"


def test_aggregate_by_task():
    import pandas as pd

    df = pd.DataFrame(
        {
            "task_name": ["a", "a", "b"],
            "participant_id": [1, 2, 3],
            "f1": [1.0, 3.0, 10.0],
            "f2": [0.0, 2.0, 0.0],
        }
    )
    out = aggregate_secondary_by_task(df, feature_cols=["f1", "f2"])
    assert set(out["task_name"]) == {"a", "b"}
    a = out.loc[out["task_name"] == "a", "f1"].item()
    assert abs(a - 2.0) < 1e-6
