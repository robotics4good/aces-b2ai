from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

# From BRIDGE2AI_COMBINED.MD §3 — sustained / prolonged phonation style activities.
SUSTAINED_PHONATION_LIKE_TASKS: frozenset[str] = frozenset(
    {
        "long-sounds",
        "silly-sounds",
        "noisy-sounds",
    }
)

# Informal activity buckets for stratified summaries (not official dataset columns).
TASK_THEME_READ_SPEECH: frozenset[str] = frozenset({"passage", "sentence"})
TASK_THEME_CONVERSATION: frozenset[str] = frozenset(
    {
        "ready-for-school",
        "favorite-show-movie-game",
        "favorite-food",
        "outside-of-school",
    }
)


def normalize_task_name(task_name: str) -> str:
    return (task_name or "").strip().lower()


def is_sustained_phonation_task(task_name: str) -> bool:
    return normalize_task_name(task_name) in SUSTAINED_PHONATION_LIKE_TASKS


def task_theme(task_name: str) -> str:
    t = normalize_task_name(task_name)
    if t in SUSTAINED_PHONATION_LIKE_TASKS:
        return "sustained_phonation_like"
    if t in TASK_THEME_READ_SPEECH:
        return "read_speech"
    if t in TASK_THEME_CONVERSATION:
        return "conversation"
    return "other"


def filter_rows_by_tasks(
    rows: Iterable[dict],
    *,
    allowed_tasks: frozenset[str] | None = None,
    sustained_phonation_only: bool = False,
) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        tn = normalize_task_name(str(r.get("task_name", "")))
        if sustained_phonation_only and tn not in SUSTAINED_PHONATION_LIKE_TASKS:
            continue
        if allowed_tasks is not None and tn not in allowed_tasks:
            continue
        out.append(r)
    return out


def aggregate_secondary_by_task(
    df: pd.DataFrame,
    *,
    feature_cols: list[str] | None = None,
    task_col: str = "task_name",
) -> pd.DataFrame:
    """Mean secondary features per task_name (activity-stratified summaries)."""
    if df.empty:
        return df
    if feature_cols is None:
        meta = {task_col, "participant_id", "session_id"}
        feature_cols = [c for c in df.columns if c not in meta and np.issubdtype(df[c].dtype, np.number)]
    if not feature_cols:
        return pd.DataFrame()
    return df.groupby(task_col, dropna=False)[feature_cols].mean().reset_index()
