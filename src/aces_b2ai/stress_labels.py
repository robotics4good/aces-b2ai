from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# --- Ground truth for “stressed environment” modeling (protocol placeholder) ---
#
# The Bridge2AI pediatric phenotype JSONs in this repo do not define a stress
# condition column. Before running stress-specific evaluation, attach labels
# via a controlled CSV (IRB + DUA compliant). Expected columns:

STRESS_LABEL_CSV_COLUMNS = (
    "participant_id",
    "session_id",
    "stress_level",
    "protocol_id",
    "notes",
)

STRESS_LEVEL_RANGE = (0.0, 1.0)


@dataclass(frozen=True)
class StressLabelSpec:
    """Schema for optional external stress annotations."""

    participant_id: str
    session_id: str
    stress_level: float
    protocol_id: str
    notes: str = ""


def load_stress_labels_csv(path: Path | str) -> pd.DataFrame:
    """Load a team-maintained stress label table; validates required columns."""
    p = Path(path)
    df = pd.read_csv(p, dtype={"participant_id": str, "session_id": str, "protocol_id": str})
    missing = [c for c in ("participant_id", "session_id", "stress_level", "protocol_id") if c not in df.columns]
    if missing:
        raise ValueError(f"stress labels CSV missing columns: {missing}")
    df["stress_level"] = pd.to_numeric(df["stress_level"], errors="coerce")
    lo, hi = STRESS_LEVEL_RANGE
    bad = df["stress_level"].lt(lo) | df["stress_level"].gt(hi)
    if bad.any():
        raise ValueError("stress_level must lie in [0, 1] for all rows")
    return df


def merge_stress_labels(
    features: pd.DataFrame,
    stress: pd.DataFrame,
    *,
    on: tuple[str, str] = ("participant_id", "session_id"),
) -> pd.DataFrame:
    """Left-join secondary features with stress labels."""
    return features.merge(stress, on=list(on), how="left")
