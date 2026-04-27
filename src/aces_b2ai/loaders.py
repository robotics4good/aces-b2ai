from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def json_checksum(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def load_feature_json(dataset_features_dir: Path, name: str) -> dict[str, Any]:
    return json.loads((dataset_features_dir / name).read_text())


def parquet_row_to_numpy_1d(cell: Any) -> np.ndarray:
    return np.asarray(cell, dtype=np.float64).ravel()


def parquet_row_to_numpy_2d(cell: Any) -> np.ndarray:
    return np.asarray(cell, dtype=np.float64)


def iter_parquet_rows(
    path: Path,
    *,
    columns: list[str] | None = None,
    batch_size: int = 64,
    max_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    pf = pq.ParquetFile(path)
    n = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        d = batch.to_pydict()
        keys = d.get("participant_id", [])
        for i in range(len(keys)):
            yield {k: d[k][i] for k in d}
            n += 1
            if max_rows is not None and n >= max_rows:
                return


def load_static_row(
    static_tsv: Path,
    *,
    participant_id: str,
    session_id: str,
    task_name: str,
) -> dict[str, float] | None:
    df = pd.read_csv(static_tsv, sep="\t", low_memory=False)
    mask = (
        (df["participant_id"].astype(str) == str(participant_id))
        & (df["session_id"].astype(str) == str(session_id))
        & (df["task_name"].astype(str) == str(task_name))
    )
    sub = df.loc[mask]
    if sub.empty:
        return None
    row = sub.iloc[0]
    meta = {"participant_id", "session_id", "task_name", "transcription"}
    out: dict[str, float] = {}
    for k, v in row.items():
        if k in meta:
            continue
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out
