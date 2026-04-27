"""CLI: batch secondary features from Torchaudio Parquet files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from aces_b2ai.loaders import iter_parquet_rows, json_checksum
from aces_b2ai.pipeline import FeatureExtractionPipeline
from aces_b2ai.tasks import filter_rows_by_tasks
from aces_b2ai.config import PipelineConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mel-parquet", type=Path, required=True)
    ap.add_argument("--mfcc-parquet", type=Path, required=True)
    ap.add_argument("--pitch-parquet", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--features-dir", type=Path, help="Dataset features/ for JSON checksums")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--sustained-phonation-only", action="store_true")
    ap.add_argument("--enable-dynamical", action="store_true")
    args = ap.parse_args()

    cfg = PipelineConfig(enable_dynamical=args.enable_dynamical)
    pipe = FeatureExtractionPipeline(cfg=cfg)
    rows_out: list[dict] = []

    def row_iter():
        mel_pf = Path(args.mel_parquet)
        mfcc_pf = Path(args.mfcc_parquet)
        pitch_pf = Path(args.pitch_parquet)
        # naive join on keys by zipping same order — caller must ensure aligned exports
        m1 = { (r["participant_id"], r["session_id"], r["task_name"]): r for r in iter_parquet_rows(mel_pf, max_rows=args.max_rows) }
        m2 = { (r["participant_id"], r["session_id"], r["task_name"]): r for r in iter_parquet_rows(mfcc_pf, max_rows=args.max_rows) }
        m3 = { (r["participant_id"], r["session_id"], r["task_name"]): r for r in iter_parquet_rows(pitch_pf, max_rows=args.max_rows) }
        keys = sorted(set(m1) & set(m2) & set(m3))
        for k in keys:
            yield m1[k], m2[k], m3[k]

    prov = {}
    if args.features_dir:
        fd = Path(args.features_dir)
        for name in ["torchaudio_mel_spectrogram.json", "torchaudio_mfcc.json", "torchaudio_pitch.json"]:
            p = fd / name
            if p.exists():
                prov[name] = json_checksum(p)

    for mel_r, mfcc_r, pitch_r in row_iter():
        rec = {
            "participant_id": mel_r["participant_id"],
            "session_id": mel_r["session_id"],
            "task_name": mel_r["task_name"],
        }
        if args.sustained_phonation_only:
            if not filter_rows_by_tasks([rec], sustained_phonation_only=True):
                continue
        res = pipe.extract(
            participant_id=str(rec["participant_id"]),
            session_id=str(rec["session_id"]),
            task_name=str(rec["task_name"]),
            mel=mel_r.get("mel_spectrogram"),
            mfcc=mfcc_r.get("mfcc"),
            pitch=pitch_r.get("pitch"),
            provenance=prov or None,
        )
        rec.update(res.features)
        rec["warnings_json"] = json.dumps(res.warnings)
        rows_out.append(rec)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
