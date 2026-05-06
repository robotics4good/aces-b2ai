from __future__ import annotations

import argparse
from pathlib import Path

from synthesis.source_filter_pipeline import SourceFilterConfig, load_uvfp_participant_ids


def main() -> None:
    p = argparse.ArgumentParser(description="Batch synthesis runner (UVFP filter).")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/batch"))
    p.add_argument(
        "--participants-tsv",
        type=Path,
        default=None,
        help="Override cfg.participants_tsv (relative to dataset_root unless absolute).",
    )
    args = p.parse_args()

    cfg = SourceFilterConfig()
    if args.participants_tsv is not None:
        cfg = SourceFilterConfig(participants_tsv=args.participants_tsv)

    # Requested integration point: get UVFP subject list before synthesis loop.
    uvfp_ids = load_uvfp_participant_ids(cfg)
    print(f"UVFP participant_id count: {len(uvfp_ids)}")

    # Note: actual batch synthesis loop is project-specific (session/task selection).
    # This script currently only resolves UVFP participant IDs and validates label plumbing.


if __name__ == "__main__":
    main()

