from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Allows running as: python synthesis/run_source_filter.py
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from synthesis.source_filter_pipeline import SourceFilterConfig, synthesize_subject  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Run UVFP source-filter synthesis from dataset features.")
    p.add_argument("--participant-id", required=True)
    p.add_argument("--session-id", required=True)
    p.add_argument("--task-name", required=True)
    p.add_argument("--out", type=Path, default=Path("outputs/source_filter.wav"))
    p.add_argument(
        "--combine-mode",
        choices=["mel_only", "phase_from_excitation_linear_mag"],
        default="phase_from_excitation_linear_mag",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = SourceFilterConfig()
    out = synthesize_subject(
        cfg,
        participant_id=args.participant_id,
        session_id=args.session_id,
        task_name=args.task_name,
        output_wav=args.out,
        combine_mode=args.combine_mode,
        seed=args.seed,
    )
    print(out)


if __name__ == "__main__":
    main()

