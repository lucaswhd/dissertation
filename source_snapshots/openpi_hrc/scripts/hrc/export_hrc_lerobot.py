#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from openpi.training import hrc_data
from openpi.training import hrc_export


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export final HRC data to LeRobot v3 for pi0.5 fine-tuning")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    parser.add_argument("--max-sources-per-split", type=int)
    parser.add_argument("--include-phase", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = hrc_data.build_manifest(args.data_root, seed=args.seed, action_horizon=args.action_horizon)
    report = hrc_export.export_manifest(
        manifest,
        args.output_root,
        include_phase=args.include_phase,
        splits=args.splits,
        max_sources_per_split=args.max_sources_per_split,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
