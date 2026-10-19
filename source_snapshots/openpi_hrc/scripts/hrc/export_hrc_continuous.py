#!/usr/bin/env python
"""Export the continuous CLEAN HRC dataset (one LeRobot episode per source).

Usage (openpi_hrc/.venv):
  .venv/bin/python scripts/hrc/export_hrc_continuous.py \
      --data-root /root/autodl-tmp/hrc_pipeline_traintest/outputs/hrc_batch_final \
      --qc-report /root/autodl-tmp/hrc_pipeline_traintest/outputs/hrc_batch_final/rescan_qc.json \
      --base-manifest /root/autodl-tmp/lerobot_datasets/hrc_pi05_v0/manifest.json \
      --output-root /root/autodl-tmp/lerobot_datasets/hrc_pi05_continuous_clean_v0 \
      --manual-qc /root/autodl-tmp/lerobot_datasets/hrc_pi05_continuous_clean_v0_manual_qc.json \
      --overwrite

``--overwrite`` may only target the new continuous output; it must never point at
the base ``hrc_pi05_v0`` (the exporter also refuses that by name).
"""

import argparse
import json
from pathlib import Path

from openpi.training.hrc_continuous_data import build_continuous_manifest
from openpi.training.hrc_continuous_export import export_continuous_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--qc-report", required=True, type=Path)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manual-qc", type=Path, default=None,
                        help="human video-review confirmation JSON for sources 25/36")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--split-spec",
        type=Path,
        default=None,
        help="frozen split spec JSON (expected_clean / manual_excluded / source_splits) for a "
        "batch other than the v0 pilot; also usable as --base-manifest",
    )
    args = parser.parse_args()

    overrides: dict = {}
    if args.split_spec is not None:
        spec = json.loads(args.split_spec.read_text(encoding="utf-8"))
        overrides = {
            "expected_clean": tuple(spec["expected_clean"]),
            "manual_excluded": tuple(spec.get("manual_excluded", ())),
            "schema_version": spec["dataset_version"],
        }
        print(
            f"[export] split spec {args.split_spec.name}: "
            f"clean={len(overrides['expected_clean'])} excluded={list(overrides['manual_excluded'])} "
            f"version={overrides['schema_version']}",
            flush=True,
        )

    print(f"[export] building continuous manifest from {args.data_root} ...", flush=True)
    manifest = build_continuous_manifest(
        args.data_root,
        args.qc_report,
        args.base_manifest,
        action_horizon=args.action_horizon,
        **overrides,
    )
    print(
        f"[export] sources={manifest['num_sources']} splits="
        f"{ {k: len(v) for k, v in manifest['source_splits'].items()} }",
        flush=True,
    )

    report = export_continuous_manifest(
        manifest, args.output_root, overwrite=args.overwrite, manual_qc_path=args.manual_qc
    )
    print("[export] done:", json.dumps(report["splits"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
