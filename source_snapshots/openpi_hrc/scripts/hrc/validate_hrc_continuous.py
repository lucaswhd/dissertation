#!/usr/bin/env python
"""Validate the exported continuous CLEAN dataset against its source of truth.

Runs the column-level contract checks (:mod:`hrc_continuous_validation`) on each
split's parquet, then the strong cross-check: rebuild every source's rows from the
raw HDF5 with :func:`load_continuous_rows` and assert the exported state23 /
action(shift-1) / context / valid-anchor / fsm / prompt match exactly. Finally an
aggregate RGB frame-count check. Prints ``VALIDATION=PASS`` only if all pass.

Usage (openpi_hrc/.venv):
  .venv/bin/python -m scripts.hrc.validate_hrc_continuous \
      --dataset-root /root/autodl-tmp/lerobot_datasets/hrc_pi05_continuous_clean_v0
"""

import argparse
import glob
import itertools
import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.training.hrc_continuous_data import CONTEXT_NAMES
from openpi.training.hrc_continuous_export import load_continuous_rows
from openpi.training.hrc_continuous_validation import check_prompt_phase_consistency
from openpi.training.hrc_continuous_validation import validate_continuous_columns

# Plan §4 (15 CLEAN, source 36 excluded).
EXPECT = {
    "train": {"sources": 13, "rows": 130000, "anchors": 128336,
              "context": {"wait": 169, "pickup": 69987, "follow": 4565, "radio": 53615}},
    "val": {"sources": 2, "rows": 18507, "anchors": 18251,
            "context": {"wait": 26, "pickup": 10298, "follow": 1057, "radio": 6870}},
}
_META_KEYS = (
    "hrc.valid_anchor", "hrc.task_id", "hrc.context_id", "hrc.source_episode_id",
    "hrc.fsm_state_id", "hrc.raw_global_frame_index", "hrc.raw_demo_id",
    "hrc.raw_demo_frame_index", "episode_index",
)


def read_split_columns(split_root: Path):
    files = sorted(glob.glob(str(split_root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {split_root}/data")
    table = pa.concat_tables([pq.read_table(f) for f in files])
    data = table.to_pydict()
    order = np.argsort(np.asarray(data["index"], dtype=np.int64))  # global row order

    columns = {
        "observation.state": np.asarray(data["observation.state"], dtype=np.float32)[order],
        "action": np.asarray(data["action"], dtype=np.float32)[order],
    }
    for key in _META_KEYS:
        dtype = bool if key == "hrc.valid_anchor" else np.int64
        columns[key] = np.asarray(data[key], dtype=dtype)[order]

    tasks = pq.read_table(split_root / "meta" / "tasks.parquet").to_pydict()
    index_to_task = {int(i): str(t) for i, t in zip(tasks["task_index"], tasks["task"], strict=True)}
    prompts = np.array([index_to_task[int(ti)] for ti in np.asarray(data["task_index"])[order]], dtype=object)
    return columns, prompts


def cross_check_sources(columns, prompts, manifest_sources, split_sources, action_horizon, failures):
    by_id = {int(s["source_episode_id"]): s for s in manifest_sources}
    for source_id in split_sources:
        rows = np.flatnonzero(columns["hrc.source_episode_id"] == source_id)
        # rows for one source are one contiguous episode; order by raw global frame.
        rows = rows[np.argsort(columns["hrc.raw_global_frame_index"][rows])]
        rebuilt = load_continuous_rows(by_id[source_id], action_horizon=action_horizon)
        n = len(rebuilt["action"])
        if len(rows) != n:
            failures.append(f"source {source_id}: exported {len(rows)} rows != source {n}")
            continue
        checks = {
            "state23": np.allclose(columns["observation.state"][rows], rebuilt["state"], atol=1e-4, rtol=1e-4),
            "action_shift1": np.allclose(columns["action"][rows], rebuilt["action"], atol=1e-4, rtol=1e-4),
            "context_id": np.array_equal(columns["hrc.context_id"][rows], rebuilt["context_id"]),
            "valid_anchor": np.array_equal(columns["hrc.valid_anchor"][rows], rebuilt["valid_anchor"]),
            "fsm_state_id": np.array_equal(columns["hrc.fsm_state_id"][rows], rebuilt["fsm_state_id"]),
            "raw_global_index": np.array_equal(
                columns["hrc.raw_global_frame_index"][rows], rebuilt["raw_global_frame_index"]
            ),
            "prompt": np.array_equal(prompts[rows], rebuilt["prompt"]),
        }
        for name, ok in checks.items():
            if not ok:
                failures.append(f"source {source_id}: raw cross-check {name} mismatch")


def count_video_frames(split_root: Path) -> int:
    total = 0
    for video in sorted(glob.glob(str(split_root / "videos" / "**" / "*.mp4"), recursive=True)):
        with av.open(video) as container:
            stream = container.streams.video[0]
            total += stream.frames or sum(1 for _ in container.decode(stream))
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--skip-rgb", action="store_true", help="skip the aggregate video frame-count check")
    parser.add_argument(
        "--expect",
        type=Path,
        default=None,
        help="frozen per-split expectations JSON (sources/rows/anchors/context) for a dataset "
        "other than the v0 pilot; splits absent from it must be empty",
    )
    args = parser.parse_args()

    expect_all = EXPECT if args.expect is None else json.loads(args.expect.read_text(encoding="utf-8"))

    manifest = json.loads((args.dataset_root / "manifest.json").read_text(encoding="utf-8"))
    action_horizon = int(manifest["action_horizon"])
    source_splits = manifest["source_splits"]
    manifest_sources = manifest["sources"]

    failures: list[str] = []
    report = {"action_horizon": action_horizon, "splits": {}}

    # cross-split: every declared split is pairwise disjoint; any split without a
    # frozen expectation must be empty (the v0 pilot declares no test split).
    declared = [split for split in ("train", "val", "test") if split in expect_all]
    for left, right in itertools.combinations(declared, 2):
        overlap = sorted(set(source_splits[left]) & set(source_splits[right]))
        if overlap:
            failures.append(f"{left} and {right} sources overlap: {overlap}")
    for split in ("train", "val", "test"):
        if split not in expect_all and source_splits.get(split):
            failures.append(f"{split} split must be empty (no frozen expectation for it)")

    for split in declared:
        split_sources = sorted(int(s) for s in source_splits[split])
        split_root = args.dataset_root / split
        columns, prompts = read_split_columns(split_root)

        try:
            stats = validate_continuous_columns(
                columns, expected_sources=set(split_sources), action_horizon=action_horizon
            )
            check_prompt_phase_consistency(prompts, columns["hrc.fsm_state_id"])
        except ValueError as exc:
            failures.append(f"[{split}] column validation: {exc}")
            report["splits"][split] = {"error": str(exc)}
            continue

        cross_check_sources(columns, prompts, manifest_sources, split_sources, action_horizon, failures)

        expect = expect_all[split]
        if stats["num_sources"] != expect["sources"]:
            failures.append(f"[{split}] sources {stats['num_sources']} != {expect['sources']}")
        if stats["num_rows"] != expect["rows"]:
            failures.append(f"[{split}] rows {stats['num_rows']} != {expect['rows']}")
        if stats["num_valid_anchors"] != expect["anchors"]:
            failures.append(f"[{split}] anchors {stats['num_valid_anchors']} != {expect['anchors']}")
        for name in CONTEXT_NAMES:
            got = stats["valid_anchors_by_context"][name]
            if got != expect["context"][name]:
                failures.append(f"[{split}] context {name} {got} != {expect['context'][name]}")

        rgb_frames = None
        if not args.skip_rgb:
            rgb_frames = count_video_frames(split_root)
            if rgb_frames != stats["num_rows"]:
                failures.append(f"[{split}] video frames {rgb_frames} != rows {stats['num_rows']}")

        report["splits"][split] = {**stats, "rgb_frames": rgb_frames}
        print(f"[{split}] rows={stats['num_rows']} anchors={stats['num_valid_anchors']} "
              f"by_context={stats['valid_anchors_by_context']} rgb_frames={rgb_frames}", flush=True)

    report["failures"] = failures
    report["result"] = "PASS" if not failures else "FAIL"
    (args.dataset_root / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if failures:
        print("\n".join(f"  ✗ {f}" for f in failures), flush=True)
        print("VALIDATION=FAIL", flush=True)
        return 1
    print("VALIDATION=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
