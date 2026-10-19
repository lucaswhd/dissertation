#!/usr/bin/env python
"""Build the Step-2 single-can overfit dataset: one grasp/release cycle, sub-instructions.

Question this dataset exists to answer: with the gripper flip made a large, explicitly
labelled fraction of a tiny dataset, can pi0.5 learn to emit it at all? The full-scale
audit measured gripper flip recall = 0.000, so this isolates that failure from data
volume, generalization and closed-loop drift.

Sub-instructions follow the MEASURED demo timeline, not an idealized one. In all 45
CLEAN episodes the robot carries the trash can in its right gripper and drops cans in
with the left, so the labels say exactly that; inventing a "walk back to the trash can"
step would attach language to actions that never happen (the task-0026 mistake).

Anchors are deliberately allowed to span a sub-instruction boundary: the chunk that
crosses the grasp frame is the one carrying the flip, and masking it would guarantee
the very failure being measured.

Usage (openpi_hrc/.venv):
  .venv/bin/python -m scripts.hrc.build_hrc_singlecan \
      --manifest /root/autodl-tmp/lerobot_datasets/hrc_pi05_continuous_rerun_v1/manifest.json \
      --source-id 38 --output-root /root/autodl-tmp/lerobot_datasets/hrc_singlecan_ep38
"""

import argparse
import itertools
import json
from pathlib import Path
import shutil

import numpy as np

from openpi.training.hrc_continuous_export import continuous_lerobot_features
from openpi.training.hrc_continuous_export import decode_rgb_frames
from openpi.training.hrc_continuous_export import load_continuous_rows
from openpi.training.hrc_continuous_export import write_continuous_episode

ACTION_HORIZON = 32
GRIPPER_DIMS = [14, 22]
PHASE = "ExecuteBehaviorTask"
PICK_INSTRUCTION = "Pick up the soda can."
DROP_INSTRUCTION = "Drop the can into the trash can you are holding."


def gripper_flip_frames(action: np.ndarray, start: int, stop: int) -> list[int]:
    """Absolute frame indices in [start, stop) where either gripper changes sign."""
    signs = np.sign(action[start:stop][:, GRIPPER_DIMS])
    changed = np.flatnonzero(np.any(signs[1:] != signs[:-1], axis=1)) + 1
    return [int(start + c) for c in changed]


def build_prompts(window_len: int, start: int, flips: list[int]) -> np.ndarray:
    """Sub-instruction per frame: 'pick up' before a close, 'drop' before an open."""
    prompts = np.empty(window_len, dtype=object)
    left_closed = False
    boundaries = [*flips, start + window_len]
    cursor = start
    for boundary in boundaries:
        text = DROP_INSTRUCTION if left_closed else PICK_INSTRUCTION
        prompts[cursor - start : boundary - start] = f"Task: {text} Phase: {PHASE}."
        left_closed = not left_closed
        cursor = boundary
    return prompts


def build_event_task_ids(window_len: int, start: int, flips: list[int], radius: int) -> np.ndarray:
    """task_id 1 inside +/-radius of a gripper flip, else 0 (drives event upsampling)."""
    task_ids = np.zeros(window_len, dtype=np.int64)
    for flip in flips:
        lo = max(0, flip - start - radius)
        hi = min(window_len, flip - start + radius + 1)
        task_ids[lo:hi] = 1
    return task_ids


def slice_rows(
    rows: dict[str, np.ndarray], start: int, stop: int, *, flip_radius: int
) -> dict[str, np.ndarray]:
    window_len = stop - start
    flips = gripper_flip_frames(rows["action"], start, stop)
    if not flips:
        raise ValueError(f"window [{start}, {stop}) contains no gripper flip to learn")

    valid = np.zeros(window_len, dtype=bool)
    valid[: max(0, window_len - ACTION_HORIZON)] = True

    sliced = {key: value[start:stop] for key, value in rows.items()}
    sliced["prompt"] = build_prompts(window_len, start, flips)
    sliced["valid_anchor"] = valid
    sliced["context_id"] = build_event_task_ids(window_len, start, flips, flip_radius)
    return sliced, flips


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-id", type=int, required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--start", type=int, default=None, help="absolute start frame")
    parser.add_argument("--stop", type=int, default=None, help="absolute stop frame")
    parser.add_argument("--flip-radius", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source = next(s for s in manifest["sources"] if int(s["source_episode_id"]) == args.source_id)
    rows = load_continuous_rows(source, action_horizon=ACTION_HORIZON)

    start, stop = args.start, args.stop
    if start is None or stop is None:
        raise ValueError("--start and --stop are required; pick them from the measured timeline")

    sliced, flips = slice_rows(rows, start, stop, flip_radius=args.flip_radius)
    # decode_rgb_frames yields lazily; materialize only the window (~121 MB) rather
    # than the whole episode, which the 90 GiB container cannot hold for long sources.
    images = list(
        itertools.islice(decode_rgb_frames(Path(str(source["head_video_path"]))), start, stop)
    )
    if len(images) != stop - start:
        raise ValueError(f"decoded {len(images)} frames for a {stop - start}-frame window")

    print(f"[singlecan] source {args.source_id} window [{start}, {stop}) = {stop - start} frames")
    print(f"[singlecan] gripper flips at {flips}")
    print(f"[singlecan] valid anchors {int(sliced['valid_anchor'].sum())}, "
          f"event anchors {int((sliced['context_id'] == 1)[sliced['valid_anchor']].sum())}")
    for text in sorted(set(sliced["prompt"].tolist())):
        count = int((sliced["prompt"] == text).sum())
        print(f"[singlecan]   {count:>4} frames :: {text}")

    from lerobot.datasets import LeRobotDataset

    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(args.output_root)
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True)

    report = {
        "source_episode_id": args.source_id,
        "window": [start, stop],
        "gripper_flips": flips,
        "flip_radius": args.flip_radius,
        "num_frames": stop - start,
        "num_valid_anchors": int(sliced["valid_anchor"].sum()),
        "splits": {},
    }
    # train and val are the same window on purpose: this is an overfit probe, not a
    # generalization measurement.
    for split in ("train", "val"):
        split_root = args.output_root / split
        dataset = LeRobotDataset.create(
            repo_id=f"hrc/{args.output_root.name}_{split}",
            root=split_root,
            robot_type="R1Pro",
            fps=30,
            features=continuous_lerobot_features(image_height=256, image_width=352),
            use_videos=True,
            video_backend="pyav",
            image_writer_threads=4,
        )
        write_continuous_episode(
            dataset, rows=sliced, images=images, source_episode_id=args.source_id
        )
        report["splits"][split] = {"dataset_root": str(split_root), "num_rows": stop - start}

    (args.output_root / "singlecan_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"[singlecan] wrote {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
