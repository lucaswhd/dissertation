#!/usr/bin/env python
"""Export the sub-instruction relabelled dataset (ablation group B).

Consumes the measured segmentation from ``relabel_hrc_subinstructions`` and rewrites
each source's Execute frames with their sub-goal prompt. Everything else -- state23,
action, shift-1 alignment, splits, non-Execute phases -- is unchanged from the
rerun_v1 continuous dataset, so a B-vs-A comparison isolates the labelling change.

Two fields that the base exporter ties together are split here:

  hrc.task_id     0/1 gripper-decision flag, read by the weighted sampler to
                  oversample decision frames (natural rate 5.5%).
  hrc.context_id  the original 4-way context, so the audit can still group by
                  wait / pickup / follow / radio.

Idle settle-out tails stay in the video but are dropped from valid_anchor: the
failure being fixed is a policy that stands still, and those frames teach exactly
that.

Usage (openpi_hrc/.venv):
  .venv/bin/python -m scripts.hrc.export_hrc_subinstructions \
      --manifest /root/autodl-tmp/lerobot_datasets/hrc_pi05_continuous_rerun_v1/manifest.json \
      --subinstructions /root/autodl-tmp/lerobot_datasets/hrc_subinstructions_rerun_v1.json \
      --output-root /root/autodl-tmp/lerobot_datasets/hrc_subinstr_rerun_v1
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from openpi.training.hrc_continuous_export import continuous_lerobot_features
from openpi.training.hrc_continuous_export import decode_rgb_frames
from openpi.training.hrc_continuous_export import load_continuous_rows

ACTION_HORIZON = 32
PHASE_TOKEN = "ExecuteBehaviorTask"


def apply_subinstructions(rows: dict, source_plan: dict, *, flip_radius: int) -> dict:
    """Rewrite Execute prompts, mask idle tails, and flag decision frames."""
    prompts = np.array(rows["prompt"], dtype=object).copy()
    valid = np.array(rows["valid_anchor"], dtype=bool).copy()
    event = np.zeros(len(prompts), dtype=bool)
    relabelled = 0

    for round_plan in source_plan.values():
        start = int(round_plan["execute_start"])
        stop = int(round_plan["execute_stop"])
        for segment in round_plan["segments"]:
            lo, hi = int(segment["start"]), int(segment["stop"])
            prompts[lo:hi] = f"Task: {segment['label']} Phase: {PHASE_TOKEN}."
            relabelled += hi - lo
        for flip in round_plan["flip_frames"]:
            event[max(start, flip - flip_radius) : min(stop, flip + flip_radius + 1)] = True
        # Idle tail: keep the frames (the video is continuous) but never anchor on them.
        valid[int(round_plan["idle_tail_from"]) : stop] = False

    if relabelled == 0:
        raise ValueError("no Execute frames were relabelled; check the plan/manifest pairing")

    out = dict(rows)
    out["prompt"] = prompts
    out["valid_anchor"] = valid
    out["event_flag"] = event
    return out


def write_episode(dataset, *, rows: dict, images, source_episode_id: int) -> None:
    """Write one episode, keeping task_id (event flag) and context_id distinct."""
    num_rows = len(rows["action"])
    image_iterator = iter(images)
    for index in range(num_rows):
        try:
            image = next(image_iterator)
        except StopIteration as exc:
            raise ValueError(f"expected {num_rows} images, got {index}") from exc
        dataset.add_frame(
            {
                "observation.rgb.zed_link_camera_0": np.asarray(image, dtype=np.uint8),
                "observation.state": np.asarray(rows["state"][index], dtype=np.float32),
                "action": np.asarray(rows["action"][index], dtype=np.float32),
                "hrc.valid_anchor": np.asarray([rows["valid_anchor"][index]], dtype=np.bool_),
                "hrc.task_id": np.asarray([int(rows["event_flag"][index])], dtype=np.int64),
                "hrc.context_id": np.asarray([int(rows["context_id"][index])], dtype=np.int64),
                "hrc.source_episode_id": np.asarray([source_episode_id], dtype=np.int64),
                "hrc.fsm_state_id": np.asarray([rows["fsm_state_id"][index]], dtype=np.int64),
                "hrc.raw_global_frame_index": np.asarray(
                    [rows["raw_global_frame_index"][index]], dtype=np.int64
                ),
                "hrc.raw_demo_id": np.asarray([rows["raw_demo_id"][index]], dtype=np.int64),
                "hrc.raw_demo_frame_index": np.asarray(
                    [rows["raw_demo_frame_index"][index]], dtype=np.int64
                ),
                "task": str(rows["prompt"][index]),
            }
        )
    dataset.save_episode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--subinstructions", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    plan = json.loads(args.subinstructions.read_text(encoding="utf-8"))
    flip_radius = int(plan["flip_radius"])
    splits = manifest["source_splits"]
    split_of = {int(s): name for name, ids in splits.items() for s in ids}

    from lerobot.datasets import LeRobotDataset

    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(args.output_root)
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True)

    out_manifest = dict(manifest)
    out_manifest["schema_version"] = "hrc_subinstr_rerun_v1"
    out_manifest["subinstruction_plan"] = str(args.subinstructions)
    (args.output_root / "manifest.json").write_text(json.dumps(out_manifest, indent=2))

    report = {"schema_version": out_manifest["schema_version"], "splits": {}}
    for split in ("train", "val", "test"):
        sources = [s for s in manifest["sources"] if split_of.get(int(s["source_episode_id"])) == split]
        if not sources:
            continue
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
        rows_total = anchors_total = events_total = 0
        for source in sorted(sources, key=lambda s: int(s["source_episode_id"])):
            source_id = int(source["source_episode_id"])
            rows = load_continuous_rows(source, action_horizon=ACTION_HORIZON)
            rows = apply_subinstructions(
                rows, plan["sources"][str(source_id)], flip_radius=flip_radius
            )
            images = decode_rgb_frames(Path(str(source["head_video_path"])))
            write_episode(dataset, rows=rows, images=images, source_episode_id=source_id)
            rows_total += len(rows["action"])
            anchors_total += int(rows["valid_anchor"].sum())
            events_total += int((rows["event_flag"] & rows["valid_anchor"]).sum())
            print(f"[export-sub] {split} source {source_id:>3}: {len(rows['action']):>6} rows, "
                  f"{int(rows['valid_anchor'].sum()):>6} anchors, "
                  f"{int((rows['event_flag'] & rows['valid_anchor']).sum()):>5} decision",
                  flush=True)
        report["splits"][split] = {
            "num_sources": len(sources),
            "num_rows": rows_total,
            "num_valid_anchors": anchors_total,
            "num_decision_anchors": events_total,
            "decision_fraction": events_total / max(anchors_total, 1),
            "dataset_root": str(split_root),
        }

    (args.output_root / "export_report.json").write_text(json.dumps(report, indent=2))
    print("\n[export-sub] done:")
    for split, stats in report["splits"].items():
        print(f"  {split:<6} {stats['num_sources']:>3} sources  {stats['num_rows']:>7,} rows  "
              f"{stats['num_valid_anchors']:>7,} anchors  "
              f"{stats['num_decision_anchors']:>6,} decision ({stats['decision_fraction']:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
