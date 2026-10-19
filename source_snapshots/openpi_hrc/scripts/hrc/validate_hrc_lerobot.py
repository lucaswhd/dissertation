"""Validate the complete HRC LeRobot export before normalization or training."""

import argparse
import json
from pathlib import Path

import numpy as np

from openpi.training import hrc_data
from openpi.training import hrc_export
from openpi.training import lerobot_compat
from openpi.training.hrc_validation import validate_exported_split

NUMERIC_KEYS = (
    "observation.state",
    "observation.privileged_state",
    "action",
    "hrc.valid_anchor",
    "hrc.task_id",
    "hrc.source_episode_id",
    "hrc.raw_frame_index",
    "episode_index",
)


def _validate_video_samples(dataset, count: int) -> list[int]:
    if count <= 0:
        return []
    indices = np.linspace(0, len(dataset) - 1, num=min(count, len(dataset)), dtype=np.int64)
    for index in indices:
        image = np.asarray(dataset[int(index)]["observation.rgb.zed_link_camera_0"])
        if image.shape != (3, 256, 352):
            raise ValueError(f"decoded RGB at row {index} has shape {image.shape}, expected [3,256,352]")
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError(f"decoded RGB at row {index} is not finite in [0,1]")
    return indices.tolist()


def _validate_raw_alignment(dataset, columns: dict[str, np.ndarray], segment: dict) -> dict[str, float | int]:
    task_id = hrc_data.TASK_NAMES.index(segment["task_name"])
    matches = np.flatnonzero(
        (columns["hrc.source_episode_id"] == int(segment["source_episode_id"]))
        & (columns["hrc.task_id"] == task_id)
        & (columns["hrc.raw_frame_index"] == int(segment["start_frame"]))
    )
    if len(matches) != 1:
        raise ValueError(f"could not uniquely locate exported first row for segment {segment}")
    row = int(matches[0])

    raw_rows = hrc_export.load_segment_rows(segment, include_phase=True, action_horizon=32)
    np.testing.assert_allclose(columns["action"][row], raw_rows["action"][0], rtol=0, atol=0)
    np.testing.assert_allclose(columns["observation.state"][row], raw_rows["state23"][0], rtol=0, atol=0)
    np.testing.assert_allclose(
        columns["observation.privileged_state"][row], raw_rows["state28"][0], rtol=0, atol=0
    )

    exported = np.asarray(dataset[row]["observation.rgb.zed_link_camera_0"])
    exported = np.rint(exported.transpose(1, 2, 0) * 255).astype(np.uint8)
    raw = hrc_export.MonotonicFrameCursor(hrc_export.decode_rgb_frames(segment["head_video_path"])).get(
        int(segment["start_frame"])
    )
    pixel_error = np.abs(exported.astype(np.float32) - raw.astype(np.float32))
    mae = float(pixel_error.mean())
    p99 = float(np.quantile(pixel_error, 0.99))
    if mae > 5.0 or p99 > 20.0:
        raise ValueError(f"exported RGB/raw video mismatch at row {row}: mae={mae:.3f}, p99={p99:.3f}")
    return {"row": row, "rgb_mae": mae, "rgb_p99": p99}


def main(dataset_root: Path, video_samples_per_split: int = 3) -> None:
    manifest = json.loads((dataset_root / "manifest.json").read_text())
    export_report = json.loads((dataset_root / "export_report.json").read_text())
    source_splits = {
        split: {int(value) for value in manifest["source_splits"][split]} for split in ("train", "val", "test")
    }
    if any(source_splits[left] & source_splits[right] for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("source leakage between train/val/test manifest splits")
    if [len(source_splits[name]) for name in ("train", "val", "test")] != [39, 5, 5]:
        raise ValueError("expected source split sizes 39/5/5")

    report = {"schema_version": manifest["schema_version"], "splits": {}}
    for split in ("train", "val", "test"):
        split_root = dataset_root / split
        repo_id = f"hrc/hrc_pi05_v0_{split}"
        dataset = lerobot_compat.LeRobotDataset(repo_id=repo_id, root=split_root)
        table = dataset.hf_dataset.with_format("numpy")
        columns = {key: np.asarray(table[key]) for key in NUMERIC_KEYS}
        split_report = validate_exported_split(
            columns,
            expected_sources=source_splits[split],
            action_horizon=int(manifest["action_horizon"]),
        )
        expected = export_report["splits"][split]
        for key in ("num_sources", "num_episodes", "num_rows", "num_valid_anchors"):
            if split_report[key] != expected[key]:
                raise ValueError(f"{split} {key} differs from export_report")
        split_report["video_sample_rows"] = _validate_video_samples(dataset, video_samples_per_split)
        alignment_segment = next(segment for segment in manifest["segments"] if segment["split"] == split)
        split_report["raw_alignment"] = _validate_raw_alignment(dataset, columns, alignment_segment)
        report["splits"][split] = split_report

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--video-samples-per-split", type=int, default=3)
    args = parser.parse_args()
    main(args.dataset_root, video_samples_per_split=args.video_samples_per_split)
