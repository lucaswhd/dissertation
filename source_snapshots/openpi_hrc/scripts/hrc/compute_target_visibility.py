#!/usr/bin/env python
"""Per-frame geometric visibility of each task object in the head camera.

Measured on the 45 CLEAN episodes: during the pickup Execute phase the target is
outside the head camera's field of view about 25% of the time, and in ~70% of
episodes it is not visible at all on the first frame -- the robot has to travel for
5-9 seconds before the can or trash can first enters the image. A training anchor
taken from such a frame asks the policy to reach for something it cannot see, so the
correct action is not recoverable from the observation and the sample is unlearnable
rather than merely hard.

This writes the per-frame mask so training can be restricted to well-posed frames,
and so the "target visible" fraction can be reported as a dataset property.

The camera convention is not assumed: it is calibrated against the recorded
``human_visible`` flag, which the collector produced independently. Only conventions
reproducing that flag are used, and the achieved agreement is written to the report
so a silent frame-convention change cannot corrupt the mask unnoticed.

Usage (openpi_hrc/.venv):
  .venv/bin/python -m scripts.hrc.compute_target_visibility \
      --batch-root /root/autodl-tmp/hrc_pipeline_traintest/outputs/hrc_batch_rerun_v1 \
      --output /root/autodl-tmp/lerobot_datasets/hrc_target_visibility_rerun_v1.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

# Head camera intrinsics recorded in every episode_meta.json.
HFOV_DEG = 105.8538533653792
VFOV_DEG = 87.81459492868048
# Height above the human's recorded foot position used when calibrating against the
# recorded visibility flag; the collector's flag tracks the body, not the floor point.
HUMAN_BODY_OFFSET_M = 1.0


def quat_to_rotation(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix from an (x, y, z, w) quaternion."""
    x, y, z, w = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def in_view(cam_pos: np.ndarray, cam_quat: np.ndarray, target: np.ndarray) -> bool:
    """True when the target lies inside the camera frustum (camera looks along -Z)."""
    local = quat_to_rotation(cam_quat).T @ (target - cam_pos)
    forward = -local[2]
    if forward <= 0:
        return False
    horizontal = np.degrees(np.arctan2(abs(local[0]), forward))
    vertical = np.degrees(np.arctan2(abs(local[1]), forward))
    return bool(horizontal < HFOV_DEG / 2 and vertical < VFOV_DEG / 2)


def calibration_agreement(episode: dict, stride: int = 37) -> float:
    """Agreement between the geometric test and the collector's own visibility flag."""
    cam_pos = episode["head_camera_world_pos"]
    cam_quat = episode["head_camera_world_quat"]
    human = episode["human_pos"].copy()
    human[:, 2] += HUMAN_BODY_OFFSET_M
    flag = episode["human_visible_flag"].astype(bool)
    checks = [
        in_view(cam_pos[t], cam_quat[t], human[t]) == flag[t]
        for t in range(0, len(cam_pos), stride)
    ]
    return float(np.mean(checks)) if checks else float("nan")


def episode_visibility(episode: dict, object_names: list[str]) -> dict:
    cam_pos = episode["head_camera_world_pos"]
    cam_quat = episode["head_camera_world_quat"]
    objects = episode["object_pos"]
    masks = {}
    for index, name in enumerate(object_names):
        masks[name] = np.array(
            [in_view(cam_pos[t], cam_quat[t], objects[t, index]) for t in range(len(cam_pos))],
            dtype=bool,
        )
    return masks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-agreement", type=float, default=0.85,
                        help="reject the run if the geometry disagrees with the recorded flag")
    args = parser.parse_args()

    manifest = json.loads((args.batch_root / "batch_manifest.json").read_text(encoding="utf-8"))
    clean = [int(e["idx"]) for e in manifest["episodes"] if e["verdict"] == "CLEAN"]

    report = {
        "batch_root": str(args.batch_root),
        "hfov_deg": HFOV_DEG,
        "vfov_deg": VFOV_DEG,
        "camera_forward_axis": "-Z",
        "quaternion_order": "xyzw",
        "sources": {},
    }
    agreements = []
    any_visible_execute = []
    per_object_execute = {}

    for source_id in clean:
        data = np.load(args.batch_root / f"ep_{source_id:04d}/data/trajectories.npz", allow_pickle=True)
        meta = json.loads(
            (args.batch_root / f"ep_{source_id:04d}/meta/episode_meta.json").read_text(encoding="utf-8")
        )
        episode = {key: data[key] for key in
                   ("head_camera_world_pos", "head_camera_world_quat", "human_pos",
                    "human_visible_flag", "object_pos", "fsm_state_id", "round_id")}
        object_names = [str(n) for n in data["object_names"]]

        agreement = calibration_agreement(episode)
        agreements.append(agreement)
        masks = episode_visibility(episode, object_names)

        execute = (episode["fsm_state_id"] == meta["fsm_states"].index("ExecuteBehaviorTask")) & (
            episode["round_id"] == 0
        )
        cans = [n for n in object_names if n.startswith("can_of_soda")]
        any_can = np.any([masks[n] for n in cans], axis=0)
        any_visible_execute.append(float(any_can[execute].mean()))
        for name in object_names:
            per_object_execute.setdefault(name, []).append(float(masks[name][execute].mean()))

        report["sources"][str(source_id)] = {
            "num_frames": int(len(episode["fsm_state_id"])),
            "flag_agreement": agreement,
            "execute_round0_frames": int(execute.sum()),
            "visible_fraction_execute": {n: float(masks[n][execute].mean()) for n in object_names},
            "any_can_visible_fraction_execute": float(any_can[execute].mean()),
            "visible_runs": {
                n: [[int(a), int(b)] for a, b in _runs(masks[n])] for n in object_names
            },
        }

    worst = float(np.min(agreements))
    report["calibration"] = {
        "mean_flag_agreement": float(np.mean(agreements)),
        "min_flag_agreement": worst,
        "min_required": args.min_agreement,
    }
    if worst < args.min_agreement:
        raise SystemExit(
            f"camera geometry only reproduces the recorded visibility flag at {worst:.1%} "
            f"(need {args.min_agreement:.0%}); refusing to emit a mask that may be wrong"
        )

    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{len(clean)} episodes; flag agreement mean {np.mean(agreements):.1%} "
          f"min {worst:.1%}")
    print(f"\nround0 Execute 段内目标可见帧占比：")
    for name, values in per_object_execute.items():
        print(f"  {name:<18} 中位 {np.median(values):.1%}  最低 {np.min(values):.1%}")
    print(f"  {'任意一个罐子可见':<18} 中位 {np.median(any_visible_execute):.1%}  "
          f"最低 {np.min(any_visible_execute):.1%}")
    print(f"\nwrote {args.output}")
    return 0


def _runs(mask: np.ndarray):
    """Contiguous [start, stop) spans where the mask is True."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2], strict=True))


if __name__ == "__main__":
    raise SystemExit(main())
