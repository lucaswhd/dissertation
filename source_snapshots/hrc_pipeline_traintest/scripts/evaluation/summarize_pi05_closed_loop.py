#!/usr/bin/env python3
"""Summarize HRC π0.5 rollout HDF5 files and compare E4 with initialization."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np


FSM_NAMES = {
    0: "WaitNearHuman",
    1: "ReceiveCommand",
    2: "GoToTask",
    3: "ExecuteBehaviorTask",
    4: "TaskSegmentDone",
    5: "ReturnToHuman",
    6: "FaceHuman",
    7: "WaitNextCommand",
    8: "AllTasksDone",
    9: "Relocate",
}
GRIPPER_DIMS = (14, 22)


def wilson_interval(successes: int, attempts: int, z: float = 1.959963984540054) -> list[float] | None:
    if attempts <= 0:
        return None
    p = successes / attempts
    denominator = 1.0 + z * z / attempts
    center = (p + z * z / (2.0 * attempts)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / attempts + z * z / (4.0 * attempts * attempts)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def paired_bootstrap_interval(values: np.ndarray, *, seed: int = 0, samples: int = 10_000) -> list[float] | None:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _json_attr(group: h5py.Group, key: str, default: Any = None) -> Any:
    if key not in group.attrs:
        return default
    value = group.attrs[key]
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value.item() if isinstance(value, np.generic) else value


def _tilt_deg(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    unit = quaternions / np.maximum(norms, 1e-12)
    x, y = unit[:, 0], unit[:, 1]
    body_up_dot_world_up = np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)
    return np.degrees(np.arccos(body_up_dot_world_up))


def summarize_rollout(hdf5_path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    phase_counts: dict[int, dict[str, int]] = {}
    round_success: dict[int, bool | None] = {}
    all_actions = []
    all_tilts = []
    follow_distances = []
    follow_visible = []
    follow_positions = []
    protocol_errors = []
    with h5py.File(hdf5_path, "r") as source:
        data = source["data"]
        metadata = {key: _json_attr(data, key) for key in data.attrs}
        for demo_name in sorted(data.keys()):
            demo = data[demo_name]
            round_id = int(demo.attrs["round_id"])
            fsm = np.asarray(demo["obs/fsm_state_id"], dtype=np.int32)
            actions = np.asarray(demo["action"], dtype=np.float32)
            quaternions = np.asarray(demo["obs/robot_quat"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 23:
                protocol_errors.append(f"{demo_name}: action shape {actions.shape}")
            if not np.isfinite(actions).all():
                protocol_errors.append(f"{demo_name}: non-finite action")
            all_actions.append(actions)
            tilts = _tilt_deg(quaternions)
            all_tilts.append(tilts)
            phase_counts[round_id] = {
                FSM_NAMES[state]: int(np.count_nonzero(fsm == state)) for state in FSM_NAMES
            }
            verdict = int(demo.attrs.get("round_success", -1))
            round_success[round_id] = None if verdict < 0 else bool(verdict)
            follow_mask = fsm == 9
            if np.any(follow_mask):
                relative = np.asarray(demo["obs/relative_human_pose"], dtype=np.float32)[follow_mask]
                visible = np.asarray(demo["obs/human_visible"], dtype=np.float32)[follow_mask]
                follow_distances.extend(np.linalg.norm(relative[:, :2], axis=1).tolist())
                follow_visible.extend(visible.tolist())
                if "human_pos" in demo["obs"]:
                    follow_positions.append(
                        np.asarray(demo["obs/human_pos"], dtype=np.float32)[follow_mask]
                    )
                else:
                    protocol_errors.append(f"{demo_name}: missing obs/human_pos for Follow")

    actions = np.concatenate(all_actions) if all_actions else np.empty((0, 23), dtype=np.float32)
    tilts = np.concatenate(all_tilts) if all_tilts else np.empty(0)
    distances = np.asarray(follow_distances, dtype=np.float64)
    visible = np.asarray(follow_visible, dtype=np.float64)
    human_positions = (
        np.concatenate(follow_positions, axis=0)
        if follow_positions
        else np.empty((0, 3), dtype=np.float32)
    )
    human_walk_m = float(
        np.linalg.norm(np.diff(human_positions[:, :2], axis=0), axis=1).sum()
    ) if len(human_positions) >= 2 else 0.0
    human_net_displacement_m = float(
        np.linalg.norm(human_positions[-1, :2] - human_positions[0, :2])
    ) if len(human_positions) >= 2 else 0.0
    follow_band_rate = float(np.mean((distances >= 0.7) & (distances <= 2.5))) if len(distances) else 0.0
    follow_visible_rate = float(np.mean(visible > 0.5)) if len(visible) else 0.0
    max_tilt = float(tilts.max()) if len(tilts) else float("nan")
    follow_success = bool(
        len(distances)
        and human_walk_m >= 0.5
        and follow_band_rate >= 0.8
        and follow_visible_rate >= 0.8
        and max_tilt < 45.0
    )

    def present(round_id: int, phase: str) -> bool:
        return phase_counts.get(round_id, {}).get(phase, 0) > 0

    partial = {
        "pickup_receive": present(0, "ReceiveCommand"),
        "pickup_go_to": present(0, "GoToTask"),
        "pickup_execute": present(0, "ExecuteBehaviorTask"),
        "pickup_return": present(0, "ReturnToHuman"),
        "follow_relocate": present(1, "Relocate"),
        "radio_receive": present(1, "ReceiveCommand"),
        "radio_go_to": present(1, "GoToTask"),
        "radio_execute": present(1, "ExecuteBehaviorTask"),
        "radio_return": present(1, "ReturnToHuman"),
    }
    task_success = {
        "pickup": round_success.get(0) is True,
        "follow": follow_success,
        "radio": round_success.get(1) is True,
    }
    gripper_metrics = {}
    for label, index in zip(("left", "right"), GRIPPER_DIMS):
        if actions.ndim != 2 or actions.shape[1] <= index:
            gripper_metrics[label] = {"switches": None, "open_fraction": None}
            continue
        signs = actions[:, index] >= 0.0 if len(actions) else np.empty(0, dtype=bool)
        gripper_metrics[label] = {
            "switches": int(np.count_nonzero(np.diff(signs))) if len(signs) else 0,
            "open_fraction": float(np.mean(signs)) if len(signs) else None,
        }
    server_metadata = metadata.get("policy_server") or {}
    expected_checkpoint = str(identity.get("checkpoint", ""))
    observed_checkpoint = str(server_metadata.get("checkpoint", ""))
    if expected_checkpoint and observed_checkpoint and expected_checkpoint != observed_checkpoint:
        protocol_errors.append(
            f"checkpoint mismatch expected={expected_checkpoint} observed={observed_checkpoint}"
        )
    live_metrics_path = hdf5_path.parent / "policy_rollout_metrics.json"
    live_metrics = None
    if live_metrics_path.exists():
        live_metrics = json.loads(live_metrics_path.read_text(encoding="utf-8"))
        if expected_checkpoint and str(live_metrics.get("checkpoint", "")) != expected_checkpoint:
            protocol_errors.append("contact metrics checkpoint does not match the run plan")

    return {
        **identity,
        "hdf5": str(hdf5_path),
        "metadata": metadata,
        "protocol_errors": protocol_errors,
        "task_success": task_success,
        "task_macro": float(np.mean(list(task_success.values()))),
        "partial_completion": partial,
        "partial_completion_rate": float(np.mean(list(partial.values()))),
        "follow": {
            "frames": int(len(distances)),
            "human_walk_m": human_walk_m,
            "human_net_displacement_m": human_net_displacement_m,
            "target_distance_m": 1.5,
            "distance_mae_m": float(np.mean(np.abs(distances - 1.5))) if len(distances) else None,
            "distance_mean_m": float(np.mean(distances)) if len(distances) else None,
            "in_band_rate_0p7_2p5": follow_band_rate,
            "visible_rate": follow_visible_rate,
        },
        "safety": {
            "max_robot_tilt_deg": max_tilt,
            "fell": bool(len(tilts) and max_tilt >= 45.0),
            "lost_human": bool(len(visible) and follow_visible_rate < 0.8),
            "timeout_or_incomplete_chain": not all(partial.values()),
            "collision": (
                bool(live_metrics["collision_frames"] > 0) if live_metrics is not None else None
            ),
            "collision_frames": (
                int(live_metrics["collision_frames"]) if live_metrics is not None else None
            ),
            "collision_bodies": (
                live_metrics.get("collision_bodies", []) if live_metrics is not None else None
            ),
        },
        "gripper": gripper_metrics,
        "navigation_arrival": (
            live_metrics.get("navigation_arrival", {}) if live_metrics is not None else {}
        ),
        "contact_and_drop": {
            "pickup_contact_frames": (
                int(live_metrics["task_contact_frames"]["pickup"])
                if live_metrics is not None else None
            ),
            "radio_contact_frames": (
                int(live_metrics["task_contact_frames"]["radio"])
                if live_metrics is not None else None
            ),
            "pickup_lifted": (
                live_metrics.get("pickup_lifted", {}) if live_metrics is not None else None
            ),
            "pickup_drop_detected": (
                live_metrics.get("pickup_drop_detected", {}) if live_metrics is not None else None
            ),
            "live_metrics_path": str(live_metrics_path) if live_metrics is not None else None,
        },
    }


def aggregate(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("split") != "val" or plan.get("sealed_test_access") is not False:
        raise ValueError("sealed test split is unavailable during closed-loop development")
    rows = []
    missing = []
    invalid = []
    for group in plan["model_groups"]:
        for run in group["runs"]:
            hdf5_path = Path(run["expected_hdf5"])
            if not hdf5_path.exists():
                missing.append(run["run_id"])
                continue
            identity = {
                key: run[key]
                for key in ("run_id", "model", "checkpoint", "source_id", "seed")
            }
            try:
                row = summarize_rollout(hdf5_path, identity)
            except Exception as exc:  # Keep one corrupt artifact from hiding the other runs.
                invalid.append(
                    {
                        **identity,
                        "hdf5": str(hdf5_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            output = Path(run["output_dir"]) / "rollout_metrics.json"
            output.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            if row["protocol_errors"]:
                invalid.append(
                    {
                        **identity,
                        "hdf5": str(hdf5_path),
                        "error": "; ".join(row["protocol_errors"]),
                    }
                )
                continue
            rows.append(row)

    per_model = {}
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        tasks = {}
        for task in ("pickup", "follow", "radio"):
            values = [bool(row["task_success"][task]) for row in model_rows]
            successes = sum(values)
            tasks[task] = {
                "successes": successes,
                "attempts": len(values),
                "rate": successes / len(values) if values else None,
                "wilson_95": wilson_interval(successes, len(values)),
            }
        navigation = {}
        for phase in ("GoToTask", "ReturnToHuman"):
            values = [
                bool(entry["arrived"])
                for row in model_rows
                for entry in row["navigation_arrival"].values()
                if entry.get("phase") == phase
            ]
            navigation[phase] = {
                "successes": sum(values),
                "attempts": len(values),
                "rate": sum(values) / len(values) if values else None,
            }
        per_model[model] = {
            "runs": len(model_rows),
            "tasks": tasks,
            "navigation_arrival": navigation,
            "task_macro_mean": float(np.mean([row["task_macro"] for row in model_rows])),
            "partial_completion_mean": float(
                np.mean([row["partial_completion_rate"] for row in model_rows])
            ),
        }

    pairs = {}
    for row in rows:
        pairs.setdefault((row["source_id"], row["seed"]), {})[row["model"]] = row
    # Read the compared model labels from the plan; fall back to the original
    # Gate 3 pair so older plans without a "comparison" field keep their behavior.
    comparison_labels = plan.get("comparison") or {}
    baseline = comparison_labels.get("baseline", "pi05_base_hrc_init")
    candidate = comparison_labels.get("candidate", "e4_5k")
    differences = []
    for pair in pairs.values():
        if candidate in pair and baseline in pair:
            differences.append(pair[candidate]["task_macro"] - pair[baseline]["task_macro"])
    difference_array = np.asarray(differences, dtype=np.float64)
    comparison = {
        "baseline": baseline,
        "candidate": candidate,
        "paired_runs": len(differences),
        "candidate_minus_baseline_task_macro": float(difference_array.mean()) if len(difference_array) else None,
        "paired_bootstrap_95": paired_bootstrap_interval(difference_array),
    }
    return {
        "gate": plan["gate"],
        "split": plan["split"],
        "sealed_test_access": plan["sealed_test_access"],
        "completed_runs": len(rows),
        "missing_runs": missing,
        "invalid_runs": invalid,
        "per_model": per_model,
        "comparison": comparison,
        "runs": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    report = aggregate(plan)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    destination = args.output or args.plan.with_name("closed_loop_report.json")
    destination.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if not report["missing_runs"] and not report["invalid_runs"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
