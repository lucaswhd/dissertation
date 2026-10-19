#!/usr/bin/env python
"""Audit whether task-target centers fall inside the recorded head-camera frustum."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC
from datetime import datetime
import itertools
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from scripts.hrc.dissertation_protocol import ACTION_HORIZON
from scripts.hrc.dissertation_protocol import file_sha256
from scripts.hrc.dissertation_protocol import load_split_rows
from scripts.hrc.dissertation_protocol import prompt_stable_mask
from scripts.hrc.dissertation_protocol import semantic_label
from scripts.hrc.dissertation_protocol import write_csv_atomic
from scripts.hrc.dissertation_protocol import write_json_atomic
from scripts.hrc.run_dissertation_conditioning_probe import SUBINSTRUCTION_PLAN
from scripts.hrc.run_dissertation_final_audit import BC_DATASET_VERSION
from scripts.hrc.run_dissertation_final_audit import DATA_ROOT
from scripts.hrc.run_dissertation_final_audit import REPO_ROOT
from scripts.hrc.run_dissertation_final_audit import SPLIT_MANIFEST

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/dissertation_target_visibility"
COLLECTION_CONFIG = Path("/root/autodl-tmp/hrc_pipeline_traintest/configs/hrc_multitask_0001.yaml")
HUMAN_BODY_OFFSET_M = 1.0

PER_FRAME_FIELDS = [
    "source_episode_id",
    "split",
    "raw_global_frame_index",
    "time_s",
    "context",
    "context_id",
    "round_id",
    "segment_index",
    "segment",
    "position",
    "segment_relative_position",
    "target_name",
    "target_position_source",
    "target_x",
    "target_y",
    "target_z",
    "target_center_in_frustum",
    "export_row",
    "valid_anchor",
    "strict_prompt_stable_anchor",
    "decision_frame_anchor",
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def quat_to_rotation(quaternion_xyzw: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError(f"invalid xyzw quaternion {quaternion}")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0:
        raise ValueError("zero-norm quaternion")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def point_in_frustum(
    camera_position: np.ndarray,
    camera_quaternion_xyzw: np.ndarray,
    target_position: np.ndarray,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
) -> bool:
    """Return whether one target point is inside a camera looking along local -Z."""
    camera_position = np.asarray(camera_position, dtype=np.float64)
    target_position = np.asarray(target_position, dtype=np.float64)
    if camera_position.shape != (3,) or target_position.shape != (3,):
        raise ValueError("camera and target positions must be three-vectors")
    if not (0 < horizontal_fov_deg < 180 and 0 < vertical_fov_deg < 180):
        raise ValueError("FOV angles must lie in (0,180) degrees")
    local = quat_to_rotation(camera_quaternion_xyzw).T @ (target_position - camera_position)
    forward = -float(local[2])
    if forward <= 0:
        return False
    horizontal = np.degrees(np.arctan2(abs(float(local[0])), forward))
    vertical = np.degrees(np.arctan2(abs(float(local[1])), forward))
    return bool(horizontal <= horizontal_fov_deg / 2 and vertical <= vertical_fov_deg / 2)


def points_in_frustum(
    camera_positions: np.ndarray,
    camera_quaternions_xyzw: np.ndarray,
    target_positions: np.ndarray,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
) -> np.ndarray:
    camera_positions = np.asarray(camera_positions, dtype=np.float64)
    quaternions = np.asarray(camera_quaternions_xyzw, dtype=np.float64)
    targets = np.asarray(target_positions, dtype=np.float64)
    if camera_positions.shape != targets.shape or camera_positions.ndim != 2 or camera_positions.shape[1] != 3:
        raise ValueError(f"camera/target arrays must both be [N,3], got {camera_positions.shape}/{targets.shape}")
    if quaternions.shape != (len(camera_positions), 4):
        raise ValueError(f"camera quaternions must be [N,4], got {quaternions.shape}")
    norms = np.linalg.norm(quaternions, axis=1)
    if not np.isfinite(camera_positions).all() or not np.isfinite(targets).all() or not np.isfinite(quaternions).all():
        raise ValueError("non-finite camera or target geometry")
    if np.any(norms <= 0):
        raise ValueError("zero-norm camera quaternion")
    q = quaternions / norms[:, None]
    x, y, z, w = q.T
    rotations = np.empty((len(q), 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotations[:, 0, 1] = 2 * (x * y - z * w)
    rotations[:, 0, 2] = 2 * (x * z + y * w)
    rotations[:, 1, 0] = 2 * (x * y + z * w)
    rotations[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotations[:, 1, 2] = 2 * (y * z - x * w)
    rotations[:, 2, 0] = 2 * (x * z - y * w)
    rotations[:, 2, 1] = 2 * (y * z + x * w)
    rotations[:, 2, 2] = 1 - 2 * (x * x + y * y)
    local = np.einsum("nji,nj->ni", rotations, targets - camera_positions)
    forward = -local[:, 2]
    horizontal = np.degrees(np.arctan2(np.abs(local[:, 0]), forward))
    vertical = np.degrees(np.arctan2(np.abs(local[:, 1]), forward))
    return (
        (forward > 0)
        & (horizontal <= horizontal_fov_deg / 2)
        & (vertical <= vertical_fov_deg / 2)
    )


def optimal_target_assignment(scores: np.ndarray, names: list[str]) -> dict[str, Any]:
    """Maximize three pickup-to-can motion scores under a one-to-one assignment."""
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (3, 3) or len(names) != 3 or not np.isfinite(values).all():
        raise ValueError(f"expected finite [3,3] scores and three names, got {values.shape}/{len(names)}")
    candidates = []
    for assignment in itertools.permutations(range(3)):
        score = float(sum(values[row, assignment[row]] for row in range(3)))
        candidates.append((score, assignment))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score, best = candidates[0]
    second_score = candidates[1][0]
    return {
        "assigned_indices": [int(value) for value in best],
        "assigned_names": [str(names[value]) for value in best],
        "best_score": best_score,
        "second_best_score": second_score,
        "margin": best_score - second_score,
        "score_matrix": values.tolist(),
        "all_assignment_scores": [
            {"assigned_indices": list(assignment), "score": score}
            for score, assignment in candidates
        ],
    }


def summarize_visibility(
    raw_frames: np.ndarray,
    in_frustum: np.ndarray,
    *,
    segment_start: int,
    segment_stop: int,
    fps: float,
) -> dict[str, float | int | None]:
    frames = np.asarray(raw_frames, dtype=np.int64)
    mask = np.asarray(in_frustum, dtype=bool)
    if frames.shape != mask.shape or frames.ndim != 1 or not len(frames):
        raise ValueError("raw_frames and in_frustum must be non-empty matching vectors")
    if segment_stop <= segment_start or fps <= 0:
        raise ValueError("invalid segment bounds or fps")
    visible_frames = frames[mask]
    first = int(visible_frames[0]) if len(visible_frames) else None
    denominator = max(1, segment_stop - segment_start - 1)
    return {
        "total_frames": len(frames),
        "in_frustum_frames": int(mask.sum()),
        "not_in_frustum_frames": int((~mask).sum()),
        "in_frustum_ratio": float(mask.mean()),
        "first_in_frustum_raw_frame": first,
        "first_in_frustum_time_s": None if first is None else float((first - segment_start) / fps),
        "first_in_frustum_segment_position": None if first is None else float((first - segment_start) / denominator),
    }


def _read_collection_config() -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(COLLECTION_CONFIG.read_text(encoding="utf-8"))
    radio = next(item for item in config["extra_objects"] if item["name"] == "radio_89")
    return {
        "fps": float(config["fps"]),
        "radio_name": str(radio["name"]),
        "radio_position": [float(value) for value in radio["position"]],
    }


def _read_intrinsics(hdf5_path: Path) -> dict[str, Any]:
    import h5py

    with h5py.File(hdf5_path, "r") as handle:
        raw = handle["data"].attrs.get("head_camera_intrinsics")
    if raw is None:
        raise ValueError(f"missing head_camera_intrinsics in {hdf5_path}")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    value = json.loads(str(raw))
    for key in ("sensor_name", "image_height", "image_width", "hfov_deg", "vfov_deg"):
        if key not in value:
            raise ValueError(f"intrinsics in {hdf5_path} lack {key}")
    return value


def _load_export_index(split: dict[str, Any]) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    index: dict[tuple[int, int], dict[str, Any]] = {}
    reports = {}
    for split_name, source_ids in split["source_splits"].items():
        rows = load_split_rows(DATA_ROOT / BC_DATASET_VERSION / split_name)
        stable = prompt_stable_mask(rows, horizon=ACTION_HORIZON)
        reports[split_name] = {
            "rows": len(rows),
            "valid_anchors": int(rows.valid.sum()),
            "decision_anchors": int(np.count_nonzero(rows.valid & (rows.task_id == 1))),
        }
        expected = {int(value) for value in source_ids}
        observed = {int(value) for value in np.unique(rows.source_id)}
        if observed != expected:
            raise ValueError(f"{split_name} source mismatch: {sorted(observed)} != {sorted(expected)}")
        for row in range(len(rows)):
            key = (int(rows.source_id[row]), int(rows.raw_frame[row]))
            if key in index:
                raise ValueError(f"duplicate exported source/frame key {key}")
            index[key] = {
                "split": split_name,
                "export_row": row,
                "valid_anchor": bool(rows.valid[row]),
                "strict_prompt_stable_anchor": bool(stable[row]),
                "decision_frame_anchor": bool(rows.valid[row] and rows.task_id[row] == 1),
                "context_id": int(rows.context_id[row]),
                "prompt": str(rows.prompts[row]),
            }
    return index, reports


def _pickup_assignment(
    positions: np.ndarray,
    object_names: list[str],
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    can_indices = [index for index, name in enumerate(object_names) if name.startswith("can_of_soda")]
    if len(can_indices) != 3:
        raise ValueError(f"expected three soda cans, found {[object_names[i] for i in can_indices]}")
    pickup_indices = [
        index for index, segment in enumerate(segments)
        if semantic_label(segment["label"]).casefold() == "pick up the soda can"
    ]
    if len(pickup_indices) != 3:
        raise ValueError(f"expected three pickup segments, found {pickup_indices}")
    scores = np.zeros((3, 3), dtype=np.float64)
    for pickup_order, segment_index in enumerate(pickup_indices):
        deposit = segments[segment_index + 1]
        start, stop = int(deposit["start"]), int(deposit["stop"])
        path_length = np.linalg.norm(np.diff(positions[start:stop], axis=0), axis=2).sum(axis=0)
        scores[pickup_order] = path_length[can_indices]
    result = optimal_target_assignment(scores, [object_names[index] for index in can_indices])
    result["pickup_segment_indices"] = pickup_indices
    result["method"] = "one-to-one maximum total object path length over each following deposit segment"
    if result["margin"] <= 0:
        raise ValueError("pickup target assignment is tied")
    return result


def _position_bucket(relative: float) -> str:
    if relative < 1 / 3:
        return "early"
    if relative < 2 / 3:
        return "middle"
    return "late"


def _target_segments(source_plan: dict[str, Any]) -> list[tuple[str, int, int, int, dict[str, Any]]]:
    selected = []
    for index, segment in enumerate(source_plan["0"]["segments"]):
        if "stand by" not in str(segment["label"]).casefold():
            selected.append(("pickup", 1, 0, index, segment))
    for index, segment in enumerate(source_plan["1"]["segments"]):
        if semantic_label(segment["label"]).casefold() == "turn on the radio receiver":
            selected.append(("radio", 3, 1, index, segment))
    return selected


def _population_summary(rows: list[dict[str, Any]], population: str) -> dict[str, Any]:
    if population == "frames":
        selected = rows
    elif population == "valid_anchors":
        selected = [row for row in rows if row["valid_anchor"]]
    elif population == "strict_prompt_stable_anchors":
        selected = [row for row in rows if row["strict_prompt_stable_anchor"]]
    elif population == "decision_frame_anchors":
        selected = [row for row in rows if row["decision_frame_anchor"]]
    else:
        raise ValueError(population)
    count = len(selected)
    visible = sum(bool(row["target_center_in_frustum"]) for row in selected)
    return {
        "population": population,
        "total": count,
        "target_center_in_frustum": visible,
        "target_center_not_in_frustum": count - visible,
        "target_center_in_frustum_ratio": visible / count if count else None,
    }


def _summary_rows(per_frame: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_fields = ("split", "context", "segment", "position")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in per_frame:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        common = dict(zip(group_fields, key, strict=True))
        output.extend(
            {**common, **_population_summary(rows, population)}
            for population in (
                "frames", "valid_anchors", "strict_prompt_stable_anchors", "decision_frame_anchors"
            )
        )
    return output


def _per_source_rows(per_frame: list[dict[str, Any]], *, fps: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_frame:
        grouped[(int(row["source_episode_id"]), str(row["split"]), str(row["context"]))].append(row)
    output = []
    for (source, split, context), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["raw_global_frame_index"]))
        first_context_frame = int(rows[0]["raw_global_frame_index"])
        visible_rows = [row for row in rows if row["target_center_in_frustum"]]
        first = visible_rows[0] if visible_rows else None
        frames = _population_summary(rows, "frames")
        valid = _population_summary(rows, "valid_anchors")
        decision = _population_summary(rows, "decision_frame_anchors")
        strict = _population_summary(rows, "strict_prompt_stable_anchors")
        output.append(
            {
                "source_episode_id": source,
                "split": split,
                "context": context,
                "total_relevant_frames": frames["total"],
                "target_center_in_frustum_frames": frames["target_center_in_frustum"],
                "target_center_not_in_frustum_frames": frames["target_center_not_in_frustum"],
                "target_center_in_frustum_ratio": frames["target_center_in_frustum_ratio"],
                "first_in_frustum_raw_frame": None if first is None else first["raw_global_frame_index"],
                "first_in_frustum_time_s_from_context_start": None if first is None else (
                    (int(first["raw_global_frame_index"]) - first_context_frame) / fps
                ),
                "first_in_frustum_segment": None if first is None else first["segment"],
                "first_in_frustum_segment_position": None if first is None else first["segment_relative_position"],
                "valid_anchors": valid["total"],
                "valid_anchor_in_frustum": valid["target_center_in_frustum"],
                "valid_anchor_not_in_frustum": valid["target_center_not_in_frustum"],
                "valid_anchor_in_frustum_ratio": valid["target_center_in_frustum_ratio"],
                "strict_prompt_stable_anchors": strict["total"],
                "strict_prompt_stable_in_frustum_ratio": strict["target_center_in_frustum_ratio"],
                "decision_frame_anchors": decision["total"],
                "decision_frame_in_frustum": decision["target_center_in_frustum"],
                "decision_frame_not_in_frustum": decision["target_center_not_in_frustum"],
                "decision_frame_in_frustum_ratio": decision["target_center_in_frustum_ratio"],
            }
        )
    return output


PER_SOURCE_FIELDS = [
    "source_episode_id", "split", "context", "total_relevant_frames",
    "target_center_in_frustum_frames", "target_center_not_in_frustum_frames",
    "target_center_in_frustum_ratio", "first_in_frustum_raw_frame",
    "first_in_frustum_time_s_from_context_start", "first_in_frustum_segment",
    "first_in_frustum_segment_position", "valid_anchors", "valid_anchor_in_frustum",
    "valid_anchor_not_in_frustum", "valid_anchor_in_frustum_ratio",
    "strict_prompt_stable_anchors", "strict_prompt_stable_in_frustum_ratio",
    "decision_frame_anchors", "decision_frame_in_frustum", "decision_frame_not_in_frustum",
    "decision_frame_in_frustum_ratio",
]

SUMMARY_FIELDS = [
    "split", "context", "segment", "position", "population", "total",
    "target_center_in_frustum", "target_center_not_in_frustum",
    "target_center_in_frustum_ratio",
]


def _git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


def run(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    final_files = [
        output_root / "per_frame.csv",
        output_root / "per_anchor.csv",
        output_root / "per_source.csv",
        output_root / "summary.csv",
        output_root / "summary.json",
        output_root / "README.md",
    ]
    existing = [str(path) for path in final_files if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing visibility outputs: {existing}")
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    plan = json.loads(SUBINSTRUCTION_PLAN.read_text(encoding="utf-8"))
    collection = _read_collection_config()
    fps = collection["fps"]
    export_index, export_report = _load_export_index(split)
    clean_sources = [int(value) for value in split["expected_clean"]]
    batch_root = Path(split["batch_root"])
    source_to_split = {
        int(source): split_name
        for split_name, sources in split["source_splits"].items()
        for source in sources
    }
    if sorted(source_to_split) != sorted(clean_sources):
        raise ValueError("source split assignment does not cover the frozen CLEAN population exactly")

    per_frame: list[dict[str, Any]] = []
    assignments: dict[str, Any] = {}
    calibrations: dict[str, float] = {}
    intrinsics_reference = None
    missing_files = []
    for source_order, source_id in enumerate(clean_sources, start=1):
        episode_root = batch_root / f"ep_{source_id:04d}"
        npz_path = episode_root / "data/trajectories.npz"
        hdf5_path = episode_root / "hrc_dataset.hdf5"
        video_path = episode_root / "videos/head_rgb.mp4"
        source_missing = [
            str(path) for path in (npz_path, hdf5_path, video_path) if not path.is_file()
        ]
        missing_files.extend(source_missing)
        if source_missing:
            continue
        intrinsics = _read_intrinsics(hdf5_path)
        if intrinsics_reference is None:
            intrinsics_reference = intrinsics
        elif intrinsics != intrinsics_reference:
            raise ValueError(f"source {source_id} camera intrinsics differ from earlier sources")
        with np.load(npz_path, allow_pickle=True) as data:
            required = {
                "head_camera_world_pos", "head_camera_world_quat", "human_pos",
                "human_visible_flag", "object_pos", "object_names", "fsm_state_id", "round_id",
            }
            absent = sorted(required - set(data.files))
            if absent:
                raise ValueError(f"{npz_path} lacks {absent}")
            camera_positions = np.asarray(data["head_camera_world_pos"], dtype=np.float64)
            camera_quaternions = np.asarray(data["head_camera_world_quat"], dtype=np.float64)
            human = np.asarray(data["human_pos"], dtype=np.float64).copy()
            human[:, 2] += HUMAN_BODY_OFFSET_M
            human_flag = np.asarray(data["human_visible_flag"], dtype=bool)
            object_positions = np.asarray(data["object_pos"], dtype=np.float64)
            object_names = [str(value) for value in data["object_names"]]
        n = len(camera_positions)
        if any(len(value) != n for value in (camera_quaternions, human, human_flag, object_positions)):
            raise ValueError(f"source {source_id} raw arrays have inconsistent lengths")
        human_geometry = points_in_frustum(
            camera_positions,
            camera_quaternions,
            human,
            float(intrinsics["hfov_deg"]),
            float(intrinsics["vfov_deg"]),
        )
        agreement = float(np.mean(human_geometry == human_flag))
        calibrations[str(source_id)] = agreement
        source_plan = plan["sources"][str(source_id)]
        pickup_segments = source_plan["0"]["segments"]
        assignment = _pickup_assignment(object_positions, object_names, pickup_segments)
        assignments[str(source_id)] = assignment
        pickup_target_by_segment = dict(
            zip(assignment["pickup_segment_indices"], assignment["assigned_names"], strict=True)
        )
        object_index = {name: index for index, name in enumerate(object_names)}
        if "trash_can_116" not in object_index:
            raise ValueError(f"source {source_id} lacks trash_can_116")

        for context, context_id, round_id, segment_index, segment in _target_segments(source_plan):
            start, stop = int(segment["start"]), int(segment["stop"])
            if not (0 <= start < stop <= n):
                raise ValueError(f"source {source_id} invalid segment [{start},{stop})")
            label = semantic_label(str(segment["label"]))
            if context == "radio":
                target_name = collection["radio_name"]
                target_source = "configured_static_world_reference_not_per_frame_recorded"
                targets = np.repeat(
                    np.asarray(collection["radio_position"], dtype=np.float64)[None, :],
                    stop - start,
                    axis=0,
                )
            elif segment_index in pickup_target_by_segment:
                target_name = pickup_target_by_segment[segment_index]
                target_source = "recorded_object_center_inferred_target_identity"
                targets = object_positions[start:stop, object_index[target_name]]
            else:
                target_name = "trash_can_116"
                target_source = "recorded_object_center"
                targets = object_positions[start:stop, object_index[target_name]]
            visible = points_in_frustum(
                camera_positions[start:stop],
                camera_quaternions[start:stop],
                targets,
                float(intrinsics["hfov_deg"]),
                float(intrinsics["vfov_deg"]),
            )
            denominator = max(1, stop - start - 1)
            for offset, frame in enumerate(range(start, stop)):
                key = (source_id, frame)
                if key not in export_index:
                    raise ValueError(f"raw relevant frame {key} has no exported row")
                exported = export_index[key]
                if exported["context_id"] != context_id:
                    raise ValueError(
                        f"frame {key} context {exported['context_id']} != planned {context_id}"
                    )
                relative = offset / denominator
                per_frame.append(
                    {
                        "source_episode_id": source_id,
                        "split": source_to_split[source_id],
                        "raw_global_frame_index": frame,
                        "time_s": frame / fps,
                        "context": context,
                        "context_id": context_id,
                        "round_id": round_id,
                        "segment_index": segment_index,
                        "segment": label,
                        "position": _position_bucket(relative),
                        "segment_relative_position": relative,
                        "target_name": target_name,
                        "target_position_source": target_source,
                        "target_x": float(targets[offset, 0]),
                        "target_y": float(targets[offset, 1]),
                        "target_z": float(targets[offset, 2]),
                        "target_center_in_frustum": bool(visible[offset]),
                        "export_row": int(exported["export_row"]),
                        "valid_anchor": bool(exported["valid_anchor"]),
                        "strict_prompt_stable_anchor": bool(exported["strict_prompt_stable_anchor"]),
                        "decision_frame_anchor": bool(exported["decision_frame_anchor"]),
                    }
                )
        print(
            f"[visibility] source {source_id} ({source_order}/{len(clean_sources)}), "
            f"calibration={agreement:.3f}, assignment_margin={assignment['margin']:.3f}",
            flush=True,
        )

    if missing_files:
        raise FileNotFoundError(f"missing raw evidence files: {missing_files}")
    if len(assignments) != len(clean_sources):
        raise ValueError(f"only {len(assignments)} of {len(clean_sources)} sources were processed")
    if intrinsics_reference is None:
        raise ValueError("no camera intrinsics loaded")

    per_frame.sort(
        key=lambda row: (
            int(row["source_episode_id"]), int(row["raw_global_frame_index"]),
            int(row["round_id"]), int(row["segment_index"]),
        )
    )
    per_anchor = [row for row in per_frame if row["valid_anchor"]]
    per_source = _per_source_rows(per_frame, fps=fps)
    summary_rows = _summary_rows(per_frame)
    write_csv_atomic(output_root / "per_frame.csv", per_frame, PER_FRAME_FIELDS)
    write_csv_atomic(output_root / "per_anchor.csv", per_anchor, PER_FRAME_FIELDS)
    write_csv_atomic(output_root / "per_source.csv", per_source, PER_SOURCE_FIELDS)
    write_csv_atomic(output_root / "summary.csv", summary_rows, SUMMARY_FIELDS)

    calibration_values = list(calibrations.values())
    split_context_summary = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_frame:
        grouped[(row["split"], row["context"])].append(row)
    for (split_name, context), rows in sorted(grouped.items()):
        split_context_summary.append(
            {
                "split": split_name,
                "context": context,
                "frames": _population_summary(rows, "frames"),
                "valid_anchors": _population_summary(rows, "valid_anchors"),
                "strict_prompt_stable_anchors": _population_summary(rows, "strict_prompt_stable_anchors"),
                "decision_frame_anchors": _population_summary(rows, "decision_frame_anchors"),
            }
        )
    report = {
        "schema_version": 1,
        "completed_at_utc": utc_now(),
        "dataset_version": BC_DATASET_VERSION,
        "split_manifest": str(SPLIT_MANIFEST),
        "split_manifest_sha256": file_sha256(SPLIT_MANIFEST),
        "subinstruction_plan": str(SUBINSTRUCTION_PLAN),
        "subinstruction_plan_sha256": file_sha256(SUBINSTRUCTION_PLAN),
        "batch_root": str(batch_root),
        "num_sources": len(clean_sources),
        "source_ids": clean_sources,
        "export_report": export_report,
        "num_relevant_frames": len(per_frame),
        "num_valid_relevant_anchors": len(per_anchor),
        "visibility_definition": {
            "reported_quantity": "target_center_in_frustum",
            "camera": intrinsics_reference["sensor_name"],
            "camera_forward_axis": "local -Z",
            "quaternion_order": "xyzw",
            "horizontal_fov_deg": float(intrinsics_reference["hfov_deg"]),
            "vertical_fov_deg": float(intrinsics_reference["vfov_deg"]),
            "intrinsics_source": "data.attrs.head_camera_intrinsics in every source hrc_dataset.hdf5",
            "extrinsics_source": "per-frame trajectories.npz head_camera_world_pos/head_camera_world_quat",
            "target_representation": "single 3-D center point",
            "bounding_box_used": False,
            "rendered_pixels_used": False,
            "occlusion_considered": False,
        },
        "radio_target": {
            "name": collection["radio_name"],
            "position": collection["radio_position"],
            "source": str(COLLECTION_CONFIG),
            "limitation": "configured static reference; the radio pose was not recorded per frame",
        },
        "pickup_target_assignments": assignments,
        "calibration": {
            "status": "descriptive_only_not_a_camera_geometry_calibration",
            "reference": "recorded human_visible_flag compared with +1.0 m body-center point",
            "human_flag_definition": "robot-base XY bearing <=42.5 degrees and distance <=3 m; ignores head-camera pose and vertical FOV",
            "threshold_applied": False,
            "mean_agreement": float(np.mean(calibration_values)),
            "minimum_agreement": float(np.min(calibration_values)),
            "by_source": calibrations,
        },
        "split_context_summary": split_context_summary,
        "missing_files": missing_files,
    }
    write_json_atomic(output_root / "summary.json", report)

    commit = _git_value("rev-parse", "HEAD")
    branch = _git_value("branch", "--show-current")
    dirty = bool(_git_value("status", "--short"))
    readme = f"""# Dissertation Target-Center Frustum Audit

Status: complete

## Provenance

- Git commit: `{commit}`
- Branch: `{branch}`
- Worktree at completion: `{'dirty' if dirty else 'clean'}`
- Dataset: `{BC_DATASET_VERSION}`
- Frozen population: all 45 CLEAN sources, split labels retained
- Batch root: `{batch_root}`
- Split manifest SHA-256: `{report['split_manifest_sha256']}`
- Subinstruction plan SHA-256: `{report['subinstruction_plan_sha256']}`
- Camera: `{intrinsics_reference['sensor_name']}`
- Image size: `{intrinsics_reference['image_width']}x{intrinsics_reference['image_height']}`
- Horizontal/vertical FOV: `{intrinsics_reference['hfov_deg']}` / `{intrinsics_reference['vfov_deg']}` degrees
- Relevant frames: `{len(per_frame)}`
- Valid relevant anchors: `{len(per_anchor)}`
- Descriptive human-flag agreement: mean `{np.mean(calibration_values):.6f}`, minimum `{np.min(calibration_values):.6f}`
- Missing files: none
- Completed UTC: `{report['completed_at_utc']}`

## Visibility Definition

The reported variable is **target center in the geometric head-camera frustum**. The camera uses per-frame recorded world position and xyzw quaternion, looks along local `-Z`, and uses intrinsics stored in each source HDF5. The target is one 3-D center point. No image detector, target bounding box, depth buffer, ray cast, or occlusion test is used. Therefore these files do not establish unobstructed visual observability.

The old script described `human_visible_flag` as an independent camera calibration, but collection code defines that flag from robot-base XY bearing (85-degree FOV) plus a 3 m distance cutoff; it ignores the head-camera pose, pitch, vertical FOV, and recorded 105.85-degree HFOV. Agreement is retained as a descriptive cross-check only and no threshold is applied. Camera convention is instead grounded in the collector's recorded head-camera transforms and explicit local `-Z` convention.

Pickup target identity was not directly recorded. It is an explicit inference: jointly assign the three cans to the three pickup segments by maximizing total recorded can path length in each following deposit segment under a one-to-one constraint. Every 3x3 score matrix, complete permutation score list, winning identity, and margin is stored in `summary.json`. Trash-targeted segments use the recorded trash-can center. Radio uses the static configured position from `{COLLECTION_CONFIG}` because no per-frame radio pose was recorded; radio results are auxiliary geometric estimates with that limitation.

Early/middle/late are segment-relative thirds. `valid_anchor` and `decision_frame_anchor` come directly from the B/C export. `strict_prompt_stable_anchor` additionally requires a complete 32-step future with constant prompt and context. Full train/val/test population geometry is descriptive only; test values did not select or alter the protocol.

## Exact Command

```bash
.venv/bin/python -m scripts.hrc.run_dissertation_target_visibility
```

## Evidence

- `per_frame.csv`: every relevant frame and geometric boolean.
- `per_anchor.csv`: the subset marked valid by the exported B/C dataset.
- `per_source.csv`: source/context counts, ratios, and first-in-frustum timing.
- `summary.csv`: split/context/segment/position results for frames, valid anchors, strict anchors, and decision anchors.
- `summary.json`: definition, calibration, input hashes, complete target assignments, and aggregate results.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    print(f"[visibility] wrote complete evidence to {output_root}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        run(args)
    except Exception as exc:
        write_json_atomic(
            args.output_root.resolve() / "failure.json",
            {
                "status": "failed",
                "failed_at_utc": utc_now(),
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "argv": sys.argv,
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
