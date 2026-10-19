#!/usr/bin/env python
"""Pure, shared protocol utilities for the frozen dissertation experiments."""

from __future__ import annotations

import csv
import dataclasses
import glob
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ACTION_DIM = 23
MODEL_ACTION_DIM = 32
ACTION_HORIZON = 32
CONTEXT_NAMES = ("wait", "pickup", "follow", "radio")
BASE_DIMS = (0, 1, 2)
GRIPPER_DIMS = (14, 22)
DELTA_DIMS = tuple(i for i in range(ACTION_DIM) if i not in {*BASE_DIMS, *GRIPPER_DIMS})
ACTION_GROUPS = {
    "arms": DELTA_DIMS,
    "base": BASE_DIMS,
    "grippers": GRIPPER_DIMS,
}


@dataclasses.dataclass(frozen=True)
class SplitRows:
    """Physical rows from one LeRobot split, ordered by the exported global index."""

    split_root: Path
    dataset_version: str
    state: np.ndarray
    action: np.ndarray
    valid: np.ndarray
    context_id: np.ndarray
    source_id: np.ndarray
    fsm_state_id: np.ndarray
    raw_frame: np.ndarray
    episode_index: np.ndarray
    task_index: np.ndarray
    prompts: np.ndarray
    task_id: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.action)


def load_split_rows(split_root: Path) -> SplitRows:
    """Load the columns needed by every dissertation experiment."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    split_root = Path(split_root).resolve()
    files = sorted(glob.glob(str(split_root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet data below {split_root}")
    table = pa.concat_tables([pq.read_table(path) for path in files])
    data = table.to_pydict()
    required = {
        "observation.state",
        "action",
        "hrc.valid_anchor",
        "hrc.task_id",
        "hrc.context_id",
        "hrc.source_episode_id",
        "hrc.fsm_state_id",
        "hrc.raw_global_frame_index",
        "episode_index",
        "task_index",
        "index",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"{split_root} is missing required columns: {missing}")

    order = np.argsort(np.asarray(data["index"], dtype=np.int64))
    tasks_path = split_root / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        raise FileNotFoundError(tasks_path)
    tasks = pq.read_table(tasks_path).to_pydict()
    index_to_task = {
        int(index): str(task)
        for index, task in zip(tasks["task_index"], tasks["task"], strict=True)
    }
    task_index = np.asarray(data["task_index"], dtype=np.int64)[order]
    try:
        prompts = np.asarray([index_to_task[int(index)] for index in task_index], dtype=object)
    except KeyError as exc:
        raise ValueError(f"task index {exc.args[0]} has no prompt in {tasks_path}") from exc

    rows = SplitRows(
        split_root=split_root,
        dataset_version=split_root.parent.name,
        state=np.asarray(data["observation.state"], dtype=np.float32)[order],
        action=np.asarray(data["action"], dtype=np.float32)[order],
        valid=np.asarray(data["hrc.valid_anchor"], dtype=bool)[order],
        task_id=np.asarray(data["hrc.task_id"], dtype=np.int64)[order],
        context_id=np.asarray(data["hrc.context_id"], dtype=np.int64)[order],
        source_id=np.asarray(data["hrc.source_episode_id"], dtype=np.int64)[order],
        fsm_state_id=np.asarray(data["hrc.fsm_state_id"], dtype=np.int64)[order],
        raw_frame=np.asarray(data["hrc.raw_global_frame_index"], dtype=np.int64)[order],
        episode_index=np.asarray(data["episode_index"], dtype=np.int64)[order],
        task_index=task_index,
        prompts=prompts,
    )
    _validate_split_rows(rows)
    return rows


def _validate_split_rows(rows: SplitRows) -> None:
    lengths = {
        len(rows.state),
        len(rows.action),
        len(rows.valid),
        len(rows.context_id),
        len(rows.source_id),
        len(rows.fsm_state_id),
        len(rows.raw_frame),
        len(rows.episode_index),
        len(rows.task_index),
        len(rows.prompts),
    }
    if rows.task_id is not None:
        lengths.add(len(rows.task_id))
    if len(lengths) != 1:
        raise ValueError(f"split columns have inconsistent lengths: {sorted(lengths)}")
    if rows.state.shape != (len(rows), ACTION_DIM) or rows.action.shape != (len(rows), ACTION_DIM):
        raise ValueError(
            f"expected state/action [N,{ACTION_DIM}], got {rows.state.shape}/{rows.action.shape}"
        )
    if not np.isfinite(rows.state).all() or not np.isfinite(rows.action).all():
        raise ValueError(f"non-finite state/action values in {rows.split_root}")
    if np.any((rows.context_id < 0) | (rows.context_id >= len(CONTEXT_NAMES))):
        raise ValueError(f"invalid context id in {rows.split_root}")


def context_stable_mask(rows: SplitRows, *, horizon: int = ACTION_HORIZON) -> np.ndarray:
    """Mask rows whose future stays within one source, episode, and task context."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    stable = np.zeros(len(rows), dtype=bool)
    for row in np.flatnonzero(rows.valid):
        stop = int(row) + horizon
        if stop >= len(rows):
            continue
        if rows.source_id[stop] != rows.source_id[row]:
            continue
        if rows.episode_index[stop] != rows.episode_index[row]:
            continue
        if rows.raw_frame[stop] - rows.raw_frame[row] != horizon:
            continue
        window = slice(int(row), stop + 1)
        stable[row] = bool(np.all(rows.context_id[window] == rows.context_id[row]))
    return stable


def prompt_stable_mask(rows: SplitRows, *, horizon: int = ACTION_HORIZON) -> np.ndarray:
    """Mask rows whose inclusive [anchor, anchor+horizon] window stays semantically fixed."""
    stable = context_stable_mask(rows, horizon=horizon)
    for row in np.flatnonzero(stable):
        stop = int(row) + horizon
        stable[row] = bool(np.all(rows.prompts[int(row) : stop + 1] == rows.prompts[row]))
    return stable


def common_anchor_mask(a: SplitRows, b: SplitRows, *, horizon: int = ACTION_HORIZON) -> np.ndarray:
    """Validate physical row equality and return the strict A/B common-anchor mask."""
    if len(a) != len(b):
        raise ValueError(f"row count differs: {len(a)} != {len(b)}")
    for name in ("source_id", "raw_frame", "context_id", "fsm_state_id", "episode_index"):
        if not np.array_equal(getattr(a, name), getattr(b, name)):
            raise ValueError(f"{name} rows differ between datasets")
    if not np.array_equal(a.state, b.state):
        raise ValueError("state rows differ between datasets")
    if not np.array_equal(a.action, b.action):
        raise ValueError("action rows differ between datasets")
    return a.valid & b.valid & context_stable_mask(a, horizon=horizon) & context_stable_mask(b, horizon=horizon)


def sample_source_balanced(
    rows: SplitRows,
    mask: np.ndarray,
    *,
    context_id: int,
    per_source: int,
    seed: int,
    allow_exhausted: bool = False,
) -> np.ndarray:
    """Choose an equal seeded quota per source, or all rows when exhaustion is allowed."""
    if per_source <= 0:
        raise ValueError("per_source must be positive")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(rows),):
        raise ValueError(f"mask must have shape {(len(rows),)}, got {mask.shape}")
    selected: list[np.ndarray] = []
    sources = sorted(int(source) for source in np.unique(rows.source_id[mask]))
    if not sources:
        raise ValueError(f"no candidates for context {context_id}")
    for source in sources:
        candidates = np.flatnonzero(mask & (rows.context_id == context_id) & (rows.source_id == source))
        if len(candidates) < per_source and not allow_exhausted:
            raise ValueError(
                f"source {source} context {context_id} has {len(candidates)} candidates, need {per_source}"
            )
        if len(candidates) <= per_source:
            chosen = candidates
        else:
            rng = np.random.default_rng(np.random.SeedSequence([seed, context_id, source]))
            chosen = np.sort(rng.choice(candidates, size=per_source, replace=False))
        selected.append(chosen)
    return np.sort(np.concatenate(selected)).astype(np.int64)


def semantic_label(prompt: str) -> str:
    value = str(prompt)
    value = value.removeprefix("Task: ")
    marker = " Phase: "
    if marker in value:
        value = value.split(marker, maxsplit=1)[0]
    return value.rstrip(".")


def build_common_audit_manifest(
    a: SplitRows,
    b: SplitRows,
    *,
    protocol_seed: int,
    per_source: int,
    horizon: int = ACTION_HORIZON,
) -> dict[str, Any]:
    common = common_anchor_mask(a, b, horizon=horizon)
    anchors: list[dict[str, Any]] = []
    candidate_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    for context_id, context in enumerate(CONTEXT_NAMES):
        candidate_counts[context] = int(np.count_nonzero(common & (a.context_id == context_id)))
        chosen = sample_source_balanced(
            a,
            common,
            context_id=context_id,
            per_source=per_source,
            seed=protocol_seed,
            allow_exhausted=context == "wait",
        )
        selected_counts[context] = len(chosen)
        for chosen_row in chosen:
            row = int(chosen_row)
            anchors.append(
                {
                    "anchor_id": f"s{int(a.source_id[row]):04d}_f{int(a.raw_frame[row]):06d}",
                    "source_episode_id": int(a.source_id[row]),
                    "raw_global_frame_index": int(a.raw_frame[row]),
                    "row_a": row,
                    "row_bc": row,
                    "context": context,
                    "context_id": context_id,
                    "fsm_state_id": int(a.fsm_state_id[row]),
                    "prompt_a": str(a.prompts[row]),
                    "prompt_bc": str(b.prompts[row]),
                    "semantic_label": semantic_label(str(b.prompts[row])),
                    "split": "val",
                    "valid_status": {
                        "a_export_valid": bool(a.valid[row]),
                        "bc_export_valid": bool(b.valid[row]),
                        "a_context_stable_through_horizon": True,
                        "bc_context_stable_through_horizon": True,
                        "prompt_change_within_horizon_allowed": True,
                        "common": True,
                    },
                }
            )
    anchors.sort(key=lambda item: (item["context_id"], item["source_episode_id"], item["raw_global_frame_index"]))
    return {
        "schema_version": 1,
        "protocol": "source-balanced strict common validation anchors",
        "protocol_seed": int(protocol_seed),
        "action_horizon": int(horizon),
        "flow_noise_shape": [int(horizon), MODEL_ACTION_DIM],
        "dataset_versions": {"A": a.dataset_version, "B_C": b.dataset_version},
        "dataset_roots": {"A": str(a.split_root), "B_C": str(b.split_root)},
        "candidate_counts": candidate_counts,
        "selected_counts": selected_counts,
        "source_ids": sorted({int(item["source_episode_id"]) for item in anchors}),
        "num_anchors": len(anchors),
        "anchors": anchors,
    }


def flow_noise(
    protocol_seed: int,
    flow_seed: int,
    source_id: int,
    frame_index: int,
    *,
    horizon: int = ACTION_HORIZON,
    action_dim: int = MODEL_ACTION_DIM,
) -> np.ndarray:
    """Standard-normal flow noise keyed solely by the frozen protocol and anchor."""
    values = (protocol_seed, flow_seed, source_id, frame_index, horizon, action_dim)
    if any(int(value) < 0 for value in values):
        raise ValueError(f"noise key values must be non-negative, got {values}")
    seed = np.random.SeedSequence([int(value) for value in values])
    return np.random.default_rng(seed).standard_normal((horizon, action_dim)).astype(np.float32)


def _validate_chunk(chunk: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(chunk, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != ACTION_DIM:
        raise ValueError(f"{name} must have shape [T,{ACTION_DIM}], got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def null_chunk(state: np.ndarray, previous_action: np.ndarray, horizon: int) -> np.ndarray:
    """Physical no-op: zero base, held grippers, and zero delta on arms/torso."""
    state = np.asarray(state, dtype=np.float32)
    previous_action = np.asarray(previous_action, dtype=np.float32)
    if state.shape != (ACTION_DIM,) or previous_action.shape != (ACTION_DIM,):
        raise ValueError(f"state and previous_action must both be [{ACTION_DIM}]")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    row = np.zeros(ACTION_DIM, dtype=np.float32)
    row[list(DELTA_DIMS)] = state[list(DELTA_DIMS)]
    row[list(GRIPPER_DIMS)] = previous_action[list(GRIPPER_DIMS)]
    return np.repeat(row[None, :], horizon, axis=0)


def group_mae(predicted: np.ndarray, expert: np.ndarray) -> dict[str, float]:
    predicted = _validate_chunk(predicted, name="predicted")
    expert = _validate_chunk(expert, name="expert")
    if predicted.shape != expert.shape:
        raise ValueError(f"predicted/expert shape mismatch: {predicted.shape} != {expert.shape}")
    return {
        group: float(np.mean(np.abs(predicted[:, list(dims)] - expert[:, list(dims)])))
        for group, dims in ACTION_GROUPS.items()
    }


def arm_motion_norm(chunk: np.ndarray, state: np.ndarray) -> float:
    chunk = _validate_chunk(chunk, name="chunk")
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (ACTION_DIM,):
        raise ValueError(f"state must have shape [{ACTION_DIM}], got {state.shape}")
    return float(np.mean(np.abs(chunk[:, list(DELTA_DIMS)] - state[list(DELTA_DIMS)])))


def _flip_events(sign_rows: np.ndarray, previous_sign: np.ndarray) -> list[tuple[int, int, float]]:
    previous = np.vstack([previous_sign[None, :], sign_rows[:-1]])
    return [
        (int(frame), int(gripper), float(sign_rows[frame, gripper]))
        for frame, gripper in np.argwhere(sign_rows != previous)
    ]


def gripper_stats(
    predicted: np.ndarray,
    expert: np.ndarray,
    previous_action: np.ndarray,
    *,
    tolerance: int = 8,
) -> dict[str, float | int | None]:
    predicted = _validate_chunk(predicted, name="predicted")
    expert = _validate_chunk(expert, name="expert")
    if predicted.shape != expert.shape:
        raise ValueError(f"predicted/expert shape mismatch: {predicted.shape} != {expert.shape}")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    previous = np.asarray(previous_action, dtype=np.float32)
    if previous.shape != (ACTION_DIM,):
        raise ValueError(f"previous_action must have shape [{ACTION_DIM}]")
    pred_sign = np.sign(predicted[:, list(GRIPPER_DIMS)])
    expert_sign = np.sign(expert[:, list(GRIPPER_DIMS)])
    previous_sign = np.sign(previous[list(GRIPPER_DIMS)])
    expert_events = _flip_events(expert_sign, previous_sign)
    predicted_events = _flip_events(pred_sign, previous_sign)
    unmatched = list(expert_events)
    true_positives = 0
    for frame, gripper, sign in predicted_events:
        for index, (expert_frame, expert_gripper, expert_sign_value) in enumerate(unmatched):
            if (
                gripper == expert_gripper
                and sign == expert_sign_value
                and abs(frame - expert_frame) <= tolerance
            ):
                true_positives += 1
                unmatched.pop(index)
                break
    precision = true_positives / len(predicted_events) if predicted_events else None
    recall = true_positives / len(expert_events) if expert_events else None
    f1 = 2 * true_positives / (len(predicted_events) + len(expert_events)) if expert_events else None
    return {
        "sign_accuracy": float(np.mean(pred_sign == expert_sign)),
        "expert_events": len(expert_events),
        "predicted_events": len(predicted_events),
        "true_positives": true_positives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tolerance_frames": tolerance,
    }


def score_anchor(
    predicted: np.ndarray,
    expert: np.ndarray,
    state: np.ndarray,
    previous_action: np.ndarray,
    *,
    gripper_tolerance: int = 8,
) -> dict[str, Any]:
    predicted = _validate_chunk(predicted, name="predicted")
    expert = _validate_chunk(expert, name="expert")
    null = null_chunk(state, previous_action, len(expert))
    raw = group_mae(predicted, expert)
    null_mae = group_mae(null, expert)
    groups = {
        group: {
            "raw_mae": raw[group],
            "null_mae": null_mae[group],
            "skill": 1.0 - raw[group] / null_mae[group] if null_mae[group] > 0 else None,
        }
        for group in ACTION_GROUPS
    }
    predicted_motion = arm_motion_norm(predicted, state)
    expert_motion = arm_motion_norm(expert, state)
    return {
        "groups": groups,
        "arm_motion_model": predicted_motion,
        "arm_motion_expert": expert_motion,
        "arm_motion_ratio": predicted_motion / expert_motion if expert_motion > 0 else None,
        "gripper": gripper_stats(
            predicted,
            expert,
            previous_action,
            tolerance=gripper_tolerance,
        ),
    }


def aggregate_anchor_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate anchor scores, pooling sparse gripper event counts."""
    if not records:
        raise ValueError("cannot aggregate an empty record list")
    groups: dict[str, dict[str, float | None]] = {}
    for group in ACTION_GROUPS:
        raw = float(np.mean([record["groups"][group]["raw_mae"] for record in records]))
        null = float(np.mean([record["groups"][group]["null_mae"] for record in records]))
        groups[group] = {
            "raw_mae": raw,
            "null_mae": null,
            "skill": 1.0 - raw / null if null > 0 else None,
        }
    model_motion = float(np.mean([record["arm_motion_model"] for record in records]))
    expert_motion = float(np.mean([record["arm_motion_expert"] for record in records]))
    expert_events = int(sum(record["gripper"]["expert_events"] for record in records))
    predicted_events = int(sum(record["gripper"]["predicted_events"] for record in records))
    true_positives = int(sum(record["gripper"]["true_positives"] for record in records))
    precision = true_positives / predicted_events if predicted_events else None
    recall = true_positives / expert_events if expert_events else None
    f1 = 2 * true_positives / (predicted_events + expert_events) if expert_events else None
    return {
        "num_anchors": len(records),
        "groups": groups,
        "arm_motion_model": model_motion,
        "arm_motion_expert": expert_motion,
        "arm_motion_ratio": model_motion / expert_motion if expert_motion > 0 else None,
        "gripper": {
            "sign_accuracy": float(np.mean([record["gripper"]["sign_accuracy"] for record in records])),
            "expert_events": expert_events,
            "predicted_events": predicted_events,
            "true_positives": true_positives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tolerance_frames": int(records[0]["gripper"]["tolerance_frames"]),
        },
    }


def bootstrap_summary(
    values: np.ndarray | list[float],
    *,
    seed: int,
    num_resamples: int = 10_000,
) -> dict[str, float | int | None]:
    """Deterministic scalar summary with a percentile bootstrap CI for the mean."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"mean": None, "median": None, "std": None, "ci95_low": None, "ci95_high": None, "n": 0}
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    if len(array) == 1 or num_resamples <= 0:
        low = high = float(array[0])
    else:
        rng = np.random.default_rng(seed)
        means = np.empty(num_resamples, dtype=np.float64)
        batch = 1_000
        for start in range(0, num_resamples, batch):
            size = min(batch, num_resamples - start)
            sample = rng.choice(array, size=(size, len(array)), replace=True)
            means[start : start + size] = sample.mean(axis=1)
        low, high = (float(value) for value in np.quantile(means, [0.025, 0.975]))
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": std,
        "ci95_low": low,
        "ci95_high": high,
        "n": len(array),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_nonfinite_json(value: Any) -> None:
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"refusing to write non-finite JSON value {value}")
    if isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite_json(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _reject_nonfinite_json(item)


def write_json_atomic(path: Path, value: Any) -> None:
    _reject_nonfinite_json(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
