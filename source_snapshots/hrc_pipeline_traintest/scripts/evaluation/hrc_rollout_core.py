"""Pure helpers shared by HRC causal and policy rollout entry points.

This module deliberately has no OmniGibson dependency so split protection,
state layout, and action safety can be tested in the lightweight test process.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


R1PRO_ACTION_DIM = 23
R1PRO_JOINT_DIM = 28
R1PRO_FINGER_DIM = 4
GRIPPER_ACTION_DIMS = (14, 22)
JOINT_ACTION_TO_ROBOT_INDEX = {
    **dict(zip(range(3, 7), range(6, 10))),
    **dict(zip(range(7, 14), range(10, 24, 2))),
    **dict(zip(range(15, 22), range(11, 24, 2))),
}
DEVELOPMENT_SPLITS = frozenset({"train", "val"})


@dataclass(frozen=True)
class CausalGateCase:
    source_id: int
    task: str
    raw_hdf5: Path
    output_dir: Path


def _load_json(path: Path | str) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def build_causal_gate_cases(
    batch_manifest_path: Path | str,
    pi05_manifest_path: Path | str,
    *,
    split: str,
) -> list[CausalGateCase]:
    """Build pickup/radio causal cases without permitting sealed test access."""

    if split == "test":
        raise ValueError("test split is sealed during development")
    if split not in DEVELOPMENT_SPLITS:
        raise ValueError(f"unsupported development split: {split!r}")

    pi05_manifest = _load_json(pi05_manifest_path)
    try:
        source_ids = sorted({int(value) for value in pi05_manifest["source_splits"][split]})
    except KeyError as exc:
        raise ValueError(f"manifest does not define split {split!r}") from exc

    batch_manifest = _load_json(batch_manifest_path)
    entries = {int(entry["idx"]): entry for entry in batch_manifest.get("episodes", [])}
    cases: list[CausalGateCase] = []
    for source_id in source_ids:
        entry = entries.get(source_id)
        if entry is None or entry.get("passed") is not True:
            raise ValueError(f"expected passed source {source_id} in batch manifest")
        source_output = Path(entry.get("out_dir", Path(batch_manifest_path).parent / f"ep_{source_id:04d}"))
        for task, key in (("pickup", "trash_hdf5"), ("radio", "radio_hdf5")):
            raw_path = entry.get(key)
            if not raw_path:
                raise ValueError(f"passed source {source_id} is missing {key}")
            cases.append(
                CausalGateCase(
                    source_id=source_id,
                    task=task,
                    raw_hdf5=Path(raw_path),
                    output_dir=source_output,
                )
            )
    return cases


def build_state23(
    joint_positions: Sequence[float] | np.ndarray,
    gripper_state: Sequence[float] | np.ndarray,
    previous_action: Sequence[float] | np.ndarray | None,
) -> np.ndarray:
    """Build the deployable state in the exact layout used by π0.5 training."""

    joints = np.asarray(joint_positions, dtype=np.float32)
    fingers = np.asarray(gripper_state, dtype=np.float32)
    if joints.shape != (R1PRO_JOINT_DIM,):
        raise ValueError(f"expected joint_positions shape (28,), got {joints.shape}")
    if fingers.shape != (R1PRO_FINGER_DIM,):
        raise ValueError(f"expected gripper_state shape (4,), got {fingers.shape}")

    if previous_action is None:
        previous_base = np.zeros(3, dtype=np.float32)
    else:
        previous = np.asarray(previous_action, dtype=np.float32)
        if previous.shape != (R1PRO_ACTION_DIM,):
            raise ValueError(f"expected previous_action shape (23,), got {previous.shape}")
        previous_base = previous[:3]

    return np.concatenate(
        [
            previous_base,
            joints[6:10],
            joints[[10, 12, 14, 16, 18, 20, 22]],
            np.asarray([fingers[:2].sum()], dtype=np.float32),
            joints[[11, 13, 15, 17, 19, 21, 23]],
            np.asarray([fingers[2:].sum()], dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)


def sanitize_r1pro_action(action: Sequence[float] | np.ndarray) -> np.ndarray:
    """Validate a physical R1Pro action and enforce deployment safety rules."""

    safe = np.asarray(action, dtype=np.float32)
    if safe.shape != (R1PRO_ACTION_DIM,):
        raise ValueError(f"expected action shape (23,), got {safe.shape}")
    if not np.isfinite(safe).all():
        raise ValueError("policy action contains non-finite values")
    safe = safe.copy()
    safe[:3] = np.clip(safe[:3], -1.0, 1.0)
    for index in GRIPPER_ACTION_DIMS:
        safe[index] = 1.0 if safe[index] >= 0.0 else -1.0
    return safe


def validate_r1pro_joint_targets(
    action: Sequence[float] | np.ndarray,
    joint_lower_limits: Sequence[float] | np.ndarray,
    joint_upper_limits: Sequence[float] | np.ndarray,
) -> None:
    """Reject physical torso/arm targets outside the live robot limits.

    OpenPI's output transform has already converted the trained deltas back to
    absolute joint targets.  Base velocities and binary grippers have separate
    safety rules, so only the 18 torso/arm target dimensions are checked here.
    """

    values = np.asarray(action, dtype=np.float32)
    if values.shape != (R1PRO_ACTION_DIM,) or not np.isfinite(values).all():
        raise ValueError("expected one finite 23-dimensional policy action")
    lower = np.asarray(joint_lower_limits, dtype=np.float32)
    upper = np.asarray(joint_upper_limits, dtype=np.float32)
    if lower.shape != (R1PRO_JOINT_DIM,) or upper.shape != (R1PRO_JOINT_DIM,):
        raise ValueError(
            f"expected joint limits shape (28,), got lower={lower.shape}, upper={upper.shape}"
        )
    if np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower > upper):
        raise ValueError("robot joint limits are invalid")

    tolerance = 1e-5
    for action_index, joint_index in JOINT_ACTION_TO_ROBOT_INDEX.items():
        value = float(values[action_index])
        minimum = float(lower[joint_index])
        maximum = float(upper[joint_index])
        if value < minimum - tolerance or value > maximum + tolerance:
            raise ValueError(
                "policy joint target outside robot limits: "
                f"action dim {action_index}={value:.6g}, robot joint {joint_index} "
                f"range=[{minimum:.6g}, {maximum:.6g}]"
            )


def clamp_r1pro_joint_targets(
    action: Sequence[float] | np.ndarray,
    joint_lower_limits: Sequence[float] | np.ndarray,
    joint_upper_limits: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Saturate torso/arm joint targets to the live robot limits for deployment.

    Same 18 dims as :func:`validate_r1pro_joint_targets`, but clamps instead of
    raising: a marginally out-of-range policy prediction saturates at the limit
    (like a real robot) rather than aborting a long closed-loop rollout. Base
    velocities and binary grippers are handled by :func:`sanitize_r1pro_action`.
    """

    values = np.asarray(action, dtype=np.float32)
    if values.shape != (R1PRO_ACTION_DIM,) or not np.isfinite(values).all():
        raise ValueError("expected one finite 23-dimensional policy action")
    lower = np.asarray(joint_lower_limits, dtype=np.float32)
    upper = np.asarray(joint_upper_limits, dtype=np.float32)
    if lower.shape != (R1PRO_JOINT_DIM,) or upper.shape != (R1PRO_JOINT_DIM,):
        raise ValueError(
            f"expected joint limits shape (28,), got lower={lower.shape}, upper={upper.shape}"
        )
    if np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower > upper):
        raise ValueError("robot joint limits are invalid")

    clamped = values.copy()
    for action_index, joint_index in JOINT_ACTION_TO_ROBOT_INDEX.items():
        clamped[action_index] = float(
            np.clip(values[action_index], lower[joint_index], upper[joint_index])
        )
    return clamped


def navigation_arrived(phase: str, distance_m: float) -> bool:
    """Apply the frozen HRC navigation-arrival definition for one endpoint."""

    distance = float(distance_m)
    if not np.isfinite(distance) or distance < 0.0:
        raise ValueError(f"navigation distance must be finite and non-negative, got {distance_m}")
    if phase == "GoToTask":
        return distance <= 0.75
    if phase == "ReturnToHuman":
        return 1.0 <= distance <= 2.0
    raise ValueError(f"navigation arrival is undefined for phase {phase!r}")


def execute_causal_action_sequence(
    actions: Sequence[Sequence[float]] | np.ndarray,
    *,
    load_initial_state: Callable[[], None],
    step_action: Callable[[int, np.ndarray], object],
    max_steps: int | None = None,
) -> int:
    """Load one initial state and causally advance only through ``step_action``.

    Validation happens before the initial-state callback.  This makes it
    impossible for a malformed action stream to leave the simulator in a
    partially initialized state, and keeps the single ``load_state`` call
    outside the execution loop by construction.
    """

    matrix = np.asarray(actions, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != R1PRO_ACTION_DIM:
        raise ValueError(f"expected actions shape [T, 23], got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("expert actions contain non-finite values")
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    count = len(matrix) if max_steps is None else min(len(matrix), max_steps)
    load_initial_state()
    for index in range(count):
        step_action(index, matrix[index].copy())
    return count
