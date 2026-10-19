from collections.abc import Mapping, Sequence
import json
from pathlib import Path

import h5py
import numpy as np

FOLLOW_INSTRUCTION = "Follow the person and maintain a safe distance."
TASK_NAMES = ("pickup", "follow", "radio")

FSM_STATE_NAMES = {
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


def split_sources(source_ids: Sequence[int], *, seed: int = 42) -> dict[str, list[int]]:
    source_ids = sorted(int(source_id) for source_id in source_ids)
    if len(source_ids) != 49 or len(set(source_ids)) != 49:
        raise ValueError("Expected exactly 49 unique passed sources")

    shuffled = np.random.default_rng(seed).permutation(source_ids).tolist()
    return {
        "train": shuffled[:39],
        "val": shuffled[39:44],
        "test": shuffled[44:49],
    }


def build_state23(
    joint_positions: np.ndarray,
    gripper_state: np.ndarray,
    previous_action: np.ndarray | None,
) -> np.ndarray:
    joint_positions = np.asarray(joint_positions, dtype=np.float32)
    gripper_state = np.asarray(gripper_state, dtype=np.float32)
    previous_base = (
        np.zeros(3, dtype=np.float32)
        if previous_action is None
        else np.asarray(previous_action, dtype=np.float32)[:3]
    )
    return np.concatenate(
        [
            previous_base,
            joint_positions[6:10],
            joint_positions[[10, 12, 14, 16, 18, 20, 22]],
            np.asarray([gripper_state[:2].sum()], dtype=np.float32),
            joint_positions[[11, 13, 15, 17, 19, 21, 23]],
            np.asarray([gripper_state[2:4].sum()], dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)


def build_state28(
    state23: np.ndarray,
    relative_human_pose: np.ndarray,
    *,
    human_visible: int | bool,
    head_pitch_deg: float,
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(state23, dtype=np.float32),
            np.asarray(relative_human_pose, dtype=np.float32),
            np.asarray([float(human_visible), head_pitch_deg], dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)


def build_prompt(instruction: str, *, fsm_state_id: int, include_phase: bool) -> str:
    instruction = instruction.strip()
    if not include_phase:
        return instruction
    try:
        phase = FSM_STATE_NAMES[int(fsm_state_id)]
    except KeyError as exc:
        raise ValueError(f"Unknown FSM state id: {fsm_state_id}") from exc
    return f"Task: {instruction} Phase: {phase}."


def valid_anchor_indices(*, segment_length: int, action_horizon: int) -> np.ndarray:
    minimum_length = action_horizon + 1
    if segment_length < minimum_length:
        raise ValueError(f"Segment must contain at least {minimum_length} frames")
    return np.arange(segment_length - action_horizon, dtype=np.int64)


def task_sample_weights(
    task_names: Sequence[str],
    *,
    target_weights: Mapping[str, float],
) -> np.ndarray:
    task_names = np.asarray(task_names, dtype=object)
    if set(task_names.tolist()) != set(target_weights):
        raise ValueError("Task names and target weight keys must match")
    target_sum = float(sum(target_weights.values()))
    if not np.isclose(target_sum, 1.0):
        raise ValueError(f"Target weights must sum to 1.0, got {target_sum}")

    weights = np.empty(len(task_names), dtype=np.float64)
    for task_name, target_weight in target_weights.items():
        mask = task_names == task_name
        count = int(mask.sum())
        if count == 0:
            raise ValueError(f"Task has no samples: {task_name}")
        weights[mask] = float(target_weight) / count
    return weights


def build_manifest(data_root: Path | str, *, seed: int = 42, action_horizon: int = 32) -> dict:
    data_root = Path(data_root).resolve()
    with (data_root / "batch_manifest.json").open() as stream:
        batch_manifest = json.load(stream)

    entries = batch_manifest["episodes"]
    passed_entries = sorted((entry for entry in entries if entry.get("passed") is True), key=lambda entry: entry["idx"])
    source_splits = split_sources([entry["idx"] for entry in passed_entries], seed=seed)
    split_by_source = {
        source_id: split
        for split, source_ids in source_splits.items()
        for source_id in source_ids
    }

    segments = []
    for entry in passed_entries:
        source_id = int(entry["idx"])
        episode_dir = Path(entry["out_dir"]).resolve()
        expected_dir = data_root / f"ep_{source_id:04d}"
        if episode_dir != expected_dir:
            episode_dir = expected_dir
        hdf5_path = episode_dir / "hrc_dataset.hdf5"
        commands_path = episode_dir / "annotations" / "commands.json"
        video_path = episode_dir / "videos" / "head_rgb.mp4"
        for path in (hdf5_path, commands_path, video_path):
            if not path.exists():
                raise FileNotFoundError(path)

        with commands_path.open() as stream:
            commands = {int(command["round_id"]): command for command in json.load(stream)}
        if set(commands) != {0, 1}:
            raise ValueError(f"Expected rounds 0 and 1 in {commands_path}, got {sorted(commands)}")

        with h5py.File(hdf5_path, "r") as hdf5_file:
            demo0 = hdf5_file["data/demo_0"]
            demo1 = hdf5_file["data/demo_1"]
            demo0_length = int(demo0["action"].shape[0])
            demo1_length = int(demo1["action"].shape[0])
            pickup_start = int(commands[0]["issued_frame"])
            radio_start = int(commands[1]["issued_frame"]) - demo0_length

            if not 0 <= pickup_start < demo0_length:
                raise ValueError(f"Invalid pickup start for source {source_id}: {pickup_start}")
            if not 0 < radio_start < demo1_length:
                raise ValueError(f"Invalid radio start for source {source_id}: {radio_start}")
            follow_fsm = np.asarray(demo1["obs/fsm_state_id"][:radio_start], dtype=np.int32)
            if follow_fsm.size == 0 or not np.all(follow_fsm == 9):
                raise ValueError(f"Follow segment is not entirely FSM=9 for source {source_id}")

        segment_specs = (
            ("pickup", "demo_0", pickup_start, demo0_length, commands[0]["instruction"], str(commands_path)),
            ("follow", "demo_1", 0, radio_start, FOLLOW_INSTRUCTION, "fsm_derived"),
            ("radio", "demo_1", radio_start, demo1_length, commands[1]["instruction"], str(commands_path)),
        )
        for task_name, demo_name, start, end, instruction, prompt_source in segment_specs:
            segment_length = end - start
            anchors = valid_anchor_indices(segment_length=segment_length, action_horizon=action_horizon)
            segments.append(
                {
                    "source_episode_id": source_id,
                    "split": split_by_source[source_id],
                    "task_name": task_name,
                    "demo_name": demo_name,
                    "start_frame": start,
                    "end_frame_exclusive": end,
                    "num_frames": segment_length,
                    "num_valid_anchors": len(anchors),
                    "instruction": instruction,
                    "prompt_source": prompt_source,
                    "hdf5_path": str(hdf5_path),
                    "head_video_path": str(video_path),
                    "commands_path": str(commands_path),
                    "privileged_human_state": False,
                }
            )

    return {
        "schema_version": "hrc_pi05_v0",
        "data_root": str(data_root),
        "seed": seed,
        "action_horizon": action_horizon,
        "action_shift": 1,
        "num_passed_sources": len(passed_entries),
        "num_failed_sources_excluded": len(entries) - len(passed_entries),
        "source_splits": source_splits,
        "segments": segments,
    }


def build_segment_arrays(
    *,
    actions: np.ndarray,
    joint_positions: np.ndarray,
    gripper_state: np.ndarray,
    relative_human_pose: np.ndarray,
    human_visible: np.ndarray,
    head_pitch_deg: np.ndarray,
    fsm_state_id: np.ndarray,
    start: int,
    end: int,
    instruction: str,
    include_phase: bool,
    action_horizon: int,
) -> dict[str, np.ndarray]:
    arrays = {
        "actions": np.asarray(actions, dtype=np.float32),
        "joint_positions": np.asarray(joint_positions, dtype=np.float32),
        "gripper_state": np.asarray(gripper_state, dtype=np.float32),
        "relative_human_pose": np.asarray(relative_human_pose, dtype=np.float32),
        "human_visible": np.asarray(human_visible),
        "head_pitch_deg": np.asarray(head_pitch_deg, dtype=np.float32),
        "fsm_state_id": np.asarray(fsm_state_id, dtype=np.int32),
    }
    total_length = len(arrays["actions"])
    if any(len(array) != total_length for array in arrays.values()):
        raise ValueError("All segment source arrays must have the same length")
    if not 0 <= start < end <= total_length:
        raise ValueError(f"Invalid segment bounds [{start}, {end}) for length {total_length}")

    segment_length = end - start
    valid_anchors = valid_anchor_indices(segment_length=segment_length, action_horizon=action_horizon)
    raw_indices = np.arange(start, end - 1, dtype=np.int64)
    previous_base = arrays["actions"][raw_indices, :3].copy()
    previous_base[0] = 0.0

    joints = arrays["joint_positions"][raw_indices]
    grippers = arrays["gripper_state"][raw_indices]
    state23 = np.concatenate(
        [
            previous_base,
            joints[:, 6:10],
            joints[:, [10, 12, 14, 16, 18, 20, 22]],
            grippers[:, :2].sum(axis=1, keepdims=True),
            joints[:, [11, 13, 15, 17, 19, 21, 23]],
            grippers[:, 2:4].sum(axis=1, keepdims=True),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    state28 = np.concatenate(
        [
            state23,
            arrays["relative_human_pose"][raw_indices],
            arrays["human_visible"][raw_indices, None].astype(np.float32),
            arrays["head_pitch_deg"][raw_indices, None],
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    valid_anchor = np.zeros(len(raw_indices), dtype=np.bool_)
    valid_anchor[: len(valid_anchors)] = True
    prompts = np.asarray(
        [
            build_prompt(instruction, fsm_state_id=int(state_id), include_phase=include_phase)
            for state_id in arrays["fsm_state_id"][raw_indices]
        ],
        dtype=object,
    )
    return {
        "state23": state23,
        "state28": state28,
        "action": arrays["actions"][raw_indices + 1].astype(np.float32, copy=False),
        "fsm_state_id": arrays["fsm_state_id"][raw_indices],
        "raw_frame_index": raw_indices,
        "valid_anchor": valid_anchor,
        "prompt": prompts,
    }
