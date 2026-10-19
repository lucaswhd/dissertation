"""Continuous full-episode HRC data contract for the E5 π0.5 pilot.

Unlike the E4 three-segment layout (one source -> pickup/follow/radio episodes),
here one source becomes ONE continuous LeRobot episode spanning four instruction
contexts (wait / pickup / follow / radio) on the merged demo_0 + demo_1 timeline.

This module is pure/array-level (plus JSON/HDF5 metadata reads) and reuses the
frozen state23 / prompt / FSM conventions from :mod:`openpi.training.hrc_data`.
"""

from collections.abc import Mapping, Sequence
import json
from pathlib import Path

import h5py
import numpy as np

from openpi.training.hrc_data import FOLLOW_INSTRUCTION
from openpi.training.hrc_data import build_prompt

WAIT_INSTRUCTION = "Wait for the human instruction."
CONTEXT_NAMES = ("wait", "pickup", "follow", "radio")
FOLLOW_FSM_STATE_ID = 9  # Relocate: the follow segment must be entirely FSM=9.
# Sources auto-CLEAN (2026-07-18 rescan) but removed by human video review because
# automated collision QC could not fully judge them. Source 36: the robot lifts
# the trash can itself instead of leaving it on the floor -> atypical manipulation,
# failed 2026-07-21 review. See constraint 7 in the plan and the manual-QC file.
MANUAL_EXCLUDED_SOURCES = (36,)
# Population after auto-QC (old_pass && verdict==CLEAN) minus MANUAL_EXCLUDED_SOURCES.
# Frozen so a silent QC drift fails fast instead of changing the experiment set.
FROZEN_CLEAN_SOURCES = (3, 4, 11, 12, 15, 18, 22, 25, 30, 34, 37, 39, 43, 53, 61)
SCHEMA_VERSION = "hrc_pi05_continuous_clean_v0"

# HDF5 obs keys required to build state23 + the FSM phase (matches hrc_export).
_ROW_KEYS = ("action", "joint_positions", "gripper_state", "fsm_state_id")


def _load_json(path: Path | str):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def build_context_ids(
    *,
    demo0_length: int,
    demo1_length: int,
    round0_issued_global: int,
    round1_issued_global: int,
) -> np.ndarray:
    """Per-frame instruction context on the merged demo_0 + demo_1 timeline.

    Context boundaries (global frame indices):
      0 wait   : [0, round0_issued_global)          -- before the first command
      1 pickup : [round0_issued_global, demo0_length)
      2 follow : [demo0_length, round1_issued_global)
      3 radio  : [round1_issued_global, total)

    ``round0`` is issued inside demo_0; ``round1`` is a GLOBAL index (demo_1's
    local radio start is ``round1 - demo0_length``), matching commands.json where
    round 1's ``issued_frame`` already includes demo_0's length.
    """

    demo0_length = int(demo0_length)
    demo1_length = int(demo1_length)
    round0 = int(round0_issued_global)
    round1 = int(round1_issued_global)
    if demo0_length <= 0 or demo1_length <= 0:
        raise ValueError("demo lengths must be positive")
    total = demo0_length + demo1_length
    if not 0 <= round0 < demo0_length:
        raise ValueError(f"round0_issued_global {round0} must lie in [0, {demo0_length})")
    if not demo0_length < round1 < total:
        raise ValueError(f"round1_issued_global {round1} must lie in ({demo0_length}, {total})")

    context = np.empty(total, dtype=np.int64)
    context[:round0] = 0
    context[round0:demo0_length] = 1
    context[demo0_length:round1] = 2
    context[round1:] = 3
    return context


def build_context_valid_anchor_mask(context_ids: np.ndarray, *, action_horizon: int) -> np.ndarray:
    """Anchors whose 32-step chunk stays inside one context and dodges the sentinel.

    Anchor ``t`` is valid iff (1) the context is constant over ``[t, t + H]``
    (the observation plus its ``H`` future action targets), and (2) ``t + H <=
    n - 1`` so every target exists and none is the non-trainable last-frame
    sentinel.
    """

    context_ids = np.asarray(context_ids, dtype=np.int64)
    action_horizon = int(action_horizon)
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    n = len(context_ids)
    mask = np.zeros(n, dtype=bool)
    if n < action_horizon + 1:
        return mask

    # remaining[t] = count of consecutive equal-context frames strictly after t.
    same_as_next = np.zeros(n, dtype=bool)
    same_as_next[:-1] = context_ids[1:] == context_ids[:-1]
    remaining = np.zeros(n, dtype=np.int64)
    for t in range(n - 2, -1, -1):
        remaining[t] = remaining[t + 1] + 1 if same_as_next[t] else 0

    positions = np.arange(n)
    valid = (remaining >= action_horizon) & (positions + action_horizon <= n - 1)
    mask[valid] = True
    return mask


def _validate_demo(demo: Mapping[str, np.ndarray], name: str) -> int:
    for key in _ROW_KEYS:
        if key not in demo:
            raise ValueError(f"{name} is missing required key {key!r}")
    length = len(demo["action"])
    if length == 0:
        raise ValueError(f"{name} has no frames")
    if any(len(demo[key]) != length for key in _ROW_KEYS):
        raise ValueError(f"{name} arrays must all share the action length {length}")
    return length


def build_continuous_rows(
    *,
    demo0: Mapping[str, np.ndarray],
    demo1: Mapping[str, np.ndarray],
    commands: Mapping[int, Mapping[str, object]],
    action_horizon: int,
) -> dict[str, np.ndarray]:
    """Merge demo_0 + demo_1 into one continuous, shift-1, sentinel-terminated episode.

    Row ``t`` holds observation ``t`` and target ``action[t] = raw_action[t + 1]``.
    The final row repeats the last raw action as a non-trainable sentinel so the
    LeRobot frame count equals the full video length. ``state23`` carries the base
    command of the action that produced the current observation (row 0 zeroed);
    the demo boundary is continuous (not re-zeroed).
    """

    demo0_length = _validate_demo(demo0, "demo0")
    demo1_length = _validate_demo(demo1, "demo1")
    if {int(key) for key in commands} != {0, 1}:
        raise ValueError("commands must contain exactly rounds 0 and 1")
    round0 = int(commands[0]["issued_frame"])
    round1 = int(commands[1]["issued_frame"])

    context_ids = build_context_ids(
        demo0_length=demo0_length,
        demo1_length=demo1_length,
        round0_issued_global=round0,
        round1_issued_global=round1,
    )
    valid_mask = build_context_valid_anchor_mask(context_ids, action_horizon=action_horizon)

    total = demo0_length + demo1_length
    merged = {key: np.concatenate([np.asarray(demo0[key]), np.asarray(demo1[key])]) for key in _ROW_KEYS}
    raw_actions = merged["action"].astype(np.float32, copy=False)
    if raw_actions.ndim != 2 or raw_actions.shape[1] != 23:
        raise ValueError(f"expected action arrays of shape [T, 23], got {raw_actions.shape}")

    action_rows = np.empty_like(raw_actions)
    action_rows[:-1] = raw_actions[1:]
    action_rows[-1] = raw_actions[-1]  # sentinel: repeat last raw action, never trainable

    previous_base = raw_actions[:, :3].copy()
    previous_base[0] = 0.0
    joints = merged["joint_positions"].astype(np.float32, copy=False)
    grippers = merged["gripper_state"].astype(np.float32, copy=False)
    if joints.shape[1] != 28 or grippers.shape[1] != 4:
        raise ValueError(f"expected joints[T,28]+grippers[T,4], got {joints.shape}/{grippers.shape}")
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

    fsm_state_id = merged["fsm_state_id"].astype(np.int64, copy=False)
    instruction_by_context = {
        0: WAIT_INSTRUCTION,
        1: str(commands[0]["instruction"]),
        2: FOLLOW_INSTRUCTION,
        3: str(commands[1]["instruction"]),
    }
    prompts = np.array(
        [
            build_prompt(instruction_by_context[int(context)], fsm_state_id=int(state), include_phase=True)
            for context, state in zip(context_ids, fsm_state_id, strict=True)
        ],
        dtype=object,
    )

    raw_demo_id = np.concatenate(
        [np.zeros(demo0_length, dtype=np.int64), np.ones(demo1_length, dtype=np.int64)]
    )
    raw_demo_frame_index = np.concatenate(
        [np.arange(demo0_length, dtype=np.int64), np.arange(demo1_length, dtype=np.int64)]
    )

    valid_anchor = valid_mask.copy()
    valid_anchor[-1] = False  # the sentinel row is never a valid anchor

    return {
        "action": action_rows,
        "state": state23,
        "prompt": prompts,
        "context_id": context_ids,
        "valid_anchor": valid_anchor,
        "fsm_state_id": fsm_state_id,
        "raw_global_frame_index": np.arange(total, dtype=np.int64),
        "raw_demo_id": raw_demo_id,
        "raw_demo_frame_index": raw_demo_frame_index,
    }


def select_clean_sources(
    batch_manifest_path: Path | str,
    qc_report_path: Path | str,
    base_pi05_manifest_path: Path | str,
    *,
    expected_clean: Sequence[int] | None = FROZEN_CLEAN_SOURCES,
    manual_excluded: Sequence[int] = MANUAL_EXCLUDED_SOURCES,
) -> dict[str, list[int]]:
    """Resolve the CLEAN train/val/test split, reusing the frozen E4 source split.

    A source is CLEAN when the rescan QC has ``old_pass`` and ``verdict=="CLEAN"``,
    minus ``manual_excluded`` (auto-CLEAN sources removed by human video review).
    The remaining set must equal ``expected_clean`` (fail fast on QC drift), every
    CLEAN source must be ``passed`` in the batch manifest, and each keeps whatever
    split it had in the base pi05 manifest (no reshuffle). ``expected_clean=None``
    skips the equality check (tests only).
    """

    qc_report = _load_json(qc_report_path)
    excluded = {int(value) for value in manual_excluded}
    clean = sorted(
        int(episode["idx"])
        for episode in qc_report["episodes"]
        if episode.get("old_pass") is True
        and episode.get("verdict") == "CLEAN"
        and int(episode["idx"]) not in excluded
    )
    if expected_clean is not None and clean != sorted(int(value) for value in expected_clean):
        raise ValueError(f"CLEAN set {clean} does not match expected {sorted(expected_clean)}")

    batch_manifest = _load_json(batch_manifest_path)
    passed = {int(entry["idx"]) for entry in batch_manifest["episodes"] if entry.get("passed") is True}
    not_passed = [source for source in clean if source not in passed]
    if not_passed:
        raise ValueError(f"CLEAN sources not marked passed in batch manifest: {not_passed}")

    base_manifest = _load_json(base_pi05_manifest_path)
    try:
        base_splits = base_manifest["source_splits"]
    except KeyError as exc:
        raise ValueError("base pi05 manifest is missing source_splits") from exc
    split_of = {int(source): split for split, sources in base_splits.items() for source in sources}

    result: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for source in clean:
        if source not in split_of:
            raise ValueError(f"CLEAN source {source} is absent from the base manifest split")
        result[split_of[source]].append(source)
    return {split: sorted(sources) for split, sources in result.items()}


def build_continuous_manifest(
    data_root: Path | str,
    qc_report_path: Path | str,
    base_pi05_manifest_path: Path | str,
    *,
    action_horizon: int = 32,
    expected_clean: Sequence[int] | None = FROZEN_CLEAN_SOURCES,
    manual_excluded: Sequence[int] = MANUAL_EXCLUDED_SOURCES,
    schema_version: str = SCHEMA_VERSION,
) -> dict:
    """Build the continuous manifest (one entry per CLEAN source) from real files.

    ``expected_clean`` / ``manual_excluded`` default to the frozen v0 pilot set; a
    later batch passes its own frozen set (see the split-spec JSON written next to
    the dataset) so a QC drift still fails fast instead of silently changing the
    experiment population.
    """

    data_root = Path(data_root).resolve()
    splits = select_clean_sources(
        data_root / "batch_manifest.json",
        qc_report_path,
        base_pi05_manifest_path,
        expected_clean=expected_clean,
        manual_excluded=manual_excluded,
    )
    split_of = {source: split for split, sources in splits.items() for source in sources}

    context_totals = {name: dict.fromkeys(splits, 0) for name in CONTEXT_NAMES}
    sources = []
    for source_id in sorted(split_of):
        split = split_of[source_id]
        episode_dir = data_root / f"ep_{source_id:04d}"
        hdf5_path = episode_dir / "hrc_dataset.hdf5"
        commands_path = episode_dir / "annotations" / "commands.json"
        video_path = episode_dir / "videos" / "head_rgb.mp4"
        for path in (hdf5_path, commands_path, video_path):
            if not path.exists():
                raise FileNotFoundError(path)

        commands = {int(command["round_id"]): command for command in _load_json(commands_path)}
        if set(commands) != {0, 1}:
            raise ValueError(f"Expected rounds 0 and 1 in {commands_path}, got {sorted(commands)}")
        round0 = int(commands[0]["issued_frame"])
        round1 = int(commands[1]["issued_frame"])

        with h5py.File(hdf5_path, "r") as hdf5_file:
            demo0_length = int(hdf5_file["data/demo_0/action"].shape[0])
            demo1_length = int(hdf5_file["data/demo_1/action"].shape[0])
            radio_start = round1 - demo0_length
            follow_fsm = np.asarray(
                hdf5_file["data/demo_1/obs/fsm_state_id"][:radio_start], dtype=np.int32
            )
        if follow_fsm.size == 0 or not np.all(follow_fsm == FOLLOW_FSM_STATE_ID):
            raise ValueError(f"Follow segment is not entirely FSM={FOLLOW_FSM_STATE_ID} for source {source_id}")

        context_ids = build_context_ids(
            demo0_length=demo0_length,
            demo1_length=demo1_length,
            round0_issued_global=round0,
            round1_issued_global=round1,
        )
        valid_mask = build_context_valid_anchor_mask(context_ids, action_horizon=action_horizon)
        per_context = {
            name: int(((context_ids == index) & valid_mask).sum())
            for index, name in enumerate(CONTEXT_NAMES)
        }
        if split in context_totals[CONTEXT_NAMES[0]]:
            for name in CONTEXT_NAMES:
                context_totals[name][split] += per_context[name]

        sources.append(
            {
                "source_episode_id": source_id,
                "split": split,
                "demo0_length": demo0_length,
                "demo1_length": demo1_length,
                "num_frames": demo0_length + demo1_length,
                "round0_issued_global": round0,
                "round1_issued_global": round1,
                "num_valid_anchors": int(valid_mask.sum()),
                "context_valid_anchors": per_context,
                "hdf5_path": str(hdf5_path),
                "head_video_path": str(video_path),
                "commands_path": str(commands_path),
            }
        )

    return {
        "schema_version": schema_version,
        "data_root": str(data_root),
        "action_horizon": action_horizon,
        "action_shift": 1,
        "num_sources": len(sources),
        "source_splits": splits,
        "context_anchor_totals": context_totals,
        "sources": sources,
    }
