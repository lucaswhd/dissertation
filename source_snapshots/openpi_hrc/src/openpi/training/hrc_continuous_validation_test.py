import numpy as np
import pytest

from openpi.training.hrc_continuous_data import build_continuous_rows
from openpi.training.hrc_continuous_validation import check_prompt_phase_consistency
from openpi.training.hrc_continuous_validation import validate_continuous_columns


def _toy_demo(length, *, seed, fsm=None):
    rng = np.random.default_rng(seed)
    return {
        "action": rng.standard_normal((length, 23)).astype(np.float32),
        "joint_positions": rng.standard_normal((length, 28)).astype(np.float32),
        "gripper_state": rng.standard_normal((length, 4)).astype(np.float32),
        "fsm_state_id": (
            np.full(length, fsm, dtype=np.int64) if fsm is not None else np.full(length, 2, dtype=np.int64)
        ),
    }


def _episode_columns(rows, *, source_id, episode_index):
    n = len(rows["action"])
    return {
        "observation.state": rows["state"],
        "action": rows["action"],
        "hrc.valid_anchor": rows["valid_anchor"].astype(bool),
        "hrc.task_id": rows["context_id"],
        "hrc.context_id": rows["context_id"],
        "hrc.source_episode_id": np.full(n, source_id, dtype=np.int64),
        "hrc.fsm_state_id": rows["fsm_state_id"],
        "hrc.raw_global_frame_index": rows["raw_global_frame_index"],
        "hrc.raw_demo_id": rows["raw_demo_id"],
        "hrc.raw_demo_frame_index": rows["raw_demo_frame_index"],
        "episode_index": np.full(n, episode_index, dtype=np.int64),
        "prompt": rows["prompt"],
    }


def _two_source_columns(action_horizon=8):
    rows0 = build_continuous_rows(
        demo0=_toy_demo(50, seed=1),
        demo1=_toy_demo(60, seed=2, fsm=9),
        commands={
            0: {"round_id": 0, "instruction": "Pick up cans.", "issued_frame": 10},
            1: {"round_id": 1, "instruction": "Turn on radio.", "issued_frame": 70},
        },
        action_horizon=action_horizon,
    )
    rows1 = build_continuous_rows(
        demo0=_toy_demo(40, seed=3),
        demo1=_toy_demo(55, seed=4, fsm=9),
        commands={
            0: {"round_id": 0, "instruction": "Pick up cans.", "issued_frame": 8},
            1: {"round_id": 1, "instruction": "Turn on radio.", "issued_frame": 60},
        },
        action_horizon=action_horizon,
    )
    c0 = _episode_columns(rows0, source_id=3, episode_index=0)
    c1 = _episode_columns(rows1, source_id=15, episode_index=1)
    return {key: np.concatenate([c0[key], c1[key]]) for key in c0}


def test_valid_continuous_split_passes_and_reports_stats():
    columns = _two_source_columns()
    stats = validate_continuous_columns(columns, expected_sources={3, 15}, action_horizon=8)
    assert stats["num_sources"] == 2
    assert stats["num_episodes"] == 2
    assert stats["num_rows"] == 110 + 95
    assert sum(stats["valid_anchors_by_context"].values()) == stats["num_valid_anchors"]


def test_rejects_wrong_expected_source_set():
    columns = _two_source_columns()
    with pytest.raises(ValueError, match="expected"):
        validate_continuous_columns(columns, expected_sources={3, 99}, action_horizon=8)


def test_rejects_corrupted_valid_anchor_mask():
    columns = _two_source_columns()
    columns["hrc.valid_anchor"] = columns["hrc.valid_anchor"].copy()
    columns["hrc.valid_anchor"][5] = not columns["hrc.valid_anchor"][5]
    with pytest.raises(ValueError, match="valid-anchor mask"):
        validate_continuous_columns(columns, expected_sources={3, 15}, action_horizon=8)


def test_rejects_task_id_context_mismatch():
    columns = _two_source_columns()
    columns["hrc.task_id"] = columns["hrc.task_id"].copy()
    columns["hrc.task_id"][0] = 2  # first row is wait(0)
    with pytest.raises(ValueError, match="task_id must equal"):
        validate_continuous_columns(columns, expected_sources={3, 15}, action_horizon=8)


def test_rejects_broken_prev_base_continuity():
    columns = _two_source_columns()
    columns["observation.state"] = columns["observation.state"].copy()
    columns["observation.state"][3, :3] += 1.0
    with pytest.raises(ValueError, match="prev-base"):
        validate_continuous_columns(columns, expected_sources={3, 15}, action_horizon=8)


def test_prompt_phase_matches_fsm():
    columns = _two_source_columns()
    check_prompt_phase_consistency(columns["prompt"], columns["hrc.fsm_state_id"])  # must not raise

    bad = list(columns["prompt"])
    bad[0] = "Task: Wait for the human instruction. Phase: FaceHuman."  # wrong phase for fsm=2
    with pytest.raises(ValueError, match="prompt phase"):
        check_prompt_phase_consistency(bad, columns["hrc.fsm_state_id"])
