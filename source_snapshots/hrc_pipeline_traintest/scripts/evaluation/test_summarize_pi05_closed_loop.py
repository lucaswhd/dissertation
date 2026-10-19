from pathlib import Path
import json

import h5py
import numpy as np
import pytest

from scripts.evaluation.summarize_pi05_closed_loop import aggregate
from scripts.evaluation.summarize_pi05_closed_loop import summarize_rollout


def _write_minimal_rollout(path: Path, action_dim: int) -> None:
    with h5py.File(path, "w") as target:
        data = target.create_group("data")
        demo = data.create_group("demo_0")
        demo.attrs["round_id"] = 0
        demo.attrs["round_success"] = 0
        obs = demo.create_group("obs")
        obs.create_dataset("fsm_state_id", data=np.asarray([1, 2, 3, 5], dtype=np.int32))
        obs.create_dataset(
            "robot_quat",
            data=np.tile(np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (4, 1)),
        )
        obs.create_dataset("relative_human_pose", data=np.zeros((4, 3), dtype=np.float32))
        obs.create_dataset("human_visible", data=np.ones(4, dtype=np.int32))
        demo.create_dataset("action", data=np.zeros((4, action_dim), dtype=np.float32))


def test_wrong_action_dimension_is_reported_without_crashing(tmp_path):
    rollout = tmp_path / "hrc_dataset.hdf5"
    _write_minimal_rollout(rollout, action_dim=22)

    result = summarize_rollout(
        rollout,
        {"run_id": "bad", "model": "e4_5k", "checkpoint": "", "source_id": 0, "seed": 0},
    )

    assert result["protocol_errors"] == ["demo_0: action shape (4, 22)"]
    assert result["gripper"]["right"]["switches"] is None


def _write_follow_rollout(path: Path, human_positions: np.ndarray) -> None:
    frames = len(human_positions)
    with h5py.File(path, "w") as target:
        data = target.create_group("data")
        demo = data.create_group("demo_0")
        demo.attrs["round_id"] = 1
        demo.attrs["round_success"] = 0
        obs = demo.create_group("obs")
        obs.create_dataset("fsm_state_id", data=np.full(frames, 9, dtype=np.int32))
        obs.create_dataset(
            "robot_quat",
            data=np.tile(np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (frames, 1)),
        )
        obs.create_dataset(
            "relative_human_pose",
            data=np.tile(np.asarray([1.5, 0.0, 0.0], dtype=np.float32), (frames, 1)),
        )
        obs.create_dataset("human_visible", data=np.ones(frames, dtype=np.int32))
        obs.create_dataset("human_pos", data=np.asarray(human_positions, dtype=np.float32))
        demo.create_dataset("action", data=np.zeros((frames, 23), dtype=np.float32))


def test_follow_success_requires_actual_route_motion(tmp_path):
    stationary = tmp_path / "stationary.hdf5"
    moving = tmp_path / "moving.hdf5"
    _write_follow_rollout(stationary, np.zeros((40, 3), dtype=np.float32))
    moving_positions = np.zeros((40, 3), dtype=np.float32)
    moving_positions[:, 0] = np.linspace(0.0, 1.0, 40)
    _write_follow_rollout(moving, moving_positions)
    identity = {"run_id": "follow", "model": "e4_5k", "checkpoint": "", "source_id": 0, "seed": 0}

    stationary_result = summarize_rollout(stationary, identity)
    moving_result = summarize_rollout(moving, identity)

    assert stationary_result["follow"]["human_walk_m"] == 0.0
    assert stationary_result["task_success"]["follow"] is False
    assert moving_result["follow"]["human_walk_m"] >= 1.0
    assert moving_result["task_success"]["follow"] is True


def test_aggregate_excludes_protocol_invalid_run_from_model_denominator(tmp_path):
    output_dir = tmp_path / "bad_run"
    output_dir.mkdir()
    rollout = output_dir / "hrc_dataset.hdf5"
    _write_minimal_rollout(rollout, action_dim=22)
    plan = {
        "gate": "gate3",
        "split": "val",
        "sealed_test_access": False,
        "model_groups": [
            {
                "model": "e4_5k",
                "runs": [
                    {
                        "run_id": "bad",
                        "model": "e4_5k",
                        "checkpoint": "",
                        "source_id": 0,
                        "seed": 0,
                        "output_dir": str(output_dir),
                        "expected_hdf5": str(rollout),
                    }
                ],
            }
        ],
    }

    report = aggregate(plan)

    assert report["completed_runs"] == 0
    assert report["per_model"] == {}
    assert report["invalid_runs"][0]["run_id"] == "bad"
    assert "action shape" in report["invalid_runs"][0]["error"]


def test_aggregate_rejects_sealed_test_plan():
    with pytest.raises(ValueError, match="sealed test"):
        aggregate(
            {
                "gate": "gate3",
                "split": "test",
                "sealed_test_access": True,
                "model_groups": [],
            }
        )


def test_navigation_arrival_sidecar_is_aggregated(tmp_path):
    output_dir = tmp_path / "arrival_run"
    output_dir.mkdir()
    rollout = output_dir / "hrc_dataset.hdf5"
    positions = np.zeros((40, 3), dtype=np.float32)
    positions[:, 0] = np.linspace(0.0, 1.0, 40)
    _write_follow_rollout(rollout, positions)
    (output_dir / "policy_rollout_metrics.json").write_text(
        json.dumps(
            {
                "checkpoint": "checkpoint",
                "collision_frames": 0,
                "collision_bodies": [],
                "task_contact_frames": {"pickup": 0, "radio": 0},
                "pickup_lifted": {},
                "pickup_drop_detected": {},
                "navigation_arrival": {
                    "round0_GoToTask": {
                        "phase": "GoToTask", "round_id": 0, "distance_m": 0.5, "arrived": True
                    },
                    "round0_ReturnToHuman": {
                        "phase": "ReturnToHuman", "round_id": 0, "distance_m": 2.2, "arrived": False
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    plan = {
        "gate": "gate3",
        "split": "val",
        "sealed_test_access": False,
        "model_groups": [
            {
                "model": "e4_5k",
                "runs": [
                    {
                        "run_id": "arrival",
                        "model": "e4_5k",
                        "checkpoint": "checkpoint",
                        "source_id": 0,
                        "seed": 0,
                        "output_dir": str(output_dir),
                        "expected_hdf5": str(rollout),
                    }
                ],
            }
        ],
    }

    report = aggregate(plan)

    assert report["runs"][0]["navigation_arrival"]["round0_GoToTask"]["arrived"] is True
    assert report["per_model"]["e4_5k"]["navigation_arrival"]["GoToTask"] == {
        "successes": 1, "attempts": 1, "rate": 1.0
    }
    assert report["per_model"]["e4_5k"]["navigation_arrival"]["ReturnToHuman"] == {
        "successes": 0, "attempts": 1, "rate": 0.0
    }
