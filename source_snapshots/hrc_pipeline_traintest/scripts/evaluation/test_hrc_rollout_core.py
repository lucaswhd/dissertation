import json

import numpy as np
import pytest

from scripts.evaluation.hrc_rollout_core import build_causal_gate_cases
from scripts.evaluation.hrc_rollout_core import build_state23
from scripts.evaluation.hrc_rollout_core import clamp_r1pro_joint_targets
from scripts.evaluation.hrc_rollout_core import execute_causal_action_sequence
from scripts.evaluation.hrc_rollout_core import navigation_arrived
from scripts.evaluation.hrc_rollout_core import sanitize_r1pro_action
from scripts.evaluation.hrc_rollout_core import validate_r1pro_joint_targets


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_build_causal_gate_cases_uses_only_requested_val_sources(tmp_path):
    data_root = tmp_path / "batch"
    data_root.mkdir()
    batch_manifest = data_root / "batch_manifest.json"
    pi05_manifest = tmp_path / "pi05_manifest.json"
    _write_json(
        batch_manifest,
        {
            "episodes": [
                {
                    "idx": source_id,
                    "passed": True,
                    "trash_hdf5": f"/raw/trash_{source_id}.hdf5",
                    "radio_hdf5": f"/raw/radio_{source_id}.hdf5",
                    "out_dir": str(data_root / f"ep_{source_id:04d}"),
                }
                for source_id in range(4)
            ]
        },
    )
    _write_json(
        pi05_manifest,
        {"source_splits": {"train": [2], "val": [3, 0], "test": [1]}},
    )

    cases = build_causal_gate_cases(
        batch_manifest_path=batch_manifest,
        pi05_manifest_path=pi05_manifest,
        split="val",
    )

    assert [(case.source_id, case.task) for case in cases] == [
        (0, "pickup"),
        (0, "radio"),
        (3, "pickup"),
        (3, "radio"),
    ]
    assert cases[0].raw_hdf5.as_posix() == "/raw/trash_0.hdf5"
    assert cases[1].raw_hdf5.as_posix() == "/raw/radio_0.hdf5"


def test_build_causal_gate_cases_rejects_test_split(tmp_path):
    batch_manifest = tmp_path / "batch.json"
    pi05_manifest = tmp_path / "pi05.json"
    _write_json(batch_manifest, {"episodes": []})
    _write_json(pi05_manifest, {"source_splits": {"train": [], "val": [], "test": [9]}})

    with pytest.raises(ValueError, match="test split is sealed"):
        build_causal_gate_cases(batch_manifest, pi05_manifest, split="test")


def test_build_causal_gate_cases_rejects_failed_or_missing_source(tmp_path):
    batch_manifest = tmp_path / "batch.json"
    pi05_manifest = tmp_path / "pi05.json"
    _write_json(batch_manifest, {"episodes": [{"idx": 7, "passed": False}]})
    _write_json(pi05_manifest, {"source_splits": {"train": [], "val": [7], "test": []}})

    with pytest.raises(ValueError, match="passed source 7"):
        build_causal_gate_cases(batch_manifest, pi05_manifest, split="val")


def test_build_state23_matches_training_layout():
    joints = np.arange(28, dtype=np.float32)
    fingers = np.asarray([0.01, 0.02, 0.03, 0.04], dtype=np.float32)
    previous_action = np.arange(23, dtype=np.float32) / 10.0

    state = build_state23(joints, fingers, previous_action)

    expected = np.concatenate(
        [
            previous_action[:3],
            joints[6:10],
            joints[[10, 12, 14, 16, 18, 20, 22]],
            [0.03],
            joints[[11, 13, 15, 17, 19, 21, 23]],
            [0.07],
        ]
    ).astype(np.float32)
    assert state.shape == (23,)
    np.testing.assert_allclose(state, expected)


def test_build_state23_zeros_previous_base_on_reset():
    state = build_state23(np.zeros(28), np.zeros(4), previous_action=None)
    np.testing.assert_array_equal(state[:3], np.zeros(3, dtype=np.float32))


def test_sanitize_r1pro_action_clips_base_and_binarizes_grippers():
    action = np.linspace(-0.5, 0.5, 23, dtype=np.float32)
    action[:3] = [-2.0, 0.25, 3.0]
    action[14] = -1e-4
    action[22] = 0.0

    safe = sanitize_r1pro_action(action)

    np.testing.assert_allclose(safe[:3], [-1.0, 0.25, 1.0])
    assert safe[14] == -1.0
    assert safe[22] == 1.0
    assert safe.dtype == np.float32


@pytest.mark.parametrize("bad", [np.zeros(22), np.full(23, np.nan), np.full(23, np.inf)])
def test_sanitize_r1pro_action_rejects_wrong_shape_or_nonfinite(bad):
    with pytest.raises(ValueError):
        sanitize_r1pro_action(bad)


def test_validate_r1pro_joint_targets_uses_live_robot_joint_limits():
    action = np.zeros(23, dtype=np.float32)
    lower = np.full(28, -10.0, dtype=np.float32)
    upper = np.full(28, 10.0, dtype=np.float32)
    upper[10] = 1.0  # Simulator joint 10 is left-arm action dimension 7.

    validate_r1pro_joint_targets(action, lower, upper)
    action[7] = 1.01
    with pytest.raises(ValueError, match="action dim 7"):
        validate_r1pro_joint_targets(action, lower, upper)


def test_clamp_r1pro_joint_targets_saturates_instead_of_raising():
    lower = np.full(28, -10.0, dtype=np.float32)
    upper = np.full(28, 10.0, dtype=np.float32)
    upper[10] = 1.0  # robot joint 10 == action dim 7
    lower[10] = -1.0

    action = np.zeros(23, dtype=np.float32)
    action[7] = 1.37  # marginally over the limit, like a noisy policy prediction
    action[3] = 0.5   # in-range dim must pass through unchanged
    clamped = clamp_r1pro_joint_targets(action, lower, upper)

    assert clamped[7] == pytest.approx(1.0)   # saturated at the upper limit
    assert clamped[3] == pytest.approx(0.5)   # untouched
    # clamping does not raise and the result now passes the strict validator.
    validate_r1pro_joint_targets(clamped, lower, upper)


def test_validate_r1pro_joint_targets_rejects_bad_limit_shapes():
    with pytest.raises(ValueError, match="joint limits shape"):
        validate_r1pro_joint_targets(np.zeros(23), np.zeros(27), np.ones(27))


def test_navigation_arrival_uses_phase_specific_distance_rules():
    assert navigation_arrived("GoToTask", 0.75) is True
    assert navigation_arrived("GoToTask", 0.751) is False
    assert navigation_arrived("ReturnToHuman", 1.0) is True
    assert navigation_arrived("ReturnToHuman", 2.0) is True
    assert navigation_arrived("ReturnToHuman", 0.99) is False
    with pytest.raises(ValueError):
        navigation_arrived("Relocate", 1.5)


def test_execute_causal_action_sequence_loads_once_then_steps_in_order():
    actions = np.arange(4 * 23, dtype=np.float32).reshape(4, 23)
    events = []

    executed = execute_causal_action_sequence(
        actions,
        load_initial_state=lambda: events.append("load"),
        step_action=lambda index, action: events.append((index, action.copy())),
    )

    assert executed == 4
    assert events[0] == "load"
    assert sum(event == "load" for event in events if isinstance(event, str)) == 1
    for index, event in enumerate(events[1:]):
        assert event[0] == index
        np.testing.assert_array_equal(event[1], actions[index])


def test_execute_causal_action_sequence_honors_max_steps():
    actions = np.zeros((5, 23), dtype=np.float32)
    stepped = []

    executed = execute_causal_action_sequence(
        actions,
        load_initial_state=lambda: None,
        step_action=lambda index, action: stepped.append(index),
        max_steps=2,
    )

    assert executed == 2
    assert stepped == [0, 1]


def test_execute_causal_action_sequence_validates_before_loading():
    loads = []

    with pytest.raises(ValueError, match="\\[T, 23\\]"):
        execute_causal_action_sequence(
            np.zeros((4, 22), dtype=np.float32),
            load_initial_state=lambda: loads.append(True),
            step_action=lambda index, action: None,
        )

    assert loads == []
