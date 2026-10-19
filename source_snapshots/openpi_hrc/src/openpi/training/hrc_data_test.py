from pathlib import Path

import numpy as np
import pytest

from openpi.training import hrc_data


def test_split_sources_is_deterministic_and_has_no_leakage():
    source_ids = list(range(49))

    split = hrc_data.split_sources(source_ids, seed=42)

    assert {name: len(ids) for name, ids in split.items()} == {"train": 39, "val": 5, "test": 5}
    assert set(split["train"]).isdisjoint(split["val"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["val"]).isdisjoint(split["test"])
    assert sorted(split["train"] + split["val"] + split["test"]) == source_ids
    assert split == hrc_data.split_sources(source_ids, seed=42)


def test_split_sources_requires_exactly_49_sources():
    with pytest.raises(ValueError, match="exactly 49"):
        hrc_data.split_sources(list(range(48)), seed=42)


def test_build_state23_uses_previous_base_command_and_all_four_trunk_joints():
    joint_positions = np.arange(28, dtype=np.float32)
    gripper_state = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32)
    previous_action = np.arange(23, dtype=np.float32) + 100

    state = hrc_data.build_state23(joint_positions, gripper_state, previous_action)

    expected = np.concatenate(
        [
            previous_action[:3],
            joint_positions[6:10],
            joint_positions[[10, 12, 14, 16, 18, 20, 22]],
            np.array([0.03], dtype=np.float32),
            joint_positions[[11, 13, 15, 17, 19, 21, 23]],
            np.array([0.07], dtype=np.float32),
        ]
    )
    np.testing.assert_array_equal(state, expected)
    assert state.shape == (23,)


def test_build_state23_uses_zero_base_command_for_first_frame():
    state = hrc_data.build_state23(np.arange(28), np.ones(4), previous_action=None)
    np.testing.assert_array_equal(state[:3], np.zeros(3, dtype=np.float32))


def test_build_state28_appends_only_privileged_fields():
    base_state = np.arange(23, dtype=np.float32)
    relative_human_pose = np.array([1.0, -2.0, 0.5], dtype=np.float32)

    state = hrc_data.build_state28(base_state, relative_human_pose, human_visible=1, head_pitch_deg=-25.0)

    np.testing.assert_array_equal(state[:23], base_state)
    np.testing.assert_array_equal(state[23:], np.array([1.0, -2.0, 0.5, 1.0, -25.0]))
    assert state.shape == (28,)


def test_build_prompt_adds_phase_only_when_requested():
    instruction = "Pick up the soda cans and put them in the trash can."

    assert hrc_data.build_prompt(instruction, fsm_state_id=3, include_phase=True) == (
        "Task: Pick up the soda cans and put them in the trash can. Phase: ExecuteBehaviorTask."
    )
    assert hrc_data.build_prompt(instruction, fsm_state_id=3, include_phase=False) == (
        "Pick up the soda cans and put them in the trash can."
    )


def test_follow_prompt_is_explicitly_derived():
    prompt = hrc_data.build_prompt(hrc_data.FOLLOW_INSTRUCTION, fsm_state_id=9, include_phase=True)
    assert prompt == "Task: Follow the person and maintain a safe distance. Phase: Relocate."


def test_valid_anchor_count_for_shift_one_horizon_32():
    anchors = hrc_data.valid_anchor_indices(segment_length=100, action_horizon=32)
    np.testing.assert_array_equal(anchors, np.arange(68, dtype=np.int64))


def test_valid_anchor_rejects_short_segment():
    with pytest.raises(ValueError, match="at least 33"):
        hrc_data.valid_anchor_indices(segment_length=32, action_horizon=32)


def test_task_sample_weights_encode_40_20_40_distribution():
    task_names = ["pickup"] * 80 + ["follow"] * 10 + ["radio"] * 60
    weights = hrc_data.task_sample_weights(
        task_names,
        target_weights={"pickup": 0.4, "follow": 0.2, "radio": 0.4},
    )

    totals = {
        task: float(weights[np.asarray(task_names) == task].sum())
        for task in ("pickup", "follow", "radio")
    }
    assert totals == pytest.approx({"pickup": 0.4, "follow": 0.2, "radio": 0.4})


@pytest.mark.manual
def test_build_manifest_from_final_dataset_has_147_valid_segments():
    data_root = Path("/root/autodl-tmp/hrc_pipeline_traintest/outputs/hrc_batch_final")
    if not data_root.exists():
        pytest.skip("Final HRC dataset is not mounted")

    manifest = hrc_data.build_manifest(data_root, seed=42, action_horizon=32)

    assert manifest["num_passed_sources"] == 49
    assert manifest["num_failed_sources_excluded"] == 16
    assert len(manifest["segments"]) == 147
    assert {split: len(ids) for split, ids in manifest["source_splits"].items()} == {
        "train": 39,
        "val": 5,
        "test": 5,
    }
    assert {task: sum(s["task_name"] == task for s in manifest["segments"]) for task in hrc_data.TASK_NAMES} == {
        "pickup": 49,
        "follow": 49,
        "radio": 49,
    }
    assert all(segment["num_valid_anchors"] > 0 for segment in manifest["segments"])
    assert all(
        segment["prompt_source"] == "fsm_derived"
        for segment in manifest["segments"]
        if segment["task_name"] == "follow"
    )


def test_build_segment_arrays_shifts_actions_and_marks_only_complete_horizons():
    length = 40
    actions = np.arange(length * 23, dtype=np.float32).reshape(length, 23)
    joint_positions = np.arange(length * 28, dtype=np.float32).reshape(length, 28)
    gripper_state = np.full((length, 4), 0.01, dtype=np.float32)
    relative_human_pose = np.arange(length * 3, dtype=np.float32).reshape(length, 3)
    human_visible = np.ones(length, dtype=np.int32)
    head_pitch = np.full(length, -25.0, dtype=np.float32)
    fsm_state_id = np.full(length, 9, dtype=np.int32)

    rows = hrc_data.build_segment_arrays(
        actions=actions,
        joint_positions=joint_positions,
        gripper_state=gripper_state,
        relative_human_pose=relative_human_pose,
        human_visible=human_visible,
        head_pitch_deg=head_pitch,
        fsm_state_id=fsm_state_id,
        start=0,
        end=length,
        instruction=hrc_data.FOLLOW_INSTRUCTION,
        include_phase=True,
        action_horizon=32,
    )

    assert rows["action"].shape == (39, 23)
    np.testing.assert_array_equal(rows["action"][0], actions[1])
    np.testing.assert_array_equal(rows["action"][-1], actions[-1])
    np.testing.assert_array_equal(rows["state23"][0, :3], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(rows["state23"][1, :3], actions[1, :3])
    assert rows["state28"].shape == (39, 28)
    np.testing.assert_array_equal(rows["valid_anchor"], np.array([True] * 8 + [False] * 31))
    assert set(rows["prompt"].tolist()) == {
        "Task: Follow the person and maintain a safe distance. Phase: Relocate."
    }


def test_build_segment_arrays_uses_segment_local_zero_previous_command():
    length = 50
    actions = np.arange(length * 23, dtype=np.float32).reshape(length, 23)
    rows = hrc_data.build_segment_arrays(
        actions=actions,
        joint_positions=np.zeros((length, 28), dtype=np.float32),
        gripper_state=np.zeros((length, 4), dtype=np.float32),
        relative_human_pose=np.zeros((length, 3), dtype=np.float32),
        human_visible=np.zeros(length, dtype=np.int32),
        head_pitch_deg=np.zeros(length, dtype=np.float32),
        fsm_state_id=np.full(length, 3, dtype=np.int32),
        start=5,
        end=45,
        instruction="Turn on the radio.",
        include_phase=False,
        action_horizon=32,
    )

    np.testing.assert_array_equal(rows["state23"][0, :3], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(rows["state23"][1, :3], actions[6, :3])
    assert set(rows["prompt"].tolist()) == {"Turn on the radio."}
