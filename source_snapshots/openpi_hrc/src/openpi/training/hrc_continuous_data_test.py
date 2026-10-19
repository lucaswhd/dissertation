import json

import numpy as np

from openpi.training.hrc_continuous_data import CONTEXT_NAMES
from openpi.training.hrc_continuous_data import FOLLOW_INSTRUCTION
from openpi.training.hrc_continuous_data import WAIT_INSTRUCTION
from openpi.training.hrc_continuous_data import build_context_ids
from openpi.training.hrc_continuous_data import build_context_valid_anchor_mask
from openpi.training.hrc_continuous_data import build_continuous_rows
from openpi.training.hrc_continuous_data import select_clean_sources


def _toy_demo(length: int, *, seed: int, fsm: int | None = None):
    rng = np.random.default_rng(seed)
    return {
        "action": rng.standard_normal((length, 23)).astype(np.float32),
        "joint_positions": rng.standard_normal((length, 28)).astype(np.float32),
        "gripper_state": rng.standard_normal((length, 4)).astype(np.float32),
        "fsm_state_id": (
            np.full(length, fsm, dtype=np.int64)
            if fsm is not None
            else rng.integers(0, 10, size=length, dtype=np.int64)
        ),
    }


def _toy_commands(round0_issued: int, round1_issued: int):
    return {
        0: {"round_id": 0, "instruction": "Pick up the soda cans.", "issued_frame": round0_issued},
        1: {"round_id": 1, "instruction": "Turn on the radio.", "issued_frame": round1_issued},
    }


def test_context_ids_partitions_the_merged_timeline():
    context = build_context_ids(
        demo0_length=50, demo1_length=60, round0_issued_global=10, round1_issued_global=70
    )
    assert len(context) == 110
    np.testing.assert_array_equal(np.unique(context[:10]), [0])  # wait
    np.testing.assert_array_equal(np.unique(context[10:50]), [1])  # pickup
    np.testing.assert_array_equal(np.unique(context[50:70]), [2])  # follow
    np.testing.assert_array_equal(np.unique(context[70:]), [3])  # radio


def test_context_ids_rejects_out_of_range_commands():
    for bad in (
        {"round0_issued_global": 50, "round1_issued_global": 70},  # round0 == demo0_length
        {"round0_issued_global": 10, "round1_issued_global": 50},  # round1 == demo0_length (follow empty)
        {"round0_issued_global": 10, "round1_issued_global": 110},  # round1 == total (radio empty)
    ):
        try:
            build_context_ids(demo0_length=50, demo1_length=60, **bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_context_mask_does_not_cross_new_instruction():
    context = np.array([0] * 45 + [1] * 100 + [2] * 50 + [3] * 100)
    mask = build_context_valid_anchor_mask(context, action_horizon=32)
    for anchor in np.flatnonzero(mask):
        assert np.all(context[anchor : anchor + 33] == context[anchor])


def test_context_mask_excludes_the_sentinel_tail():
    context = np.zeros(100, dtype=np.int64)
    mask = build_context_valid_anchor_mask(context, action_horizon=32)
    # single context: anchors 0..67 valid (t + 32 <= 99), rest excluded.
    assert mask[:68].all()
    assert not mask[68:].any()


def test_continuous_rows_shift_sentinel_and_length():
    demo0 = _toy_demo(50, seed=1)
    demo1 = _toy_demo(60, seed=2)
    commands = _toy_commands(10, 70)
    rows = build_continuous_rows(demo0=demo0, demo1=demo1, commands=commands, action_horizon=8)

    concatenated = np.concatenate([demo0["action"], demo1["action"]])
    assert len(rows["action"]) == 110
    np.testing.assert_array_equal(rows["action"][:-1], concatenated[1:])
    np.testing.assert_array_equal(rows["action"][-1], concatenated[-1])  # sentinel repeats last raw action
    assert rows["valid_anchor"][-1] == np.False_


def test_continuous_rows_state_prev_base_is_producing_action():
    demo0 = _toy_demo(50, seed=3)
    demo1 = _toy_demo(60, seed=4)
    rows = build_continuous_rows(
        demo0=demo0, demo1=demo1, commands=_toy_commands(10, 70), action_horizon=8
    )
    # Row 0 prev-base zeroed; demo_1 first frame (global 50) uses its own producing action.
    np.testing.assert_array_equal(rows["state"][0, :3], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(rows["state"][50, :3], demo1["action"][0, :3])


def test_continuous_rows_prompts_and_anchors_respect_context():
    demo0 = _toy_demo(50, seed=5)
    demo1 = _toy_demo(60, seed=6, fsm=9)
    commands = _toy_commands(10, 70)
    rows = build_continuous_rows(demo0=demo0, demo1=demo1, commands=commands, action_horizon=8)

    context = rows["context_id"]
    # Prompt instruction is chosen by context.
    assert WAIT_INSTRUCTION in rows["prompt"][0]
    assert "Pick up the soda cans." in rows["prompt"][10]
    assert FOLLOW_INSTRUCTION in rows["prompt"][50]
    assert "Turn on the radio." in rows["prompt"][70]
    # No valid anchor crosses a context boundary.
    for anchor in np.flatnonzero(rows["valid_anchor"]):
        assert np.all(context[anchor : anchor + 9] == context[anchor])


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_select_clean_sources_reuses_base_split(tmp_path):
    qc = _write_json(
        tmp_path / "rescan_qc.json",
        {
            "episodes": [
                {"idx": 1, "old_pass": True, "verdict": "CLEAN"},
                {"idx": 2, "old_pass": True, "verdict": "CLEAN"},
                {"idx": 3, "old_pass": True, "verdict": "CLEAN"},
                {"idx": 4, "old_pass": True, "verdict": "TRIM"},
                {"idx": 5, "old_pass": False, "verdict": "CLEAN"},
            ]
        },
    )
    batch = _write_json(
        tmp_path / "batch_manifest.json",
        {"episodes": [{"idx": i, "passed": True} for i in (1, 2, 3, 4, 5)]},
    )
    base = _write_json(
        tmp_path / "base_manifest.json",
        {"source_splits": {"train": [1, 3, 9], "val": [2, 8], "test": [7]}},
    )

    splits = select_clean_sources(batch, qc, base, expected_clean=(1, 2, 3))
    assert splits == {"train": [1, 3], "val": [2], "test": []}


def test_select_clean_sources_fails_fast_on_qc_drift(tmp_path):
    qc = _write_json(
        tmp_path / "rescan_qc.json",
        {"episodes": [{"idx": i, "old_pass": True, "verdict": "CLEAN"} for i in (1, 2, 3)]},
    )
    batch = _write_json(
        tmp_path / "batch_manifest.json",
        {"episodes": [{"idx": i, "passed": True} for i in (1, 2, 3)]},
    )
    base = _write_json(
        tmp_path / "base_manifest.json",
        {"source_splits": {"train": [1, 3], "val": [2], "test": []}},
    )
    try:
        select_clean_sources(batch, qc, base, expected_clean=(1, 2))  # data has an extra CLEAN 3
    except ValueError:
        return
    raise AssertionError("expected ValueError on CLEAN-set mismatch")


def test_context_names_are_wait_pickup_follow_radio():
    assert CONTEXT_NAMES == ("wait", "pickup", "follow", "radio")
