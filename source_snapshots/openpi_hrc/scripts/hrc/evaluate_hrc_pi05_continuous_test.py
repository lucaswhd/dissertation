import numpy as np
import pytest

from scripts.hrc.evaluate_hrc_pi05_continuous import ACTION_GROUPS
from scripts.hrc.evaluate_hrc_pi05_continuous import action_group_mae
from scripts.hrc.evaluate_hrc_pi05_continuous import aggregate_group_mae
from scripts.hrc.evaluate_hrc_pi05_continuous import sample_anchors_per_context


def test_action_groups_partition_all_23_dims():
    covered = sorted(i for idx in ACTION_GROUPS.values() for i in idx)
    assert covered == list(range(23))


def test_action_group_mae_zero_on_identical_and_isolates_groups():
    predicted = np.zeros((32, 23), dtype=np.float32)
    assert all(v == 0.0 for v in action_group_mae(predicted, predicted.copy()).values())

    expert = predicted.copy()
    expert[:, 14] = 2.0  # perturb only the left gripper dim
    mae = action_group_mae(predicted, expert)
    assert mae["left_gripper"] == pytest.approx(2.0)
    assert mae["base"] == 0.0
    assert mae["arms"] == 0.0
    assert mae["right_gripper"] == 0.0


def test_action_group_mae_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="23"):
        action_group_mae(np.zeros((32, 22)), np.zeros((32, 22)))


def test_sample_anchors_caps_per_context_and_is_seeded():
    context = np.array([0] * 10 + [1] * 500 + [2] * 5 + [3] * 400, dtype=np.int64)
    valid = np.ones(len(context), dtype=bool)
    sampled = sample_anchors_per_context(context, valid, max_per_context=300, seed=42)

    assert len(sampled[0]) == 10  # fewer than cap -> all kept
    assert len(sampled[1]) == 300  # capped
    assert len(sampled[2]) == 5
    assert len(sampled[3]) == 300
    # indices stay within their context and are sorted
    assert np.all(context[sampled[1]] == 1)
    assert np.array_equal(sampled[1], np.sort(sampled[1]))
    # deterministic under the same seed
    again = sample_anchors_per_context(context, valid, max_per_context=300, seed=42)
    assert np.array_equal(sampled[1], again[1])


def test_sample_anchors_respects_valid_mask():
    context = np.zeros(100, dtype=np.int64)
    valid = np.zeros(100, dtype=bool)
    valid[[3, 7, 42]] = True
    sampled = sample_anchors_per_context(context, valid, max_per_context=300, seed=0)
    assert np.array_equal(sampled[0], [3, 7, 42])


def test_aggregate_group_mae_averages_chunks():
    chunks = [
        {"base": 1.0, "arms": 2.0, "left_gripper": 0.0, "right_gripper": 0.0},
        {"base": 3.0, "arms": 4.0, "left_gripper": 1.0, "right_gripper": 1.0},
    ]
    agg = aggregate_group_mae(chunks)
    assert agg["base"] == pytest.approx(2.0)
    assert agg["arms"] == pytest.approx(3.0)
    assert np.isnan(aggregate_group_mae([])["base"])
