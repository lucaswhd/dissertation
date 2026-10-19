from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.hrc.dissertation_protocol import ACTION_DIM
from scripts.hrc.dissertation_protocol import SplitRows
from scripts.hrc.dissertation_protocol import arm_motion_norm
from scripts.hrc.dissertation_protocol import common_anchor_mask
from scripts.hrc.dissertation_protocol import flow_noise
from scripts.hrc.dissertation_protocol import gripper_stats
from scripts.hrc.dissertation_protocol import null_chunk
from scripts.hrc.dissertation_protocol import sample_source_balanced
from scripts.hrc.dissertation_protocol import score_anchor


def _rows(
    *,
    prompts: list[str],
    sources: list[int] | None = None,
    frames: list[int] | None = None,
    contexts: list[int] | None = None,
) -> SplitRows:
    n = len(prompts)
    sources = sources or [0] * n
    frames = frames or list(range(n))
    contexts = contexts or [1] * n
    return SplitRows(
        split_root=Path("/dataset/val"),
        dataset_version="fixture",
        state=np.zeros((n, ACTION_DIM), dtype=np.float32),
        action=np.zeros((n, ACTION_DIM), dtype=np.float32),
        valid=np.ones(n, dtype=bool),
        context_id=np.asarray(contexts, dtype=np.int64),
        source_id=np.asarray(sources, dtype=np.int64),
        fsm_state_id=np.full(n, 3, dtype=np.int64),
        raw_frame=np.asarray(frames, dtype=np.int64),
        episode_index=np.asarray(sources, dtype=np.int64),
        task_index=np.zeros(n, dtype=np.int64),
        prompts=np.asarray(prompts, dtype=object),
    )


def test_common_anchor_mask_allows_subinstructions_but_rejects_context_and_source_boundaries() -> None:
    sources = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    frames = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    contexts = [1, 1, 2, 2, 2, 1, 1, 1, 1, 1]
    a = _rows(prompts=["a"] * 10, sources=sources, frames=frames, contexts=contexts)
    b = _rows(
        prompts=["b0", "b0", "b1", "b1", "b1", "b2", "b2", "b2", "b2", "b2"],
        sources=sources,
        frames=frames,
        contexts=contexts,
    )

    got = common_anchor_mask(a, b, horizon=2)

    # Rows 0/1 cross context 1->2; rows 3/4 and 8/9 run beyond their source.
    # Prompt changes alone are allowed because B's gripper decisions define its
    # subinstruction boundaries inside a single task context.
    assert np.flatnonzero(got).tolist() == [2, 5, 6, 7]


def test_common_anchor_mask_fails_on_misaligned_physical_rows() -> None:
    a = _rows(prompts=["a"] * 4)
    b = _rows(prompts=["b"] * 4)
    b.action[1, 7] = 0.25

    with pytest.raises(ValueError, match="action rows differ"):
        common_anchor_mask(a, b, horizon=1)


def test_source_balanced_sampling_is_deterministic_and_enforces_quotas() -> None:
    rows = _rows(
        prompts=["x"] * 12,
        sources=[0] * 6 + [1] * 6,
        frames=list(range(6)) * 2,
        contexts=[1] * 12,
    )
    mask = np.ones(12, dtype=bool)

    first = sample_source_balanced(rows, mask, context_id=1, per_source=3, seed=17)
    second = sample_source_balanced(rows, mask, context_id=1, per_source=3, seed=17)

    assert first.tolist() == second.tolist()
    assert len(first) == 6
    assert np.count_nonzero(rows.source_id[first] == 0) == 3
    assert np.count_nonzero(rows.source_id[first] == 1) == 3


def test_flow_noise_is_keyed_by_anchor_and_seed() -> None:
    a = flow_noise(20260907, 3, 15, 1234, horizon=4, action_dim=5)
    same = flow_noise(20260907, 3, 15, 1234, horizon=4, action_dim=5)
    other_seed = flow_noise(20260907, 4, 15, 1234, horizon=4, action_dim=5)
    other_anchor = flow_noise(20260907, 3, 15, 1235, horizon=4, action_dim=5)

    assert a.shape == (4, 5)
    assert a.dtype == np.float32
    np.testing.assert_array_equal(a, same)
    assert not np.array_equal(a, other_seed)
    assert not np.array_equal(a, other_anchor)


def test_null_chunk_holds_pose_and_grippers_but_zeros_base() -> None:
    state = np.arange(ACTION_DIM, dtype=np.float32)
    previous = np.full(ACTION_DIM, -9.0, dtype=np.float32)

    got = null_chunk(state, previous, horizon=2)

    np.testing.assert_array_equal(got[:, :3], np.zeros((2, 3), dtype=np.float32))
    np.testing.assert_array_equal(got[:, 14], np.full(2, -9.0, dtype=np.float32))
    np.testing.assert_array_equal(got[:, 22], np.full(2, -9.0, dtype=np.float32))
    np.testing.assert_array_equal(got[:, 3:7], np.tile(state[3:7], (2, 1)))
    np.testing.assert_array_equal(got[:, 7:14], np.tile(state[7:14], (2, 1)))
    np.testing.assert_array_equal(got[:, 15:22], np.tile(state[15:22], (2, 1)))


def test_gripper_event_matching_reports_tp_fp_and_f1() -> None:
    predicted = np.zeros((4, ACTION_DIM), dtype=np.float32)
    expert = np.zeros_like(predicted)
    previous = np.zeros(ACTION_DIM, dtype=np.float32)
    previous[[14, 22]] = -1.0
    predicted[:, [14, 22]] = -1.0
    expert[:, [14, 22]] = -1.0
    expert[1:, 14] = 1.0
    predicted[2:, 14] = 1.0
    predicted[3:, 22] = 1.0

    got = gripper_stats(predicted, expert, previous, tolerance=1)

    assert got["expert_events"] == 1
    assert got["predicted_events"] == 2
    assert got["true_positives"] == 1
    assert got["precision"] == pytest.approx(0.5)
    assert got["recall"] == pytest.approx(1.0)
    assert got["f1"] == pytest.approx(2.0 / 3.0)


def test_gripper_f1_is_zero_when_expert_event_has_no_true_positive() -> None:
    predicted = np.zeros((4, ACTION_DIM), dtype=np.float32)
    expert = np.zeros_like(predicted)
    previous = np.zeros(ACTION_DIM, dtype=np.float32)
    previous[[14, 22]] = -1.0
    predicted[:, [14, 22]] = -1.0
    expert[:, [14, 22]] = -1.0
    expert[1:, 14] = 1.0

    got = gripper_stats(predicted, expert, previous, tolerance=0)

    assert got["expert_events"] == 1
    assert got["predicted_events"] == 0
    assert got["true_positives"] == 0
    assert got["precision"] is None
    assert got["recall"] == 0.0
    assert got["f1"] == 0.0


def test_score_anchor_uses_null_skill_and_delta_motion() -> None:
    state = np.zeros(ACTION_DIM, dtype=np.float32)
    previous = np.zeros(ACTION_DIM, dtype=np.float32)
    expert = np.zeros((2, ACTION_DIM), dtype=np.float32)
    expert[:, 3:7] = 2.0
    predicted = np.zeros_like(expert)
    predicted[:, 3:7] = 1.0

    got = score_anchor(predicted, expert, state, previous)

    assert got["groups"]["arms"]["raw_mae"] == pytest.approx(4.0 / 18.0)
    assert got["groups"]["arms"]["null_mae"] == pytest.approx(8.0 / 18.0)
    assert got["groups"]["arms"]["skill"] == pytest.approx(0.5)
    assert arm_motion_norm(predicted, state) == pytest.approx(4.0 / 18.0)
    assert got["arm_motion_ratio"] == pytest.approx(0.5)
