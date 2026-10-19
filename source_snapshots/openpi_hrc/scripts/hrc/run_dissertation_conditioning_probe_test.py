from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.hrc.dissertation_protocol import ACTION_DIM
from scripts.hrc.dissertation_protocol import DELTA_DIMS
from scripts.hrc.dissertation_protocol import SplitRows
from scripts.hrc.run_dissertation_conditioning_probe import assign_donors
from scripts.hrc.run_dissertation_conditioning_probe import choose_segment_positions
from scripts.hrc.run_dissertation_conditioning_probe import comparison_space
from scripts.hrc.run_dissertation_conditioning_probe import safe_ratio


def _rows() -> SplitRows:
    n = 12
    return SplitRows(
        split_root=Path("/dataset/val"),
        dataset_version="fixture",
        state=np.zeros((n, ACTION_DIM), dtype=np.float32),
        action=np.zeros((n, ACTION_DIM), dtype=np.float32),
        valid=np.ones(n, dtype=bool),
        context_id=np.ones(n, dtype=np.int64),
        source_id=np.zeros(n, dtype=np.int64),
        fsm_state_id=np.full(n, 3, dtype=np.int64),
        raw_frame=np.arange(100, 112, dtype=np.int64),
        episode_index=np.zeros(n, dtype=np.int64),
        task_index=np.zeros(n, dtype=np.int64),
        prompts=np.asarray(["Task: Pick up the soda can. Phase: ExecuteBehaviorTask."] * n, dtype=object),
    )


def test_segment_positions_choose_nearest_valid_stable_rows() -> None:
    rows = _rows()
    stable = np.ones(len(rows), dtype=bool)
    stable[[3, 6, 8]] = False

    got = choose_segment_positions(
        rows,
        stable,
        source_id=0,
        context_id=1,
        start=100,
        stop=112,
        label="Pick up the soda can.",
        fractions=(0.25, 0.5, 0.75),
    )

    # Targets are raw frames 103, 106, 109; invalid targets move to the lower
    # equally-near frame so selection is deterministic and hand-checkable.
    assert [(item["position"], item["row"], item["raw_global_frame_index"]) for item in got] == [
        ("early", 2, 102),
        ("middle", 5, 105),
        ("late", 9, 109),
    ]


def test_donors_are_deterministic_and_cross_source() -> None:
    anchors = [
        {"anchor_id": "a", "split": "val", "source_episode_id": 0, "raw_global_frame_index": 10, "row": 10, "semantic_label": "pick", "position": "early"},
        {"anchor_id": "b", "split": "val", "source_episode_id": 1, "raw_global_frame_index": 20, "row": 20, "semantic_label": "pick", "position": "early"},
        {"anchor_id": "c", "split": "val", "source_episode_id": 2, "raw_global_frame_index": 30, "row": 30, "semantic_label": "pick", "position": "early"},
    ]

    first = assign_donors(anchors, seed=9)
    second = assign_donors(anchors, seed=9)

    assert first == second
    source_by_id = {item["anchor_id"]: item["source_episode_id"] for item in anchors}
    assert all(source_by_id[item["anchor_id"]] != source_by_id[item["donor_anchor_id"]] for item in first)


def test_comparison_space_removes_anchor_state_translation_on_delta_dims() -> None:
    state_a = np.zeros(ACTION_DIM, dtype=np.float32)
    state_b = np.zeros(ACTION_DIM, dtype=np.float32)
    state_b[list(DELTA_DIMS)] = 5.0
    delta = np.zeros((2, ACTION_DIM), dtype=np.float32)
    delta[:, list(DELTA_DIMS)] = 0.25
    prediction_a = delta.copy()
    prediction_b = delta.copy()
    prediction_b[:, list(DELTA_DIMS)] += state_b[list(DELTA_DIMS)]

    np.testing.assert_array_equal(
        comparison_space(prediction_a, state_a),
        comparison_space(prediction_b, state_b),
    )


def test_safe_ratio_is_undefined_at_zero_noise_floor() -> None:
    assert safe_ratio(0.2, 0.1) == 2.0
    assert safe_ratio(0.2, 0.0) is None
