from __future__ import annotations

import numpy as np
import pytest

from scripts.hrc.verify_dissertation_results import independent_anchor_score


def test_independent_anchor_score_recomputes_skill_motion_and_zero_f1() -> None:
    state = np.zeros(23, dtype=np.float32)
    previous = np.zeros(23, dtype=np.float32)
    previous[[14, 22]] = -1.0
    predicted = np.zeros((4, 23), dtype=np.float32)
    expert = np.zeros_like(predicted)
    predicted[:, [14, 22]] = -1.0
    expert[:, [14, 22]] = -1.0
    predicted[:, 3:7] = 1.0
    expert[:, 3:7] = 2.0
    expert[1:, 14] = 1.0

    result = independent_anchor_score(predicted, expert, state, previous, tolerance=0)

    assert result["arms_raw_mae"] == pytest.approx(4.0 / 18.0)
    assert result["arms_null_mae"] == pytest.approx(8.0 / 18.0)
    assert result["arms_skill"] == pytest.approx(0.5)
    assert result["arm_motion_ratio"] == pytest.approx(0.5)
    assert result["gripper_expert_events"] == 1
    assert result["gripper_predicted_events"] == 0
    assert result["gripper_true_positives"] == 0
    assert result["gripper_f1"] == 0.0
