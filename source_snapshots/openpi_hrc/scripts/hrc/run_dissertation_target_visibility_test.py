from __future__ import annotations

import numpy as np
import pytest

from scripts.hrc.run_dissertation_target_visibility import optimal_target_assignment
from scripts.hrc.run_dissertation_target_visibility import point_in_frustum
from scripts.hrc.run_dissertation_target_visibility import summarize_visibility


def test_point_in_identity_camera_frustum_uses_negative_z_forward() -> None:
    camera_position = np.zeros(3)
    identity_xyzw = np.array([0.0, 0.0, 0.0, 1.0])

    assert point_in_frustum(camera_position, identity_xyzw, np.array([0.5, 0.0, -1.0]), 90.0, 90.0)
    assert not point_in_frustum(camera_position, identity_xyzw, np.array([2.0, 0.0, -1.0]), 90.0, 90.0)
    assert not point_in_frustum(camera_position, identity_xyzw, np.array([0.0, 0.0, 1.0]), 90.0, 90.0)


def test_point_in_rotated_camera_frustum_uses_xyzw_quaternion() -> None:
    camera_position = np.zeros(3)
    # +90 degrees around world Y maps the camera's local -Z direction to world -X.
    half = np.sqrt(0.5)
    quaternion_xyzw = np.array([0.0, half, 0.0, half])

    assert point_in_frustum(camera_position, quaternion_xyzw, np.array([-1.0, 0.0, 0.0]), 60.0, 60.0)
    assert not point_in_frustum(camera_position, quaternion_xyzw, np.array([1.0, 0.0, 0.0]), 60.0, 60.0)


def test_optimal_target_assignment_is_one_to_one_and_reports_margin() -> None:
    scores = np.array(
        [
            [9.0, 1.0, 1.0],
            [1.0, 8.0, 2.0],
            [2.0, 1.0, 7.0],
        ]
    )

    got = optimal_target_assignment(scores, ["can_a", "can_b", "can_c"])

    assert got["assigned_names"] == ["can_a", "can_b", "can_c"]
    assert got["best_score"] == pytest.approx(24.0)
    assert got["second_best_score"] == pytest.approx(12.0)
    assert got["margin"] == pytest.approx(12.0)


def test_visibility_summary_reports_first_frame_time_and_relative_position() -> None:
    frames = np.array([100, 101, 102, 103, 104])
    mask = np.array([False, False, True, True, False])

    got = summarize_visibility(frames, mask, segment_start=100, segment_stop=105, fps=10.0)

    assert got["total_frames"] == 5
    assert got["in_frustum_frames"] == 2
    assert got["not_in_frustum_frames"] == 3
    assert got["in_frustum_ratio"] == pytest.approx(0.4)
    assert got["first_in_frustum_raw_frame"] == 102
    assert got["first_in_frustum_time_s"] == pytest.approx(0.2)
    assert got["first_in_frustum_segment_position"] == pytest.approx(0.5)
