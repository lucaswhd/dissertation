import numpy as np

from openpi.training.hrc_norm_stats import RunningMoments
from openpi.training.hrc_norm_stats import RunningQuantiles
from openpi.training.hrc_norm_stats import combine_task_moments
from openpi.training.hrc_norm_stats import compute_hrc_norm_stats_from_arrays
from openpi.training.hrc_norm_stats import quantiles_from_histogram


def test_running_quantiles_match_numpy_percentile():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(20_000, 3)) * [1.0, 5.0, 0.1] + [0.0, -3.0, 7.0]
    histogram = RunningQuantiles.over_range(values.min(axis=0), values.max(axis=0))
    for start in range(0, len(values), 512):
        histogram.update(values[start : start + 512])

    assert histogram.total == len(values)
    q01, q99 = quantiles_from_histogram(histogram.edges, histogram.counts, (0.01, 0.99))
    # 5000 bins over the observed range; tolerance is a few bin widths.
    spread = values.max(axis=0) - values.min(axis=0)
    assert np.allclose(q01, np.percentile(values, 1, axis=0), atol=spread * 1e-3)
    assert np.allclose(q99, np.percentile(values, 99, axis=0), atol=spread * 1e-3)


def test_running_quantiles_handle_a_constant_dimension():
    histogram = RunningQuantiles.over_range([2.0], [2.0])
    histogram.update(np.full((100, 1), 2.0))
    q01, q99 = quantiles_from_histogram(histogram.edges, histogram.counts, (0.01, 0.99))
    assert np.isclose(q01[0], 2.0)
    assert np.isclose(q99[0], 2.0)


def test_quantiles_follow_task_weights_not_row_counts():
    # Task 0 is 100x more numerous but must only carry half the weight, so the
    # pooled 99th percentile has to land in task 1's range, not task 0's.
    grid = (np.array([0.0]), np.array([100.0]))
    histograms = {}
    for task_id, value, rows in ((0, 1.0, 10_000), (1, 99.0, 100)):
        histogram = RunningQuantiles.over_range(*grid)
        histogram.update(np.full((rows, 1), value))
        histograms[task_id] = histogram

    moments = {}
    for task_id, value, rows in ((0, 1.0, 10_000), (1, 99.0, 100)):
        running = RunningMoments()
        running.update(np.full((rows, 1), value))
        moments[task_id] = running

    stats = combine_task_moments(moments, task_weights=(0.5, 0.5), quantiles=histograms)
    assert np.isclose(stats.mean[0], 50.0)
    assert stats.q01[0] < 2.0
    assert stats.q99[0] > 98.0


def test_combine_task_moments_uses_target_distribution_not_row_counts():
    moments = {}
    for task_id, values in {
        0: np.array([[0.0], [2.0]]),
        1: np.array([[10.0], [10.0], [10.0], [10.0]]),
        2: np.array([[-2.0], [2.0]]),
    }.items():
        running = RunningMoments()
        running.update(values)
        moments[task_id] = running

    stats = combine_task_moments(moments, task_weights=(0.4, 0.2, 0.4))
    expected_mean = 0.4 * 1.0 + 0.2 * 10.0 + 0.4 * 0.0
    expected_second_moment = 0.4 * 2.0 + 0.2 * 100.0 + 0.4 * 4.0
    assert np.isclose(stats.mean[0], expected_mean)
    assert np.isclose(stats.std[0] ** 2, expected_second_moment - expected_mean**2)
    assert stats.q01 is None
    assert stats.q99 is None


def test_array_stats_use_only_valid_anchors_and_future_action_chunks():
    states = np.arange(6, dtype=np.float32)[:, None]
    actions = (np.arange(6, dtype=np.float32) * 10)[:, None]
    valid = np.array([True, True, False, True, False, False])
    task_ids = np.array([0, 0, 0, 1, 1, 1])

    stats = compute_hrc_norm_stats_from_arrays(
        states=states,
        actions=actions,
        valid_mask=valid,
        task_ids=task_ids,
        action_horizon=2,
        delta_mappings=(),
        task_weights=None,
    )

    np.testing.assert_allclose(stats["state"].mean, np.mean(states[[0, 1, 3]], axis=0))
    expected_actions = np.array([[0.0], [10.0], [10.0], [20.0], [30.0], [40.0]])
    np.testing.assert_allclose(stats["actions"].mean, np.mean(expected_actions, axis=0))


def test_delta_mapping_subtracts_anchor_state_from_each_future_action():
    states = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    actions = np.array([[2.0], [4.0], [8.0]], dtype=np.float32)
    stats = compute_hrc_norm_stats_from_arrays(
        states=states,
        actions=actions,
        valid_mask=np.array([True, True, False]),
        task_ids=np.array([0, 0, 0]),
        action_horizon=2,
        delta_mappings=(([0], [0]),),
        task_weights=None,
    )
    expected = np.array([[1.0], [3.0], [2.0], [6.0]])
    np.testing.assert_allclose(stats["actions"].mean, expected.mean(axis=0))
