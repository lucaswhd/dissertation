# ruff: noqa: SLF001

import collections
import types

import numpy as np

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training.hrc_dataset import HrcTaskSampler
from openpi.training.hrc_dataset import HrcValidAnchorDataset
from openpi.training.hrc_dataset import _uniform_time_positions
from openpi.training.hrc_dataset import select_diag_subset_positions


class _FakeDataset:
    def __init__(self, size: int):
        self.values = list(range(size))

    def __getitem__(self, index):
        return self.values[index]

    def __len__(self):
        return len(self.values)


def test_valid_anchor_dataset_exposes_only_valid_rows():
    dataset = HrcValidAnchorDataset(
        _FakeDataset(6),
        valid_mask=np.array([True, False, True, False, False, True]),
        task_ids=np.array([0, 0, 1, 1, 2, 2]),
    )

    assert len(dataset) == 3
    assert [dataset[i] for i in range(len(dataset))] == [0, 2, 5]
    np.testing.assert_array_equal(dataset.raw_indices, [0, 2, 5])
    np.testing.assert_array_equal(dataset.task_ids, [0, 1, 2])


def test_valid_anchor_dataset_rejects_mismatched_metadata():
    with np.testing.assert_raises_regex(ValueError, "same length"):
        HrcValidAnchorDataset(
            _FakeDataset(3),
            valid_mask=np.array([True, False]),
            task_ids=np.array([0, 1, 2]),
        )


def test_fixed_overfit_subset_is_deterministic_and_contains_all_tasks():
    task_ids = np.array([0] * 200 + [1] * 50 + [2] * 100)
    dataset = HrcValidAnchorDataset(
        _FakeDataset(len(task_ids)),
        valid_mask=np.ones(len(task_ids), dtype=bool),
        task_ids=task_ids,
        max_valid_anchors=128,
    )
    counts = np.bincount(dataset.task_ids, minlength=3)
    assert len(dataset) == 128
    assert counts.sum() == 128
    assert counts.min() >= 42
    np.testing.assert_array_equal(dataset.raw_indices, np.sort(dataset.raw_indices))


def test_task_sampler_matches_40_20_40_with_imbalanced_rows():
    task_ids = np.array([0] * 100 + [1] * 10 + [2] * 50)
    sampler = HrcTaskSampler(
        task_ids,
        task_weights=(0.4, 0.2, 0.4),
        num_samples=20_000,
        seed=42,
    )

    counts = collections.Counter(int(task_ids[index]) for index in sampler)
    frequencies = np.array([counts[i] / 20_000 for i in range(3)])
    np.testing.assert_allclose(frequencies, [0.4, 0.2, 0.4], atol=0.02)


def test_task_sampler_rejects_missing_positive_weight_task():
    with np.testing.assert_raises_regex(ValueError, "task 1"):
        HrcTaskSampler(
            np.array([0, 0, 2]),
            task_weights=(0.4, 0.2, 0.4),
            num_samples=10,
            seed=42,
        )


class _FakeHfDataset(dict):
    pass


class _FakeLeRobotDataset(_FakeDataset):
    def __init__(self, *, root, delta_timestamps, repo_id):
        del root, delta_timestamps, repo_id
        super().__init__(8)
        self.hf_dataset = _FakeHfDataset(
            {
                "hrc.valid_anchor": [True, False, True, True, False, True, True, True],
                "hrc.task_id": [0, 0, 1, 1, 1, 2, 2, 2],
            }
        )


def test_create_b1k_dataset_wraps_hrc_valid_anchors(monkeypatch):
    monkeypatch.setattr(
        _data_loader._lerobot_compat,
        "LeRobotDatasetMetadata",
        lambda **kwargs: types.SimpleNamespace(fps=30),
    )
    config = _config.HrcDataConfig(
        repo_id="hrc/test",
        dataset_root="/unused",
        data_cls=_FakeLeRobotDataset,
        action_sequence_keys=("action",),
        task_weights=(0.4, 0.2, 0.4),
    )

    dataset = _data_loader.create_b1k_dataset(config, action_horizon=32)

    assert isinstance(dataset, HrcValidAnchorDataset)
    assert len(dataset) == 6
    np.testing.assert_array_equal(dataset.task_ids, [0, 1, 1, 2, 2, 2])


def _diag_fixture(*, n_sources: int = 14, per_bucket: int = 20, n_context: int = 4):
    """Flat timeline: every source contributes ``per_bucket`` anchors per context,
    contexts in time order 0..3, sources concatenated."""
    task_ids: list[int] = []
    source_ids: list[int] = []
    for source in range(n_sources):
        for context in range(n_context):
            task_ids.extend([context] * per_bucket)
            source_ids.extend([source] * per_bucket)
    return np.array(task_ids, dtype=np.int64), np.array(source_ids, dtype=np.int64)


def test_uniform_time_positions_spreads_instead_of_prefix():
    np.testing.assert_array_equal(_uniform_time_positions(20, 3), [0, 6, 13])
    np.testing.assert_array_equal(_uniform_time_positions(20, 2), [0, 10])
    # count >= available returns every offset
    np.testing.assert_array_equal(_uniform_time_positions(3, 5), [0, 1, 2])


def test_diag_subset_stratifies_over_task_source_and_time():
    task_ids, source_ids = _diag_fixture()
    dataset = HrcValidAnchorDataset(
        _FakeDataset(len(task_ids)),
        valid_mask=np.ones(len(task_ids), dtype=bool),
        task_ids=task_ids,
        source_ids=source_ids,
        max_valid_anchors=128,
        diag_subset_seed=42,
    )

    assert len(dataset) == 128
    np.testing.assert_array_equal(np.bincount(dataset.task_ids, minlength=4), [32, 32, 32, 32])
    # Each context's 32 picks cover all 14 sources (round-robin), never clustered.
    for context in range(4):
        sources = np.unique(dataset.source_ids[dataset.task_ids == context])
        assert len(sources) == 14
    np.testing.assert_array_equal(dataset.raw_indices, np.sort(dataset.raw_indices))


def test_diag_subset_differs_from_first_n_and_is_deterministic():
    task_ids, source_ids = _diag_fixture()
    common = {
        "valid_mask": np.ones(len(task_ids), dtype=bool),
        "task_ids": task_ids,
        "max_valid_anchors": 128,
    }
    strat = HrcValidAnchorDataset(
        _FakeDataset(len(task_ids)), source_ids=source_ids, diag_subset_seed=42, **common
    )
    strat_again = HrcValidAnchorDataset(
        _FakeDataset(len(task_ids)), source_ids=source_ids, diag_subset_seed=42, **common
    )
    legacy = HrcValidAnchorDataset(_FakeDataset(len(task_ids)), **common)  # first-N per task

    # Same seed is reproducible.
    np.testing.assert_array_equal(strat.raw_indices, strat_again.raw_indices)
    # Legacy first-N stays inside the earliest ~2 sources for a context; stratified spans all 14.
    legacy_sources = np.unique(source_ids[legacy.raw_indices][legacy.task_ids == 0])
    assert len(legacy_sources) <= 2
    assert not np.array_equal(strat.raw_indices, legacy.raw_indices)


def test_diag_subset_seed_requires_source_ids():
    task_ids = np.array([0, 1, 2, 3] * 40, dtype=np.int64)
    with np.testing.assert_raises_regex(ValueError, "source_ids"):
        HrcValidAnchorDataset(
            _FakeDataset(len(task_ids)),
            valid_mask=np.ones(len(task_ids), dtype=bool),
            task_ids=task_ids,
            max_valid_anchors=128,
            diag_subset_seed=42,
        )


def test_select_diag_subset_tops_up_when_a_context_is_thin():
    # context 3 only has 5 anchors but its quota is 32 -> selector must still
    # return exactly 128 by topping up from the remaining timeline.
    task_ids = np.array([0] * 200 + [1] * 200 + [2] * 200 + [3] * 5, dtype=np.int64)
    source_ids = np.zeros_like(task_ids)
    positions = select_diag_subset_positions(
        task_ids, source_ids, max_valid_anchors=128, seed=42
    )
    assert len(positions) == 128
    assert len(set(positions)) == 128
