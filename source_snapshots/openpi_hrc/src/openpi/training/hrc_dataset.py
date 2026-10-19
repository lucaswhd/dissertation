from collections.abc import Iterator, Sequence
from typing import SupportsIndex

import numpy as np
import torch


def _uniform_time_positions(available: int, count: int) -> np.ndarray:
    """Pick ``count`` evenly time-spaced offsets out of ``available`` ordered rows.

    ``available`` rows are assumed to already be in timeline order, so evenly
    spaced offsets sample the whole segment instead of clustering at its start.
    For ``count >= available`` every offset is returned. The step ``available /
    count`` is ``>= 1`` when ``count < available``, so the floored offsets are
    strictly increasing and therefore distinct.
    """

    if count >= available:
        return np.arange(available, dtype=np.int64)
    return np.floor(np.arange(count) * (available / count)).astype(np.int64)


def _round_robin_allocation(quota: int, capacities: Sequence[int]) -> list[int]:
    """Spread ``quota`` picks across buckets one at a time, capped per capacity.

    One pass hands every non-full bucket a single slot, so the counts stay within
    one of each other (source balance) until a bucket runs out of anchors.
    """

    allocation = [0] * len(capacities)
    remaining = quota
    while remaining > 0 and any(allocation[i] < capacities[i] for i in range(len(capacities))):
        for i in range(len(capacities)):
            if remaining <= 0:
                break
            if allocation[i] < capacities[i]:
                allocation[i] += 1
                remaining -= 1
    return allocation


def select_diag_subset_positions(
    valid_task_ids: np.ndarray,
    valid_source_ids: np.ndarray,
    *,
    max_valid_anchors: int,
    seed: int,
) -> list[int]:
    """Deterministic stratified diagnostic subset over (task, source, time).

    Returns sorted positions into the ``valid_*`` arrays. ``max_valid_anchors`` is
    split evenly across the present tasks (the first ``remainder`` tasks get one
    extra). Within a task the quota is dealt round-robin across that task's source
    episodes, and within each source the picks are spaced uniformly along time.
    ``seed`` only shuffles the per-task source visitation order so the ``+1`` from
    round-robin is not systematically biased toward the lowest source id; the
    within-source time sampling is RNG-free. Any shortfall from rounding or thin
    sources is topped up in timeline order so the result is exactly the quota.
    """

    valid_task_ids = np.asarray(valid_task_ids, dtype=np.int64)
    valid_source_ids = np.asarray(valid_source_ids, dtype=np.int64)
    if valid_task_ids.shape != valid_source_ids.shape:
        raise ValueError("valid_task_ids and valid_source_ids must have the same shape")
    total = len(valid_task_ids)
    if max_valid_anchors > total:
        raise ValueError("max_valid_anchors exceeds the number of valid anchors")

    unique_tasks = np.unique(valid_task_ids)
    base, remainder = divmod(max_valid_anchors, len(unique_tasks))
    rng = np.random.default_rng(seed)

    selected: list[int] = []
    for task_offset, task_id in enumerate(unique_tasks):
        quota = base + int(task_offset < remainder)
        if quota <= 0:
            continue
        task_positions = np.flatnonzero(valid_task_ids == task_id)  # timeline order
        task_sources = valid_source_ids[task_positions]
        source_order = np.unique(task_sources)
        rng.shuffle(source_order)
        buckets = [task_positions[task_sources == source] for source in source_order]
        allocation = _round_robin_allocation(quota, [len(bucket) for bucket in buckets])
        for bucket, count in zip(buckets, allocation, strict=True):
            if count <= 0:
                continue
            offsets = _uniform_time_positions(len(bucket), count)
            selected.extend(int(position) for position in bucket[offsets])

    # Top up if rounding or thin sources left us short of the exact target.
    if len(selected) < max_valid_anchors:
        already = set(selected)
        for position in range(total):
            if position not in already:
                selected.append(position)
                already.add(position)
                if len(selected) >= max_valid_anchors:
                    break
    return sorted(set(selected))[:max_valid_anchors]


class HrcValidAnchorDataset:
    """Index view that removes anchors without a complete future action chunk."""

    def __init__(
        self,
        dataset,
        *,
        valid_mask: np.ndarray,
        task_ids: np.ndarray,
        max_valid_anchors: int | None = None,
        source_ids: np.ndarray | None = None,
        diag_subset_seed: int | None = None,
    ):
        valid_mask = np.asarray(valid_mask, dtype=bool)
        task_ids = np.asarray(task_ids, dtype=np.int64)
        if len(dataset) != len(valid_mask) or len(dataset) != len(task_ids):
            raise ValueError("dataset, valid_mask, and task_ids must have the same length")
        if source_ids is not None:
            source_ids = np.asarray(source_ids, dtype=np.int64)
            if len(source_ids) != len(dataset):
                raise ValueError("source_ids must have the same length as dataset")
        if diag_subset_seed is not None and source_ids is None:
            raise ValueError("diag_subset_seed requires source_ids for stratified selection")

        raw_indices = np.flatnonzero(valid_mask).astype(np.int64)
        valid_task_ids = task_ids[raw_indices]
        valid_source_ids = source_ids[raw_indices] if source_ids is not None else None
        if max_valid_anchors is not None:
            if max_valid_anchors <= 0:
                raise ValueError("max_valid_anchors must be positive")
            if max_valid_anchors < len(raw_indices):
                if diag_subset_seed is not None:
                    # Stratified diagnostic subset: even task quota, round-robin
                    # over source episodes, uniform in time within a source.
                    selected_positions = select_diag_subset_positions(
                        valid_task_ids,
                        valid_source_ids,
                        max_valid_anchors=max_valid_anchors,
                        seed=diag_subset_seed,
                    )
                else:
                    # Legacy: first N anchors per task (kept for callers without
                    # source metadata / diagnostic seed).
                    selected_positions = []
                    unique_tasks = np.unique(valid_task_ids)
                    base, remainder = divmod(max_valid_anchors, len(unique_tasks))
                    for task_offset, task_id in enumerate(unique_tasks):
                        quota = base + int(task_offset < remainder)
                        positions = np.flatnonzero(valid_task_ids == task_id)
                        selected_positions.extend(positions[:quota].tolist())
                    if len(selected_positions) < max_valid_anchors:
                        selected_set = set(selected_positions)
                        remaining = (i for i in range(len(raw_indices)) if i not in selected_set)
                        selected_positions.extend(
                            next(remaining) for _ in range(max_valid_anchors - len(selected_positions))
                        )
                    selected_positions = sorted(selected_positions[:max_valid_anchors])
                raw_indices = raw_indices[selected_positions]
                valid_task_ids = valid_task_ids[selected_positions]
                if valid_source_ids is not None:
                    valid_source_ids = valid_source_ids[selected_positions]

        self._dataset = dataset
        self._raw_indices = raw_indices
        self._task_ids = valid_task_ids
        self._source_ids = valid_source_ids

    @classmethod
    def from_lerobot(
        cls,
        dataset,
        *,
        metadata_dataset=None,
        max_valid_anchors: int | None = None,
        diag_subset_seed: int | None = None,
    ) -> "HrcValidAnchorDataset":
        """Build the index without materializing video frames."""
        metadata_dataset = metadata_dataset if metadata_dataset is not None else dataset
        try:
            hf_dataset = metadata_dataset.hf_dataset
            valid_values = hf_dataset["hrc.valid_anchor"]
            task_values = hf_dataset["hrc.task_id"]
        except (AttributeError, KeyError) as exc:
            raise ValueError("HRC LeRobot dataset is missing anchor/task metadata") from exc

        valid_mask = np.fromiter((bool(value) for value in valid_values), dtype=bool, count=len(dataset))
        task_ids = np.fromiter((int(value) for value in task_values), dtype=np.int64, count=len(dataset))

        source_ids = None
        if diag_subset_seed is not None:
            try:
                source_values = hf_dataset["hrc.source_episode_id"]
            except KeyError as exc:
                raise ValueError(
                    "diag_subset_seed requires hrc.source_episode_id metadata for stratified selection"
                ) from exc
            source_ids = np.fromiter(
                (int(value) for value in source_values), dtype=np.int64, count=len(dataset)
            )

        return cls(
            dataset,
            valid_mask=valid_mask,
            task_ids=task_ids,
            max_valid_anchors=max_valid_anchors,
            source_ids=source_ids,
            diag_subset_seed=diag_subset_seed,
        )

    @property
    def raw_indices(self) -> np.ndarray:
        return self._raw_indices.copy()

    @property
    def task_ids(self) -> np.ndarray:
        return self._task_ids.copy()

    @property
    def source_ids(self) -> np.ndarray | None:
        return None if self._source_ids is None else self._source_ids.copy()

    def __getitem__(self, index: SupportsIndex):
        return self._dataset[int(self._raw_indices[int(index)])]

    def __len__(self) -> int:
        return len(self._raw_indices)


class HrcTaskSampler(torch.utils.data.Sampler[int]):
    """Replacement sampler whose marginal task distribution matches target weights."""

    def __init__(
        self,
        task_ids: np.ndarray,
        *,
        task_weights: tuple[float, float, float],
        num_samples: int,
        seed: int,
    ):
        task_ids = np.asarray(task_ids, dtype=np.int64)
        target = np.asarray(task_weights, dtype=np.float64)
        if target.shape != (3,) or np.any(target < 0) or not np.isfinite(target).all() or target.sum() <= 0:
            raise ValueError("task_weights must contain three finite non-negative values")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        target /= target.sum()

        counts = np.bincount(task_ids, minlength=3)
        for task_id, (weight, count) in enumerate(zip(target, counts, strict=True)):
            if weight > 0 and count == 0:
                raise ValueError(f"task {task_id} has positive weight but no valid anchors")
        if np.any((task_ids < 0) | (task_ids >= 3)):
            raise ValueError("task_ids must be 0, 1, or 2")

        per_row = np.array([target[task_id] / counts[task_id] for task_id in task_ids], dtype=np.float64)
        self._weights = torch.as_tensor(per_row, dtype=torch.double)
        self._num_samples = int(num_samples)
        self._generator = torch.Generator().manual_seed(seed)

    def __iter__(self) -> Iterator[int]:
        indices = torch.multinomial(
            self._weights,
            self._num_samples,
            replacement=True,
            generator=self._generator,
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self._num_samples
