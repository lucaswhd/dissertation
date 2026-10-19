from collections.abc import Mapping, Sequence
import dataclasses

import numpy as np

from openpi.shared.normalize import NormStats


# Matches shared/normalize.py RunningStats, so HRC quantiles carry the same
# discretization error as every other openpi dataset.
NUM_QUANTILE_BINS = 5000


@dataclasses.dataclass
class RunningQuantiles:
    """Fixed-grid weighted histogram, for quantiles that honour ``task_weights``.

    ``shared.normalize.RunningStats`` already computes streaming quantiles, but each
    instance owns a private grid that it rescales as new extremes arrive, and it has
    no notion of the per-task reweighting ``combine_task_moments`` applies to the
    moments. Pinning one grid per key up front instead lets the *same* weights drive
    mean/std and q01/q99, so the two halves of a NormStats cannot end up describing
    different distributions.
    """

    low: np.ndarray  # [feature]
    high: np.ndarray  # [feature]
    counts: np.ndarray  # [feature, NUM_QUANTILE_BINS]

    @classmethod
    def over_range(cls, low, high, *, bins: int = NUM_QUANTILE_BINS) -> "RunningQuantiles":
        low = np.asarray(low, dtype=np.float64).ravel()
        high = np.asarray(high, dtype=np.float64).ravel()
        if low.shape != high.shape:
            raise ValueError("low and high must share a shape")
        if bins <= 0:
            raise ValueError("bins must be positive")
        # A constant dim still needs a non-degenerate grid, else every value lands
        # outside the edges and the quantiles collapse onto the bound.
        pad = np.maximum(np.abs(high - low), 1.0) * 1e-9
        return cls(low=low - pad, high=high + pad, counts=np.zeros((len(low), bins), dtype=np.float64))

    @property
    def edges(self) -> np.ndarray:
        """[feature, bins + 1] bin edges."""
        steps = np.linspace(0.0, 1.0, self.counts.shape[-1] + 1)
        return self.low[:, None] + steps[None, :] * (self.high - self.low)[:, None]

    @property
    def total(self) -> float:
        return float(self.counts[0].sum()) if len(self.counts) else 0.0

    def update(self, batch: np.ndarray) -> None:
        dim, bins = self.counts.shape
        batch = np.asarray(batch, dtype=np.float64).reshape(-1, dim)
        if len(batch) == 0:
            return
        scale = bins / np.maximum(self.high - self.low, 1e-300)
        index = ((batch - self.low) * scale).astype(np.int64)
        np.clip(index, 0, bins - 1, out=index)
        # One bincount over dim-major flat indices, rather than `dim` histogram calls:
        # the action key is ~10M rows x 23 dims, where the per-dim loop dominates.
        index += np.arange(dim, dtype=np.int64) * bins
        self.counts += np.bincount(index.ravel(), minlength=dim * bins).reshape(dim, bins)


def quantiles_from_histogram(edges: np.ndarray, counts: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    """[len(quantiles), feature] linear-interpolated quantiles of a weighted histogram."""
    out = np.zeros((len(quantiles), len(counts)), dtype=np.float64)
    for feature, row in enumerate(counts):
        total = row.sum()
        if total <= 0:
            out[:, feature] = edges[feature][0]
            continue
        cdf = np.cumsum(row) / total
        for j, q in enumerate(quantiles):
            index = min(int(np.searchsorted(cdf, q, side="left")), len(row) - 1)
            below = cdf[index - 1] if index > 0 else 0.0
            span = cdf[index] - below
            fraction = 0.0 if span <= 0 else (q - below) / span
            lo, hi = edges[feature][index], edges[feature][index + 1]
            out[j, feature] = lo + np.clip(fraction, 0.0, 1.0) * (hi - lo)
    return out


@dataclasses.dataclass
class RunningMoments:
    count: int = 0
    mean: np.ndarray | None = None
    mean_of_squares: np.ndarray | None = None

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 0:
            raise ValueError("moment input must have a feature dimension")
        batch = batch.reshape(-1, batch.shape[-1])
        if len(batch) == 0:
            return
        batch_mean = batch.mean(axis=0)
        batch_second = np.square(batch).mean(axis=0)
        if self.count == 0:
            self.mean = batch_mean
            self.mean_of_squares = batch_second
            self.count = len(batch)
            return
        if self.mean is None or self.mean.shape != batch_mean.shape:
            raise ValueError("moment feature dimension changed")
        new_count = self.count + len(batch)
        ratio = len(batch) / new_count
        self.mean += (batch_mean - self.mean) * ratio
        self.mean_of_squares += (batch_second - self.mean_of_squares) * ratio
        self.count = new_count


def combine_task_moments(
    moments: Mapping[int, RunningMoments],
    *,
    task_weights: Sequence[float] | None,
    quantiles: Mapping[int, RunningQuantiles] | None = None,
) -> NormStats:
    usable = {task_id: value for task_id, value in moments.items() if value.count > 0}
    if not usable:
        raise ValueError("no task moments were collected")
    if task_weights is None:
        total = sum(value.count for value in usable.values())
        weights = {task_id: value.count / total for task_id, value in usable.items()}
    else:
        target = np.asarray(task_weights, dtype=np.float64)
        if target.ndim != 1 or np.any(target < 0) or not np.isfinite(target).all() or target.sum() <= 0:
            raise ValueError("task_weights must be finite, non-negative, and non-empty")
        target /= target.sum()
        for task_id, weight in enumerate(target):
            if weight > 0 and task_id not in usable:
                raise ValueError(f"task {task_id} has positive weight but no moments")
        weights = {task_id: target[task_id] for task_id in usable if target[task_id] > 0}

    first = next(iter(usable.values()))
    assert first.mean is not None
    assert first.mean_of_squares is not None
    mean = np.zeros_like(first.mean)
    second = np.zeros_like(first.mean_of_squares)
    for task_id, weight in weights.items():
        task = usable[task_id]
        assert task.mean is not None
        assert task.mean_of_squares is not None
        mean += weight * task.mean
        second += weight * task.mean_of_squares
    std = np.sqrt(np.maximum(0.0, second - np.square(mean)))

    q01 = q99 = None
    if quantiles:
        # Reweight each task's histogram to the same target distribution the moments
        # use, so q01/q99 and mean/std describe one distribution rather than two.
        usable_hist = {t: h for t, h in quantiles.items() if t in weights and h.total > 0}
        if usable_hist:
            reference = next(iter(usable_hist.values()))
            pooled = np.zeros_like(reference.counts)
            for task_id, histogram in usable_hist.items():
                if histogram.counts.shape != pooled.shape:
                    raise ValueError("task histograms must share a grid")
                per_task_total = histogram.counts.sum(axis=-1, keepdims=True)
                pooled += weights[task_id] * histogram.counts / np.maximum(per_task_total, 1.0)
            q01, q99 = quantiles_from_histogram(reference.edges, pooled, (0.01, 0.99))

    return NormStats(mean=mean, std=std, q01=q01, q99=q99)


def compute_hrc_norm_stats_from_arrays(
    *,
    states: np.ndarray,
    actions: np.ndarray,
    valid_mask: np.ndarray,
    task_ids: np.ndarray,
    action_horizon: int,
    delta_mappings: Sequence[tuple[Sequence[int], Sequence[int]]],
    task_weights: Sequence[float] | None,
    chunk_size: int = 4096,
) -> dict[str, NormStats]:
    states = np.asarray(states)
    actions = np.asarray(actions)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    task_ids = np.asarray(task_ids, dtype=np.int64)
    row_count = len(states)
    if not (len(actions) == len(valid_mask) == len(task_ids) == row_count):
        raise ValueError("states, actions, valid_mask, and task_ids must have the same length")
    if action_horizon <= 0 or chunk_size <= 0:
        raise ValueError("action_horizon and chunk_size must be positive")

    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) == 0:
        raise ValueError("no valid HRC anchors")
    if int(valid_indices.max()) + action_horizon > row_count:
        raise ValueError("valid anchor action chunk exceeds the dataset")

    def iter_blocks():
        """(task_id, anchor states, delta-mapped action chunks) per anchor block."""
        for task_id in np.unique(task_ids[valid_indices]):
            task_indices = valid_indices[task_ids[valid_indices] == task_id]
            for start in range(0, len(task_indices), chunk_size):
                anchors = task_indices[start : start + chunk_size]
                anchor_states = np.asarray(states[anchors], dtype=np.float64)
                offsets = anchors[:, None] + np.arange(action_horizon)[None, :]
                chunks = np.asarray(actions[offsets], dtype=np.float64).copy()
                for action_indices, state_indices in delta_mappings:
                    if len(action_indices) != len(state_indices):
                        raise ValueError("delta action/state mapping lengths must match")
                    chunks[..., action_indices] -= anchor_states[:, None, state_indices]
                yield int(task_id), anchor_states, chunks

    # Pass 1: moments, plus the extent every task's histogram will share. The grid
    # has to be fixed before any counting starts, otherwise tasks end up on
    # different edges and their histograms cannot be pooled.
    state_moments: dict[int, RunningMoments] = {}
    action_moments: dict[int, RunningMoments] = {}
    bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for task_id, anchor_states, chunks in iter_blocks():
        state_moments.setdefault(task_id, RunningMoments()).update(anchor_states)
        action_moments.setdefault(task_id, RunningMoments()).update(chunks)
        flat_chunks = chunks.reshape(-1, chunks.shape[-1])
        for key, block in (("state", anchor_states), ("actions", flat_chunks)):
            low, high = block.min(axis=0), block.max(axis=0)
            if key in bounds:
                low = np.minimum(bounds[key][0], low)
                high = np.maximum(bounds[key][1], high)
            bounds[key] = (low, high)

    # Pass 2: fill the histograms now that the grid is known.
    state_hist: dict[int, RunningQuantiles] = {}
    action_hist: dict[int, RunningQuantiles] = {}
    for task_id, anchor_states, chunks in iter_blocks():
        state_hist.setdefault(task_id, RunningQuantiles.over_range(*bounds["state"])).update(anchor_states)
        action_hist.setdefault(task_id, RunningQuantiles.over_range(*bounds["actions"])).update(chunks)

    return {
        "state": combine_task_moments(state_moments, task_weights=task_weights, quantiles=state_hist),
        "actions": combine_task_moments(action_moments, task_weights=task_weights, quantiles=action_hist),
    }
