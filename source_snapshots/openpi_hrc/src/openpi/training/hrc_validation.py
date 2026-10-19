from collections.abc import Mapping, Set

import numpy as np


def validate_exported_split(
    columns: Mapping[str, np.ndarray],
    *,
    expected_sources: Set[int] | set[int],
    action_horizon: int,
) -> dict[str, object]:
    required = (
        "observation.state",
        "observation.privileged_state",
        "action",
        "hrc.valid_anchor",
        "hrc.task_id",
        "hrc.source_episode_id",
        "hrc.raw_frame_index",
        "episode_index",
    )
    missing = [key for key in required if key not in columns]
    if missing:
        raise ValueError(f"missing exported columns: {missing}")
    arrays = {key: np.asarray(columns[key]) for key in required}
    row_count = len(arrays["action"])
    if row_count == 0 or any(len(value) != row_count for value in arrays.values()):
        raise ValueError("all exported columns must have the same non-zero row count")
    if arrays["observation.state"].shape != (row_count, 23):
        raise ValueError("main state must have shape [N,23]")
    if arrays["observation.privileged_state"].shape != (row_count, 28):
        raise ValueError("privileged state must have shape [N,28]")
    if arrays["action"].shape != (row_count, 23):
        raise ValueError("action must have shape [N,23]")
    for key in ("observation.state", "observation.privileged_state", "action"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"non-finite values in {key}")

    sources = {int(value) for value in np.unique(arrays["hrc.source_episode_id"])}
    if sources != set(expected_sources):
        raise ValueError(f"exported source set {sorted(sources)} does not match expected {sorted(expected_sources)}")
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")

    episodes = np.unique(arrays["episode_index"])
    episodes_by_source: dict[int, list[int]] = {source: [] for source in sources}
    for episode_id in episodes:
        indices = np.flatnonzero(arrays["episode_index"] == episode_id)
        if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError(f"episode {episode_id} rows are not contiguous")
        source_values = np.unique(arrays["hrc.source_episode_id"][indices])
        task_values = np.unique(arrays["hrc.task_id"][indices])
        if len(source_values) != 1 or len(task_values) != 1:
            raise ValueError(f"episode {episode_id} mixes source/task metadata")
        source = int(source_values[0])
        episodes_by_source[source].append(int(episode_id))

        raw_frames = arrays["hrc.raw_frame_index"][indices]
        if not np.all(np.diff(raw_frames) == 1):
            raise ValueError(f"episode {episode_id} raw frame indices are not consecutive")
        episode_length = len(indices)
        if episode_length < action_horizon:
            raise ValueError(f"episode {episode_id} is shorter than action horizon")
        expected_valid = np.arange(episode_length) < episode_length - action_horizon + 1
        if not np.array_equal(arrays["hrc.valid_anchor"][indices].astype(bool), expected_valid):
            raise ValueError(f"episode {episode_id} has an invalid valid-anchor mask")

        states = arrays["observation.state"][indices]
        actions = arrays["action"][indices]
        if not np.allclose(states[0, :3], 0.0, atol=1e-6):
            raise ValueError(f"episode {episode_id} first previous base command is not zero")
        if not np.allclose(states[1:, :3], actions[:-1, :3], rtol=1e-6, atol=1e-6):
            raise ValueError(f"episode {episode_id} previous base command does not match prior action")

    for source, source_episodes in episodes_by_source.items():
        if len(source_episodes) != 3:
            raise ValueError(f"source {source} must have exactly three task episodes")
        tasks = {
            int(arrays["hrc.task_id"][np.flatnonzero(arrays["episode_index"] == episode)[0]])
            for episode in source_episodes
        }
        if tasks != {0, 1, 2}:
            raise ValueError(f"source {source} does not contain pickup/follow/radio exactly once")

    valid = arrays["hrc.valid_anchor"].astype(bool)
    task_counts = np.bincount(arrays["hrc.task_id"][valid].astype(np.int64), minlength=3)
    return {
        "num_sources": len(sources),
        "num_episodes": len(episodes),
        "num_rows": row_count,
        "num_valid_anchors": int(valid.sum()),
        "valid_anchors_by_task": task_counts.tolist(),
    }
