"""Pure validation of an exported continuous HRC split (column-level, no LeRobot).

Given the per-row columns of one exported split, check the continuous data
contract: one episode per source, 4-context task ids, context-aware valid-anchor
mask (no cross-context / no sentinel), shift-1 prev-base continuity, finiteness,
and prompt phase ↔ FSM consistency. Heavy cross-checks against the raw HDF5/MP4
(state23 per-dim, action==raw[t+1], RGB, demo-boundary joints/grippers) live in
the CLI, which can rebuild rows with :mod:`hrc_continuous_data`.
"""

from collections.abc import Mapping, Sequence
import re

import numpy as np

from openpi.training.hrc_continuous_data import CONTEXT_NAMES
from openpi.training.hrc_continuous_data import build_context_valid_anchor_mask
from openpi.training.hrc_data import FSM_STATE_NAMES

_PHASE_RE = re.compile(r"Phase:\s*([A-Za-z]+)\.")


def check_prompt_phase_consistency(prompts: Sequence[str], fsm_state_ids: Sequence[int]) -> None:
    """Every prompt's ``Phase: <Name>.`` must equal ``FSM_STATE_NAMES[fsm_id]``."""

    prompts = list(prompts)
    fsm_state_ids = np.asarray(fsm_state_ids, dtype=np.int64)
    if len(prompts) != len(fsm_state_ids):
        raise ValueError("prompts and fsm_state_ids must have equal length")
    for index, (prompt, fsm_id) in enumerate(zip(prompts, fsm_state_ids, strict=True)):
        match = _PHASE_RE.search(str(prompt))
        if match is None:
            raise ValueError(f"row {index}: prompt has no 'Phase: <Name>.' segment: {prompt!r}")
        expected = FSM_STATE_NAMES.get(int(fsm_id))
        if expected is None:
            raise ValueError(f"row {index}: unknown FSM state id {int(fsm_id)}")
        if match.group(1) != expected:
            raise ValueError(
                f"row {index}: prompt phase {match.group(1)!r} != FSM {int(fsm_id)}->{expected!r}"
            )


def validate_continuous_columns(
    columns: Mapping[str, np.ndarray],
    *,
    expected_sources: set[int] | Sequence[int],
    action_horizon: int,
) -> dict[str, object]:
    """Validate one exported continuous split from its per-row columns."""

    required = (
        "observation.state",
        "action",
        "hrc.valid_anchor",
        "hrc.task_id",
        "hrc.context_id",
        "hrc.source_episode_id",
        "hrc.fsm_state_id",
        "hrc.raw_global_frame_index",
        "hrc.raw_demo_id",
        "hrc.raw_demo_frame_index",
        "episode_index",
    )
    missing = [key for key in required if key not in columns]
    if missing:
        raise ValueError(f"missing exported columns: {missing}")
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")

    arrays = {key: np.asarray(columns[key]) for key in required}
    row_count = len(arrays["action"])
    if row_count == 0 or any(len(value) != row_count for value in arrays.values()):
        raise ValueError("all exported columns must share the same non-zero row count")
    if arrays["observation.state"].shape != (row_count, 23):
        raise ValueError("state must have shape [N, 23]")
    if arrays["action"].shape != (row_count, 23):
        raise ValueError("action must have shape [N, 23]")
    for key in ("observation.state", "action"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"non-finite values in {key}")

    # task_id must equal context_id everywhere (0..3).
    task_id = arrays["hrc.task_id"].reshape(-1).astype(np.int64)
    context_id = arrays["hrc.context_id"].reshape(-1).astype(np.int64)
    if not np.array_equal(task_id, context_id):
        raise ValueError("hrc.task_id must equal hrc.context_id for the continuous export")
    if task_id.min() < 0 or task_id.max() > 3:
        raise ValueError("context ids must lie in [0, 3]")

    sources = {int(value) for value in np.unique(arrays["hrc.source_episode_id"])}
    if sources != {int(value) for value in expected_sources}:
        raise ValueError(
            f"exported sources {sorted(sources)} != expected {sorted(int(v) for v in expected_sources)}"
        )

    episodes = np.unique(arrays["episode_index"])
    episodes_by_source: dict[int, int] = {}
    for episode_id in episodes:
        rows = np.flatnonzero(arrays["episode_index"] == episode_id)
        if not np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
            raise ValueError(f"episode {episode_id} rows are not contiguous")
        source_values = np.unique(arrays["hrc.source_episode_id"][rows])
        if len(source_values) != 1:
            raise ValueError(f"episode {episode_id} mixes source ids")
        source = int(source_values[0])
        episodes_by_source[source] = episodes_by_source.get(source, 0) + 1

        length = len(rows)
        if length < action_horizon + 1:
            raise ValueError(f"episode {episode_id} shorter than action_horizon + 1")

        # raw indices: global consecutive from 0; demo id 0..1 with one 0->1 step.
        if not np.array_equal(arrays["hrc.raw_global_frame_index"][rows], np.arange(length)):
            raise ValueError(f"episode {episode_id} raw_global_frame_index is not 0..N-1")
        demo_ids = arrays["hrc.raw_demo_id"][rows].astype(np.int64)
        transitions = np.flatnonzero(np.diff(demo_ids) != 0)
        if set(np.unique(demo_ids).tolist()) != {0, 1} or len(transitions) != 1:
            raise ValueError(f"episode {episode_id} demo id must go 0..0 then 1..1 exactly once")

        # context-aware valid-anchor mask must match exactly (checks 9 + 10).
        episode_context = context_id[rows]
        expected_mask = build_context_valid_anchor_mask(episode_context, action_horizon=action_horizon)
        if not np.array_equal(arrays["hrc.valid_anchor"][rows].astype(bool), expected_mask):
            raise ValueError(f"episode {episode_id} valid-anchor mask disagrees with context mask")
        if bool(arrays["hrc.valid_anchor"][rows][-1]):
            raise ValueError(f"episode {episode_id} sentinel row is marked valid")

        # shift-1 prev-base continuity (row 0 zeroed; continuous across demo boundary).
        states = arrays["observation.state"][rows]
        actions = arrays["action"][rows]
        if not np.allclose(states[0, :3], 0.0, atol=1e-6):
            raise ValueError(f"episode {episode_id} first prev-base is not zero")
        if not np.allclose(states[1:, :3], actions[:-1, :3], rtol=1e-5, atol=1e-6):
            raise ValueError(f"episode {episode_id} prev-base does not match prior action base")

    multi = {source: count for source, count in episodes_by_source.items() if count != 1}
    if multi:
        raise ValueError(f"each source must map to exactly one episode; offenders: {multi}")

    valid = arrays["hrc.valid_anchor"].astype(bool).reshape(-1)
    by_context = {
        name: int(((context_id == index) & valid).sum()) for index, name in enumerate(CONTEXT_NAMES)
    }
    return {
        "num_sources": len(sources),
        "num_episodes": len(episodes),
        "num_rows": row_count,
        "num_valid_anchors": int(valid.sum()),
        "valid_anchors_by_context": by_context,
    }
