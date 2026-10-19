import numpy as np

from openpi.training.hrc_validation import validate_exported_split


def _make_columns(rows_per_episode=5, horizon=3):
    columns = {key: [] for key in (
        "observation.state",
        "observation.privileged_state",
        "action",
        "hrc.valid_anchor",
        "hrc.task_id",
        "hrc.source_episode_id",
        "hrc.raw_frame_index",
        "episode_index",
    )}
    episode = 0
    for source in (10, 11):
        for task in range(3):
            actions = np.zeros((rows_per_episode, 23), dtype=np.float32)
            actions[:, :3] = source + task + np.arange(rows_per_episode)[:, None]
            states = np.zeros((rows_per_episode, 23), dtype=np.float32)
            states[1:, :3] = actions[:-1, :3]
            privileged = np.pad(states, ((0, 0), (0, 5)))
            valid = np.arange(rows_per_episode) < rows_per_episode - horizon + 1
            columns["observation.state"].extend(states)
            columns["observation.privileged_state"].extend(privileged)
            columns["action"].extend(actions)
            columns["hrc.valid_anchor"].extend(valid)
            columns["hrc.task_id"].extend([task] * rows_per_episode)
            columns["hrc.source_episode_id"].extend([source] * rows_per_episode)
            columns["hrc.raw_frame_index"].extend(range(100, 100 + rows_per_episode))
            columns["episode_index"].extend([episode] * rows_per_episode)
            episode += 1
    return {key: np.asarray(value) for key, value in columns.items()}


def test_validate_exported_split_checks_episode_and_shift_contract():
    report = validate_exported_split(
        _make_columns(),
        expected_sources={10, 11},
        action_horizon=3,
    )
    assert report["num_sources"] == 2
    assert report["num_episodes"] == 6
    assert report["num_rows"] == 30
    assert report["valid_anchors_by_task"] == [6, 6, 6]


def test_validate_exported_split_rejects_leakage_padding_and_shift_errors():
    columns = _make_columns()
    with np.testing.assert_raises_regex(ValueError, "source set"):
        validate_exported_split(columns, expected_sources={10}, action_horizon=3)

    bad_valid = {**columns, "hrc.valid_anchor": columns["hrc.valid_anchor"].copy()}
    bad_valid["hrc.valid_anchor"][4] = True
    with np.testing.assert_raises_regex(ValueError, "valid-anchor mask"):
        validate_exported_split(bad_valid, expected_sources={10, 11}, action_horizon=3)

    bad_state = {**columns, "observation.state": columns["observation.state"].copy()}
    bad_state["observation.state"][1, 0] += 1
    with np.testing.assert_raises_regex(ValueError, "previous base"):
        validate_exported_split(bad_state, expected_sources={10, 11}, action_horizon=3)
