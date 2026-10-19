import numpy as np
import pytest

from scripts.hrc.serve_hrc_pi05_rollout import validate_action_chunk
from scripts.hrc.serve_hrc_pi05_rollout import validate_act_request
from scripts.hrc.serve_hrc_pi05_rollout import validate_execute_horizon
from scripts.hrc.serve_hrc_pi05_rollout import canonical_checkpoint_path


def test_validate_execute_horizon_accepts_required_replanning_protocol():
    assert validate_execute_horizon(8) == 8


def test_validate_execute_horizon_allows_replan_rate_experiments_but_rejects_impossible_ones():
    # Off-protocol rates are a deliberate experiment knob (warned about, not blocked);
    # a rate outside [1, action_horizon] is still impossible and must fail.
    assert validate_execute_horizon(1) == 1
    assert validate_execute_horizon(32) == 32
    for bad in (0, -1, 33):
        with pytest.raises(ValueError, match=r"within \[1, 32\]"):
            validate_execute_horizon(bad)


def test_checkpoint_identity_is_always_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert canonical_checkpoint_path("checkpoints/example/5000") == tmp_path / "checkpoints/example/5000"


def test_validate_act_request_normalizes_real_observation():
    state, head = validate_act_request(
        {"state": np.zeros(23), "head": np.zeros((224, 224, 3), dtype=np.uint8)}
    )
    assert state.shape == (23,)
    assert state.dtype == np.float32
    assert head.shape == (224, 224, 3)
    assert head.dtype == np.uint8


@pytest.mark.parametrize(
    "message",
    [
        {"state": np.zeros(22), "head": np.zeros((8, 8, 3), dtype=np.uint8)},
        {"state": np.full(23, np.nan), "head": np.zeros((8, 8, 3), dtype=np.uint8)},
        {"state": np.zeros(23), "head": np.zeros((8, 8), dtype=np.uint8)},
    ],
)
def test_validate_act_request_rejects_bad_observation(message):
    with pytest.raises(ValueError):
        validate_act_request(message)


def test_validate_action_chunk_requires_32_finite_action23_steps():
    actions = validate_action_chunk(np.zeros((32, 23), dtype=np.float32))
    assert actions.shape == (32, 23)
    with pytest.raises(ValueError, match="\[32, 23\]"):
        validate_action_chunk(np.zeros((8, 23)))
    bad = np.zeros((32, 23))
    bad[3, 5] = np.inf
    with pytest.raises(ValueError, match="finite"):
        validate_action_chunk(bad)
