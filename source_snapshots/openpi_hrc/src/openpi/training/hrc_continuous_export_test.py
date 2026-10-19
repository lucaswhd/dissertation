import json

import numpy as np
import pytest

from openpi.training.hrc_continuous_data import WAIT_INSTRUCTION
from openpi.training.hrc_continuous_data import build_continuous_rows
from openpi.training.hrc_continuous_export import continuous_lerobot_features
from openpi.training.hrc_continuous_export import require_manual_qc_confirmation
from openpi.training.hrc_continuous_export import write_continuous_episode


class _FakeWriter:
    def __init__(self):
        self.frames = []
        self.saved = 0

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved += 1


def _toy_demo(length, *, seed, fsm=None):
    rng = np.random.default_rng(seed)
    return {
        "action": rng.standard_normal((length, 23)).astype(np.float32),
        "joint_positions": rng.standard_normal((length, 28)).astype(np.float32),
        "gripper_state": rng.standard_normal((length, 4)).astype(np.float32),
        "fsm_state_id": (
            np.full(length, fsm, dtype=np.int64) if fsm is not None else np.full(length, 2, dtype=np.int64)
        ),
    }


def _toy_rows(action_horizon=8):
    demo0 = _toy_demo(50, seed=1)
    demo1 = _toy_demo(60, seed=2, fsm=9)
    commands = {
        0: {"round_id": 0, "instruction": "Pick up the soda cans.", "issued_frame": 10},
        1: {"round_id": 1, "instruction": "Turn on the radio.", "issued_frame": 70},
    }
    return build_continuous_rows(demo0=demo0, demo1=demo1, commands=commands, action_horizon=action_horizon)


def test_features_are_state23_without_privileged_and_carry_context():
    features = continuous_lerobot_features()
    assert features["observation.state"]["shape"] == (23,)
    assert "observation.privileged_state" not in features  # no human coordinates
    assert "hrc.context_id" in features
    assert "hrc.raw_global_frame_index" in features


def test_write_continuous_episode_single_episode_ordered_and_context_prompts():
    rows = _toy_rows()
    n = len(rows["action"])
    # distinct per-frame images so we can assert order preservation.
    images = [np.full((2, 2, 3), i % 256, dtype=np.uint8) for i in range(n)]
    writer = _FakeWriter()

    write_continuous_episode(writer, rows=rows, images=images, source_episode_id=15)

    assert writer.saved == 1  # exactly one episode per source
    assert len(writer.frames) == n
    # image order strictly preserved
    for i, frame in enumerate(writer.frames):
        assert int(frame["observation.rgb.zed_link_camera_0"][0, 0, 0]) == i % 256
    # task_id mirrors the per-frame context id, and prompt changes across contexts
    assert int(writer.frames[0]["hrc.task_id"][0]) == 0
    assert int(writer.frames[70]["hrc.task_id"][0]) == 3
    assert WAIT_INSTRUCTION in writer.frames[0]["task"]
    assert "Turn on the radio." in writer.frames[70]["task"]
    assert writer.frames[0]["task"] != writer.frames[70]["task"]
    # last frame exists but is never a valid anchor
    assert bool(writer.frames[-1]["hrc.valid_anchor"][0]) is False


def test_write_continuous_episode_rejects_image_count_mismatch():
    rows = _toy_rows()
    n = len(rows["action"])
    writer = _FakeWriter()
    try:
        write_continuous_episode(
            writer, rows=rows, images=[np.zeros((2, 2, 3), np.uint8)] * (n - 1), source_episode_id=3
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError on too few images")


def test_manual_qc_gate_blocks_25_36_without_confirmation(tmp_path):
    with pytest.raises(ValueError, match="人工视频复核"):
        require_manual_qc_confirmation(None, [25, 36])

    partial = tmp_path / "qc.json"
    partial.write_text(json.dumps({"source_25": "PASS"}), encoding="utf-8")  # 36 missing
    try:
        require_manual_qc_confirmation(partial, [25, 36])
    except ValueError:
        pass
    else:
        raise AssertionError("expected failure when a source is not PASS")


def test_manual_qc_gate_passes_when_confirmed(tmp_path):
    confirmation = tmp_path / "qc.json"
    confirmation.write_text(json.dumps({"source_25": "PASS", "source_36": "PASS"}), encoding="utf-8")
    require_manual_qc_confirmation(confirmation, [25, 36])  # must not raise
    require_manual_qc_confirmation(None, [])  # no gated sources -> no-op
