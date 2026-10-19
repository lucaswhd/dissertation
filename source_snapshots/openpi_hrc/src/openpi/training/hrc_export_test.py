from pathlib import Path

import h5py
import numpy as np

from openpi.training import hrc_data
from openpi.training import hrc_export


class CollectingWriter:
    def __init__(self):
        self.frames = []
        self.saved = 0

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved += 1


def test_lerobot_features_preserve_both_states_and_raw_action23():
    features = hrc_export.lerobot_features(image_height=256, image_width=352)

    assert features["observation.rgb.zed_link_camera_0"]["shape"] == (256, 352, 3)
    assert features["observation.state"]["shape"] == (23,)
    assert features["observation.privileged_state"]["shape"] == (28,)
    assert features["action"]["shape"] == (23,)
    assert features["hrc.valid_anchor"]["shape"] == (1,)


def test_write_segment_episode_keeps_row_alignment_and_metadata():
    writer = CollectingWriter()
    rows = {
        "state23": np.arange(3 * 23, dtype=np.float32).reshape(3, 23),
        "state28": np.arange(3 * 28, dtype=np.float32).reshape(3, 28),
        "action": np.arange(3 * 23, dtype=np.float32).reshape(3, 23) + 100,
        "fsm_state_id": np.array([2, 3, 5], dtype=np.int32),
        "raw_frame_index": np.array([10, 11, 12], dtype=np.int64),
        "valid_anchor": np.array([True, False, False]),
        "prompt": np.array(["phase 2", "phase 3", "phase 5"], dtype=object),
    }
    images = [np.full((4, 6, 3), value, dtype=np.uint8) for value in (10, 11, 12)]

    hrc_export.write_segment_episode(
        writer,
        rows=rows,
        images=images,
        source_episode_id=7,
        task_name="pickup",
    )

    assert writer.saved == 1
    assert len(writer.frames) == 3
    assert writer.frames[0]["task"] == "phase 2"
    assert writer.frames[2]["hrc.task_id"].item() == 0
    assert writer.frames[1]["hrc.source_episode_id"].item() == 7
    assert writer.frames[2]["hrc.raw_frame_index"].item() == 12
    assert not writer.frames[1]["hrc.valid_anchor"].item()
    np.testing.assert_array_equal(writer.frames[1]["observation.rgb.zed_link_camera_0"], images[1])
    np.testing.assert_array_equal(writer.frames[1]["action"], rows["action"][1])


def test_select_monotonic_frames_advances_decoder_once():
    frames = (np.full((2, 3, 3), index, dtype=np.uint8) for index in range(8))

    selected = hrc_export.select_monotonic_frames(frames, [2, 3, 6])

    assert [int(frame[0, 0, 0]) for frame in selected] == [2, 3, 6]


def test_monotonic_frame_cursor_streams_without_buffering_whole_video():
    frames = (np.full((2, 3, 3), index, dtype=np.uint8) for index in range(8))
    cursor = hrc_export.MonotonicFrameCursor(frames)

    assert int(cursor.get(2)[0, 0, 0]) == 2
    assert int(cursor.get(3)[0, 0, 0]) == 3
    assert int(cursor.get(6)[0, 0, 0]) == 6

    with np.testing.assert_raises_regex(ValueError, "strictly increasing"):
        cursor.get(5)


def test_load_segment_rows_matches_raw_shifted_action():
    data_root = Path("/root/autodl-tmp/hrc_pipeline_traintest/outputs/hrc_batch_final")
    manifest = hrc_data.build_manifest(data_root, seed=42, action_horizon=32)
    segment = next(segment for segment in manifest["segments"] if segment["source_episode_id"] == 0)

    rows = hrc_export.load_segment_rows(segment, include_phase=True, action_horizon=32)

    with h5py.File(segment["hdf5_path"], "r") as hdf5_file:
        demo = hdf5_file[f"data/{segment['demo_name']}"]
        raw_actions = demo["action"][:]
    assert len(rows["action"]) == segment["num_frames"] - 1
    np.testing.assert_array_equal(rows["action"][0], raw_actions[segment["start_frame"] + 1])
    assert int(rows["valid_anchor"].sum()) == segment["num_valid_anchors"]
