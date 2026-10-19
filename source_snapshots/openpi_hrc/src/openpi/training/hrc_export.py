from collections.abc import Iterable, Mapping
import json
from pathlib import Path
import shutil
from typing import Protocol

import av
import h5py
import numpy as np

from openpi.training import hrc_data


class EpisodeWriter(Protocol):
    def add_frame(self, frame: dict) -> None: ...

    def save_episode(self) -> None: ...


def lerobot_features(*, image_height: int, image_width: int) -> dict[str, dict]:
    scalar = {"shape": (1,), "names": ["value"]}
    return {
        "observation.rgb.zed_link_camera_0": {
            "dtype": "video",
            "shape": (image_height, image_width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {"dtype": "float32", "shape": (23,), "names": ["state"]},
        "observation.privileged_state": {
            "dtype": "float32",
            "shape": (28,),
            "names": ["privileged_state"],
        },
        "action": {"dtype": "float32", "shape": (23,), "names": ["action"]},
        "hrc.valid_anchor": {"dtype": "bool", **scalar},
        "hrc.task_id": {"dtype": "int64", **scalar},
        "hrc.source_episode_id": {"dtype": "int64", **scalar},
        "hrc.fsm_state_id": {"dtype": "int64", **scalar},
        "hrc.raw_frame_index": {"dtype": "int64", **scalar},
    }


def write_segment_episode(
    writer: EpisodeWriter,
    *,
    rows: Mapping[str, np.ndarray],
    images: Iterable[np.ndarray],
    source_episode_id: int,
    task_name: str,
) -> None:
    try:
        task_id = hrc_data.TASK_NAMES.index(task_name)
    except ValueError as exc:
        raise ValueError(f"Unknown HRC task: {task_name}") from exc

    num_rows = len(rows["action"])
    if any(len(rows[key]) != num_rows for key in rows):
        raise ValueError("All exported row arrays must have equal length")

    image_iterator = iter(images)
    for index in range(num_rows):
        try:
            image = next(image_iterator)
        except StopIteration as exc:
            raise ValueError(f"Expected {num_rows} images, got {index}") from exc
        writer.add_frame(
            {
                "observation.rgb.zed_link_camera_0": np.asarray(image, dtype=np.uint8),
                "observation.state": np.asarray(rows["state23"][index], dtype=np.float32),
                "observation.privileged_state": np.asarray(rows["state28"][index], dtype=np.float32),
                "action": np.asarray(rows["action"][index], dtype=np.float32),
                "hrc.valid_anchor": np.asarray([rows["valid_anchor"][index]], dtype=np.bool_),
                "hrc.task_id": np.asarray([task_id], dtype=np.int64),
                "hrc.source_episode_id": np.asarray([source_episode_id], dtype=np.int64),
                "hrc.fsm_state_id": np.asarray([rows["fsm_state_id"][index]], dtype=np.int64),
                "hrc.raw_frame_index": np.asarray([rows["raw_frame_index"][index]], dtype=np.int64),
                "task": str(rows["prompt"][index]),
            }
        )
    try:
        next(image_iterator)
    except StopIteration:
        pass
    else:
        raise ValueError(f"Expected {num_rows} images, got more")
    writer.save_episode()


def select_monotonic_frames(frames: Iterable[np.ndarray], indices: Iterable[int]) -> list[np.ndarray]:
    indices = [int(index) for index in indices]
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError("Frame indices must be strictly increasing")
    if not indices:
        return []

    selected = []
    target_position = 0
    for frame_index, frame in enumerate(frames):
        if frame_index == indices[target_position]:
            selected.append(np.asarray(frame))
            target_position += 1
            if target_position == len(indices):
                return selected
    raise ValueError(f"Video ended before frame {indices[-1]}")


class MonotonicFrameCursor:
    def __init__(self, frames: Iterable[np.ndarray]):
        self._frames = iter(frames)
        self._frame_index = -1
        self._frame = None

    def get(self, target_index: int) -> np.ndarray:
        target_index = int(target_index)
        if target_index <= self._frame_index:
            raise ValueError(
                f"Frame targets must be strictly increasing: current={self._frame_index}, requested={target_index}"
            )
        while self._frame_index < target_index:
            try:
                self._frame = next(self._frames)
            except StopIteration as exc:
                raise ValueError(f"Video ended before frame {target_index}") from exc
            self._frame_index += 1
        return np.asarray(self._frame)


def load_segment_rows(
    segment: Mapping[str, object],
    *,
    include_phase: bool,
    action_horizon: int,
) -> dict[str, np.ndarray]:
    with h5py.File(str(segment["hdf5_path"]), "r") as hdf5_file:
        demo = hdf5_file[f"data/{segment['demo_name']}"]
        return hrc_data.build_segment_arrays(
            actions=demo["action"][:],
            joint_positions=demo["obs/joint_positions"][:],
            gripper_state=demo["obs/gripper_state"][:],
            relative_human_pose=demo["obs/relative_human_pose"][:],
            human_visible=demo["obs/human_visible"][:],
            head_pitch_deg=demo["obs/head_camera_pitch_deg"][:],
            fsm_state_id=demo["obs/fsm_state_id"][:],
            start=int(segment["start_frame"]),
            end=int(segment["end_frame_exclusive"]),
            instruction=str(segment["instruction"]),
            include_phase=include_phase,
            action_horizon=action_horizon,
        )


def decode_rgb_frames(video_path: Path | str) -> Iterable[np.ndarray]:
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            yield frame.to_ndarray(format="rgb24")


def export_manifest(
    manifest: Mapping[str, object],
    output_root: Path | str,
    *,
    include_phase: bool,
    splits: Iterable[str] = ("train", "val", "test"),
    max_sources_per_split: int | None = None,
    overwrite: bool = False,
) -> dict:
    from lerobot.datasets import LeRobotDataset

    output_root = Path(output_root).resolve()
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    with (output_root / "manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=2)

    report = {"schema_version": manifest["schema_version"], "splits": {}}
    all_segments = list(manifest["segments"])
    for split in splits:
        split_segments = [segment for segment in all_segments if segment["split"] == split]
        source_ids = sorted({int(segment["source_episode_id"]) for segment in split_segments})
        if max_sources_per_split is not None:
            source_ids = source_ids[:max_sources_per_split]
            split_segments = [segment for segment in split_segments if int(segment["source_episode_id"]) in source_ids]

        split_root = output_root / split
        dataset = LeRobotDataset.create(
            repo_id=f"hrc/hrc_pi05_v0_{split}",
            root=split_root,
            robot_type="R1Pro",
            fps=30,
            features=lerobot_features(image_height=256, image_width=352),
            use_videos=True,
            video_backend="pyav",
            image_writer_threads=4,
        )
        episode_count = 0
        row_count = 0
        valid_anchor_count = 0
        task_counts = dict.fromkeys(hrc_data.TASK_NAMES, 0)
        for source_id in source_ids:
            source_segments = [
                segment for segment in split_segments if int(segment["source_episode_id"]) == source_id
            ]
            source_segments.sort(key=lambda segment: (segment["demo_name"], int(segment["start_frame"])))
            with h5py.File(str(source_segments[0]["hdf5_path"]), "r") as hdf5_file:
                demo0_length = int(hdf5_file["data/demo_0/action"].shape[0])
            cursor = MonotonicFrameCursor(decode_rgb_frames(Path(str(source_segments[0]["head_video_path"]))))

            for segment in source_segments:
                rows = load_segment_rows(
                    segment,
                    include_phase=include_phase,
                    action_horizon=int(manifest["action_horizon"]),
                )
                demo_offset = 0 if segment["demo_name"] == "demo_0" else demo0_length
                global_indices = demo_offset + rows["raw_frame_index"]
                images = (cursor.get(int(frame_index)) for frame_index in global_indices)
                write_segment_episode(
                    dataset,
                    rows=rows,
                    images=images,
                    source_episode_id=source_id,
                    task_name=str(segment["task_name"]),
                )
                episode_count += 1
                row_count += len(rows["action"])
                valid_anchor_count += int(rows["valid_anchor"].sum())
                task_counts[str(segment["task_name"])] += int(rows["valid_anchor"].sum())

        report["splits"][split] = {
            "sources": source_ids,
            "num_sources": len(source_ids),
            "num_episodes": episode_count,
            "num_rows": row_count,
            "num_valid_anchors": valid_anchor_count,
            "valid_anchors_by_task": task_counts,
            "dataset_root": str(split_root),
        }

    with (output_root / "export_report.json").open("w") as stream:
        json.dump(report, stream, indent=2)
    return report
