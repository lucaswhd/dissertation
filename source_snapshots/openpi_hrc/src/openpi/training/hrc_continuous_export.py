"""Export CLEAN sources as continuous one-episode-per-source LeRobot datasets.

Mirrors :mod:`openpi.training.hrc_export` but writes a single continuous episode
per source (merged demo_0 + demo_1, four instruction contexts) instead of three
task segments. ``hrc.task_id`` carries the 4-way context id so the existing
loader / E0 stratified selector work unchanged; the privileged 28-d human state
is intentionally dropped (main model must not see human coordinates).
"""

from collections.abc import Iterable, Mapping
import json
from pathlib import Path
import shutil
from typing import Protocol

import h5py
import numpy as np

from openpi.training import hrc_continuous_data as hcd
from openpi.training.hrc_export import decode_rgb_frames

# rows key -> HDF5 dataset path under data/demo_N.
_HDF5_KEYS = {
    "action": "action",
    "joint_positions": "obs/joint_positions",
    "gripper_state": "obs/gripper_state",
    "fsm_state_id": "obs/fsm_state_id",
}
MANUAL_QC_SOURCES = (25, 36)
_BASE_DATASET_NAME = "hrc_pi05_v0"


class EpisodeWriter(Protocol):
    def add_frame(self, frame: dict) -> None: ...

    def save_episode(self) -> None: ...


def continuous_lerobot_features(*, image_height: int = 256, image_width: int = 352) -> dict[str, dict]:
    scalar = {"shape": (1,), "names": ["value"]}
    return {
        "observation.rgb.zed_link_camera_0": {
            "dtype": "video",
            "shape": (image_height, image_width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {"dtype": "float32", "shape": (23,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (23,), "names": ["action"]},
        "hrc.valid_anchor": {"dtype": "bool", **scalar},
        "hrc.task_id": {"dtype": "int64", **scalar},
        "hrc.context_id": {"dtype": "int64", **scalar},
        "hrc.source_episode_id": {"dtype": "int64", **scalar},
        "hrc.fsm_state_id": {"dtype": "int64", **scalar},
        "hrc.raw_global_frame_index": {"dtype": "int64", **scalar},
        "hrc.raw_demo_id": {"dtype": "int64", **scalar},
        "hrc.raw_demo_frame_index": {"dtype": "int64", **scalar},
    }


def write_continuous_episode(
    writer: EpisodeWriter,
    *,
    rows: Mapping[str, np.ndarray],
    images: Iterable[np.ndarray],
    source_episode_id: int,
) -> None:
    """Write one continuous episode; ``task_id`` == per-frame context id."""

    num_rows = len(rows["action"])
    if any(len(rows[key]) != num_rows for key in rows):
        raise ValueError("All exported row arrays must have equal length")

    image_iterator = iter(images)
    for index in range(num_rows):
        try:
            image = next(image_iterator)
        except StopIteration as exc:
            raise ValueError(f"Expected {num_rows} images, got {index}") from exc
        context_id = int(rows["context_id"][index])
        writer.add_frame(
            {
                "observation.rgb.zed_link_camera_0": np.asarray(image, dtype=np.uint8),
                "observation.state": np.asarray(rows["state"][index], dtype=np.float32),
                "action": np.asarray(rows["action"][index], dtype=np.float32),
                "hrc.valid_anchor": np.asarray([rows["valid_anchor"][index]], dtype=np.bool_),
                "hrc.task_id": np.asarray([context_id], dtype=np.int64),
                "hrc.context_id": np.asarray([context_id], dtype=np.int64),
                "hrc.source_episode_id": np.asarray([source_episode_id], dtype=np.int64),
                "hrc.fsm_state_id": np.asarray([rows["fsm_state_id"][index]], dtype=np.int64),
                "hrc.raw_global_frame_index": np.asarray(
                    [rows["raw_global_frame_index"][index]], dtype=np.int64
                ),
                "hrc.raw_demo_id": np.asarray([rows["raw_demo_id"][index]], dtype=np.int64),
                "hrc.raw_demo_frame_index": np.asarray(
                    [rows["raw_demo_frame_index"][index]], dtype=np.int64
                ),
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


def load_continuous_rows(source: Mapping[str, object], *, action_horizon: int) -> dict[str, np.ndarray]:
    """Read demo_0 + demo_1 obs from HDF5 and build the continuous rows."""

    commands = {
        int(command["round_id"]): command
        for command in hcd._load_json(source["commands_path"])  # noqa: SLF001
    }
    with h5py.File(str(source["hdf5_path"]), "r") as hdf5_file:
        demo0 = {key: hdf5_file[f"data/demo_0/{path}"][:] for key, path in _HDF5_KEYS.items()}
        demo1 = {key: hdf5_file[f"data/demo_1/{path}"][:] for key, path in _HDF5_KEYS.items()}
    return hcd.build_continuous_rows(
        demo0=demo0, demo1=demo1, commands=commands, action_horizon=action_horizon
    )


def require_manual_qc_confirmation(
    manual_qc_path: Path | str | None, sources: Iterable[int]
) -> None:
    """Refuse to export possibly-contaminated sources without human video review.

    ``sources`` (e.g. 25/36) require an explicit confirmation file marking each
    ``source_<id>`` as ``PASS``; otherwise export stops with a review prompt.
    """

    sources = sorted(int(source) for source in sources)
    if not sources:
        return
    if manual_qc_path is None or not Path(manual_qc_path).exists():
        raise ValueError(f"等待人工视频复核 (source {sources}): 需提供含 source_<id>=PASS 的 --manual-qc 确认文件")
    confirmation = hcd._load_json(manual_qc_path)  # noqa: SLF001
    missing = [source for source in sources if confirmation.get(f"source_{source}") != "PASS"]
    if missing:
        raise ValueError(f"人工复核未通过/缺失: source {missing} 在 {manual_qc_path} 中未标记为 PASS")


def export_continuous_manifest(
    manifest: Mapping[str, object],
    output_root: Path | str,
    *,
    overwrite: bool = False,
    manual_qc_path: Path | str | None = None,
) -> dict:
    """Write one continuous LeRobot episode per CLEAN source, split into train/val."""

    output_root = Path(output_root).resolve()
    if output_root.name == _BASE_DATASET_NAME:
        raise ValueError(f"refusing to write into the base dataset {_BASE_DATASET_NAME!r}")

    sources = list(manifest["sources"])
    present = {int(source["source_episode_id"]) for source in sources}
    require_manual_qc_confirmation(
        manual_qc_path, [source for source in MANUAL_QC_SOURCES if source in present]
    )

    from lerobot.datasets import LeRobotDataset

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    with (output_root / "manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=2)

    action_horizon = int(manifest["action_horizon"])
    report = {"schema_version": manifest["schema_version"], "splits": {}}
    for split in ("train", "val", "test"):
        split_sources = sorted(
            (source for source in sources if source["split"] == split),
            key=lambda source: int(source["source_episode_id"]),
        )
        if not split_sources:
            continue
        split_root = output_root / split
        dataset = LeRobotDataset.create(
            repo_id=f"hrc/{output_root.name}_{split}",
            root=split_root,
            robot_type="R1Pro",
            fps=30,
            features=continuous_lerobot_features(image_height=256, image_width=352),
            use_videos=True,
            video_backend="pyav",
            image_writer_threads=4,
        )
        episode_count = 0
        row_count = 0
        valid_anchor_count = 0
        context_counts = dict.fromkeys(hcd.CONTEXT_NAMES, 0)
        for source in split_sources:
            source_id = int(source["source_episode_id"])
            rows = load_continuous_rows(source, action_horizon=action_horizon)
            images = decode_rgb_frames(Path(str(source["head_video_path"])))
            write_continuous_episode(dataset, rows=rows, images=images, source_episode_id=source_id)
            episode_count += 1
            row_count += len(rows["action"])
            valid_anchor_count += int(rows["valid_anchor"].sum())
            for index, name in enumerate(hcd.CONTEXT_NAMES):
                context_counts[name] += int(((rows["context_id"] == index) & rows["valid_anchor"]).sum())

        report["splits"][split] = {
            "sources": [int(source["source_episode_id"]) for source in split_sources],
            "num_sources": len(split_sources),
            "num_episodes": episode_count,
            "num_rows": row_count,
            "num_valid_anchors": valid_anchor_count,
            "valid_anchors_by_context": context_counts,
            "dataset_root": str(split_root),
        }

    with (output_root / "export_report.json").open("w") as stream:
        json.dump(report, stream, indent=2)
    return report
