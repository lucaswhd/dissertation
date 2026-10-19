"""Compute train-only HRC mean/std without decoding RGB videos."""

import argparse
import json
from pathlib import Path

import numpy as np

from openpi.shared import normalize
from openpi.training import config as _config
from openpi.training import hrc_norm_stats
from openpi.training import lerobot_compat

DELTA_MAPPINGS = (
    (tuple(range(3, 7)), tuple(range(3, 7))),
    (tuple(range(7, 14)), tuple(range(7, 14))),
    (tuple(range(15, 22)), tuple(range(15, 22))),
)


def _load_numpy_columns(dataset_root: Path, repo_id: str, state_key: str) -> dict[str, np.ndarray]:
    dataset = lerobot_compat.LeRobotDataset(repo_id=repo_id, root=dataset_root)
    table = dataset.hf_dataset.with_format("numpy")
    keys = (state_key, "action", "hrc.valid_anchor", "hrc.task_id", "hrc.source_episode_id", "episode_index")
    return {key: np.asarray(table[key]) for key in keys}


def _validate_train_only(factory: _config.LeRobotHrcDataConfig, columns: dict[str, np.ndarray]) -> None:
    if factory.split != "train":
        raise ValueError("HRC normalization statistics may only be computed from split='train'")
    manifest_path = Path(factory.manifest_path)
    manifest = json.loads(manifest_path.read_text())
    train_sources = {int(value) for value in manifest["source_splits"]["train"]}
    val_test_sources = {
        int(value) for split in ("val", "test") for value in manifest["source_splits"][split]
    }
    observed_sources = {int(value) for value in np.unique(columns["hrc.source_episode_id"])}
    if not observed_sources or not observed_sources.issubset(train_sources):
        raise ValueError("dataset contains sources outside the manifest train split")
    if observed_sources & val_test_sources:
        raise ValueError("validation/test source leakage detected in normalization data")


def main(config_name: str, *, overwrite: bool = False) -> None:
    train_config = _config.get_config(config_name)
    factory = train_config.data
    if not isinstance(factory, _config.LeRobotHrcDataConfig):
        raise TypeError(f"config {config_name!r} is not an HRC LeRobot config")

    dataset_root = Path(factory.dataset_root)
    columns = _load_numpy_columns(dataset_root, str(factory.repo_id), factory.state_key)
    _validate_train_only(factory, columns)

    valid_indices = np.flatnonzero(columns["hrc.valid_anchor"])
    horizon = train_config.model.action_horizon
    episode_ids = columns["episode_index"]
    if np.any(episode_ids[valid_indices] != episode_ids[valid_indices + horizon - 1]):
        raise ValueError("valid HRC action chunks cross an episode boundary")

    stats = hrc_norm_stats.compute_hrc_norm_stats_from_arrays(
        states=columns[factory.state_key],
        actions=columns["action"],
        valid_mask=columns["hrc.valid_anchor"],
        task_ids=columns["hrc.task_id"],
        action_horizon=horizon,
        delta_mappings=DELTA_MAPPINGS,
        task_weights=factory.task_weights,
    )

    asset_id = factory.assets.asset_id
    assets_dir = factory.assets.assets_dir
    if not asset_id or not assets_dir:
        raise ValueError("HRC config must define an explicit norm stats assets_dir and asset_id")
    output_dir = Path(assets_dir) / asset_id
    output_path = output_dir / "norm_stats.json"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output_path}; pass --overwrite")
    normalize.save(output_dir, stats)

    valid_task_ids = columns["hrc.task_id"][valid_indices]
    # Support 4 continuous context ids while keeping the legacy 3-task output shape.
    counts = np.bincount(valid_task_ids, minlength=max(3, int(valid_task_ids.max()) + 1))
    print(
        json.dumps(
            {
                "config": config_name,
                "dataset_root": str(dataset_root),
                "num_valid_anchors": len(valid_indices),
                "valid_anchors_by_task": counts.tolist(),
                "task_weights": factory.task_weights,
                "state_dim": int(columns[factory.state_key].shape[-1]),
                "action_dim": int(columns["action"].shape[-1]),
                "output": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.config_name, overwrite=args.overwrite)
