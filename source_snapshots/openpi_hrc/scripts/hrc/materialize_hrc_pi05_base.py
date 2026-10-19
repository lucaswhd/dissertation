#!/usr/bin/env python3
"""Materialize the fair, zero-update HRC π0.5 initialization checkpoint.

Official ``pi05_base/params`` is only a partial parameter source for the HRC
LoRA graph and has no HRC normalization assets.  This command builds the exact
``pi05_hrc_e4_main`` graph, merges the official base weights through the normal
training weight loader, performs zero optimizer updates, and saves step 0 with
the same train-only HRC norm stats.  Gate 3 should call this checkpoint the
"HRC initialization" or "pi05_base+HRC interface", not a trained HRC model.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path


DEFAULT_BASE_PARAMS = Path(
    "/root/autodl-tmp/.cache/openpi/openpi-assets/checkpoints/pi05_base/params"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_hrc_e4_main")
    parser.add_argument("--exp-name", default="hrc_base_init_seed42")
    parser.add_argument("--checkpoint-base-dir", default="./checkpoints")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from openpi.training.config import get_config

    original = get_config(args.config)
    config = dataclasses.replace(
        original,
        exp_name=args.exp_name,
        checkpoint_base_dir=args.checkpoint_base_dir,
        overwrite=args.overwrite,
        resume=False,
        wandb_enabled=False,
        num_workers=0,
    )
    report = {
        "config": config.name,
        "seed": config.seed,
        "official_base_params": str(DEFAULT_BASE_PARAMS),
        "output_checkpoint": str(config.checkpoint_dir / "0"),
        "optimizer_updates": 0,
        "norm_stats": "train-only HRC stats from the selected data config",
    }
    print(json.dumps(report, indent=2))
    if args.dry_run:
        return 0
    if not DEFAULT_BASE_PARAMS.exists():
        raise FileNotFoundError(DEFAULT_BASE_PARAMS)

    import jax

    from openpi.training import checkpoints
    from openpi.training import data_loader
    from openpi.training import sharding
    from scripts.b1k import train_b1k

    mesh = sharding.make_mesh(config.fsdp_devices)
    loader = data_loader.create_b1k_data_loader(
        config,
        shuffle=False,
        num_batches=1,
        skip_norm_stats=False,
    )
    rng = jax.random.key(config.seed)
    _, init_rng = jax.random.split(rng)
    state, _ = train_b1k.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    if int(state.step) != 0:
        raise RuntimeError(f"expected zero-update state, got step={int(state.step)}")

    manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=False,
    )
    if resuming:
        raise RuntimeError("base materialization must not resume")
    checkpoints.save_state(manager, state, loader, 0)
    manager.wait_until_finished()
    if not (config.checkpoint_dir / "0" / "params").exists():
        raise RuntimeError("checkpoint save finished without params")
    print(f"MATERIALIZED {config.checkpoint_dir / '0'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
