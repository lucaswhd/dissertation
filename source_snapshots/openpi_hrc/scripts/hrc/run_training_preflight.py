"""Run real pi0.5 forward/backward steps and enforce the 4090 memory gate."""

import argparse
import dataclasses
import functools
import json
import os
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from openpi.training import config as _config
from openpi.training import data_loader
from openpi.training import sharding
from openpi.training.hrc_preflight import validate_model_batch
from scripts.b1k import train_b1k


def _gpu_process_memory_mib() -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    )
    current_pid = os.getpid()
    for line in output.splitlines():
        pid, memory = (part.strip() for part in line.split(",", maxsplit=1))
        if int(pid) == current_pid:
            return int(memory)
    return 0


def _audit_trainable_tree(config: _config.TrainConfig, state) -> dict[str, int]:
    trainable = state.params.filter(config.trainable_filter)
    paths = ["/".join(str(part) for part in path) for path, _ in trainable.flat_state().items()]
    forbidden = [
        path
        for path in paths
        if "PaliGemma/img" in path or ("PaliGemma/llm" in path and "lora" not in path.lower())
    ]
    if forbidden:
        raise ValueError(f"frozen SigLIP/Gemma base parameters remain trainable: {forbidden[:5]}")
    required_fragments = ("lora", "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")
    missing = [fragment for fragment in required_fragments if not any(fragment in path for path in paths)]
    if missing:
        raise ValueError(f"required trainable parameter groups are missing: {missing}")
    parameter_count = sum(int(np.prod(variable.value.shape)) for variable in trainable.flat_state().values())
    return {"trainable_leaves": len(paths), "trainable_parameters": parameter_count}


def main(
    config_name: str,
    physical_batch_size: int,
    *,
    steps: int = 10,
    max_memory_mib: int = 23 * 1024,
    dataset_root: Path | None = None,
    skip_norm_stats: bool = False,
) -> None:
    if physical_batch_size not in (1, 2):
        raise ValueError("physical_batch_size must be 1 or 2")
    if steps < 1:
        raise ValueError("steps must be positive")
    config = _config.get_config(config_name)
    if dataset_root is not None:
        config = dataclasses.replace(
            config,
            data=dataclasses.replace(
                config.data,
                dataset_root=str(dataset_root),
                manifest_path=str(dataset_root.parent / "manifest.json"),
            ),
        )
    config = dataclasses.replace(
        config,
        batch_size=physical_batch_size,
        grad_accum_steps=1,
        num_workers=0,
        wandb_enabled=False,
    )

    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", "/root/autodl-tmp/.cache/jax")
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    accumulated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(None, sharding.DATA_AXIS)
    )
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    loader = data_loader.create_b1k_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        num_batches=steps,
        skip_norm_stats=skip_norm_stats,
    )
    iterator = iter(loader)
    first_observation, first_actions = next(iterator)
    shape_report = validate_model_batch(first_observation, first_actions)

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    state, state_sharding = train_b1k.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    tree_report = _audit_trainable_tree(config, state)
    train_step = jax.jit(
        functools.partial(train_b1k.train_step, config),
        in_shardings=(replicated, state_sharding, accumulated_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )

    losses = []
    max_observed_memory = _gpu_process_memory_mib()
    batch = (first_observation, first_actions)
    for step in range(steps):
        if step > 0:
            batch = next(iterator)
        accumulated_batch = jax.tree.map(lambda value: jnp.expand_dims(value, axis=0), batch)
        with sharding.set_mesh(mesh):
            state, info = train_step(train_rng, state, accumulated_batch)
        jax.block_until_ready(state)
        loss = float(info["loss"])
        if not np.isfinite(loss):
            raise ValueError(f"non-finite loss at preflight step {step}: {loss}")
        losses.append(loss)
        max_observed_memory = max(max_observed_memory, _gpu_process_memory_mib())
        if max_observed_memory > max_memory_mib:
            raise MemoryError(
                f"GPU process memory {max_observed_memory} MiB exceeds gate {max_memory_mib} MiB"
            )

    print(
        json.dumps(
            {
                "config": config_name,
                "physical_batch_size": physical_batch_size,
                "steps": steps,
                "final_optimizer_step": int(state.step),
                "max_gpu_memory_mib": max_observed_memory,
                "losses": losses,
                "shape_report": shape_report,
                "trainable_tree": tree_report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pi05_hrc_e4_main")
    parser.add_argument("--physical-batch-size", type=int, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max-memory-mib", type=int, default=23 * 1024)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--skip-norm-stats", action="store_true")
    args = parser.parse_args()
    main(
        args.config_name,
        args.physical_batch_size,
        steps=args.steps,
        max_memory_mib=args.max_memory_mib,
        dataset_root=args.dataset_root,
        skip_norm_stats=args.skip_norm_stats,
    )
