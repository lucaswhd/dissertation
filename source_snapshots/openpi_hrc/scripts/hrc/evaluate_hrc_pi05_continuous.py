#!/usr/bin/env python
"""Offline teacher-forced evaluation of the continuous π0.5 pilot checkpoint.

Two modes (both feed RECORDED observations, so this is NOT closed-loop success):
  metrics : per context, sample up to N valid anchors, run policy.infer on the
            anchor observation and compare the predicted 32-step chunk to the
            expert action rows -> per-group MAE (base / arms / grippers).
  render  : for one val source, re-predict every ``--replan-every`` frames on the
            recorded observations and write a head-RGB comparison video plus the
            predicted/expert action arrays.

Usage (openpi_hrc/.venv, needs GPU):
  .venv/bin/python -m scripts.hrc.evaluate_hrc_pi05_continuous \
      --config pi05_hrc_e5_continuous_main \
      --checkpoint checkpoints/pi05_hrc_e5_continuous_main/hrc_e5_continuous_seed42/5000 \
      --dataset-root /root/autodl-tmp/lerobot_datasets/hrc_pi05_continuous_clean_v0/val \
      --mode metrics --max-anchors-per-context 300 --seed 42 \
      --output-dir outputs/hrc_e5_continuous
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ACTION_DIM = 23
# 23-dim R1Pro physical action groups (base velocity, binary grippers, arm/torso).
ACTION_GROUPS = {
    "base": (0, 1, 2),
    "left_gripper": (14,),
    "right_gripper": (22,),
    "arms": tuple(i for i in range(ACTION_DIM) if i not in {0, 1, 2, 14, 22}),
}
CONTEXT_NAMES = ("wait", "pickup", "follow", "radio")


def action_group_mae(predicted: np.ndarray, expert: np.ndarray) -> dict[str, float]:
    """Mean absolute error per action group over a [T, 23] chunk (or [23])."""

    predicted = np.asarray(predicted, dtype=np.float32)
    expert = np.asarray(expert, dtype=np.float32)
    if predicted.shape != expert.shape or predicted.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected matching [...,23] chunks, got {predicted.shape} vs {expert.shape}")
    return {
        name: float(np.mean(np.abs(predicted[..., list(idx)] - expert[..., list(idx)])))
        for name, idx in ACTION_GROUPS.items()
    }


def sample_anchors_per_context(
    context_ids: np.ndarray,
    valid_mask: np.ndarray,
    *,
    max_per_context: int,
    seed: int,
) -> dict[int, np.ndarray]:
    """Per context, the valid anchor row indices, uniformly capped at N (seeded)."""

    if max_per_context <= 0:
        raise ValueError("max_per_context must be positive")
    context_ids = np.asarray(context_ids, dtype=np.int64)
    valid = np.asarray(valid_mask, dtype=bool)
    rng = np.random.default_rng(seed)
    sampled: dict[int, np.ndarray] = {}
    for context in range(len(CONTEXT_NAMES)):
        idx = np.flatnonzero((context_ids == context) & valid)
        if len(idx) > max_per_context:
            idx = np.sort(rng.choice(idx, size=max_per_context, replace=False))
        sampled[context] = idx
    return sampled


def aggregate_group_mae(per_chunk: list[dict[str, float]]) -> dict[str, float]:
    """Average per-group MAE across a list of per-chunk MAE dicts."""

    if not per_chunk:
        return {name: float("nan") for name in ACTION_GROUPS}
    return {
        name: float(np.mean([chunk[name] for chunk in per_chunk])) for name in ACTION_GROUPS
    }


# --------------------------------------------------------------------------- #
# Everything below needs the dataset + GPU policy; kept out of the pure core.
# --------------------------------------------------------------------------- #
def _read_columns(split_root: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(split_root / "data" / "**" / "*.parquet"), recursive=True))
    table = pa.concat_tables([pq.read_table(f) for f in files])
    data = table.to_pydict()
    order = np.argsort(np.asarray(data["index"], dtype=np.int64))
    columns = {
        "observation.state": np.asarray(data["observation.state"], dtype=np.float32)[order],
        "action": np.asarray(data["action"], dtype=np.float32)[order],
        "hrc.valid_anchor": np.asarray(data["hrc.valid_anchor"], dtype=bool)[order],
        "hrc.context_id": np.asarray(data["hrc.context_id"], dtype=np.int64)[order],
        "hrc.source_episode_id": np.asarray(data["hrc.source_episode_id"], dtype=np.int64)[order],
        "hrc.fsm_state_id": np.asarray(data["hrc.fsm_state_id"], dtype=np.int64)[order],
        "hrc.raw_global_frame_index": np.asarray(data["hrc.raw_global_frame_index"], dtype=np.int64)[order],
        "episode_index": np.asarray(data["episode_index"], dtype=np.int64)[order],
    }
    tasks = pq.read_table(split_root / "meta" / "tasks.parquet").to_pydict()
    index_to_task = {int(i): str(t) for i, t in zip(tasks["task_index"], tasks["task"], strict=True)}
    prompts = np.array([index_to_task[int(ti)] for ti in np.asarray(data["task_index"])[order]], dtype=object)
    return columns, prompts


def _decode_source_frames(dataset_root: Path, source_id: int) -> np.ndarray:
    """Decode one source's full head video into an [N, H, W, 3] uint8 array."""
    import av

    manifest = json.loads((dataset_root.parent / "manifest.json").read_text(encoding="utf-8"))
    source = next(s for s in manifest["sources"] if int(s["source_episode_id"]) == source_id)
    with av.open(str(source["head_video_path"])) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(container.streams.video[0])]
    return np.asarray(frames, dtype=np.uint8)


def _load_policy(config_name: str, checkpoint: Path):
    from openpi.policies.policy_config import create_trained_policy
    from openpi.training.config import get_config

    return create_trained_policy(get_config(config_name), checkpoint)


def _predict_chunk(policy, image: np.ndarray, state: np.ndarray, prompt: str) -> np.ndarray:
    result = policy.infer(
        {
            "observation/image_0": np.asarray(image, dtype=np.uint8),
            "observation/state": np.asarray(state, dtype=np.float32),
            "prompt": str(prompt),
        }
    )
    return np.asarray(result["actions"], dtype=np.float32)


def run_metrics(args) -> int:
    columns, prompts = _read_columns(args.dataset_root)
    sampled = sample_anchors_per_context(
        columns["hrc.context_id"], columns["hrc.valid_anchor"],
        max_per_context=args.max_anchors_per_context, seed=args.seed,
    )
    policy = _load_policy(args.config, args.checkpoint)

    # Decode each source's frames once and serve its sampled anchors from memory.
    source_ids = sorted({int(s) for s in np.unique(columns["hrc.source_episode_id"])})
    frames_by_source = {sid: _decode_source_frames(args.dataset_root, sid) for sid in source_ids}

    per_context_chunks: dict[int, list[dict[str, float]]] = {c: [] for c in sampled}
    for context, anchors in sampled.items():
        for row in anchors:
            source_id = int(columns["hrc.source_episode_id"][row])
            frame_index = int(columns["hrc.raw_global_frame_index"][row])
            image = frames_by_source[source_id][frame_index]
            predicted = _predict_chunk(policy, image, columns["observation.state"][row], prompts[row])
            expert = columns["action"][row : row + predicted.shape[0]]
            per_context_chunks[context].append(action_group_mae(predicted[: len(expert)], expert))

    report = {
        "config": args.config,
        "checkpoint": str(args.checkpoint),
        "max_anchors_per_context": args.max_anchors_per_context,
        "by_context": {
            CONTEXT_NAMES[c]: {
                "num_anchors": len(per_context_chunks[c]),
                "group_mae": aggregate_group_mae(per_context_chunks[c]),
            }
            for c in sampled
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "offline_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["by_context"], indent=2, ensure_ascii=False), flush=True)
    return 0


def run_render(args) -> int:
    import av

    from openpi.training.hrc_data import FSM_STATE_NAMES

    columns, prompts = _read_columns(args.dataset_root)
    rows = np.flatnonzero(columns["hrc.source_episode_id"] == args.source_id)
    rows = rows[np.argsort(columns["hrc.raw_global_frame_index"][rows])]
    frames = _decode_source_frames(args.dataset_root, args.source_id)
    policy = _load_policy(args.config, args.checkpoint)

    n = len(rows)
    predicted_rows = np.zeros((n, ACTION_DIM), dtype=np.float32)
    chunk = None
    for local in range(n):
        if local % args.replan_every == 0:
            row = rows[local]
            frame_index = int(columns["hrc.raw_global_frame_index"][row])
            chunk = _predict_chunk(policy, frames[frame_index], columns["observation.state"][row], prompts[row])
        predicted_rows[local] = chunk[min(local % args.replan_every, len(chunk) - 1)]

    expert_rows = columns["action"][rows]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "anchor_indices.npy", rows)
    np.save(out / "predicted_actions.npy", predicted_rows)
    np.save(out / "expert_actions.npy", expert_rows)

    cumulative = {name: [] for name in ACTION_GROUPS}
    with av.open(str(out / "teacher_forced_comparison.mp4"), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.height, stream.width, stream.pix_fmt = frames.shape[1], frames.shape[2], "yuv420p"
        for local in range(n):
            mae = action_group_mae(predicted_rows[local], expert_rows[local])
            for name in ACTION_GROUPS:
                cumulative[name].append(mae[name])
            frame = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(frames[int(columns["hrc.raw_global_frame_index"][rows[local]])]),
                format="rgb24",
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    metrics = {
        "source_id": args.source_id,
        "num_frames": n,
        "replan_every": args.replan_every,
        "group_mae": {name: float(np.mean(cumulative[name])) for name in ACTION_GROUPS},
        "fsm_states_seen": sorted({FSM_STATE_NAMES[int(f)] for f in columns["hrc.fsm_state_id"][rows]}),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("metrics", "render"), required=True)
    parser.add_argument("--max-anchors-per-context", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-id", type=int)
    parser.add_argument("--replan-every", type=int, default=8)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.mode == "metrics":
        return run_metrics(args)
    if args.source_id is None:
        raise ValueError("--source-id is required for render mode")
    return run_render(args)


if __name__ == "__main__":
    raise SystemExit(main())
