#!/usr/bin/env python
"""Visual-conditioning probe: does the policy's arm output depend on the image at all?

Motivation
----------
The HRC arm skill score is negative in every context while the commanded arm
magnitude sits at ~0.5x the expert's. That signature is what you get when a
policy cannot infer *where to move* from its observation and therefore falls
back to the conditional mean. The single channel that could carry "where is the
can" is the head RGB image -- ``state23`` holds only base velocity commands,
joint angles and gripper commands (no object pose, no robot global pose), the
prompt is a fixed string per sub-instruction, and both wrist images are zeroed
with ``image_mask=False``. That one channel is encoded by a *fully frozen*
vision tower (``PathRegex(".*PaliGemma/img.*")``).

This probe measures whether that channel is actually used, without training
anything.

Design
------
``Policy.infer`` accepts an explicit ``noise`` array. With``noise`` pinned the
flow-matching ODE integration is deterministic, so a single forward pass is a
pure function of (image, state, prompt). That allows strict one-variable
swaps instead of having to average out sampling noise.

We build a pool of N anchors that share the *same* prompt but come from N
different source episodes (so the can sits somewhere else in each), then run
an N x N grid twice:

  swap_image : hold (state_i, prompt), feed image_j     -> sensitivity to vision
  swap_state : hold (image_i, prompt), feed state_j-> sensitivity to proprioception
  swap_noise : hold (image_i, state_i, prompt), vary noise -> sampling floor

and report, per action group, the dispersion of the predicted chunk across the
swapped axis.

Arm dims are compared in *delta* space (``chunk[:, DELTA] - state[DELTA]``,
matching ``audit_hrc_teacher_forced.arm_motion_norm``). Without that, swapping
the state would trivially shift the absolute output and inflate
``swap_state`` for purely coordinate reasons.

Controls
--------
A flat ``swap_image`` on its own is not conclusive, for two reasons: the
mismatched (image_i, state_j) pairs are off the training manifold, and a broken
probe (image never reaching the model) would look identical. Two controls
bound both failure modes.

1. Image degradation,``degrade_*``. The anchor's own state, prompt and noise
   are kept perfectly matched and *only* the image is replaced -- by black,
   white, gaussian noise, or a horizontal flip. Nothing here is a mismatched
   pair, so OOD pairing cannot explain a flat response. A horizontal flip is
   the sharpest of these: it moves the can from one side of the frame to the
   other, so any policy with usable spatial vision must react. If even a fully
   black image shifts the arm output less than re-rolling the sampling noise
   does, the visual channel is not being used for the arms.

2. A navigation prompt. Base velocity is the one thing the model demonstrably
   learned (base skill +0.27..+0.42; the B20 rollout navigated to within
   3.6 cm), but it need not move much *during* a manipulation sub-instruction,
   so base under a pickup prompt is the wrong place to look. Re-run this probe
   with ``--prompt "Task: Walk to the trash can and pick it up. Phase:
   ExecuteBehaviorTask."`` and read the base row: if base reacts to images
   there while arms stay flat here, the visual pathway works and the deficit is
   specific to the arms.

Note that ``base`` reacting strongly to ``swap_state`` is *not* evidence of
visual navigation: state dims 0-2 are the previous base command, so a policy
that simply continues its last command reproduces that pattern with no vision
at all.

Usage (openpi_hrc/.venv, needs GPU):
  .venv/bin/python -m scripts.hrc.probe_hrc_visual_conditioning \
      --config pi05_hrc_b_subinstr_20k \
      --checkpoint checkpoints/pi05_hrc_b_subinstr_20k/b_subinstr_seed42/20000 \
      --dataset-root /root/autodl-tmp/lerobot_datasets/hrc_subinstr_rerun_v1/train \
      --output-dir outputs/hrc_b_visual_probe
"""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.hrc.evaluate_hrc_pi05_continuous import CONTEXT_NAMES
from scripts.hrc.evaluate_hrc_pi05_continuous import _decode_source_frames
from scripts.hrc.evaluate_hrc_pi05_continuous import _load_policy
from scripts.hrc.evaluate_hrc_pi05_continuous import _read_columns

# Same dim grouping as the teacher-forced audit (config.py:522).
DELTA_DIMS = [*range(3, 7), *range(7, 14), *range(15, 22)]
BASE_DIMS = [0, 1, 2]
GRIPPER_DIMS = [14, 22]

DEFAULT_PROMPT = "Task: Pick up the soda can. Phase: ExecuteBehaviorTask."


def select_anchor_pool(
    columns: dict,
    prompts: np.ndarray,
    *,
    prompt: str,
    context_name: str,
    num_anchors: int,
    segment_fraction: float,
) -> list[dict]:
    """One anchor per source episode, taken inside that source's first matching segment.

    Anchors must share a prompt (so the swap stays single-variable) and must come
    from distinct sources (so the swapped images really do show different can
    placements). Within a source we take the first contiguous run of the prompt
    -- the approach to the first can -- and sample ``segment_fraction`` into it,
    i.e. while the arm is still reaching rather than already closed on the can.
    """
    if not 0.0 <= segment_fraction < 1.0:
        raise ValueError(f"segment_fraction must be in [0, 1), got {segment_fraction}")

    context_id = CONTEXT_NAMES.index(context_name)
    match = (
        (prompts == prompt)
        & columns["hrc.valid_anchor"]
        & (columns["hrc.context_id"] == context_id)
    )
    if not match.any():
        raise ValueError(f"no valid anchors for prompt {prompt!r} in context {context_name!r}")

    source_of = columns["hrc.source_episode_id"]
    frame_of = columns["hrc.raw_global_frame_index"]

    pool: list[dict] = []
    for source_id in sorted({int(s) for s in np.unique(source_of[match])}):
        rows = np.flatnonzero(match & (source_of == source_id))
        frames = frame_of[rows]
        # Split into contiguous runs; a gap means a different sub-instruction segment.
        breaks = np.flatnonzero(np.diff(frames) > 1)
        segment = np.split(rows, breaks + 1)[0]
        row = int(segment[int(len(segment) * segment_fraction)])
        pool.append(
            {
                "row": row,
                "source_episode_id": source_id,
                "raw_global_frame_index": int(frame_of[row]),
                "segment_length": int(len(segment)),
            }
        )
        if len(pool) >= num_anchors:
            break

    if len(pool) < 2:
        raise ValueError(f"need >=2 distinct sources for a swap probe, found {len(pool)}")
    return pool


def gather_pool_images(dataset_root: Path, pool: list[dict]) -> np.ndarray:
    """Decode one source at a time and keep only that source's single anchor frame.

    The full train split is ~86 GiB of frames against a 90 GiB container cap
    (see handoff section 9 #3), so videos are never held simultaneously.
    """
    images = []
    for order, entry in enumerate(pool, start=1):
        frames = _decode_source_frames(dataset_root, entry["source_episode_id"])
        images.append(np.array(frames[entry["raw_global_frame_index"]], dtype=np.uint8))
        print(
            f"[probe] source {entry['source_episode_id']} ({order}/{len(pool)}): "
            f"frame {entry['raw_global_frame_index']} of {len(frames)} decoded",
            flush=True,
        )
        del frames
    return np.stack(images)


def dispersion(stack: np.ndarray, dims: list[int], *, axis: int) -> float:
    """Mean over (chunk step, dim) of the std taken across ``axis``."""
    return float(np.mean(np.std(stack[..., dims], axis=axis)))


def mean_pairwise_abs(stack: np.ndarray, dims: list[int]) -> float:
    """Mean |a - b| over distinct pairs along axis 1, averaged over anchors.

    Same units as the degradation shifts below, so the two are directly
    comparable: this is the output change you get for free by re-rolling the
    sampling noise, and it is the floor any real effect has to clear.
    """
    values = stack[..., dims]
    count = values.shape[1]
    pairs = [
        float(np.mean(np.abs(values[:, a] - values[:, b])))
        for a in range(count)
        for b in range(a + 1, count)
    ]
    return float(np.mean(pairs)) if pairs else float("nan")


def image_variants(images: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Degraded copies of each anchor image, for the matched-input control.

    ``hflip`` is the informative one: it relocates the can to the mirrored side
    of the frame while leaving colour statistics untouched, so a policy that
    reads spatial layout has to respond to it. ``black`` doubles as a sanity
    reference because it is exactly what the unused wrist inputs already carry.
    """
    return {
        "original": images,
        "black": np.zeros_like(images),
        "white": np.full_like(images, 255),
        "gaussian": rng.integers(0, 256, size=images.shape, dtype=np.uint8),
        "hflip": np.ascontiguousarray(images[:, :, ::-1]),
    }


def _arm_delta(chunk: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Absolute prediction -> delta from the anchor pose, for the 18 delta dims."""
    out = np.array(chunk, dtype=np.float32)
    out[..., DELTA_DIMS] -= np.asarray(state, dtype=np.float32)[DELTA_DIMS]
    return out


def run_probe(args) -> int:
    columns, prompts = _read_columns(args.dataset_root)
    pool = select_anchor_pool(
        columns,
        prompts,
        prompt=args.prompt,
        context_name=args.context,
        num_anchors=args.num_anchors,
        segment_fraction=args.segment_fraction,
    )
    rows = [entry["row"] for entry in pool]
    states = columns["observation.state"][rows]
    images = gather_pool_images(args.dataset_root, pool)
    n = len(pool)

    from openpi.training.config import get_config

    model_config = get_config(args.config).model
    horizon, action_dim = model_config.action_horizon, model_config.action_dim
    rng = np.random.default_rng(args.seed)
    noises = rng.standard_normal((max(args.num_swap_noise, args.num_floor_noise), horizon, action_dim))
    noises = noises.astype(np.float32)

    policy = _load_policy(args.config, args.checkpoint)

    def predict(image: np.ndarray, state: np.ndarray, noise: np.ndarray) -> np.ndarray:
        result = policy.infer(
            {
                "observation/image_0": np.asarray(image, dtype=np.uint8),
                "observation/state": np.asarray(state, dtype=np.float32),
                "prompt": args.prompt,
            },
            noise=noise,
        )
        return np.asarray(result["actions"], dtype=np.float32)

    variants = image_variants(images, np.random.default_rng(args.seed + 1))
    total = (
        n * n * args.num_swap_noise * 2
        + n * args.num_floor_noise
        + n * (len(variants) - 1) * args.num_swap_noise
    )
    print(f"[probe] {n} anchors, {total} inferences", flush=True)

    # swap_image[i, j, k]: hold state_i, feed image_j -> delta vs state_i.
    swap_image = np.zeros((n, n, args.num_swap_noise, horizon, 23), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            for k in range(args.num_swap_noise):
                swap_image[i, j, k] = _arm_delta(predict(images[j], states[i], noises[k]), states[i])
        print(f"[probe] swap_image row {i + 1}/{n} done", flush=True)

    # swap_state[i, j, k]: hold image_i, feed state_j -> delta vs the state that was fed.
    swap_state = np.zeros((n, n, args.num_swap_noise, horizon, 23), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            for k in range(args.num_swap_noise):
                swap_state[i, j, k] = _arm_delta(predict(images[i], states[j], noises[k]), states[j])
        print(f"[probe] swap_state row {i + 1}/{n} done", flush=True)

    # swap_noise[i, k]: matched (image_i, state_i), vary noise -> the sampling floor.
    swap_noise = np.zeros((n, args.num_floor_noise, horizon, 23), dtype=np.float32)
    for i in range(n):
        for k in range(args.num_floor_noise):
            swap_noise[i, k] = _arm_delta(predict(images[i], states[i], noises[k]), states[i])
    print("[probe] swap_noise done", flush=True)

    # degrade[name][i, k]: matched (state_i, prompt, noise_k), image replaced by a
    # variant. Inputs stay on-manifold apart from the image itself, so this is the
    # control that OOD pairing cannot explain away.
    degrade: dict[str, np.ndarray] = {"original": swap_noise[:, : args.num_swap_noise]}
    for name, variant in variants.items():
        if name == "original":
            continue
        arr = np.zeros((n, args.num_swap_noise, horizon, 23), dtype=np.float32)
        for i in range(n):
            for k in range(args.num_swap_noise):
                arr[i, k] = _arm_delta(predict(variant[i], states[i], noises[k]), states[i])
        degrade[name] = arr
        print(f"[probe] degrade {name} done", flush=True)

    # The matched diagonal is computed three times from identical inputs; if pinned
    # noise really makes inference deterministic these must agree bit-for-bit.
    diag_image = np.stack([swap_image[i, i] for i in range(n)])
    diag_state = np.stack([swap_state[i, i] for i in range(n)])
    diag_floor = swap_noise[:, : args.num_swap_noise]
    determinism = max(
        float(np.max(np.abs(diag_image - diag_state))),
        float(np.max(np.abs(diag_image - diag_floor))),
    )

    groups = {"arms": DELTA_DIMS, "base": BASE_DIMS, "grippers": GRIPPER_DIMS}
    disp = {
        name: {
            # std across the swapped axis (j), averaged over anchors and noises.
            "swap_image": dispersion(swap_image, dims, axis=1),
            "swap_state": dispersion(swap_state, dims, axis=1),
            "swap_noise": dispersion(swap_noise, dims, axis=1),
        }
        for name, dims in groups.items()
    }

    # How far each degraded image moves the output, against the do-nothing-but-
    # re-roll-the-noise floor. Same units (mean absolute difference) throughout.
    noise_floor = {name: mean_pairwise_abs(swap_noise, dims) for name, dims in groups.items()}
    degrade_shift = {
        variant_name: {
            group_name: float(
                np.mean(np.abs(arr[..., dims] - degrade["original"][..., dims]))
            )
            for group_name, dims in groups.items()
        }
        for variant_name, arr in degrade.items()
        if variant_name != "original"
    }
    degrade_over_floor = {
        variant_name: {
            group_name: (value / noise_floor[group_name]) if noise_floor[group_name] > 0 else None
            for group_name, value in shifts.items()
        }
        for variant_name, shifts in degrade_shift.items()
    }

    # Scale references: how big is the expert's motion, and the model's own output.
    action = columns["action"]
    expert_arm_motion = float(
        np.mean(
            [
                np.mean(np.abs(_arm_delta(action[row : row + horizon], states[i])[..., DELTA_DIMS]))
                for i, row in enumerate(rows)
            ]
        )
    )
    scale = {
        "expert_arm_motion": expert_arm_motion,
        "pred_arm_motion": float(np.mean(np.abs(swap_noise[..., DELTA_DIMS]))),
        "pred_base_magnitude": float(np.mean(np.abs(swap_noise[..., BASE_DIMS]))),
    }

    def ratio(a: float, b: float) -> float | None:
        return float(a / b) if b > 0 else None

    ratios = {
        "arms_image_over_noise": ratio(disp["arms"]["swap_image"], disp["arms"]["swap_noise"]),
        "arms_state_over_noise": ratio(disp["arms"]["swap_state"], disp["arms"]["swap_noise"]),
        "arms_image_over_state": ratio(disp["arms"]["swap_image"], disp["arms"]["swap_state"]),
        "arms_image_over_expert_motion": ratio(disp["arms"]["swap_image"], expert_arm_motion),
        # Read against a navigation prompt (see module docstring) -- base need not
        # move during a manipulation sub-instruction.
        "base_image_over_noise": ratio(disp["base"]["swap_image"], disp["base"]["swap_noise"]),
        # NOT evidence of visual navigation: state dims 0-2 are the previous base command.
        "base_state_over_noise": ratio(disp["base"]["swap_state"], disp["base"]["swap_noise"]),
    }

    # Confirm the swapped inputs really differ, so a flat response cannot be an
    # artefact of feeding near-identical images or states.
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    image_l1 = [float(np.mean(np.abs(images[i].astype(np.float32) - images[j].astype(np.float32)))) for i, j in pairs]
    state_l1 = [float(np.mean(np.abs(states[i][DELTA_DIMS] - states[j][DELTA_DIMS]))) for i, j in pairs]

    per_anchor = [
        {
            **pool[i],
            "arms_swap_image": dispersion(swap_image[i], DELTA_DIMS, axis=0),
            "arms_swap_state": dispersion(swap_state[i], DELTA_DIMS, axis=0),
            "arms_swap_noise": dispersion(swap_noise[i : i + 1], DELTA_DIMS, axis=1),
            "base_swap_image": dispersion(swap_image[i], BASE_DIMS, axis=0),
            "base_swap_noise": dispersion(swap_noise[i : i + 1], BASE_DIMS, axis=1),
        }
        for i in range(n)
    ]

    report = {
        "config": args.config,
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "prompt": args.prompt,
        "context": args.context,
        "seed": args.seed,
        "num_anchors": n,
        "num_swap_noise": args.num_swap_noise,
        "num_floor_noise": args.num_floor_noise,
        "num_inferences": total,
        "sanity": {
            "determinism_max_abs_diff": determinism,
            "image_pairwise_l1_mean": float(np.mean(image_l1)),
            "image_pairwise_l1_min": float(np.min(image_l1)),
            "state_arm_pairwise_l1_mean": float(np.mean(state_l1)),
            "state_arm_pairwise_l1_min": float(np.min(state_l1)),
        },
        "dispersion": disp,
        "noise_floor": noise_floor,
        "degrade_shift": degrade_shift,
        "degrade_over_noise_floor": degrade_over_floor,
        "scale": scale,
        "ratios": ratios,
        "per_anchor": per_anchor,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "visual_conditioning_probe.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\nprompt: {args.prompt}")
    print(f"anchors: {n} distinct sources {[e['source_episode_id'] for e in pool]}")
    print(f"\ndeterminism check (pinned noise, must be 0.0): {determinism:.3e}")
    print(f"image pairwise L1: mean {np.mean(image_l1):.2f}, min {np.min(image_l1):.2f} (uint8 scale)")
    print(f"state arm pairwise L1: mean {np.mean(state_l1):.4f}, min {np.min(state_l1):.4f} (rad)")

    print(f"\n{'group':<10} {'swap_image':>12} {'swap_state':>12} {'swap_noise':>12}")
    for name in groups:
        d = disp[name]
        print(f"{name:<10} {d['swap_image']:>12.6f} {d['swap_state']:>12.6f} {d['swap_noise']:>12.6f}")

    print(f"\nexpert arm motion {expert_arm_motion:.6f} | model arm motion {scale['pred_arm_motion']:.6f}")
    print("\nratios:")
    for key, value in ratios.items():
        print(f"  {key:<32} {'n/a' if value is None else f'{value:.3f}'}")

    print("\nimage degradation, matched state/prompt/noise (x noise floor; >1 = beats resampling):")
    print(f"  {'variant':<10} {'arms':>18} {'base':>18} {'grippers':>18}")
    for variant_name, shifts in degrade_shift.items():
        cells = []
        for group_name in groups:
            over = degrade_over_floor[variant_name][group_name]
            cells.append(f"{shifts[group_name]:.6f} ({'n/a' if over is None else f'{over:.2f}x'})")
        print(f"  {variant_name:<10} " + " ".join(f"{c:>18}" for c in cells))
    print(f"  {'noise floor':<10} " + " ".join(f"{noise_floor[g]:>18.6f}" for g in groups))

    print(f"\nwrote {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--context", default="pickup", choices=CONTEXT_NAMES)
    parser.add_argument("--num-anchors", type=int, default=8)
    parser.add_argument("--num-swap-noise", type=int, default=3)
    parser.add_argument("--num-floor-noise", type=int, default=8)
    parser.add_argument("--segment-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return run_probe(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
