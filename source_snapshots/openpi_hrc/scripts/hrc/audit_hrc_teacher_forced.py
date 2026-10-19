#!/usr/bin/env python
"""Teacher-forced audit of an HRC π0.5 checkpoint, scored against a null predictor.

Raw teacher-forced MAE is not interpretable on this data: the 18 arm/torso action
dims are learned as deltas from the anchor state, and those deltas average ~0.02 rad,
so "predict no movement" already scores ~0.02 MAE. This audit therefore reports a
skill score against an explicit do-nothing baseline

    skill = 1 - MAE(model) / MAE(null)

where null = hold the current joint angles (zero arm delta), zero base velocity, and
keep the previous gripper command. skill <= 0 means the model is no better than not
moving, whatever the absolute MAE looks like.

It also reports, per context and per pickup sub-stage (segmented at expert gripper
flips): commanded arm-motion magnitude vs the expert's, gripper sign accuracy, gripper
flip recall, and across-seed sample spread (to separate mode collapse from a model
that is multi-modal but unconditioned).

Usage (openpi_hrc/.venv, needs GPU):
  .venv/bin/python -m scripts.hrc.audit_hrc_teacher_forced \
      --config pi05_hrc_e6_rerun_20k \
      --checkpoint checkpoints/pi05_hrc_e6_rerun_20k/hrc_e6_rerun_seed42/20000 \
      --dataset-root /root/autodl-tmp/lerobot_datasets/hrc_pi05_continuous_rerun_v1/train \
      --max-anchors-per-context 200 --seed 42 \
      --output-dir outputs/hrc_e6_audit_train
"""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.hrc.evaluate_hrc_pi05_continuous import ACTION_DIM
from scripts.hrc.evaluate_hrc_pi05_continuous import ACTION_GROUPS
from scripts.hrc.evaluate_hrc_pi05_continuous import CONTEXT_NAMES
from scripts.hrc.evaluate_hrc_pi05_continuous import _decode_source_frames
from scripts.hrc.evaluate_hrc_pi05_continuous import _load_policy
from scripts.hrc.evaluate_hrc_pi05_continuous import _predict_chunk
from scripts.hrc.evaluate_hrc_pi05_continuous import _read_columns
from scripts.hrc.evaluate_hrc_pi05_continuous import sample_anchors_per_context

# Action dims whose label is a delta from the same-index state dim (config.py:521).
DELTA_DIMS = [*range(3, 7), *range(7, 14), *range(15, 22)]
BASE_DIMS = [0, 1, 2]
GRIPPER_DIMS = [14, 22]


def null_chunk(state: np.ndarray, previous_action: np.ndarray, horizon: int) -> np.ndarray:
    """The do-nothing chunk in absolute action space: hold pose, no base motion."""
    row = np.zeros(ACTION_DIM, dtype=np.float32)
    row[DELTA_DIMS] = np.asarray(state, dtype=np.float32)[DELTA_DIMS]
    row[BASE_DIMS] = 0.0
    row[GRIPPER_DIMS] = np.asarray(previous_action, dtype=np.float32)[GRIPPER_DIMS]
    return np.repeat(row[None, :], horizon, axis=0)


def group_mae(predicted: np.ndarray, expert: np.ndarray) -> dict[str, float]:
    return {
        name: float(np.mean(np.abs(predicted[..., list(idx)] - expert[..., list(idx)])))
        for name, idx in ACTION_GROUPS.items()
    }


def arm_motion_norm(chunk: np.ndarray, state: np.ndarray) -> float:
    """Mean |commanded arm displacement from the current pose| over the chunk."""
    return float(np.mean(np.abs(chunk[..., DELTA_DIMS] - np.asarray(state)[DELTA_DIMS])))


def _flip_events(sign_rows: np.ndarray, prev_sign: np.ndarray) -> list[tuple[int, int, float]]:
    """(frame, gripper index, new sign) for every sign change in a chunk."""
    previous = np.vstack([prev_sign[None, :], sign_rows[:-1]])
    changed = np.argwhere(sign_rows != previous)
    return [(int(t), int(g), float(sign_rows[t, g])) for t, g in changed]


def gripper_stats(
    predicted: np.ndarray, expert: np.ndarray, previous: np.ndarray, *, tolerance: int = 8
) -> dict:
    """Sign accuracy plus matched precision/recall over gripper flip events.

    Recall alone is gameable: a policy that toggles the gripper every frame would
    "recall" every expert flip. Precision is therefore reported from the same
    matching -- a predicted flip counts only if a real flip of the same gripper in
    the same direction happens within ``tolerance`` frames.

    ``flip_recall_lenient`` keeps the earlier, looser definition (did the chunk
    leave the previous sign at or before the expert flip) so numbers already
    reported against it stay comparable.
    """
    pred_sign = np.sign(predicted[:, GRIPPER_DIMS])
    exp_sign = np.sign(expert[:, GRIPPER_DIMS])
    accuracy = float(np.mean(pred_sign == exp_sign))
    prev_sign = np.sign(np.asarray(previous, dtype=np.float32)[GRIPPER_DIMS])

    expert_events = _flip_events(exp_sign, prev_sign)
    pred_events = _flip_events(pred_sign, prev_sign)

    unmatched = list(expert_events)
    true_positives = 0
    for frame, gripper, sign in pred_events:
        for index, (e_frame, e_gripper, e_sign) in enumerate(unmatched):
            if e_gripper == gripper and e_sign == sign and abs(e_frame - frame) <= tolerance:
                true_positives += 1
                unmatched.pop(index)
                break

    precision = true_positives / len(pred_events) if pred_events else None
    recall = true_positives / len(expert_events) if expert_events else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    lenient = None
    if expert_events:
        flips = exp_sign != np.vstack([prev_sign[None, :], exp_sign[:-1]])
        pred_left_prev = np.cumsum(pred_sign != prev_sign[None, :], axis=0) > 0
        lenient = float(pred_left_prev[flips].mean())

    return {
        "sign_accuracy": accuracy,
        "expert_flips": len(expert_events),
        "predicted_flips": len(pred_events),
        "true_positives": true_positives,
        "flip_precision": precision,
        "flip_recall": recall,
        "flip_f1": f1,
        "flip_recall_lenient": lenient,
        "tolerance": tolerance,
    }


def pickup_substage_ids(action: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Sub-stage index per row, incremented at every expert gripper sign change."""
    signs = np.sign(action[rows][:, GRIPPER_DIMS])
    changed = np.any(signs[1:] != signs[:-1], axis=1)
    return np.concatenate([[0], np.cumsum(changed)]).astype(np.int64)


def _summarize(records: list[dict]) -> dict:
    if not records:
        return {}
    out: dict = {"num_anchors": len(records)}
    for name in ACTION_GROUPS:
        model = float(np.mean([r["mae_model"][name] for r in records]))
        null = float(np.mean([r["mae_null"][name] for r in records]))
        out[name] = {
            "mae_model": model,
            "mae_null": null,
            "skill": float(1.0 - model / null) if null > 0 else None,
        }
    out["arm_motion_model"] = float(np.mean([r["arm_motion_model"] for r in records]))
    out["arm_motion_expert"] = float(np.mean([r["arm_motion_expert"] for r in records]))
    out["arm_motion_ratio"] = (
        out["arm_motion_model"] / out["arm_motion_expert"] if out["arm_motion_expert"] > 0 else None
    )
    out["gripper_sign_accuracy"] = float(np.mean([r["gripper"]["sign_accuracy"] for r in records]))
    # Pool the event counts rather than averaging per-chunk ratios: most chunks hold
    # zero or one flip, so a mean of per-chunk precisions would be dominated by chunks
    # that contain a single event.
    expert_flips = sum(r["gripper"]["expert_flips"] for r in records)
    predicted_flips = sum(r["gripper"]["predicted_flips"] for r in records)
    true_positives = sum(r["gripper"]["true_positives"] for r in records)
    precision = true_positives / predicted_flips if predicted_flips else None
    recall = true_positives / expert_flips if expert_flips else None
    f1 = None
    if precision and recall:
        f1 = 2 * precision * recall / (precision + recall)
    lenient = [r["gripper"]["flip_recall_lenient"] for r in records
               if r["gripper"]["flip_recall_lenient"] is not None]
    out["gripper_expert_flips"] = expert_flips
    out["gripper_predicted_flips"] = predicted_flips
    out["gripper_true_positives"] = true_positives
    out["gripper_flip_precision"] = precision
    out["gripper_flip_recall"] = recall
    out["gripper_flip_f1"] = f1
    out["gripper_flip_anchors"] = len(lenient)
    out["gripper_flip_recall_lenient"] = float(np.mean(lenient)) if lenient else None
    return out


def run_audit(args) -> int:
    columns, prompts = _read_columns(args.dataset_root)
    sampled = sample_anchors_per_context(
        columns["hrc.context_id"],
        columns["hrc.valid_anchor"],
        max_per_context=args.max_anchors_per_context,
        seed=args.seed,
    )
    policy = _load_policy(args.config, args.checkpoint)
    source_ids = sorted({int(s) for s in np.unique(columns["hrc.source_episode_id"])})

    action = columns["action"]
    state = columns["observation.state"]
    source_of = columns["hrc.source_episode_id"]
    context_of = columns["hrc.context_id"]

    per_context: dict[int, list[dict]] = {c: [] for c in sampled}
    per_substage: dict[int, list[dict]] = {}
    spread_records: list[dict] = []

    # Sub-stage labels come from the expert action stream only, so they can be
    # resolved before any video is decoded.
    substage_of: dict[int, int] = {}
    pickup_context = CONTEXT_NAMES.index("pickup")
    for sid in source_ids:
        rows = np.flatnonzero((source_of == sid) & (context_of == pickup_context))
        if len(rows):
            for row, stage in zip(rows, pickup_substage_ids(action, rows), strict=True):
                substage_of[int(row)] = int(stage)

    # Decode one source at a time: the full train split is ~86 GiB of frames and the
    # container caps at 90 GiB, so holding them all at once is an OOM (exit 137).
    context_by_row = {int(row): context for context, anchors in sampled.items() for row in anchors}
    rows_by_source: dict[int, list[int]] = {}
    for row in sorted(context_by_row):
        rows_by_source.setdefault(int(source_of[row]), []).append(row)

    for order, (sid, rows) in enumerate(sorted(rows_by_source.items()), start=1):
        frames = _decode_source_frames(args.dataset_root, sid)
        print(f"[audit] source {sid} ({order}/{len(rows_by_source)}): {len(rows)} anchors, "
              f"{len(frames)} frames decoded", flush=True)
        for row in rows:
            context = context_by_row[row]
            frame_index = int(columns["hrc.raw_global_frame_index"][row])
            image = frames[frame_index]
            anchor_state = state[row]
            previous = action[row - 1] if row > 0 and int(source_of[row - 1]) == sid else action[row]

            predicted = _predict_chunk(policy, image, anchor_state, prompts[row])
            expert = action[row : row + predicted.shape[0]]
            predicted = predicted[: len(expert)]
            null = null_chunk(anchor_state, previous, len(expert))

            record = {
                "mae_model": group_mae(predicted, expert),
                "mae_null": group_mae(null, expert),
                "arm_motion_model": arm_motion_norm(predicted, anchor_state),
                "arm_motion_expert": arm_motion_norm(expert, anchor_state),
                "gripper": gripper_stats(predicted, expert, previous),
            }
            per_context[context].append(record)
            if row in substage_of and context == pickup_context:
                per_substage.setdefault(substage_of[row], []).append(record)

            if len(spread_records) < args.spread_anchors and context == pickup_context:
                samples = [predicted] + [
                    _predict_chunk(policy, image, anchor_state, prompts[row])[: len(expert)]
                    for _ in range(args.spread_seeds - 1)
                ]
                stack = np.stack(samples)
                spread_records.append(
                    {
                        "row": row,
                        "arms_std_across_seeds": float(np.mean(np.std(stack[..., DELTA_DIMS], axis=0))),
                        "base_std_across_seeds": float(np.mean(np.std(stack[..., BASE_DIMS], axis=0))),
                        "expert_arm_motion": record["arm_motion_expert"],
                    }
                )
        del frames

    report = {
        "config": args.config,
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "max_anchors_per_context": args.max_anchors_per_context,
        "seed": args.seed,
        "by_context": {
            CONTEXT_NAMES[c]: _summarize(records) for c, records in per_context.items() if records
        },
        "pickup_by_substage": {
            str(stage): _summarize(records) for stage, records in sorted(per_substage.items())
        },
        "sample_spread": {
            "seeds": args.spread_seeds,
            "anchors": len(spread_records),
            "arms_std_across_seeds": float(
                np.mean([r["arms_std_across_seeds"] for r in spread_records])
            )
            if spread_records
            else None,
            "base_std_across_seeds": float(
                np.mean([r["base_std_across_seeds"] for r in spread_records])
            )
            if spread_records
            else None,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "teacher_forced_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    def fmt(value, spec=".4f", width=11):
        # skill is None when the null predictor is already exact (null MAE == 0),
        # which happens for groups the expert never moves in a context.
        text = "n/a" if value is None else format(value, spec)
        return f"{text:>{width}}"

    print(f"\n{'context':<9} {'anchors':>7} {'arms skill':>11} {'arm mot m/e':>14} "
          f"{'flipsGT/pred':>13} {'precision':>10} {'recall':>8} {'F1':>8}")
    for name, summary in report["by_context"].items():
        print(f"{name:<9} {summary['num_anchors']:>7} {fmt(summary['arms']['skill'])} "
              f"{summary['arm_motion_model']:>6.4f}/{summary['arm_motion_expert']:<7.4f} "
              f"{summary['gripper_expert_flips']:>6}/{summary['gripper_predicted_flips']:<6} "
              f"{fmt(summary['gripper_flip_precision'], width=10)} "
              f"{fmt(summary['gripper_flip_recall'], width=8)} "
              f"{fmt(summary['gripper_flip_f1'], width=8)}")
    if report["pickup_by_substage"]:
        print("\npickup 子阶段 (按专家夹爪翻转切分):")
        for stage, summary in report["pickup_by_substage"].items():
            print(f"  stage {stage:<3} n={summary['num_anchors']:<5} "
                  f"arms skill={fmt(summary['arms']['skill'], width=8)} "
                  f"arm motion={summary['arm_motion_model']:.4f}/{summary['arm_motion_expert']:.4f}")
    spread = report["sample_spread"]
    print(f"\n多 seed 采样发散度 ({spread['seeds']} seeds, {spread['anchors']} anchors): "
          f"arms std={spread['arms_std_across_seeds']}, base std={spread['base_std_across_seeds']}")
    print(f"\nwrote {args.output_dir / 'teacher_forced_audit.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-anchors-per-context", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spread-seeds", type=int, default=5)
    parser.add_argument("--spread-anchors", type=int, default=10)
    return run_audit(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
