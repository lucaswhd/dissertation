#!/usr/bin/env python
"""Zero-observation lookup-table baseline, scored on the same ruler as the model audit.

Why this exists (HANDOFF §6.6): the visual-conditioning probe measured that arm output
responds to neither the image (0.201x) nor the state (0.732x) more than to resampling
the sampling noise. That is consistent with the policy having collapsed onto a
shortcut -- "read the sub-instruction, emit that instruction's average motion" -- but
sensitivity alone cannot say *how good* that fixed component is. This script builds the
shortcut explicitly and scores it against the model.

The predictor:

    for each prompt p and chunk offset k
        delta[p, k]   = mean over train anchors of (action[t+k] - state[t])   # arm/torso
        base[p, k]    = mean over train anchors of action[t+k][0:3]           # absolute
        gripper[p, k] = mean over train anchors of action[t+k][14, 22]        # absolute

    predict(t) = state[t] + delta[prompt[t]]   on the 18 delta dims
                 base[prompt[t]]               on dims 0-2
                 gripper[prompt[t]]            on dims 14, 22

It reads no image and no state beyond the anchor pose that the *null* baseline already
uses as its origin, so it sits exactly one rung above null: null says "zero delta",
this says "the average delta for this instruction". Everything else -- anchor sampling,
skill formula, arm-motion ratio, gripper matching, sub-stage split -- is imported from
the model audit so the numbers are directly comparable to §4.2.

Both the table and the evaluation use the train split, matching how §4.2 audits the
model on train anchors: this is a train-set-vs-train-set comparison on both sides.

Usage (openpi_hrc/.venv, no GPU, no video decoding):
  .venv/bin/python -m scripts.hrc.audit_hrc_lookup_baseline \
      --dataset-root /root/autodl-tmp/lerobot_datasets/hrc_subinstr_rerun_v1/train \
      --output-dir outputs/hrc_b_lookup_baseline \
      --compare-audit outputs/hrc_b_audit_train/teacher_forced_audit.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.hrc.audit_hrc_teacher_forced import BASE_DIMS
from scripts.hrc.audit_hrc_teacher_forced import DELTA_DIMS
from scripts.hrc.audit_hrc_teacher_forced import GRIPPER_DIMS
from scripts.hrc.audit_hrc_teacher_forced import _summarize
from scripts.hrc.audit_hrc_teacher_forced import arm_motion_norm
from scripts.hrc.audit_hrc_teacher_forced import group_mae
from scripts.hrc.audit_hrc_teacher_forced import gripper_stats
from scripts.hrc.audit_hrc_teacher_forced import null_chunk
from scripts.hrc.audit_hrc_teacher_forced import pickup_substage_ids
from scripts.hrc.evaluate_hrc_pi05_continuous import ACTION_DIM
from scripts.hrc.evaluate_hrc_pi05_continuous import CONTEXT_NAMES
from scripts.hrc.evaluate_hrc_pi05_continuous import _read_columns
from scripts.hrc.evaluate_hrc_pi05_continuous import sample_anchors_per_context

HORIZON = 32


def build_lookup_table(
    action: np.ndarray,
    state: np.ndarray,
    prompts: np.ndarray,
    anchor_rows: np.ndarray,
    *,
    horizon: int = HORIZON,
    batch: int = 20_000,
) -> dict[str, np.ndarray]:
    """Mean [horizon, 23] chunk per prompt; delta dims are relative to the anchor pose."""
    tables: dict[str, np.ndarray] = {}
    offsets = np.arange(horizon)
    for prompt in sorted({str(p) for p in prompts[anchor_rows]}):
        rows = anchor_rows[prompts[anchor_rows] == prompt]
        total = np.zeros((horizon, ACTION_DIM), dtype=np.float64)
        for start in range(0, len(rows), batch):
            block = rows[start : start + batch]
            chunk = action[block[:, None] + offsets[None, :]].astype(np.float64)
            anchor = state[block][:, None, :].astype(np.float64)
            chunk[:, :, DELTA_DIMS] -= anchor[:, :, DELTA_DIMS]
            total += chunk.sum(axis=0)
        tables[prompt] = (total / len(rows)).astype(np.float32)
    return tables


def lookup_chunk(
    table: np.ndarray, anchor_state: np.ndarray, horizon: int, *, copycat_base: bool = False
) -> np.ndarray:
    """Absolute [horizon, 23] prediction from the table and the anchor pose."""
    out = np.array(table[:horizon], dtype=np.float32, copy=True)
    out[:, DELTA_DIMS] += np.asarray(anchor_state, dtype=np.float32)[DELTA_DIMS]
    if copycat_base:
        # state[0:3] is the previous base command (HANDOFF §2.2). Holding it tests
        # whether base skill is explained by inertia rather than by the table.
        out[:, BASE_DIMS] = np.asarray(anchor_state, dtype=np.float32)[BASE_DIMS]
    return out


def lookup_residuals(
    action: np.ndarray,
    state: np.ndarray,
    prompts: np.ndarray,
    anchor_rows: np.ndarray,
    tables: dict[str, np.ndarray],
    *,
    horizon: int = HORIZON,
    batch: int = 20_000,
) -> np.ndarray:
    """Per-anchor mean |expert - lookup| over the 18 arm/torso dims, in delta space.

    High residual = "the instruction average cannot explain this frame", i.e. exactly
    the frames where an observation is required. Intended as a sampling weight that is
    measured rather than guessed (B's decision frames use gripper flips as a proxy).
    """
    residual = np.zeros(len(anchor_rows), dtype=np.float32)
    offsets = np.arange(horizon)
    for prompt, table in tables.items():
        mask = prompts[anchor_rows] == prompt
        rows = anchor_rows[mask]
        if not len(rows):
            continue
        out = np.zeros(len(rows), dtype=np.float32)
        for start in range(0, len(rows), batch):
            block = rows[start : start + batch]
            chunk = action[block[:, None] + offsets[None, :]].astype(np.float32)
            anchor = state[block][:, None, :]
            chunk[:, :, DELTA_DIMS] -= anchor[:, :, DELTA_DIMS]
            err = np.abs(chunk[:, :, DELTA_DIMS] - table[None, :horizon, DELTA_DIMS])
            out[start : start + len(block)] = err.mean(axis=(1, 2))
        residual[mask] = out
    return residual


def _evaluate(args, columns, prompts, tables, sampled, *, copycat_base: bool) -> dict:
    action = columns["action"]
    state = columns["observation.state"]
    source_of = columns["hrc.source_episode_id"]
    context_of = columns["hrc.context_id"]

    pickup_context = CONTEXT_NAMES.index("pickup")
    substage_of: dict[int, int] = {}
    for sid in sorted({int(s) for s in np.unique(source_of)}):
        rows = np.flatnonzero((source_of == sid) & (context_of == pickup_context))
        if len(rows):
            for row, stage in zip(rows, pickup_substage_ids(action, rows), strict=True):
                substage_of[int(row)] = int(stage)

    per_context: dict[int, list[dict]] = {c: [] for c in sampled}
    per_substage: dict[int, list[dict]] = {}

    for context, rows in sampled.items():
        for row in rows:
            row = int(row)
            sid = int(source_of[row])
            anchor_state = state[row]
            previous = action[row - 1] if row > 0 and int(source_of[row - 1]) == sid else action[row]

            expert = action[row : row + HORIZON]
            predicted = lookup_chunk(
                tables[str(prompts[row])], anchor_state, len(expert), copycat_base=copycat_base
            )
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

    return {
        "by_context": {
            CONTEXT_NAMES[c]: _summarize(records) for c, records in per_context.items() if records
        },
        "pickup_by_substage": {
            str(stage): _summarize(records) for stage, records in sorted(per_substage.items())
        },
    }


def _verdict(lookup: dict, model: dict | None) -> dict | None:
    """Side-by-side on the metrics §8.3 M2A2 says to judge on."""
    if model is None:
        return None
    out: dict = {}
    for context in ("pickup", "radio"):
        lut = lookup["by_context"].get(context)
        mdl = model.get("by_context", {}).get(context)
        if not lut or not mdl:
            continue
        out[context] = {
            "arms_skill_lookup": lut["arms"]["skill"],
            "arms_skill_model": mdl["arms"]["skill"],
            "arms_skill_gap_model_minus_lookup": (
                None
                if lut["arms"]["skill"] is None or mdl["arms"]["skill"] is None
                else mdl["arms"]["skill"] - lut["arms"]["skill"]
            ),
            "arm_motion_ratio_lookup": lut["arm_motion_ratio"],
            "arm_motion_ratio_model": mdl["arm_motion_ratio"],
            "gripper_f1_lookup": lut["gripper_flip_f1"],
            "gripper_f1_model": mdl["gripper_flip_f1"],
        }
    return out


def run(args) -> int:
    columns, prompts = _read_columns(args.dataset_root)
    valid = columns["hrc.valid_anchor"]
    # t + HORIZON must stay inside the same source; valid_anchor already guarantees
    # t+32 <= n-1 within a source (HANDOFF §2.2), assert rather than silently clip.
    anchor_rows = np.flatnonzero(valid)
    source_of = columns["hrc.source_episode_id"]
    if not np.all(source_of[anchor_rows] == source_of[np.minimum(anchor_rows + HORIZON, len(valid) - 1)]):
        raise ValueError("some valid anchors would read a chunk across a source boundary")

    print(f"[lut] {len(anchor_rows):,} train valid anchors, "
          f"{len({str(p) for p in prompts[anchor_rows]})} distinct prompts", flush=True)

    tables = build_lookup_table(columns["action"], columns["observation.state"], prompts, anchor_rows)
    sampled = sample_anchors_per_context(
        columns["hrc.context_id"],
        valid,
        max_per_context=args.max_anchors_per_context,
        seed=args.seed,
    )

    model = None
    if args.compare_audit is not None:
        model = json.loads(args.compare_audit.read_text(encoding="utf-8"))

    plain = _evaluate(args, columns, prompts, tables, sampled, copycat_base=False)
    copycat = _evaluate(args, columns, prompts, tables, sampled, copycat_base=True)

    residual = lookup_residuals(
        columns["action"], columns["observation.state"], prompts, anchor_rows, tables
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "lookup_residuals.npz",
        anchor_rows=anchor_rows,
        arms_residual=residual,
        source_episode_id=source_of[anchor_rows],
        context_id=columns["hrc.context_id"][anchor_rows],
        prompt_index=np.array(
            [sorted(tables).index(str(p)) for p in prompts[anchor_rows]], dtype=np.int64
        ),
        prompt_order=np.array(sorted(tables), dtype=object),
    )

    per_prompt = {}
    for i, prompt in enumerate(sorted(tables)):
        mask = prompts[anchor_rows] == prompt
        per_prompt[prompt] = {
            "anchors": int(mask.sum()),
            "mean_arms_residual": float(residual[mask].mean()),
            "p90_arms_residual": float(np.percentile(residual[mask], 90)),
        }

    report = {
        "predictor": "prompt-keyed mean action chunk (zero observation)",
        "dataset_root": str(args.dataset_root),
        "max_anchors_per_context": args.max_anchors_per_context,
        "seed": args.seed,
        "horizon": HORIZON,
        "table_fitted_on_anchors": int(len(anchor_rows)),
        "lookup": plain,
        "lookup_copycat_base": copycat,
        "compare_audit": str(args.compare_audit) if args.compare_audit else None,
        "verdict": _verdict(plain, model),
        "residual_by_prompt": per_prompt,
    }
    (args.output_dir / "lookup_baseline.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    def fmt(value, width=11):
        return f"{'n/a' if value is None else format(value, '.4f'):>{width}}"

    print(f"\n{'context':<9} {'anchors':>7} {'arms skill':>11} {'base skill':>11} "
          f"{'arm mot l/e':>14} {'flipsGT/pred':>13} {'F1':>8}")
    for name, summary in plain["by_context"].items():
        print(f"{name:<9} {summary['num_anchors']:>7} {fmt(summary['arms']['skill'])} "
              f"{fmt(summary['base']['skill'])} "
              f"{summary['arm_motion_model']:>6.4f}/{summary['arm_motion_expert']:<7.4f} "
              f"{summary['gripper_expert_flips']:>6}/{summary['gripper_predicted_flips']:<6} "
              f"{fmt(summary['gripper_flip_f1'], 8)}")

    print("\ncopycat-base variant (base = state[0:3], the previous base command):")
    for name, summary in copycat["by_context"].items():
        print(f"  {name:<9} base skill={fmt(summary['base']['skill'], 9)}")

    if report["verdict"]:
        print("\n=== 判决：模型 vs 查表器 ===")
        for context, row in report["verdict"].items():
            print(f"  [{context}] arms skill  lookup={fmt(row['arms_skill_lookup'], 9)}"
                  f"  model={fmt(row['arms_skill_model'], 9)}"
                  f"  gap={fmt(row['arms_skill_gap_model_minus_lookup'], 9)}")
            print(f"  {'':<{len(context) + 2}} 幅度比     lookup={fmt(row['arm_motion_ratio_lookup'], 9)}"
                  f"  model={fmt(row['arm_motion_ratio_model'], 9)}")
            print(f"  {'':<{len(context) + 2}} 夹爪 F1    lookup={fmt(row['gripper_f1_lookup'], 9)}"
                  f"  model={fmt(row['gripper_f1_model'], 9)}")

    print(f"\nwrote {args.output_dir / 'lookup_baseline.json'}")
    print(f"wrote {args.output_dir / 'lookup_residuals.npz'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--compare-audit", type=Path, default=None,
                        help="teacher_forced_audit.json to score against (same split/seed/cap)")
    parser.add_argument("--max-anchors-per-context", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
