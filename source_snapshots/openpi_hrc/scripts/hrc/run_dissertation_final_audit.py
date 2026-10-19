#!/usr/bin/env python
"""Freeze the common-anchor A20/B20/C20 dissertation teacher-forced audit."""

from __future__ import annotations

import argparse
import dataclasses
from datetime import UTC
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from scripts.hrc.audit_hrc_lookup_baseline import build_lookup_table
from scripts.hrc.audit_hrc_lookup_baseline import lookup_chunk
from scripts.hrc.dissertation_protocol import ACTION_DIM
from scripts.hrc.dissertation_protocol import ACTION_GROUPS
from scripts.hrc.dissertation_protocol import ACTION_HORIZON
from scripts.hrc.dissertation_protocol import CONTEXT_NAMES
from scripts.hrc.dissertation_protocol import MODEL_ACTION_DIM
from scripts.hrc.dissertation_protocol import aggregate_anchor_records
from scripts.hrc.dissertation_protocol import bootstrap_summary
from scripts.hrc.dissertation_protocol import build_common_audit_manifest
from scripts.hrc.dissertation_protocol import file_sha256
from scripts.hrc.dissertation_protocol import flow_noise
from scripts.hrc.dissertation_protocol import load_split_rows
from scripts.hrc.dissertation_protocol import null_chunk
from scripts.hrc.dissertation_protocol import score_anchor
from scripts.hrc.dissertation_protocol import write_csv_atomic
from scripts.hrc.dissertation_protocol import write_json_atomic
from scripts.hrc.evaluate_hrc_pi05_continuous import _decode_source_frames
from scripts.hrc.evaluate_hrc_pi05_continuous import _load_policy

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/root/autodl-tmp/lerobot_datasets")
A_DATASET_VERSION = "hrc_pi05_continuous_rerun_v1"
BC_DATASET_VERSION = "hrc_subinstr_rerun_v1"
SPLIT_MANIFEST = DATA_ROOT / "hrc_pi05_continuous_rerun_v1_split.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/dissertation_final_audit"
DEFAULT_PROTOCOL_SEED = 20260907
DEFAULT_FLOW_SEEDS = (0, 1, 2, 3, 4)
GRIPPER_TOLERANCE = 8


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    label: str
    config_name: str
    checkpoint: Path
    dataset_version: str
    prompt_field: str
    row_field: str
    normalization: str

    @property
    def split_root(self) -> Path:
        return DATA_ROOT / self.dataset_version / "val"


MODEL_SPECS = {
    "A20": ModelSpec(
        label="A20",
        config_name="pi05_hrc_e6_rerun_20k",
        checkpoint=REPO_ROOT / "checkpoints/pi05_hrc_e6_rerun_20k/hrc_e6_rerun_seed42/20000",
        dataset_version=A_DATASET_VERSION,
        prompt_field="prompt_a",
        row_field="row_a",
        normalization="z-score",
    ),
    "B20": ModelSpec(
        label="B20",
        config_name="pi05_hrc_b_subinstr_20k",
        checkpoint=REPO_ROOT / "checkpoints/pi05_hrc_b_subinstr_20k/b_subinstr_seed42/20000",
        dataset_version=BC_DATASET_VERSION,
        prompt_field="prompt_bc",
        row_field="row_bc",
        normalization="z-score",
    ),
    "C20": ModelSpec(
        label="C20",
        config_name="pi05_hrc_c_qnorm_20k",
        checkpoint=REPO_ROOT / "checkpoints/pi05_hrc_c_qnorm_20k/c_qnorm_seed42/20000",
        dataset_version=BC_DATASET_VERSION,
        prompt_field="prompt_bc",
        row_field="row_bc",
        normalization="quantile",
    ),
}

PER_ANCHOR_FIELDS = [
    "anchor_id",
    "source_episode_id",
    "raw_global_frame_index",
    "context",
    "semantic_label",
    "flow_seed",
    "arms_raw_mae",
    "arms_null_mae",
    "arms_skill",
    "base_raw_mae",
    "base_null_mae",
    "base_skill",
    "grippers_raw_mae",
    "grippers_null_mae",
    "grippers_skill",
    "arm_motion_model",
    "arm_motion_expert",
    "arm_motion_ratio",
    "gripper_sign_accuracy",
    "gripper_expert_events",
    "gripper_predicted_events",
    "gripper_true_positives",
    "gripper_precision",
    "gripper_recall",
    "gripper_f1",
]

COMPARISON_FIELDS = ["model", "context", "action_group", "metric", "mean", "std", "n", "baseline"]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def validate_model_binding(
    spec: ModelSpec,
    *,
    config_name: str,
    checkpoint: Path,
    dataset_version: str,
) -> None:
    if config_name != spec.config_name:
        raise ValueError(f"config name {config_name!r} does not match {spec.label} {spec.config_name!r}")
    if Path(checkpoint).resolve() != spec.checkpoint.resolve():
        raise ValueError(f"checkpoint {checkpoint} does not match frozen {spec.label} path {spec.checkpoint}")
    if dataset_version != spec.dataset_version:
        raise ValueError(
            f"dataset version {dataset_version!r} does not match {spec.label} {spec.dataset_version!r}"
        )


def validate_complete_seed_records(
    records: dict[int, list[dict[str, Any]]],
    *,
    expected_anchor_ids: list[str],
    expected_seeds: list[int],
) -> None:
    expected = set(expected_anchor_ids)
    missing_seeds = sorted(set(expected_seeds) - records.keys())
    extra_seeds = sorted(records.keys() - set(expected_seeds))
    if missing_seeds or extra_seeds:
        raise ValueError(f"seed mismatch: missing={missing_seeds}, extra={extra_seeds}")
    for seed in expected_seeds:
        ids = [str(record["anchor_id"]) for record in records[seed]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"seed {seed} has duplicate anchor ids")
        missing = sorted(expected - set(ids))
        extra = sorted(set(ids) - expected)
        if missing or extra:
            raise ValueError(f"seed {seed} anchor mismatch: missing={missing}, extra={extra}")


def _mean_std(values: list[float | int | None]) -> dict[str, float | int | None]:
    finite = np.asarray([float(value) for value in values if value is not None], dtype=np.float64)
    if not len(finite):
        return {"mean": None, "std": None, "n": 0}
    if not np.isfinite(finite).all():
        raise ValueError("non-finite per-seed metric")
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "n": len(finite),
    }


def _flatten_context_summary(summary: dict[str, Any]) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {}
    for group in ACTION_GROUPS:
        for metric in ("raw_mae", "null_mae", "skill"):
            metrics[f"{group}.{metric}"] = summary["groups"][group][metric]
    metrics["arm_motion_model"] = summary["arm_motion_model"]
    metrics["arm_motion_expert"] = summary["arm_motion_expert"]
    metrics["arm_motion_ratio"] = summary["arm_motion_ratio"]
    for metric in (
        "sign_accuracy",
        "expert_events",
        "predicted_events",
        "true_positives",
        "precision",
        "recall",
        "f1",
    ):
        metrics[f"grippers.{metric}"] = summary["gripper"][metric]
    return metrics


def _context_records(records: list[dict[str, Any]], context: str) -> list[dict[str, Any]]:
    return [record["score"] for record in records if record["context"] == context]


def _make_seed_summary(
    spec: ModelSpec,
    flow_seed_value: int,
    records: list[dict[str, Any]],
    *,
    manifest_sha256: str,
    started_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model": spec.label,
        "config": spec.config_name,
        "checkpoint": str(spec.checkpoint),
        "dataset_version": spec.dataset_version,
        "flow_seed": int(flow_seed_value),
        "manifest_sha256": manifest_sha256,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "num_anchors": len(records),
        "by_context": {
            context: aggregate_anchor_records(_context_records(records, context))
            for context in CONTEXT_NAMES
            if _context_records(records, context)
        },
    }


def _flat_anchor_record(record: dict[str, Any], flow_seed_value: int) -> dict[str, Any]:
    score = record["score"]
    row = {
        "anchor_id": record["anchor_id"],
        "source_episode_id": record["source_episode_id"],
        "raw_global_frame_index": record["raw_global_frame_index"],
        "context": record["context"],
        "semantic_label": record["semantic_label"],
        "flow_seed": flow_seed_value,
        "arm_motion_model": score["arm_motion_model"],
        "arm_motion_expert": score["arm_motion_expert"],
        "arm_motion_ratio": score["arm_motion_ratio"],
        "gripper_sign_accuracy": score["gripper"]["sign_accuracy"],
        "gripper_expert_events": score["gripper"]["expert_events"],
        "gripper_predicted_events": score["gripper"]["predicted_events"],
        "gripper_true_positives": score["gripper"]["true_positives"],
        "gripper_precision": score["gripper"]["precision"],
        "gripper_recall": score["gripper"]["recall"],
        "gripper_f1": score["gripper"]["f1"],
    }
    for group in ACTION_GROUPS:
        row[f"{group}_raw_mae"] = score["groups"][group]["raw_mae"]
        row[f"{group}_null_mae"] = score["groups"][group]["null_mae"]
        row[f"{group}_skill"] = score["groups"][group]["skill"]
    return row


def _write_predictions(path: Path, predictions: np.ndarray, anchor_ids: list[str]) -> None:
    if predictions.ndim != 3 or predictions.shape[1:] != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError(f"invalid prediction array shape {predictions.shape}")
    if not np.isfinite(predictions).all():
        raise ValueError("predictions contain non-finite values")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        predictions=predictions.astype(np.float32),
        anchor_ids=np.asarray(anchor_ids, dtype=str),
    )
    temporary.replace(path)


def _read_prediction_file(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as data:
        predictions = np.asarray(data["predictions"], dtype=np.float32)
        anchor_ids = [str(value) for value in data["anchor_ids"]]
    return predictions, anchor_ids


def validate_saved_prediction_bundle(
    predictions: np.ndarray,
    anchor_ids: list[str],
    expected_anchor_ids: list[str],
) -> None:
    expected_shape = (len(expected_anchor_ids), ACTION_HORIZON, ACTION_DIM)
    if predictions.shape != expected_shape:
        raise ValueError(f"saved predictions have shape {predictions.shape}, expected {expected_shape}")
    if not np.isfinite(predictions).all():
        raise ValueError("saved predictions contain non-finite values")
    if anchor_ids != expected_anchor_ids:
        raise ValueError("saved prediction anchor order does not exactly match the frozen manifest")


def _score_saved_predictions(
    spec: ModelSpec,
    manifest: dict[str, Any],
    rows: Any,
    predictions: np.ndarray,
    anchor_ids: list[str],
) -> list[dict[str, Any]]:
    expected_anchor_ids = [str(anchor["anchor_id"]) for anchor in manifest["anchors"]]
    validate_saved_prediction_bundle(predictions, anchor_ids, expected_anchor_ids)
    records = []
    for index, anchor in enumerate(manifest["anchors"]):
        row = int(anchor[spec.row_field])
        source_id = int(anchor["source_episode_id"])
        if int(rows.source_id[row]) != source_id or int(rows.raw_frame[row]) != int(
            anchor["raw_global_frame_index"]
        ):
            raise ValueError(f"manifest row mismatch for {anchor['anchor_id']}")
        previous = (
            rows.action[row - 1]
            if row > 0 and rows.source_id[row - 1] == source_id
            else rows.action[row]
        )
        expert = rows.action[row : row + ACTION_HORIZON]
        if len(expert) != ACTION_HORIZON or np.any(
            rows.source_id[row : row + ACTION_HORIZON] != source_id
        ):
            raise ValueError(f"invalid future chunk for {anchor['anchor_id']}")
        records.append(
            {
                "anchor_id": anchor["anchor_id"],
                "source_episode_id": source_id,
                "raw_global_frame_index": int(anchor["raw_global_frame_index"]),
                "context": anchor["context"],
                "semantic_label": anchor["semantic_label"],
                "score": score_anchor(
                    predictions[index],
                    expert,
                    rows.state[row],
                    previous,
                    gripper_tolerance=GRIPPER_TOLERANCE,
                ),
            }
        )
    return records


def _make_model_summary(
    spec: ModelSpec,
    manifest: dict[str, Any],
    seed_summaries: dict[int, dict[str, Any]],
    prediction_by_seed: dict[int, np.ndarray],
    *,
    manifest_sha256: str,
    determinism_max_abs_diff: float,
) -> dict[str, Any]:
    seeds = sorted(seed_summaries)
    stack = np.stack([prediction_by_seed[seed] for seed in seeds], axis=0)
    if stack.shape[:2] != (len(seeds), len(manifest["anchors"])):
        raise ValueError(f"prediction stack does not match manifest: {stack.shape}")
    dispersion = np.std(stack, axis=0, ddof=1)
    context_by_anchor = [anchor["context"] for anchor in manifest["anchors"]]
    by_context: dict[str, Any] = {}
    for context in CONTEXT_NAMES:
        seed_contexts = [seed_summaries[seed]["by_context"][context] for seed in seeds]
        metric_keys = sorted(_flatten_context_summary(seed_contexts[0]))
        metrics = {
            key: _mean_std([_flatten_context_summary(value)[key] for value in seed_contexts])
            for key in metric_keys
        }
        indices = [index for index, value in enumerate(context_by_anchor) if value == context]
        group_dispersion = {}
        for group, dims in ACTION_GROUPS.items():
            per_anchor = np.mean(dispersion[indices][:, :, list(dims)], axis=(1, 2))
            group_dispersion[group] = bootstrap_summary(
                per_anchor,
                seed=DEFAULT_PROTOCOL_SEED + CONTEXT_NAMES.index(context) * 100 + len(group),
                num_resamples=10_000,
            )
        by_context[context] = {
            "num_anchors": len(indices),
            "metrics": metrics,
            "dispersion": group_dispersion,
        }
    return {
        "schema_version": 1,
        "model": spec.label,
        "config": spec.config_name,
        "checkpoint": str(spec.checkpoint),
        "dataset_version": spec.dataset_version,
        "normalization": spec.normalization,
        "manifest_sha256": manifest_sha256,
        "flow_seeds": seeds,
        "num_anchors": len(manifest["anchors"]),
        "determinism_max_abs_diff": determinism_max_abs_diff,
        "by_context": by_context,
        "completed_at_utc": utc_now(),
        "baseline": "null",
    }


def build_comparison_rows(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, summary in sorted(summaries.items()):
        baseline = summary.get("baseline", "null")
        for context, context_summary in summary["by_context"].items():
            for key, stats in context_summary.get("metrics", {}).items():
                if "." in key:
                    group, name = key.split(".", maxsplit=1)
                    metric = f"gripper_{name}" if group == "grippers" and name in {
                        "sign_accuracy",
                        "expert_events",
                        "predicted_events",
                        "true_positives",
                        "precision",
                        "recall",
                        "f1",
                    } else name
                elif key.startswith("arm_motion_"):
                    group, metric = "arms", key
                else:
                    continue
                rows.append(
                    {
                        "model": model,
                        "context": context,
                        "action_group": group,
                        "metric": metric,
                        "mean": stats["mean"],
                        "std": stats["std"],
                        "n": stats["n"],
                        "baseline": baseline,
                    }
                )
            for group, stats in context_summary.get("dispersion", {}).items():
                rows.append(
                    {
                        "model": model,
                        "context": context,
                        "action_group": group,
                        "metric": "output_dispersion",
                        "mean": stats["mean"],
                        "std": stats["std"],
                        "n": stats["n"],
                        "baseline": baseline,
                    }
                )
    order = {name: index for index, name in enumerate(CONTEXT_NAMES)}
    return sorted(rows, key=lambda row: (row["model"], order.get(row["context"], 99), row["action_group"], row["metric"]))


def create_manifest(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    path = output_root / "anchor_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("protocol_seed") != args.protocol_seed or existing.get("flow_seeds") != args.flow_seeds:
            raise ValueError(f"existing manifest protocol differs: {path}")
        print(f"[manifest] reusing {path} ({existing['num_anchors']} anchors)", flush=True)
        return
    a = load_split_rows(DATA_ROOT / A_DATASET_VERSION / "val")
    bc = load_split_rows(DATA_ROOT / BC_DATASET_VERSION / "val")
    manifest = build_common_audit_manifest(
        a,
        bc,
        protocol_seed=args.protocol_seed,
        per_source=args.per_source,
        horizon=ACTION_HORIZON,
    )
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    expected_sources = [int(value) for value in split["source_splits"]["val"]]
    if manifest["source_ids"] != expected_sources:
        raise ValueError(f"manifest val sources {manifest['source_ids']} != frozen {expected_sources}")
    manifest.update(
        {
            "created_at_utc": utc_now(),
            "flow_seeds": args.flow_seeds,
            "noise_scheme": "numpy SeedSequence([protocol_seed, flow_seed, source, frame, horizon, action_dim])",
            "selection_quota_per_source": args.per_source,
            "split_manifest": str(SPLIT_MANIFEST),
            "split_manifest_sha256": file_sha256(SPLIT_MANIFEST),
        }
    )
    write_json_atomic(path, manifest)
    print(f"[manifest] wrote {path} ({manifest['num_anchors']} anchors)", flush=True)


def _validate_runtime_config(spec: ModelSpec) -> None:
    from openpi.training.config import get_config

    config = get_config(spec.config_name)
    validate_model_binding(
        spec,
        config_name=config.name,
        checkpoint=spec.checkpoint,
        dataset_version=spec.dataset_version,
    )
    if config.num_train_steps != 20_000:
        raise ValueError(f"{spec.label} config has {config.num_train_steps} train steps")
    if config.model.action_horizon != ACTION_HORIZON or config.model.action_dim != MODEL_ACTION_DIM:
        raise ValueError(
            f"{spec.label} model shape is horizon={config.model.action_horizon}, dim={config.model.action_dim}"
        )
    if config.data.state_dim != ACTION_DIM:
        raise ValueError(f"{spec.label} state dim is {config.data.state_dim}, expected {ACTION_DIM}")
    if bool(config.data.quantile_norm) != (spec.normalization == "quantile"):
        raise ValueError(f"{spec.label} normalization does not match frozen spec")
    if not (spec.checkpoint / "params").is_dir():
        raise FileNotFoundError(spec.checkpoint / "params")


def _predict(policy: Any, image: np.ndarray, state: np.ndarray, prompt: str, noise: np.ndarray) -> np.ndarray:
    result = policy.infer(
        {
            "observation/image_0": np.asarray(image, dtype=np.uint8),
            "observation/state": np.asarray(state, dtype=np.float32),
            "prompt": str(prompt),
        },
        noise=np.asarray(noise, dtype=np.float32),
    )
    prediction = np.asarray(result["actions"], dtype=np.float32)
    if prediction.shape != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError(f"policy returned {prediction.shape}, expected {(ACTION_HORIZON, ACTION_DIM)}")
    if not np.isfinite(prediction).all():
        raise ValueError("policy returned non-finite actions")
    return prediction


def run_model(args: argparse.Namespace) -> None:
    spec = MODEL_SPECS[args.model]
    _validate_runtime_config(spec)
    manifest_path = args.output_root.resolve() / "anchor_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"create the manifest first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.flow_seeds != manifest["flow_seeds"]:
        raise ValueError(f"flow seeds {args.flow_seeds} != frozen {manifest['flow_seeds']}")
    manifest_sha = file_sha256(manifest_path)
    rows = load_split_rows(spec.split_root)
    validate_model_binding(
        spec,
        config_name=spec.config_name,
        checkpoint=spec.checkpoint,
        dataset_version=rows.dataset_version,
    )
    anchors = manifest["anchors"]
    for anchor in anchors:
        row = int(anchor[spec.row_field])
        if int(rows.source_id[row]) != anchor["source_episode_id"] or int(rows.raw_frame[row]) != anchor["raw_global_frame_index"]:
            raise ValueError(f"manifest row mismatch for {anchor['anchor_id']}")
        if str(rows.prompts[row]) != anchor[spec.prompt_field]:
            raise ValueError(f"manifest prompt mismatch for {anchor['anchor_id']}")

    model_root = args.output_root.resolve() / spec.label
    if (model_root / "summary.json").exists():
        raise FileExistsError(f"refusing to overwrite completed model result {model_root / 'summary.json'}")
    model_root.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    write_json_atomic(
        model_root / "run_status.json",
        {
            "status": "running",
            "started_at_utc": started_at,
            "pid": os.getpid(),
            "model": spec.label,
            "manifest_sha256": manifest_sha,
        },
    )
    policy = _load_policy(spec.config_name, spec.checkpoint)

    check_anchor = next(anchor for anchor in anchors if anchor["context"] == "pickup")
    check_row = int(check_anchor[spec.row_field])
    check_frames = _decode_source_frames(spec.split_root, int(check_anchor["source_episode_id"]))
    check_image = check_frames[int(check_anchor["raw_global_frame_index"])]
    check_noise = flow_noise(
        manifest["protocol_seed"],
        args.flow_seeds[0],
        int(check_anchor["source_episode_id"]),
        int(check_anchor["raw_global_frame_index"]),
    )
    first = _predict(policy, check_image, rows.state[check_row], check_anchor[spec.prompt_field], check_noise)
    second = _predict(policy, check_image, rows.state[check_row], check_anchor[spec.prompt_field], check_noise)
    determinism = float(np.max(np.abs(first - second)))
    del check_frames
    if determinism != 0.0:
        raise RuntimeError(f"fixed-noise determinism failed for {spec.label}: max abs diff {determinism}")
    print(f"[{spec.label}] fixed-noise determinism max abs diff = {determinism:.1f}", flush=True)

    anchor_index = {anchor["anchor_id"]: index for index, anchor in enumerate(anchors)}
    predictions = {
        seed: np.empty((len(anchors), ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
        for seed in args.flow_seeds
    }
    records = {seed: [] for seed in args.flow_seeds}
    anchors_by_source: dict[int, list[dict[str, Any]]] = {}
    for anchor in anchors:
        anchors_by_source.setdefault(int(anchor["source_episode_id"]), []).append(anchor)

    for source_order, (source_id, source_anchors) in enumerate(sorted(anchors_by_source.items()), start=1):
        frames = _decode_source_frames(spec.split_root, source_id)
        max_frame = max(int(anchor["raw_global_frame_index"]) for anchor in source_anchors)
        if max_frame >= len(frames):
            raise ValueError(f"source {source_id} needs frame {max_frame}, video has {len(frames)}")
        print(
            f"[{spec.label}] source {source_id} ({source_order}/{len(anchors_by_source)}), "
            f"{len(source_anchors)} anchors, {len(frames)} frames",
            flush=True,
        )
        for local_index, anchor in enumerate(source_anchors, start=1):
            row = int(anchor[spec.row_field])
            image = frames[int(anchor["raw_global_frame_index"])]
            state = rows.state[row]
            previous = rows.action[row - 1] if row > 0 and rows.source_id[row - 1] == source_id else rows.action[row]
            expert = rows.action[row : row + ACTION_HORIZON]
            if len(expert) != ACTION_HORIZON or np.any(rows.source_id[row : row + ACTION_HORIZON] != source_id):
                raise ValueError(f"invalid future chunk for {anchor['anchor_id']}")
            for seed in args.flow_seeds:
                noise = flow_noise(
                    manifest["protocol_seed"],
                    seed,
                    source_id,
                    int(anchor["raw_global_frame_index"]),
                )
                predicted = _predict(policy, image, state, anchor[spec.prompt_field], noise)
                index = anchor_index[anchor["anchor_id"]]
                predictions[seed][index] = predicted
                records[seed].append(
                    {
                        "anchor_id": anchor["anchor_id"],
                        "source_episode_id": source_id,
                        "raw_global_frame_index": int(anchor["raw_global_frame_index"]),
                        "context": anchor["context"],
                        "semantic_label": anchor["semantic_label"],
                        "score": score_anchor(
                            predicted,
                            expert,
                            state,
                            previous,
                            gripper_tolerance=GRIPPER_TOLERANCE,
                        ),
                    }
                )
            if local_index % 25 == 0 or local_index == len(source_anchors):
                print(f"[{spec.label}] source {source_id}: {local_index}/{len(source_anchors)} anchors", flush=True)
        del frames

    validate_complete_seed_records(
        records,
        expected_anchor_ids=[anchor["anchor_id"] for anchor in anchors],
        expected_seeds=args.flow_seeds,
    )
    seed_summaries: dict[int, dict[str, Any]] = {}
    for seed in args.flow_seeds:
        records[seed].sort(key=lambda record: anchor_index[record["anchor_id"]])
        seed_root = model_root / "per_seed" / f"seed_{seed}"
        seed_summary = _make_seed_summary(
            spec,
            seed,
            records[seed],
            manifest_sha256=manifest_sha,
            started_at=started_at,
        )
        write_json_atomic(seed_root / "summary.json", seed_summary)
        write_csv_atomic(
            seed_root / "per_anchor.csv",
            [_flat_anchor_record(record, seed) for record in records[seed]],
            PER_ANCHOR_FIELDS,
        )
        _write_predictions(
            seed_root / "predictions.npz",
            predictions[seed],
            [anchor["anchor_id"] for anchor in anchors],
        )
        seed_summaries[seed] = seed_summary

    summary = _make_model_summary(
        spec,
        manifest,
        seed_summaries,
        predictions,
        manifest_sha256=manifest_sha,
        determinism_max_abs_diff=determinism,
    )
    write_json_atomic(model_root / "summary.json", summary)
    write_json_atomic(
        model_root / "run_status.json",
        {
            "status": "complete",
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "model": spec.label,
            "manifest_sha256": manifest_sha,
            "determinism_max_abs_diff": determinism,
        },
    )
    print(f"[{spec.label}] wrote {model_root / 'summary.json'}", flush=True)


def rescore_model(args: argparse.Namespace) -> None:
    """Recompute metrics from frozen raw prediction chunks without policy inference."""
    spec = MODEL_SPECS[args.model]
    _validate_runtime_config(spec)
    output_root = args.output_root.resolve()
    manifest_path = output_root / "anchor_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.flow_seeds != manifest["flow_seeds"]:
        raise ValueError(f"flow seeds {args.flow_seeds} != frozen {manifest['flow_seeds']}")
    manifest_sha = file_sha256(manifest_path)

    source_root = args.prediction_root.resolve()
    source_summary_path = source_root / "summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    if source_summary.get("model") != spec.label:
        raise ValueError(f"prediction archive model {source_summary.get('model')} != {spec.label}")
    if source_summary.get("manifest_sha256") != manifest_sha:
        raise ValueError("prediction archive was generated from a different manifest")
    if source_summary.get("flow_seeds") != args.flow_seeds:
        raise ValueError("prediction archive flow seeds do not match the frozen protocol")
    determinism = source_summary.get("determinism_max_abs_diff")
    if determinism != 0.0:
        raise ValueError(f"prediction archive did not pass exact determinism: {determinism}")

    model_root = output_root / spec.label
    if model_root.exists():
        raise FileExistsError(f"refusing to overwrite existing model result {model_root}")
    model_root.mkdir(parents=True)
    started_at = utc_now()
    write_json_atomic(
        model_root / "run_status.json",
        {
            "status": "rescoring",
            "started_at_utc": started_at,
            "model": spec.label,
            "manifest_sha256": manifest_sha,
            "prediction_archive": str(source_root),
        },
    )

    rows = load_split_rows(spec.split_root)
    prediction_by_seed: dict[int, np.ndarray] = {}
    records_by_seed: dict[int, list[dict[str, Any]]] = {}
    expected_anchor_ids = [str(anchor["anchor_id"]) for anchor in manifest["anchors"]]
    for seed in args.flow_seeds:
        source_path = source_root / "per_seed" / f"seed_{seed}" / "predictions.npz"
        predictions, anchor_ids = _read_prediction_file(source_path)
        records = _score_saved_predictions(spec, manifest, rows, predictions, anchor_ids)
        prediction_by_seed[seed] = predictions
        records_by_seed[seed] = records

    validate_complete_seed_records(
        records_by_seed,
        expected_anchor_ids=expected_anchor_ids,
        expected_seeds=args.flow_seeds,
    )
    seed_summaries: dict[int, dict[str, Any]] = {}
    for seed in args.flow_seeds:
        seed_root = model_root / "per_seed" / f"seed_{seed}"
        seed_summary = _make_seed_summary(
            spec,
            seed,
            records_by_seed[seed],
            manifest_sha256=manifest_sha,
            started_at=started_at,
        )
        write_json_atomic(seed_root / "summary.json", seed_summary)
        write_csv_atomic(
            seed_root / "per_anchor.csv",
            [_flat_anchor_record(record, seed) for record in records_by_seed[seed]],
            PER_ANCHOR_FIELDS,
        )
        _write_predictions(seed_root / "predictions.npz", prediction_by_seed[seed], expected_anchor_ids)
        seed_summaries[seed] = seed_summary

    summary = _make_model_summary(
        spec,
        manifest,
        seed_summaries,
        prediction_by_seed,
        manifest_sha256=manifest_sha,
        determinism_max_abs_diff=determinism,
    )
    summary["rescored_from"] = str(source_root)
    summary["metric_correction"] = (
        "F1 is zero when expert events exist and there are no true positives; "
        "precision remains undefined when no events are predicted."
    )
    write_json_atomic(model_root / "summary.json", summary)
    write_json_atomic(
        model_root / "run_status.json",
        {
            "status": "complete",
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "model": spec.label,
            "manifest_sha256": manifest_sha,
            "prediction_archive": str(source_root),
            "determinism_max_abs_diff": determinism,
        },
    )
    print(f"[{spec.label}] rescored frozen predictions into {model_root}", flush=True)


def _deterministic_summary(
    label: str,
    records: list[dict[str, Any]],
    *,
    baseline: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    by_context = {}
    for context in CONTEXT_NAMES:
        scores = _context_records(records, context)
        aggregate = aggregate_anchor_records(scores)
        flat = _flatten_context_summary(aggregate)
        by_context[context] = {
            "num_anchors": len(scores),
            "metrics": {key: _mean_std([value]) for key, value in flat.items()},
            "dispersion": {},
        }
    return {
        "schema_version": 1,
        "model": label,
        "manifest_sha256": manifest_sha256,
        "num_anchors": len(records),
        "by_context": by_context,
        "baseline": baseline,
        "completed_at_utc": utc_now(),
    }


def _evaluate_baseline(
    label: str,
    manifest: dict[str, Any],
    val_rows: Any,
    *,
    row_field: str,
    prompt_field: str,
    tables: dict[str, np.ndarray] | None,
    copycat_base: bool,
    null_only: bool,
) -> list[dict[str, Any]]:
    records = []
    for anchor in manifest["anchors"]:
        row = int(anchor[row_field])
        state = val_rows.state[row]
        source_id = int(anchor["source_episode_id"])
        previous = val_rows.action[row - 1] if row > 0 and val_rows.source_id[row - 1] == source_id else val_rows.action[row]
        expert = val_rows.action[row : row + ACTION_HORIZON]
        if null_only:
            predicted = null_chunk(state, previous, ACTION_HORIZON)
        else:
            assert tables is not None
            prompt = str(anchor[prompt_field])
            if prompt not in tables:
                raise ValueError(f"{label} has no train lookup table for prompt {prompt!r}")
            predicted = lookup_chunk(
                tables[prompt],
                state,
                ACTION_HORIZON,
                copycat_base=copycat_base,
            )
        records.append(
            {
                "anchor_id": anchor["anchor_id"],
                "source_episode_id": source_id,
                "raw_global_frame_index": int(anchor["raw_global_frame_index"]),
                "context": anchor["context"],
                "semantic_label": anchor["semantic_label"],
                "score": score_anchor(predicted, expert, state, previous),
            }
        )
    return records


def run_baselines(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    output_path = output_root / "baselines/summary.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    manifest_path = output_root / "anchor_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = file_sha256(manifest_path)
    a_train = load_split_rows(DATA_ROOT / A_DATASET_VERSION / "train")
    bc_train = load_split_rows(DATA_ROOT / BC_DATASET_VERSION / "train")
    a_val = load_split_rows(DATA_ROOT / A_DATASET_VERSION / "val")
    bc_val = load_split_rows(DATA_ROOT / BC_DATASET_VERSION / "val")
    a_train_anchors = np.flatnonzero(a_train.valid)
    bc_train_anchors = np.flatnonzero(bc_train.valid)
    print(f"[baseline] fitting A lookup on {len(a_train_anchors):,} exported train anchors", flush=True)
    a_tables = build_lookup_table(a_train.action, a_train.state, a_train.prompts, a_train_anchors)
    print(f"[baseline] fitting B/C lookup on {len(bc_train_anchors):,} exported train anchors", flush=True)
    bc_tables = build_lookup_table(bc_train.action, bc_train.state, bc_train.prompts, bc_train_anchors)
    variants = {
        "null": _evaluate_baseline(
            "null",
            manifest,
            a_val,
            row_field="row_a",
            prompt_field="prompt_a",
            tables=None,
            copycat_base=False,
            null_only=True,
        ),
        "lookup_A": _evaluate_baseline(
            "lookup_A", manifest, a_val, row_field="row_a", prompt_field="prompt_a",
            tables=a_tables, copycat_base=False, null_only=False,
        ),
        "lookup_A_copycat_base": _evaluate_baseline(
            "lookup_A_copycat_base", manifest, a_val, row_field="row_a", prompt_field="prompt_a",
            tables=a_tables, copycat_base=True, null_only=False,
        ),
        "lookup_B": _evaluate_baseline(
            "lookup_B", manifest, bc_val, row_field="row_bc", prompt_field="prompt_bc",
            tables=bc_tables, copycat_base=False, null_only=False,
        ),
        "lookup_B_copycat_base": _evaluate_baseline(
            "lookup_B_copycat_base", manifest, bc_val, row_field="row_bc", prompt_field="prompt_bc",
            tables=bc_tables, copycat_base=True, null_only=False,
        ),
    }
    summaries = {
        label: _deterministic_summary(
            label,
            records,
            baseline="self" if label == "null" else "null",
            manifest_sha256=manifest_sha,
        )
        for label, records in variants.items()
    }
    report = {
        "schema_version": 1,
        "manifest_sha256": manifest_sha,
        "fit_split": "train",
        "evaluation_split": "val",
        "train_anchor_counts": {"A": len(a_train_anchors), "B_C": len(bc_train_anchors)},
        "models": summaries,
        "completed_at_utc": utc_now(),
    }
    write_json_atomic(output_path, report)
    print(f"[baseline] wrote {output_path}", flush=True)


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def finalize(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    manifest_path = output_root / "anchor_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = file_sha256(manifest_path)
    summaries = {}
    for label in MODEL_SPECS:
        path = output_root / label / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("manifest_sha256") != manifest_sha:
            raise ValueError(f"{label} summary was generated from a different manifest")
        if summary.get("flow_seeds") != manifest["flow_seeds"]:
            raise ValueError(f"{label} flow seeds do not match manifest")
        if summary.get("determinism_max_abs_diff") != 0.0:
            raise ValueError(f"{label} determinism check did not pass exactly")
        summaries[label] = summary
    baseline_path = output_root / "baselines/summary.json"
    if baseline_path.is_file():
        baseline_report = json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline_report.get("manifest_sha256") != manifest_sha:
            raise ValueError("baseline summary was generated from a different manifest")
        summaries.update(baseline_report["models"])
    comparison = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "models": summaries,
        "completed_at_utc": utc_now(),
    }
    comparison_path = output_root / "comparison.json"
    csv_path = output_root / "comparison.csv"
    if comparison_path.exists() or csv_path.exists():
        raise FileExistsError("refusing to overwrite an existing final comparison")
    write_json_atomic(comparison_path, comparison)
    write_csv_atomic(csv_path, build_comparison_rows(summaries), COMPARISON_FIELDS)

    branch = _git_value("branch", "--show-current")
    commit = _git_value("rev-parse", "HEAD")
    dirty = _git_value("status", "--short")
    readme = f"""# Dissertation Final Teacher-Forced Audit

Status: complete

## Provenance

- Git commit: `{commit}`
- Branch: `{branch}`
- Worktree at finalization: `{'dirty' if dirty else 'clean'}`
- Pre-existing user modification preserved: `docs/HRC_PI05_MANIPULATION_HANDOFF.md`
- Completed UTC: `{comparison['completed_at_utc']}`
- Split: validation only
- Source IDs: `{manifest['source_ids']}`
- Dataset A: `{A_DATASET_VERSION}`
- Dataset B/C: `{BC_DATASET_VERSION}`
- Split manifest: `{SPLIT_MANIFEST}`
- Split manifest SHA-256: `{manifest['split_manifest_sha256']}`
- Anchor manifest SHA-256: `{manifest_sha}`
- Anchors: `{manifest['selected_counts']}` (wait exhausted at all available anchors)
- Flow-noise seeds: `{manifest['flow_seeds']}`
- Flow-noise scheme: `{manifest['noise_scheme']}`

## Checkpoints And Configs

| Model | Config | Checkpoint | State normalization |
|---|---|---|---|
"""
    for spec in MODEL_SPECS.values():
        readme += f"| {spec.label} | `{spec.config_name}` | `{spec.checkpoint}` | {spec.normalization} |\n"
    readme += f"""

## Protocol

Rows are joined by source episode and raw frame. State, action, context, and FSM rows must be identical across the A and B/C exports. An anchor is eligible only when both exports mark it valid and its inclusive `[t, t+32]` window remains inside one source, episode, and task context. B/C subinstruction changes inside that task window are intentionally allowed: those boundaries coincide with gripper decisions, so excluding them would make event F1 undefined. Non-wait contexts use 50 seeded anchors per validation source; all 13 wait anchors per source are retained.

The model receives an explicit float32 `[32,32]` flow-noise tensor. A/B/C receive the same tensor for the same anchor and flow seed. Each checkpoint passed an exact repeated-inference check (`max_abs_diff == 0.0`) before statistics were computed.

`arms` contains the 18 torso/arm delta-style dimensions, `base` contains dimensions 0-2, and `grippers` contains dimensions 14 and 22. Raw MAE is compared with a physical null predictor: zero base velocity, held gripper commands, and zero arm/torso delta. Skill is `1 - MAE(model) / MAE(null)` and is undefined when pooled null MAE is zero. Arm motion ratio is mean predicted absolute arm delta divided by mean expert absolute arm delta. Gripper events are sign changes matched greedily by gripper, direction, and +/-{GRIPPER_TOLERANCE} frames; counts are pooled before precision, recall, and F1. F1 is zero when expert events exist but no true positives are found; precision remains undefined when no event is predicted. Reported model means and sample standard deviations are over the five flow-noise seeds. Output dispersion is the mean within-anchor sample standard deviation across the five predicted action chunks.

Lookup tables are fitted only on exported valid train anchors and evaluated on this validation manifest. `lookup_*_copycat_base` retains the corresponding lookup arm/gripper outputs while replacing base with the previous recorded base command.

## Exact Commands

```bash
.venv/bin/python -m scripts.hrc.run_dissertation_final_audit manifest
# Generated the unchanged A20 raw prediction tensors.
.venv/bin/python -m scripts.hrc.run_dissertation_final_audit run-model --model A20
# After correcting the zero-TP F1 edge case, preserved the old summaries and rescored those tensors.
.venv/bin/python -m scripts.hrc.run_dissertation_final_audit rescore-model --model A20 --prediction-root outputs/dissertation_final_audit/_invalid_f1_null_protocol_20260907/A20
.venv/bin/python -m scripts.hrc.run_dissertation_final_audit run-model --model B20
.venv/bin/python -m scripts.hrc.run_dissertation_final_audit run-model --model C20
.venv/bin/python -m scripts.hrc.verify_dissertation_results
.venv/bin/python -m scripts.hrc.run_dissertation_final_audit baselines
.venv/bin/python -m scripts.hrc.run_dissertation_final_audit finalize
```

## Evidence

- `anchor_manifest.json`: frozen questions and prompts.
- `A20|B20|C20/per_seed/seed_N/per_anchor.csv`: anchor-level scores.
- `A20|B20|C20/per_seed/seed_N/predictions.npz`: physical predicted chunks.
- `A20|B20|C20/summary.json`: mean/std and multi-seed dispersion.
- `baselines/summary.json`: train-fit, validation-evaluated baselines.
- `comparison.csv` and `comparison.json`: paper-facing tables.
"""
    readme_path = output_root / "README.md"
    if readme_path.exists():
        raise FileExistsError(f"refusing to overwrite {readme_path}")
    readme_path.write_text(readme, encoding="utf-8")
    print(f"[finalize] wrote {comparison_path}, {csv_path}, and {readme_path}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    manifest.add_argument("--protocol-seed", type=int, default=DEFAULT_PROTOCOL_SEED)
    manifest.add_argument("--per-source", type=int, default=50)
    manifest.add_argument("--flow-seeds", type=int, nargs="+", default=list(DEFAULT_FLOW_SEEDS))
    manifest.set_defaults(function=create_manifest)

    model = subparsers.add_parser("run-model")
    model.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    model.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    model.add_argument("--flow-seeds", type=int, nargs="+", default=list(DEFAULT_FLOW_SEEDS))
    model.set_defaults(function=run_model)

    rescore = subparsers.add_parser("rescore-model")
    rescore.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    rescore.add_argument("--prediction-root", type=Path, required=True)
    rescore.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    rescore.add_argument("--flow-seeds", type=int, nargs="+", default=list(DEFAULT_FLOW_SEEDS))
    rescore.set_defaults(function=rescore_model)

    baselines = subparsers.add_parser("baselines")
    baselines.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    baselines.set_defaults(function=run_baselines)

    final = subparsers.add_parser("finalize")
    final.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    final.set_defaults(function=finalize)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if hasattr(args, "flow_seeds"):
        args.flow_seeds = [int(seed) for seed in args.flow_seeds]
        if len(args.flow_seeds) != len(set(args.flow_seeds)) or any(seed < 0 for seed in args.flow_seeds):
            raise ValueError(f"flow seeds must be unique non-negative integers: {args.flow_seeds}")
    try:
        args.function(args)
    except Exception as exc:
        output_root = getattr(args, "output_root", None)
        if output_root is not None:
            failure = Path(output_root).resolve() / f"failure_{args.command}.json"
            write_json_atomic(
                failure,
                {
                    "status": "failed",
                    "command": args.command,
                    "model": getattr(args, "model", None),
                    "failed_at_utc": utc_now(),
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "argv": sys.argv,
                },
            )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
