#!/usr/bin/env python
"""Independently verify frozen dissertation evidence and write a QA report."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from openpi.training import config as training_config
from openpi.training import hrc_norm_stats
from scripts.hrc.compute_hrc_norm_stats import DELTA_MAPPINGS
from scripts.hrc.compute_hrc_norm_stats import _load_numpy_columns
from scripts.hrc.compute_hrc_norm_stats import _validate_train_only
from scripts.hrc.dissertation_protocol import ACTION_DIM
from scripts.hrc.dissertation_protocol import ACTION_GROUPS
from scripts.hrc.dissertation_protocol import ACTION_HORIZON
from scripts.hrc.dissertation_protocol import DELTA_DIMS
from scripts.hrc.dissertation_protocol import GRIPPER_DIMS
from scripts.hrc.dissertation_protocol import file_sha256
from scripts.hrc.dissertation_protocol import load_split_rows
from scripts.hrc.dissertation_protocol import write_json_atomic
from scripts.hrc.run_dissertation_conditioning_probe import DEFAULT_OUTPUT_ROOT as PROBE_ROOT
from scripts.hrc.run_dissertation_final_audit import DEFAULT_FLOW_SEEDS
from scripts.hrc.run_dissertation_final_audit import DEFAULT_OUTPUT_ROOT as AUDIT_ROOT
from scripts.hrc.run_dissertation_final_audit import MODEL_SPECS
from scripts.hrc.run_dissertation_final_audit import REPO_ROOT
from scripts.hrc.run_dissertation_final_audit import SPLIT_MANIFEST
from scripts.hrc.run_dissertation_target_visibility import DEFAULT_OUTPUT_ROOT as VISIBILITY_ROOT

DEFAULT_OUTPUT = REPO_ROOT / "outputs/dissertation_quality_control/qa_report.json"
PROTOCOL_SEED = 20260907
SPOT_CHECK_COUNT = 10


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _events(sign_rows: np.ndarray, previous_sign: np.ndarray) -> list[tuple[int, int, float]]:
    earlier = np.vstack((previous_sign[None, :], sign_rows[:-1]))
    return [
        (int(frame), int(gripper), float(sign_rows[frame, gripper]))
        for frame, gripper in np.argwhere(sign_rows != earlier)
    ]


def independent_anchor_score(
    predicted: np.ndarray,
    expert: np.ndarray,
    state: np.ndarray,
    previous: np.ndarray,
    *,
    tolerance: int = 8,
) -> dict[str, float | int | None]:
    """Reimplement the paper metrics without calling the production scorer."""
    predicted = np.asarray(predicted, dtype=np.float64)
    expert = np.asarray(expert, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    previous = np.asarray(previous, dtype=np.float64)
    if predicted.shape != expert.shape or predicted.ndim != 2 or predicted.shape[1] != ACTION_DIM:
        raise ValueError("invalid action chunks")
    null = np.zeros_like(expert)
    null[:, list(DELTA_DIMS)] = state[list(DELTA_DIMS)]
    null[:, list(GRIPPER_DIMS)] = previous[list(GRIPPER_DIMS)]
    result: dict[str, float | int | None] = {}
    for group, dims in ACTION_GROUPS.items():
        raw = float(np.mean(np.abs(predicted[:, list(dims)] - expert[:, list(dims)])))
        null_mae = float(np.mean(np.abs(null[:, list(dims)] - expert[:, list(dims)])))
        result[f"{group}_raw_mae"] = raw
        result[f"{group}_null_mae"] = null_mae
        result[f"{group}_skill"] = 1.0 - raw / null_mae if null_mae > 0 else None
    model_motion = float(
        np.mean(np.abs(predicted[:, list(DELTA_DIMS)] - state[list(DELTA_DIMS)]))
    )
    expert_motion = float(
        np.mean(np.abs(expert[:, list(DELTA_DIMS)] - state[list(DELTA_DIMS)]))
    )
    result["arm_motion_model"] = model_motion
    result["arm_motion_expert"] = expert_motion
    result["arm_motion_ratio"] = model_motion / expert_motion if expert_motion > 0 else None

    previous_sign = np.sign(previous[list(GRIPPER_DIMS)])
    predicted_events = _events(np.sign(predicted[:, list(GRIPPER_DIMS)]), previous_sign)
    expert_events = _events(np.sign(expert[:, list(GRIPPER_DIMS)]), previous_sign)
    unmatched = list(expert_events)
    true_positives = 0
    for frame, gripper, direction in predicted_events:
        for index, (expert_frame, expert_gripper, expert_direction) in enumerate(unmatched):
            if (
                gripper == expert_gripper
                and direction == expert_direction
                and abs(frame - expert_frame) <= tolerance
            ):
                true_positives += 1
                unmatched.pop(index)
                break
    result["gripper_expert_events"] = len(expert_events)
    result["gripper_predicted_events"] = len(predicted_events)
    result["gripper_true_positives"] = true_positives
    result["gripper_precision"] = (
        true_positives / len(predicted_events) if predicted_events else None
    )
    result["gripper_recall"] = true_positives / len(expert_events) if expert_events else None
    result["gripper_f1"] = (
        2 * true_positives / (len(predicted_events) + len(expert_events))
        if expert_events
        else None
    )
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _compare_value(actual: float | int | None, serialized: str, label: str) -> float:
    if actual is None:
        if serialized != "":
            raise AssertionError(f"{label}: expected empty value, got {serialized!r}")
        return 0.0
    if serialized == "":
        raise AssertionError(f"{label}: unexpectedly empty")
    difference = abs(float(actual) - float(serialized))
    scale = max(abs(float(actual)), abs(float(serialized)), 1.0)
    scaled_difference = difference / scale
    if difference > 1e-7 and scaled_difference > 1e-6:
        raise AssertionError(
            f"{label}: independent difference {difference} (scaled {scaled_difference})"
        )
    return scaled_difference


def _audit_checks(split: dict[str, Any]) -> dict[str, Any]:
    manifest_path = AUDIT_ROOT / "anchor_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = file_sha256(manifest_path)
    val_sources = sorted(int(value) for value in split["source_splits"]["val"])
    anchor_splits = {str(anchor["split"]) for anchor in manifest["anchors"]}
    if anchor_splits != {"val"} or manifest["source_ids"] != val_sources:
        raise AssertionError("audit manifest is not bound to the frozen validation sources")
    if len(manifest["anchors"]) != 652:
        raise AssertionError("audit manifest does not contain 652 anchors")
    rng = np.random.default_rng(PROTOCOL_SEED)
    spot_indices = sorted(
        int(value) for value in rng.choice(len(manifest["anchors"]), SPOT_CHECK_COUNT, replace=False)
    )
    spot_ids = [str(manifest["anchors"][index]["anchor_id"]) for index in spot_indices]
    model_reports = {}
    for label, spec in MODEL_SPECS.items():
        summary = json.loads((AUDIT_ROOT / label / "summary.json").read_text(encoding="utf-8"))
        if summary["manifest_sha256"] != manifest_sha:
            raise AssertionError(f"{label} summary manifest mismatch")
        if summary["flow_seeds"] != list(DEFAULT_FLOW_SEEDS):
            raise AssertionError(f"{label} flow seeds mismatch")
        if summary["determinism_max_abs_diff"] != 0.0:
            raise AssertionError(f"{label} fixed-noise determinism failed")
        rows = load_split_rows(spec.split_root)
        max_difference = 0.0
        seed_reports = {}
        for seed in DEFAULT_FLOW_SEEDS:
            seed_root = AUDIT_ROOT / label / "per_seed" / f"seed_{seed}"
            with np.load(seed_root / "predictions.npz", allow_pickle=False) as archive:
                predictions = np.asarray(archive["predictions"], dtype=np.float32)
                anchor_ids = [str(value) for value in archive["anchor_ids"]]
            expected_ids = [str(anchor["anchor_id"]) for anchor in manifest["anchors"]]
            if predictions.shape != (652, ACTION_HORIZON, ACTION_DIM):
                raise AssertionError(f"{label} seed {seed} prediction shape mismatch")
            if anchor_ids != expected_ids or not np.isfinite(predictions).all():
                raise AssertionError(f"{label} seed {seed} prediction identity/finite check failed")
            csv_rows = _read_csv(seed_root / "per_anchor.csv")
            if [row["anchor_id"] for row in csv_rows] != expected_ids:
                raise AssertionError(f"{label} seed {seed} CSV anchor order mismatch")
            if seed == DEFAULT_FLOW_SEEDS[0]:
                for index in spot_indices:
                    anchor = manifest["anchors"][index]
                    row_index = int(anchor[spec.row_field])
                    source = int(anchor["source_episode_id"])
                    previous = (
                        rows.action[row_index - 1]
                        if row_index > 0 and int(rows.source_id[row_index - 1]) == source
                        else rows.action[row_index]
                    )
                    expert = rows.action[row_index : row_index + ACTION_HORIZON]
                    independent = independent_anchor_score(
                        predictions[index], expert, rows.state[row_index], previous
                    )
                    for metric, value in independent.items():
                        max_difference = max(
                            max_difference,
                            _compare_value(
                                value,
                                csv_rows[index][metric],
                                f"{label}/{seed}/{anchor['anchor_id']}/{metric}",
                            ),
                        )
            seed_reports[str(seed)] = {
                "num_predictions": len(predictions),
                "all_finite": True,
                "anchor_order_exact": True,
                "near_zero_null_mae_anchors": {
                    group: sum(abs(float(row[f"{group}_null_mae"])) < 1e-12 for row in csv_rows)
                    for group in ACTION_GROUPS
                },
            }
        model_reports[label] = {
            "determinism_max_abs_diff": 0.0,
            "manifest_sha256": manifest_sha,
            "independent_spot_check_max_scaled_difference": max_difference,
            "independent_spot_check_tolerance": {"absolute": 1e-7, "relative": 1e-6},
            "seeds": seed_reports,
        }
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "split": "val",
        "source_ids": val_sources,
        "num_anchors": len(manifest["anchors"]),
        "spot_check_anchor_ids": spot_ids,
        "models": model_reports,
    }


def _recompute_stats(config_name: str) -> tuple[dict[str, Any], set[int], Any]:
    config = training_config.get_config(config_name)
    factory = config.data
    columns = _load_numpy_columns(Path(factory.dataset_root), str(factory.repo_id), factory.state_key)
    _validate_train_only(factory, columns)
    sources = {int(value) for value in np.unique(columns["hrc.source_episode_id"])}
    stats = hrc_norm_stats.compute_hrc_norm_stats_from_arrays(
        states=columns[factory.state_key],
        actions=columns["action"],
        valid_mask=columns["hrc.valid_anchor"],
        task_ids=columns["hrc.task_id"],
        action_horizon=config.model.action_horizon,
        delta_mappings=DELTA_MAPPINGS,
        task_weights=factory.task_weights,
    )
    return stats, sources, factory


def _stats_max_difference(
    computed: dict[str, Any],
    saved_path: Path,
    *,
    check_quantiles: bool,
) -> float:
    saved = json.loads(saved_path.read_text(encoding="utf-8"))["norm_stats"]
    maximum = 0.0
    for key in ("state", "actions"):
        for statistic in ("mean", "std", "q01", "q99"):
            if statistic.startswith("q") and not check_quantiles:
                continue
            computed_value = getattr(computed[key], statistic)
            saved_value = saved[key][statistic]
            if computed_value is None or saved_value is None:
                if computed_value is not None or saved_value is not None:
                    raise AssertionError(f"normalization {key}.{statistic} presence mismatch")
                continue
            maximum = max(
                maximum,
                float(np.max(np.abs(np.asarray(computed_value) - np.asarray(saved_value)))),
            )
    if maximum > 1e-10:
        raise AssertionError(f"normalization stats differ by {maximum}")
    return maximum


def _normalization_checks(split: dict[str, Any]) -> dict[str, Any]:
    expected_train = {int(value) for value in split["source_splits"]["train"]}
    a_stats, a_sources, a_factory = _recompute_stats(MODEL_SPECS["A20"].config_name)
    bc_stats, bc_sources, bc_factory = _recompute_stats(MODEL_SPECS["C20"].config_name)
    if a_sources != expected_train or bc_sources != expected_train:
        raise AssertionError("normalization arrays do not exactly contain the frozen train sources")
    assets = {
        "A20": Path(a_factory.assets.assets_dir) / str(a_factory.assets.asset_id) / "norm_stats.json",
        "B20": Path(bc_factory.assets.assets_dir) / "subinstr_rerun_v1/norm_stats.json",
        "C20": Path(bc_factory.assets.assets_dir) / str(bc_factory.assets.asset_id) / "norm_stats.json",
    }
    differences = {
        "A20": _stats_max_difference(a_stats, assets["A20"], check_quantiles=False),
        "B20": _stats_max_difference(bc_stats, assets["B20"], check_quantiles=False),
        "C20": _stats_max_difference(bc_stats, assets["C20"], check_quantiles=True),
    }
    hashes = {}
    for label, spec in MODEL_SPECS.items():
        checkpoint_assets = list((spec.checkpoint / "assets").glob("*/norm_stats.json"))
        if len(checkpoint_assets) != 1:
            raise AssertionError(f"{label} checkpoint has {len(checkpoint_assets)} norm assets")
        if file_sha256(checkpoint_assets[0]) != file_sha256(assets[label]):
            raise AssertionError(f"{label} checkpoint and repository norm stats differ")
        hashes[label] = {
            "repository_asset": str(assets[label]),
            "checkpoint_asset": str(checkpoint_assets[0]),
            "sha256": file_sha256(assets[label]),
            "recomputed_max_abs_diff": differences[label],
        }
    return {
        "train_source_ids": sorted(expected_train),
        "val_test_source_overlap": [],
        "assets": hashes,
    }


def _conditioning_checks(split: dict[str, Any]) -> dict[str, Any]:
    manifest_path = PROBE_ROOT / "anchor_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_sources = {int(value) for value in split["source_splits"]["test"]}
    observed_sources = {int(anchor["source_episode_id"]) for anchor in manifest["anchors"]}
    if observed_sources & test_sources:
        raise AssertionError("conditioning manifest contains test sources")
    if len(manifest["anchors"]) != 324:
        raise AssertionError("conditioning manifest does not contain 324 anchors")
    reports = {}
    for label in ("B20", "C20"):
        summary = json.loads((PROBE_ROOT / label / "summary.json").read_text(encoding="utf-8"))
        rows = _read_csv(PROBE_ROOT / label / "per_anchor.csv")
        ratio_fields = [field for field in rows[0] if field.endswith("_over_noise")]
        nonfinite = sum(
            not np.isfinite(float(row[field])) for row in rows for field in ratio_fields
        )
        if summary["determinism_max_abs_diff"] != 0.0 or len(rows) != 324 or nonfinite:
            raise AssertionError(f"{label} conditioning completeness/finite check failed")
        reports[label] = {
            "num_anchors": len(rows),
            "num_inferences": summary["num_inferences"],
            "determinism_max_abs_diff": summary["determinism_max_abs_diff"],
            "nonfinite_ratios": nonfinite,
        }
    return {
        "manifest": str(manifest_path),
        "source_ids": sorted(observed_sources),
        "test_source_overlap": [],
        "num_anchors": len(manifest["anchors"]),
        "models": reports,
    }


def _visibility_checks() -> dict[str, Any]:
    summary = json.loads((VISIBILITY_ROOT / "summary.json").read_text(encoding="utf-8"))
    if summary["missing_files"] or summary["num_sources"] != 45:
        raise AssertionError("visibility population is incomplete")
    frame_count = sum(1 for _ in (VISIBILITY_ROOT / "per_frame.csv").open(encoding="utf-8")) - 1
    anchor_count = sum(1 for _ in (VISIBILITY_ROOT / "per_anchor.csv").open(encoding="utf-8")) - 1
    if frame_count != summary["num_relevant_frames"]:
        raise AssertionError("visibility per-frame row count mismatch")
    if anchor_count != summary["num_valid_relevant_anchors"]:
        raise AssertionError("visibility per-anchor row count mismatch")
    return {
        "num_sources": summary["num_sources"],
        "num_relevant_frames": frame_count,
        "num_valid_relevant_anchors": anchor_count,
        "missing_files": summary["missing_files"],
        "occlusion_considered": summary["visibility_definition"]["occlusion_considered"],
    }


def run(output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    source_sets = {
        name: {int(value) for value in values}
        for name, values in split["source_splits"].items()
    }
    if any(source_sets[left] & source_sets[right] for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise AssertionError("frozen source splits overlap")
    report = {
        "schema_version": 1,
        "status": "passed",
        "completed_at_utc": utc_now(),
        "protocol_seed": PROTOCOL_SEED,
        "split_manifest": str(SPLIT_MANIFEST),
        "split_manifest_sha256": file_sha256(SPLIT_MANIFEST),
        "source_split_counts": {key: len(value) for key, value in source_sets.items()},
        "audit": _audit_checks(split),
        "normalization": _normalization_checks(split),
        "conditioning": _conditioning_checks(split),
        "visibility": _visibility_checks(),
    }
    write_json_atomic(output, report)
    print(f"[qa] all checks passed; wrote {output}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
