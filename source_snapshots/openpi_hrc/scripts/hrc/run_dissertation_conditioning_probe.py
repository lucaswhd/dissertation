#!/usr/bin/env python
"""Run the expanded, fixed-noise dissertation observation-conditioning probe."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from scripts.hrc.dissertation_protocol import ACTION_DIM
from scripts.hrc.dissertation_protocol import ACTION_GROUPS
from scripts.hrc.dissertation_protocol import ACTION_HORIZON
from scripts.hrc.dissertation_protocol import DELTA_DIMS
from scripts.hrc.dissertation_protocol import SplitRows
from scripts.hrc.dissertation_protocol import bootstrap_summary
from scripts.hrc.dissertation_protocol import file_sha256
from scripts.hrc.dissertation_protocol import flow_noise
from scripts.hrc.dissertation_protocol import load_split_rows
from scripts.hrc.dissertation_protocol import prompt_stable_mask
from scripts.hrc.dissertation_protocol import semantic_label
from scripts.hrc.dissertation_protocol import write_csv_atomic
from scripts.hrc.dissertation_protocol import write_json_atomic
from scripts.hrc.evaluate_hrc_pi05_continuous import _decode_source_frames
from scripts.hrc.evaluate_hrc_pi05_continuous import _load_policy
from scripts.hrc.run_dissertation_final_audit import BC_DATASET_VERSION
from scripts.hrc.run_dissertation_final_audit import DATA_ROOT
from scripts.hrc.run_dissertation_final_audit import MODEL_SPECS
from scripts.hrc.run_dissertation_final_audit import REPO_ROOT
from scripts.hrc.run_dissertation_final_audit import SPLIT_MANIFEST
from scripts.hrc.run_dissertation_final_audit import _validate_runtime_config

SUBINSTRUCTION_PLAN = DATA_ROOT / "hrc_subinstructions_rerun_v1.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/dissertation_conditioning_probe"
DEFAULT_PROTOCOL_SEED = 20260907
POSITIONS = (("early", 0.25), ("middle", 0.50), ("late", 0.75))
PROBE_MODELS = ("B20", "C20")
PERTURBATIONS = ("image", "state", "noise", "hflip", "black", "white", "random_image")
RATIO_PERTURBATIONS = ("image", "state", "hflip", "black", "white", "random_image")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_label(value: str) -> str:
    return semantic_label(value).strip().rstrip(".").casefold()


def choose_segment_positions(
    rows: SplitRows,
    stable_mask: np.ndarray,
    *,
    source_id: int,
    context_id: int,
    start: int,
    stop: int,
    label: str,
    fractions: tuple[float, ...] = tuple(value for _, value in POSITIONS),
) -> list[dict[str, Any]]:
    """Pick the nearest valid, prompt-stable row to each requested segment fraction."""
    if stop <= start:
        raise ValueError(f"empty segment [{start}, {stop})")
    if len(fractions) != len(POSITIONS):
        raise ValueError(f"expected {len(POSITIONS)} fractions, got {len(fractions)}")
    stable_mask = np.asarray(stable_mask, dtype=bool)
    candidates = np.flatnonzero(
        stable_mask
        & rows.valid
        & (rows.source_id == source_id)
        & (rows.context_id == context_id)
        & (rows.raw_frame >= start)
        & (rows.raw_frame < stop)
        & np.asarray([_normalize_label(str(prompt)) == _normalize_label(label) for prompt in rows.prompts])
    )
    if not len(candidates):
        raise ValueError("no valid prompt-stable rows with the requested source/context/label")
    selected = []
    used: set[int] = set()
    for (position, _), fraction in zip(POSITIONS, fractions, strict=True):
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"segment fraction must be in (0,1), got {fraction}")
        target = start + fraction * (stop - start)
        ordering = np.lexsort((rows.raw_frame[candidates], np.abs(rows.raw_frame[candidates] - target)))
        row = next((int(candidates[index]) for index in ordering if int(candidates[index]) not in used), None)
        if row is None:
            raise ValueError("segment has fewer than three distinct eligible rows")
        used.add(row)
        selected.append(
            {
                "position": position,
                "fraction": float(fraction),
                "target_raw_frame": float(target),
                "row": row,
                "raw_global_frame_index": int(rows.raw_frame[row]),
            }
        )
    return selected


def assign_donors(anchors: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """Attach a deterministic same-stratum, different-source donor to each anchor."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for anchor in anchors:
        key = (str(anchor["split"]), str(anchor["semantic_label"]), str(anchor["position"]))
        grouped[key].append(anchor)
    result = []
    for anchor in anchors:
        key = (str(anchor["split"]), str(anchor["semantic_label"]), str(anchor["position"]))
        candidates = sorted(
            (
                candidate
                for candidate in grouped[key]
                if int(candidate["source_episode_id"]) != int(anchor["source_episode_id"])
            ),
            key=lambda item: str(item["anchor_id"]),
        )
        if not candidates:
            raise ValueError(f"no cross-source donor for {anchor['anchor_id']} in stratum {key}")
        digest = hashlib.sha256(f"{seed}|{anchor['anchor_id']}".encode()).digest()
        donor = candidates[int.from_bytes(digest[:8], "little") % len(candidates)]
        enriched = dict(anchor)
        enriched["donor_anchor_id"] = donor["anchor_id"]
        enriched["donor_source_episode_id"] = int(donor["source_episode_id"])
        enriched["donor_raw_global_frame_index"] = int(donor["raw_global_frame_index"])
        enriched["donor_row"] = int(donor["row"])
        result.append(enriched)
    return result


def comparison_space(prediction: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Convert physical output to a coordinate-free arm-delta comparison space."""
    output = np.asarray(prediction, dtype=np.float32).copy()
    state = np.asarray(state, dtype=np.float32)
    if output.shape != (ACTION_HORIZON, ACTION_DIM) and output.shape[-1] != ACTION_DIM:
        raise ValueError(f"prediction must end in {ACTION_DIM} dimensions, got {output.shape}")
    if state.shape != (ACTION_DIM,):
        raise ValueError(f"state must have shape {(ACTION_DIM,)}, got {state.shape}")
    output[..., list(DELTA_DIMS)] -= state[list(DELTA_DIMS)]
    return output


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def _select_train_sources(split: dict[str, Any], *, seed: int, count: int = 8) -> list[int]:
    train = np.asarray(split["source_splits"]["train"], dtype=np.int64)
    if len(train) < count:
        raise ValueError(f"frozen train split has only {len(train)} sources, need {count}")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 901]))
    return sorted(int(value) for value in rng.choice(train, size=count, replace=False))


def create_manifest(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    path = output_root / "anchor_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("protocol_seed") != args.protocol_seed:
            raise ValueError(f"existing conditioning manifest uses another protocol seed: {path}")
        print(f"[probe-manifest] reusing {path} ({existing['num_anchors']} anchors)", flush=True)
        return
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    plan = json.loads(SUBINSTRUCTION_PLAN.read_text(encoding="utf-8"))
    source_sets = {
        "val": [int(value) for value in split["source_splits"]["val"]],
        "train_coverage": _select_train_sources(split, seed=args.protocol_seed),
    }
    rows_by_split = {
        "val": load_split_rows(DATA_ROOT / BC_DATASET_VERSION / "val"),
        "train_coverage": load_split_rows(DATA_ROOT / BC_DATASET_VERSION / "train"),
    }
    stable_by_split = {name: prompt_stable_mask(rows) for name, rows in rows_by_split.items()}
    anchors: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for split_name, source_ids in source_sets.items():
        rows = rows_by_split[split_name]
        stable = stable_by_split[split_name]
        for source_id in source_ids:
            source_plan = plan["sources"][str(source_id)]
            segment_specs = []
            for index, segment in enumerate(source_plan["0"]["segments"]):
                if "stand by" not in segment["label"].casefold():
                    segment_specs.append(("pickup", 1, 0, index, segment))
            for index, segment in enumerate(source_plan["1"]["segments"]):
                if _normalize_label(segment["label"]) == _normalize_label("Turn on the radio receiver."):
                    segment_specs.append(("radio", 3, 1, index, segment))
            for context, context_id, round_id, segment_index, segment in segment_specs:
                try:
                    choices = choose_segment_positions(
                        rows,
                        stable,
                        source_id=source_id,
                        context_id=context_id,
                        start=int(segment["start"]),
                        stop=int(segment["stop"]),
                        label=str(segment["label"]),
                    )
                except ValueError as exc:
                    rejected.append(
                        {
                            "split": split_name,
                            "source_episode_id": source_id,
                            "context": context,
                            "round_id": round_id,
                            "segment_index": segment_index,
                            "segment_label": str(segment["label"]),
                            "start": int(segment["start"]),
                            "stop": int(segment["stop"]),
                            "reason": str(exc),
                        }
                    )
                    continue
                for choice in choices:
                    row = int(choice["row"])
                    anchor_id = (
                        f"{split_name}_s{source_id:04d}_c{context_id}_r{round_id}_"
                        f"g{segment_index:02d}_{choice['position']}"
                    )
                    anchors.append(
                        {
                            "anchor_id": anchor_id,
                            "split": split_name,
                            "dataset_split": "val" if split_name == "val" else "train",
                            "source_episode_id": source_id,
                            "row": row,
                            "raw_global_frame_index": int(choice["raw_global_frame_index"]),
                            "context": context,
                            "context_id": context_id,
                            "round_id": round_id,
                            "segment_index": segment_index,
                            "segment_label": str(segment["label"]),
                            "semantic_label": semantic_label(str(rows.prompts[row])),
                            "segment_start": int(segment["start"]),
                            "segment_stop": int(segment["stop"]),
                            "segment_length": int(segment["stop"] - segment["start"]),
                            "position": choice["position"],
                            "position_fraction": choice["fraction"],
                            "target_raw_frame": choice["target_raw_frame"],
                            "prompt": str(rows.prompts[row]),
                            "valid_status": {
                                "export_valid": bool(rows.valid[row]),
                                "prompt_context_stable_through_horizon": bool(stable[row]),
                            },
                        }
                    )
    anchors.sort(
        key=lambda item: (
            item["split"], item["source_episode_id"], item["context_id"],
            item["segment_index"], item["position_fraction"],
        )
    )
    anchors = assign_donors(anchors, seed=args.protocol_seed)
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "protocol_seed": args.protocol_seed,
        "dataset_version": BC_DATASET_VERSION,
        "split_manifest": str(SPLIT_MANIFEST),
        "split_manifest_sha256": file_sha256(SPLIT_MANIFEST),
        "subinstruction_plan": str(SUBINSTRUCTION_PLAN),
        "subinstruction_plan_sha256": file_sha256(SUBINSTRUCTION_PLAN),
        "source_sets": source_sets,
        "positions": dict(POSITIONS),
        "reference_flow_seed": 0,
        "alternate_flow_seed": 1,
        "target_segments": "all non-idle pickup Execute subinstructions plus radio Turn on receiver",
        "num_anchors": len(anchors),
        "num_rejected_segments": len(rejected),
        "rejected_segments": rejected,
        "anchors": anchors,
    }
    write_json_atomic(path, manifest)
    print(
        f"[probe-manifest] wrote {path}: {len(anchors)} anchors, "
        f"{len(rejected)} rejected segments, sources={source_sets}",
        flush=True,
    )


def _predict(policy: Any, image: np.ndarray, state: np.ndarray, prompt: str, noise: np.ndarray) -> np.ndarray:
    result = policy.infer(
        {
            "observation/image_0": np.asarray(image, dtype=np.uint8),
            "observation/state": np.asarray(state, dtype=np.float32),
            "prompt": str(prompt),
        },
        noise=np.asarray(noise, dtype=np.float32),
    )
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(actions).all():
        raise ValueError(f"invalid policy actions shape/values: {actions.shape}")
    return actions


def _gather_images(manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for anchor in manifest["anchors"]:
        grouped[(anchor["dataset_split"], int(anchor["source_episode_id"]))].append(anchor)
    for order, ((dataset_split, source_id), anchors) in enumerate(sorted(grouped.items()), start=1):
        split_root = DATA_ROOT / BC_DATASET_VERSION / dataset_split
        frames = _decode_source_frames(split_root, source_id)
        for anchor in anchors:
            frame = int(anchor["raw_global_frame_index"])
            if frame >= len(frames):
                raise ValueError(f"{anchor['anchor_id']} frame {frame} exceeds video length {len(frames)}")
            images[anchor["anchor_id"]] = np.array(frames[frame], dtype=np.uint8, copy=True)
        print(
            f"[probe-images] {dataset_split} source {source_id} ({order}/{len(grouped)}): "
            f"kept {len(anchors)} of {len(frames)} frames",
            flush=True,
        )
        del frames
    return images


def _difference(a: np.ndarray, b: np.ndarray, dims: tuple[int, ...]) -> float:
    return float(np.mean(np.abs(a[:, list(dims)] - b[:, list(dims)])))


def _metric_columns() -> list[str]:
    columns = []
    for group in ACTION_GROUPS:
        columns.extend(f"{group}_D_{name}" for name in PERTURBATIONS)
        columns.extend(f"{group}_{name}_over_noise" for name in RATIO_PERTURBATIONS)
    return columns


PER_ANCHOR_FIELDS = [
    "checkpoint",
    "split",
    "dataset_split",
    "anchor_id",
    "source_episode_id",
    "raw_global_frame_index",
    "context",
    "segment_index",
    "segment",
    "position",
    "position_fraction",
    "donor_anchor_id",
    "donor_source_episode_id",
    "donor_raw_global_frame_index",
    "image_donor_l1_uint8",
    "state_donor_l1",
    *_metric_columns(),
]


def _random_image(shape: tuple[int, ...], *, seed: int, source_id: int, frame: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 777, source_id, frame]))
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def _summary_seed(parts: tuple[Any, ...]) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _aggregate_table(
    rows: list[dict[str, Any]],
    *,
    group_fields: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, members in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        common = dict(zip(group_fields, key, strict=True))
        for metric in _metric_columns():
            values = [member[metric] for member in members if member[metric] is not None]
            stats = bootstrap_summary(values, seed=_summary_seed((*key, metric)))
            output.append({**common, "metric": metric, **stats})
    return output


def run_model(args: argparse.Namespace) -> None:
    spec = MODEL_SPECS[args.model]
    if spec.label not in PROBE_MODELS:
        raise ValueError(f"conditioning probe supports {PROBE_MODELS}, not {spec.label}")
    _validate_runtime_config(spec)
    output_root = args.output_root.resolve()
    manifest_path = output_root / "anchor_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = file_sha256(manifest_path)
    model_root = output_root / spec.label
    if (model_root / "summary.json").exists():
        raise FileExistsError(f"refusing to overwrite {model_root / 'summary.json'}")
    rows_by_split = {
        "val": load_split_rows(DATA_ROOT / BC_DATASET_VERSION / "val"),
        "train": load_split_rows(DATA_ROOT / BC_DATASET_VERSION / "train"),
    }
    for anchor in manifest["anchors"]:
        rows = rows_by_split[anchor["dataset_split"]]
        row = int(anchor["row"])
        if int(rows.source_id[row]) != anchor["source_episode_id"] or int(rows.raw_frame[row]) != anchor["raw_global_frame_index"]:
            raise ValueError(f"manifest row mismatch for {anchor['anchor_id']}")
        if str(rows.prompts[row]) != anchor["prompt"]:
            raise ValueError(f"manifest prompt mismatch for {anchor['anchor_id']}")
    images = _gather_images(manifest)
    policy = _load_policy(spec.config_name, spec.checkpoint)

    first_anchor = manifest["anchors"][0]
    first_rows = rows_by_split[first_anchor["dataset_split"]]
    first_state = first_rows.state[int(first_anchor["row"])]
    reference_noise = flow_noise(
        manifest["protocol_seed"], manifest["reference_flow_seed"],
        int(first_anchor["source_episode_id"]), int(first_anchor["raw_global_frame_index"]),
    )
    first_prediction = _predict(policy, images[first_anchor["anchor_id"]], first_state, first_anchor["prompt"], reference_noise)
    repeated_prediction = _predict(policy, images[first_anchor["anchor_id"]], first_state, first_anchor["prompt"], reference_noise)
    determinism = float(np.max(np.abs(first_prediction - repeated_prediction)))
    if determinism != 0.0:
        raise RuntimeError(f"fixed-noise determinism failed for {spec.label}: {determinism}")
    print(f"[{spec.label} probe] fixed-noise determinism max abs diff = {determinism:.1f}", flush=True)

    rows_out = []
    by_id = {anchor["anchor_id"]: anchor for anchor in manifest["anchors"]}
    for index, anchor in enumerate(manifest["anchors"], start=1):
        donor = by_id[anchor["donor_anchor_id"]]
        rows = rows_by_split[anchor["dataset_split"]]
        donor_rows = rows_by_split[donor["dataset_split"]]
        state = rows.state[int(anchor["row"])]
        donor_state = donor_rows.state[int(donor["row"])]
        image = images[anchor["anchor_id"]]
        donor_image = images[donor["anchor_id"]]
        if anchor["prompt"] != donor["prompt"]:
            raise ValueError(f"donor prompt mismatch for {anchor['anchor_id']} -> {donor['anchor_id']}")
        noise = flow_noise(
            manifest["protocol_seed"], manifest["reference_flow_seed"],
            int(anchor["source_episode_id"]), int(anchor["raw_global_frame_index"]),
        )
        alternate_noise = flow_noise(
            manifest["protocol_seed"], manifest["alternate_flow_seed"],
            int(anchor["source_episode_id"]), int(anchor["raw_global_frame_index"]),
        )
        random_image = _random_image(
            image.shape,
            seed=manifest["protocol_seed"],
            source_id=int(anchor["source_episode_id"]),
            frame=int(anchor["raw_global_frame_index"]),
        )
        predictions = {
            "reference": (_predict(policy, image, state, anchor["prompt"], noise), state),
            "image": (_predict(policy, donor_image, state, anchor["prompt"], noise), state),
            "state": (_predict(policy, image, donor_state, anchor["prompt"], noise), donor_state),
            "noise": (_predict(policy, image, state, anchor["prompt"], alternate_noise), state),
            "hflip": (_predict(policy, np.ascontiguousarray(image[:, ::-1]), state, anchor["prompt"], noise), state),
            "black": (_predict(policy, np.zeros_like(image), state, anchor["prompt"], noise), state),
            "white": (_predict(policy, np.full_like(image, 255), state, anchor["prompt"], noise), state),
            "random_image": (_predict(policy, random_image, state, anchor["prompt"], noise), state),
        }
        comparable = {
            name: comparison_space(prediction, fed_state)
            for name, (prediction, fed_state) in predictions.items()
        }
        output = {
            "checkpoint": spec.label,
            "split": anchor["split"],
            "dataset_split": anchor["dataset_split"],
            "anchor_id": anchor["anchor_id"],
            "source_episode_id": anchor["source_episode_id"],
            "raw_global_frame_index": anchor["raw_global_frame_index"],
            "context": anchor["context"],
            "segment_index": anchor["segment_index"],
            "segment": anchor["semantic_label"],
            "position": anchor["position"],
            "position_fraction": anchor["position_fraction"],
            "donor_anchor_id": donor["anchor_id"],
            "donor_source_episode_id": donor["source_episode_id"],
            "donor_raw_global_frame_index": donor["raw_global_frame_index"],
            "image_donor_l1_uint8": float(
                np.mean(np.abs(image.astype(np.float32) - donor_image.astype(np.float32)))
            ),
            "state_donor_l1": float(np.mean(np.abs(state - donor_state))),
        }
        for group, dims in ACTION_GROUPS.items():
            for perturbation in PERTURBATIONS:
                output[f"{group}_D_{perturbation}"] = _difference(
                    comparable["reference"], comparable[perturbation], dims
                )
            noise_floor = output[f"{group}_D_noise"]
            for perturbation in RATIO_PERTURBATIONS:
                output[f"{group}_{perturbation}_over_noise"] = safe_ratio(
                    output[f"{group}_D_{perturbation}"], noise_floor
                )
        rows_out.append(output)
        if index % 20 == 0 or index == len(manifest["anchors"]):
            print(f"[{spec.label} probe] {index}/{len(manifest['anchors'])} anchors", flush=True)

    model_root.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(model_root / "per_anchor.csv", rows_out, PER_ANCHOR_FIELDS)
    by_position = _aggregate_table(
        rows_out,
        group_fields=["checkpoint", "split", "context", "segment", "position"],
    )
    by_segment = _aggregate_table(
        rows_out,
        group_fields=["checkpoint", "split", "context", "segment"],
    )
    by_context = _aggregate_table(
        rows_out,
        group_fields=["checkpoint", "split", "context"],
    )
    by_source = _aggregate_table(
        rows_out,
        group_fields=["checkpoint", "split", "source_episode_id", "context"],
    )
    summary_fields = [
        "checkpoint", "split", "context", "segment", "position", "metric",
        "mean", "median", "std", "ci95_low", "ci95_high", "n",
    ]
    write_csv_atomic(model_root / "by_position.csv", by_position, summary_fields)
    write_csv_atomic(
        model_root / "by_segment.csv",
        by_segment,
        [field for field in summary_fields if field != "position"],
    )
    write_csv_atomic(
        model_root / "by_context.csv",
        by_context,
        [field for field in summary_fields if field not in {"segment", "position"}],
    )
    write_csv_atomic(
        model_root / "by_source.csv",
        by_source,
        [
            "checkpoint", "split", "source_episode_id", "context", "metric",
            "mean", "median", "std", "ci95_low", "ci95_high", "n",
        ],
    )
    report = {
        "schema_version": 1,
        "checkpoint": spec.label,
        "config": spec.config_name,
        "checkpoint_path": str(spec.checkpoint),
        "dataset_version": BC_DATASET_VERSION,
        "manifest_sha256": manifest_sha,
        "num_anchors": len(rows_out),
        "num_inferences": 2 + len(rows_out) * 8,
        "determinism_max_abs_diff": determinism,
        "reference_flow_seed": manifest["reference_flow_seed"],
        "alternate_flow_seed": manifest["alternate_flow_seed"],
        "by_context": by_context,
        "completed_at_utc": utc_now(),
    }
    write_json_atomic(model_root / "summary.json", report)
    print(f"[{spec.label} probe] wrote {model_root}", flush=True)


def _git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


def finalize(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    manifest_path = output_root / "anchor_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = file_sha256(manifest_path)
    all_position_rows = []
    summaries = {}
    for label in PROBE_MODELS:
        summary_path = output_root / label / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["manifest_sha256"] != manifest_sha or summary["determinism_max_abs_diff"] != 0.0:
            raise ValueError(f"{label} summary failed manifest/determinism validation")
        summaries[label] = summary
        import csv

        with (output_root / label / "by_position.csv").open(encoding="utf-8", newline="") as handle:
            all_position_rows.extend(list(csv.DictReader(handle)))
    combined_path = output_root / "by_position.csv"
    if combined_path.exists():
        raise FileExistsError(f"refusing to overwrite {combined_path}")
    fields = [
        "checkpoint", "split", "context", "segment", "position", "metric",
        "mean", "median", "std", "ci95_low", "ci95_high", "n",
    ]
    write_csv_atomic(combined_path, all_position_rows, fields)
    write_json_atomic(
        output_root / "summary.json",
        {
            "schema_version": 1,
            "manifest_sha256": manifest_sha,
            "models": summaries,
            "completed_at_utc": utc_now(),
        },
    )
    commit = _git_value("rev-parse", "HEAD")
    branch = _git_value("branch", "--show-current")
    dirty = bool(_git_value("status", "--short"))
    readme = f"""# Dissertation Observation-Conditioning Probe

Status: complete

## Provenance

- Git commit: `{commit}`
- Branch: `{branch}`
- Worktree at finalization: `{'dirty' if dirty else 'clean'}`
- Dataset: `{BC_DATASET_VERSION}`
- Primary split/source IDs: validation `{manifest['source_sets']['val']}`
- Separate coverage split/source IDs: train `{manifest['source_sets']['train_coverage']}`
- Test split used: no
- Anchor manifest SHA-256: `{manifest_sha}`
- Anchors: `{manifest['num_anchors']}`
- Rejected segments: `{manifest['num_rejected_segments']}` (listed in `anchor_manifest.json`)
- Completed UTC: `{utc_now()}`

## Protocol

Anchors are the nearest valid, 32-step prompt-stable rows to 25%, 50%, and 75% of every non-idle pickup Execute subinstruction and the radio `Turn on the radio receiver` segment. The primary result uses all four validation sources. Eight train sources selected once by protocol seed provide a separately labelled coverage check; they are never merged into a validation statistic.

For each anchor a donor from another source with the same split, semantic label, and relative position is fixed in the manifest. Reference, swap-image, swap-state, change-noise, horizontal-flip, black, white, and deterministic random-image conditions change one variable at a time. Reference noise is seed 0 and alternate noise is seed 1, both keyed by protocol seed/source/frame and passed explicitly to `Policy.infer`. Arms are compared after subtracting the state actually fed, so a swapped anchor pose cannot create a false response.

`D_*` is mean absolute action-output change from the reference over the named action group. `*_over_noise` divides that change by the same anchor's alternate-noise change. A zero denominator produces an undefined value rather than an infinite ratio. Summaries report mean, median, sample standard deviation, seeded percentile-bootstrap 95% CI, and n. The probe measures output sensitivity, not task success or causal reliance.

## Exact Commands

```bash
.venv/bin/python -m scripts.hrc.run_dissertation_conditioning_probe manifest
.venv/bin/python -m scripts.hrc.run_dissertation_conditioning_probe run-model --model B20
.venv/bin/python -m scripts.hrc.run_dissertation_conditioning_probe run-model --model C20
.venv/bin/python -m scripts.hrc.run_dissertation_conditioning_probe finalize
```

## Evidence

- `anchor_manifest.json`: source/segment/position anchors and fixed donors.
- `B20/per_anchor.csv`, `C20/per_anchor.csv`: every one-variable comparison.
- Per-model `by_position.csv`, `by_segment.csv`, `by_context.csv`, and `by_source.csv`.
- Root `by_position.csv`: paper-facing combined B20/C20 table.
"""
    readme_path = output_root / "README.md"
    if readme_path.exists():
        raise FileExistsError(f"refusing to overwrite {readme_path}")
    readme_path.write_text(readme, encoding="utf-8")
    print(f"[probe-finalize] wrote {combined_path} and {readme_path}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    manifest.add_argument("--protocol-seed", type=int, default=DEFAULT_PROTOCOL_SEED)
    manifest.set_defaults(function=create_manifest)
    model = commands.add_parser("run-model")
    model.add_argument("--model", choices=PROBE_MODELS, required=True)
    model.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    model.set_defaults(function=run_model)
    final = commands.add_parser("finalize")
    final.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    final.set_defaults(function=finalize)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        args.function(args)
    except Exception as exc:
        write_json_atomic(
            args.output_root.resolve() / f"failure_{args.command}.json",
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
