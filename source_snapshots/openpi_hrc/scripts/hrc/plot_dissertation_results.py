#!/usr/bin/env python
"""Render dissertation figures from frozen result CSVs only."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.hrc.run_dissertation_final_audit import REPO_ROOT

AUDIT_CSV = REPO_ROOT / "outputs/dissertation_final_audit/comparison.csv"
PROBE_CSV = REPO_ROOT / "outputs/dissertation_conditioning_probe/by_position.csv"
VISIBILITY_CSV = REPO_ROOT / "outputs/dissertation_target_visibility/summary.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/dissertation_figures"
MODELS = ("A20", "B20", "C20")
COLORS = ("#374151", "#2563EB", "#15803D")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_columns(rows: list[dict[str, str]], columns: set[str], label: str) -> None:
    if not rows:
        raise ValueError(f"{label} CSV is empty")
    missing = sorted(columns - rows[0].keys())
    if missing:
        raise ValueError(f"{label} CSV missing columns: {missing}")


def _save(figure: plt.Figure, output_root: Path, stem: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_root / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_root / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_audit(rows: list[dict[str, str]], output_root: Path) -> None:
    require_columns(
        rows,
        {"model", "context", "action_group", "metric", "mean", "std"},
        "audit",
    )
    contexts = ("pickup", "radio", "follow")
    groups = ("arms", "base")
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharex=True)
    x = np.arange(len(contexts), dtype=np.float64)
    width = 0.24
    for axis, group in zip(axes, groups, strict=True):
        for model_index, (model, color) in enumerate(zip(MODELS, COLORS, strict=True)):
            selected = {
                row["context"]: row
                for row in rows
                if row["model"] == model
                and row["action_group"] == group
                and row["metric"] == "skill"
                and row["context"] in contexts
            }
            if set(selected) != set(contexts):
                raise ValueError(f"audit CSV lacks {model}/{group} skill rows")
            means = [float(selected[context]["mean"]) for context in contexts]
            stds = [float(selected[context]["std"]) for context in contexts]
            axis.bar(
                x + (model_index - 1) * width,
                means,
                width,
                yerr=stds,
                capsize=2,
                label=model,
                color=color,
            )
        axis.axhline(0.0, color="#111827", linewidth=0.8)
        axis.set_title(f"{group.capitalize()} skill")
        axis.set_xticks(x, contexts)
        axis.set_ylabel("Skill score")
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.8)
    axes[0].legend(frameon=False, ncols=3, loc="lower left")
    figure.suptitle("Unified validation teacher-forced audit")
    figure.tight_layout()
    _save(figure, output_root, "teacher_forced_skill")


def plot_conditioning(rows: list[dict[str, str]], output_root: Path) -> None:
    require_columns(
        rows,
        {
            "checkpoint",
            "split",
            "context",
            "segment",
            "position",
            "metric",
            "mean",
            "ci95_low",
            "ci95_high",
        },
        "conditioning",
    )
    positions = ("early", "middle", "late")
    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    for model, color in zip(("B20", "C20"), COLORS[1:], strict=True):
        selected = {
            row["position"]: row
            for row in rows
            if row["checkpoint"] == model
            and row["split"] == "val"
            and row["context"] == "pickup"
            and row["segment"] == "Pick up the soda can"
            and row["metric"] == "arms_image_over_noise"
        }
        if set(selected) != set(positions):
            raise ValueError(f"conditioning CSV lacks {model} validation pickup positions")
        means = np.asarray([float(selected[position]["mean"]) for position in positions])
        lows = np.asarray([float(selected[position]["ci95_low"]) for position in positions])
        highs = np.asarray([float(selected[position]["ci95_high"]) for position in positions])
        axis.errorbar(
            positions,
            means,
            yerr=np.vstack((means - lows, highs - means)),
            marker="o",
            capsize=3,
            linewidth=1.5,
            color=color,
            label=model,
        )
    axis.axhline(1.0, color="#111827", linestyle="--", linewidth=0.9, label="Noise reference")
    axis.set_title("Validation pickup image conditioning by segment position")
    axis.set_xlabel("Segment position")
    axis.set_ylabel("Arms D_image / D_noise")
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.8)
    axis.legend(frameon=False)
    figure.tight_layout()
    _save(figure, output_root, "conditioning_by_position")


def plot_visibility(rows: list[dict[str, str]], output_root: Path) -> None:
    require_columns(
        rows,
        {
            "split",
            "context",
            "segment",
            "position",
            "population",
            "target_center_in_frustum_ratio",
        },
        "visibility",
    )
    segments = (
        "Walk to the trash can and pick it up",
        "Pick up the soda can",
        "Drop the can into the trash can you are holding",
        "Put the trash can down",
    )
    labels = ("Acquire bin", "Pick can", "Deposit can", "Place bin")
    positions = ("early", "middle", "late")
    selected = {
        (row["segment"], row["position"]): row
        for row in rows
        if row["split"] == "val"
        and row["context"] == "pickup"
        and row["population"] == "valid_anchors"
    }
    if any((segment, position) not in selected for segment in segments for position in positions):
        raise ValueError("visibility CSV lacks validation pickup segment-position rows")
    figure, axis = plt.subplots(figsize=(8.0, 3.8))
    x = np.arange(len(segments), dtype=np.float64)
    width = 0.24
    position_colors = ("#2563EB", "#15803D", "#B45309")
    for index, (position, color) in enumerate(zip(positions, position_colors, strict=True)):
        values = [
            float(selected[(segment, position)]["target_center_in_frustum_ratio"])
            for segment in segments
        ]
        axis.bar(x + (index - 1) * width, values, width, color=color, label=position.capitalize())
    axis.set_ylim(0.0, 1.05)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Target center in frustum ratio")
    axis.set_title("Validation pickup target-center frustum coverage")
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.8)
    axis.legend(frameon=False, ncols=3)
    figure.tight_layout()
    _save(figure, output_root, "target_visibility")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    plot_audit(read_csv(AUDIT_CSV), args.output_root)
    plot_conditioning(read_csv(PROBE_CSV), args.output_root)
    plot_visibility(read_csv(VISIBILITY_CSV), args.output_root)
    print(f"[plots] wrote PNG and PDF figures to {args.output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
