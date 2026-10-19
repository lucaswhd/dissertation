#!/usr/bin/env python3
"""Generate sealed-test-safe Gate 2/3 server and rollout commands.

This command never starts a model or simulator.  It groups runs by checkpoint
because one 4090 can host only one π0.5 server at a time, and writes every
source/seed/output/checkpoint choice to a reproducible JSON plan.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.evaluation.hrc_rollout_core import build_causal_gate_cases


DEFAULT_BATCH_MANIFEST = _REPO_ROOT / "outputs/hrc_batch_final/batch_manifest.json"
DEFAULT_PI05_MANIFEST = Path("/root/autodl-tmp/lerobot_datasets/hrc_pi05_v0/manifest.json")
DEFAULT_OPENPI = Path("/root/autodl-tmp/openpi_hrc")
DEFAULT_CONFIG = _REPO_ROOT / "configs/hrc_multitask_0001.yaml"
DEFAULT_OUTPUT = _REPO_ROOT / "outputs/pi05_closed_loop"


def _quote_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(value) for value in parts)


def _entries(batch_manifest: Path) -> dict[int, dict]:
    with batch_manifest.open(encoding="utf-8") as stream:
        return {int(row["idx"]): row for row in json.load(stream)["episodes"]}


def build_plan(args: argparse.Namespace) -> dict:
    cases = build_causal_gate_cases(
        args.batch_manifest,
        args.pi05_manifest,
        split="val",
    )
    source_ids = sorted({case.source_id for case in cases})
    if len(source_ids) != 5:
        raise ValueError(f"Gate 3 requires exactly five val sources, got {source_ids}")
    entries = _entries(args.batch_manifest)
    models = {
        "pi05_base_hrc_init": Path(args.base_checkpoint),
        "e4_5k": Path(args.e4_checkpoint),
    }
    selected_models = models if args.gate == "gate3" else {"e4_5k": models["e4_5k"]}
    selected_sources = source_ids if args.gate == "gate3" else source_ids[:1]
    selected_seeds = list(args.seed) if args.gate == "gate3" else [args.seed[0]]
    execute_steps = args.execute_steps if args.gate == "gate3" else args.smoke_execute_steps

    groups = []
    for model_label, checkpoint in selected_models.items():
        server_command = _quote_command(
            [
                str(args.openpi_root / ".venv/bin/python"),
                "scripts/hrc/serve_hrc_pi05_rollout.py",
                "--config", "pi05_hrc_e4_main",
                "--checkpoint", str(checkpoint),
                "--host", args.policy_host,
                "--port", str(args.policy_port),
                "--execute-horizon", "8",
            ]
        )
        runs = []
        for source_id in selected_sources:
            entry = entries[source_id]
            for seed in selected_seeds:
                run_id = f"{model_label}_source{source_id:04d}_seed{seed}"
                output_dir = args.output_root / args.gate / run_id
                rollout_command = "HRC_RECORD_THIRD=1 " + _quote_command(
                    [
                        "/root/autodl-tmp/behavior-venv/bin/python",
                        "scripts/collection/hrc_multitask_demo.py",
                        "--config", str(args.config),
                        "--raw-hdf5", entry["trash_hdf5"],
                        "--external-hdf5", f"1:{entry['radio_hdf5']}",
                        "--output-dir", str(output_dir),
                        "--policy-rollout",
                        "--policy-host", args.policy_host,
                        "--policy-port", str(args.policy_port),
                        "--rollout-seed", str(seed),
                        "--policy-execute-steps", str(execute_steps),
                    ]
                )
                runs.append(
                    {
                        "run_id": run_id,
                        "model": model_label,
                        "checkpoint": str(checkpoint),
                        "source_id": source_id,
                        "seed": seed,
                        "output_dir": str(output_dir),
                        "expected_hdf5": str(output_dir / "hrc_dataset.hdf5"),
                        "expected_head_video": str(output_dir / "videos/head_rgb.mp4"),
                        "expected_third_video": str(output_dir / "videos/third_person_rgb.mp4"),
                        "command": rollout_command,
                    }
                )
        groups.append(
            {
                "model": model_label,
                "checkpoint": str(checkpoint),
                "server_workdir": str(args.openpi_root),
                "server_command": server_command,
                "runs": runs,
            }
        )
    return {
        "gate": args.gate,
        "split": "val",
        "sealed_test_access": False,
        "source_ids": selected_sources,
        "policy_seeds": selected_seeds,
        "execute_horizon": 8,
        "action_horizon": 32,
        "execute_steps_per_round": execute_steps,
        "rollout_workdir": str(_REPO_ROOT),
        "model_groups": groups,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=("gate2", "gate3"), default="gate3")
    parser.add_argument("--batch-manifest", type=Path, default=DEFAULT_BATCH_MANIFEST)
    parser.add_argument("--pi05-manifest", type=Path, default=DEFAULT_PI05_MANIFEST)
    parser.add_argument("--openpi-root", type=Path, default=DEFAULT_OPENPI)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=DEFAULT_OPENPI / "checkpoints/pi05_hrc_e4_main/hrc_base_init_seed42/0",
    )
    parser.add_argument(
        "--e4-checkpoint",
        type=Path,
        default=DEFAULT_OPENPI / "checkpoints/pi05_hrc_e4_main/hrc_e4_seed42/5000",
    )
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--execute-steps", type=int, default=600)
    parser.add_argument("--smoke-execute-steps", type=int, default=32)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=18765)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.seed = args.seed or [0, 1]
    if args.gate == "gate3" and len(args.seed) != 2:
        raise ValueError("Gate 3 requires exactly two fixed policy seeds")
    return args


def main() -> int:
    args = parse_args()
    plan = build_plan(args)
    rendered = json.dumps(plan, indent=2, ensure_ascii=False)
    print(rendered)
    if not args.dry_run:
        destination = args.output_root / args.gate / "run_plan.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
        print(f"WROTE {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
