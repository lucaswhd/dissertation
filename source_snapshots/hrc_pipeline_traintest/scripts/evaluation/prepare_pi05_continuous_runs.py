#!/usr/bin/env python3
"""Generate the E5 continuous closed-loop run plan (server + rollout commands).

Never starts a model or simulator. Emits a reproducible JSON plan for the 8
full-chain runs (source_ids [15,18] x seeds [0,1] x {continuous_init,
e5_continuous_5k}), grouped by checkpoint because one 4090 hosts one π0.5 server
at a time. Carries a ``comparison`` field so ``summarize_pi05_closed_loop.py``
compares e5_continuous_5k against continuous_init (not the hardcoded e4 pair).
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

DEFAULT_BATCH_MANIFEST = _REPO_ROOT / "outputs/hrc_batch_final/batch_manifest.json"
DEFAULT_CONTINUOUS_MANIFEST = Path(
    "/root/autodl-tmp/lerobot_datasets/hrc_pi05_continuous_clean_v0/manifest.json"
)
DEFAULT_OPENPI = Path("/root/autodl-tmp/openpi_hrc")
DEFAULT_CONFIG = _REPO_ROOT / "configs/hrc_multitask_0001.yaml"
DEFAULT_OUTPUT = _REPO_ROOT / "outputs/pi05_continuous_clean/closed_loop"
DEFAULT_OPENPI_CONFIG_NAME = "pi05_hrc_e5_continuous_main"
DEFAULT_VAL_SOURCE_IDS = (15, 18)
POLICY_SEEDS = (0, 1)


def _quote_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(value) for value in parts)


def _entries(batch_manifest: Path) -> dict[int, dict]:
    with batch_manifest.open(encoding="utf-8") as stream:
        return {int(row["idx"]): row for row in json.load(stream)["episodes"]}


def _resolve_sources(continuous_manifest: Path, split: str, expected: tuple[int, ...]) -> list[int]:
    """Sources for one split, guarded against silently evaluating a different set.

    The guard is what keeps a sealed split from leaking into an eval by accident:
    the caller must state which sources it expects, and a manifest that disagrees
    fails fast instead of quietly changing what the numbers mean.
    """
    manifest = json.loads(continuous_manifest.read_text(encoding="utf-8"))
    sources = sorted(int(value) for value in manifest["source_splits"][split])
    if sources != sorted(expected):
        raise ValueError(f"continuous manifest {split} {sources} != expected {sorted(expected)}")
    return sources


def build_plan(args: argparse.Namespace) -> dict:
    split = getattr(args, "split", "val")
    expected = tuple(getattr(args, "expect_sources", None) or DEFAULT_VAL_SOURCE_IDS)
    openpi_config = getattr(args, "openpi_config", None) or DEFAULT_OPENPI_CONFIG_NAME
    baseline_label = getattr(args, "baseline_label", None) or "continuous_init"
    candidate_label = getattr(args, "candidate_label", None) or "e5_continuous_5k"

    source_ids = _resolve_sources(args.continuous_manifest, split, expected)
    entries = _entries(args.batch_manifest)
    models = {}
    if args.init_checkpoint:
        models[baseline_label] = Path(args.init_checkpoint)
    models[candidate_label] = Path(args.candidate_checkpoint)
    if args.smoke:
        # Prove the loop on one source/seed with the trained model only.
        models = {candidate_label: models[candidate_label]}
        source_ids = source_ids[:1]
        seeds = [POLICY_SEEDS[0]]
        execute_steps = args.smoke_execute_steps
    else:
        seeds = list(POLICY_SEEDS)
        execute_steps = args.execute_steps

    groups = []
    for model_label, checkpoint in models.items():
        server_command = _quote_command(
            [
                str(args.openpi_root / ".venv/bin/python"),
                "scripts/hrc/serve_hrc_pi05_rollout.py",
                "--config", openpi_config,
                "--checkpoint", str(checkpoint),
                "--host", args.policy_host,
                "--port", str(args.policy_port),
                "--execute-horizon", "8",
            ]
        )
        runs = []
        for source_id in source_ids:
            entry = entries[source_id]
            for seed in seeds:
                run_id = f"{model_label}_source{source_id:04d}_seed{seed}"
                output_dir = args.output_root / run_id
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
                        "--policy-execute-max-seconds", str(args.execute_max_seconds),
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
                        "expected_metrics": str(output_dir / "policy_rollout_metrics.json"),
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
        "gate": f"{candidate_label}_closed_loop",
        "split": split,
        "sealed_test_access": split == "test",
        "source_ids": source_ids,
        "policy_seeds": seeds,
        "execute_horizon": 8,
        "action_horizon": 32,
        "execute_steps_per_round": execute_steps,
        "execute_max_seconds": args.execute_max_seconds,
        "config_name": openpi_config,
        "rollout_workdir": str(_REPO_ROOT),
        "comparison": (
            {"baseline": baseline_label, "candidate": candidate_label}
            if args.init_checkpoint
            else {"baseline": None, "candidate": candidate_label}
        ),
        "model_groups": groups,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-manifest", type=Path, default=DEFAULT_BATCH_MANIFEST)
    parser.add_argument("--continuous-manifest", type=Path, default=DEFAULT_CONTINUOUS_MANIFEST)
    parser.add_argument("--openpi-root", type=Path, default=DEFAULT_OPENPI)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=DEFAULT_OPENPI
        / "checkpoints/pi05_hrc_e5_continuous_main/hrc_e5_continuous_init_seed42/0",
        help="optimizer-step-0 init, materialized from the same config/norm stats; pass an "
             "empty string to plan the trained model alone (no paired baseline)",
    )
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=DEFAULT_OPENPI
        / "checkpoints/pi05_hrc_e5_continuous_main/hrc_e5_continuous_seed42/5000",
    )
    parser.add_argument(
        "--execute-steps", type=str, default="3000,2400",
        help="ExecuteBehaviorTask budget per round (pickup,radio). The expert needs "
             "~2505 / ~1956 steps, so the old flat 600 capped pickup and radio well "
             "below completion no matter how good the policy was; these defaults keep "
             "~20%% headroom over the expert.",
    )
    parser.add_argument(
        "--execute-max-seconds", type=float, default=1800.0,
        help="Wall-clock cap per ExecuteBehaviorTask phase, so a stalled policy cannot "
             "hold a run open indefinitely once the step budget is this large.",
    )
    parser.add_argument(
        "--split", default="val", choices=("val", "test"),
        help="which frozen split to evaluate; 'test' marks the plan sealed_test_access",
    )
    parser.add_argument(
        "--expect-sources", type=int, nargs="+", default=None,
        help="sources this plan expects in --split; a manifest that disagrees fails fast "
             "so an eval can never silently run on a different population",
    )
    parser.add_argument(
        "--openpi-config", default=None,
        help=f"openpi train config the policy server loads (default {DEFAULT_OPENPI_CONFIG_NAME})",
    )
    parser.add_argument("--baseline-label", default=None, help="run-id prefix for the init model")
    parser.add_argument("--candidate-label", default=None, help="run-id prefix for the trained model")
    parser.add_argument("--smoke-execute-steps", type=str, default="32")
    parser.add_argument("--smoke", action="store_true", help="one source/seed, trained model only")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=18765)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_plan(args)
    rendered = json.dumps(plan, indent=2, ensure_ascii=False)
    print(rendered)
    if not args.dry_run:
        destination = args.output_root / "run_plan.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
        print(f"WROTE {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
