#!/usr/bin/env python3
"""Gate 0: execute raw expert actions causally from one loaded initial state.

The sealed test split is intentionally unavailable from this command.  A real
case restores scene metadata, loads ``state[0]`` exactly once, then advances
the simulator only through ``env.step(action[t])``.  Use ``--dry-run`` first;
full physics runs are intentionally left to an explicit user invocation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.evaluation.hrc_rollout_core import CausalGateCase
from scripts.evaluation.hrc_rollout_core import build_causal_gate_cases
from scripts.evaluation.hrc_rollout_core import execute_causal_action_sequence


DEFAULT_BATCH_MANIFEST = _REPO_ROOT / "outputs/hrc_batch_final/batch_manifest.json"
DEFAULT_PI05_MANIFEST = Path("/root/autodl-tmp/lerobot_datasets/hrc_pi05_v0/manifest.json")
DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "outputs/pi05_closed_loop/gate0_causal_expert"
TASK_PREDICATES = {
    "pickup": {
        "type": "inside",
        "objects": ["can_of_soda_113", "can_of_soda_114", "can_of_soda_115"],
        "container": "trash_can_116",
    },
    "radio": {"type": "toggled_on", "object": "radio_89"},
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _select_cases(args: argparse.Namespace) -> list[CausalGateCase]:
    cases = build_causal_gate_cases(
        args.batch_manifest,
        args.pi05_manifest,
        split=args.split,
    )
    if args.source_id:
        selected = set(args.source_id)
        cases = [case for case in cases if case.source_id in selected]
        missing = sorted(selected - {case.source_id for case in cases})
        if missing:
            raise ValueError(f"requested source IDs are not in {args.split}: {missing}")
    if args.task != "all":
        cases = [case for case in cases if case.task == args.task]
    return cases


def _inspect_case(case: CausalGateCase) -> dict[str, Any]:
    import h5py

    with h5py.File(case.raw_hdf5, "r") as source:
        group = source["data/demo_0"]
        actions = group["action"]
        states = group["state"]
        transitions = json.loads(group.attrs.get("transitions", "{}"))
        if actions.ndim != 2 or actions.shape[1] != 23:
            raise ValueError(f"{case.raw_hdf5}: expected action [T,23], got {actions.shape}")
        if len(states) < len(actions) + 1:
            raise ValueError(
                f"{case.raw_hdf5}: causal stream needs T+1 states; "
                f"actions={len(actions)} states={len(states)}"
            )
        if transitions:
            raise ValueError(
                f"{case.raw_hdf5}: dynamic transitions are not supported by Gate 0; "
                "handle them explicitly before interpreting causality"
            )
        action_matrix = np.asarray(actions, dtype=np.float32)
        if not np.isfinite(action_matrix).all():
            raise ValueError(f"{case.raw_hdf5}: action contains non-finite values")
        config = json.loads(source["data"].attrs["config"])
        frequency = float(config["env"]["action_frequency"])
    return {
        "source_id": case.source_id,
        "task": case.task,
        "raw_hdf5": str(case.raw_hdf5),
        "num_actions": int(action_matrix.shape[0]),
        "action_dim": int(action_matrix.shape[1]),
        "action_frequency_hz": frequency,
        "left_gripper_switches": int(np.count_nonzero(np.diff(action_matrix[:, 14] >= 0))),
        "right_gripper_switches": int(np.count_nonzero(np.diff(action_matrix[:, 22] >= 0))),
    }


def _evaluate_predicate(env: Any, task: str) -> tuple[bool | None, str]:
    predicate = TASK_PREDICATES[task]
    if predicate["type"] == "toggled_on":
        from omnigibson.object_states import ToggledOn

        name = predicate["object"]
        obj = env.scene.object_registry("name", name)
        if obj is None or ToggledOn not in obj.states:
            return None, f"cannot evaluate ToggledOn({name})"
        value = bool(obj.states[ToggledOn].get_value())
        return value, f"ToggledOn({name})={value}"

    from omnigibson.object_states import Inside

    container_name = predicate["container"]
    container = env.scene.object_registry("name", container_name)
    if container is None:
        return None, f"container {container_name!r} not found"
    hits = []
    missing = []
    for name in predicate["objects"]:
        obj = env.scene.object_registry("name", name)
        if obj is None:
            missing.append(name)
            continue
        if Inside in obj.states and bool(obj.states[Inside].get_value(container)):
            hits.append(name)
    if missing and not hits:
        return None, f"missing objects={missing}"
    return bool(hits), f"Inside(any, {container_name}) hits={hits}"


def _run_case(case: CausalGateCase, args: argparse.Namespace, run_root: Path) -> dict[str, Any]:
    import h5py
    import imageio.v2 as imageio
    import omnigibson as og
    import torch as th
    from omnigibson.utils.python_utils import h5py_group_to_torch

    collection_dir = _REPO_ROOT / "scripts/collection"
    if str(collection_dir) not in sys.path:
        sys.path.insert(0, str(collection_dir))
    from hrc_utils import extract_first_rgb
    from nav_online_driver import build_contacts_on_playback_wrapper

    case_dir = run_root / f"source_{case.source_id:04d}_{case.task}"
    case_dir.mkdir(parents=True, exist_ok=True)
    wrapper_output = case_dir / "wrapper_output.hdf5"
    video_path = case_dir / "head.mp4"
    env = None
    writer = None
    start_time = time.monotonic()
    try:
        env = build_contacts_on_playback_wrapper(
            raw_hdf5=str(case.raw_hdf5),
            output_hdf5_path=str(wrapper_output),
            resolution=args.resolution,
        )
        with h5py.File(case.raw_hdf5, "r") as source:
            raw_group = source["data/demo_0"]
            transitions = json.loads(raw_group.attrs.get("transitions", "{}"))
            if transitions:
                raise ValueError("Gate 0 refuses per-frame scene transitions")
            group = h5py_group_to_torch(raw_group)
            actions = np.asarray(group["action"], dtype=np.float32)
            states = group["state"]
            state_sizes = group["state_size"]
            init_metadata = group["init_metadata"]

        def load_initial_state() -> None:
            env.scene.restore(env.scene_file, update_initial_file=True)
            with og.sim.stopped():
                for index, obj in enumerate(env.scene.objects):
                    for attribute, values in init_metadata.items():
                        value = values[index]
                        setattr(obj, attribute, value.item() if value.ndim == 0 else value)
            env.reset()
            size = int(state_sizes[0])
            og.sim.load_state(states[0, :size], serialized=True)

        writer = imageio.get_writer(video_path, fps=args.video_fps)
        last_info: dict[str, Any] = {}

        def step_action(index: int, action: np.ndarray) -> None:
            nonlocal last_info
            obs, reward, terminated, truncated, info = env.env.step(
                action=th.as_tensor(action, dtype=th.float32),
                n_render_iterations=env.n_render_iterations,
            )
            last_info = {
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
            frame, _ = extract_first_rgb(obs)
            if frame is not None:
                writer.append_data(frame)
            if args.log_every and (index + 1) % args.log_every == 0:
                print(f"[gate0] source={case.source_id} task={case.task} step={index + 1}", flush=True)

        executed = execute_causal_action_sequence(
            actions,
            load_initial_state=load_initial_state,
            step_action=step_action,
            max_steps=args.max_steps,
        )
        success, detail = _evaluate_predicate(env, case.task)
        partial = args.max_steps is not None and executed < len(actions)
        result = {
            **_inspect_case(case),
            "executed_steps": executed,
            "initial_state_loads": 1,
            "step_only_after_initialization": True,
            "partial_smoke": partial,
            "success": success,
            "predicate_detail": detail,
            "last_step_info": last_info,
            "video": str(video_path),
            "elapsed_s": round(time.monotonic() - start_time, 3),
        }
    except Exception as exc:
        result = {
            **_inspect_case(case),
            "success": None,
            "error": f"{type(exc).__name__}: {exc}",
            "partial_smoke": args.max_steps is not None,
            "elapsed_s": round(time.monotonic() - start_time, 3),
        }
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    _write_json(case_dir / "result.json", result)
    return result


def _summarize(results: list[dict[str, Any]], expected_cases: int) -> dict[str, Any]:
    per_task = {}
    for task in ("pickup", "radio"):
        rows = [row for row in results if row["task"] == task]
        successes = sum(row.get("success") is True and not row.get("partial_smoke") for row in rows)
        per_task[task] = {"successes": successes, "attempts": len(rows)}
    full = len(results) == expected_cases and all(not row.get("partial_smoke") for row in results)
    passed = full and all(per_task[task] == {"successes": 5, "attempts": 5} for task in per_task)
    return {
        "gate": "Gate 0 expert-action causality",
        "full_validation": full,
        "passed": passed,
        "required": "pickup 5/5 and radio 5/5",
        "per_task": per_task,
        "results": results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-manifest", type=Path, default=DEFAULT_BATCH_MANIFEST)
    parser.add_argument("--pi05-manifest", type=Path, default=DEFAULT_PI05_MANIFEST)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--source-id", type=int, action="append")
    parser.add_argument("--task", choices=("all", "pickup", "radio"), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--log-every", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = _select_cases(args)
    inspected = [_inspect_case(case) for case in cases]
    if args.dry_run:
        print(json.dumps({"split": args.split, "cases": inspected}, indent=2, ensure_ascii=False))
        return 0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = args.output_root / timestamp
    run_root.mkdir(parents=True, exist_ok=False)
    config_payload = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    _write_json(run_root / "config.json", config_payload)
    results = []
    for case in cases:
        print(f"[gate0] starting source={case.source_id} task={case.task}", flush=True)
        results.append(_run_case(case, args, run_root))
        _write_json(run_root / "gate_summary.json", _summarize(results, len(cases)))
    summary = _summarize(results, len(cases))
    _write_json(run_root / "gate_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
