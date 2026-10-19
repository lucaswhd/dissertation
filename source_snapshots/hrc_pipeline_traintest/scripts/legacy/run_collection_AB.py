#!/usr/bin/env python3
# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
"""Batch collector for combo A: task-0026 (base) × task-0000 (radio).

Each output episode is the result of running ``hrc_multitask_demo.py`` with one
matched pair of input HDF5 files (same episode number on both sides). Each
episode is launched as a fresh Python subprocess to fully isolate Isaac Sim
state, with a per-run timeout, automatic retries, and a ``progress.json`` for
crash-safe resume.

Usage
-----
    python scripts/run_collection_AB.py [options]

Examples
--------
    # default: 85 episodes, common episode ids between task-0026 and task-0000
    python scripts/run_collection_AB.py

    # custom episode count, custom output root, dry run to inspect command list
    python scripts/run_collection_AB.py --num-episodes 5 --output-root /tmp/hrc_AB --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional


PROJECT_ROOT = Path("/root/hrc_pipeline")
DEFAULT_VENV_PY = Path("/root/behavior-venv/bin/python")
DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/datasets/2025-challenge-rawdata")
DEFAULT_OUTPUT_ROOT = Path("/root/autodl-tmp/outputs/hrc_collection_AB")
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "hrc_multitask_v01.yaml"
DEFAULT_BASE_TASK = "task-0026"   # round 1 = assembling_gift_baskets (base scene)
DEFAULT_EXT_TASK = "task-0000"    # round 0 = turning_on_radio (external)


def _list_episode_ids(dir_path: Path, task_id: str) -> List[int]:
    """Find every ``episode_TTTTNNNN.hdf5`` in ``dir_path`` and return the NNNN list."""
    digits = re.match(r"task-(\d{4})", task_id)
    if not digits:
        raise ValueError(f"Bad task id format: {task_id!r}")
    prefix = f"episode_{digits.group(1)}"
    out: List[int] = []
    for fn in os.listdir(dir_path):
        m = re.fullmatch(rf"{re.escape(prefix)}(\d{{4}})\.hdf5", fn)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def _episode_path(data_root: Path, task_id: str, ep_num: int) -> Path:
    digits = re.match(r"task-(\d{4})", task_id).group(1)  # type: ignore[union-attr]
    return data_root / task_id / f"episode_{digits}{ep_num:04d}.hdf5"


def _load_progress(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[run_collection] WARN: corrupt progress file ({exc}); starting fresh.",
              flush=True)
        return {}


def _save_progress(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(path)


def _validate_output(out_dir: Path) -> bool:
    """Quick sanity check: hrc_dataset.hdf5 exists with at least one demo group."""
    hdf5 = out_dir / "hrc_dataset.hdf5"
    if not hdf5.is_file():
        return False
    try:
        import h5py  # imported lazily — only needed for validation
    except ImportError:
        # Without h5py we can only check file existence.
        return hdf5.stat().st_size > 1024
    try:
        with h5py.File(str(hdf5), "r", locking=False) as f:
            data = f.get("data")
            if data is None:
                return False
            num_demos = int(data.attrs.get("num_demos", 0))
            return num_demos >= 1
    except Exception as exc:
        print(f"[run_collection] WARN: validation read failed for {hdf5}: {exc}",
              flush=True)
        return False


def _run_one(
    *,
    py: Path,
    config: Path,
    raw_hdf5: Path,
    external_hdf5: Path,
    output_dir: Path,
    timeout_s: int,
    log_path: Path,
    extra_args: List[str],
) -> tuple[bool, str]:
    """Spawn ``hrc_multitask_demo.py`` for a single (base_ep, ext_ep) pair.

    Returns (success, status_string).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        str(py),
        str(PROJECT_ROOT / "scripts" / "hrc_multitask_demo.py"),
        "--config", str(config),
        "--raw-hdf5", str(raw_hdf5),
        "--output-dir", str(output_dir),
        "--episode-id", "0",  # each rawdata file only contains demo_0
        "--external-hdf5", f"0:{external_hdf5}",
        *extra_args,
    ]

    print(f"[run_collection] CMD: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "8")

    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"# CMD: {' '.join(shlex.quote(c) for c in cmd)}\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
            start_new_session=True,
        )
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            print(f"[run_collection] TIMEOUT after {timeout_s}s — killing pgid={proc.pid}",
                  flush=True)
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(5)
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            elapsed = time.monotonic() - started
            return False, f"timeout_after_{int(elapsed)}s"

    elapsed = time.monotonic() - started

    if rc != 0:
        return False, f"nonzero_exit_{rc}_after_{int(elapsed)}s"

    if not _validate_output(output_dir):
        return False, f"validation_failed_after_{int(elapsed)}s"

    return True, f"ok_in_{int(elapsed)}s"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch collect HRC episodes for combo A "
                    "(task-0026 base × task-0000 radio).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="YAML config (used as a template; raw_hdf5 / external_hdf5 "
                             "/ output_dir are overridden per episode).")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="Root containing task-XXXX/episode_XXXXNNNN.hdf5.")
    parser.add_argument("--base-task", default=DEFAULT_BASE_TASK,
                        help="Base task id (round 1 source).")
    parser.add_argument("--external-task", default=DEFAULT_EXT_TASK,
                        help="External task id (round 0 source).")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="Where to put episode_NNNN/ subdirectories.")
    parser.add_argument("--num-episodes", type=int, default=85,
                        help="Number of episodes to collect.")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Skip the first N selected episodes (for re-segmenting work).")
    parser.add_argument("--episode-ids", type=str, default=None,
                        help="Comma-separated explicit external episode ids "
                             "(overrides --num-episodes / --start-index).")
    parser.add_argument(
        "--fixed-base-ep", type=int, default=None,
        help="If set, use this single base task episode number (e.g. 10 -> "
             "episode_00260010.hdf5) for ALL output episodes; only the external "
             "episode varies. Otherwise base/external are paired by id.",
    )
    parser.add_argument(
        "--external-sample-mode", choices=("sequential", "random"),
        default="sequential",
        help="How to pick external episode ids when --fixed-base-ep is set: "
             "'sequential' = sorted by id; 'random' = random sample (use --seed for "
             "reproducibility).",
    )
    parser.add_argument(
        "--seed", type=int, default=2026,
        help="Random seed for --external-sample-mode=random.",
    )
    parser.add_argument("--timeout-s", type=int, default=30 * 60,
                        help="Per-episode wall-clock timeout.")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Number of retries on failure (total attempts = 1 + retries).")
    parser.add_argument("--python", type=Path, default=DEFAULT_VENV_PY,
                        help="Python interpreter (the behavior-venv one).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan only; print the (base_ep, ext_ep) pairs and exit.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run episodes already marked done in progress.json.")
    parser.add_argument(
        "--extra-arg", action="append", default=None,
        help="Extra args forwarded to hrc_multitask_demo.py (repeatable).",
    )
    args = parser.parse_args()

    if not args.python.is_file():
        print(f"[run_collection] ERROR: python not found: {args.python}", file=sys.stderr)
        return 2
    if not args.config.is_file():
        print(f"[run_collection] ERROR: config not found: {args.config}", file=sys.stderr)
        return 2

    base_dir = args.data_root / args.base_task
    ext_dir = args.data_root / args.external_task
    if not base_dir.is_dir():
        print(f"[run_collection] ERROR: base task dir not found: {base_dir}", file=sys.stderr)
        return 2
    if not ext_dir.is_dir():
        print(f"[run_collection] ERROR: external task dir not found: {ext_dir}", file=sys.stderr)
        return 2

    base_ids = _list_episode_ids(base_dir, args.base_task)
    ext_ids = _list_episode_ids(ext_dir, args.external_task)
    common_ids = sorted(set(base_ids) & set(ext_ids))

    # ---- Build the list of (base_ep, ext_ep) pairs to run ------------------
    pairs: List[tuple[int, int]] = []
    if args.fixed_base_ep is not None:
        if args.fixed_base_ep not in base_ids:
            print(f"[run_collection] ERROR: --fixed-base-ep {args.fixed_base_ep} is not "
                  f"in {args.base_task} (have {len(base_ids)} episodes; "
                  f"sample: {base_ids[:5]}...)", file=sys.stderr)
            return 2

        if args.episode_ids:
            wanted = [int(x.strip()) for x in args.episode_ids.split(",") if x.strip()]
            missing = [i for i in wanted if i not in ext_ids]
            if missing:
                print(f"[run_collection] ERROR: external episode ids not present "
                      f"in {args.external_task}: {missing}", file=sys.stderr)
                return 2
            ext_pool = wanted
        elif args.external_sample_mode == "random":
            # Shuffle the full pool once with `seed`, then take the first N.
            # This guarantees that for the same seed, --num-episodes=5 is the
            # *prefix* of --num-episodes=85 — so a 5-episode dry run can be
            # extended to 85 without re-running the first 5 (`progress.json`
            # will mark them done and they get SKIPped).
            import random
            rng = random.Random(args.seed)
            shuffled = list(ext_ids)
            rng.shuffle(shuffled)
            n_pick = min(args.num_episodes, len(shuffled))
            ext_pool = shuffled[:n_pick]
            # Sort what we picked just so progress display / output dir order
            # is monotonic by ext_ep id (does not change *which* episodes we
            # pick, only their visit order).
            ext_pool = sorted(ext_pool)
        else:  # sequential
            ext_pool = ext_ids[: args.num_episodes]

        # Apply --start-index after the pool is built (only meaningful for non --episode-ids).
        if not args.episode_ids:
            if args.start_index < 0 or args.start_index >= len(ext_pool):
                print(f"[run_collection] ERROR: --start-index {args.start_index} out of "
                      f"range (pool size {len(ext_pool)})", file=sys.stderr)
                return 2
            ext_pool = ext_pool[args.start_index:]

        pairs = [(args.fixed_base_ep, e) for e in ext_pool]
    else:
        # Legacy mode: pair by identical episode id from the common set.
        if not common_ids:
            print(f"[run_collection] ERROR: no common episode ids between "
                  f"{args.base_task} and {args.external_task}", file=sys.stderr)
            return 2
        if args.episode_ids:
            wanted = [int(x.strip()) for x in args.episode_ids.split(",") if x.strip()]
            missing = [i for i in wanted if i not in common_ids]
            if missing:
                print(f"[run_collection] ERROR: episode ids not in common set: {missing}",
                      file=sys.stderr)
                return 2
            selected = wanted
        else:
            if args.start_index < 0 or args.start_index >= len(common_ids):
                print(f"[run_collection] ERROR: --start-index {args.start_index} out of "
                      f"range (have {len(common_ids)} matched episodes)", file=sys.stderr)
                return 2
            end = min(args.start_index + args.num_episodes, len(common_ids))
            selected = common_ids[args.start_index:end]
        pairs = [(e, e) for e in selected]

    print(f"[run_collection] base_task={args.base_task} (count={len(base_ids)})")
    print(f"[run_collection] external_task={args.external_task} (count={len(ext_ids)})")
    print(f"[run_collection] common pool size: {len(common_ids)}")
    if args.fixed_base_ep is not None:
        print(f"[run_collection] mode: FIXED-BASE  base_ep={args.fixed_base_ep:04d}  "
              f"sample_mode={args.external_sample_mode}  seed={args.seed}")
    else:
        print(f"[run_collection] mode: PAIRED-BY-ID")
    print(f"[run_collection] selected count: {len(pairs)}")
    preview = [f"(b={b:04d},e={e:04d})" for b, e in pairs[:8]]
    print(f"[run_collection] selected pairs: {preview}{' ...' if len(pairs) > 8 else ''}")
    print(f"[run_collection] output_root: {args.output_root}")
    print(f"[run_collection] timeout_s: {args.timeout_s}, max_retries: {args.max_retries}")

    if args.dry_run:
        print("[run_collection] DRY-RUN; exiting.")
        for base_ep, ext_ep in pairs:
            base = _episode_path(args.data_root, args.base_task, base_ep)
            ext = _episode_path(args.data_root, args.external_task, ext_ep)
            print(f"  out=episode_{ext_ep:04d}  base={base}  ext={ext}")
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.json"
    progress = _load_progress(progress_path)
    failures_path = args.output_root / "failures.json"
    log_dir = args.output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    summary = {"done": 0, "skipped": 0, "failed": 0}
    overall_t0 = time.monotonic()

    for i, (base_ep, ext_ep) in enumerate(pairs):
        # Output dirs/keys are indexed by external ep so that re-running with the
        # same seed/list resumes idempotently.
        key = (f"{args.base_task}-ep{base_ep:04d}__"
               f"{args.external_task}-ep{ext_ep:04d}")
        out_dir = args.output_root / f"episode_{ext_ep:04d}"
        log_path = log_dir / f"episode_{ext_ep:04d}.log"

        prior = progress.get(key, {})
        if not args.force and prior.get("status") == "done" and _validate_output(out_dir):
            summary["skipped"] += 1
            print(f"[run_collection] [{i+1}/{len(pairs)}] base={base_ep:04d} "
                  f"ext={ext_ep:04d} SKIP (already done)", flush=True)
            continue

        base_path = _episode_path(args.data_root, args.base_task, base_ep)
        ext_path = _episode_path(args.data_root, args.external_task, ext_ep)
        if not base_path.is_file() or not ext_path.is_file():
            progress[key] = {"status": "missing_input", "base": str(base_path),
                             "ext": str(ext_path)}
            _save_progress(progress_path, progress)
            summary["failed"] += 1
            print(f"[run_collection] [{i+1}/{len(pairs)}] base={base_ep:04d} "
                  f"ext={ext_ep:04d} MISSING INPUT", flush=True)
            continue

        succeeded = False
        last_status = ""
        for attempt in range(1, args.max_retries + 2):
            print(f"\n[run_collection] [{i+1}/{len(pairs)}] base={base_ep:04d} "
                  f"ext={ext_ep:04d} attempt {attempt}/{args.max_retries + 1}  "
                  f"(elapsed_total={timedelta(seconds=int(time.monotonic() - overall_t0))})",
                  flush=True)
            ok, status = _run_one(
                py=args.python,
                config=args.config,
                raw_hdf5=base_path,
                external_hdf5=ext_path,
                output_dir=out_dir,
                timeout_s=args.timeout_s,
                log_path=log_path if attempt == 1 else log_dir / f"episode_{ext_ep:04d}.try{attempt}.log",
                extra_args=list(args.extra_arg or []),
            )
            last_status = status
            print(f"[run_collection]   -> attempt {attempt}: {status}", flush=True)
            if ok:
                succeeded = True
                break

        progress[key] = {
            "status": "done" if succeeded else "failed",
            "last_status": last_status,
            "attempts": attempt,
            "base_ep": base_ep,
            "ext_ep": ext_ep,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "output_dir": str(out_dir),
        }
        _save_progress(progress_path, progress)

        if succeeded:
            summary["done"] += 1
        else:
            summary["failed"] += 1
            failures = _load_progress(failures_path)
            failures[key] = progress[key]
            _save_progress(failures_path, failures)

    elapsed = timedelta(seconds=int(time.monotonic() - overall_t0))
    print("\n[run_collection] === SUMMARY ===")
    print(f"  total selected: {len(pairs)}")
    print(f"  done:           {summary['done']}")
    print(f"  skipped:        {summary['skipped']}")
    print(f"  failed:         {summary['failed']}")
    print(f"  total elapsed:  {elapsed}")
    print(f"  progress file:  {progress_path}")
    print(f"  output root:    {args.output_root}")

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
