#!/usr/bin/env python3
"""Parse every available BEHAVIOR-1K task hdf5 (full-task dirs + sampled dir) and:

1. Print a task manifest: task_id -> activity_name, scene_model, rooms, #steps, etc.
2. Dump the detailed hdf5 internal structure (groups/datasets, shapes, dtypes) plus
   representative numeric info (action dim, reward sparsity, terminated step, ...).
3. Compare tasks against the current HRC combos and recommend replacements for the
   "hard for VLA" combos D (cook) / B (decor+fire) / E (wash).

Read-only. Run after fetch_task_samples.py finishes:
    /root/behavior-venv/bin/python scripts/parse_task_samples.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/root/autodl-tmp/datasets"))
FULL_ROOT = DATA_ROOT / "2025-challenge-rawdata"
SAMPLE_ROOT = DATA_ROOT / "task_samples"

# Current 5 HRC combos (from docs/data_collection_plan.md).
COMBOS = {
    "A": ("task-0026", "task-0000", "客厅·礼篮+收音机"),
    "B": ("task-0002", "task-0030", "客厅·节日+壁炉"),
    "C": ("task-0012", "task-0010", "厨房·咖啡+午餐盒"),
    "D": ("task-0046", "task-0045", "厨房·热狗+培根"),
    "E": ("task-0033", "task-0032", "工具房·清洗"),
}

# Keyword-based difficulty heuristic for VLA suitability.
HARD_KW = ("cook", "fire", "wash", "clean", "spray", "freeze", "can_", "canning",
           "microwave", "chop", "slic", "pizza", "cabbage", "onion", "meat")
EASY_KW = ("radio", "pick", "put", "tidy", "sort", "organiz", "assembl", "load",
           "carry", "bring", "store", "stor", "box", "shoe", "rack", "basket",
           "lunch", "coffee", "groc", "desk")


def find_one_hdf5(task_id: str) -> Path | None:
    """Prefer the full-task dir; fall back to the sampled dir."""
    for root in (FULL_ROOT, SAMPLE_ROOT):
        d = root / task_id
        if d.is_dir():
            files = sorted(d.glob("*.hdf5"))
            if files:
                return files[0]
    return None


def safe_json(attrs: Any, key: str) -> dict:
    if key not in attrs:
        return {}
    val = attrs[key]
    if isinstance(val, bytes):
        val = val.decode("utf-8", "ignore")
    try:
        return json.loads(val)
    except Exception:
        return {}


def parse_task(task_id: str, path: Path) -> dict:
    info: dict[str, Any] = {"task_id": task_id, "file": path.name,
                            "source": "full" if FULL_ROOT in path.parents else "sample"}
    with h5py.File(path, "r") as f:
        data = f.get("data")
        if data is None:
            info["error"] = "no /data group"
            return info
        cfg = safe_json(data.attrs, "config")
        info["activity"] = cfg.get("task", {}).get("activity_name", "?")
        info["scene"] = cfg.get("scene", {}).get("scene_model", "?")
        # rooms (best-effort from task instance config)
        info["n_episodes"] = int(data.attrs.get("n_episodes", 0) or 0)
        demos = [k for k in data.keys() if k.startswith("demo_")]
        info["n_demos"] = len(demos)
        if not demos:
            info["error"] = "no demo_*"
            return info
        g = data[demos[0]]
        info["demo_keys"] = list(g.keys())
        # action
        if "action" in g:
            a = g["action"]
            info["action_shape"] = tuple(a.shape)
            info["action_dim"] = int(a.shape[1]) if a.ndim == 2 else None
        # steps
        n_steps = None
        for k in ("action", "reward", "terminated"):
            if k in g:
                n_steps = int(g[k].shape[0]); break
        info["n_steps"] = n_steps
        # reward sparsity
        if "reward" in g:
            r = g["reward"][:]
            nz = np.nonzero(r)[0]
            info["reward_total"] = float(np.nansum(r))
            info["reward_max"] = float(np.nanmax(r)) if r.size else 0.0
            info["reward_nonzero_steps"] = nz.tolist()[:10]
            info["reward_n_nonzero"] = int(nz.size)
            info["reward_first_spike"] = int(nz[0]) if nz.size else -1
            info["reward_sparsity_pct"] = round(100.0 * nz.size / max(r.size, 1), 4)
        # terminated
        if "terminated" in g:
            t = g["terminated"][:]
            tn = np.nonzero(t)[0]
            info["terminated_first"] = int(tn[0]) if tn.size else -1
    return info


def difficulty(activity: str) -> str:
    a = activity.lower()
    if any(k in a for k in HARD_KW):
        return "hard"
    if any(k in a for k in EASY_KW):
        return "easy"
    return "medium"


def dump_structure(task_id: str, path: Path) -> None:
    print(f"\n----- STRUCTURE: {task_id}  ({path.name}) -----")
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            indent = "  " * name.count("/")
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}{name:50s} shape={str(obj.shape):16s} dtype={obj.dtype}")
            else:
                print(f"{indent}{name}/  (group)")
        f.visititems(visit)
        data = f.get("data")
        if data is not None:
            print("  data.attrs keys:", list(data.attrs.keys()))


def main() -> None:
    all_tasks = [f"task-{i:04d}" for i in range(50)]
    rows: list[dict] = []
    missing: list[str] = []
    for t in all_tasks:
        p = find_one_hdf5(t)
        if p is None:
            missing.append(t)
            continue
        try:
            rows.append(parse_task(t, p))
        except Exception as exc:
            rows.append({"task_id": t, "file": p.name,
                         "source": "full" if FULL_ROOT in p.parents else "sample",
                         "error": f"{type(exc).__name__}: {exc}",
                         "activity": "?", "scene": "?"})

    # ---------- 1. task manifest ----------
    print("=" * 110)
    print(" TASK MANIFEST  (50 tasks)")
    print("=" * 110)
    hdr = f"{'task':9s} {'src':5s} {'activity':42s} {'scene':26s} {'steps':>7s} {'adim':>4s} {'diff':6s}"
    print(hdr); print("-" * len(hdr))
    by_scene: dict[str, list[str]] = {}
    for r in sorted(rows, key=lambda x: x["task_id"]):
        act = r.get("activity", "?")
        sc = r.get("scene", "?")
        by_scene.setdefault(sc, []).append(r["task_id"])
        r["difficulty"] = difficulty(act)
        print(f"{r['task_id']:9s} {r['source']:5s} {act[:42]:42s} {sc[:26]:26s} "
              f"{str(r.get('n_steps','?')):>7s} {str(r.get('action_dim','?')):>4s} {r['difficulty']:6s}")
    if missing:
        print(f"\n[!] No hdf5 found for: {missing}")

    # ---------- scene grouping ----------
    print("\n" + "=" * 110)
    print(" SCENE -> TASKS")
    print("=" * 110)
    for sc, ts in sorted(by_scene.items()):
        print(f"  {sc:28s} ({len(ts):2d}): {', '.join(ts)}")

    # ---------- 2. detailed structure (one full + a couple of samples) ----------
    print("\n" + "=" * 110)
    print(" DETAILED HDF5 STRUCTURE (representative)")
    print("=" * 110)
    shown = 0
    for t in all_tasks:
        p = find_one_hdf5(t)
        if p is None:
            continue
        dump_structure(t, p)
        shown += 1
        if shown >= 3:
            break

    # ---------- per-task numeric detail ----------
    print("\n" + "=" * 110)
    print(" PER-TASK NUMERIC DETAIL (reward sparsity / terminated / demo keys)")
    print("=" * 110)
    for r in sorted(rows, key=lambda x: x["task_id"]):
        print(f"\n{r['task_id']} [{r.get('activity','?')}] scene={r.get('scene','?')}")
        print(f"   steps={r.get('n_steps')} action={r.get('action_shape')} "
              f"reward_max={r.get('reward_max')} n_nonzero={r.get('reward_n_nonzero')} "
              f"first_spike={r.get('reward_first_spike')} sparsity%={r.get('reward_sparsity_pct')} "
              f"terminated_first={r.get('terminated_first')}")
        print(f"   demo_keys={r.get('demo_keys')}")

    # ---------- 3. combo replacement analysis ----------
    print("\n" + "=" * 110)
    print(" COMBO REPLACEMENT ANALYSIS (alternatives to D / B / E for VLA)")
    print("=" * 110)
    info_by_id = {r["task_id"]: r for r in rows}

    def line(tid: str) -> str:
        r = info_by_id.get(tid, {})
        return (f"{tid} [{r.get('activity','?')}] scene={r.get('scene','?')} "
                f"steps={r.get('n_steps')} spike={r.get('reward_first_spike')} "
                f"sparsity%={r.get('reward_sparsity_pct')} diff={r.get('difficulty','?')}")

    print("\nCurrent combos under review:")
    for cid in ("D", "B", "E"):
        base, ext, theme = COMBOS[cid]
        print(f"  Combo {cid} ({theme}):")
        print(f"     base : {line(base)}")
        print(f"     ext  : {line(ext)}")

    # Candidate easy pick-place tasks grouped by scene, sorted by horizon then sparsity.
    print("\nEASY (pick-place-like) candidates per scene, sorted by (steps, sparsity):")
    for sc, ts in sorted(by_scene.items()):
        cands = [info_by_id[t] for t in ts
                 if info_by_id.get(t, {}).get("difficulty") == "easy"
                 and info_by_id[t].get("n_steps")]
        cands.sort(key=lambda r: (r.get("n_steps") or 1e9, r.get("reward_sparsity_pct") or 0))
        if not cands:
            continue
        print(f"\n  scene={sc}")
        for r in cands[:8]:
            print(f"     - {line(r['task_id'])}")

    # Auto-suggest same-scene easy pairs as drop-in replacements.
    print("\nSuggested same-scene EASY pairs (auto):")
    for sc, ts in sorted(by_scene.items()):
        easy = [t for t in ts if info_by_id.get(t, {}).get("difficulty") == "easy"
                and info_by_id[t].get("n_steps")]
        easy.sort(key=lambda t: info_by_id[t].get("n_steps") or 1e9)
        if len(easy) >= 2:
            a, b = easy[0], easy[1]
            print(f"  [{sc}]  {a} + {b}  "
                  f"(steps {info_by_id[a].get('n_steps')} / {info_by_id[b].get('n_steps')})")


if __name__ == "__main__":
    main()
