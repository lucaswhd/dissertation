# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
# See LICENSE in the project root for full terms.

"""Sim-free pre-screen + pairing for the multitask "trash + radio" scale-up.

Each output episode = one task-0001 raw (Round 0 = pick up trash) replayed as the
base, plus one task-0000 raw (Round 1 = turn on radio) replayed externally. This
tool reads ONLY the reward / terminated arrays (no Isaac boot) to reject
structurally-broken source raws BEFORE we spend ~40 min/episode on them, then
writes a deterministic pairing manifest the batch driver consumes.

"Clean" criteria (measured distribution over the 200+200 raws, 2026-06-30):
  - TRASH (task-0001): exactly N reward spikes (default {3} = the 3 soda cans;
    174/200 match) AND terminated at/after the last spike AND length in band.
  - RADIO (task-0000): >=1 reward spike (the toggle; 200/200 match) AND length
    in band.
Shorter (but complete) trash raws are preferred — episode wall-time scales with
total replay steps, and the median trash raw (5352) is shorter than the
hand-validated ep0010 (7667).

We also verify each selected trash raw's scene_file still contains the predicate's
instance names (can_of_soda_113/114/115 + trash_can_116). They were identical
across every sampled raw (same scene instance), so this is a cheap guard, not an
expected failure.

Usage:
  python scripts/collection/prescreen_sources.py --n 50 \
      --out outputs/hrc_batch_0001/pairing_manifest.json
"""
import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

# Round-0 success predicate instance names (configs/hrc_multitask_0001.yaml).
PREDICATE_CANS = ["can_of_soda_113", "can_of_soda_114", "can_of_soda_115"]
PREDICATE_TRASH = "trash_can_116"
_NAME_RE = re.compile(r"(can_of_soda|trash_can)", re.I)


def _scan_raw(path: str) -> Dict[str, Any]:
    """Read reward/terminated for one raw (cheap, no sim)."""
    with h5py.File(path, "r") as f:
        g = f[f"data/demo_0"]
        reward = g["reward"][:]
        term = g["terminated"][:]
    pos = np.flatnonzero(reward > 0)
    term_idx = np.flatnonzero(term)
    return {
        "path": path,
        "name": os.path.basename(path),
        "length": int(len(reward)),
        "n_events": int(len(pos)),
        "first_event": int(pos[0]) if len(pos) else -1,
        "last_event": int(pos[-1]) if len(pos) else -1,
        "term_start": int(term_idx[0]) if len(term_idx) else -1,
    }


def _scene_names(path: str) -> set:
    """Object instance names matching can_of_soda / trash_can in the raw's scene_file."""
    names: set = set()
    try:
        with h5py.File(path, "r") as f:
            sf = f["data"].attrs.get("scene_file", "")
        j = json.loads(sf)

        def walk(o: Any) -> None:
            if isinstance(o, dict):
                for k, v in o.items():
                    if isinstance(k, str) and _NAME_RE.search(k):
                        names.add(k)
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)

        walk(j)
    except Exception:
        pass
    return names


def _trash_clean(r: Dict[str, Any], events: set, lmin: int, lmax: int) -> bool:
    return (r["n_events"] in events
            and r["term_start"] >= r["last_event"] >= 0
            and lmin <= r["length"] <= lmax)


def _radio_clean(r: Dict[str, Any], lmin: int, lmax: int) -> bool:
    return r["n_events"] >= 1 and lmin <= r["length"] <= lmax


def _select(dirpath: str, predicate, sort_key, n: int) -> List[Dict[str, Any]]:
    rows = [_scan_raw(p) for p in sorted(glob.glob(os.path.join(dirpath, "*.hdf5")))]
    clean = [r for r in rows if predicate(r)]
    clean.sort(key=sort_key)
    return clean[:n], rows, clean


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trash-dir", default="/root/autodl-tmp/datasets/2025-challenge-rawdata/task-0001")
    ap.add_argument("--radio-dir", default="/root/autodl-tmp/datasets/2025-challenge-rawdata/task-0000")
    ap.add_argument("--n", type=int, default=50, help="episodes to pair (= clean raws of each kind to pick)")
    ap.add_argument("--oversample", type=float, default=1.3,
                    help="select n*oversample of each so the batch has margin for sim-time yield loss")
    ap.add_argument("--trash-events", default="3",
                    help="comma list of accepted reward-spike counts for trash (default '3')")
    ap.add_argument("--trash-len-min", type=int, default=2500)
    ap.add_argument("--trash-len-max", type=int, default=9000)
    ap.add_argument("--radio-len-min", type=int, default=1200)
    ap.add_argument("--radio-len-max", type=int, default=3500)
    ap.add_argument("--out", default="outputs/hrc_batch_0001/pairing_manifest.json")
    args = ap.parse_args()

    events = {int(x) for x in str(args.trash_events).split(",") if x.strip()}
    pick = max(int(round(args.n * args.oversample)), args.n)

    # Prefer SHORT-but-complete trash raws (faster episodes), tie-break by name
    # (deterministic). Radio: just deterministic by name (all are ~uniform).
    trash_sel, trash_all, trash_clean = _select(
        args.trash_dir, lambda r: _trash_clean(r, events, args.trash_len_min, args.trash_len_max),
        lambda r: (r["length"], r["name"]), pick)
    radio_sel, radio_all, radio_clean = _select(
        args.radio_dir, lambda r: _radio_clean(r, args.radio_len_min, args.radio_len_max),
        lambda r: r["name"], pick)

    n_pair = min(len(trash_sel), len(radio_sel), pick)
    if n_pair < args.n:
        print(f"[prescreen] WARN: only {n_pair} clean pairs available "
              f"(trash_clean={len(trash_clean)} radio_clean={len(radio_clean)}); "
              f"loosen criteria if you need {args.n}.", flush=True)

    # Pair with (a) predicate instance-name check and (b) SCENE-OBJECT COMPATIBILITY:
    # the radio round replays task-0000 state INTO the scene built from the trash
    # raw's scene_file; every object in the radio state must exist there, else
    # og.sim.load_state asserts "Could not find object while deserializing" and the
    # radio round never runs (measured: trash episode_00010020 has a REDUCED 102-obj
    # scene missing doors/downlights -> incompatible with EVERY radio raw; the other
    # 64 trash raws were fine). Incompatible trash raws are skipped and the next
    # clean trash takes their place (radio side kept — radio raws were uniform).
    def _scene_objs(path: str) -> set:
        with h5py.File(path, "r") as f:
            sf = json.loads(f["data"].attrs["scene_file"])
        return (set(sf.get("objects_info", {}).get("init_info", {}).keys())
                | set(sf.get("state", {}).get("object_registry", {}).keys()))

    want = set(PREDICATE_CANS) | {PREDICATE_TRASH}
    episodes: List[Dict[str, Any]] = []
    name_warn = 0
    compat_skip = 0
    _objs_cache: Dict[str, set] = {}
    def _objs(p: str) -> set:
        if p not in _objs_cache:
            _objs_cache[p] = _scene_objs(p)
        return _objs_cache[p]
    trash_pool = list(trash_sel) + [t for t in trash_clean if t not in trash_sel]
    ti = 0
    for i in range(n_pair):
        rd = radio_sel[i]
        t = None
        while ti < len(trash_pool):
            cand_t = trash_pool[ti]; ti += 1
            try:
                extra = _objs(rd["path"]) - _objs(cand_t["path"]) - {"radio_89"}
            except Exception as exc:
                print(f"[prescreen] WARN: compat check failed for {cand_t['name']}: {exc}", flush=True)
                extra = None
            if extra:
                compat_skip += 1
                print(f"[prescreen] SKIP trash {cand_t['name']}: radio state has "
                      f"{len(extra)} object(s) missing from its scene (e.g. {sorted(extra)[:3]})", flush=True)
                continue
            t = cand_t
            break
        if t is None:
            print(f"[prescreen] WARN: ran out of compatible trash raws at pair {i}", flush=True)
            break
        missing = sorted(want - _scene_names(t["path"]))
        if missing:
            name_warn += 1
            print(f"[prescreen] WARN ep{i:04d}: {t['name']} missing predicate names {missing}", flush=True)
        episodes.append({
            "idx": i,
            "trash_hdf5": t["path"], "trash_len": t["length"], "trash_events": t["n_events"],
            "radio_hdf5": rd["path"], "radio_len": rd["length"],
            "predicate_names_ok": not missing,
            "est_replay_steps": t["length"] + rd["length"],
        })

    manifest = {
        "criteria": {"n": args.n, "oversample": args.oversample, "trash_events": sorted(events),
                     "trash_len": [args.trash_len_min, args.trash_len_max],
                     "radio_len": [args.radio_len_min, args.radio_len_max]},
        "stats": {"trash_total": len(trash_all), "trash_clean": len(trash_clean),
                  "radio_total": len(radio_all), "radio_clean": len(radio_clean),
                  "paired": n_pair, "predicate_name_warnings": name_warn},
        "episodes": episodes,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(manifest, open(args.out, "w"), indent=2)
    print(f"[prescreen] trash clean {len(trash_clean)}/{len(trash_all)}, "
          f"radio clean {len(radio_clean)}/{len(radio_all)} -> paired {n_pair} "
          f"(name warnings: {name_warn})", flush=True)
    tot = sum(e["est_replay_steps"] for e in episodes)
    print(f"[prescreen] selected total replay steps={tot} "
          f"(mean {tot // max(n_pair,1)}/ep; ep0010 was 9623); manifest -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
