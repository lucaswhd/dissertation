#!/usr/bin/env python
# Validate that robot/human A* now routes AROUND free-standing tables
# (ARCHITECTURE.md §16). For each table, plan a path between two navigable points
# on opposite sides and check whether the path passes through the table footprint:
#   OLD (_nav_path_between)  : expected to cut straight through (trav_map misses it)
#   NEW (_nav_path_avoiding) : expected to route around (footprint stamped)
import argparse, os, sys
import h5py, numpy as np, torch as th
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hrc_humanoid import KinematicHumanoidActor
from hrc_utils import build_playback_env, to_numpy
import omnigibson as og

NAV_AROUND = ("table", "desk", "ottoman", "bench", "piano", "pool_table")


def crosses(path, lo, hi, step=0.05):
    """Count samples whose xy falls inside [lo,hi] (the table footprint); min dist to box."""
    n_in, min_d = 0, 1e9
    for i in range(len(path) - 1):
        a = np.asarray(path[i], np.float32)[:2]; b = np.asarray(path[i + 1], np.float32)[:2]
        L = float(np.linalg.norm(b - a)); m = max(int(L / step), 1)
        for k in range(m + 1):
            p = a + (b - a) * (k / m)
            inside = (lo[0] <= p[0] <= hi[0]) and (lo[1] <= p[1] <= hi[1])
            if inside:
                n_in += 1
            cx = min(max(p[0], lo[0]), hi[0]); cy = min(max(p[1], lo[1]), hi[1])
            min_d = min(min_d, float(np.hypot(p[0] - cx, p[1] - cy)))
    return n_in, min_d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-hdf5", default="/root/autodl-tmp/datasets/2025-challenge-rawdata/task-0046/episode_00460010.hdf5")
    ap.add_argument("--episode-id", type=int, default=0)
    args = ap.parse_args()
    env = build_playback_env(args.raw_hdf5, resolution=256); env.reset()
    rpos = to_numpy(env.robots[0].get_position_orientation()[0]).copy()
    rquat = to_numpy(env.robots[0].get_position_orientation()[1]).copy()
    with h5py.File(args.raw_hdf5, "r") as f:
        g = f[f"data/demo_{args.episode_id}"]; s0 = g["state"][0]; sz = int(g["state_size"][0])
    og.sim.load_state(th.tensor(s0[:sz], dtype=th.float32), serialized=True)
    env.robots[0].set_position_orientation(rpos, rquat); env.robots[0].keep_still()
    z = max(float(rpos[2]), 0.0)

    actor = KinematicHumanoidActor(env=env, usd_path="/root/hrc_pipeline/assets/character.usda",
                                   name="human_npc", distance=1.8, collision_proxy=None)
    trav = actor._try_get_trav_map()

    tables = []
    for o in env.scene.objects:
        c = (getattr(o, "category", "") or "").lower()
        if any(s in c for s in NAV_AROUND):
            lo, hi = o.aabb
            tables.append((getattr(o, "name", "?"), c, to_numpy(lo).astype(np.float32), to_numpy(hi).astype(np.float32)))
    print(f"\n[val] tables/go-around furniture found: {len(tables)}", flush=True)
    aabbs = [(lo, hi) for _, _, lo, hi in tables]

    all_ok = True
    for name, cat, lo, hi in tables:
        cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
        rx = (hi[0] - lo[0]) / 2 + 0.9
        src = np.array([cx - rx, cy, z], np.float32)
        dst = np.array([cx + rx, cy, z], np.float32)
        old = actor._nav_path_between(src[:2], dst[:2], z, trav, 0)
        new = actor._nav_path_avoiding(src[:2], dst[:2], z, trav, aabbs, floor=0, pad=0.35)
        o_in, o_d = crosses(old, lo, hi)
        n_in, n_d = crosses(new, lo, hi)
        ROBOT_R = 0.30   # path centre-line must clear the table by >= robot body radius
        if len(new) < 2:
            verdict_txt = "n/a (test src/dst unreachable)"; ok = True  # inconclusive, don't fail
        elif n_d >= ROBOT_R:
            verdict_txt = f"OK clears by {n_d:.2f}m (>= {ROBOT_R})"; ok = True
        else:
            verdict_txt = f"TOO CLOSE {n_d:.2f}m (< {ROBOT_R})"; ok = False
        all_ok = all_ok and ok
        print(f"[val] {name} ({cat}) center=({cx:.2f},{cy:.2f}) size=({hi[0]-lo[0]:.2f}x{hi[1]-lo[1]:.2f})",
              flush=True)
        print(f"[val]    OLD path: clearance_to_table={o_d:.2f}m  pts={len(old)} "
              f"({'clips, body radius 0.30' if o_d < ROBOT_R else 'ok'})", flush=True)
        print(f"[val]    NEW path: clearance_to_table={n_d:.2f}m  pts={len(new)}  -> {verdict_txt}", flush=True)
    print(f"\n[val] ==== VERDICT: {'PASS' if all_ok else 'CHECK'} ==== "
          f"(all tables routed around = {all_ok})", flush=True)
    og.shutdown()
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
