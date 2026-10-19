#!/usr/bin/env python
# Validate the final-ReturnToHuman fix (ARCHITECTURE.md §18): when the standoff
# target is an isolated navigable pocket behind a counter (A* unreachable), the
# robot must NOT straight-line through the counter — it should reroute to the
# nearest reachable point. Reproduces the logged scenario src=(5.22,-0.17),
# dst=(7.26,1.16).
import argparse, os, sys
import h5py, numpy as np, torch as th
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hrc_humanoid import KinematicHumanoidActor
from hrc_utils import build_playback_env, to_numpy
import hrc_multitask_demo as M
from hrc_multitask_demo import _plan_robot_path, _collect_nav_obstacles
import omnigibson as og

# solid furniture the robot body must not pass through (for the clip check)
BLOCK = ("countertop", "counter", "bar", "table", "cabinet", "fridge", "oven",
         "stove", "dishwasher", "sink", "shelf", "burner", "range_hood")


def path_clear_of(path, aabbs, robot_r=0.25, step=0.05):
    mind, who = 1e9, None
    for i in range(len(path) - 1):
        a = np.asarray(path[i], np.float32)[:2]; b = np.asarray(path[i + 1], np.float32)[:2]
        L = float(np.linalg.norm(b - a)); n = max(int(L / step), 1)
        for k in range(n + 1):
            p = a + (b - a) * (k / n)
            for name, lo, hi in aabbs:
                cx = min(max(p[0], lo[0]), hi[0]); cy = min(max(p[1], lo[1]), hi[1])
                d = float(np.hypot(p[0] - cx, p[1] - cy))
                if d < mind:
                    mind, who = d, name
    return mind, who


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-hdf5", default="/root/autodl-tmp/datasets/2025-challenge-rawdata/task-0046/episode_00460010.hdf5")
    ap.add_argument("--src", default="5.22,-0.17")
    ap.add_argument("--dst", default="7.26,1.16")
    args = ap.parse_args()
    env = build_playback_env(args.raw_hdf5, resolution=256); env.reset()
    rpos = to_numpy(env.robots[0].get_position_orientation()[0]).copy()
    rquat = to_numpy(env.robots[0].get_position_orientation()[1]).copy()
    with h5py.File(args.raw_hdf5, "r") as f:
        g = f["data/demo_0"]; s0 = g["state"][0]; sz = int(g["state_size"][0])
    og.sim.load_state(th.tensor(s0[:sz], dtype=th.float32), serialized=True)
    env.robots[0].set_position_orientation(rpos, rquat); env.robots[0].keep_still()
    z = max(float(rpos[2]), 0.0)

    actor = KinematicHumanoidActor(env=env, usd_path="/root/hrc_pipeline/assets/character.usda",
                                   name="human_npc", distance=1.8, collision_proxy=None)
    _collect_nav_obstacles(env)   # populates M._NAV_OBSTACLE_AABBS (tables)
    trav = actor._try_get_trav_map()
    sx, sy = [float(v) for v in args.src.split(",")]
    dx, dy = [float(v) for v in args.dst.split(",")]
    src = np.array([sx, sy, z], np.float32); dst = np.array([dx, dy], np.float32)
    actor.base_pos = np.array([dx + 0.8, dy + 0.5, z], np.float32)  # ~human, for human-deflect

    blocking = []
    for o in env.scene.objects:
        c = (getattr(o, "category", "") or "").lower()
        if any(s in c for s in BLOCK):
            lo, hi = o.aabb
            blocking.append((getattr(o, "name", "?"), to_numpy(lo).astype(np.float32), to_numpy(hi).astype(np.float32)))

    old = actor._nav_path_between(src[:2], dst, z, trav, 0)
    new = _plan_robot_path(actor, src, dst, z=z, avoid_objects=[])
    o_d, o_who = path_clear_of([np.array([sx, sy]), np.array([dx, dy])], blocking)  # straight line clip
    n_d, n_who = path_clear_of(new, blocking)

    print(f"\n[val] src=({sx:.2f},{sy:.2f}) dst=({dx:.2f},{dy:.2f})  blocking_furniture={len(blocking)}", flush=True)
    print(f"[val] OLD direct A*: pts={len(old)}  ({'UNREACHABLE -> straight line' if len(old) < 2 else 'reachable'})", flush=True)
    print(f"[val] OLD straight line clears blocking by {o_d:.2f}m (worst {o_who})  "
          f"{'CLIPS' if o_d < 0.25 else 'ok'}", flush=True)
    print(f"[val] NEW _plan_robot_path: pts={len(new)}  end=({new[-1][0]:.2f},{new[-1][1]:.2f})", flush=True)
    print(f"[val] NEW path clears blocking by {n_d:.2f}m (worst {n_who})  "
          f"{'CLIPS' if n_d < 0.20 else 'OK clears'}", flush=True)
    ok = (len(new) >= 2) and (n_d >= 0.20)
    print(f"\n[val] ==== VERDICT: {'PASS' if ok else 'CHECK'} ==== "
          f"(rerouted={len(new)>=2}, clears_furniture={n_d>=0.20})", flush=True)
    og.shutdown()
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
