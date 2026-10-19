#!/usr/bin/env python
# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
#
# Offline preview / validator for the general collision-free human spawn
# (ARCHITECTURE.md §14).  Loads a base-task HDF5, restores its step-0 state, and
# runs the SAME placement logic the collection pipeline uses
# (KinematicHumanoidActor._compute_spawn_pose_collision_free) — so what you see
# here is exactly what collection will do.  It then INDEPENDENTLY re-checks that
# the chosen body box overlaps no scene-object AABB and prints the clearance.
#
# Because collision-free placement needs no per-scene anchor, this script is no
# longer required before collection — it is just a fast (no full episode replay)
# way to eyeball / sanity-check a new scene, and to optionally print a
# `human_spawn:` line if you want to pin the chosen spot.
import argparse
import json
import math
import os
import sys

import h5py
import numpy as np
import torch as th

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hrc_humanoid import KinematicHumanoidActor
from hrc_utils import build_playback_env, to_numpy

import omnigibson as og


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-hdf5", required=True)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--human-usd", default="/root/hrc_pipeline/assets/character.usda")
    parser.add_argument("--distance", type=float, default=1.8)
    parser.add_argument("--jitter-xy", type=float, default=0.0,
                        help="per-episode xy jitter (m); 0 = none (for a clean preview)")
    parser.add_argument("--jitter-yaw", type=float, default=0.0, help="per-episode yaw jitter (deg)")
    parser.add_argument("--search-json", default=None,
                        help="optional JSON overriding human_spawn_search params")
    args = parser.parse_args()

    env = build_playback_env(args.raw_hdf5, resolution=256)
    env.reset()

    with h5py.File(args.raw_hdf5, "r") as f:
        grp = f[f"data/demo_{args.episode_id}"]
        state0 = grp["state"][0]
        size0 = int(grp["state_size"][0])
    og.sim.load_state(th.tensor(state0[:size0], dtype=th.float32), serialized=True)

    robot = env.robots[0]
    rpos = to_numpy(robot.get_position_orientation()[0]).astype(np.float32)
    print(f"[probe] robot initial xy=({rpos[0]:.3f}, {rpos[1]:.3f})", flush=True)

    jitter = None
    if args.jitter_xy > 0 or args.jitter_yaw > 0:
        jitter = {"xy_m": args.jitter_xy, "yaw_deg": args.jitter_yaw}
    search = json.loads(args.search_json) if args.search_json else None

    # Construct the actor WITHOUT spawning its visual (spawn_near_robot is not
    # called) and run the real placement routine.
    actor = KinematicHumanoidActor(
        env=env,
        usd_path=args.human_usd,
        name="human_npc",
        distance=args.distance,
        collision_proxy=None,
        spawn_collision_free=True,
        spawn_jitter=jitter,
        spawn_seed=args.episode_id,
        spawn_search=search,
    )
    pos, yaw, debug = actor._compute_spawn_pose(args.distance)
    rec = actor.spawn_record

    # Independent verification: re-collect AABBs and measure clearance of the
    # chosen body box to the nearest obstacle (negative => overlap => clipping).
    body_radius = float(rec["search"]["body_radius"])
    body_height = float(rec["search"]["body_height"])
    base_lift = float(rec["search"]["base_lift"])
    z = float(pos[2])
    aabbs = actor._collect_obstacle_aabbs(pos[:2].astype(np.float32),
                                          max_range=float(rec["search"]["max_dist"]) + body_radius + 1.0)
    x0, x1 = pos[0] - body_radius, pos[0] + body_radius
    y0, y1 = pos[1] - body_radius, pos[1] + body_radius
    z0, z1 = z + base_lift, z + body_height
    nearest = None
    nearest_gap = 1e9
    overlap_any = False
    for name, cat, lo, hi in aabbs:
        # signed gap per axis: >0 means separated by that much on the axis
        gx = max(lo[0] - x1, x0 - hi[0])
        gy = max(lo[1] - y1, y0 - hi[1])
        gz = max(lo[2] - z1, z0 - hi[2])
        sep = max(gx, gy, gz)  # boxes overlap iff sep < 0
        if sep < 0:
            overlap_any = True
        if sep < nearest_gap:
            nearest_gap = sep
            nearest = (name, cat)

    print("\n" + "=" * 64, flush=True)
    print(f"[probe] chosen spawn: xy=({pos[0]:.3f}, {pos[1]:.3f}) yaw={yaw:.4f}", flush=True)
    print(f"[probe] tier={rec['tier']} tag={rec['tag']} obstacles_checked={len(aabbs)}", flush=True)
    print(f"[probe] nearest obstacle: {nearest} gap={nearest_gap:+.3f} m "
          f"(>0 = clear, <0 = OVERLAP)", flush=True)
    if overlap_any:
        print("[probe] ❌ FAIL: chosen spawn OVERLAPS geometry (would clip).", flush=True)
    else:
        print("[probe] ✅ PASS: chosen spawn is collision-free.", flush=True)
    print(f"[probe] (optional) pin it:  human_spawn: {{x: {pos[0]:.3f}, y: {pos[1]:.3f}, "
          f"yaw: {yaw:.4f}}}", flush=True)
    print("=" * 64 + "\n", flush=True)
    print(f"[probe] full spawn_record = {json.dumps(rec, default=float)}", flush=True)

    og.shutdown()
    return 1 if overlap_any else 0


if __name__ == "__main__":
    sys.exit(main())
