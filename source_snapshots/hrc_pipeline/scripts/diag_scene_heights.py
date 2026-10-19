#!/usr/bin/env python
"""Analyze object AABB height (z) distribution in a scene to choose a sensible
``body_height`` for the collision-free human-spawn clearance check
(``_human_box_clear`` uses a vertical column [z+base_lift, z+body_height]).

It replicates the spawn-time scene state (like diag_scene_walls.py), snapshots
every obstacle AABB (excluding floor/ceiling shell + robot), and reports:
  * z-range per object (bottom lo[2] / top hi[2]),
  * how the "blocker count" grows as the column height H increases,
  * which objects are "overhead-only" (bottom above the floor zone) and would
    wrongly veto a floor standoff at H=1.8 but not at a lower H.
"""
import argparse, os, sys
from collections import Counter
import h5py, numpy as np, torch as th
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hrc_utils import build_playback_env, to_numpy
import omnigibson as og

SKIP = {"floors", "floor", "ceilings", "ceiling", "lawn", "driveway", "roof", "rug", "carpet"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-hdf5", default="/root/autodl-tmp/datasets/2025-challenge-rawdata/task-0046/episode_00460010.hdf5")
    ap.add_argument("--episode-id", type=int, default=0)
    ap.add_argument("--base-lift", type=float, default=0.10)
    args = ap.parse_args()

    env = build_playback_env(args.raw_hdf5, resolution=256)
    env.reset()
    rpos = to_numpy(env.robots[0].get_position_orientation()[0]).copy()
    rquat = to_numpy(env.robots[0].get_position_orientation()[1]).copy()
    with h5py.File(args.raw_hdf5, "r") as f:
        g = f[f"data/demo_{args.episode_id}"]
        s0 = g["state"][0]; sz = int(g["state_size"][0])
    og.sim.load_state(th.tensor(s0[:sz], dtype=th.float32), serialized=True)
    env.robots[0].set_position_orientation(rpos, rquat)
    env.robots[0].keep_still()

    robot_names = {getattr(r, "name", None) for r in env.robots}
    floor_z = 0.0  # human stands on floor; column base ~ floor_z + base_lift

    rows = []  # (name, cat, lo, hi)
    for o in env.scene.objects:
        cat = (getattr(o, "category", "") or "?").lower()
        if cat in SKIP or getattr(o, "name", None) in robot_names:
            continue
        try:
            lo, hi = o.aabb
            rows.append((getattr(o, "name", "?"), cat, to_numpy(lo), to_numpy(hi)))
        except Exception:
            pass

    print(f"\n[scene] raw_hdf5={os.path.basename(args.raw_hdf5)}  obstacle objects={len(rows)}  base_lift={args.base_lift}")

    # ---- 1. blocker count vs column height H (xy ignored => pure vertical overlap) ----
    z0 = floor_z + args.base_lift
    print("\n[A] 垂直重叠物体数 vs 柱高 H  (柱 = [z0, z0+H], z0={:.2f}):".format(z0))
    print(f"    {'H(m)':>6} {'overlap_objs':>12}   (越高否决越多)")
    Hs = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    for H in Hs:
        z1 = z0 + H
        n = sum(1 for _, _, lo, hi in rows if (z0 <= hi[2] and z1 >= lo[2]))
        bar = "#" * n
        print(f"    {H:>6.2f} {n:>12}   {bar}")

    # ---- 2. objects by bottom height lo[2]: floor-standing vs overhead ----
    print("\n[B] 物体底部高度 lo[2] 分布 (决定'是否落地障碍'):")
    buckets = [(-1, 0.10, "贴地(真正挡底盘)"), (0.10, 0.30, "近地"),
               (0.30, 0.70, "矮台下沿"), (0.70, 1.10, "台面/桌面层"),
               (1.10, 1.60, "壁柜/搁架下沿"), (1.60, 99, "高处/吊顶层")]
    for lo_b, hi_b, label in buckets:
        names = [(c, lo, hi) for _, c, lo, hi in rows if lo_b <= lo[2] < hi_b]
        print(f"    lo[2]∈[{lo_b:>5.2f},{hi_b:>5.2f}) {label:14s}: {len(names):3d} 个")

    # ---- 3. overhead-only objects that veto at H=1.8 but not at H=1.0 ----
    print("\n[C] '悬空/高处'物体 (lo[2] >= 1.0, 底盘下方其实可通行) —— 1.8m 柱会错误否决:")
    overhead = sorted([(lo[2], hi[2], c, n) for n, c, lo, hi in rows if lo[2] >= 1.0], key=lambda r: r[0])
    cat_hist = Counter(c for _, _, c, _ in [(a, b, c, d) for a, b, c, d in overhead])
    for z_lo, z_hi, c, n in overhead[:25]:
        print(f"      z=[{z_lo:.2f},{z_hi:.2f}]  {c:18s} {n}")
    print(f"    overhead(lo>=1.0) 物体总数 = {len(overhead)}；类别: {dict(cat_hist)}")

    # ---- 4. tall floor-standing furniture (the ones we DO want to avoid) ----
    print("\n[D] 落地高家具 (lo[2]<0.3 且 hi[2]>0.9，真正需要避开的台面/电器/柜):")
    tall = sorted([(hi[2], c, n, lo) for n, c, lo, hi in rows if lo[2] < 0.3 and hi[2] > 0.9], key=lambda r: -r[0])
    for z_hi, c, n, lo in tall[:25]:
        print(f"      top={z_hi:.2f}  {c:18s} {n}")

    og.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
