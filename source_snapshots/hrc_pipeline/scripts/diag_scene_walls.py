#!/usr/bin/env python
# One-shot diagnostic: how are walls / building shell represented in this scene,
# and why does a "collision-free" spawn still clip? Replicates the EXACT spawn-
# time scene state (env.reset robot pose + step-0 object positions) used by
# hrc_multitask_demo, then inspects object categories, wall AABBs, the trav_map,
# and the two suspect points (spawn 3.11,-0.09 and relocate 0.40,0.70).
import argparse, os, sys
import h5py, numpy as np, torch as th
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hrc_utils import build_playback_env, to_numpy
import omnigibson as og


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-hdf5", default="/root/autodl-tmp/datasets/2025-challenge-rawdata/task-0046/episode_00460010.hdf5")
    ap.add_argument("--episode-id", type=int, default=0)
    ap.add_argument("--points", default="3.11,-0.09;0.40,0.70")
    args = ap.parse_args()

    env = build_playback_env(args.raw_hdf5, resolution=256)
    env.reset()
    # Replicate spawn-time state: keep env.reset robot pose, load step-0 objects.
    rpos = to_numpy(env.robots[0].get_position_orientation()[0]).copy()
    rquat = to_numpy(env.robots[0].get_position_orientation()[1]).copy()
    with h5py.File(args.raw_hdf5, "r") as f:
        g = f[f"data/demo_{args.episode_id}"]
        s0 = g["state"][0]; sz = int(g["state_size"][0])
    og.sim.load_state(th.tensor(s0[:sz], dtype=th.float32), serialized=True)
    env.robots[0].set_position_orientation(rpos, rquat)
    env.robots[0].keep_still()
    print(f"\n[diag] robot pose at spawn = ({rpos[0]:.3f}, {rpos[1]:.3f}, {rpos[2]:.3f})", flush=True)

    scene = og.sim.scenes[0]
    objs = list(env.scene.objects)
    print(f"[diag] env.scene.objects count = {len(objs)}", flush=True)

    # 1) category histogram
    from collections import Counter
    cats = Counter((getattr(o, "category", "") or "?").lower() for o in objs)
    print("[diag] categories (sorted):", flush=True)
    for c, n in sorted(cats.items()):
        print(f"       {c:28s} x{n}", flush=True)

    # 2) wall / structure objects + their AABBs
    print("\n[diag] WALL/STRUCTURE-like objects and AABBs:", flush=True)
    structure_like = []
    for o in objs:
        c = (getattr(o, "category", "") or "").lower()
        if any(k in c for k in ("wall", "ceiling", "floor", "door", "window", "structure", "roof")):
            try:
                lo, hi = o.aabb
                lo = to_numpy(lo); hi = to_numpy(hi)
                structure_like.append((getattr(o, "name", "?"), c, lo, hi))
                print(f"       {c:14s} {getattr(o,'name','?'):24s} "
                      f"lo=({lo[0]:.2f},{lo[1]:.2f},{lo[2]:.2f}) hi=({hi[0]:.2f},{hi[1]:.2f},{hi[2]:.2f})", flush=True)
            except Exception as e:
                print(f"       {c:14s} {getattr(o,'name','?'):24s} AABB-ERR {e}", flush=True)
    print(f"[diag] structure-like count = {len(structure_like)}", flush=True)

    # 3) trav_map presence + resolution
    tmap = getattr(scene, "trav_map", None)
    if tmap is not None and getattr(tmap, "floor_map", None) is not None and len(tmap.floor_map):
        fmap = tmap.floor_map[0]
        print(f"\n[diag] trav_map: floor_map shape={tuple(fmap.shape)} "
              f"resolution={getattr(tmap,'map_resolution','?')} default_erosion={getattr(tmap,'default_erosion_radius','?')}", flush=True)
    else:
        print("\n[diag] trav_map: NONE", flush=True)
        fmap = None

    # 4) probe the suspect points
    def navigable(x, y):
        if fmap is None: return None
        ij = tmap.world_to_map(th.tensor([float(x), float(y)], dtype=th.float32)).tolist()
        i, j = int(ij[0]), int(ij[1])
        H, W = int(fmap.shape[0]), int(fmap.shape[1])
        return bool(0 <= i < H and 0 <= j < W and int(fmap[i, j]) > 0)

    # snapshot all object AABBs once
    all_aabb = []
    for o in objs:
        try:
            lo, hi = o.aabb
            all_aabb.append((getattr(o, "name", "?"), (getattr(o, "category", "") or "").lower(),
                             to_numpy(lo), to_numpy(hi)))
        except Exception:
            pass

    for pstr in args.points.split(";"):
        x, y = [float(v) for v in pstr.split(",")]
        print(f"\n[diag] ===== POINT ({x:.2f}, {y:.2f}) =====", flush=True)
        print(f"       trav_map navigable? {navigable(x, y)}", flush=True)
        # human body box at this point
        br, bh, lift = 0.33, 1.80, 0.10
        z = float(rpos[2]); z = max(z, 0.0)
        bx0, bx1, by0, by1, bz0, bz1 = x-br, x+br, y-br, y+br, z+lift, z+bh
        # nearest objects by xy gap
        rows = []
        for name, cat, lo, hi in all_aabb:
            gx = max(lo[0]-bx1, bx0-hi[0]); gy = max(lo[1]-by1, by0-hi[1]); gz = max(lo[2]-bz1, bz0-hi[2])
            sep3 = max(gx, gy, gz)             # 3D box gap (<0 => overlap)
            gxy = max(gx, gy)                  # xy-only gap
            rows.append((gxy, sep3, name, cat, lo, hi))
        rows.sort(key=lambda r: r[0])
        print(f"       nearest objects (by xy gap; body box z=[{bz0:.2f},{bz1:.2f}]):", flush=True)
        for gxy, sep3, name, cat, lo, hi in rows[:8]:
            flag = "  <-- 3D OVERLAP" if sep3 < 0 else ""
            print(f"         xy_gap={gxy:+.3f} 3d_gap={sep3:+.3f} {cat:14s} {name:22s} "
                  f"z=[{lo[2]:.2f},{hi[2]:.2f}]{flag}", flush=True)

    og.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
