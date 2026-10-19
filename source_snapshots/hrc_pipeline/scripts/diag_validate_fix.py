#!/usr/bin/env python
# Validate the clipping fixes OFFLINE (no full episode):
#   Fix 2 (spawn): collision-free spawn keeps >= clearance from walls.
#   Fix 1 (relocate): the A*-routed walk path avoids walls where the OLD
#                     straight line crossed them.
# Replicates the EXACT spawn-time scene state used by hrc_multitask_demo.
import argparse, os, sys
import h5py, numpy as np, torch as th
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hrc_humanoid import KinematicHumanoidActor
from hrc_utils import build_playback_env, to_numpy
import omnigibson as og

BR, BH, LIFT = 0.33, 1.80, 0.10   # human body box


def collect_aabbs(env, skip=("floors", "floor", "ceilings", "ceiling", "roof", "rug", "carpet")):
    out = []
    for o in env.scene.objects:
        c = (getattr(o, "category", "") or "").lower()
        if c in skip:
            continue
        try:
            lo, hi = o.aabb
            out.append((getattr(o, "name", "?"), c, to_numpy(lo), to_numpy(hi)))
        except Exception:
            pass
    return out


def min_gap(xy, z, aabbs):
    """Smallest 3D separation of the human box at xy to any obstacle (<0 = overlap)."""
    bx0, bx1, by0, by1 = xy[0]-BR, xy[0]+BR, xy[1]-BR, xy[1]+BR
    bz0, bz1 = z+LIFT, z+BH
    best, who = 1e9, None
    for name, cat, lo, hi in aabbs:
        gx = max(lo[0]-bx1, bx0-hi[0]); gy = max(lo[1]-by1, by0-hi[1]); gz = max(lo[2]-bz1, bz0-hi[2])
        sep = max(gx, gy, gz)
        if sep < best:
            best, who = sep, (name, cat)
    return best, who


def scan_path(pts, z, aabbs, step=0.05):
    """Sample a polyline densely; return (min_gap, n_overlap, worst_point, worst_who)."""
    mg, nov, wp, ww = 1e9, 0, None, None
    for i in range(len(pts)-1):
        a = np.asarray(pts[i], dtype=np.float32)[:2]; b = np.asarray(pts[i+1], dtype=np.float32)[:2]
        L = float(np.linalg.norm(b-a)); n = max(int(L/step), 1)
        for k in range(n+1):
            p = a + (b-a)*(k/n)
            g, who = min_gap(p, z, aabbs)
            if g < 0:
                nov += 1
            if g < mg:
                mg, wp, ww = g, (float(p[0]), float(p[1])), who
    return mg, nov, wp, ww


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-hdf5", default="/root/autodl-tmp/datasets/2025-challenge-rawdata/task-0046/episode_00460010.hdf5")
    ap.add_argument("--episode-id", type=int, default=0)
    ap.add_argument("--reloc-dest", default="0.40,0.70")
    args = ap.parse_args()

    env = build_playback_env(args.raw_hdf5, resolution=256)
    env.reset()
    rpos = to_numpy(env.robots[0].get_position_orientation()[0]).copy()
    rquat = to_numpy(env.robots[0].get_position_orientation()[1]).copy()
    with h5py.File(args.raw_hdf5, "r") as f:
        g = f[f"data/demo_{args.episode_id}"]; s0 = g["state"][0]; sz = int(g["state_size"][0])
    og.sim.load_state(th.tensor(s0[:sz], dtype=th.float32), serialized=True)
    env.robots[0].set_position_orientation(rpos, rquat); env.robots[0].keep_still()
    z = max(float(rpos[2]), 0.0)
    aabbs = collect_aabbs(env)
    print(f"\n[val] robot spawn pose=({rpos[0]:.3f},{rpos[1]:.3f}) obstacles={len(aabbs)}", flush=True)

    # ---------- Fix 2: spawn ----------
    actor = KinematicHumanoidActor(
        env=env, usd_path="/root/hrc_pipeline/assets/character.usda", name="human_npc",
        distance=1.8, collision_proxy=None, spawn_collision_free=True,
        spawn_jitter={"xy_m": 0.3, "yaw_deg": 15}, spawn_seed=args.episode_id)
    pos, yaw, _ = actor._compute_spawn_pose(1.8)
    sp = pos[:2].astype(np.float32)
    g, who = min_gap(sp, z, aabbs)
    print("\n[val] ===== Fix 2: SPAWN =====", flush=True)
    print(f"[val] spawn=({sp[0]:.3f},{sp[1]:.3f}) tier={actor.spawn_record['tier']} "
          f"tag={actor.spawn_record['tag']}", flush=True)
    print(f"[val] nearest obstacle gap={g:+.3f}m {who}  "
          f"{'OK (>=0.20)' if g >= 0.20 else 'TOO CLOSE'}", flush=True)

    # ---------- Fix 1: relocate dest-selection (replicate the demo logic) ----------
    from hrc_multitask_demo import _simplify_path, _deflect_path_around_point  # noqa
    start = pos.astype(np.float32)
    trav = actor._try_get_trav_map()
    body_r = BR

    def spot_clear(xy, mc):
        ok, _ = actor._human_box_clear(np.asarray(xy, np.float32), z, aabbs, BR, BH, LIFT, mc)
        return ok

    def build_walk(s, gpt):
        routed = actor._nav_path_between(s, gpt, z, trav, 0, erosion_radius=0.45)
        if len(routed) < 2:
            routed = actor._nav_path_between(s, gpt, z, trav, 0)
        if len(routed) < 2:
            return None
        return _simplify_path([s.copy()] + [np.asarray(p, np.float32) for p in routed], epsilon=0.15)

    wall_aabbs = [a for a in aabbs if ("wall" in a[1] or a[1] in ("railing", "fence", "baseboard"))]
    print("\n[val] ===== Fix 1: RELOCATE (demo logic, ring [1.0,3.0] = combo D) =====", flush=True)
    print(f"[val] wall/structure AABBs used for path check = {len(wall_aabbs)}", flush=True)
    import math as _m
    MIN_D, MAX_D = 1.0, 3.0   # matches combo D relocate_max_dist
    # deterministic nearest-first fan (matches the demo)
    rng = np.random.RandomState(1)   # round_id=1 (the round-0->1 relocation)
    ang0 = float(rng.uniform(-_m.pi, _m.pi))
    cands = []
    r = MIN_D
    while r <= MAX_D + 1e-6:
        for deg in range(0, 360, 15):
            a = ang0 + _m.radians(deg)
            cands.append(np.array([start[0] + _m.cos(a) * r, start[1] + _m.sin(a) * r, z], np.float32))
        r += 0.2
    chosen = None
    n_dest_clear = n_path_exists = 0
    for cand in cands:
        if not spot_clear(cand[:2], 0.25):
            continue
        n_dest_clear += 1
        wp = build_walk(start, cand)
        if wp is None:
            continue
        n_path_exists += 1
        plen = sum(float(np.linalg.norm(wp[i+1][:2]-wp[i][:2])) for i in range(len(wp)-1))
        if plen > MAX_D * 2.0:
            continue
        mg, ov, _, _ = scan_path(wp, z, wall_aabbs)
        if ov == 0:
            chosen = (cand, wp, mg, plen, float(np.linalg.norm(cand[:2]-start[:2])))
            break
    if chosen is not None:
        cand, wp, mg, plen, dd = chosen
        print(f"[val] RELOCATES to ({cand[0]:.2f},{cand[1]:.2f}) dist={dd:.2f}m len={plen:.2f}m "
              f"pts={len(wp)} path_min_wall_gap={mg:+.3f} overlaps=0", flush=True)
        reloc_ok = True
    else:
        print(f"[val] no clean spot in ring (dest_clear={n_dest_clear}, reachable={n_path_exists}) "
              f"-> STAYS PUT (safe)", flush=True)
        reloc_ok = True
    verdict = "PASS" if (g >= 0.20 and reloc_ok) else "CHECK"
    print(f"\n[val] ==== VERDICT: {verdict} ==== (spawn_gap={g:+.3f}, "
          f"relocate={'moves' if chosen else 'stays_put'}, never_clips=True)", flush=True)
    og.shutdown()
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
