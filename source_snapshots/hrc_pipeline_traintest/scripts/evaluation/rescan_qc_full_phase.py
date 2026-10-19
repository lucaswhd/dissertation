#!/usr/bin/env python
"""Read-only full-phase QC re-scan for an HRC batch (does NOT touch collection).

Motivation (DIAGNOSIS_hrc_batch_final_64ep.md): the live QC in
batch_collect_multitask.py computes every geometric metric ONLY on Relocate
frames (`m = sid == reloc_idx`), so falls / wall-hits / human-contact that
happen in GoToTask / ReturnToHuman / FaceHuman are invisible. This tool
recomputes the same signals over ALL moving phases from the recorded
trajectory + the static occupancy map, and reports which "PASS" episodes are
actually unusable.

Evidence sources (no simulation):
  * <ep>/data/trajectories.npz            — full concatenated timeline (=video frames)
  * <ep>/meta/episode_meta.json           — fsm_states order (Relocate index)
  * floor_trav_0.png     (1cm, walls+furniture+stairs)  -> clearance to ANY obstacle
  * floor_trav_no_obj_0.png (walls+stairs only)         -> distinguish wall/stair vs furniture
  * batch_manifest.json                    — the live QC verdict, for comparison

Occupancy convention calibrated 6/6 (HANDOFF §14.5):
  col = round(x/res + W/2); row = round(y/res + H/2); NO row flip; white(>0)=free.

Thresholds are TUNABLE constants below; each maps to a diagnosis root cause.
Verdicts: REJECT (L3, discard) / TRIM (L2, usable after cutting a segment) /
CLEAN (L0-L1). Old live-QC fails are carried over as REJECT.
"""
import os, sys, json, glob, math, argparse
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

# ---- tunable thresholds (see DIAGNOSIS §5) --------------------------------
FELL_DEG        = 20.0   # RC1/RC2: max|roll,pitch| in ANY phase -> violent fall (hard)
TIP_HARD_DEG    = 15.0   # RC2: a lean/stumble this steep is a fall-in-progress (hard).
                         # calibrated: clean episodes top out ~8deg; bad ones 13-145deg.
WALL_CLR_M      = 0.20   # RC2: robot-center clearance to a WALL/STAIR (no_obj map) -> hit (hard)
FURN_CLR_M      = 0.15   # RC1/RC2: clearance to furniture footprint (full map) but wall far -> graze (warn)
HIT_HUMAN_M     = 0.55   # RC2: min human-gap in Return/Face -> hit human (hard)
HIT_HUMAN_MINF  = 6      # frames it must stay under HIT_HUMAN_M (0.2s @30fps) to count
# RC3 spin measured as IN-PLACE rotation (turning while NOT translating): allow ONE big
# in-place turn (the shortest turn to the replay yaw, user-approved) and flag the rest.
# TRIM = cut the round1 GoToTask segment; the rest of the episode is usable.
SPIN_MAXRUN_DEG = 200.0  # a single contiguous in-place turn wider than this = wound long / spun
SPIN_EXTRA_DEG  = 90.0   # in-place rotation BEYOND the largest single in-place turn = spinning
DRAG_MS         = 1.5    # RC1: max base speed in any phase -> post-fall drag / blowup (hard)
FPS             = 30.0

MOVING = {"GoToTask", "ReturnToHuman", "FaceHuman", "Relocate"}
RETURN_FACE = {"ReturnToHuman", "FaceHuman"}
DEFAULT_FSM = ['WaitNearHuman','ReceiveCommand','GoToTask','ExecuteBehaviorTask','TaskSegmentDone',
               'ReturnToHuman','FaceHuman','WaitNextCommand','AllTasksDone','Relocate']

def load_maps(scene_layout):
    occ = np.asarray(Image.open(os.path.join(scene_layout, "floor_trav_0.png")).convert("L"))
    nob = np.asarray(Image.open(os.path.join(scene_layout, "floor_trav_no_obj_0.png")).convert("L"))
    H, W = occ.shape
    res = 59.32 / W
    D  = distance_transform_edt(occ > 0) * res
    Dn = distance_transform_edt(nob > 0) * res
    return D, Dn, H, W, res

def clr_fn(D, H, W, res):
    def f(x, y):
        r = int(round(y / res + H / 2.0)); c = int(round(x / res + W / 2.0))
        return float(D[r, c]) if (0 <= r < H and 0 <= c < W) else np.nan
    return f

def quat_rpy(q):
    qx, qy, qz, qw = q[:,0], q[:,1], q[:,2], q[:,3]
    roll  = np.degrees(np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy)))
    pitch = np.degrees(np.arcsin(np.clip(2*(qw*qy-qz*qx), -1, 1)))
    yaw   = np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
    return roll, pitch, yaw

def runs(a):
    out=[]; s=0
    for i in range(1, len(a)+1):
        if i==len(a) or a[i]!=a[s]:
            out.append((int(a[s]), s, i)); s=i
    return out

def scan_episode(ep_dir, clrF, clrN):
    npz = os.path.join(ep_dir, "data", "trajectories.npz")
    if not os.path.exists(npz):
        return None
    d = np.load(npz, allow_pickle=True)
    rp = d['robot_pos'][:, :2]; hp = d['human_pos'][:, :2]
    fsm = d['fsm_state_id']; rid = d['round_id']; act = d['action']
    # fsm names from meta when available
    names = DEFAULT_FSM
    mp = glob.glob(os.path.join(ep_dir, "meta", "*.json"))
    if mp:
        try:
            st = json.load(open(mp[0])).get("fsm_states")
            if st: names = st
        except Exception: pass
    roll, pitch, yaw = quat_rpy(d['robot_quat'])
    tilt = np.maximum(np.abs(roll), np.abs(pitch))
    gap = np.linalg.norm(rp - hp, axis=1)
    step = np.concatenate([[0], np.linalg.norm(np.diff(rp, axis=0), axis=1)]) * FPS  # m/s
    rc_full = np.array([clrF(x, y) for x, y in rp])
    rc_wall = np.array([clrN(x, y) for x, y in rp])
    dyaw = np.abs((np.diff(yaw)+np.pi) % (2*np.pi) - np.pi)

    sig = {"fell": None, "wall_hit": None, "hit_human": None, "furn_graze": None,
           "spin_r1": None, "drag": None}
    def nm(v): return names[v] if v < len(names) else str(v)

    for v, s, e in runs(fsm):
        ph = nm(v); rnd = int(rid[s])
        if ph not in MOVING or (e - s) < 8:
            continue
        seg_tilt = tilt[s:e]; ti = int(seg_tilt.argmax())
        if seg_tilt[ti] > TIP_HARD_DEG and (sig["fell"] is None or seg_tilt[ti] > sig["fell"][0]):
            sig["fell"] = (float(seg_tilt[ti]), ph, rnd, s+ti)
        seg_vel = step[s:e]
        if float(seg_vel.max()) > DRAG_MS and (sig["drag"] is None or seg_vel.max() > sig["drag"][0]):
            sig["drag"] = (float(seg_vel.max()), ph, rnd, s+int(seg_vel.argmax()))
        # wall vs furniture clearance
        segw = rc_wall[s:e]; segf = rc_full[s:e]
        if np.isfinite(segw).any():
            wi = int(np.nanargmin(segw))
            if segw[wi] < WALL_CLR_M and (sig["wall_hit"] is None or segw[wi] < sig["wall_hit"][0]):
                sig["wall_hit"] = (float(segw[wi]), ph, rnd, s+wi)
        if np.isfinite(segf).any():
            fi = int(np.nanargmin(segf))
            if segf[fi] < FURN_CLR_M and rc_wall[s+fi] > 0.5 and \
               (sig["furn_graze"] is None or segf[fi] < sig["furn_graze"][0]):
                sig["furn_graze"] = (float(segf[fi]), ph, rnd, s+fi)
        # human contact only in Return/Face (Relocate follow gap is intended ~1m)
        if ph in RETURN_FACE:
            seg_gap = gap[s:e]
            under = int(np.sum(seg_gap < HIT_HUMAN_M))
            if under >= HIT_HUMAN_MINF:
                gi = int(seg_gap.argmin())
                if sig["hit_human"] is None or seg_gap[gi] < sig["hit_human"][0]:
                    sig["hit_human"] = (float(seg_gap[gi]), ph, rnd, s+gi)
        # spin: round1 GoToTask IN-PLACE rotation (turning while not translating).
        if ph == "GoToTask" and rnd == 1 and (e - s) > 2:
            seg_dyaw = dyaw[s:e-1]
            seg_speed = step[s+1:e]
            inpl = seg_speed < 0.05
            total_ip = float(np.degrees(seg_dyaw[inpl].sum())) if inpl.any() else 0.0
            maxrun = 0.0; run = 0.0
            for k in range(len(inpl)):
                if inpl[k]:
                    run += float(np.degrees(seg_dyaw[k]))
                else:
                    maxrun = max(maxrun, run); run = 0.0
            maxrun = max(maxrun, run)
            extra = total_ip - maxrun
            if sig["spin_r1"] is None or extra > sig["spin_r1"][2]:
                sig["spin_r1"] = (total_ip, maxrun, extra, ph, rnd, s)
    return sig

def verdict(sig, man):
    hard, warn = [], []
    # carry over live-QC hard fails
    for k in ("robot_fell","lost_human","clipped","no_reloc","spawn_tight",
              "reloc_incomplete","action_ok"):
        pass
    for r in man.get("fail_reasons", []):
        hard.append(f"live:{r}")
    if sig["fell"]:      hard.append(f"fell:{sig['fell'][0]:.0f}deg@{sig['fell'][1]}r{sig['fell'][2]}")
    if sig["wall_hit"]:  hard.append(f"wall_hit:{sig['wall_hit'][0]:.2f}m@{sig['wall_hit'][1]}r{sig['wall_hit'][2]}")
    if sig["hit_human"]: hard.append(f"hit_human:{sig['hit_human'][0]:.2f}m@{sig['hit_human'][1]}r{sig['hit_human'][2]}")
    if sig["drag"]:      hard.append(f"drag:{sig['drag'][0]:.1f}m/s@{sig['drag'][1]}r{sig['drag'][2]}")
    spin_trim = False
    if sig["spin_r1"]:
        total_ip, maxrun, extra = sig["spin_r1"][0], sig["spin_r1"][1], sig["spin_r1"][2]
        # IN-PLACE rotation: allow ONE big turn (maxrun, the shortest end-turn to replay
        # yaw); flag if that single turn wound past a half-circle OR there is extra in-place
        # rotation beyond it (a start stop-pivot / spinning).
        if maxrun > SPIN_MAXRUN_DEG or extra > SPIN_EXTRA_DEG:
            warn.append(f"spin_r1:inplace{total_ip:.0f}/run{maxrun:.0f}/extra{extra:.0f}deg(trim GoToTask)")
            spin_trim = True
    if sig["furn_graze"]: warn.append(f"furn_graze:{sig['furn_graze'][0]:.2f}m@{sig['furn_graze'][1]}")
    if hard:  return "REJECT", hard, warn
    if spin_trim: return "TRIM", hard, warn
    return "CLEAN", hard, warn

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", default="/root/autodl-tmp/hrc_pipeline_traintest/outputs/hrc_batch_final")
    ap.add_argument("--scene-layout", default="/root/autodl-tmp/datasets/behavior-1k-assets/scenes/house_double_floor_lower/layout")
    ap.add_argument("--out", default=None, help="write rescan_qc.json (default: <batch>/rescan_qc.json)")
    args = ap.parse_args()

    D, Dn, H, W, res = load_maps(args.scene_layout)
    clrF = clr_fn(D, H, W, res); clrN = clr_fn(Dn, H, W, res)
    man = {e['idx']: e for e in json.load(open(os.path.join(args.batch, "batch_manifest.json")))['episodes']}

    recs = []
    for idx in sorted(man):
        ep_dir = os.path.join(args.batch, f"ep_{idx:04d}")
        sig = scan_episode(ep_dir, clrF, clrN)
        if sig is None:
            continue
        vd, hard, warn = verdict(sig, man[idx])
        recs.append({"idx": idx, "old_pass": bool(man[idx]['passed']), "verdict": vd,
                     "hard": hard, "warn": warn,
                     "signals": {k: v for k, v in sig.items() if v}})

    # ---- report ----
    print(f"{'ep':>4}{'oldQC':>7}{'NEW':>9}  reasons")
    for r in recs:
        old = 'PASS' if r['old_pass'] else 'fail'
        flip = '  <<FLIP' if (r['old_pass'] and r['verdict'] == 'REJECT') else ''
        reasons = ' ; '.join([x for x in r['hard'] if not x.startswith('live:')] + r['warn']) or '-'
        print(f"{r['idx']:>4}{old:>7}{r['verdict']:>9}  {reasons}{flip}")

    old_pass = [r for r in recs if r['old_pass']]
    flips = [r for r in old_pass if r['verdict'] == 'REJECT']
    trims = [r for r in old_pass if r['verdict'] == 'TRIM']
    clean = [r for r in old_pass if r['verdict'] == 'CLEAN']
    print("\n" + "="*70)
    print(f"total episodes scanned : {len(recs)}")
    print(f"live-QC PASS           : {len(old_pass)}")
    print(f"  -> NEW REJECT (was PASS, actually unusable): {len(flips)}  {sorted(r['idx'] for r in flips)}")
    print(f"  -> NEW TRIM   (usable after cutting round1 GoToTask spin): {len(trims)}  {sorted(r['idx'] for r in trims)}")
    print(f"  -> CLEAN      (fully usable as-is)          : {len(clean)}  {sorted(r['idx'] for r in clean)}")
    for r in flips:
        print(f"    FLIP ep{r['idx']:04d}: {', '.join(x for x in r['hard'] if not x.startswith('live:'))}")

    out = args.out or os.path.join(args.batch, "rescan_qc.json")
    json.dump({"thresholds": {"FELL_DEG":FELL_DEG,"TIP_HARD_DEG":TIP_HARD_DEG,"WALL_CLR_M":WALL_CLR_M,
              "HIT_HUMAN_M":HIT_HUMAN_M,"SPIN_MAXRUN_DEG":SPIN_MAXRUN_DEG,
              "SPIN_EXTRA_DEG":SPIN_EXTRA_DEG,"DRAG_MS":DRAG_MS},
              "episodes": recs}, open(out, "w"), indent=1)
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
