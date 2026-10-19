# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
# See LICENSE in the project root for full terms.

"""Batch driver for the multitask "trash + radio" scale-up (HANDOFF §6.2).

Runs each episode in its OWN subprocess (Isaac Sim is a singleton — it cannot be
re-initialised in-process), overriding the Round-0 trash source (--raw-hdf5) and
the Round-1 radio source (--external-hdf5 1:<file>) per the pairing manifest from
prescreen_sources.py. After each run it parses the episode's commands.json and
keeps the episode only if BOTH rounds report success (round 0 Inside(cans,trash)
AND round 1 ToggledOn(radio)). Episodes are run SERIALLY: one sim owns the single
GPU at a time, fully boots, runs, and exits before the next starts.

Stepwise by design (the user collects 50 in batches, not one shot):
  --start / --count  : run manifest[start : start+count] only
  --resume           : skip episodes whose output already has a commands.json
A cumulative batch_manifest.json under --out-root records per-episode rc, wall
time, both round successes, and the running yield, so a later batch appends to it.

Example (first pilot of 5):
  python scripts/collection/batch_collect_multitask.py \
      --manifest outputs/hrc_batch_0001/pairing_manifest.json \
      --out-root outputs/hrc_batch_0001 --start 0 --count 5
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))  # .../hrc_pipeline_traintest
_DEMO = os.path.join(_PROJECT, "scripts", "hrc_multitask_demo.py")


def _parse_round_success(out_dir: str) -> Dict[str, Any]:
    """Read <out_dir>/annotations/commands.json -> per-round success."""
    cj = os.path.join(out_dir, "annotations", "commands.json")
    res = {"round0": None, "round1": None, "detail": {}, "commands_json": cj}
    if not os.path.exists(cj):
        return res
    try:
        cmds = json.load(open(cj))
        for c in cmds:
            rid = c.get("round_id")
            res[f"round{rid}"] = bool(c.get("success"))
            res["detail"][f"round{rid}"] = c.get("success_detail", "")
    except Exception as exc:
        res["detail"]["parse_error"] = str(exc)
    return res


def _load_manifest_done(batch_manifest: str) -> Dict[int, Dict[str, Any]]:
    if os.path.exists(batch_manifest):
        try:
            return {e["idx"]: e for e in json.load(open(batch_manifest)).get("episodes", [])}
        except Exception:
            pass
    return {}


# "Loosest" data-correctness QC (2026-07-07): the bar is TRAINABLE data, not nav
# polish. Only TRUE errors are rejected; imperfect-but-correct follows (robot lags,
# jams ~5s at a mirror, gap large) are ACCEPTED — their actions/observations are still
# consistent. Thresholds are deliberately permissive.
FPS = 30  # recording rate; config `fps`. Only used to convert frame counts to seconds.
GAP_MIN_M = 0.3  # min human-robot dist (m) during Relocate; below = an actual clip/穿模
#                  (was 0.6 = graze; relaxed — a 0.3-0.6m near-pass neither knocks the
#                  collision proxy nor corrupts replay state, so it is NOT a data error).
RELOC_MIN_WALK_M = 0.5  # min human displacement (m) during Relocate; below = the
#                         relocation never happened -> no follow segment to learn from.
RELOC_LOST_MIN_S = 0.5  # `lost_human` needs the gap to STAY over RELOC_MAX_GAP_M this long.
#                         A peak alone is the wrong test: across 16 measured episodes the
#                         gap exceeded 2.0m only as a 2-frame (0.07s) transient on the
#                         longest walk, while a genuinely jammed robot sat far behind for
#                         seconds. Peak-only falsely failed a clean 5.7m follow.
RELOC_MAX_GAP_M = 2.0   # max human-robot dist during Relocate; above = the robot fell
#                         far behind / jammed and stopped following (ep0002 4.41m,
#                         ep0003 2.47m — both judged unusable on video review).
RELOC_MAX_SHORT_M = 0.6  # max |human_end - planned_dest| at the end of Relocate. The
#                          human halts whenever the robot is within hard_collision_r; if
#                          the robot cannot clear the lane the follow loop runs out of
#                          frames and the human is stranded mid-path. `human_walk` cannot
#                          see this (a 1.27m-of-4.21m walk still clears RELOC_MIN_WALK_M),
#                          so compare against the endpoint the searcher actually chose.
#                          Measured: reached 0.11-0.24m, stranded 2.61-4.88m. No overlap.
ROBOT_MAX_TILT_DEG = 20.0  # (SUPERSEDED by ROBOT_TIP_DEG, which checks ALL moving phases,
#                            not just Relocate — see the full-phase QC block below.) Kept for
#                            reference: a robot tipped past this (ep0004 180°, ep0005 22°) is
#                            lying on the floor, making every later obs/action meaningless.
#                            `robot_tilt_deg` (Relocate tilt) is still reported for continuity.
SPAWN_MIN_CLEAR_M = 0.30  # min distance from the human's SPAWN to the nearest obstacle.
#                         Measured on the 8-ep batch: every BAD episode spawned <=0.24m
#                         from furniture/stairs (ep0005 0.03m -> feet clipped the stove);
#                         good ones 0.61-1.19m. The spawn logic now prefers >=0.50m, so
#                         this is a backstop for episodes that fell to the relaxed tier.

# ---------------------------------------------------------------------------
# Full-phase safety QC (2026-07-16, docs/DIAGNOSIS_hrc_batch_final_64ep.md §5/§7).
#
# The Relocate-only checks above (min_gap / max_gap / robot_tilt / short_by) are
# computed on Relocate frames ONLY (`m = sid == reloc_idx`). That blind spot
# passed ep0031/0035/0052/0057/0064 — every one of which fell over, drove into a
# wall, or walked into the human DURING ReturnToHuman / FaceHuman, phases the old
# QC never inspected. The `_safety_qc` scan below covers ALL moving phases
# (GoToTask / ReturnToHuman / FaceHuman / Relocate) using only recorded obs + the
# static maps (no simulation). Thresholds calibrated on the 64-ep review (§7):
# CLEAN episodes top out ~8° tilt / >0.4m clearance, bad ones 13-145° / <0.2m.
_MOVING_PHASES = ("GoToTask", "ReturnToHuman", "FaceHuman", "Relocate")
_APPROACH_PHASES = ("ReturnToHuman", "FaceHuman")  # the robot walks UP TO the human here
ROBOT_TIP_DEG = 15.0     # max |roll|/|pitch| in ANY moving phase; > this = tipped / fell.
WALL_CLR_M = 0.20        # robot-centre clearance to a WALL/STAIR (no_obj map) in any moving
#                          phase; < this = drove into a wall pocket (ep0031/0064: 0.20m + 21°).
HUMAN_HIT_M = 0.55       # min robot-human distance during ReturnToHuman/FaceHuman; < this =
#                          walked into the human (ep0035 0.40m / ep0052 0.46m). The reach gate
#                          stops at ≥1.0m, so a sustained <0.55m is a genuine over-approach.
HUMAN_HIT_MIN_S = 0.2    # ... it must STAY under HUMAN_HIT_M this long (reject contact, not a
#                          1-frame graze). At 30fps = 6 frames.
DRAG_MAX_MS = 1.5        # max base speed (m/s) in any phase; > this = a fallen robot dragged by
#                          the nav primitive / physics blow-up (ep0028 6.7 m/s post-fall).
# RC3 spin is measured as IN-PLACE rotation (turning while NOT translating), not total
# rotation: after the 2026-07-17 fix GoToTask turns WHILE walking (acceptable), and only a
# single shortest in-place turn to the replay yaw remains (user-allowed). So we allow ONE big
# in-place turn (the largest contiguous in-place run) and flag the REST. goto_spin (-> TRIM,
# segment-level, not REJECT) if a single in-place turn winds past a half-circle, OR there is
# extra in-place rotation beyond that one allowed turn (a stop-pivot / spinning).
GOTO_SPIN_MAXRUN_DEG = 200.0  # a single contiguous in-place turn wider than this = wound long / spun
GOTO_SPIN_EXTRA_DEG = 90.0    # in-place rotation BEYOND the largest single in-place turn = spinning
OBJ_KNOCK_M = 0.08       # a tracked task object that MOVES >this WITHIN a navigation phase
#                          (GoToTask/Return/Face/Relocate) was knocked — the robot is not
#                          manipulating anything then, so any motion is a collision (ep0036
#                          trash can). Needs recorder object tracking (2026-07-16); a no-op
#                          on datasets without obs/object_pos (old batches unaffected).
OBJ_POLLUTE_M = 0.05     # a RECEPTACLE (trash/ashcan/basket/radio) moving >this WITHIN an
#                          ExecuteBehaviorTask replay = replay-state pollution (§12.D red line;
#                          cans legitimately move during pickup so only receptacles are checked).
_OBJ_RECEPTACLE_TOKENS = ("ashcan", "trash", "basket", "bin", "radio", "receptacle")
_OBJ_NAV_PHASES = ("GoToTask", "ReturnToHuman", "FaceHuman", "Relocate")
FURN_GRAZE_M = 0.15      # clearance to a furniture FOOTPRINT (full map, wall far) -> WARN only:
#                          ambiguous (map footprint often exceeds the collision mesh; the robot
#                          does not tip). Recorded for review, never rejects. Needs contact
#                          logging to decide (DIAGNOSIS limitation #3).

# Scene traversability maps (1cm/px). Verified transform 6/6: col=100*x+W/2, row=100*y+H/2,
# no row flip, white(>0)=free.  floor_trav_0 = walls+furniture+stairs (ANY obstacle);
# floor_trav_no_obj_0 = walls+stairs only (furniture removed) -> lets us separate an
# unrecoverable WALL/STAIR hit from a furniture-footprint graze. Scene-specific (this
# task-0001/0000 pipeline shares house_double_floor_lower across both rounds).
_SCENE_LAYOUT = ("/root/autodl-tmp/datasets/behavior-1k-assets/scenes/"
                 "house_double_floor_lower/layout")
_FLOOR_TRAV_PNG = os.path.join(_SCENE_LAYOUT, "floor_trav_0.png")
_NOOBJ_TRAV_PNG = os.path.join(_SCENE_LAYOUT, "floor_trav_no_obj_0.png")
_TRAV_CACHE: Any = None   # (dist_full, dist_noobj_or_None, H, W); metres to nearest obstacle


def _trav_maps():
    """Cache & return (dist_full, dist_noobj, H, W) — distance transforms (m) of the two
    traversability maps. dist_noobj is None if the no_obj map is missing. Returns None on
    any failure (QC then simply skips the map-based checks; it never blocks an episode)."""
    global _TRAV_CACHE
    if _TRAV_CACHE is not None:
        return _TRAV_CACHE if _TRAV_CACHE != "FAIL" else None
    try:
        import numpy as np
        from PIL import Image
        from scipy.ndimage import distance_transform_edt
        full = np.asarray(Image.open(_FLOOR_TRAV_PNG).convert("L"))
        df = distance_transform_edt(full > 0) * 0.01
        H, W = full.shape
        try:
            nob = np.asarray(Image.open(_NOOBJ_TRAV_PNG).convert("L"))
            dn = distance_transform_edt(nob > 0) * 0.01
        except Exception:
            dn = None
        _TRAV_CACHE = (df, dn, H, W)
        return _TRAV_CACHE
    except Exception:
        _TRAV_CACHE = "FAIL"
        return None


def _clr_array(dist, xy):
    """Vectorised clearance (m) at world xy (N,2) on a distance map; NaN off-map."""
    import numpy as np
    if dist is None:
        return None
    H, W = dist.shape
    cols = np.round(100.0 * xy[:, 0] + W / 2.0).astype(int)
    rows = np.round(100.0 * xy[:, 1] + H / 2.0).astype(int)
    out = np.full(len(xy), np.nan)
    ok = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    out[ok] = dist[rows[ok], cols[ok]]
    return out


def _spawn_clearance(spawn_xy) -> Optional[float]:
    """Metres from the human's spawn point to the nearest obstacle (full trav map).
    Returns None when uncomputable (never blocks an episode)."""
    m = _trav_maps()
    if m is None:
        return None
    df, _dn, H, W = m
    r = int(round(100.0 * float(spawn_xy[1]) + H / 2.0))
    c = int(round(100.0 * float(spawn_xy[0]) + W / 2.0))
    if 0 <= r < H and 0 <= c < W:
        return float(df[r, c])
    return None


def _reloc_dest(out_dir: str):
    """Planned relocation endpoint, read back from the episode log line
    `[relocation] dest=(x,y) ...`. Parsing the log avoids threading the value through
    the hdf5 writer; the line is printed for every dest_mode (auto/predefined/banded)."""
    ep = os.path.basename(os.path.normpath(out_dir))
    log = os.path.join(os.path.dirname(os.path.normpath(out_dir)), "logs", f"{ep}.log")
    if not os.path.exists(log):
        return None
    dest = None
    try:
        with open(log, errors="ignore") as fh:
            for ln in fh:
                m = re.search(r"\[relocation\] dest=\(([-\d.]+),\s*([-\d.]+)\)", ln)
                if m:
                    dest = (float(m.group(1)), float(m.group(2)))
    except Exception:
        return None
    return dest


def _reloc_qc(out_dir: str) -> Dict[str, Any]:
    """One-pass data-correctness QC over the merged hdf5:
      min_gap    – min human-robot distance during Relocate (< GAP_MIN_M => clip/穿模)
      human_walk – total human displacement during Relocate (< RELOC_MIN_WALK_M => the
                   relocation never happened, so there is no follow segment to learn)
      action_ok  – all recorded actions finite (no NaN/Inf => data integrity)
    None/True when uncomputable (rc/predicate already gate those episodes)."""
    out: Dict[str, Any] = {"min_gap": None, "max_gap": None, "human_walk": None,
                           "spawn_clear": None, "robot_tilt": None, "action_ok": True,
                           "short_by": None, "gap_over_s": 0.0}
    h = os.path.join(out_dir, "hrc_dataset.hdf5")
    if not os.path.exists(h):
        return out
    try:
        import glob
        import h5py
        import numpy as np
    except Exception:
        return out
    reloc_idx = 9  # default fsm-state order; override from meta when present
    try:
        mp = glob.glob(os.path.join(out_dir, "meta", "*.json"))
        if mp:
            st = json.load(open(mp[0])).get("fsm_states")
            if st and "Relocate" in st:
                reloc_idx = st.index("Relocate")
    except Exception:
        pass
    try:
        gmin = float("inf")
        gmax = 0.0
        walk = 0.0
        tilt_max = 0.0
        spawn_xy = None
        reloc_end_xy = None
        with h5py.File(h, "r") as f:
            if "data/demo_0/obs/human_pos" in f:
                spawn_xy = f["data/demo_0/obs/human_pos"][0, :2]
            for demo in ("demo_0", "demo_1"):
                if f"data/{demo}" not in f:
                    continue
                g = f[f"data/{demo}"]
                if "action" in g and not bool(np.all(np.isfinite(g["action"][:]))):
                    out["action_ok"] = False
                rp = g["obs/robot_pos"][:][:, :2]
                hp = g["obs/human_pos"][:][:, :2]
                sid = g["obs/fsm_state_id"][:].astype(int).ravel()
                m = sid == reloc_idx
                if int(m.sum()) > 0:
                    d = np.linalg.norm(rp[m] - hp[m], axis=1)
                    gmin = min(gmin, float(d.min()))
                    # max_gap answers "did the robot fall behind while FOLLOWING?" — so it
                    # must be measured only while the human is actually walking. The
                    # Relocate phase also contains the pre-walk `_yield_robot_off_path`
                    # manoeuvre, in which the robot deliberately drives clear of the
                    # human's planned path; that alone reads 2.64m and falsely tripped
                    # lost_human on a v4 episode whose follow never exceeded 2.05m.
                    hp_r = hp[m]
                    stepd = np.linalg.norm(np.diff(hp_r, axis=0), axis=1)
                    walk += float(np.sum(stepd))
                    mv = np.nonzero(stepd > 1e-3)[0]
                    if mv.size >= 5:                       # a real follow segment exists
                        lo, hi = int(mv.min()), int(mv.max()) + 1
                        dfollow = d[lo:hi + 1]
                        gmax = max(gmax, float(dfollow.max()))
                        over = int(np.sum(dfollow > RELOC_MAX_GAP_M))
                        out["gap_over_s"] = max(out["gap_over_s"], over / float(FPS))
                    # else: the human never walked -> no_reloc / reloc_incomplete gate it
                    # Robot knocked over by the kinematic human walking through it.
                    q = g["obs/robot_quat"][:][m]
                    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                    roll = np.degrees(np.arctan2(2 * (qw * qx + qy * qz),
                                                 1 - 2 * (qx * qx + qy * qy)))
                    pitch = np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)))
                    tilt_max = max(tilt_max,
                                   float(np.max(np.maximum(np.abs(roll), np.abs(pitch)))))
                    reloc_end_xy = hp[m][-1]
        dest = _reloc_dest(out_dir)
        if dest is not None and reloc_end_xy is not None:
            out["short_by"] = float(np.linalg.norm(np.asarray(reloc_end_xy, dtype=float)
                                                   - np.asarray(dest, dtype=float)))
        out["min_gap"] = gmin if gmin < float("inf") else None
        out["max_gap"] = gmax if gmax > 0 else None
        out["human_walk"] = walk
        out["robot_tilt"] = tilt_max if tilt_max > 0 else None
        if spawn_xy is not None:
            out["spawn_clear"] = _spawn_clearance(spawn_xy)
    except Exception:
        pass
    return out


def _safety_qc(out_dir: str) -> Dict[str, Any]:
    """Full-phase safety scan over the merged hdf5 (DIAGNOSIS §5/§7). Unlike
    `_reloc_qc` (Relocate frames only), this covers GoToTask / ReturnToHuman /
    FaceHuman too, so falls / wall-hits / human-contact in the return-and-face
    phases are no longer invisible. Reads only recorded obs + the static trav
    maps (no simulation). All fields None/0 when uncomputable (predicate + rc still
    gate those episodes). Fields:
      tilt_max   – max |roll|/|pitch| (deg) over ALL moving phases
      wall_clr   – min robot clearance to a WALL/STAIR (no_obj map)
      furn_clr   – min robot clearance to a furniture footprint (wall far) [warn]
      human_hit  – min robot-human distance during Return/Face, IF sustained
      spin_*     – round-1 GoToTask total / net / excess heading rotation (deg)
      drag_max   – max base speed (m/s) over all phases
    Each *_phase field tags where the extreme occurred (e.g. "FaceHuman#r1")."""
    out: Dict[str, Any] = {"tilt_max": 0.0, "tilt_phase": None,
                           "wall_clr": None, "wall_phase": None, "furn_clr": None,
                           "human_hit": None, "human_hit_phase": None,
                           "spin_inplace": None, "spin_maxrun": None, "spin_extra": None,
                           "drag_max": 0.0, "drag_phase": None}
    h = os.path.join(out_dir, "hrc_dataset.hdf5")
    if not os.path.exists(h):
        return out
    try:
        import glob
        import h5py
        import numpy as np
    except Exception:
        return out
    names = ['WaitNearHuman', 'ReceiveCommand', 'GoToTask', 'ExecuteBehaviorTask',
             'TaskSegmentDone', 'ReturnToHuman', 'FaceHuman', 'WaitNextCommand',
             'AllTasksDone', 'Relocate']
    try:
        mp = glob.glob(os.path.join(out_dir, "meta", "*.json"))
        if mp:
            st = json.load(open(mp[0])).get("fsm_states")
            if st:
                names = st
    except Exception:
        pass
    maps = _trav_maps()
    df, dn = (maps[0], maps[1]) if maps is not None else (None, None)

    def _runs(a):
        segs = []; s = 0
        for i in range(1, len(a) + 1):
            if i == len(a) or a[i] != a[s]:
                segs.append((int(a[s]), s, i)); s = i
        return segs

    def _nm(v):
        return names[v] if 0 <= v < len(names) else str(v)

    hit_min = max(int(HUMAN_HIT_MIN_S * FPS), 1)
    try:
        with h5py.File(h, "r") as f:
            for demo in ("demo_0", "demo_1"):
                if f"data/{demo}" not in f:
                    continue
                g = f[f"data/{demo}"]
                rp = g["obs/robot_pos"][:][:, :2]
                hp = g["obs/human_pos"][:][:, :2]
                q = g["obs/robot_quat"][:]
                sid = g["obs/fsm_state_id"][:].astype(int).ravel()
                rid = (g["round_id"][:].astype(int).ravel()
                       if "round_id" in g else np.zeros(len(sid), int))
                qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                roll = np.degrees(np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))
                pitch = np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1)))
                yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                tilt = np.maximum(np.abs(roll), np.abs(pitch))
                gap = np.linalg.norm(rp - hp, axis=1)
                vel = np.concatenate([[0.0], np.linalg.norm(np.diff(rp, axis=0), axis=1)]) * FPS
                dyaw = np.abs((np.diff(yaw) + np.pi) % (2 * np.pi) - np.pi)
                cf = _clr_array(df, rp)  # clearance to any obstacle (furniture incl.)
                cw = _clr_array(dn, rp)  # clearance to walls/stairs only
                for v, s, e in _runs(sid):
                    ph = _nm(v)
                    if ph not in _MOVING_PHASES or (e - s) < 8:
                        continue
                    tag = f"{ph}#r{int(rid[s])}"
                    tmax = float(tilt[s:e].max())
                    if tmax > out["tilt_max"]:
                        out["tilt_max"], out["tilt_phase"] = tmax, tag
                    vmax = float(vel[s:e].max())
                    if vmax > out["drag_max"]:
                        out["drag_max"], out["drag_phase"] = vmax, tag
                    if cw is not None:
                        segw = cw[s:e]
                        if np.isfinite(segw).any():
                            wmin = float(np.nanmin(segw))
                            if wmin < WALL_CLR_M and (out["wall_clr"] is None or wmin < out["wall_clr"]):
                                out["wall_clr"], out["wall_phase"] = wmin, tag
                    if cf is not None and cw is not None:
                        segf, segw = cf[s:e], cw[s:e]
                        m2 = np.isfinite(segf) & np.isfinite(segw) & (segf < FURN_GRAZE_M) & (segw > 0.5)
                        if m2.any():
                            fmin = float(np.nanmin(segf[m2]))
                            if out["furn_clr"] is None or fmin < out["furn_clr"]:
                                out["furn_clr"] = fmin
                    if ph in _APPROACH_PHASES:
                        seg_gap = gap[s:e]
                        if int(np.sum(seg_gap < HUMAN_HIT_M)) >= hit_min:
                            gmin = float(seg_gap.min())
                            if out["human_hit"] is None or gmin < out["human_hit"]:
                                out["human_hit"], out["human_hit_phase"] = gmin, tag
                    if ph == "GoToTask" and int(rid[s]) == 1 and (e - s) > 2:
                        seg_dyaw = dyaw[s:e - 1]                  # per-step |dyaw| (rad)
                        seg_speed = vel[s + 1:e]                  # m/s aligned to the dyaw steps
                        inpl = seg_speed < 0.05                   # rotating ~in place
                        total_ip = float(np.degrees(seg_dyaw[inpl].sum())) if inpl.any() else 0.0
                        maxrun = 0.0; run = 0.0                   # largest contiguous in-place run (deg)
                        for k in range(len(inpl)):
                            if inpl[k]:
                                run += float(np.degrees(seg_dyaw[k]))
                            else:
                                maxrun = max(maxrun, run); run = 0.0
                        maxrun = max(maxrun, run)
                        extra = total_ip - maxrun
                        if out["spin_extra"] is None or extra > out["spin_extra"]:
                            out["spin_inplace"] = total_ip
                            out["spin_maxrun"] = maxrun
                            out["spin_extra"] = extra
    except Exception:
        pass
    return out


def _object_qc(out_dir: str) -> Dict[str, Any]:
    """Task-object pose scan over the merged hdf5 (2026-07-16). Needs the recorder's
    obs/object_pos (N,K,3) + data attr `object_names` — a NO-OP on datasets that lack
    them (old batches unaffected). Flags:
      object_knocked   – a tracked object moved >OBJ_KNOCK_M WITHIN a navigation phase
                         (robot isn't manipulating -> a collision knocked it; ep0036).
      replay_pollution – a RECEPTACLE moved >OBJ_POLLUTE_M WITHIN ExecuteBehaviorTask
                         (§12.D data-integrity red line; cans move legitimately so only
                         receptacles are checked). Displacement is intra-phase (from the
                         phase's first frame), so a can that legitimately came to rest
                         displaced from origin in an earlier phase is NOT re-flagged."""
    out: Dict[str, Any] = {"object_knocked": None, "obj_knock_name": None, "obj_knock_phase": None,
                           "replay_pollution": None, "obj_pollute_name": None}
    h = os.path.join(out_dir, "hrc_dataset.hdf5")
    if not os.path.exists(h):
        return out
    try:
        import glob
        import h5py
        import numpy as np
    except Exception:
        return out
    names = ['WaitNearHuman', 'ReceiveCommand', 'GoToTask', 'ExecuteBehaviorTask',
             'TaskSegmentDone', 'ReturnToHuman', 'FaceHuman', 'WaitNextCommand',
             'AllTasksDone', 'Relocate']
    try:
        mp = glob.glob(os.path.join(out_dir, "meta", "*.json"))
        if mp:
            st = json.load(open(mp[0])).get("fsm_states")
            if st:
                names = st
    except Exception:
        pass

    def _runs(a):
        segs = []; s = 0
        for i in range(1, len(a) + 1):
            if i == len(a) or a[i] != a[s]:
                segs.append((int(a[s]), s, i)); s = i
        return segs

    def _nm(v):
        return names[v] if 0 <= v < len(names) else str(v)
    try:
        with h5py.File(h, "r") as f:
            if "data" not in f or "object_names" not in f["data"].attrs:
                return out                                   # dataset predates object tracking
            try:
                obj_names = json.loads(f["data"].attrs["object_names"])
            except Exception:
                obj_names = None
            if not obj_names:
                return out
            for demo in ("demo_0", "demo_1"):
                g = f.get(f"data/{demo}")
                if g is None or "obs/object_pos" not in g:
                    continue
                op = g["obs/object_pos"][:]                  # (N, K, 3)
                sid = g["obs/fsm_state_id"][:].astype(int).ravel()
                if op.ndim != 3 or op.shape[0] != len(sid):
                    continue
                for v, s, e in _runs(sid):
                    ph = _nm(v)
                    if (e - s) < 8:
                        continue
                    for k, onm in enumerate(obj_names):
                        if k >= op.shape[1]:
                            break
                        pos = op[s:e, k, :]                  # (phase_len, 3)
                        dmax = float(np.linalg.norm(pos - pos[0], axis=1).max())
                        if ph in _OBJ_NAV_PHASES and dmax > OBJ_KNOCK_M:
                            if out["object_knocked"] is None or dmax > out["object_knocked"]:
                                out["object_knocked"] = dmax
                                out["obj_knock_name"] = str(onm)
                                out["obj_knock_phase"] = f"{ph}#r{int(demo[-1])}"
                        if (ph == "ExecuteBehaviorTask" and dmax > OBJ_POLLUTE_M
                                and any(t in str(onm).lower() for t in _OBJ_RECEPTACLE_TOKENS)):
                            if out["replay_pollution"] is None or dmax > out["replay_pollution"]:
                                out["replay_pollution"] = dmax
                                out["obj_pollute_name"] = str(onm)
    except Exception:
        pass
    return out


def _classify(rs: Dict[str, Any], qc: Dict[str, Any], sq: Dict[str, Any],
              rc: int, timed_out: bool, oq: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The three-tier success standard (docs/DIAGNOSIS_hrc_batch_final_64ep.md §7).
    Pure function of parsed round success + the two QC scans + run status, so it is
    testable and reusable (batch _finish and any offline re-scan call it identically).

      REJECT — any HARD failure below: the whole episode is unusable, discard it.
      TRIM   — predicate-OK, no hard failure, but one SEGMENT is bad (round-1 GoToTask
               wound-long spin). Usable after cutting that segment.
      CLEAN  — predicate-OK and nothing flagged: directly trainable as-is.

    Hard failures fall in four groups:
      (1) task semantics + run    : r0, r1 predicate, rc, timeout, action_nan
      (2) Relocate data-correctness: clip, no_reloc, lost_human, reloc_incomplete, spawn_tight
      (3) full-phase safety (NEW)  : robot_fell(any phase), wall_hit, hit_human, robot_drag
    goto_spin is a TRIM; furn_graze is a WARN (recorded, never blocks)."""
    gap, walk, short_by = qc["min_gap"], qc["human_walk"], qc["short_by"]
    sclr = qc["spawn_clear"]
    oq = oq or {}
    flags = {
        # (1)
        "action_nan": not qc["action_ok"],
        # (2) Relocate-phase (existing loose philosophy: only true errors)
        "clip": gap is not None and gap < GAP_MIN_M,
        "no_reloc": walk is not None and walk < RELOC_MIN_WALK_M,
        "lost_human": qc["gap_over_s"] >= RELOC_LOST_MIN_S,
        "reloc_incomplete": short_by is not None and short_by > RELOC_MAX_SHORT_M,
        "spawn_tight": sclr is not None and sclr < SPAWN_MIN_CLEAR_M,
        # (3) full-phase safety (new: GoToTask/ReturnToHuman/FaceHuman covered)
        "robot_fell": sq["tilt_max"] > ROBOT_TIP_DEG,
        "wall_hit": sq["wall_clr"] is not None and sq["wall_clr"] < WALL_CLR_M,
        "hit_human": sq["human_hit"] is not None,          # already gated by HUMAN_HIT_MIN_S
        "robot_drag": sq["drag_max"] > DRAG_MAX_MS,
        # (3b) task-object integrity (no-op when object tracking is absent)
        "object_knocked": oq.get("object_knocked") is not None,
        "replay_pollution": oq.get("replay_pollution") is not None,
        # (4) segment-level / warning
        "goto_spin": sq["spin_maxrun"] is not None and (
            sq["spin_maxrun"] > GOTO_SPIN_MAXRUN_DEG or sq["spin_extra"] > GOTO_SPIN_EXTRA_DEG),
        "furn_graze": sq["furn_clr"] is not None,
    }
    pred_ok = bool(rs["round0"]) and bool(rs["round1"])
    # Object-displacement checks are WARN-ONLY (record for review, never auto-reject):
    # a task object moving is ambiguous without a source-trajectory comparison (TODO).
    # Three false-positive modes were observed on ep0063: (a) replay_pollution — Execute is
    # FAITHFUL REPLAY, so a receptacle moving there is the demonstrated behaviour (the source
    # relocates trash_can_116 by 3.3m); (b) object_knocked — physics settling/drift of the
    # round-0 cans+bin by 0.17-0.23m during round-1 GoToTask (robot is far, navigating to the
    # radio); (c) sources that legitimately relocate the bin. A real knock (ep0036) is large,
    # sudden and robot-adjacent — separating it needs robot-proximity + velocity gating or a
    # source diff, so for now both only warn.
    hard_keys = ("clip", "no_reloc", "action_nan", "lost_human", "reloc_incomplete",
                 "spawn_tight", "robot_fell", "wall_hit", "hit_human", "robot_drag")
    hard = ([k for k, v in (("rc", rc != 0), ("timeout", timed_out),
                            ("r0", not rs["round0"]), ("r1", not rs["round1"])) if v]
            + [k for k in hard_keys if flags[k]])
    trim = [k for k in ("goto_spin",) if flags[k]]
    warn = [k for k in ("furn_graze", "object_knocked", "replay_pollution") if flags[k]]
    if hard or not pred_ok or rc != 0 or timed_out:
        verdict = "REJECT"
    elif trim:
        verdict = "TRIM"
    else:
        verdict = "CLEAN"
    return {"verdict": verdict, "passed": verdict == "CLEAN",
            "usable_after_trim": verdict in ("CLEAN", "TRIM"),
            "hard": hard, "trim": trim, "warn": warn, **flags}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="pairing manifest from prescreen_sources.py")
    ap.add_argument("--config", default=os.path.join(_PROJECT, "configs", "hrc_multitask_0001.yaml"))
    ap.add_argument("--out-root", default=os.path.join(_PROJECT, "outputs", "hrc_batch_0001"))
    ap.add_argument("--start", type=int, default=0, help="first manifest index (inclusive)")
    ap.add_argument("--count", type=int, default=None, help="how many to run from --start (default: all remaining)")
    ap.add_argument("--timeout-min", type=float, default=90.0, help="per-episode hard timeout")
    ap.add_argument("--parallel", type=int, default=1,
                    help="concurrent episodes, each its own Isaac sim process (~12GB VRAM each; 2 fits a 24GB GPU)")
    ap.add_argument("--stagger-sec", type=float, default=20.0,
                    help="delay between launching concurrent episodes, to offset sim-boot/cache contention")
    ap.add_argument("--resume", action="store_true", help="skip episodes that already have a commands.json")
    ap.add_argument("--online-nav", dest="online_nav", action="store_true", default=True)
    ap.add_argument("--no-online-nav", dest="online_nav", action="store_false")
    ap.add_argument("--dry-run", action="store_true", help="print the per-episode commands and exit")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    episodes: List[Dict[str, Any]] = manifest["episodes"]
    end = len(episodes) if args.count is None else min(args.start + args.count, len(episodes))
    todo = episodes[args.start:end]

    os.makedirs(args.out_root, exist_ok=True)
    log_dir = os.path.join(args.out_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    batch_manifest = os.path.join(args.out_root, "batch_manifest.json")
    done = _load_manifest_done(batch_manifest)

    env = dict(os.environ)
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    env.setdefault("OMP_NUM_THREADS", "8")

    print(f"[batch] {len(todo)} episode(s) [{args.start}:{end}) -> {args.out_root} "
          f"(resume={args.resume}, timeout={args.timeout_min}min)", flush=True)

    # Resolve the pending list up front (apply resume-skip / dry-run here), so the
    # pool loop below only deals with episodes that actually need running.
    pending = []
    for ep in todo:
        idx = ep["idx"]
        out_dir = os.path.join(args.out_root, f"ep_{idx:04d}")
        cj = os.path.join(out_dir, "annotations", "commands.json")
        cmd = [sys.executable, _DEMO,
               "--config", args.config,
               "--raw-hdf5", ep["trash_hdf5"],
               "--external-hdf5", f"1:{ep['radio_hdf5']}",
               "--output-dir", out_dir]
        if args.online_nav:
            cmd.append("--online-nav")
        if args.dry_run:
            print(f"[batch] ep{idx:04d}: {' '.join(cmd)}", flush=True)
            continue
        if args.resume and os.path.exists(cj):
            rs = _parse_round_success(out_dir)
            print(f"[batch] ep{idx:04d}: SKIP (resume; r0={rs['round0']} r1={rs['round1']})", flush=True)
            done[idx] = {**done.get(idx, {}), "idx": idx, **rs, "skipped_resume": True}
            continue
        pending.append((ep, out_dir, cmd))
    if args.dry_run:
        return

    def _finish(ep, out_dir, rc, timed_out, dt):
        rs = _parse_round_success(out_dir)
        qc = _reloc_qc(out_dir)
        gap, walk = qc["min_gap"], qc["human_walk"]
        gmax, sclr, tilt = qc["max_gap"], qc["spawn_clear"], qc["robot_tilt"]
        short_by = qc["short_by"]
        sq = _safety_qc(out_dir)
        oq = _object_qc(out_dir)
        # Three-tier success standard (REJECT / TRIM / CLEAN). See _classify docstring.
        cl = _classify(rs, qc, sq, rc, timed_out, oq=oq)
        verdict, passed = cl["verdict"], cl["passed"]
        usable_after_trim = cl["usable_after_trim"]
        fails, trim, warn = cl["hard"], cl["trim"], cl["warn"]
        clipped, no_reloc, action_bad = cl["clip"], cl["no_reloc"], cl["action_nan"]
        lost_human, reloc_incomplete, spawn_tight = cl["lost_human"], cl["reloc_incomplete"], cl["spawn_tight"]
        robot_fell, wall_hit, hit_human = cl["robot_fell"], cl["wall_hit"], cl["hit_human"]
        robot_drag, goto_spin = cl["robot_drag"], cl["goto_spin"]
        object_knocked, replay_pollution = cl["object_knocked"], cl["replay_pollution"]

        done[ep["idx"]] = {"idx": ep["idx"], "trash_hdf5": ep["trash_hdf5"], "radio_hdf5": ep["radio_hdf5"],
                           "rc": rc, "timed_out": timed_out, "duration_s": round(dt, 1),
                           "round0": rs["round0"], "round1": rs["round1"],
                           # --- Relocate-phase metrics (existing) ---
                           "reloc_min_gap": (round(gap, 3) if gap is not None else None),
                           "reloc_max_gap": (round(gmax, 3) if gmax is not None else None),
                           "reloc_human_walk": (round(walk, 3) if walk is not None else None),
                           "spawn_clear": (round(sclr, 3) if sclr is not None else None),
                           "robot_tilt_deg": (round(tilt, 1) if tilt is not None else None),
                           "reloc_gap_over_s": round(qc["gap_over_s"], 2),
                           "reloc_short_by": (round(short_by, 3) if short_by is not None else None),
                           # --- full-phase safety metrics (new) ---
                           "tilt_max_all_deg": round(sq["tilt_max"], 1), "tilt_phase": sq["tilt_phase"],
                           "wall_clr": (round(sq["wall_clr"], 3) if sq["wall_clr"] is not None else None),
                           "wall_phase": sq["wall_phase"],
                           "human_hit_gap": (round(sq["human_hit"], 3) if sq["human_hit"] is not None else None),
                           "human_hit_phase": sq["human_hit_phase"],
                           "goto_spin_inplace_deg": (round(sq["spin_inplace"], 0) if sq["spin_inplace"] is not None else None),
                           "goto_spin_maxrun_deg": (round(sq["spin_maxrun"], 0) if sq["spin_maxrun"] is not None else None),
                           "goto_spin_extra_deg": (round(sq["spin_extra"], 0) if sq["spin_extra"] is not None else None),
                           "drag_max_ms": round(sq["drag_max"], 2),
                           "furn_clr": (round(sq["furn_clr"], 3) if sq["furn_clr"] is not None else None),
                           # --- task-object metrics (new; null on datasets without tracking) ---
                           "obj_knock_m": (round(oq["object_knocked"], 3) if oq["object_knocked"] is not None else None),
                           "obj_knock_name": oq["obj_knock_name"], "obj_knock_phase": oq["obj_knock_phase"],
                           "obj_pollute_m": (round(oq["replay_pollution"], 3) if oq["replay_pollution"] is not None else None),
                           "obj_pollute_name": oq["obj_pollute_name"],
                           # --- flags ---
                           "action_ok": qc["action_ok"], "clipped": clipped, "no_reloc": no_reloc,
                           "lost_human": lost_human, "spawn_tight": spawn_tight, "robot_fell": robot_fell,
                           "wall_hit": wall_hit, "hit_human": hit_human, "robot_drag": robot_drag,
                           "object_knocked": object_knocked, "replay_pollution": replay_pollution,
                           "reloc_incomplete": reloc_incomplete, "goto_spin": goto_spin,
                           # --- verdict ---
                           "verdict": verdict, "passed": passed,
                           "usable_after_trim": usable_after_trim,
                           "fail_reasons": fails, "trim_reasons": trim, "warn_reasons": warn,
                           "detail": rs["detail"], "out_dir": out_dir}
        # Persist after every episode so a crash/interrupt loses nothing.
        all_recs = [done[k] for k in sorted(done.keys())]
        n_clean = sum(1 for r in all_recs if r.get("verdict") == "CLEAN")
        n_trim = sum(1 for r in all_recs if r.get("verdict") == "TRIM")
        n_pass = n_clean  # backward-compat alias
        json.dump({"manifest": args.manifest, "episodes": all_recs,
                   "n_run": len(all_recs), "n_pass": n_clean,
                   "n_clean": n_clean, "n_trim": n_trim,
                   "n_usable_after_trim": n_clean + n_trim},
                  open(batch_manifest, "w"), indent=2)
        gtxt = f"{gap:.2f}m" if gap is not None else "n/a"
        ttxt = f"{sq['tilt_max']:.0f}°" if sq["tilt_max"] else "n/a"
        wtxt = f"{sq['wall_clr']:.2f}m" if sq["wall_clr"] is not None else "-"
        htxt = f"{sq['human_hit']:.2f}m" if sq["human_hit"] is not None else "-"
        sptxt = (f"{sq['spin_maxrun']:.0f}/{sq['spin_extra']:.0f}°"
                 if sq["spin_maxrun"] is not None else "-")
        allflags = fails + trim + warn
        print(f"[batch] ep{ep['idx']:04d}: {verdict} "
              f"(rc={rc} r0={rs['round0']} r1={rs['round1']} gap={gtxt} tilt={ttxt} "
              f"wall={wtxt} humanhit={htxt} spin_run/extra={sptxt}"
              f"{' [' + ','.join(allflags) + ']' if allflags else ''} "
              f"{dt / 60:.1f}min) | clean {n_clean} +trim {n_trim} = {n_clean + n_trim}/{len(all_recs)}",
              flush=True)

    # Concurrent pool: up to --parallel episodes run at once, each its own Isaac
    # sim process (Isaac is a singleton PER PROCESS, so separate processes are
    # fine; the bottleneck is GPU VRAM). Single-threaded poll loop -> manifest
    # writes stay serialized. Launches are staggered to avoid two sim boots
    # hammering the kit asset-cache / GPU init at the same instant.
    par = max(int(args.parallel), 1)
    print(f"[batch] running {len(pending)} episode(s), parallel={par}, "
          f"stagger={args.stagger_sec}s, timeout={args.timeout_min}min", flush=True)
    running: Dict[int, Dict[str, Any]] = {}   # idx -> popen/lf/ep/out_dir/t0
    while pending or running:
        while pending and len(running) < par:
            ep, out_dir, cmd = pending.pop(0)
            idx = ep["idx"]
            log_path = os.path.join(log_dir, f"ep_{idx:04d}.log")
            print(f"[batch] ep{idx:04d}: START trash={os.path.basename(ep['trash_hdf5'])} "
                  f"radio={os.path.basename(ep['radio_hdf5'])} -> {log_path}", flush=True)
            lf = open(log_path, "w")
            p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
            running[idx] = {"popen": p, "lf": lf, "ep": ep, "out_dir": out_dir, "t0": time.time()}
            if pending and len(running) < par:
                time.sleep(max(args.stagger_sec, 0.0))
        finished = []
        for idx, info in running.items():
            rc = info["popen"].poll()
            dt = time.time() - info["t0"]
            timed_out = False
            if rc is None and dt > args.timeout_min * 60:
                info["popen"].kill()
                try:
                    info["popen"].wait(30)
                except Exception:
                    pass
                rc = info["popen"].returncode if info["popen"].returncode is not None else -9
                timed_out = True
            if rc is not None:
                try:
                    info["lf"].close()
                except Exception:
                    pass
                _finish(info["ep"], info["out_dir"], rc, timed_out, dt)
                finished.append(idx)
        for idx in finished:
            del running[idx]
        if running and not finished:
            time.sleep(3)

    all_recs = [done[k] for k in sorted(done.keys())]
    clean = [r for r in all_recs if r.get("verdict") == "CLEAN"]
    trim = [r for r in all_recs if r.get("verdict") == "TRIM"]
    reject = [r for r in all_recs if r.get("verdict") == "REJECT"]
    print(f"\n[batch] DONE. CLEAN {len(clean)} / TRIM {len(trim)} / REJECT {len(reject)} "
          f"of {len(all_recs)} -> usable_after_trim {len(clean) + len(trim)}. "
          f"manifest -> {batch_manifest}", flush=True)
    print(f"[batch] CLEAN (directly trainable, {len(clean)}):", flush=True)
    for r in clean:
        print(f"    {r['out_dir']}", flush=True)
    if trim:
        print(f"[batch] TRIM (usable after cutting round-1 GoToTask spin, {len(trim)}):", flush=True)
        for r in trim:
            print(f"    {r['out_dir']}  (goto_spin_excess={r.get('goto_spin_excess_deg')}°)", flush=True)


if __name__ == "__main__":
    main()
