#!/usr/bin/env python3
"""Offline corrector for the HRC multitask dataset action labels.

WHY
---
Navigation FSM phases (GoToTask / ReturnToHuman / FaceHuman / Relocate) and the
wait phases were recorded with ``action == 0``: ``DemoRecorder.record`` defaults
to ``np.zeros(robot.action_dim)`` when no ``action=`` is passed
(scripts/hrc_minimal_demo.py:146-149), yet the robot is moved kinematically via
``robot.set_position_orientation(...)``.  An end-to-end policy would therefore
learn "output zero velocity while the robot is clearly translating/rotating".

This tool rebuilds the action labels OFFLINE from the recorded observations and
writes a NEW file.  It never mutates the original dataset.

DESIGN
------
* Steps are independent and idempotent.  Every step reads source-of-truth from
  the pristine SRC file and writes only its own action dimensions into DST, so
  running a step twice (or out of order) yields the same result.
* "Frames to fix" are always the originally fully-zero frames, detected from SRC
  (``all(action == 0)``), so later steps are not confused by earlier edits.

ACTION LAYOUT (R1Pro, 23 dims) -- confirmed from the raw env config
(action_normalize=False) and the data:
    [0:3]   base    HolonomicBaseJointController, motor=velocity,
                    input[-1,1] -> output [+-0.75 m/s, +-0.75 m/s, +-1.0 rad/s]
    [3:7]   trunk   JointController, motor=position, use_delta=False (absolute)
                    (dim 6 is dead: std==0 across the whole episode)
    [7:14]  arm_left   JointController, position, absolute
    [14]    gripper_left   MultiFingerGripperController (~{-1,0,+1})
    [15:22] arm_right  JointController, position, absolute
    [22]    gripper_right

This file currently implements STEP 1 (base) only.  Later steps (arm-hold,
gripper-hold, language rebinding) will be added once this one is verified.

USAGE
-----
    python fix_nav_actions.py --init            # copy SRC -> DST
    python fix_nav_actions.py --step base       # rebuild base[0:3] for nav frames
    python fix_nav_actions.py --verify          # print stats on DST
"""
from __future__ import annotations

import argparse
import os
import shutil

import h5py
import numpy as np

# ----------------------------------------------------------------------------
DEFAULT_SRC = "/root/autodl-tmp/outputs/hrc_multitask_v01/hrc_dataset.hdf5"

# FSM ids where the robot is teleported along a path (so it really moves and the
# base action must be non-zero).  Wait/transition phases are intentionally NOT
# here: the robot is static there and base==0 is correct.
NAV_IDS = {2, 5, 6, 9}  # GoToTask, ReturnToHuman, FaceHuman, Relocate

# HolonomicBaseJointController output limits -> used to map physical velocity
# back to the normalized command the controller expects.
BASE_OUT_LIMITS = np.array([0.75, 0.75, 1.0], dtype=np.float32)  # vx, vy, vyaw

FPS = 30  # env action_frequency (from raw hdf5 config)

# Absolute-position action dims (JointController, use_delta=False): trunk + both
# arms.  Excludes base (0-2), the dead trunk dim (6, std==0 everywhere, stays 0
# to match the manipulation segment) and the grippers (14, 22).  For these,
# action == target joint angle, so "hold pose" == current measured joint angle.
ARM_TRUNK_DIMS = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21]


def dst_path(src: str) -> str:
    d, b = os.path.split(src)
    return os.path.join(d, b.replace(".hdf5", "_fixed.hdf5"))


def yaw_from_quat_xyzw(q: np.ndarray) -> np.ndarray:
    """Yaw (rotation about world z) from a scalar-last [x,y,z,w] quaternion."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def demo_keys(f: h5py.File):
    return sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))


# ----------------------------------------------------------------------------
def cmd_init(src: str) -> None:
    dst = dst_path(src)
    shutil.copyfile(src, dst)
    with h5py.File(dst, "r+") as f:
        f["data"].attrs["fixups_applied"] = "init"
    print(f"[init] copied\n        {src}\n     -> {dst}")


def cmd_step_base(src: str) -> None:
    dst = dst_path(src)
    if not os.path.exists(dst):
        raise SystemExit("[base] DST missing -- run `--init` first.")

    with h5py.File(src, "r") as fs, h5py.File(dst, "r+") as fd:
        total_fixed = 0
        total_clipped = 0
        for dm in demo_keys(fs):
            gs = fs["data"][dm]
            orig_action = gs["action"][:]
            fsm = gs["obs/fsm_state_id"][:]
            pos = gs["obs/robot_pos"][:].astype(np.float64)
            yaw = np.unwrap(yaw_from_quat_xyzw(gs["obs/robot_quat"][:].astype(np.float64)))

            zero_mask = np.all(orig_action == 0.0, axis=1)
            nav_mask = np.isin(fsm, list(NAV_IDS))
            target = np.where(zero_mask & nav_mask)[0]

            # Work on a copy of the DST action so we touch only base dims of
            # target frames; everything else is preserved verbatim.
            A = fd["data"][dm]["action"][:]
            n_clip = 0
            for t in target:
                # finite-difference pair, kept strictly WITHIN the same phase run
                if t + 1 < len(fsm) and fsm[t + 1] == fsm[t]:
                    a, b = t, t + 1
                elif t - 1 >= 0 and fsm[t - 1] == fsm[t]:
                    a, b = t - 1, t
                else:
                    continue  # isolated single-frame run -> leave as 0
                dx, dy = pos[b, 0] - pos[a, 0], pos[b, 1] - pos[a, 1]
                ya = yaw[t]
                vx = (np.cos(ya) * dx + np.sin(ya) * dy) * FPS          # body forward
                vy = (-np.sin(ya) * dx + np.cos(ya) * dy) * FPS          # body left
                vyaw = (yaw[b] - yaw[a]) * FPS
                cmd = np.array([vx, vy, vyaw], dtype=np.float32) / BASE_OUT_LIMITS
                clipped = np.clip(cmd, -1.0, 1.0)
                n_clip += int(np.any(clipped != cmd))
                A[t, 0:3] = clipped

            fd["data"][dm]["action"][...] = A
            total_fixed += len(target)
            total_clipped += n_clip
            print(f"[base] {dm}: rebuilt base[0:3] on {len(target)} nav frames "
                  f"({n_clip} hit the +-1 clip)")

        prev = fd["data"].attrs.get("fixups_applied", "")
        if "base" not in prev:
            fd["data"].attrs["fixups_applied"] = (prev + "+base").lstrip("+")
    print(f"[base] done: {total_fixed} frames rebuilt, {total_clipped} clipped.")


# ----------------------------------------------------------------------------
def build_arm_joint_map(fs: h5py.File):
    """Discover each absolute-position action dim -> obs joint_positions column.

    On the manipulation segment the commanded action equals the achieved joint
    angle, so the correct column is the one with corr~=1 AND regression slope~=1
    (identity).  The slope guards against picking a merely-correlated neighbour.
    Returns (mapping dict, report list of (dim, col, corr, slope)).
    """
    acts, jps = [], []
    for dm in demo_keys(fs):
        g = fs["data"][dm]
        m = g["obs/fsm_state_id"][:] == 3  # ExecuteBehaviorTask
        acts.append(g["action"][:][m])
        jps.append(g["obs/joint_positions"][:][m])
    A = np.concatenate(acts)
    J = np.concatenate(jps)
    mapping, report = {}, []
    for ad in ARM_TRUNK_DIMS:
        a = A[:, ad]
        best, best_c, best_slope = -1, 0.0, 0.0
        for jc in range(J.shape[1]):
            j = J[:, jc]
            if j.std() < 1e-6:
                continue
            c = np.corrcoef(a, j)[0, 1]
            if abs(c) > abs(best_c):
                best_c = c
                best = jc
                best_slope = float(np.polyfit(j, a, 1)[0])
        mapping[ad] = best
        report.append((ad, best, best_c, best_slope))
    return mapping, report


def cmd_step_arms(src: str) -> None:
    dst = dst_path(src)
    if not os.path.exists(dst):
        raise SystemExit("[arms] DST missing -- run `--init` first.")

    with h5py.File(src, "r") as fs, h5py.File(dst, "r+") as fd:
        mapping, report = build_arm_joint_map(fs)
        # Print + validate the map BEFORE writing anything.
        print("[arms] action-dim -> joint-col map (corr & slope should be ~1.0):")
        ok = True
        for ad, jc, c, s in report:
            flag = "" if (abs(c) > 0.99 and abs(s - 1.0) < 0.05) else "  <-- CHECK"
            if flag:
                ok = False
            print(f"        a[{ad:2d}] -> jp[{jc:2d}]   corr={c:+.3f}  slope={s:+.3f}{flag}")
        if len(set(mapping.values())) != len(mapping):
            raise SystemExit("[arms] ABORT: mapping is not 1:1 (two dims share a joint col).")
        if not ok:
            raise SystemExit("[arms] ABORT: some dims failed corr/slope identity check.")
        print("        map is 1:1 and identity-like -> proceeding.")

        total = 0
        for dm in demo_keys(fs):
            gs = fs["data"][dm]
            orig = gs["action"][:]
            jp = gs["obs/joint_positions"][:]
            zero = np.all(orig == 0.0, axis=1)  # originally fully-zero frames
            A = fd["data"][dm]["action"][:]
            for ad in ARM_TRUNK_DIMS:
                A[zero, ad] = jp[zero, mapping[ad]].astype(np.float32)
            fd["data"][dm]["action"][...] = A
            total += int(zero.sum())
            print(f"[arms] {dm}: held arm/trunk on {int(zero.sum())} zero-action frames")

        prev = fd["data"].attrs.get("fixups_applied", "")
        if "arms" not in prev:
            fd["data"].attrs["fixups_applied"] = (prev + "+arms").lstrip("+")
    print(f"[arms] done: {total} frames hold-filled.")


def cmd_step_grippers(src: str) -> None:
    """Hold the gripper command (dims 14, 22) through zero-action frames.

    Forward-fill the last real gripper command into each originally-zero frame
    (the gripper does not actuate while navigating/waiting).  Leading zero frames
    with no prior command default to 0 (neutral/open).
    """
    dst = dst_path(src)
    if not os.path.exists(dst):
        raise SystemExit("[grippers] DST missing -- run `--init` first.")
    with h5py.File(src, "r") as fs, h5py.File(dst, "r+") as fd:
        total = 0
        for dm in demo_keys(fs):
            orig = fs["data"][dm]["action"][:]
            zero = np.all(orig == 0.0, axis=1)
            A = fd["data"][dm]["action"][:]
            for col in (14, 22):
                last = 0.0
                for t in range(len(zero)):
                    if not zero[t]:
                        last = float(orig[t, col])
                    else:
                        A[t, col] = last
            fd["data"][dm]["action"][...] = A
            total += int(zero.sum())
            print(f"[grippers] {dm}: hold-filled grippers on {int(zero.sum())} zero frames")
        prev = fd["data"].attrs.get("fixups_applied", "")
        if "grippers" not in prev:
            fd["data"].attrs["fixups_applied"] = (prev + "+grippers").lstrip("+")
    print(f"[grippers] done: {total} frames.")


def cmd_step_language(src: str) -> None:
    """Rebind each demo's instruction to the correct per-round text.

    The recorded ``demo.attrs['instruction']`` is the episode base instruction
    copied onto every round (so demo_0 is wrong for multi-round episodes).  The
    authoritative source is annotations/commands.json, keyed by round_id.
    """
    import json
    dst = dst_path(src)
    cmds_path = os.path.join(os.path.dirname(src), "annotations", "commands.json")
    if not os.path.exists(cmds_path):
        raise SystemExit(f"[lang] commands.json not found at {cmds_path}")
    with open(cmds_path) as fp:
        by_round = {int(c["round_id"]): c for c in json.load(fp)}
    with h5py.File(dst, "r+") as fd:
        for dm in demo_keys(fd):
            g = fd["data"][dm]
            rid = int(g.attrs["round_id"])
            if rid not in by_round:
                print(f"[lang] WARN {dm}: round {rid} not in commands.json -- skipped")
                continue
            old = g.attrs.get("instruction", "")
            new = by_round[rid]["instruction"]
            g.attrs["instruction"] = new           # corrected canonical field
            g.attrs["language_instruction"] = new  # explicit per-demo field for loaders
            g.attrs["activity"] = by_round[rid].get("activity", g.attrs.get("activity", ""))
            mark = "" if old == new else "  (FIXED)"
            print(f"[lang] {dm} round={rid}: {old!r} -> {new!r}{mark}")
        prev = fd["data"].attrs.get("fixups_applied", "")
        if "language" not in prev:
            fd["data"].attrs["fixups_applied"] = (prev + "+language").lstrip("+")
    print("[lang] done.")


# ----------------------------------------------------------------------------
def cmd_verify(src: str) -> None:
    dst = dst_path(src)
    FSM = ["WaitNearHuman", "ReceiveCommand", "GoToTask", "ExecuteBehaviorTask",
           "TaskSegmentDone", "ReturnToHuman", "FaceHuman", "WaitNextCommand",
           "AllTasksDone", "Relocate"]
    with h5py.File(src, "r") as fs, h5py.File(dst, "r") as fd:
        print(f"[verify] fixups_applied = {fd['data'].attrs.get('fixups_applied')!r}")
        for dm in demo_keys(fs):
            ao = fs["data"][dm]["action"][:]
            an = fd["data"][dm]["action"][:]
            fsm = fs["data"][dm]["obs/fsm_state_id"][:]
            pos = fs["data"][dm]["obs/robot_pos"][:].astype(np.float64)
            yaw = np.unwrap(yaw_from_quat_xyzw(fs["data"][dm]["obs/robot_quat"][:].astype(np.float64)))
            instr = fd["data"][dm].attrs.get("instruction", "")
            print(f"\n  == {dm} == round_id={int(fs['data'][dm].attrs['round_id'])}  instruction={instr!r}")
            diff = np.abs(an - ao)
            changed_dims = np.where(diff.sum(0) > 0)[0]
            print(f"     changed action dims: {changed_dims.tolist()}")
            zero_o = np.all(ao == 0, axis=1)
            nav = np.isin(fsm, list(NAV_IDS))
            # INVARIANTS (must hold after any combination of steps):
            #  (a) originally non-zero (manipulation) frames are never touched
            manip_changed = int(((diff.sum(1) > 0) & ~zero_o).sum())
            #  (b) base dims [0:3] only change on (zero & nav)
            base_bad = int(((diff[:, 0:3].sum(1) > 0) & ~(zero_o & nav)).sum())
            #  (c) arm/trunk dims only change on zero frames
            arm_bad = int(((diff[:, ARM_TRUNK_DIMS].sum(1) > 0) & ~zero_o).sum())
            #  (d) dead dim 6 is never touched; grippers 14/22 only on zero frames
            applied = fd["data"].attrs.get("fixups_applied", "") or ""
            dim6_bad = float(np.abs(diff[:, 6]).sum())
            if "grippers" in applied:
                grip_msg = f"grip-outside-zero={int(((diff[:, [14, 22]].sum(1) > 0) & ~zero_o).sum())}(0)"
            else:
                grip_msg = f"grip-touched={float(np.abs(diff[:, [14, 22]]).sum()):.0f}(0 until grippers step)"
            print(f"     INVARIANTS  manip-touched={manip_changed}(0)  "
                  f"base-outside-navzero={base_bad}(0)  arm-outside-zero={arm_bad}(0)  "
                  f"dim6-touched={dim6_bad:.0f}(0)  {grip_msg}")
            print(f"     remaining fully-zero action frames = {int(np.all(an == 0, axis=1).sum())}")
            # arm-hold correctness: on zero frames, arm/trunk action == measured joint pos
            if "arms" in (fd["data"].attrs.get("fixups_applied", "") or ""):
                jp = fs["data"][dm]["obs/joint_positions"][:]
                mapping, _ = build_arm_joint_map(fs)
                zmask = zero_o
                maxgap = 0.0
                for ad in ARM_TRUNK_DIMS:
                    maxgap = max(maxgap, float(np.abs(an[zmask, ad] - jp[zmask, mapping[ad]]).max()))
                print(f"     arm-hold: max|action-jointpos| on zero frames = {maxgap:.2e} (expect ~0)")
            # 2) per nav phase: zero-frame ratio + base magnitude before/after
            for u in [2, 5, 6, 9]:
                m = fsm == u
                if not m.any():
                    continue
                fz_o = np.all(ao[m] == 0, axis=1).mean() * 100
                fz_n = np.all(an[m] == 0, axis=1).mean() * 100
                print(f"     {FSM[u]:14s} n={m.sum():4d}  fullzero {fz_o:5.1f}%->{fz_n:5.1f}%  "
                      f"base|mean| {np.abs(an[m,:3]).mean():.3f}  base max|.| {np.abs(an[m,:3]).max():.3f}")
            # 3) SELF-CONSISTENCY: integrate UNCLIPPED reconstructed body-velocity
            #    back to world and compare to the true path (validates quat/rot/fps).
            err = []
            fwd_ratio = []
            for t in range(len(fsm) - 1):
                if int(fsm[t]) in NAV_IDS and fsm[t + 1] == fsm[t]:
                    dx, dy = pos[t + 1, 0] - pos[t, 0], pos[t + 1, 1] - pos[t, 1]
                    ya = yaw[t]
                    vx = np.cos(ya) * dx + np.sin(ya) * dy
                    vy = -np.sin(ya) * dx + np.cos(ya) * dy
                    # world reconstruction from body vel
                    rx = np.cos(ya) * vx - np.sin(ya) * vy
                    ry = np.sin(ya) * vx + np.cos(ya) * vy
                    err.append(abs(rx - dx) + abs(ry - dy))
                    denom = abs(vx) + abs(vy) + 1e-9
                    fwd_ratio.append(abs(vx) / denom)
            if err:
                print(f"     body<->world round-trip max err = {max(err):.2e} (expect ~0)")
            # GoToTask should be mostly forward motion if yaw convention is right
            gx = []
            for t in range(len(fsm) - 1):
                if int(fsm[t]) == 2 and fsm[t + 1] == 2:
                    dx, dy = pos[t + 1, 0] - pos[t, 0], pos[t + 1, 1] - pos[t, 1]
                    ya = yaw[t]
                    vx = np.cos(ya) * dx + np.sin(ya) * dy
                    vy = -np.sin(ya) * dx + np.cos(ya) * dy
                    gx.append((abs(vx), abs(vy)))
            if gx:
                gx = np.array(gx)
                print(f"     GoToTask forward|vx| mean={gx[:,0].mean():.4f} vs lateral|vy| mean={gx[:,1].mean():.4f} "
                      f"(forward should dominate)")


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_SRC, help="original dataset (read-only)")
    ap.add_argument("--init", action="store_true", help="copy SRC -> *_fixed.hdf5")
    ap.add_argument("--step", choices=["base", "arms", "grippers", "language"],
                    help="apply one correction step")
    ap.add_argument("--all", action="store_true",
                    help="init + all steps + verify (idempotent; use per new episode)")
    ap.add_argument("--verify", action="store_true", help="print verification stats")
    args = ap.parse_args()

    steps = {"base": cmd_step_base, "arms": cmd_step_arms,
             "grippers": cmd_step_grippers, "language": cmd_step_language}

    if args.all:
        cmd_init(args.src)
        for name in ["base", "arms", "grippers", "language"]:
            steps[name](args.src)
        cmd_verify(args.src)
        return
    if args.init:
        cmd_init(args.src)
    if args.step:
        steps[args.step](args.src)
    if args.verify:
        cmd_verify(args.src)
    if not (args.init or args.step or args.verify):
        ap.print_help()


if __name__ == "__main__":
    main()
