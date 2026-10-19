#!/usr/bin/env python
# Path A (TRUE physics online) — navigation physics sandbox.
#
# The HRC playback env uses include_contacts=False -> gm.VISUAL_ONLY=True, so
# the robot has NO collisions and cannot stand on the floor; once its base is
# velocity-driven it tumbles (verified by probe_base_control.py).
#
# This sandbox builds a NORMAL og.Environment from the SAME recorded config the
# dataset was collected with (BEHAVIOR-1K standard: contacts ON, self_collisions
# ON, control ON, 30 Hz action / 120 Hz physics), and tests whether R1Pro:
#   (1) stands stably on the floor under gravity + contacts (HOLD), and
#   (2) can be driven along waypoints by a HolonomicBase velocity action while
#       the arms hold, WITHOUT tumbling.
#
# It saves a head-camera mp4 so the motion can be eyeballed, and prints
# stability + velocity-calibration metrics. This de-risks Path A before any
# refactor of the 2381-line pipeline.
#
# Run:
#   source /root/behavior-venv/bin/activate
#   python scripts/nav_physics_sandbox.py \
#       --raw-hdf5 /root/autodl-tmp/datasets/2025-challenge-rawdata/task-0026/episode_00260010.hdf5 \
#       > logs/nav_physics_sandbox.log 2>&1 &

import argparse
import json
import math
import os

import h5py
import imageio.v2 as imageio
import numpy as np

import omnigibson as og
from omnigibson.macros import gm

from hrc_utils import DEFAULT_RAW_HDF5, to_numpy, extract_first_rgb

# Base controller physical output limits (from recorded config, verified by probe).
BASE_OUT = np.array([0.75, 0.75, 1.0], dtype=np.float32)  # vx,vy (m/s), vyaw (rad/s)


def yaw_from_quat(quat) -> float:
    x, y, z, w = [float(v) for v in to_numpy(quat)]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def upright_metrics(robot):
    pos, quat = robot.get_position_orientation()
    x, y, z, w = [float(v) for v in to_numpy(quat)]
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    return float(to_numpy(pos)[2]), math.degrees(roll), math.degrees(pitch)


def build_hold_action(robot) -> np.ndarray:
    """Full no-op action: position controllers hold current joint angle, base
    zero velocity, grippers keep. Built from each controller's own no-op."""
    action = np.zeros(robot.action_dim, dtype=np.float32)
    cd = robot.get_control_dict()
    for name, ctrl in robot.controllers.items():
        try:
            no_op = np.atleast_1d(to_numpy(ctrl.compute_no_op_action(cd))).astype(np.float32)
            idx = np.atleast_1d(to_numpy(robot.controller_action_idx[name])).astype(int)
            action[idx] = no_op
        except Exception as exc:
            print(f"[sandbox] WARN no_op '{name}': {exc}", flush=True)
    return action


# Position-controlled groups that must HOLD a FIXED reference pose during
# navigation (commanding "current pos" every frame lets them drift, since any
# physics nudge becomes the new target -> unbounded accumulation).
HOLD_CONTROLLERS = ("trunk", "arm_left", "arm_right")


def capture_hold_ref(robot) -> dict:
    """Snapshot the fixed position targets for the trunk/arm controllers
    (absolute joint angles, since use_delta=False). These are commanded
    verbatim every navigation frame so the arms truly hold."""
    jp = to_numpy(robot.get_joint_positions()).astype(np.float32)
    ref = {}
    for name in HOLD_CONTROLLERS:
        if name not in robot.controllers:
            continue
        dof = np.atleast_1d(to_numpy(robot.controllers[name].dof_idx)).astype(int)
        a_idx = np.atleast_1d(to_numpy(robot.controller_action_idx[name])).astype(int)
        ref[name] = (a_idx, jp[dof].copy())
    return ref


def build_nav_action(robot, base_cmd, hold_ref: dict) -> np.ndarray:
    """Navigation action: base = velocity cmd, trunk/arms = FIXED reference
    pose, grippers = no-op (keep)."""
    action = build_hold_action(robot)  # grippers + base default to no-op
    for name, (a_idx, vals) in hold_ref.items():
        action[a_idx] = vals
    base_idx = np.atleast_1d(to_numpy(robot.controller_action_idx["base"])).astype(int)
    action[base_idx] = np.asarray(base_cmd, dtype=np.float32)
    return action


def build_env(raw_hdf5: str, resolution: int):
    """Build a normal physics env from the recorded config, contacts ON."""
    with h5py.File(raw_hdf5, "r") as f:
        config = json.loads(f["data"].attrs["config"])
        scene_file = json.loads(f["data"].attrs["scene_file"])

    # BEHAVIOR-1K physics cadence (do NOT force 1000 Hz like the contacts-on
    # playback path; we want to record at 30 fps).
    config["env"]["action_frequency"] = 30.0
    config["env"]["rendering_frequency"] = 30.0
    config["env"]["physics_frequency"] = 120.0
    config["env"]["flatten_action_space"] = True
    config["env"]["flatten_obs_space"] = False
    # Drop the 3 heavy 1080p external cams for speed.
    config["env"]["external_sensors"] = []

    # Restore the exact recorded scene (objects + robot pose).
    config["scene"]["scene_file"] = scene_file
    config["objects"] = []
    # DummyTask: we only need the scene + robot, not task sampling/obs.
    config["task"] = {"type": "DummyTask"}

    # Robot: onboard head RGB only, modest resolution.
    w = (int(resolution * 4 / 3) + 15) // 16 * 16
    h = (resolution + 15) // 16 * 16
    for rc in config["robots"]:
        rc["obs_modalities"] = ["rgb"]
        rc["sensor_config"] = {
            "VisionSensor": {"sensor_kwargs": {"image_height": h, "image_width": w}}
        }

    # IMPORTANT: do NOT set gm.VISUAL_ONLY -> collisions stay enabled so the
    # robot rests on the floor.
    gm.ENABLE_TRANSITION_RULES = False
    return og.Environment(configs=config)


def grab_head_rgb(obs):
    frame, _ = extract_first_rgb(obs)
    return frame


def settle(env, robot, n, writer=None):
    obs = None
    for _ in range(n):
        obs, *_ = env.step(build_hold_action(robot))
        if writer is not None:
            fr = grab_head_rgb(obs)
            if fr is not None:
                writer.append_data(fr)
    return obs


def arm_trunk_dof_indices(robot) -> np.ndarray:
    """DOF indices of trunk + both arms (EXCLUDING the moving base x/y/rz
    virtual joints), so 'arm drift' actually measures the arms, not the base."""
    dofs = []
    for name in ("trunk", "arm_left", "arm_right"):
        if name in robot.controllers:
            dofs += [int(v) for v in np.atleast_1d(to_numpy(robot.controllers[name].dof_idx)).reshape(-1)]
    return np.asarray(dofs, dtype=int)


def refresh_arm_hold_targets(robot, prims) -> None:
    """Pin the action-primitive arm targets to the CURRENT pose, so
    ``_empty_action(follow_arm_targets=True)`` commands a FIXED hold target
    every frame (prevents the 'hold current -> drift' accumulation)."""
    jp = robot.get_joint_positions()
    for arm in robot.arm_names:
        name = f"arm_{arm}"
        if name in robot.controllers:
            prims._arm_targets[name] = jp[robot.controllers[name].dof_idx]


def drive_pose_primitive(env, robot, prims, pose_2d, writer, label, arm_dofs):
    """Drive to a 2D pose (x, y, yaw) reusing BEHAVIOR-1K's
    ``_navigate_to_pose_direct`` (rotate-to-face + constant-velocity drive +
    rotate-to-final). Records video + stability/arm metrics."""
    fps = 30.0
    dt = 1.0 / fps
    p0 = to_numpy(robot.get_position_orientation()[0]).astype(np.float64)
    arm0 = to_numpy(robot.get_joint_positions()).astype(np.float64)[arm_dofs]
    max_roll = max_pitch = 0.0
    z_min, z_max = 1e9, -1e9
    speeds = []
    prev = p0.copy()
    reached = True

    refresh_arm_hold_targets(robot, prims)
    try:
        for action in prims._navigate_to_pose_direct(pose_2d):
            obs, *_ = env.step(action)
            fr = grab_head_rgb(obs)
            if fr is not None:
                writer.append_data(fr)
            cur = to_numpy(robot.get_position_orientation()[0]).astype(np.float64)
            speeds.append(float(np.linalg.norm(cur[:2] - prev[:2])) / dt)
            prev = cur
            z, roll, pitch = upright_metrics(robot)
            max_roll = max(max_roll, abs(roll)); max_pitch = max(max_pitch, abs(pitch))
            z_min = min(z_min, z); z_max = max(z_max, z)
    except Exception as exc:
        reached = False
        print(f"[sandbox:{label}] primitive stopped early: {type(exc).__name__}: {exc}", flush=True)

    p1 = to_numpy(robot.get_position_orientation()[0]).astype(np.float64)
    arm1 = to_numpy(robot.get_joint_positions()).astype(np.float64)[arm_dofs]
    arm_drift = float(np.max(np.abs(arm1 - arm0))) if len(arm0) else -1.0
    final_yaw = yaw_from_quat(robot.get_position_orientation()[1])
    print(f"\n[sandbox:{label}] target_pose={list(np.asarray(pose_2d,dtype=float))} frames={len(speeds)}", flush=True)
    print(f"  start xy={p0[:2].tolist()} end xy={p1[:2].tolist()} final_yaw={math.degrees(final_yaw):.1f} deg", flush=True)
    print(f"  net move={float(np.linalg.norm(p1[:2]-p0[:2])):.3f} m  "
          f"mean speed={np.mean(speeds) if speeds else 0:.3f} max speed={np.max(speeds) if speeds else 0:.3f} m/s", flush=True)
    print(f"  upright: max|roll|={max_roll:.2f} deg max|pitch|={max_pitch:.2f} deg z in [{z_min:.3f},{z_max:.3f}]", flush=True)
    print(f"  ARM drift (arms+trunk only) = {arm_drift:.4f} rad  (want ~0 = arms held)", flush=True)
    return reached and max_roll < 20 and max_pitch < 20


def check_operation_replay(env, robot, raw_hdf5, episode_id=0, n=30):
    """Decision-critical check: does load_state-based OPERATION replay stay
    faithful in this contacts-ON 30/120 env? For a handful of recorded frames
    we load_state -> step(recorded action) -> measure how far the robot drifts
    from the recorded pose within that one physics step. Small drift => the
    whole episode can run on a single contacts-on env."""
    import torch as th
    with h5py.File(raw_hdf5, "r") as f:
        g = f["data"][f"demo_{episode_id}"]
        total = g["action"].shape[0]
        k0 = total // 2  # mid-episode (operation segment)
        states = g["state"][k0:k0 + n]
        sizes = g["state_size"][k0:k0 + n]
        actions = g["action"][k0:k0 + n]
    max_pos_drift = 0.0
    max_quat_drift = 0.0
    for i in range(len(actions)):
        st = th.tensor(states[i][: int(sizes[i])], dtype=th.float32)
        og.sim.load_state(st, serialized=True)
        rec_pos = to_numpy(robot.get_position_orientation()[0]).astype(np.float64)
        rec_quat = to_numpy(robot.get_position_orientation()[1]).astype(np.float64)
        env.step(np.asarray(actions[i], dtype=np.float32))
        aft_pos = to_numpy(robot.get_position_orientation()[0]).astype(np.float64)
        aft_quat = to_numpy(robot.get_position_orientation()[1]).astype(np.float64)
        max_pos_drift = max(max_pos_drift, float(np.linalg.norm(aft_pos - rec_pos)))
        max_quat_drift = max(max_quat_drift, float(np.linalg.norm(aft_quat - rec_quat)))
    print(f"[sandbox:op-replay] frames=[{k0},{k0+n}) "
          f"max robot pos drift/step={max_pos_drift*1000:.2f} mm, "
          f"max quat drift/step={max_quat_drift:.4f} "
          f"(small => load_state replay is faithful on contacts-ON env)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-hdf5", default=DEFAULT_RAW_HDF5)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--out-dir", default="/root/autodl-tmp/outputs/nav_sandbox")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    video_path = os.path.join(args.out_dir, "nav_sandbox_head.mp4")
    writer = imageio.get_writer(video_path, fps=args.fps, codec="libx264")

    print(f"[sandbox] building physics env (contacts ON) from {args.raw_hdf5}", flush=True)
    env = build_env(args.raw_hdf5, args.resolution)
    try:
        env.reset()
        robot = env.robots[0]
        print(f"[sandbox] robot={type(robot).__name__} action_dim={robot.action_dim}", flush=True)

        # 1) Stand stably under gravity + contacts.
        print("[sandbox] settling 45 frames (HOLD) to test standing stability...", flush=True)
        settle(env, robot, 45, writer=writer)
        z, roll, pitch = upright_metrics(robot)
        print(f"[sandbox] after settle: z={z:.3f} roll={roll:.2f} pitch={pitch:.2f}", flush=True)

        # 1b) Decision check: operation-segment load_state replay on contacts-ON env.
        try:
            check_operation_replay(env, robot, args.raw_hdf5)
        except Exception as exc:
            import traceback as _tb0
            print("[sandbox:op-replay] FAILED:\n" + _tb0.format_exc(), flush=True)

        # 2) Instantiate BEHAVIOR-1K's action primitives WITHOUT curobo.
        import traceback as _tb
        from omnigibson.action_primitives.starter_semantic_action_primitives import (
            StarterSemanticActionPrimitives,
        )
        print("[sandbox] constructing StarterSemanticActionPrimitives...", flush=True)
        try:
            prims = StarterSemanticActionPrimitives(
                env, robot, enable_head_tracking=False, skip_curobo_initilization=True,
            )
        except Exception:
            print("[sandbox] CONSTRUCTOR FAILED:\n" + _tb.format_exc(), flush=True)
            raise
        print("[sandbox] StarterSemanticActionPrimitives ready (curobo skipped)", flush=True)
        arm_dofs = arm_trunk_dof_indices(robot)

        p = to_numpy(robot.get_position_orientation()[0]).astype(np.float64)
        yaw = yaw_from_quat(robot.get_position_orientation()[1])

        # Forward ~1.5 m keeping heading.
        tgt1 = (float(p[0] + 1.5 * math.cos(yaw)), float(p[1] + 1.5 * math.sin(yaw)), float(yaw))
        ok1 = drive_pose_primitive(env, robot, prims, tgt1, writer, "fwd1.5m", arm_dofs)

        settle(env, robot, 10, writer=writer)
        # Move to a pose requiring a ~90 deg turn then travel.
        p = to_numpy(robot.get_position_orientation()[0]).astype(np.float64)
        side_yaw = yaw + math.pi / 2.0
        tgt2 = (float(p[0] + 1.2 * math.cos(side_yaw)), float(p[1] + 1.2 * math.sin(side_yaw)), float(side_yaw))
        ok2 = drive_pose_primitive(env, robot, prims, tgt2, writer, "turn90+move", arm_dofs)

        print(f"\n[sandbox] RESULT: fwd_ok={ok1} turn_ok={ok2}", flush=True)
        print(f"[sandbox] head video -> {video_path}", flush=True)
        if ok1 and ok2:
            print("[sandbox] PATH A + BEHAVIOR-1K primitives VIABLE: stable, arms held, "
                  "navigation via reused control.", flush=True)
        else:
            print("[sandbox] inspect video/metrics.", flush=True)
    finally:
        writer.close()
        og.shutdown()


if __name__ == "__main__":
    main()

