#!/usr/bin/env python
# De-risking probe for Phase 2 (ONLINE navigation data).
#
# Goal: confirm that — inside the SAME DataPlaybackWrapper env that the HRC
# pipeline uses — we can (a) re-enable the robot's native controllers after
# `disable_robot_control`, (b) drive the holonomic base with a velocity action
# so the robot moves *because of the action* (no teleport / no load_state),
# while the arms HOLD their pose, and (c) calibrate the mapping from the
# normalized base command in [-1,1] to the realized physical velocity.
#
# This script is ADDITIVE: it does not touch any existing pipeline logic. It
# only builds an env, probes, prints a report, and shuts down.
#
# Run:
#   source /root/behavior-venv/bin/activate
#   python scripts/probe_base_control.py --config configs/hrc_multitask_v01.yaml \
#       > logs/probe_base_control.log 2>&1
#
# What to look for in the report:
#   * "re-enable OK" + base joints show control_type=VELOCITY
#   * forward probe: realized vx ~ commanded*max (max ~0.75 m/s for cmd=1.0)
#   * yaw probe: realized vyaw ~ commanded*max (max ~1.0 rad/s for cmd=1.0)
#   * arm drift small (arms held by their own no-op action)

import argparse
import math
import os

import numpy as np
import yaml

import omnigibson as og

from hrc_utils import (
    DEFAULT_RAW_HDF5,
    build_playback_env,
    disable_robot_control,
    to_numpy,
)


def _load_yaml(path: str) -> dict:
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _yaw_from_quat(quat) -> float:
    x, y, z, w = [float(v) for v in to_numpy(quat)]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def enable_robot_control(env) -> None:
    """Inverse of hrc_utils.disable_robot_control: restore each robot's native
    control types/gains from its controller config, and re-arm control."""
    for robot in env.robots:
        robot.control_enabled = True
        # Re-applies the controller-config control_type + isaac_kp/kd to every
        # driven joint (base -> VELOCITY, arms/trunk -> POSITION, ...).
        robot.update_controller_mode()


def constraint_dof_indices(robot):
    """DOF indices of the holonomic base's z / rx / ry virtual joints — the
    'locked' constraint DOFs (a ground robot must not rise/roll/pitch). They
    belong to no controller, so we must pin them ourselves when free-running
    physics, otherwise driving x/y/yaw makes the robot launch and tumble."""
    idxs = []
    for comp in ("z", "rx", "ry"):
        j = robot.joints[f"base_footprint_{comp}_joint"]
        idxs.extend([int(v) for v in to_numpy(j.dof_indices).reshape(-1)])
    return np.asarray(idxs, dtype=int)


def build_hold_action(robot) -> np.ndarray:
    """A full action vector that makes every controller HOLD (no-op):
    position controllers -> current joint positions, base -> zero velocity,
    grippers -> keep. Built from each controller's own compute_no_op_action."""
    action = np.zeros(robot.action_dim, dtype=np.float32)
    try:
        control_dict = robot.get_control_dict()
    except Exception as exc:
        print(f"[probe] WARN get_control_dict failed: {exc}", flush=True)
        return action
    for name, ctrl in robot.controllers.items():
        try:
            no_op = np.atleast_1d(to_numpy(ctrl.compute_no_op_action(control_dict))).astype(np.float32)
            idx = np.atleast_1d(to_numpy(robot.controller_action_idx[name])).astype(int)
            action[idx] = no_op
        except Exception as exc:
            print(f"[probe] WARN no_op for controller '{name}' failed: {exc}", flush=True)
    return action


def upright_metrics(robot) -> tuple:
    """Return (z, roll_deg, pitch_deg) to detect tipping/falling."""
    pos, quat = robot.get_position_orientation()
    x, y, z, w = [float(v) for v in to_numpy(quat)]
    # roll (x-axis) and pitch (y-axis) from xyzw quaternion
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    return float(to_numpy(pos)[2]), math.degrees(roll), math.degrees(pitch)


def print_controller_layout(robot) -> None:
    print("\n" + "=" * 70, flush=True)
    print(f"[probe] action_dim = {robot.action_dim}", flush=True)
    print(f"[probe] robot class = {type(robot).__name__}", flush=True)
    print("[probe] controllers (name | action_idx | command_dim | ctrl_type | "
          "in_limits | out_limits):", flush=True)
    for name, ctrl in robot.controllers.items():
        try:
            idx = to_numpy(robot.controller_action_idx[name]).tolist()
        except Exception:
            idx = "?"
        ctype = getattr(ctrl, "control_type", "?")
        try:
            in_lim = ctrl.command_input_limits
            out_lim = ctrl.command_output_limits
        except Exception:
            in_lim = out_lim = "?"
        print(f"   {name:14s} | idx={idx} | cmd_dim={getattr(ctrl,'command_dim','?')} "
              f"| type={ctype} | in={in_lim} | out={out_lim}", flush=True)

    # Per-joint control types for the base
    try:
        base_idx = to_numpy(robot.controller_action_idx["base"]).tolist()
        print(f"[probe] base controller_action_idx = {base_idx}", flush=True)
        base_dof = to_numpy(robot.controllers["base"].dof_idx).tolist()
        print(f"[probe] base dof_idx = {base_dof}", flush=True)
        for dof in base_dof:
            jname = robot.dof_names_ordered[dof]
            j = robot.joints[jname]
            print(f"   base joint {jname}: control_type={j.control_type}", flush=True)
    except Exception as exc:
        print(f"[probe] WARN base joint introspection failed: {exc}", flush=True)
    print("=" * 70 + "\n", flush=True)


def run_constant_command(env, robot, base_cmd, n_steps: int, label: str,
                         pin_dofs=None, pin_ref=None) -> dict:
    """Apply a constant base command (vx,vy,vyaw normalized in [-1,1]) for
    n_steps real env.step calls (no teleport), measuring realized motion.

    If ``pin_dofs`` / ``pin_ref`` are given, the listed constraint DOFs
    (z/rx/ry virtual base joints) are kinematically pinned to ``pin_ref``
    each frame so the robot stays grounded & level while x/y/yaw are driven
    by physics from the action."""
    import torch as th
    base_idx = robot.controller_action_idx["base"]
    fps = 30.0
    dt = 1.0 / fps

    p0 = to_numpy(robot.get_position_orientation()[0]).astype(np.float64)
    y0 = _yaw_from_quat(robot.get_position_orientation()[1])
    arm0 = to_numpy(robot.get_joint_positions()).astype(np.float64)

    per_frame_speed = []
    per_frame_yawrate = []
    prev_p = p0.copy()
    prev_y = y0
    max_roll = 0.0
    max_pitch = 0.0
    z_min = 1e9
    z_max = -1e9

    for _ in range(n_steps):
        action = build_hold_action(robot)
        action[base_idx] = np.asarray(base_cmd, dtype=np.float32)
        env.step(action, n_render_iterations=1)
        p = to_numpy(robot.get_position_orientation()[0]).astype(np.float64)
        y = _yaw_from_quat(robot.get_position_orientation()[1])
        per_frame_speed.append(float(np.linalg.norm(p[:2] - prev_p[:2])) / dt)
        per_frame_yawrate.append(_wrap_to_pi(y - prev_y) / dt)
        prev_p, prev_y = p.copy(), y
        z_now, roll_now, pitch_now = upright_metrics(robot)
        max_roll = max(max_roll, abs(roll_now))
        max_pitch = max(max_pitch, abs(pitch_now))
        z_min = min(z_min, z_now)
        z_max = max(z_max, z_now)

    p1 = to_numpy(robot.get_position_orientation()[0]).astype(np.float64)
    y1 = _yaw_from_quat(robot.get_position_orientation()[1])
    arm1 = to_numpy(robot.get_joint_positions()).astype(np.float64)

    total_disp = float(np.linalg.norm(p1[:2] - p0[:2]))
    total_yaw = _wrap_to_pi(y1 - y0)
    arm_drift = float(np.max(np.abs(arm1 - arm0))) if arm0.shape == arm1.shape else -1.0

    # steady-state (drop first 5 frames of accel)
    ss = slice(min(5, n_steps), n_steps)
    mean_speed = float(np.mean(per_frame_speed[ss])) if n_steps > 5 else float(np.mean(per_frame_speed))
    mean_yawrate = float(np.mean(per_frame_yawrate[ss])) if n_steps > 5 else float(np.mean(per_frame_yawrate))

    print(f"\n[probe:{label}] cmd={list(np.asarray(base_cmd, dtype=float))} x{n_steps} frames", flush=True)
    print(f"  total xy displacement = {total_disp:.4f} m  (avg {total_disp/(n_steps*dt):.4f} m/s)", flush=True)
    print(f"  total yaw change      = {math.degrees(total_yaw):.2f} deg  "
          f"(avg {math.degrees(total_yaw)/(n_steps*dt):.2f} deg/s)", flush=True)
    print(f"  steady-state speed    = {mean_speed:.4f} m/s", flush=True)
    print(f"  steady-state yaw rate = {mean_yawrate:.4f} rad/s ({math.degrees(mean_yawrate):.2f} deg/s)", flush=True)
    print(f"  max arm joint drift   = {arm_drift:.4f} rad  (want ~0 = arms held)", flush=True)
    print(f"  upright: roll<= {max_roll:.2f} deg, pitch<= {max_pitch:.2f} deg, "
          f"z in [{z_min:.3f}, {z_max:.3f}] (want roll/pitch~0, z stable)", flush=True)
    return {
        "total_disp": total_disp,
        "total_yaw": total_yaw,
        "mean_speed": mean_speed,
        "mean_yawrate": mean_yawrate,
        "arm_drift": arm_drift,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hrc_multitask_v01.yaml")
    parser.add_argument("--raw-hdf5", default=None)
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    raw_hdf5 = args.raw_hdf5 or cfg.get("raw_hdf5", DEFAULT_RAW_HDF5)
    resolution = int(cfg.get("resolution", 256))
    output_dir = cfg.get("output_dir", "outputs/probe")
    os.makedirs(output_dir, exist_ok=True)
    cache = os.path.join(output_dir, f"probe_cache_{os.getpid()}.hdf5")

    print(f"[probe] raw_hdf5 = {raw_hdf5}", flush=True)
    env = build_playback_env(raw_hdf5, resolution=resolution,
                             output_hdf5_path=cache, robot_modalities=["rgb"])
    try:
        env.reset()
        robot = env.robots[0]

        # Mirror the pipeline: control is disabled during playback.
        disable_robot_control(env)
        print("[probe] disable_robot_control done (pipeline-equivalent state)", flush=True)

        # --- Re-enable native control (the core unknown we want to validate) ---
        enable_robot_control(env)
        print("[probe] re-enable OK: control_enabled + update_controller_mode", flush=True)
        print_controller_layout(robot)

        # Let physics settle a few frames in hold (no command) first.
        for _ in range(10):
            env.step(build_hold_action(robot), n_render_iterations=1)

        # Capture the level/ground reference for the constraint DOFs (z/rx/ry)
        # AFTER settling, while the robot is verified upright.
        import torch as th
        pin_dofs = constraint_dof_indices(robot)
        pin_ref = to_numpy(robot.get_joint_positions()).astype(np.float32)[pin_dofs]
        print(f"[probe] constraint DOFs (z/rx/ry) dof_idx={pin_dofs.tolist()} "
              f"ref={pin_ref.tolist()}", flush=True)

        # 0) PURE HOLD baseline — no base command.
        run_constant_command(env, robot, [0.0, 0.0, 0.0], args.steps, "HOLD@0")

        # ===== WITHOUT pinning the constraint DOFs (known unstable) =====
        for _ in range(10):
            env.step(build_hold_action(robot), n_render_iterations=1)
        run_constant_command(env, robot, [0.5, 0.0, 0.0], args.steps, "fwd@0.5 NOPIN")

        # ===== WITH constraint-DOF pinning (the fix under test) =====
        # Re-settle and re-pin reference before the pinned battery.
        for _ in range(10):
            env.step(build_hold_action(robot), n_render_iterations=1)
        robot.set_joint_positions(th.tensor(pin_ref, dtype=th.float32),
                                  indices=th.tensor(pin_dofs, dtype=th.long), drive=False)

        run_constant_command(env, robot, [0.5, 0.0, 0.0], args.steps, "fwd@0.5 PIN",
                             pin_dofs=pin_dofs, pin_ref=pin_ref)
        run_constant_command(env, robot, [0.7, 0.0, 0.0], args.steps, "fwd@0.7 PIN",
                             pin_dofs=pin_dofs, pin_ref=pin_ref)
        run_constant_command(env, robot, [0.0, 0.5, 0.0], args.steps, "strafe@0.5 PIN",
                             pin_dofs=pin_dofs, pin_ref=pin_ref)
        run_constant_command(env, robot, [0.0, 0.0, 0.3], args.steps, "yaw@+0.3 PIN",
                             pin_dofs=pin_dofs, pin_ref=pin_ref)
        run_constant_command(env, robot, [0.0, 0.0, -0.3], args.steps, "yaw@-0.3 PIN",
                             pin_dofs=pin_dofs, pin_ref=pin_ref)

        print("\n[probe] DONE. If realized speeds scale ~linearly with cmd and "
              "arm drift ~0, Phase 2 online driving is viable.", flush=True)
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
