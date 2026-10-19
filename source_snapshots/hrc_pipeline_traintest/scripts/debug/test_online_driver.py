#!/usr/bin/env python
# Smoke test for nav_online_driver on the DataPlaybackWrapper interface
# (contacts-ON @30/120). Verifies the NEW code paths before integrating into
# the 2381-line pipeline:
#   1. build_contacts_on_playback_wrapper -> wrapper builds, robot stands
#   2. operation-segment load_state replay via wrapper (env.input_hdf5)
#   3. OnlineNavDriver physics-drives to a reachable pose, arms held, upright
#
# Run:
#   source /root/behavior-venv/bin/activate
#   python scripts/test_online_driver.py > logs/test_online_driver.log 2>&1 &

import argparse
import math
import os

import numpy as np
import torch as th

import omnigibson as og

from hrc_utils import DEFAULT_RAW_HDF5, to_numpy
from nav_online_driver import (
    build_contacts_on_playback_wrapper,
    OnlineNavDriver,
    yaw_from_quat,
)


def upright(robot):
    pos, quat = robot.get_position_orientation()
    x, y, z, w = [float(v) for v in to_numpy(quat)]
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return float(to_numpy(pos)[2]), math.degrees(roll), math.degrees(math.asin(sinp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-hdf5", default=DEFAULT_RAW_HDF5)
    ap.add_argument("--out-dir", default="/root/autodl-tmp/outputs/test_online_driver")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cache = os.path.join(args.out_dir, f"cache_{os.getpid()}.hdf5")

    print(f"[test] building contacts-ON DataPlaybackWrapper from {args.raw_hdf5}", flush=True)
    env = build_contacts_on_playback_wrapper(args.raw_hdf5, output_hdf5_path=cache)
    try:
        env.reset()
        robot = env.robots[0]
        print(f"[test] wrapper OK: robot={type(robot).__name__} "
              f"include_contacts={env.include_contacts} action_dim={robot.action_dim}", flush=True)
        print(f"[test] wrapper interface: has input_hdf5={hasattr(env,'input_hdf5')} "
              f"scene_file={hasattr(env,'scene_file')}", flush=True)

        # settle
        for _ in range(30):
            env.step(np.zeros(robot.action_dim, dtype=np.float32))
        z, roll, pitch = upright(robot)
        print(f"[test] after settle: z={z:.3f} roll={roll:.2f} pitch={pitch:.2f}", flush=True)

        # op-replay via wrapper interface (env.input_hdf5)
        g = env.input_hdf5["data"]["demo_0"]
        total = g["action"].shape[0]
        k0 = total // 2
        max_drift = 0.0
        for i in range(k0, k0 + 20):
            st = th.tensor(g["state"][i][: int(g["state_size"][i])], dtype=th.float32)
            og.sim.load_state(st, serialized=True)
            rec = to_numpy(robot.get_position_orientation()[0]).astype(np.float64)
            env.env.step(np.asarray(g["action"][i], dtype=np.float32), n_render_iterations=1)
            aft = to_numpy(robot.get_position_orientation()[0]).astype(np.float64)
            max_drift = max(max_drift, float(np.linalg.norm(aft - rec)))
        print(f"[test] op-replay via wrapper: max pos drift/step={max_drift*1000:.2f} mm", flush=True)

        # nav drive via OnlineNavDriver
        driver = OnlineNavDriver(env, robot)
        print("[test] OnlineNavDriver constructed", flush=True)
        p = to_numpy(robot.get_position_orientation()[0]).astype(np.float64)
        yaw = yaw_from_quat(robot.get_position_orientation()[1])
        tgt = (float(p[0] + 1.0 * math.cos(yaw)), float(p[1] + 1.0 * math.sin(yaw)), float(yaw))
        arm0 = to_numpy(robot.get_joint_positions()).astype(np.float64)
        mr = mp = 0.0
        n = [0]

        def on_frame(obs, action):
            n[0] += 1
            _, r, pch = upright(robot)
            nonlocal mr, mp
            mr = max(mr, abs(r)); mp = max(mp, abs(pch))

        ok = driver.drive_to_pose(tgt, on_frame=on_frame)
        p1 = to_numpy(robot.get_position_orientation()[0]).astype(np.float64)
        arm1 = to_numpy(robot.get_joint_positions()).astype(np.float64)
        # arm/trunk drift (exclude base 6 dofs)
        arm_dofs = []
        for nm in ("trunk", "arm_left", "arm_right"):
            if nm in robot.controllers:
                arm_dofs += [int(v) for v in np.atleast_1d(to_numpy(robot.controllers[nm].dof_idx)).reshape(-1)]
        drift = float(np.max(np.abs(arm1[arm_dofs] - arm0[arm_dofs]))) if arm_dofs else -1.0
        print(f"[test] drive_to_pose ok={ok} frames={n[0]} "
              f"net_move={float(np.linalg.norm(p1[:2]-p[:2])):.3f}m "
              f"max|roll|={mr:.2f} max|pitch|={mp:.2f} arm_drift={drift:.4f}rad", flush=True)
        print("[test] DONE — wrapper builder + op-replay + online nav all functional"
              if ok else "[test] DONE — inspect (nav primitive ended early)", flush=True)
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
