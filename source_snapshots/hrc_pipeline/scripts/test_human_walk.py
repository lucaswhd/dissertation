# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
# See LICENSE in the project root for full terms.

#!/usr/bin/env python3
"""Minimal test: spawn a KinematicHumanoidActor and make it walk in a loop.

Outputs a short video (head cam + third-person) proving the walk animation works.
Usage:
    python scripts/test_human_walk.py --config configs/hrc_v01.yaml
"""
import argparse
import os
import sys
import math
import traceback
from pathlib import Path

import numpy as np
import imageio
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hrc_utils import (
    DEFAULT_RAW_HDF5,
    build_playback_env,
    to_numpy,
    quat_from_yaw,
    ensure_parent,
)
from hrc_humanoid import KinematicHumanoidActor, HumanPhaseMotion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hrc_v01.yaml")
    parser.add_argument("--walk-steps", type=int, default=150,
                        help="Number of sim steps for the walk test (default 150 = 5s at 30fps)")
    args = parser.parse_args()

    # --- Load config ---
    cfg = {}
    if os.path.isfile(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    raw_hdf5 = cfg.get("raw_hdf5", DEFAULT_RAW_HDF5)
    human_usd = cfg.get("human_usd")
    resolution = int(cfg.get("resolution", 256))
    fps = int(cfg.get("fps", 30))
    human_distance = float(cfg.get("human_distance", 1.8))
    walk_steps = args.walk_steps

    output_dir = os.path.join(str(cfg.get("output_dir", "outputs/walk_test")), "walk_test")
    os.makedirs(output_dir, exist_ok=True)

    playback_cache = os.path.join(output_dir, f"playback_cache_{os.getpid()}.hdf5")
    ensure_parent(playback_cache)

    print(f"[walk_test] Building env from: {raw_hdf5}", flush=True)
    env = build_playback_env(raw_hdf5, resolution=resolution, output_hdf5_path=playback_cache)

    try:
        env.reset()

        # --- Load initial state from HDF5 so we have a valid scene ---
        import h5py, json, torch as th
        import omnigibson as og
        from omnigibson.utils.python_utils import h5py_group_to_torch

        f = h5py.File(raw_hdf5, "r")
        traj = f["data"]["demo_0"]
        traj_t = h5py_group_to_torch(traj)
        state = traj_t["state"]
        state_size = traj_t["state_size"]
        action = traj_t["action"]
        # Load first frame state
        og.sim.load_state(state[0, : int(state_size[0])], serialized=True)
        env.env.step(action=action[0], n_render_iterations=5)
        f.close()
        print("[walk_test] Scene loaded at step 0", flush=True)

        # --- Create animated human ---
        actor = KinematicHumanoidActor(
            env=env,
            usd_path=human_usd,
            name="human_npc",
            distance=human_distance,
            collision_proxy=None,  # no collision for this test
            phase_motion={
                "Walk": HumanPhaseMotion(
                    mode="waypoint_loop",
                    face_robot=False,
                    speed=0.5,
                    patrol_radius=1.2,
                    loop=True,
                ),
            },
            step_dt=1.0 / fps,
        )
        # Disable collision proxy for lightweight test
        actor.collision_proxy.enabled = False

        visual = actor.spawn_near_robot()
        print(f"[walk_test] Human spawned at: {actor.base_pos.tolist()}", flush=True)
        print(f"[walk_test] Animation runtime: {actor.animation_runtime is not None}", flush=True)
        if actor.animation_runtime is not None:
            print(f"[walk_test]   walk frames: {actor.animation_runtime.frame_count}", flush=True)
            print(f"[walk_test]   walk fps: {actor.animation_runtime.walk_fps}", flush=True)
            print(f"[walk_test]   nominal speed: {actor.animation_runtime.nominal_speed_mps:.3f} m/s", flush=True)
            print(f"[walk_test]   visual usd: {actor.animation_runtime.visual_usd_path}", flush=True)

        # --- Set walk phase and run loop ---
        actor.set_phase("Walk")

        head_path = os.path.join(output_dir, "walk_test_head.mp4")
        third_path = os.path.join(output_dir, "walk_test_third.mp4")
        head_writer = imageio.get_writer(head_path, fps=fps, codec="libx264", quality=8)
        third_writer = imageio.get_writer(third_path, fps=fps, codec="libx264", quality=8)

        zero_action = np.zeros(env.robots[0].action_dim, dtype=np.float32)
        robot = env.robots[0]

        # --- Pin the robot in place so only the human moves ---
        from hrc_utils import disable_robot_control
        disable_robot_control(env)
        robot_fixed_pos, robot_fixed_quat = robot.get_position_orientation()
        print(f"[walk_test] Robot pinned at: {to_numpy(robot_fixed_pos).tolist()}", flush=True)

        # --- Helper: look-at quaternion (USD camera convention: forward = -Z local) ---
        def look_at_quat(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
            """Compute xyzw quaternion for camera at *eye* looking at *target*."""
            fwd = target - eye
            fwd_norm = np.linalg.norm(fwd)
            if fwd_norm < 1e-6:
                return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            fwd = fwd / fwd_norm
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            right = np.cross(fwd, world_up)
            right_norm = np.linalg.norm(right)
            if right_norm < 1e-6:
                right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                right = right / right_norm
            up = np.cross(right, fwd)
            # Rotation matrix columns: +X=right, +Y=up, +Z=-fwd  (USD cam looks -Z)
            R = np.stack([right, up, -fwd], axis=-1)  # 3x3
            # Matrix to quaternion (Shepperd's method)
            tr = R[0, 0] + R[1, 1] + R[2, 2]
            if tr > 0:
                s = 0.5 / math.sqrt(tr + 1.0)
                w = 0.25 / s
                x = (R[2, 1] - R[1, 2]) * s
                y = (R[0, 2] - R[2, 0]) * s
                z = (R[1, 0] - R[0, 1]) * s
            elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
            q = np.array([x, y, z, w], dtype=np.float32)
            q /= max(np.linalg.norm(q), 1e-8)
            return q

        # --- Position third-person camera: midpoint of robot & human, elevated ---
        # Use robot position as anchor (guaranteed inside room)
        def compute_third_cam_pose(human_pos: np.ndarray, robot_pos: np.ndarray):
            """Camera at midpoint between robot and human, offset up and to the side."""
            mid = (human_pos + robot_pos) / 2.0
            # Direction from human to robot (to place camera on the robot's side)
            delta = robot_pos[:2] - human_pos[:2]
            dist = max(np.linalg.norm(delta), 0.5)
            side_dir = np.array([-delta[1], delta[0]], dtype=np.float32)
            side_dir /= max(np.linalg.norm(side_dir), 1e-6)
            cam_pos = np.array([
                mid[0] + side_dir[0] * 1.5,   # 1.5m to the side
                mid[1] + side_dir[1] * 1.5,
                mid[2] + 1.8,                  # 1.8m above ground
            ], dtype=np.float32)
            # Look at human's chest
            look_target = human_pos + np.array([0.0, 0.0, 1.0], dtype=np.float32)
            cam_q = look_at_quat(cam_pos, look_target)
            return cam_pos, cam_q

        try:
            robot_pos_np = to_numpy(robot_fixed_pos).astype(np.float32)
            cam_pos, cam_q = compute_third_cam_pose(actor.base_pos, robot_pos_np)
            og.sim.viewer_camera.set_position_orientation(position=cam_pos, orientation=cam_q)
            print(f"[walk_test] Third-person camera at: {cam_pos.tolist()} looking at human", flush=True)
        except Exception as e:
            print(f"[walk_test] Warning: could not set viewer camera: {e}", flush=True)

        print(f"[walk_test] Running {walk_steps} walk steps ({walk_steps/fps:.1f}s)...", flush=True)
        for step_i in range(walk_steps):
            snap = actor.step("Walk", robot)
            # Keep robot absolutely still
            robot.set_position_orientation(robot_fixed_pos, robot_fixed_quat)
            robot.keep_still()

            # Track human with third-person camera
            try:
                cam_pos, cam_q = compute_third_cam_pose(actor.base_pos, robot_pos_np)
                og.sim.viewer_camera.set_position_orientation(position=cam_pos, orientation=cam_q)
            except Exception:
                pass

            obs, _, _, _, info = env.step(zero_action, n_render_iterations=1)

            # Extract head camera frame
            head_frame = None
            for key, val in obs.items():
                if "rgb" in key.lower():
                    frame = val
                    if hasattr(frame, "numpy"):
                        frame = frame.numpy()
                    frame = np.asarray(frame)
                    if frame.ndim == 3 and frame.shape[-1] in (3, 4):
                        head_frame = frame[:, :, :3].astype(np.uint8)
                        break

            # Third-person frame from viewer camera
            third_frame = None
            try:
                cam = og.sim.viewer_camera
                cam_obs, _ = cam.get_obs()
                for mod_key, mod_val in cam_obs.items():
                    if "rgb" in mod_key.lower():
                        arr = mod_val
                        if hasattr(arr, "numpy"):
                            arr = arr.numpy()
                        arr = np.asarray(arr)
                        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                            third_frame = arr[:, :, :3].astype(np.uint8)
                            break
            except Exception:
                pass

            if head_frame is not None:
                head_writer.append_data(head_frame)
            if third_frame is not None:
                third_writer.append_data(third_frame)

            if step_i % 30 == 0:
                print(
                    f"  step {step_i}/{walk_steps}: "
                    f"motion={snap.motion_state} speed={snap.linear_speed:.3f} "
                    f"pos={snap.base_pos[:2].tolist()} "
                    f"anim_frame={actor.animation_frame_cursor:.1f}",
                    flush=True,
                )

        head_writer.close()
        third_writer.close()

        # Save last head frame as diagnostic image
        if head_frame is not None:
            diag_path = os.path.join(output_dir, "walk_test_last_frame.png")
            imageio.imwrite(diag_path, head_frame)
            print(f"[walk_test] Last frame saved: {diag_path}", flush=True)

        print(f"[walk_test] SUCCESS", flush=True)
        print(f"  Head video:  {head_path}", flush=True)
        print(f"  Third video: {third_path}", flush=True)
        print(f"  Steps: {walk_steps} ({walk_steps/fps:.1f}s)", flush=True)
        print(f"  Final human pos: {actor.base_pos.tolist()}", flush=True)

    except Exception:
        print("[walk_test] ERROR", flush=True)
        traceback.print_exc()
        raise
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
