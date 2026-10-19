#!/usr/bin/env python
# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
#
# TEMP / debug-only: render JUST the "all tasks done -> robot walks back to the
# human" segment (ReturnToHuman + FaceHuman + AllTasksDone), so the final-return
# path fix (ARCHITECTURE.md §18) can be eyeballed WITHOUT the full ~100-min
# collection. It reuses the real phase code (run_return_phase_smooth etc.), so
# what you see is exactly what collection would produce for this segment.
#
# It reproduces the logged combo-D scenario: robot finishes bacon at ~(5.22,-0.17)
# (SW of the kitchen bar/island) and the human is NE of the bar — the geometry
# that made the stand-off land in an isolated pocket behind the bar.
import argparse
import math
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hrc_humanoid import KinematicHumanoidActor
from hrc_utils import build_playback_env, to_numpy
from hrc_minimal_demo import DemoRecorder
import hrc_multitask_demo as M
from hrc_multitask_demo import (
    _build_phase_motion,
    _collect_nav_obstacles,
    run_return_phase_smooth,
    run_face_human_phase_smooth,
    run_zero_phase_pinned,
    tilt_head_camera,
    set_head_camera_aperture,
)

import omnigibson as og


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/root/hrc_pipeline/configs/hrc_combo_D.yaml")
    ap.add_argument("--robot-xy", default="", help="optional xy override; default = recorded bacon-end pose")
    ap.add_argument("--human-xy", default="8.10,1.71", help="human (relocated) xy")
    ap.add_argument("--bacon-end-step", type=int, default=8390,
                    help="recording step to restore for the robot's grounded bacon-end pose")
    ap.add_argument("--output-dir", default="/root/autodl-tmp/outputs/hrc_combo_D_finalreturn")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    raw_hdf5 = cfg["raw_hdf5"]
    human_usd = cfg.get("human_usd", "/root/hrc_pipeline/assets/character.usda")
    fps = int(cfg.get("fps", 30))
    resolution = int(cfg.get("resolution", 256))
    return_steps = int(cfg.get("return_steps", 90))
    face_human_steps = int(cfg.get("face_human_steps", 45))
    wait_steps = int(cfg.get("wait_steps", 5))
    stand_off = float(cfg.get("stand_off", 1.0))

    hx, hy = [float(v) for v in args.human_xy.split(",")]

    env = build_playback_env(raw_hdf5, resolution=resolution,
                             output_hdf5_path=os.path.join(args.output_dir, f"_cache_{os.getpid()}.hdf5"),
                             robot_modalities=["rgb"])
    env.reset()
    tilt_head_camera(env, float(cfg.get("head_camera_pitch_deg", 0.0)),
                     axis=str(cfg.get("head_camera_pitch_axis", "x")))
    set_head_camera_aperture(env, float(cfg.get("head_camera_horizontal_aperture", 0.0)))

    # --- Restore the robot to its REAL bacon-end pose from the recording, so it
    #     sits ON THE GROUND (using env.reset's default height made it look like it
    #     floats) at the realistic spot it finishes cooking (SW of the bar). The
    #     recorded orientation/arm pose is kept (natural just-finished-cooking look).
    import h5py
    import torch as th
    with h5py.File(raw_hdf5, "r") as f:
        g = f["data/demo_0"]
        n = int(g["state"].shape[0])
        st_i = min(max(args.bacon_end_step, 0), n - 1)
        st = g["state"][st_i]
        ssz = int(g["state_size"][st_i])
    og.sim.load_state(th.tensor(st[:ssz], dtype=th.float32), serialized=True)
    robot = env.robots[0]
    rpose, rquat = robot.get_position_orientation()
    rpose = to_numpy(rpose).astype(np.float32)
    if args.robot_xy:  # optional xy override, keep recorded (grounded) z + orientation
        rx_o, ry_o = [float(v) for v in args.robot_xy.split(",")]
        robot.set_position_orientation(np.array([rx_o, ry_o, float(rpose[2])], dtype=np.float32),
                                       to_numpy(rquat).astype(np.float32))
    robot.keep_still()
    rpose = to_numpy(robot.get_position_orientation()[0]).astype(np.float32)
    rx, ry, z = float(rpose[0]), float(rpose[1]), float(rpose[2])
    print(f"[final-return] robot bacon-end pose=({rx:.2f},{ry:.2f},{z:.3f}) [recorded, grounded]", flush=True)

    # --- add the human, then move it to the relocated NE-of-bar spot, facing robot
    human_actor = KinematicHumanoidActor(
        env=env, usd_path=human_usd, name="human_npc", distance=1.8,
        collision_proxy=None, phase_motion=_build_phase_motion(), step_dt=1.0 / max(fps, 1))
    human_actor.collision_proxy.enabled = False
    human_actor.spawn_near_robot()
    hz = float(human_actor.base_pos[2])
    human_actor.base_pos = np.array([hx, hy, hz], dtype=np.float32)
    human_actor.base_yaw = math.atan2(ry - hy, rx - hx)  # face the robot
    human_actor.anchor_pos = human_actor.base_pos.copy()
    human_actor.anchor_yaw = human_actor.base_yaw
    human_actor.apply_pose()
    human = human_actor
    print(f"[final-return] human at=({hx:.2f},{hy:.2f}) facing robot", flush=True)

    # --- go-around furniture (bar/tables) for the path planner
    _collect_nav_obstacles(env)

    recorder = DemoRecorder(env=env, human=human, activity="cook_bacon",
                            instruction="(debug) return to human after all tasks",
                            output_dir=args.output_dir, fps=fps, input_hdf5_path=raw_hdf5)
    recorder._human_actor = human_actor

    # --- the segment: ReturnToHuman -> FaceHuman -> AllTasksDone
    run_return_phase_smooth(env, recorder, human, return_steps, round_id=0,
                            stand_off=stand_off, human_actor=human_actor,
                            avoid_objects=[])
    run_face_human_phase_smooth(env, recorder, human, face_human_steps,
                                round_id=0, human_actor=human_actor)
    run_zero_phase_pinned(env, recorder, "AllTasksDone", max(wait_steps, 15),
                          round_id=0, human_actor=human_actor)

    recorder.close()
    print(f"\n[final-return] DONE. third-person video: {recorder.third_path}", flush=True)
    print(f"[final-return]            head video: {recorder.head_path}", flush=True)
    og.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
