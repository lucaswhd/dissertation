# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
# See LICENSE in the project root for full terms.

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import imageio.v2 as imageio
import numpy as np
import yaml

import omnigibson as og

from hrc_utils import (
    DEFAULT_HUMAN_USD,
    DEFAULT_RAW_HDF5,
    FSM_STATES,
    add_human_to_scene,
    build_playback_env,
    compute_human_visible,
    compute_relative_pose,
    ensure_parent,
    extract_first_rgb,
    get_activity_from_hdf5,
    interpolate_positions,
    normalize_rgb,
    orient_robot_towards_target,
    playback_episode_with_callback,
    quat_from_yaw,
    resolve_instruction,
    to_numpy,
    update_viewer_camera,
)


@dataclass
class EpisodeArtifacts:
    episode_root: str
    videos_dir: str
    data_dir: str
    annotations_dir: str
    meta_dir: str
    head_video_path: str
    third_video_path: str


class HRCCollector:
    def __init__(self, env: Any, human: Any, activity: str, instruction: str, artifacts: EpisodeArtifacts, fps: int):
        self.env = env
        self.human = human
        self.activity = activity
        self.instruction = instruction
        self.artifacts = artifacts
        self.fps = fps
        self.head_writer = imageio.get_writer(artifacts.head_video_path, fps=fps, codec="libx264")
        self.third_writer = imageio.get_writer(artifacts.third_video_path, fps=fps, codec="libx264")
        self.frame_rows: List[Dict[str, Any]] = []
        self.commands: List[Dict[str, Any]] = []
        self.primary_camera_key = None
        self.frame_idx = 0

    def emit_command(self, round_id: int, command_text: str) -> None:
        self.commands.append(
            {
                "round_id": round_id,
                "issued_frame": self.frame_idx,
                "activity": self.activity,
                "instruction": command_text,
            }
        )

    def record(self, phase: str, round_id: int, obs: Dict[str, Any], info: Dict[str, Any]) -> None:
        robot = self.env.robots[0]
        robot_pos, robot_quat = robot.get_position_orientation()
        human_pos, human_quat = self.human.get_position_orientation()

        robot_pos_np = to_numpy(robot_pos)
        robot_quat_np = to_numpy(robot_quat)
        human_pos_np = to_numpy(human_pos)
        human_quat_np = to_numpy(human_quat)

        head_frame, camera_key = extract_first_rgb(obs)
        if head_frame is None:
            raise RuntimeError("No robot RGB frame was found while collecting HRC data.")
        if self.primary_camera_key is None:
            self.primary_camera_key = camera_key

        update_viewer_camera(robot_pos_np, human_pos_np)
        viewer_obs, _ = og.sim.viewer_camera.get_obs()
        third_frame = normalize_rgb(to_numpy(viewer_obs["rgb"]))

        self.head_writer.append_data(head_frame)
        if third_frame is not None:
            self.third_writer.append_data(third_frame)

        relative = compute_relative_pose(robot_pos_np, human_pos_np)
        self.frame_rows.append(
            {
                "frame_idx": self.frame_idx,
                "round_id": round_id,
                "fsm_state": phase,
                "fsm_state_id": FSM_STATES[phase],
                "robot_pos": robot_pos_np.tolist(),
                "robot_quat": robot_quat_np.tolist(),
                "human_pos": human_pos_np.tolist(),
                "human_quat": human_quat_np.tolist(),
                "relative_human_pose": relative.tolist(),
                "human_visible_flag": compute_human_visible(robot_pos_np, robot_quat_np, human_pos_np),
                "info_keys": sorted(info.keys()) if isinstance(info, dict) else [],
            }
        )
        self.frame_idx += 1

    def save(self) -> None:
        self.head_writer.close()
        self.third_writer.close()

        np.savez_compressed(
            os.path.join(self.artifacts.data_dir, "hrc_data.npz"),
            robot_pos=np.asarray([row["robot_pos"] for row in self.frame_rows], dtype=np.float32),
            robot_quat=np.asarray([row["robot_quat"] for row in self.frame_rows], dtype=np.float32),
            human_pos=np.asarray([row["human_pos"] for row in self.frame_rows], dtype=np.float32),
            human_quat=np.asarray([row["human_quat"] for row in self.frame_rows], dtype=np.float32),
            relative_human_pose=np.asarray([row["relative_human_pose"] for row in self.frame_rows], dtype=np.float32),
            human_visible_flag=np.asarray([row["human_visible_flag"] for row in self.frame_rows], dtype=np.int32),
            fsm_state_id=np.asarray([row["fsm_state_id"] for row in self.frame_rows], dtype=np.int32),
            round_id=np.asarray([row["round_id"] for row in self.frame_rows], dtype=np.int32),
        )
        with open(os.path.join(self.artifacts.data_dir, "frame_log.json"), "w", encoding="utf-8") as f:
            json.dump(self.frame_rows, f, indent=2)
        with open(os.path.join(self.artifacts.annotations_dir, "commands.json"), "w", encoding="utf-8") as f:
            json.dump(self.commands, f, indent=2)
        with open(os.path.join(self.artifacts.meta_dir, "episode_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "activity": self.activity,
                    "instruction": self.instruction,
                    "camera_key": self.primary_camera_key,
                    "num_frames": self.frame_idx,
                    "fsm_states": list(FSM_STATES.keys()),
                    "videos": {
                        "head_rgb": self.artifacts.head_video_path,
                        "third_person_rgb": self.artifacts.third_video_path,
                    },
                    "schema": {
                        "data": [
                            "robot_pos",
                            "robot_quat",
                            "human_pos",
                            "human_quat",
                            "relative_human_pose",
                            "human_visible_flag",
                            "fsm_state_id",
                            "round_id",
                        ]
                    },
                },
                f,
                indent=2,
            )


def create_artifacts(output_root: str, episode_name: str) -> EpisodeArtifacts:
    episode_root = os.path.join(os.path.abspath(output_root), episode_name)
    videos_dir = os.path.join(episode_root, "videos")
    data_dir = os.path.join(episode_root, "data")
    annotations_dir = os.path.join(episode_root, "annotations")
    meta_dir = os.path.join(episode_root, "meta")
    for folder in [videos_dir, data_dir, annotations_dir, meta_dir]:
        os.makedirs(folder, exist_ok=True)
    return EpisodeArtifacts(
        episode_root=episode_root,
        videos_dir=videos_dir,
        data_dir=data_dir,
        annotations_dir=annotations_dir,
        meta_dir=meta_dir,
        head_video_path=os.path.join(videos_dir, "head_rgb.mp4"),
        third_video_path=os.path.join(videos_dir, "third_person_rgb.mp4"),
    )


def zero_phase(env: Any, collector: HRCCollector, phase: str, steps: int, round_id: int) -> None:
    robot = env.robots[0]
    zero_action = np.zeros(robot.action_dim, dtype=np.float32)
    for _ in range(steps):
        obs, _, _, _, info = env.step(zero_action, n_render_iterations=1)
        collector.record(phase, round_id, obs, info)


def return_phase(env: Any, collector: HRCCollector, human: Any, steps: int, round_id: int, stand_off: float) -> None:
    robot = env.robots[0]
    robot_pos, _ = robot.get_position_orientation()
    human_pos, _ = human.get_position_orientation()
    robot_pos = to_numpy(robot_pos)
    human_pos = to_numpy(human_pos)
    delta = robot_pos[:2] - human_pos[:2]
    dist = np.linalg.norm(delta)
    if dist < 1e-6:
        delta = np.array([1.0, 0.0], dtype=np.float32)
    else:
        delta = delta / dist
    target = human_pos.copy()
    target[0] += delta[0] * stand_off
    target[1] += delta[1] * stand_off
    target[2] = robot_pos[2]

    robot = env.robots[0]
    zero_action = np.zeros(robot.action_dim, dtype=np.float32)
    for waypoint in interpolate_positions(robot_pos, target, steps):
        yaw = np.arctan2(float(human_pos[1] - waypoint[1]), float(human_pos[0] - waypoint[0]))
        robot.set_position_orientation(waypoint, quat_from_yaw(float(yaw)))
        obs, _, _, _, info = env.step(zero_action, n_render_iterations=1)
        collector.record("ReturnToHuman", round_id, obs, info)


def collect_episode_from_args(args: argparse.Namespace) -> str:
    activity = args.activity or get_activity_from_hdf5(args.raw_hdf5)
    instruction = args.instruction or resolve_instruction(activity)
    episode_name = args.episode_name or f"{activity}_hrc_episode"
    artifacts = create_artifacts(args.output_root, episode_name)

    env = build_playback_env(args.raw_hdf5, resolution=args.resolution)
    try:
        env.reset()
        human = add_human_to_scene(env, args.human_usd, distance=args.human_distance)
        orient_robot_towards_target(env.robots[0], to_numpy(human.get_position_orientation()[0]))
        env.scene.update_initial_file()
        env.scene_file = env.scene.save(as_dict=True)

        collector = HRCCollector(env, human, activity, instruction, artifacts, fps=args.fps)
        zero_phase(env, collector, "WaitNearHuman", args.wait_steps, round_id=0)
        collector.emit_command(round_id=0, command_text=instruction)
        zero_phase(env, collector, "ReceiveCommand", args.command_steps, round_id=0)
        playback_episode_with_callback(
            env,
            episode_id=args.episode_id,
            callback=lambda phase, step_idx, obs, info, action=None: collector.record(phase, 0, obs, info),
        )
        return_phase(env, collector, human, args.return_steps, round_id=0, stand_off=args.stand_off)
        zero_phase(env, collector, "WaitNextCommand", args.command_steps, round_id=1)
        collector.save()

        with open(os.path.join(artifacts.meta_dir, "collector_config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)
        print(f"[HRC Collector] Saved episode to: {artifacts.episode_root}")
        return artifacts.episode_root
    finally:
        og.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-hdf5", default=DEFAULT_RAW_HDF5)
    parser.add_argument("--output-root", default="outputs/hrc_dataset")
    parser.add_argument("--episode-name", default=None)
    parser.add_argument("--activity", default=None)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--human-usd", default=DEFAULT_HUMAN_USD)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--wait-steps", type=int, default=20)
    parser.add_argument("--command-steps", type=int, default=10)
    parser.add_argument("--return-steps", type=int, default=30)
    parser.add_argument("--human-distance", type=float, default=1.8)
    parser.add_argument("--stand-off", type=float, default=1.0)
    parser.add_argument("--config", default=None, help="Optional YAML config. CLI args override config values.")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        for key, value in config.items():
            if getattr(args, key, None) == parser.get_default(key):
                setattr(args, key, value)
    ensure_parent(os.path.join(args.output_root, "placeholder.txt"))
    return args


if __name__ == "__main__":
    collect_episode_from_args(parse_args())
