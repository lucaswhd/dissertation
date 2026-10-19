# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
# See LICENSE in the project root for full terms.

import argparse
import json
import math
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import h5py
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
    disable_robot_control,
    discover_reward_segments,
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
from hrc_humanoid import KinematicHumanoidActor, HumanPhaseMotion, HUMAN_MOTION_STATE_IDS


class DemoRecorder:
    def __init__(self, env: Any, human: Any, activity: str, instruction: str, output_dir: str, fps: int, input_hdf5_path: str = ""):
        self.env = env
        self.human = human
        self.activity = activity
        self.instruction = instruction
        self.output_dir = os.path.abspath(output_dir)
        self.fps = fps
        self.input_hdf5_path = input_hdf5_path

        self.videos_dir = os.path.join(self.output_dir, "videos")
        self.data_dir = os.path.join(self.output_dir, "data")
        self.annotations_dir = os.path.join(self.output_dir, "annotations")
        self.meta_dir = os.path.join(self.output_dir, "meta")
        for folder in [self.videos_dir, self.data_dir, self.annotations_dir, self.meta_dir]:
            os.makedirs(folder, exist_ok=True)

        self.head_path = os.path.join(self.videos_dir, "head_rgb.mp4")
        self.third_path = os.path.join(self.videos_dir, "third_person_rgb.mp4")
        self.head_writer = imageio.get_writer(self.head_path, fps=fps, codec="libx264")
        self.third_writer = imageio.get_writer(self.third_path, fps=fps, codec="libx264")

        self.phase_history: List[str] = []
        self.command_log: List[Dict[str, Any]] = []
        self.records: List[Dict[str, Any]] = []
        self.frame_idx = 0
        self.primary_camera_key = None
        self._human_actor: Any = None
        self._segments: List[Tuple[int, int]] = []
        self._round_exec_stats: Dict[int, Dict[str, Any]] = {}
        # Optional callback fired whenever the FSM phase changes
        # (signature: hook(old_phase: str, new_phase: str) -> None).
        # Used by hrc_multitask_demo to re-pitch the head camera per FSM phase.
        self.phase_change_hook: Any = None
        # Optional callback fired BEFORE every frame is rendered/recorded
        # (signature: hook() -> None). Used by hrc_multitask_demo to step the
        # smooth head-camera pitch interpolation each frame.
        self.pre_record_hook: Any = None
        # Optional per-frame snapshotter for head-camera state
        # (signature: provider() -> Dict[str, Any]). Returned dict is stored in
        # each record and written into HDF5 ``obs/`` so that the
        # camera-pose / pitch information stays in sync with the RGB frames.
        self.camera_state_provider: Any = None
        # Optional dataset-level metadata (env args, gaze policy, intrinsics).
        # Each (str -> JSON-serializable) entry is dumped to ``data`` attrs at
        # write time. Used to make the dataset reproducible per robomimic /
        # LeRobot v3 conventions.
        self.dataset_metadata: Dict[str, Any] = {}
        self._last_recorded_phase: str = ""

    def add_command(self, round_id: int) -> None:
        self.command_log.append(
            {
                "round_id": round_id,
                "instruction": self.instruction,
                "activity": self.activity,
                "issued_frame": self.frame_idx,
            }
        )

    def record(self, phase: str, obs: Dict[str, Any], info: Dict[str, Any], round_id: int = 0, action: Any = None) -> None:
        # Fire the optional phase-change hook BEFORE we render this frame, so any
        # camera-pose change made by the hook is reflected in the head_rgb frame
        # we are about to capture.
        if phase != self._last_recorded_phase:
            if self.phase_change_hook is not None:
                try:
                    self.phase_change_hook(self._last_recorded_phase, phase)
                except Exception as exc:
                    print(f"[DemoRecorder] phase_change_hook failed ({self._last_recorded_phase} -> {phase}): {exc}", flush=True)
            self._last_recorded_phase = phase
        # Per-frame hook (e.g. smooth head-camera pitch interpolation).
        if self.pre_record_hook is not None:
            try:
                self.pre_record_hook()
            except Exception as exc:
                print(f"[DemoRecorder] pre_record_hook failed: {exc}", flush=True)
        robot = self.env.robots[0]
        robot_pos, robot_quat = robot.get_position_orientation()
        human_pos, human_quat = self.human.get_position_orientation()

        robot_pos_np = to_numpy(robot_pos)
        robot_quat_np = to_numpy(robot_quat)
        human_pos_np = to_numpy(human_pos)
        human_quat_np = to_numpy(human_quat)

        update_viewer_camera(robot_pos_np, human_pos_np)
        viewer_obs, _ = og.sim.viewer_camera.get_obs()
        third_frame = normalize_rgb(to_numpy(viewer_obs["rgb"]))

        head_frame, camera_key = extract_first_rgb(obs)
        if head_frame is None:
            raise RuntimeError("Could not extract robot RGB frame.")
        if self.primary_camera_key is None:
            self.primary_camera_key = camera_key

        self.head_writer.append_data(head_frame)
        if third_frame is not None:
            self.third_writer.append_data(third_frame)

        # --- new: action vector ---
        if action is not None:
            action_np = to_numpy(action).astype(np.float32)
        else:
            action_np = np.zeros(robot.action_dim, dtype=np.float32)

        # --- new: joint positions ---
        joint_pos_np = to_numpy(robot.get_joint_positions()).astype(np.float32)

        # --- new: gripper state (per-arm gripper qpos) ---
        gripper_parts = []
        if hasattr(robot, 'gripper_control_idx') and hasattr(robot, 'arm_names'):
            all_joint_pos = to_numpy(robot.get_joint_positions())
            for arm in robot.arm_names:
                gripper_parts.append(all_joint_pos[robot.gripper_control_idx[arm]])
        gripper_np = np.concatenate(gripper_parts, axis=0).astype(np.float32) if gripper_parts else np.zeros(2, dtype=np.float32)

        # --- new: human motion state (0=hold, 1=walk, 2=turn) ---
        human_motion_state = 0
        if self._human_actor is not None:
            ms = getattr(self._human_actor, 'motion_state', 'hold')
            human_motion_state = HUMAN_MOTION_STATE_IDS.get(ms, 0)

        # --- new: task_success ---
        # Derive from recorded reward: once a positive reward has been seen
        # in this round, all subsequent frames are marked as success.
        task_success = 0
        if isinstance(info, dict):
            done_block = info.get("done")
            if isinstance(done_block, dict) and done_block.get("success", False):
                task_success = 1
            elif info.get("reward", 0.0) > 0:
                task_success = 1

        # --- new: reward / terminated / truncated ---
        reward_val = 0.0
        terminated_val = 0
        truncated_val = 0
        if isinstance(info, dict):
            raw_rew = info.get("reward", 0.0)
            reward_val = float(raw_rew) if isinstance(raw_rew, (int, float)) else 0.0
            raw_term = info.get("terminated", False)
            terminated_val = int(bool(raw_term)) if isinstance(raw_term, (bool, int, float)) else 0
            raw_trunc = info.get("truncated", False)
            truncated_val = int(bool(raw_trunc)) if isinstance(raw_trunc, (bool, int, float)) else 0

        rel = compute_relative_pose(robot_pos_np, human_pos_np)
        # --- new: per-frame head-camera state (pitch, local/world pose) ---
        cam_state: Dict[str, Any] = {}
        if self.camera_state_provider is not None:
            try:
                cam_state = self.camera_state_provider() or {}
            except Exception as exc:
                print(f"[DemoRecorder] camera_state_provider failed: {exc}", flush=True)
                cam_state = {}

        self.records.append(
            {
                "frame_idx": self.frame_idx,
                "round_id": round_id,
                "fsm_state": phase,
                "fsm_state_id": FSM_STATES[phase],
                "robot_pos": robot_pos_np.tolist(),
                "robot_quat": robot_quat_np.tolist(),
                "human_pos": human_pos_np.tolist(),
                "human_quat": human_quat_np.tolist(),
                "relative_human_pose": rel.tolist(),
                "human_visible_flag": compute_human_visible(robot_pos_np, robot_quat_np, human_pos_np),
                "reward_keys": sorted(info.keys()) if isinstance(info, dict) else [],
                "action": action_np.tolist(),
                "joint_positions": joint_pos_np.tolist(),
                "gripper_state": gripper_np.tolist(),
                "human_motion_state_id": human_motion_state,
                "task_success": task_success,
                "reward": reward_val,
                "terminated": terminated_val,
                "truncated": truncated_val,
                "head_camera_pitch_deg": float(cam_state.get("pitch_deg", 0.0)),
                "head_camera_local_pos": cam_state.get("local_pos", [0.0, 0.0, 0.0]),
                "head_camera_local_quat": cam_state.get("local_quat", [0.0, 0.0, 0.0, 1.0]),
                "head_camera_world_pos": cam_state.get("world_pos", [0.0, 0.0, 0.0]),
                "head_camera_world_quat": cam_state.get("world_quat", [0.0, 0.0, 0.0, 1.0]),
            }
        )
        self.phase_history.append(phase)
        self.frame_idx += 1

    def close(self) -> None:
        self.head_writer.close()
        self.third_writer.close()

        np.savez_compressed(
            os.path.join(self.data_dir, "trajectories.npz"),
            robot_pos=np.asarray([row["robot_pos"] for row in self.records], dtype=np.float32),
            robot_quat=np.asarray([row["robot_quat"] for row in self.records], dtype=np.float32),
            human_pos=np.asarray([row["human_pos"] for row in self.records], dtype=np.float32),
            human_quat=np.asarray([row["human_quat"] for row in self.records], dtype=np.float32),
            relative_human_pose=np.asarray([row["relative_human_pose"] for row in self.records], dtype=np.float32),
            human_visible_flag=np.asarray([row["human_visible_flag"] for row in self.records], dtype=np.int32),
            fsm_state_id=np.asarray([row["fsm_state_id"] for row in self.records], dtype=np.int32),
            round_id=np.asarray([row["round_id"] for row in self.records], dtype=np.int32),
            action=np.asarray([row["action"] for row in self.records], dtype=np.float32),
            joint_positions=np.asarray([row["joint_positions"] for row in self.records], dtype=np.float32),
            gripper_state=np.asarray([row["gripper_state"] for row in self.records], dtype=np.float32),
            human_motion_state_id=np.asarray([row["human_motion_state_id"] for row in self.records], dtype=np.int32),
            task_success=np.asarray([row["task_success"] for row in self.records], dtype=np.int32),
            head_camera_pitch_deg=np.asarray([row["head_camera_pitch_deg"] for row in self.records], dtype=np.float32),
            head_camera_local_pos=np.asarray([row["head_camera_local_pos"] for row in self.records], dtype=np.float32),
            head_camera_local_quat=np.asarray([row["head_camera_local_quat"] for row in self.records], dtype=np.float32),
            head_camera_world_pos=np.asarray([row["head_camera_world_pos"] for row in self.records], dtype=np.float32),
            head_camera_world_quat=np.asarray([row["head_camera_world_quat"] for row in self.records], dtype=np.float32),
        )

        with open(os.path.join(self.annotations_dir, "commands.json"), "w", encoding="utf-8") as f:
            json.dump(self.command_log, f, indent=2)
        with open(os.path.join(self.meta_dir, "episode_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "activity": self.activity,
                    "instruction": self.instruction,
                    "head_video": self.head_path,
                    "third_person_video": self.third_path,
                    "camera_key": self.primary_camera_key,
                    "num_frames": self.frame_idx,
                    "fsm_states": list(FSM_STATES.keys()),
                },
                f,
                indent=2,
            )

        self.write_hdf5()

    def write_hdf5(self) -> None:
        """Write merged HDF5 dataset: each round → data/demo_N/ with obs/ + action + metadata."""
        if not self.records:
            print("[HDF5] Skipped: no records to write", flush=True)
            return

        # --- 1) Also write into original HDF5 (backward compat) ---
        # Wrapped in try/except: writing into the read-only dataset source can
        # fail with an HDF5 file-lock conflict if another handle is still open.
        # That must not block the (more important) merged dataset write below.
        if self.input_hdf5_path and os.path.isfile(self.input_hdf5_path):
            try:
                self._write_hdf5_legacy()
            except Exception as exc:
                print(f"[HDF5-legacy] SKIPPED due to error: {exc}", flush=True)

        # --- 2) Write new merged dataset file ---
        merged_path = os.path.join(self.output_dir, "hrc_dataset.hdf5")
        rounds: Dict[int, List[Dict[str, Any]]] = {}
        for r in self.records:
            rounds.setdefault(r["round_id"], []).append(r)

        obs_keys = [
            ("robot_pos", "robot_pos", np.float32),
            ("robot_quat", "robot_quat", np.float32),
            ("joint_positions", "joint_positions", np.float32),
            ("gripper_state", "gripper_state", np.float32),
            ("human_pos", "human_pos", np.float32),
            ("human_quat", "human_quat", np.float32),
            ("relative_human_pose", "relative_human_pose", np.float32),
            ("human_visible", "human_visible_flag", np.int32),
            ("human_motion_state_id", "human_motion_state_id", np.int32),
            ("fsm_state_id", "fsm_state_id", np.int32),
            ("head_camera_pitch_deg", "head_camera_pitch_deg", np.float32),
            ("head_camera_local_pos", "head_camera_local_pos", np.float32),
            ("head_camera_local_quat", "head_camera_local_quat", np.float32),
            ("head_camera_world_pos", "head_camera_world_pos", np.float32),
            ("head_camera_world_quat", "head_camera_world_quat", np.float32),
        ]
        top_keys = [
            ("action", "action", np.float32),
            ("reward", "reward", np.float32),
            ("terminated", "terminated", np.int32),
            ("truncated", "truncated", np.int32),
            ("task_success", "task_success", np.int32),
            ("round_id", "round_id", np.int32),
        ]

        with h5py.File(merged_path, "w") as f:
            data_grp = f.create_group("data")

            for rid in sorted(rounds.keys()):
                recs = rounds[rid]
                demo_grp = data_grp.create_group(f"demo_{rid}")
                obs_grp = demo_grp.create_group("obs")

                for ds_name, rec_key, dtype in obs_keys:
                    arr = np.asarray([r[rec_key] for r in recs], dtype=dtype)
                    obs_grp.create_dataset(ds_name, data=arr, compression="gzip")

                for ds_name, rec_key, dtype in top_keys:
                    arr = np.asarray([r[rec_key] for r in recs], dtype=dtype)
                    demo_grp.create_dataset(ds_name, data=arr, compression="gzip")

                # Per-demo attrs
                demo_grp.attrs["round_id"] = rid
                demo_grp.attrs["num_frames"] = len(recs)
                demo_grp.attrs["activity"] = self.activity
                demo_grp.attrs["instruction"] = self.instruction
                if rid < len(self._segments):
                    demo_grp.attrs["segment_range"] = json.dumps(list(self._segments[rid]))
                if rid in self._round_exec_stats:
                    demo_grp.attrs["execute_stats"] = json.dumps(self._round_exec_stats[rid])

                print(f"[HDF5-merged] data/demo_{rid}: {len(recs)} frames, "
                      f"obs={len(obs_keys)} fields, top={len(top_keys)} fields", flush=True)

            # Global attrs
            data_grp.attrs["num_demos"] = len(rounds)
            data_grp.attrs["total_frames"] = len(self.records)
            data_grp.attrs["fsm_states"] = json.dumps(list(FSM_STATES.keys()))
            data_grp.attrs["activity"] = self.activity
            data_grp.attrs["instruction"] = self.instruction
            data_grp.attrs["segment_mode"] = "reward"
            if self._segments:
                data_grp.attrs["segments"] = json.dumps([list(s) for s in self._segments])
            data_grp.attrs["human_motion_states"] = json.dumps(HUMAN_MOTION_STATE_IDS)

            # Dataset-level reproducibility metadata (camera intrinsics, gaze
            # policy, env args, ...). Each entry is JSON-serialized so the
            # HDF5 stays self-describing.
            for meta_key, meta_val in (self.dataset_metadata or {}).items():
                try:
                    data_grp.attrs[str(meta_key)] = json.dumps(meta_val, default=str)
                except (TypeError, ValueError) as exc:
                    print(f"[HDF5-merged] Skipping non-serializable metadata "
                          f"'{meta_key}': {exc}", flush=True)

        print(f"[HDF5-merged] Merged dataset written to: {merged_path}", flush=True)

    def _write_hdf5_legacy(self) -> None:
        """Backward compat: write into original HDF5 as data/hrc/round_N/."""
        rounds: Dict[int, List[Dict[str, Any]]] = {}
        for r in self.records:
            rounds.setdefault(r["round_id"], []).append(r)

        dataset_keys = [
            ("robot_pos", "robot_pos", np.float32),
            ("robot_quat", "robot_quat", np.float32),
            ("human_pos", "human_pos", np.float32),
            ("human_quat", "human_quat", np.float32),
            ("relative_human_pose", "relative_human_pose", np.float32),
            ("human_visible", "human_visible_flag", np.int32),
            ("fsm_state_id", "fsm_state_id", np.int32),
            ("round_id", "round_id", np.int32),
            ("action", "action", np.float32),
            ("joint_positions", "joint_positions", np.float32),
            ("gripper_state", "gripper_state", np.float32),
            ("human_motion_state_id", "human_motion_state_id", np.int32),
            ("task_success", "task_success", np.int32),
            ("head_camera_pitch_deg", "head_camera_pitch_deg", np.float32),
            ("head_camera_local_pos", "head_camera_local_pos", np.float32),
            ("head_camera_local_quat", "head_camera_local_quat", np.float32),
            ("head_camera_world_pos", "head_camera_world_pos", np.float32),
            ("head_camera_world_quat", "head_camera_world_quat", np.float32),
        ]

        # Make sure no stale read-only handles to the same file are alive,
        # otherwise HDF5's file lock will refuse the read-write open below.
        import gc
        gc.collect()
        try:
            target_abs = os.path.abspath(self.input_hdf5_path)
            for fid in list(h5py.h5f.get_obj_ids(types=h5py.h5f.OBJ_FILE)):
                try:
                    raw_name = h5py.h5i.get_name(fid)
                    name = raw_name.decode("utf-8", "ignore") if raw_name else ""
                except Exception:
                    name = ""
                if name and os.path.abspath(name) == target_abs:
                    try:
                        h5py.h5f.close(fid)
                        print(f"[HDF5-legacy] Closed stale handle to {name}", flush=True)
                    except Exception as exc_close:
                        print(f"[HDF5-legacy] WARN: failed closing stale handle: {exc_close}", flush=True)
        except Exception as exc:
            print(f"[HDF5-legacy] WARN: pre-open cleanup failed: {exc}", flush=True)

        # locking=False avoids HDF5 file-lock contention when other parts of
        # the same process briefly held the file in read-only mode.
        try:
            f_ctx = h5py.File(self.input_hdf5_path, "a", locking=False)
        except TypeError:
            # Older h5py without `locking` kwarg fallback.
            f_ctx = h5py.File(self.input_hdf5_path, "a")

        with f_ctx as f:
            if "data/hrc" in f:
                del f["data/hrc"]
            hrc_grp = f.create_group("data/hrc")
            for rid in sorted(rounds.keys()):
                recs = rounds[rid]
                rg = hrc_grp.create_group(f"round_{rid}")
                for ds_name, rec_key, dtype in dataset_keys:
                    arr = np.asarray([r[rec_key] for r in recs], dtype=dtype)
                    rg.create_dataset(ds_name, data=arr, compression="gzip")
                rg.attrs["round_id"] = rid
                rg.attrs["activity"] = self.activity
                rg.attrs["instruction"] = self.instruction
                rg.attrs["num_frames"] = len(recs)
            hrc_grp.attrs["num_rounds"] = len(rounds)
            hrc_grp.attrs["fsm_states"] = json.dumps(list(FSM_STATES.keys()))
            hrc_grp.attrs["total_hrc_frames"] = len(self.records)

        print(f"[HDF5-legacy] Written to: {self.input_hdf5_path}", flush=True)


def run_return_phase(env: Any, recorder: DemoRecorder, human: Any, steps: int, round_id: int, stand_off: float, human_actor: Any = None) -> None:
    """Smooth return: gradually turn toward human over first turn_frames, then walk."""
    robot = env.robots[0]
    robot_pos, robot_quat = robot.get_position_orientation()
    human_pos, _ = human.get_position_orientation()
    robot_pos = to_numpy(robot_pos)
    robot_quat = to_numpy(robot_quat)
    human_pos = to_numpy(human_pos)

    direction = robot_pos[:2] - human_pos[:2]
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = np.array([1.0, 0.0], dtype=np.float32)
    else:
        direction = direction / norm
    target = human_pos.copy()
    target[0] += direction[0] * stand_off
    target[1] += direction[1] * stand_off
    target[2] = robot_pos[2]

    start_yaw = _yaw_from_quat(robot_quat)
    waypoints = interpolate_positions(robot_pos, target, steps)
    zero_action = np.zeros(robot.action_dim, dtype=np.float32)
    turn_frames = min(10, max(steps // 3, 1))

    for step_i, waypoint in enumerate(waypoints):
        cur_human_pos = to_numpy(human.get_position_orientation()[0])
        face_yaw = math.atan2(float(cur_human_pos[1] - waypoint[1]), float(cur_human_pos[0] - waypoint[0]))
        if step_i < turn_frames:
            alpha = (step_i + 1) / turn_frames
            yaw = start_yaw + _wrap_to_pi(face_yaw - start_yaw) * alpha
        else:
            yaw = face_yaw
        robot.set_position_orientation(waypoint, quat_from_yaw(float(yaw)))
        if human_actor is not None:
            human_actor.step("ReturnToHuman", robot)
        obs, rew, term, trunc, info = env.step(zero_action, n_render_iterations=1)
        info["reward"] = float(rew) if rew is not None else 0.0
        info["terminated"] = bool(term) if term is not None else False
        info["truncated"] = bool(trunc) if trunc is not None else False
        recorder.record("ReturnToHuman", obs, info, round_id=round_id)


def run_face_human_phase(env: Any, recorder: DemoRecorder, human: Any, steps: int, round_id: int, human_actor: Any = None) -> None:
    """Robot stays in place and smoothly rotates to face the human."""
    robot = env.robots[0]
    robot_pos, robot_quat = robot.get_position_orientation()
    robot_pos_np = to_numpy(robot_pos)
    start_yaw = _yaw_from_quat(to_numpy(robot_quat))
    human_pos_np = to_numpy(human.get_position_orientation()[0])
    target_yaw = math.atan2(float(human_pos_np[1] - robot_pos_np[1]), float(human_pos_np[0] - robot_pos_np[0]))

    zero_action = np.zeros(robot.action_dim, dtype=np.float32)
    n = max(steps, 1)
    for i in range(n):
        alpha = (i + 1) / n
        yaw = start_yaw + _wrap_to_pi(target_yaw - start_yaw) * alpha
        cur_pos = to_numpy(robot.get_position_orientation()[0])
        robot.set_position_orientation(cur_pos, quat_from_yaw(float(yaw)))
        if human_actor is not None:
            human_actor.step("FaceHuman", robot)
        obs, rew, term, trunc, info = env.step(zero_action, n_render_iterations=1)
        info["reward"] = float(rew) if rew is not None else 0.0
        info["terminated"] = bool(term) if term is not None else False
        info["truncated"] = bool(trunc) if trunc is not None else False
        recorder.record("FaceHuman", obs, info, round_id=round_id)


def run_go_to_task_phase(env: Any, recorder: DemoRecorder, human: Any, steps: int, round_id: int,
                         target_pos: np.ndarray, human_actor: Any = None) -> None:
    """Smoothly walk robot from current position (near human) to task start position."""
    robot = env.robots[0]
    robot_pos = to_numpy(robot.get_position_orientation()[0])
    waypoints = interpolate_positions(robot_pos, target_pos, max(steps, 1))
    zero_action = np.zeros(robot.action_dim, dtype=np.float32)

    start_yaw = _yaw_from_quat(to_numpy(robot.get_position_orientation()[1]))
    walk_yaw = math.atan2(float(target_pos[1] - robot_pos[1]), float(target_pos[0] - robot_pos[0]))
    turn_frames = min(5, max(steps // 4, 1))

    for step_i, waypoint in enumerate(waypoints):
        if step_i < turn_frames:
            alpha = (step_i + 1) / turn_frames
            yaw = start_yaw + _wrap_to_pi(walk_yaw - start_yaw) * alpha
        else:
            yaw = walk_yaw
        robot.set_position_orientation(waypoint, quat_from_yaw(float(yaw)))
        if human_actor is not None:
            human_actor.step("GoToTask", robot)
        obs, rew, term, trunc, info = env.step(zero_action, n_render_iterations=1)
        info["reward"] = float(rew) if rew is not None else 0.0
        info["terminated"] = bool(term) if term is not None else False
        info["truncated"] = bool(trunc) if trunc is not None else False
        recorder.record("GoToTask", obs, info, round_id=round_id)


def peek_robot_position_at_step(env: Any, episode_id: int, step: int) -> np.ndarray:
    """Load state at a given step and return the robot position (3D)."""
    import omnigibson as og
    data_grp = env.input_hdf5["data"]
    traj_grp = data_grp[f"demo_{episode_id}"]
    state_raw = traj_grp["state"][step]
    state_size = int(traj_grp["state_size"][step])
    import torch as th
    state_t = th.tensor(state_raw[:state_size])
    og.sim.load_state(state_t, serialized=True)
    robot_pos = to_numpy(env.robots[0].get_position_orientation()[0])
    return robot_pos.copy()


def run_zero_phase(env: Any, recorder: DemoRecorder, phase: str, steps: int, round_id: int, human_actor: Any = None) -> None:
    """Record `steps` frames of a stationary phase (WaitNearHuman, ReceiveCommand, etc.)."""
    robot = env.robots[0]
    zero_action = np.zeros(robot.action_dim, dtype=np.float32)
    for _ in range(max(steps, 1)):
        if human_actor is not None:
            human_actor.step(phase, robot)
        obs, rew, term, trunc, info = env.step(zero_action, n_render_iterations=1)
        info["reward"] = float(rew) if rew is not None else 0.0
        info["terminated"] = bool(term) if term is not None else False
        info["truncated"] = bool(trunc) if trunc is not None else False
        recorder.record(phase, obs, info, round_id=round_id)


def _load_yaml_config(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a mapping: {path}")
    return data


def _resolve(cli_value: Any, cfg: Dict[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    return cfg.get(key, default)


def _normalize_seg_vis(arr: np.ndarray) -> Optional[np.ndarray]:
    seg = to_numpy(arr)
    if seg.ndim == 4:
        seg = seg[0]
    if seg.ndim == 3 and seg.shape[0] == 1 and seg.shape[-1] != 1:
        seg = np.transpose(seg, (1, 2, 0))
    if seg.ndim == 3 and seg.shape[-1] == 1:
        seg = seg[..., 0]
    if seg.ndim != 2:
        return None

    seg = np.nan_to_num(seg, nan=0.0).astype(np.int32, copy=False)
    vis = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    # Deterministic pseudo-color map for quick visual debugging.
    vis[..., 0] = ((seg * 37) % 255).astype(np.uint8)
    vis[..., 1] = ((seg * 67) % 255).astype(np.uint8)
    vis[..., 2] = ((seg * 97) % 255).astype(np.uint8)
    return vis


def _capture_live_robot_sensor_frames(robot: Any) -> Tuple[Optional[np.ndarray], Optional[str], Optional[np.ndarray], Optional[str]]:
    rgb_candidates: List[Tuple[str, np.ndarray]] = []
    seg_candidates: List[Tuple[str, np.ndarray]] = []
    sensors = list(getattr(robot, "sensors", {}).items())
    sensors.sort(key=lambda kv: (0 if "zed" in kv[0].lower() else 1, kv[0]))

    for sensor_name, sensor in sensors:
        if not hasattr(sensor, "get_obs"):
            continue
        try:
            sensor_obs, _ = sensor.get_obs()
        except Exception:
            continue
        if not isinstance(sensor_obs, dict):
            continue
        for modality, value in sensor_obs.items():
            key = f"{sensor_name}:{modality}"
            m = str(modality).lower()
            if m == "rgb":
                frame = normalize_rgb(to_numpy(value))
                if frame is not None:
                    rgb_candidates.append((key, frame))
            elif m in {"seg_semantic", "seg_instance", "seg_instance_id"}:
                seg_frame = _normalize_seg_vis(to_numpy(value))
                if seg_frame is not None:
                    seg_candidates.append((key, seg_frame))

    rgb = rgb_candidates[0] if rgb_candidates else (None, None)
    seg = seg_candidates[0] if seg_candidates else (None, None)
    return rgb[1], rgb[0], seg[1], seg[0]


def _draw_bbox(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]
    x0 = max(0, min(w - 1, int(x0)))
    x1 = max(0, min(w - 1, int(x1)))
    y0 = max(0, min(h - 1, int(y0)))
    y1 = max(0, min(h - 1, int(y1)))
    if x1 <= x0 or y1 <= y0:
        return vis
    vis[y0 : y0 + 2, x0:x1] = color
    vis[max(y1 - 2, 0) : y1, x0:x1] = color
    vis[y0:y1, x0 : x0 + 2] = color
    vis[y0:y1, max(x1 - 2, 0) : x1] = color
    return vis


def _capture_bbox_proof(robot: Any, human_name: str = "human_npc") -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "camera_key": None,
        "bbox_modality": None,
        "human_boxes": [],
        "all_boxes_count": 0,
    }
    sensor_name, sensor = _find_evidence_sensor(robot)
    if sensor is None or not hasattr(sensor, "get_obs"):
        return result
    try:
        obs, info = sensor.get_obs()
    except Exception:
        return result
    if not isinstance(obs, dict):
        return result

    bbox_modality = None
    for candidate in ("bbox_2d_tight", "bbox_2d_loose"):
        if candidate in obs:
            bbox_modality = candidate
            break
    if bbox_modality is None:
        return result

    result["camera_key"] = f"{sensor_name}:rgb"
    result["bbox_modality"] = bbox_modality
    boxes = obs.get(bbox_modality, [])
    result["all_boxes_count"] = len(boxes) if hasattr(boxes, "__len__") else 0
    id_to_labels = {}
    if isinstance(info, dict) and isinstance(info.get(bbox_modality), dict):
        id_to_labels = info[bbox_modality].get("idToLabels", {}) or {}

    parsed_boxes: List[Dict[str, Any]] = []
    for box in boxes:
        try:
            semantic_id = int(box[0])
            x0, y0, x1, y1 = [int(v) for v in box[1:5]]
            label = str(id_to_labels.get(str(semantic_id), id_to_labels.get(semantic_id, ""))).lower()
            if human_name.lower() in label or "human" in label or "male_adult" in label:
                parsed_boxes.append(
                    {
                        "semantic_id": semantic_id,
                        "label": label,
                        "bbox_xyxy": [x0, y0, x1, y1],
                        "width": max(0, x1 - x0),
                        "height": max(0, y1 - y0),
                    }
                )
        except Exception:
            continue
    result["human_boxes"] = parsed_boxes
    return result


def _yaw_from_quat(quat: np.ndarray) -> float:
    x, y, z, w = [float(v) for v in to_numpy(quat)]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _find_evidence_sensor(robot: Any) -> Tuple[Optional[str], Optional[Any]]:
    sensors = list(getattr(robot, "sensors", {}).items())
    if not sensors:
        return None, None
    sensors.sort(key=lambda kv: (0 if "zed" in kv[0].lower() else 1, kv[0]))
    return sensors[0][0], sensors[0][1]


def _align_robot_base_for_evidence_capture(robot: Any, human_pos: np.ndarray) -> Tuple[Optional[str], Optional[List[float]], Optional[List[float]]]:
    sensor_name, sensor = _find_evidence_sensor(robot)
    if sensor is None or not hasattr(sensor, "get_position_orientation"):
        return None, None, None

    robot_pos, robot_quat = robot.get_position_orientation()
    robot_pos_np = to_numpy(robot_pos).astype(np.float32)
    robot_quat_np = to_numpy(robot_quat).astype(np.float32)
    before_pose = robot_pos_np.tolist() + robot_quat_np.tolist()

    s_pos, s_quat = sensor.get_position_orientation()
    s_pos_np = to_numpy(s_pos).astype(np.float32)
    s_quat_np = to_numpy(s_quat).astype(np.float32)
    desired_sensor_yaw = math.atan2(float(human_pos[1] - s_pos_np[1]), float(human_pos[0] - s_pos_np[0]))
    sensor_yaw = _yaw_from_quat(s_quat_np)
    delta_yaw = _wrap_to_pi(desired_sensor_yaw - sensor_yaw)

    new_robot_yaw = _yaw_from_quat(robot_quat_np) + delta_yaw
    robot.set_position_orientation(robot_pos_np, quat_from_yaw(new_robot_yaw))
    r_pos_after, r_quat_after = robot.get_position_orientation()
    after_pose = to_numpy(r_pos_after).tolist() + to_numpy(r_quat_after).tolist()
    return sensor_name, before_pose, after_pose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hrc_v01.yaml")
    parser.add_argument("--raw-hdf5", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--human-usd", default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--return-steps", type=int, default=None)
    parser.add_argument("--face-human-steps", type=int, default=None)
    parser.add_argument("--go-to-task-steps", type=int, default=None)
    parser.add_argument("--wait-steps", type=int, default=None)
    parser.add_argument("--human-distance", type=float, default=None)
    parser.add_argument("--stand-off", type=float, default=None)
    parser.add_argument("--episode-id", type=int, default=None)
    parser.add_argument("--max-playback-steps", type=int, default=None)
    parser.add_argument("--human-motion-mode", default=None)
    parser.add_argument("--align-camera-to-human-for-capture", action="store_true")
    parser.add_argument("--visibility-proof-modality", default=None, choices=["bbox_2d_tight", "bbox_2d_loose", "seg_instance", "seg_instance_id"])
    parser.add_argument("--require-task-success", action="store_true")
    args = parser.parse_args()

    cfg = _load_yaml_config(args.config)
    raw_hdf5 = _resolve(args.raw_hdf5, cfg, "raw_hdf5", DEFAULT_RAW_HDF5)
    output_dir = _resolve(args.output_dir, cfg, "output_dir", "outputs/hrc_v01_static")
    human_usd = _resolve(args.human_usd, cfg, "human_usd", DEFAULT_HUMAN_USD)
    resolution = int(_resolve(args.resolution, cfg, "resolution", 256))
    fps = int(_resolve(args.fps, cfg, "fps", 30))
    return_steps = int(_resolve(args.return_steps, cfg, "return_steps", 30))
    face_human_steps = int(_resolve(args.face_human_steps, cfg, "face_human_steps", 5))
    go_to_task_steps = int(_resolve(args.go_to_task_steps, cfg, "go_to_task_steps", 30))
    wait_steps = int(_resolve(args.wait_steps, cfg, "wait_steps", 3))
    human_distance = float(_resolve(args.human_distance, cfg, "human_distance", 1.8))
    stand_off = float(_resolve(args.stand_off, cfg, "stand_off", 1.0))
    episode_id = int(_resolve(args.episode_id, cfg, "episode_id", 0))
    max_playback_steps = _resolve(args.max_playback_steps, cfg, "max_playback_steps", None)
    max_playback_steps = None if max_playback_steps is None else int(max_playback_steps)
    human_motion_mode = str(_resolve(args.human_motion_mode, cfg, "human_motion_mode", "static")).lower()
    align_camera_to_human_for_capture = bool(
        args.align_camera_to_human_for_capture or cfg.get("align_camera_to_human_for_capture", False)
    )
    visibility_proof_modality = _resolve(args.visibility_proof_modality, cfg, "visibility_proof_modality", None)
    require_task_success = bool(args.require_task_success or cfg.get("require_task_success", False))

    if human_motion_mode not in ("static", "walk"):
        raise ValueError(f"Supported human_motion_mode: 'static', 'walk'. Got: {human_motion_mode}")

    ensure_parent(os.path.join(output_dir, "placeholder.txt"))
    print(f"[HRC v0.1] Config loaded from: {os.path.abspath(args.config)}", flush=True)
    print(f"[HRC v0.1] Output dir: {os.path.abspath(output_dir)}", flush=True)
    print(f"[HRC v0.1] Smoke max_playback_steps: {max_playback_steps}", flush=True)
    activity = get_activity_from_hdf5(raw_hdf5)
    instruction = resolve_instruction(activity)
    print(f"[HRC v0.1] Building playback env from: {raw_hdf5}", flush=True)
    playback_cache_hdf5 = os.path.join(output_dir, f"playback_cache_{os.getpid()}.hdf5")
    print(f"[HRC v0.1] Playback cache hdf5: {playback_cache_hdf5}", flush=True)
    robot_modalities = ["rgb"]
    if visibility_proof_modality is not None:
        robot_modalities.append(str(visibility_proof_modality))
    env = build_playback_env(
        raw_hdf5,
        resolution=resolution,
        output_hdf5_path=playback_cache_hdf5,
        robot_modalities=robot_modalities,
    )

    try:
        env.reset()

        # --- Load human BEFORE playback so it is visible throughout the episode ---
        human_actor = None  # set when walk mode
        if human_motion_mode == "walk":
            human_actor = KinematicHumanoidActor(
                env=env,
                usd_path=human_usd,
                name="human_npc",
                distance=human_distance,
                collision_proxy=None,
                phase_motion={
                    "ExecuteBehaviorTask": HumanPhaseMotion(
                        mode="waypoint_loop", face_robot=False, speed=0.4, patrol_radius=0.9, loop=True,
                    ),
                    "GoToTask": HumanPhaseMotion(
                        mode="hold", face_robot=True, speed=0.0, patrol_radius=0.8, loop=True,
                    ),
                    "TaskSegmentDone": HumanPhaseMotion(
                        mode="hold", face_robot=True, speed=0.0, patrol_radius=0.8, loop=True,
                    ),
                    "ReturnToHuman": HumanPhaseMotion(
                        mode="hold", face_robot=True, speed=0.0, patrol_radius=0.8, loop=True,
                    ),
                    "FaceHuman": HumanPhaseMotion(
                        mode="hold", face_robot=True, speed=0.0, patrol_radius=0.8, loop=True,
                    ),
                    "WaitNearHuman": HumanPhaseMotion(
                        mode="hold", face_robot=True, speed=0.0, patrol_radius=0.8, loop=True,
                    ),
                    "ReceiveCommand": HumanPhaseMotion(
                        mode="hold", face_robot=True, speed=0.0, patrol_radius=0.8, loop=True,
                    ),
                    "WaitNextCommand": HumanPhaseMotion(
                        mode="hold", face_robot=True, speed=0.0, patrol_radius=0.8, loop=True,
                    ),
                    "AllTasksDone": HumanPhaseMotion(
                        mode="hold", face_robot=True, speed=0.0, patrol_radius=0.8, loop=True,
                    ),
                },
                step_dt=1.0 / max(fps, 1),
            )
            human_actor.collision_proxy.enabled = False
            human_actor.spawn_near_robot()
            human = human_actor  # duck-typed: has get_position_orientation(), name
            print(f"[HRC v0.1] Human loaded (walk mode) at {human_actor.base_pos.tolist()}", flush=True)
        else:
            human = add_human_to_scene(env, human_usd, distance=human_distance)
            print("[HRC v0.1] Human loaded into scene (static)", flush=True)

        recorder = DemoRecorder(
            env=env,
            human=human,
            activity=activity,
            instruction=instruction,
            output_dir=output_dir,
            fps=fps,
            input_hdf5_path=raw_hdf5,
        )
        recorder._human_actor = human_actor
        print("[HRC v0.2] Recorder initialized", flush=True)

        # --- Discover reward-based segments ---
        segments = discover_reward_segments(raw_hdf5, episode_id=episode_id)
        num_rounds = len(segments)
        recorder._segments = segments
        print(f"[HRC v0.2] Discovered {num_rounds} sub-task segments:", flush=True)
        for si, (seg_s, seg_e) in enumerate(segments):
            print(f"  Round {si}: steps [{seg_s}, {seg_e}) = {seg_e - seg_s} steps", flush=True)

        # --- Shared helpers ---
        def _make_execute_callback(round_id: int, stats: Dict[str, Any]):
            def _cb(phase: str, step_idx: int, obs: Dict[str, Any], info: Dict[str, Any], action: Any = None) -> None:
                stats["steps"] = max(stats["steps"], int(step_idx) + 1)
                if isinstance(info, dict) and isinstance(info.get("done"), dict):
                    done_info = info["done"]
                    stats["saw_done"] = stats["saw_done"] or bool(done_info.get("done", False))
                    stats["success"] = stats["success"] or bool(done_info.get("success", False))
                recorder.record(phase, obs, info, round_id=round_id, action=action)
            return _cb

        def _make_pre_step(round_id: int):
            """Pre-step callback that syncs the walking human during playback."""
            _waypoints_set = {"done": False}

            def _pre_step(step_idx: int) -> None:
                if human_actor is None:
                    return
                # One-shot waypoint setup on the first frame of round 0 only
                if round_id == 0 and not _waypoints_set["done"] and step_idx <= 1:
                    _waypoints_set["done"] = True
                    spawn = human_actor.base_pos.copy()
                    z = float(spawn[2])
                    sp = np.array([spawn[0], spawn[1]], dtype=np.float32)
                    doors = []
                    for obj in env.scene.objects:
                        cat = (getattr(obj, "category", "") or "").lower()
                        if "door" in cat:
                            p = obj.get_position_orientation()[0]
                            p = p.cpu().numpy() if hasattr(p, "cpu") else p
                            doors.append((obj.name, float(p[0]), float(p[1])))
                    if doors:
                        trav_map = human_actor._try_get_trav_map()
                        best = None
                        for name, dx, dy in doors:
                            dr = np.array([dx, dy], dtype=np.float32)
                            v = dr - sp
                            v_n = v / (float(np.linalg.norm(v)) + 1e-6)
                            approach = dr - v_n * 0.8
                            if trav_map is None:
                                score = 0
                            else:
                                seg = human_actor._nav_path_between(
                                    np.array([sp[0], sp[1], z], dtype=np.float32),
                                    np.array([approach[0], approach[1], z], dtype=np.float32),
                                    z, trav_map, 0,
                                )
                                score = len(seg)
                            if best is None or score > best[0]:
                                best = (score, name, dx, dy, approach)
                        score, door_name, dx, dy, approach = best
                        print(f"[waypoint] round {round_id}: picked door {door_name} score={score}", flush=True)
                        human_actor.set_waypoints(
                            [[float(approach[0]), float(approach[1]), z],
                             [float(spawn[0]), float(spawn[1]), z]],
                            mode="waypoint_once", phase="ExecuteBehaviorTask",
                            loop=False, expand_with_astar=True, face_robot=False,
                        )
                human_actor.sync_after_replay_load()
                human_actor.step("ExecuteBehaviorTask", env.robots[0])
            return _pre_step

        # ======= MULTI-ROUND LOOP =======
        round_stats_all = []
        for round_id, (seg_start, seg_end) in enumerate(segments):
            is_first = (round_id == 0)
            is_last = (round_id == num_rounds - 1)
            print(f"\n{'='*60}", flush=True)
            print(f"[HRC v0.2] === ROUND {round_id}/{num_rounds-1}  steps [{seg_start}, {seg_end}) ===", flush=True)
            print(f"{'='*60}", flush=True)

            recorder.add_command(round_id=round_id)

            # 1) WaitNearHuman
            run_zero_phase(env, recorder, "WaitNearHuman", wait_steps, round_id=round_id, human_actor=human_actor)

            # 2) ReceiveCommand
            run_zero_phase(env, recorder, "ReceiveCommand", wait_steps, round_id=round_id, human_actor=human_actor)

            # 3) GoToTask — smooth walk from near-human to task start (skip for first round)
            if not is_first:
                saved_robot_pos = to_numpy(env.robots[0].get_position_orientation()[0]).copy()
                saved_robot_quat = to_numpy(env.robots[0].get_position_orientation()[1]).copy()
                task_start_pos = peek_robot_position_at_step(env, episode_id, seg_start)
                # Restore robot to near-human position (scene objects are now at seg_start state, which is fine)
                env.robots[0].set_position_orientation(saved_robot_pos, saved_robot_quat)
                print(f"[HRC v0.2] GoToTask: walking from {saved_robot_pos[:2].tolist()} to {task_start_pos[:2].tolist()}", flush=True)
                run_go_to_task_phase(env, recorder, human, go_to_task_steps, round_id=round_id,
                                     target_pos=task_start_pos, human_actor=human_actor)

            # 4) ExecuteBehaviorTask (segment playback)
            exec_stats: Dict[str, Any] = {"steps": 0, "saw_done": False, "success": False}
            cb = _make_execute_callback(round_id, exec_stats)
            pre_step_cb = _make_pre_step(round_id) if human_actor is not None else None

            playback_episode_with_callback(
                env, episode_id=episode_id,
                callback=cb,
                max_playback_steps=max_playback_steps,
                pre_step_callback=pre_step_cb,
                start_step=seg_start,
                end_step=seg_end,
                skip_init=not is_first,
            )
            print(f"[HRC v0.2] Round {round_id} execute: {exec_stats['steps']} steps, success={exec_stats['success']}", flush=True)

            # 5) TaskSegmentDone (1 frame marker)
            run_zero_phase(env, recorder, "TaskSegmentDone", 1, round_id=round_id, human_actor=human_actor)

            # 6) ReturnToHuman (smooth turn + walk, no instant orient)
            run_return_phase(env, recorder, human, return_steps, round_id=round_id, stand_off=stand_off, human_actor=human_actor)

            # 7) FaceHuman (smooth rotation to face human)
            run_face_human_phase(env, recorder, human, face_human_steps, round_id=round_id, human_actor=human_actor)

            # 8) WaitNextCommand (skip on last round, use AllTasksDone instead)
            if not is_last:
                run_zero_phase(env, recorder, "WaitNextCommand", wait_steps, round_id=round_id, human_actor=human_actor)
            else:
                run_zero_phase(env, recorder, "AllTasksDone", wait_steps, round_id=round_id, human_actor=human_actor)

            round_stats_all.append(exec_stats)
            recorder._round_exec_stats[round_id] = exec_stats
            print(f"[HRC v0.2] Round {round_id} complete. Total recorded frames so far: {recorder.frame_idx}", flush=True)

        # ======= CLOSE & SUMMARY =======
        recorder.close()

        print(f"\n[HRC v0.2] SUCCESS — {num_rounds} rounds completed")
        print(f"  Activity: {activity}")
        print(f"  Instruction: {instruction}")
        print(f"  Output dir: {os.path.abspath(output_dir)}")
        print(f"  Total frames: {recorder.frame_idx}")
        for ri, rs in enumerate(round_stats_all):
            seg_s, seg_e = segments[ri]
            print(f"  Round {ri}: segment [{seg_s},{seg_e}), exec_steps={rs['steps']}, success={rs['success']}")
        print(f"  Head video: {recorder.head_path}")
        print(f"  Third video: {recorder.third_path}")
    except Exception:
        print("[HRC v0.1] ERROR: exception during run", flush=True)
        traceback.print_exc()
        raise
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
