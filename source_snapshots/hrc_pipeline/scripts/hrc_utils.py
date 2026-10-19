# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
# See LICENSE in the project root for full terms.

import json
import math
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
import torch as th
import yaml

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.envs import DataPlaybackWrapper
from omnigibson.macros import gm, macros
from omnigibson.objects.usd_object import USDObject
from omnigibson.systems.macro_particle_system import MacroPhysicalParticleSystem
from omnigibson.controllers.controller_base import ControlType
from omnigibson.utils.python_utils import create_object_from_init_info, h5py_group_to_torch

gm.HEADLESS = True
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = False
gm.USE_PBR_MATERIALS = True


DEFAULT_RAW_HDF5 = (
    "/root/autodl-tmp/datasets/2025-challenge-rawdata/task-0026/episode_00260010.hdf5"
)
DEFAULT_TASK_DATA_JSON = (
    "/root/hrc_pipeline/OmniGibson/docs/challenge/task_data.json"
)
DEFAULT_HUMAN_USD = (
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/2022.2.1/Isaac/People/Characters/"
    "original_male_adult_construction_03/male_adult_construction_03.usd"
)
FALLBACK_HUMAN_URLS = [
    DEFAULT_HUMAN_USD,
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/2023.1.1/Isaac/People/Characters/"
    "original_male_adult_construction_03/male_adult_construction_03.usd",
]

FSM_STATES = {
    "WaitNearHuman": 0,
    "ReceiveCommand": 1,
    "GoToTask": 2,
    "ExecuteBehaviorTask": 3,
    "TaskSegmentDone": 4,
    "ReturnToHuman": 5,
    "FaceHuman": 6,
    "WaitNextCommand": 7,
    "AllTasksDone": 8,
    "Relocate": 9,
}


def discover_reward_segments(hdf5_path: str, episode_id: int = 0) -> List[Tuple[int, int]]:
    """Discover sub-task segments from reward spikes in the original HDF5.

    Returns a list of (start_step, end_step) tuples. Each segment ends at
    the step right after a positive reward (i.e. sub-task completion point).
    """
    with h5py.File(hdf5_path, "r") as f:
        reward = f[f"data/demo_{episode_id}/reward"][:]
    pos_idx = np.where(reward > 0)[0]
    if len(pos_idx) == 0:
        return [(0, len(reward))]
    segments = []
    prev = 0
    for idx in pos_idx:
        end = int(idx) + 1
        segments.append((prev, end))
        prev = end
    if prev < len(reward):
        segments.append((prev, len(reward)))
    return segments


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, th.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def iter_arrays(obj: Any, prefix: str = "") -> Iterable[Tuple[str, np.ndarray]]:
    if isinstance(obj, dict):
        for key, val in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_arrays(val, next_prefix)
        return
    if isinstance(obj, (list, tuple)):
        for i, val in enumerate(obj):
            next_prefix = f"{prefix}[{i}]"
            yield from iter_arrays(val, next_prefix)
        return
    if isinstance(obj, (np.ndarray, th.Tensor)):
        yield prefix, to_numpy(obj)


def normalize_rgb(arr: np.ndarray) -> Optional[np.ndarray]:
    arr = to_numpy(arr)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3:
        return None
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] != 3:
        return None
    if np.issubdtype(arr.dtype, np.floating):
        max_val = float(np.nanmax(arr)) if arr.size else 0.0
        if max_val <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def extract_first_rgb(obs: Dict[str, Any], preference: Optional[List[str]] = None) -> Tuple[Optional[np.ndarray], Optional[str]]:
    preference = preference or ["zed_link", "zed", "eyes", "head", "robot", "camera", "rgb"]
    candidates: List[Tuple[str, np.ndarray]] = []
    for key, arr in iter_arrays(obs):
        if "rgb" not in key.lower():
            continue
        frame = normalize_rgb(arr)
        if frame is not None:
            candidates.append((key, frame))
    if not candidates:
        return None, None
    for token in preference:
        for key, frame in candidates:
            if token in key.lower():
                return frame, key
    return candidates[0][1], candidates[0][0]


def download_human_usd(usd_path: str = DEFAULT_HUMAN_USD) -> str:
    if os.path.exists(usd_path):
        return usd_path
    if not (usd_path.startswith("http://") or usd_path.startswith("https://")):
        raise FileNotFoundError(f"Human USD not found: {usd_path}")
    local_path = os.path.join(tempfile.gettempdir(), os.path.basename(usd_path) or "human_npc.usd")
    if os.path.exists(local_path):
        return local_path
    last_err = None
    for url in [usd_path, *[u for u in FALLBACK_HUMAN_URLS if u != usd_path]]:
        try:
            urllib.request.urlretrieve(url, local_path)
            return local_path
        except Exception as exc:  # pragma: no cover - network failure handling
            last_err = exc
    raise RuntimeError(f"Failed to download human USD: {last_err}")


def yaw_from_quat(quat: np.ndarray) -> float:
    x, y, z, w = [float(v) for v in to_numpy(quat)]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)], dtype=np.float32)


def forward_xy_from_quat(quat: np.ndarray) -> np.ndarray:
    yaw = yaw_from_quat(quat)
    return np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)


def pose_in_front(position: np.ndarray, orientation: np.ndarray, distance: float) -> np.ndarray:
    pos = to_numpy(position).astype(np.float32)
    forward = forward_xy_from_quat(orientation)
    return pos + forward * float(distance)


def lookat_quat(cam_pos: np.ndarray, target: np.ndarray) -> th.Tensor:
    direction = target - cam_pos
    direction = direction / max(np.linalg.norm(direction), 1e-6)
    xy_norm = float(np.linalg.norm(direction[:2]))
    pan = 0.0 if xy_norm < 1e-8 else math.atan2(-direction[0], direction[1])
    tilt = math.asin(float(direction[2]))
    return T.euler2quat(th.tensor([math.pi / 2 + tilt, 0.0, pan], dtype=th.float32))


_ema_cam_pos: Optional[np.ndarray] = None
_ema_look_target: Optional[np.ndarray] = None


def reset_camera_smooth() -> None:
    global _ema_cam_pos, _ema_look_target
    _ema_cam_pos = None
    _ema_look_target = None


def update_viewer_camera(robot_pos: np.ndarray, human_pos: np.ndarray, side_offset: float = 2.0, camera_height: float = 2.0, smooth: float = 0.12) -> None:
    global _ema_cam_pos, _ema_look_target
    robot_pos = to_numpy(robot_pos).astype(np.float32)
    human_pos = to_numpy(human_pos).astype(np.float32)
    mid = (robot_pos + human_pos) / 2.0
    delta = robot_pos[:2] - human_pos[:2]
    side_dir = np.array([-delta[1], delta[0]], dtype=np.float32)
    side_norm = float(np.linalg.norm(side_dir))
    if side_norm > 1e-6:
        side_dir /= side_norm
    else:
        side_dir = np.array([1.0, 0.0], dtype=np.float32)
    raw_cam_pos = np.array([
        mid[0] + side_dir[0] * side_offset,
        mid[1] + side_dir[1] * side_offset,
        max(mid[2], 0.0) + camera_height,
    ], dtype=np.float32)
    raw_look = mid + np.array([0.0, 0.0, 0.9], dtype=np.float32)
    if _ema_cam_pos is None:
        _ema_cam_pos = raw_cam_pos.copy()
        _ema_look_target = raw_look.copy()
    else:
        _ema_cam_pos += smooth * (raw_cam_pos - _ema_cam_pos)
        _ema_look_target += smooth * (raw_look - _ema_look_target)
    quat = lookat_quat(_ema_cam_pos, _ema_look_target)
    og.sim.viewer_camera.set_position_orientation(position=th.tensor(_ema_cam_pos), orientation=quat)
    for _ in range(2):
        og.sim.render()


def get_activity_from_hdf5(raw_hdf5_path: str) -> str:
    import h5py

    with h5py.File(raw_hdf5_path, "r") as f:
        config = json.loads(f["data"].attrs["config"])
    return config["task"]["activity_name"]


def resolve_instruction(activity_name: str, task_data_path: str = DEFAULT_TASK_DATA_JSON) -> str:
    with open(task_data_path, "r", encoding="utf-8") as f:
        task_data = json.load(f)
    for task in task_data.get("tasks", []):
        if task.get("id") == activity_name:
            return task.get("instruction", activity_name)
    return activity_name


def compute_relative_pose(robot_pos: np.ndarray, human_pos: np.ndarray) -> np.ndarray:
    robot_pos = to_numpy(robot_pos).astype(np.float32)
    human_pos = to_numpy(human_pos).astype(np.float32)
    return human_pos - robot_pos


def compute_human_visible(robot_pos: np.ndarray, robot_quat: np.ndarray, human_pos: np.ndarray, max_distance: float = 3.0, fov_deg: float = 85.0) -> int:
    rel = compute_relative_pose(robot_pos, human_pos)
    dist = float(np.linalg.norm(rel[:2]))
    if dist > max_distance:
        return 0
    forward = forward_xy_from_quat(robot_quat)[:2]
    rel_xy = rel[:2] / max(dist, 1e-6)
    cosine = float(np.clip(np.dot(forward, rel_xy), -1.0, 1.0))
    angle_deg = math.degrees(math.acos(cosine))
    return int(angle_deg <= fov_deg / 2.0)


def add_human_to_scene(env: og.Environment, human_usd_path: str, name: str = "human_npc", distance: float = 1.8) -> Any:
    existing = env.scene.object_registry("name", name)
    robot = env.robots[0]
    robot_pos, robot_quat = robot.get_position_orientation()
    # Prefer evidence camera (zed) frame and apply deterministic lateral offset candidates
    # to avoid repeated occlusion from centerline clutter.
    spawn_pos = to_numpy(robot_pos).astype(np.float32)
    spawn_quat = to_numpy(robot_quat).astype(np.float32)
    evidence_sensor_name = None
    if hasattr(robot, "sensors") and robot.sensors:
        sensor_items = list(robot.sensors.items())
        sensor_items.sort(key=lambda kv: (0 if "zed" in kv[0].lower() else 1, kv[0]))
        for sensor_name, sensor in sensor_items:
            if not hasattr(sensor, "get_position_orientation"):
                continue
            try:
                s_pos, s_quat = sensor.get_position_orientation()
                spawn_pos, spawn_quat = to_numpy(s_pos).astype(np.float32), to_numpy(s_quat).astype(np.float32)
                evidence_sensor_name = sensor_name
                break
            except Exception:
                continue

    forward = forward_xy_from_quat(spawn_quat)
    left = np.array([-forward[1], forward[0], 0.0], dtype=np.float32)
    lateral_candidates = [0.45, -0.45, 0.0]
    chosen_lateral = lateral_candidates[0]
    human_pos = spawn_pos + forward * float(distance) + left * float(chosen_lateral)
    robot_xy = to_numpy(robot_pos).astype(np.float32)[:2]
    for lateral in lateral_candidates:
        candidate = spawn_pos + forward * float(distance) + left * float(lateral)
        candidate_xy_dist = float(np.linalg.norm(candidate[:2] - robot_xy))
        # Keep the first deterministic candidate that is not too close to the robot body.
        if candidate_xy_dist >= 1.1:
            human_pos = candidate
            chosen_lateral = float(lateral)
            break

    human_pos[2] = float(max(to_numpy(robot_pos)[2], 0.0))
    spawn_debug = {
        "evidence_sensor_name": evidence_sensor_name,
        "evidence_sensor_pos": spawn_pos.tolist(),
        "evidence_sensor_quat": spawn_quat.tolist(),
        "forward_distance": float(distance),
        "lateral_candidates": lateral_candidates,
        "chosen_lateral": float(chosen_lateral),
        "chosen_human_pos": human_pos.tolist(),
    }
    if existing is not None:
        # Keep placement deterministic: even if a human object already exists, reposition it relative to robot pose.
        existing.set_position_orientation(
            position=human_pos,
            orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
        setattr(existing, "_hrc_spawn_debug", spawn_debug)
        return existing

    resolved_usd = download_human_usd(human_usd_path)
    # Detect Habitat converted package: source_asset.json sits next to character.usda
    is_habitat_package = Path(resolved_usd).parent.joinpath("source_asset.json").exists()

    if is_habitat_package:
        # Habitat UsdSkel packages are not compatible with OmniGibson's USDObject/EntityPrim
        # (no physics root link). Use the lightweight ReferencedHumanVisual instead.
        from hrc_humanoid import ReferencedHumanVisual
        human = ReferencedHumanVisual(
            relative_prim_path=f"/{name}",
            name=name,
            usd_path=resolved_usd,
        )
        human.load(env.scene)
        human.initialize()
        # Habitat assets are Y-up, OmniGibson stage is Z-up.
        # Apply 90-degree X rotation: quat_xyzw = [sin(45), 0, 0, cos(45)]
        y_up_to_z_up_quat = np.array([0.7071068, 0.0, 0.0, 0.7071068], dtype=np.float32)
        human.set_position_orientation(
            position=human_pos,
            orientation=y_up_to_z_up_quat,
        )
    else:
        human = USDObject(
            name=name,
            usd_path=resolved_usd,
            category="human",
            visual_only=True,
            position=human_pos,
            orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
        env.scene.add_object(human)
        human.set_position_orientation(
            position=human_pos,
            orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
    setattr(human, "_hrc_spawn_debug", spawn_debug)
    return human


def spawn_extra_object(
    env: Any,
    name: str,
    category: str,
    model: str,
    position: Iterable[float],
    orientation: Iterable[float] = (0.0, 0.0, 0.0, 1.0),
    in_rooms: Optional[List[str]] = None,
) -> Any:
    """Spawn an extra BEHAVIOR-1K DatasetObject on top of an existing scene.

    The new object is *not* in the recorded HDF5 state, so OmniGibson's UUID-based
    deserialization (``Registry.deserialize`` -> hash_key lookup) skips it during
    ``og.sim.load_state(serialized=True)``. The object therefore retains its
    spawned pose throughout playback and behaves like a static prop.
    """
    from omnigibson.objects.dataset_object import DatasetObject

    existing = env.scene.object_registry("name", name)
    if existing is not None:
        existing.set_position_orientation(
            position=np.asarray(position, dtype=np.float32),
            orientation=np.asarray(orientation, dtype=np.float32),
        )
        return existing

    kwargs: Dict[str, Any] = dict(
        name=name,
        category=category,
        model=model,
        position=np.asarray(position, dtype=np.float32),
        orientation=np.asarray(orientation, dtype=np.float32),
    )
    if in_rooms:
        kwargs["in_rooms"] = list(in_rooms)
    obj = DatasetObject(**kwargs)
    env.scene.add_object(obj)
    obj.set_position_orientation(
        position=np.asarray(position, dtype=np.float32),
        orientation=np.asarray(orientation, dtype=np.float32),
    )
    og.sim.step()
    obj.keep_still()
    return obj


def orient_robot_towards_target(robot: Any, target_pos: np.ndarray) -> None:
    robot_pos = to_numpy(robot.get_position_orientation()[0]).astype(np.float32)
    target_pos = to_numpy(target_pos).astype(np.float32)
    delta = target_pos - robot_pos
    yaw = math.atan2(float(delta[1]), float(delta[0]))
    robot.set_position_orientation(robot_pos, quat_from_yaw(yaw))


def build_playback_env(
    raw_hdf5_path: str,
    resolution: int,
    output_hdf5_path: Optional[str] = None,
    robot_modalities: Optional[List[str]] = None,
) -> DataPlaybackWrapper:
    gm.ENABLE_TRANSITION_RULES = False

    if output_hdf5_path is None:
        output_hdf5_path = os.path.join(tempfile.gettempdir(), "hrc_playback_tmp.hdf5")
    ensure_parent(output_hdf5_path)
    robot_modalities = robot_modalities or ["rgb"]

    width = (int(resolution * 4 / 3) + 15) // 16 * 16
    resolution = (resolution + 15) // 16 * 16
    robot_sensor_config = {
        "VisionSensor": {
            "modalities": robot_modalities,
            "sensor_kwargs": {
                "image_height": resolution,
                "image_width": width,
            },
        }
    }
    return DataPlaybackWrapper.create_from_hdf5(
        input_path=raw_hdf5_path,
        output_path=output_hdf5_path,
        robot_obs_modalities=robot_modalities,
        robot_sensor_config=robot_sensor_config,
        external_sensors_config=dict(),
        n_render_iterations=1,
        only_successes=False,
        include_robot_control=False,
        include_contacts=False,
    )


def disable_robot_control(env: DataPlaybackWrapper) -> None:
    if env.include_robot_control:
        return
    for robot in env.robots:
        robot.control_enabled = False
        for controller in robot.controllers.values():
            for dof in controller.dof_idx:
                try:
                    joint = robot.joints[robot.dof_names_ordered[dof]]
                    joint.set_control_type(control_type=ControlType.EFFORT, kp=None, kd=None)
                except (KeyError, IndexError) as exc:
                    print(f"[hrc_utils] Warning: could not disable control for dof {dof}: {exc}")


def playback_episode_with_callback(
    env: DataPlaybackWrapper,
    episode_id: int,
    callback: Callable[[str, int, Dict[str, Any], Dict[str, Any], Any], None],
    max_playback_steps: Optional[int] = None,
    pre_step_callback: Optional[Callable[[int], None]] = None,
    start_step: int = 0,
    end_step: Optional[int] = None,
    skip_init: bool = False,
) -> None:
    data_grp = env.input_hdf5["data"]
    traj_grp = data_grp[f"demo_{episode_id}"]
    transitions = json.loads(traj_grp.attrs["transitions"])
    traj_grp = h5py_group_to_torch(traj_grp)
    init_metadata = traj_grp["init_metadata"]
    action = traj_grp["action"]
    state = traj_grp["state"]
    state_size = traj_grp["state_size"]
    reward = traj_grp["reward"]
    terminated = traj_grp["terminated"]
    truncated = traj_grp["truncated"]

    total_steps = len(action)
    if end_step is None:
        end_step = total_steps
    end_step = min(end_step, total_steps)

    if not skip_init:
        env.scene.restore(env.scene_file, update_initial_file=True)
        with og.sim.stopped():
            for i, obj in enumerate(env.scene.objects):
                for attr, vals in init_metadata.items():
                    val = vals[i]
                    setattr(obj, attr, val.item() if val.ndim == 0 else val)

        env.reset()
        disable_robot_control(env)

    og.sim.load_state(state[start_step, : int(state_size[start_step])], serialized=True)
    if pre_step_callback is not None:
        pre_step_callback(start_step)
    env.current_obs, init_rew, init_term, init_trunc, init_info = env.env.step(
        action=action[start_step], n_render_iterations=env.n_render_iterations + 10
    )
    init_info["reward"] = float(reward[start_step])
    init_info["terminated"] = bool(terminated[start_step])
    init_info["truncated"] = bool(truncated[start_step])
    callback("ExecuteBehaviorTask", start_step, env.current_obs, init_info, action[start_step])

    for i in range(start_step, end_step - 1):
        if max_playback_steps is not None and (i - start_step) >= int(max_playback_steps):
            break
        next_i = i + 1
        if str(i) in transitions:
            cur_transitions = transitions[str(i)]
            scene = og.sim.scenes[0]
            for add_sys_name in cur_transitions["systems"]["add"]:
                scene.get_system(add_sys_name, force_init=True)
            for remove_sys_name in cur_transitions["systems"]["remove"]:
                scene.clear_system(remove_sys_name)
            for remove_obj_name in cur_transitions["objects"]["remove"]:
                obj = scene.object_registry("name", remove_obj_name)
                scene.remove_object(obj)
            for j, add_obj_info in enumerate(cur_transitions["objects"]["add"]):
                obj = create_object_from_init_info(add_obj_info)
                scene.add_object(obj)
                obj.set_position(th.ones(3) * 100.0 + th.ones(3) * 5 * j)
            og.sim.step()

        og.sim.load_state(state[next_i, : int(state_size[next_i])], serialized=True)
        if not env.include_contacts:
            for obj in env.scene.objects:
                obj.keep_still()
            for system in env.scene.systems:
                if isinstance(system, MacroPhysicalParticleSystem):
                    zeros = th.zeros((system.n_particles, 3))
                    system.set_particles_velocities(lin_vels=zeros, ang_vels=zeros)
        if pre_step_callback is not None:
            pre_step_callback(next_i)
        a = action[next_i]
        env.current_obs, rew, term, trunc, info = env.env.step(action=a, n_render_iterations=env.n_render_iterations)
        info["reward"] = float(reward[next_i])
        info["terminated"] = bool(terminated[next_i])
        info["truncated"] = bool(truncated[next_i])
        callback("ExecuteBehaviorTask", next_i, env.current_obs, info, a)
        env.current_episode_step_count += 1
        env.step_count += 1
        if bool(terminated[next_i]) or bool(truncated[next_i]):
            break


def playback_external_segment(
    env: Any,
    ext_hdf5_path: str,
    episode_id: int,
    callback: Callable[[str, int, Dict[str, Any], Dict[str, Any], Any], None],
    start_step: int = 0,
    end_step: Optional[int] = None,
    pre_step_callback: Optional[Callable[[int], None]] = None,
    max_playback_steps: Optional[int] = None,
    phase_label: str = "ExecuteBehaviorTask",
) -> int:
    """Replay an external HDF5's recorded segment on the current ``env``.

    Used to splice multiple tasks' recordings into one multi-task demo: the env
    was built from one HDF5, but we want to replay state/action sequences from
    a *different* HDF5 (whose objects largely overlap with the current scene).
    Object UUIDs that aren't present in the current scene are silently skipped
    by ``Registry.deserialize``; UUIDs that match are restored to the recorded
    pose. Spawned extras (whose UUIDs aren't in the external recording) keep
    their current pose.

    Returns the number of frames actually played.
    """
    n_played = 0
    with h5py.File(ext_hdf5_path, "r") as f:
        traj = f[f"data/demo_{episode_id}"]
        state = traj["state"]
        state_size = traj["state_size"]
        action = traj["action"]
        reward = traj["reward"]
        terminated = traj["terminated"]
        truncated = traj["truncated"]
        n_steps = action.shape[0]
        if end_step is None:
            end_step = n_steps
        end_step = int(min(end_step, n_steps))
        start_step = int(max(0, start_step))
        if max_playback_steps is not None:
            end_step = int(min(end_step, start_step + int(max_playback_steps)))

        for i in range(start_step, end_step):
            state_t = th.tensor(state[i, : int(state_size[i])], dtype=th.float32)
            og.sim.load_state(state_t, serialized=True)
            if not getattr(env, "include_contacts", False):
                for obj in env.scene.objects:
                    obj.keep_still()
                for system in env.scene.systems:
                    if isinstance(system, MacroPhysicalParticleSystem):
                        zeros = th.zeros((system.n_particles, 3))
                        system.set_particles_velocities(lin_vels=zeros, ang_vels=zeros)
            if pre_step_callback is not None:
                pre_step_callback(i)
            a = np.asarray(action[i], dtype=np.float32)
            obs, rew, term, trunc, info = env.env.step(
                action=a,
                n_render_iterations=getattr(env, "n_render_iterations", 1),
            )
            info["reward"] = float(reward[i])
            info["terminated"] = bool(terminated[i])
            info["truncated"] = bool(truncated[i])
            env.current_obs = obs
            callback(phase_label, i, obs, info, a)
            n_played += 1
            try:
                env.current_episode_step_count += 1
                env.step_count += 1
            except Exception:
                pass
    return n_played


def peek_robot_pos_from_external_hdf5(
    ext_hdf5_path: str,
    episode_id: int,
    step: int,
    return_quat: bool = False,
):
    """Read the robot xyz position (and optionally orientation) at a given step
    of an external HDF5.

    The recorded ``state`` is a serialized OmniGibson scene state, so the only
    safe way to extract the robot pose is to actually deserialize it. To avoid
    perturbing the scene, we snapshot the current state, ``load_state(...)``
    the recorded frame to read the robot pose, then restore the snapshot.
    """
    snapshot = og.sim.dump_state(serialized=True)
    try:
        with h5py.File(ext_hdf5_path, "r") as f:
            traj = f[f"data/demo_{episode_id}"]
            state = traj["state"]
            state_size = traj["state_size"]
            state_t = th.tensor(state[step, : int(state_size[step])], dtype=th.float32)
        og.sim.load_state(state_t, serialized=True)
        scene = og.sim.scenes[0]
        if hasattr(scene, "robots") and len(scene.robots) > 0:
            pos = to_numpy(scene.robots[0].get_position_orientation()[0]).astype(np.float32)
            quat = to_numpy(scene.robots[0].get_position_orientation()[1]).astype(np.float32)
        else:
            pos = np.zeros(3, dtype=np.float32)
            quat = np.array([0, 0, 0, 1], dtype=np.float32)
    finally:
        og.sim.load_state(snapshot, serialized=True)
    if return_quat:
        return pos, quat
    return pos


def load_recorded_human_trajectory(
    hdf5_path: str,
    round_key: str = "round_0",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load recorded ``human_pos`` (N,3) and ``human_quat`` (N,4 xyzw) arrays
    from the ``hrc/<round_key>`` group of an HDF5 file.

    These arrays are the natural human walking trajectory that was recorded
    when the original HRC demo for this task was run. Replaying them back
    on top of the env (alongside its task-state replay) makes the human
    walk exactly the way they did in the recorded demo — already synced
    with the robot's motion, no scripted send_home needed.

    Returns ``None`` if the group / datasets don't exist.
    """
    try:
        with h5py.File(hdf5_path, "r") as f:
            grp_path = f"hrc/{round_key}"
            if grp_path not in f:
                return None
            grp = f[grp_path]
            if "human_pos" not in grp or "human_quat" not in grp:
                return None
            pos = np.array(grp["human_pos"], dtype=np.float32)
            quat = np.array(grp["human_quat"], dtype=np.float32)
        if len(pos) == 0 or len(pos) != len(quat):
            return None
        return pos, quat
    except Exception as exc:
        print(f"[hrc_utils] load_recorded_human_trajectory({hdf5_path}, {round_key}) failed: {exc}", flush=True)
        return None


def drive_human_from_recording(
    human_actor: Any,
    traj_pos: np.ndarray,
    traj_quat: np.ndarray,
    progress: float,
    walk_threshold_m: float = 1e-3,
) -> None:
    """Set ``human_actor.base_pos / base_yaw`` to the recorded pose at the
    given normalized progress in [0, 1] along the recorded trajectory.

    Crucially, this also mimics what :meth:`KinematicHumanoidActor.step` does
    internally so that the walking animation actually plays:

      1. Compute the xy displacement from the previous frame
      2. Update ``_last_displacement_m`` / ``linear_speed`` / ``motion_state``
      3. Advance the animation timeline (``_advance_animation``)
      4. Update ``gait_phase`` for the synthetic keypoint swing
      5. Write the new pose to the USD (``apply_pose``)

    Without steps 1-4, ``base_pos`` teleports per frame but the human mesh
    looks like it is *sliding* (T-pose / no leg motion).

    The recorded trajectory is resampled by index, so the human follows the
    exact recorded path regardless of replay-vs-recording length mismatch.
    The human's z is preserved from spawn (recordings sometimes include
    floor-height that doesn't match our merged scene).
    """
    if human_actor is None or traj_pos is None or len(traj_pos) == 0:
        return
    n = len(traj_pos)
    progress = min(max(float(progress), 0.0), 1.0)
    idx = min(int(progress * (n - 1)), n - 1)
    pos = traj_pos[idx].astype(np.float32).copy()
    # Keep our spawned z to avoid floating/sinking due to scene-z mismatch.
    pos[2] = float(human_actor.base_pos[2])

    # --- 1. xy displacement from previous frame (drives animation tempo) ---
    prev_pos = np.asarray(human_actor.base_pos, dtype=np.float32)
    dx = float(pos[0] - prev_pos[0])
    dy = float(pos[1] - prev_pos[1])
    displacement = math.sqrt(dx * dx + dy * dy)

    # --- 2. write new base pose ---
    human_actor.base_pos = pos
    human_actor.base_yaw = float(yaw_from_quat(traj_quat[idx]))

    # --- 3. set walking kinematics so _advance_animation will advance frames ---
    step_dt = float(getattr(human_actor, "step_dt", 1.0 / 30.0))
    human_actor._last_displacement_m = displacement
    human_actor.linear_speed = displacement / max(step_dt, 1e-6)
    if displacement >= float(walk_threshold_m):
        human_actor.motion_state = "walk"
    else:
        human_actor.motion_state = "hold"

    # --- 4. advance animation timeline + synthetic gait phase ---
    if hasattr(human_actor, "_advance_animation"):
        human_actor._advance_animation()
    human_actor.gait_phase = float(getattr(human_actor, "gait_phase", 0.0)) + displacement * 2.0
    human_actor.phase_step = int(getattr(human_actor, "phase_step", 0)) + 1

    # --- 5. write pose to USD (visual mesh + skeletal animation pose) ---
    human_actor.apply_pose()


def interpolate_positions(start: np.ndarray, target: np.ndarray, n_steps: int) -> List[np.ndarray]:
    start = to_numpy(start).astype(np.float32)
    target = to_numpy(target).astype(np.float32)
    if n_steps <= 1:
        return [target]
    return [start + (target - start) * ((i + 1) / n_steps) for i in range(n_steps)]


def write_yaml(path: str, data: Dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
