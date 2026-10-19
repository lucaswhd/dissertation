# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
# See LICENSE in the project root for full terms.

"""Multi-task HRC demo.

Combines two BEHAVIOR-1K tasks in one scene:
  - Round 0 (scripted)  : Turn on the radio sitting on the coffee table
  - Round 1 (replay)    : Place the gift basket on the coffee table (task-0026 segment 0)

Both tasks share the same ``house_double_floor_lower`` scene. We use task-0026's
HDF5 as the playback base (so all gift-basket / bow / cookie objects are loaded)
and additionally spawn the radio from task-0000 onto the same coffee table.

Because OmniGibson's ``Registry.deserialize`` looks objects up by UUID and only
iterates the recorded ``n_objects`` entries, an extra spawned object whose UUID
isn't in the recording is simply skipped during ``og.sim.load_state(serialized=True)``.
The radio therefore stays put through Round 1's playback.
"""

import argparse
import json
import math
import os
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch as th
import yaml

import omnigibson as og

from hrc_utils import (
    DEFAULT_HUMAN_USD,
    build_playback_env,
    disable_robot_control,
    discover_reward_segments,
    ensure_parent,
    drive_human_from_recording,
    get_activity_from_hdf5,
    load_recorded_human_trajectory,
    peek_robot_pos_from_external_hdf5,
    playback_episode_with_callback,
    playback_external_segment,
    quat_from_yaw,
    resolve_instruction,
    spawn_extra_object,
    to_numpy,
)
from hrc_humanoid import KinematicHumanoidActor, HumanPhaseMotion
from hrc_minimal_demo import (
    DemoRecorder,
    _wrap_to_pi,
    _yaw_from_quat,
    peek_robot_position_at_step,
    run_zero_phase,
)
from hrc_utils import interpolate_positions

import h5py


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two xyzw quaternions (a then b in a's local frame)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


# Cache for the head camera's resting (un-pitched) local pose. Captured the first
# time tilt_head_camera() / set_head_camera_pitch_absolute() runs so that
# subsequent absolute pitch changes are NOT applied on top of an already-pitched
# camera (which would accumulate / drift).
_HEAD_CAM_REST_QUAT: Optional[np.ndarray] = None
_HEAD_CAM_REST_POS: Optional[np.ndarray] = None
_HEAD_CAM_LAST_PITCH_DEG: Optional[float] = None
# Smooth-gaze state: the camera's CURRENT displayed pitch and the TARGET pitch
# it is interpolating toward. ``step_head_camera_pitch_smooth`` advances current
# toward target by at most max_step_deg per call (typically called once per
# recorded frame). Phase-change hooks only update the target — the actual
# rotation happens incrementally in the per-frame step, producing a smooth
# transition instead of a snap.
_HEAD_CAM_CURRENT_PITCH_DEG: Optional[float] = None
_HEAD_CAM_TARGET_PITCH_DEG: Optional[float] = None
_HEAD_CAM_PITCH_AXIS: str = "x"


def _find_head_camera_sensor(env: Any) -> Optional[Tuple[str, Any]]:
    """Locate the robot's head ZED VisionSensor (or fallback to first sensor
    with get_position_orientation). Returns (name, sensor) or None."""
    robot = env.robots[0]
    sensors = list(getattr(robot, "sensors", {}).items())
    sensors.sort(key=lambda kv: (0 if "zed" in kv[0].lower() else 1, kv[0]))
    for name, sensor in sensors:
        if "zed" in name.lower() and hasattr(sensor, "set_position_orientation"):
            return name, sensor
    for name, sensor in sensors:
        if hasattr(sensor, "set_position_orientation"):
            return name, sensor
    return None


def _capture_head_camera_rest_pose(env: Any) -> bool:
    """Cache the camera's current local pose as the 'rest' baseline. Idempotent.
    Returns True iff a rest pose is now available."""
    global _HEAD_CAM_REST_QUAT, _HEAD_CAM_REST_POS
    if _HEAD_CAM_REST_QUAT is not None:
        return True
    found = _find_head_camera_sensor(env)
    if found is None:
        return False
    _, sensor = found
    pos, quat = sensor.get_position_orientation(frame="parent")
    _HEAD_CAM_REST_POS = to_numpy(pos).astype(np.float64).copy()
    _HEAD_CAM_REST_QUAT = to_numpy(quat).astype(np.float64).copy()
    return True


def tilt_head_camera(env: Any, pitch_deg: float, axis: str = "x") -> None:
    """Pitch the robot's onboard ZED camera down by ``pitch_deg`` degrees.

    The camera is an XFormPrim rigidly parented to the robot's ``zed_link``.
    Its local offset is NOT part of the serialized scene state, so a one-time
    set here persists across ``og.sim.load_state`` replay calls.

    Positive ``pitch_deg`` tilts the camera nose-down (look toward the floor /
    tabletop).  ``axis`` selects which local axis to rotate about in case the
    camera's convention differs from the USD default (look down -Z, up +Y).

    NOTE: This function applies the rotation **on top of the camera's current
    local pose**.  For phase-conditioned dynamic gaze (where pitch must be
    re-set every time the FSM phase changes) use
    :func:`set_head_camera_pitch_absolute` instead, which rotates from the
    cached rest pose and therefore does not accumulate.
    """
    if not pitch_deg:
        # Still capture the rest pose so subsequent absolute calls have a baseline.
        _capture_head_camera_rest_pose(env)
        return
    found = _find_head_camera_sensor(env)
    if found is None:
        print("[multitask] tilt_head_camera: no VisionSensor with set_position_orientation found", flush=True)
        return

    name, sensor = found
    # Cache rest BEFORE applying the first pitch — used by absolute pitch later.
    _capture_head_camera_rest_pose(env)
    pos, quat = sensor.get_position_orientation(frame="parent")  # quat is xyzw
    pos = to_numpy(pos).astype(np.float64)
    quat = to_numpy(quat).astype(np.float64)

    half = math.radians(pitch_deg) / 2.0
    s, c = math.sin(half), math.cos(half)
    axis = axis.lower()
    if axis == "x":
        dq = np.array([s, 0.0, 0.0, c])
    elif axis == "y":
        dq = np.array([0.0, s, 0.0, c])
    else:  # "z"
        dq = np.array([0.0, 0.0, s, c])

    # Apply the pitch in the camera's OWN local frame: q_new = q ⊗ dq.
    new_quat = _quat_mul_xyzw(quat, dq)
    new_quat = new_quat / (np.linalg.norm(new_quat) + 1e-12)

    sensor.set_position_orientation(
        position=th.tensor(pos, dtype=th.float32),
        orientation=th.tensor(new_quat, dtype=th.float32),
        frame="parent",
    )
    global _HEAD_CAM_LAST_PITCH_DEG, _HEAD_CAM_CURRENT_PITCH_DEG, _HEAD_CAM_TARGET_PITCH_DEG, _HEAD_CAM_PITCH_AXIS
    _HEAD_CAM_LAST_PITCH_DEG = float(pitch_deg)
    # Seed the smooth-gaze state so that subsequent phase changes interpolate
    # from this initial static pitch instead of jumping from 0°.
    _HEAD_CAM_CURRENT_PITCH_DEG = float(pitch_deg)
    _HEAD_CAM_TARGET_PITCH_DEG = float(pitch_deg)
    _HEAD_CAM_PITCH_AXIS = axis
    print(f"[multitask] tilt_head_camera: '{name}' pitched {pitch_deg:+.1f}° about local {axis} "
          f"(quat {quat.round(3).tolist()} -> {new_quat.round(3).tolist()})", flush=True)


def set_head_camera_pitch_absolute(env: Any, pitch_deg: float, axis: str = "x",
                                   verbose: bool = True) -> bool:
    """Set the head camera pitch to ``pitch_deg`` **relative to its rest pose**
    (i.e. the un-pitched local pose captured the first time the camera was
    touched). Idempotent / safe to call every frame — pitches are not
    accumulated, they are absolute about the rest baseline.

    Returns True iff the pitch was applied.
    """
    if not _capture_head_camera_rest_pose(env):
        if verbose:
            print("[multitask] set_head_camera_pitch_absolute: no head camera found", flush=True)
        return False
    found = _find_head_camera_sensor(env)
    if found is None:
        return False
    name, sensor = found
    rest_pos = _HEAD_CAM_REST_POS
    rest_quat = _HEAD_CAM_REST_QUAT
    half = math.radians(float(pitch_deg)) / 2.0
    s, c = math.sin(half), math.cos(half)
    axis = axis.lower()
    if axis == "x":
        dq = np.array([s, 0.0, 0.0, c])
    elif axis == "y":
        dq = np.array([0.0, s, 0.0, c])
    else:
        dq = np.array([0.0, 0.0, s, c])
    new_quat = _quat_mul_xyzw(rest_quat, dq)
    new_quat = new_quat / (np.linalg.norm(new_quat) + 1e-12)
    sensor.set_position_orientation(
        position=th.tensor(rest_pos, dtype=th.float32),
        orientation=th.tensor(new_quat, dtype=th.float32),
        frame="parent",
    )
    global _HEAD_CAM_LAST_PITCH_DEG, _HEAD_CAM_CURRENT_PITCH_DEG, _HEAD_CAM_TARGET_PITCH_DEG, _HEAD_CAM_PITCH_AXIS
    _HEAD_CAM_LAST_PITCH_DEG = float(pitch_deg)
    _HEAD_CAM_CURRENT_PITCH_DEG = float(pitch_deg)
    _HEAD_CAM_TARGET_PITCH_DEG = float(pitch_deg)
    _HEAD_CAM_PITCH_AXIS = axis
    if verbose:
        print(f"[multitask] set_head_camera_pitch_absolute: '{name}' pitch -> "
              f"{pitch_deg:+.1f}° (about local {axis}, from rest quat)", flush=True)
    return True


def step_head_camera_pitch_smooth(env: Any, max_step_deg: float = 1.5) -> None:
    """Advance the head-camera pitch toward ``_HEAD_CAM_TARGET_PITCH_DEG`` by
    at most ``max_step_deg`` degrees, using ease-out interpolation toward
    the remaining error. Call once per recorded frame.

    Uses ``set_head_camera_pitch_absolute`` internally (so no drift). When the
    current pitch is already within ``max_step_deg`` of the target, the
    function performs a final exact set and stops printing.

    Setting ``max_step_deg <= 0`` disables smoothing (snap behaviour).
    """
    global _HEAD_CAM_CURRENT_PITCH_DEG, _HEAD_CAM_TARGET_PITCH_DEG
    if _HEAD_CAM_TARGET_PITCH_DEG is None:
        return
    if _HEAD_CAM_CURRENT_PITCH_DEG is None:
        _HEAD_CAM_CURRENT_PITCH_DEG = float(_HEAD_CAM_TARGET_PITCH_DEG)
    cur = float(_HEAD_CAM_CURRENT_PITCH_DEG)
    tgt = float(_HEAD_CAM_TARGET_PITCH_DEG)
    err = tgt - cur
    if abs(err) < 1e-3:
        return
    if max_step_deg is None or max_step_deg <= 0:
        new_pitch = tgt
    else:
        # Ease-out: take min(max_step, |err|*0.35) so the camera decelerates
        # as it approaches the target instead of stopping abruptly.
        eased = abs(err) * 0.35
        step = min(float(max_step_deg), max(eased, 0.25))
        step = min(step, abs(err))
        new_pitch = cur + math.copysign(step, err)
    set_head_camera_pitch_absolute(env, new_pitch, axis=_HEAD_CAM_PITCH_AXIS, verbose=False)
    # set_head_camera_pitch_absolute() resets target=current; restore target
    # so the interpolation can keep advancing on the next frame.
    _HEAD_CAM_TARGET_PITCH_DEG = tgt


def make_phase_pitch_hook(env: Any, phase_pitch_map: Dict[str, float],
                          axis: str = "x", default_pitch: Optional[float] = None,
                          smooth: bool = True
                          ) -> Callable[[str, str], None]:
    """Build a callback ``hook(old_phase, new_phase)`` that updates the head
    camera's TARGET pitch whenever the FSM phase changes (using
    ``phase_pitch_map[new_phase]`` or ``default_pitch`` as fallback).

    When ``smooth=True`` (default) the hook only updates the target — the
    actual pitch is interpolated frame-by-frame by
    :func:`step_head_camera_pitch_smooth` (called from the per-frame recorder
    step). When ``smooth=False`` the hook snaps the camera immediately
    (legacy behaviour).
    """
    if not phase_pitch_map and default_pitch is None:
        return lambda *_a, **_kw: None

    def _hook(old_phase: str, new_phase: str) -> None:
        if new_phase in phase_pitch_map:
            pitch = float(phase_pitch_map[new_phase])
        elif default_pitch is not None:
            pitch = float(default_pitch)
        else:
            return
        global _HEAD_CAM_TARGET_PITCH_DEG, _HEAD_CAM_PITCH_AXIS
        _HEAD_CAM_PITCH_AXIS = axis
        if smooth:
            # Just retarget — interpolation happens in the per-frame step.
            _HEAD_CAM_TARGET_PITCH_DEG = pitch
            cur = _HEAD_CAM_CURRENT_PITCH_DEG if _HEAD_CAM_CURRENT_PITCH_DEG is not None else float("nan")
            print(f"[multitask] phase '{old_phase}' -> '{new_phase}': head-cam "
                  f"pitch retargeted {cur:+.1f}° -> {pitch:+.1f}° (smooth)", flush=True)
        else:
            set_head_camera_pitch_absolute(env, pitch, axis=axis, verbose=True)

    return _hook


def set_head_camera_aperture(env: Any, horizontal_aperture_mm: float) -> None:
    """Widen (or narrow) the robot's onboard ZED camera FOV by setting its
    USD ``horizontalAperture`` attribute (in mm).

    With the default ``focal_length=17 mm``:
        * horizontal_aperture=20.995  -> HFOV ≈ 63°  (default, narrow)
        * horizontal_aperture=40.0    -> HFOV ≈ 99°  (BEHAVIOR-1K official wide)
        * horizontal_aperture=50.0    -> HFOV ≈ 111° (very wide)

    Like the camera pitch, the aperture is a USD-prim attribute that lives on
    the sensor's prim and is NOT part of the serialized scene state, so a
    one-time set here persists across ``og.sim.load_state`` replay calls.
    """
    if not horizontal_aperture_mm or horizontal_aperture_mm <= 0:
        return
    robot = env.robots[0]
    sensors = list(getattr(robot, "sensors", {}).items())
    sensors.sort(key=lambda kv: (0 if "zed" in kv[0].lower() else 1, kv[0]))
    target = None
    for name, sensor in sensors:
        if "zed" in name.lower() and hasattr(sensor, "horizontal_aperture"):
            target = (name, sensor)
            break
    if target is None:
        for name, sensor in sensors:
            if hasattr(sensor, "horizontal_aperture"):
                target = (name, sensor)
                break
    if target is None:
        print("[multitask] set_head_camera_aperture: no VisionSensor with horizontal_aperture found", flush=True)
        return

    name, sensor = target
    try:
        old_aperture = float(sensor.horizontal_aperture)
    except Exception:
        old_aperture = float("nan")
    try:
        focal = float(sensor.focal_length)
    except Exception:
        focal = 17.0
    sensor.horizontal_aperture = float(horizontal_aperture_mm)
    # Trigger a few render iterations so Replicator picks up the new aperture.
    for _ in range(3):
        og.sim.render()
    new_aperture = float(horizontal_aperture_mm)
    hfov_old = math.degrees(2.0 * math.atan(old_aperture / (2.0 * focal))) if focal > 0 else float("nan")
    hfov_new = math.degrees(2.0 * math.atan(new_aperture / (2.0 * focal))) if focal > 0 else float("nan")
    print(
        f"[multitask] set_head_camera_aperture: '{name}' horizontal_aperture "
        f"{old_aperture:.3f}mm -> {new_aperture:.3f}mm "
        f"(HFOV {hfov_old:.1f}° -> {hfov_new:.1f}° @ focal={focal:.2f}mm)",
        flush=True,
    )


def get_head_camera_intrinsics(env: Any) -> Dict[str, Any]:
    """Read static head-camera intrinsics from the live USD prim. Returns a
    JSON-serializable dict, suitable for embedding in HDF5 attrs (env_args).
    """
    found = _find_head_camera_sensor(env)
    if found is None:
        return {}
    name, sensor = found
    intr: Dict[str, Any] = {"sensor_name": name}
    for attr in ("focal_length", "horizontal_aperture", "focus_distance", "fstop"):
        try:
            intr[attr] = float(getattr(sensor, attr))
        except Exception:
            pass
    for attr in ("image_height", "image_width"):
        try:
            intr[attr] = int(getattr(sensor, attr))
        except Exception:
            pass
    fl = intr.get("focal_length", 0.0)
    ha = intr.get("horizontal_aperture", 0.0)
    if fl > 0 and ha > 0:
        intr["hfov_deg"] = math.degrees(2.0 * math.atan(ha / (2.0 * fl)))
        ih = intr.get("image_height", 0)
        iw = intr.get("image_width", 0)
        if ih > 0 and iw > 0:
            va = ha * (ih / iw)
            intr["vertical_aperture_mm"] = float(va)
            intr["vfov_deg"] = math.degrees(2.0 * math.atan(va / (2.0 * fl)))
    if _HEAD_CAM_REST_QUAT is not None:
        intr["rest_local_quat_xyzw"] = _HEAD_CAM_REST_QUAT.tolist()
    if _HEAD_CAM_REST_POS is not None:
        intr["rest_local_pos"] = _HEAD_CAM_REST_POS.tolist()
    return intr


def get_head_camera_state(env: Any) -> Dict[str, Any]:
    """Snapshot the head camera's per-frame extrinsics + current pitch. Cheap
    enough to call every recorded frame.

    Returns a dict with float / list-of-float entries (no torch tensors), or an
    empty dict if no head camera is found.
    """
    found = _find_head_camera_sensor(env)
    if found is None:
        return {}
    _, sensor = found
    state: Dict[str, Any] = {}
    # Current desired pitch (whatever step_head_camera_pitch_smooth set last).
    if _HEAD_CAM_CURRENT_PITCH_DEG is not None:
        state["pitch_deg"] = float(_HEAD_CAM_CURRENT_PITCH_DEG)
    if _HEAD_CAM_TARGET_PITCH_DEG is not None:
        state["target_pitch_deg"] = float(_HEAD_CAM_TARGET_PITCH_DEG)
    state["pitch_axis"] = _HEAD_CAM_PITCH_AXIS
    # Local pose (relative to zed_link).
    try:
        lpos, lquat = sensor.get_position_orientation(frame="parent")
        state["local_pos"] = to_numpy(lpos).astype(np.float32).tolist()
        state["local_quat"] = to_numpy(lquat).astype(np.float32).tolist()
    except Exception:
        pass
    # World pose.
    try:
        wpos, wquat = sensor.get_position_orientation()
        state["world_pos"] = to_numpy(wpos).astype(np.float32).tolist()
        state["world_quat"] = to_numpy(wquat).astype(np.float32).tolist()
    except Exception:
        pass
    return state


def peek_robot_pose_at_step(env: Any, episode_id: int, step: int
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Like peek_robot_position_at_step but also returns orientation (quaternion).
    Caller must wrap in dump_state/load_state to avoid side effects."""
    data_grp = env.input_hdf5["data"]
    traj_grp = data_grp[f"demo_{episode_id}"]
    state_raw = traj_grp["state"][step]
    state_size = int(traj_grp["state_size"][step])
    state_t = th.tensor(state_raw[:state_size])
    og.sim.load_state(state_t, serialized=True)
    pos = to_numpy(env.robots[0].get_position_orientation()[0]).copy()
    quat = to_numpy(env.robots[0].get_position_orientation()[1]).copy()
    return pos, quat


# --------------------------------------------------------------------------- #
# Layer 1: Pre-scan robot trajectory from HDF5
# --------------------------------------------------------------------------- #

def scan_robot_trajectory(
    env: Any,
    episode_id: int,
    start_step: int,
    end_step: int,
    num_samples: int = 15,
) -> List[np.ndarray]:
    """Sample robot XY positions from HDF5 at evenly-spaced intervals.

    Uses dump_state / load_state to snapshot-restore the full sim, so no
    side-effects on the running scene.
    """
    snapshot = og.sim.dump_state(serialized=True)
    positions: List[np.ndarray] = []
    try:
        data_grp = env.input_hdf5["data"]
        traj_grp = data_grp[f"demo_{episode_id}"]
        steps = np.linspace(start_step, max(end_step - 1, start_step), num_samples, dtype=int)
        steps = list(dict.fromkeys(steps))
        for s in steps:
            state_raw = traj_grp["state"][s]
            state_sz = int(traj_grp["state_size"][s])
            import torch as th
            state_t = th.tensor(state_raw[:state_sz])
            og.sim.load_state(state_t, serialized=True)
            pos = to_numpy(env.robots[0].get_position_orientation()[0]).astype(np.float32)
            positions.append(pos[:2].copy())
    except Exception as exc:
        print(f"[trajectory-scan] WARN: {exc}", flush=True)
    finally:
        og.sim.load_state(snapshot, serialized=True)
    print(f"[trajectory-scan] sampled {len(positions)} robot positions "
          f"from steps [{start_step},{end_step})", flush=True)
    return positions


def scan_robot_trajectory_external(
    ext_hdf5_path: str,
    episode_id: int,
    start_step: int,
    end_step: int,
    env: Any,
    num_samples: int = 15,
) -> List[np.ndarray]:
    """Same as scan_robot_trajectory but reads from an external HDF5."""
    snapshot = og.sim.dump_state(serialized=True)
    positions: List[np.ndarray] = []
    try:
        with h5py.File(ext_hdf5_path, "r") as f:
            traj = f[f"data/demo_{episode_id}"]
            state = traj["state"]
            state_size = traj["state_size"]
            n_steps = state.shape[0]
            actual_end = min(end_step, n_steps) if end_step is not None else n_steps
            steps = np.linspace(start_step, max(actual_end - 1, start_step),
                                num_samples, dtype=int)
            steps = list(dict.fromkeys(steps))
            for s in steps:
                import torch as th
                state_t = th.tensor(state[s, :int(state_size[s])], dtype=th.float32)
                og.sim.load_state(state_t, serialized=True)
                pos = to_numpy(env.robots[0].get_position_orientation()[0]).astype(np.float32)
                positions.append(pos[:2].copy())
    except Exception as exc:
        print(f"[trajectory-scan-ext] WARN: {exc}", flush=True)
    finally:
        og.sim.load_state(snapshot, serialized=True)
    print(f"[trajectory-scan-ext] sampled {len(positions)} robot positions "
          f"from {os.path.basename(ext_hdf5_path)} [{start_step},{end_step})", flush=True)
    return positions


# --------------------------------------------------------------------------- #
# Layer 2: Find safe human position outside the trajectory envelope
# --------------------------------------------------------------------------- #

def find_safe_human_position(
    human_actor: Any,
    trajectory_points: List[np.ndarray],
    margin: float = 1.5,
    max_dist_from_task: float = 4.0,
    max_attempts: int = 100,
) -> Optional[np.ndarray]:
    """Find a navigable point outside the robot's trajectory envelope but
    still close to the task area (trajectory centroid).

    Constraints:
      - At least ``margin`` away from every sampled trajectory point
      - At most ``max_dist_from_task`` from the trajectory centroid
      - On the traversable map with a valid A* path
    """
    if not trajectory_points:
        return None
    trav_map = human_actor._try_get_trav_map()
    if trav_map is None:
        return None
    traj_arr = np.array(trajectory_points, dtype=np.float32)
    traj_center = np.mean(traj_arr, axis=0)
    human_start = human_actor.base_pos.copy()
    z = float(human_start[2])
    ref_pt = np.array([traj_center[0], traj_center[1], z], dtype=np.float32)

    best: Optional[np.ndarray] = None
    best_min_traj_dist = -1.0

    for _ in range(max_attempts):
        try:
            _, rand_pt = trav_map.get_random_point(
                floor=0,
                reference_point=th.tensor(ref_pt, dtype=th.float32),
            )
            candidate = np.array([float(rand_pt[0]), float(rand_pt[1])], dtype=np.float32)
            dist_from_task = float(np.linalg.norm(candidate - traj_center))
            if dist_from_task > max_dist_from_task:
                continue
            dists = np.linalg.norm(traj_arr - candidate[None, :], axis=1)
            min_traj_dist = float(dists.min())
            if min_traj_dist >= margin:
                path_test = human_actor._nav_path_between(
                    human_start,
                    np.array([candidate[0], candidate[1], z], dtype=np.float32),
                    z, trav_map, 0,
                )
                if len(path_test) >= 2:
                    print(f"[safe-pos] found position ({candidate[0]:.2f}, {candidate[1]:.2f}) "
                          f"min_traj_dist={min_traj_dist:.2f}m, dist_from_task={dist_from_task:.2f}m",
                          flush=True)
                    return np.array([candidate[0], candidate[1], z], dtype=np.float32)
            elif best is None or min_traj_dist > best_min_traj_dist:
                if dist_from_task <= max_dist_from_task:
                    best = np.array([candidate[0], candidate[1], z], dtype=np.float32)
                    best_min_traj_dist = min_traj_dist
        except Exception:
            pass

    if best is not None and best_min_traj_dist > margin * 0.4:
        print(f"[safe-pos] using best-effort position (min_traj_dist={best_min_traj_dist:.2f}m)", flush=True)
        return best
    return None


def move_human_to_safe_position(
    human_actor: Any,
    safe_pos: np.ndarray,
    face_target: Optional[np.ndarray] = None,
) -> None:
    """Teleport the human to safe_pos and orient them toward face_target."""
    human_actor.base_pos = safe_pos.copy()
    if face_target is not None:
        dx = float(face_target[0] - safe_pos[0])
        dy = float(face_target[1] - safe_pos[1])
        if abs(dx) + abs(dy) > 0.01:
            human_actor.base_yaw = math.atan2(dy, dx)
    human_actor.motion_state = "hold"
    human_actor.linear_speed = 0.0
    human_actor.apply_pose()
    human_actor.anchor_pos = safe_pos.copy()
    print(f"[safe-pos] human moved to ({safe_pos[0]:.2f}, {safe_pos[1]:.2f})", flush=True)


# --------------------------------------------------------------------------- #
# Layer 3: Real-time dodge during replay
# --------------------------------------------------------------------------- #

def dodge_robot_if_needed(
    human_actor: Any,
    robot: Any,
    min_dist: float = 1.0,
    dodge_step: float = 0.08,
) -> None:
    """If robot is within min_dist of the human, nudge human away smoothly."""
    robot_xy = to_numpy(robot.get_position_orientation()[0]).astype(np.float32)[:2]
    human_xy = human_actor.base_pos[:2].astype(np.float32)
    diff = human_xy - robot_xy
    dist = float(np.linalg.norm(diff))
    if dist >= min_dist:
        return
    if dist < 1e-4:
        direction = np.array([1.0, 0.0], dtype=np.float32)
    else:
        direction = diff / dist
    new_xy = human_xy + direction * dodge_step
    new_pos = human_actor.base_pos.copy()
    new_pos[0] = float(new_xy[0])
    new_pos[1] = float(new_xy[1])
    human_actor.base_pos = new_pos
    human_actor.apply_pose()


# --------------------------------------------------------------------------- #
# Position-pinning version of run_zero_phase
# --------------------------------------------------------------------------- #

_pinned_scene_objects: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

# World AABBs (lo_xyz, hi_xyz) of free-standing "go-around" furniture (tables,
# desks...) whose thin legs leave the floor under them "navigable" in the 2-D
# trav_map, so plain A* would route the robot/human straight through the top.
# Populated once in main(); consumed by _plan_robot_path and the relocate walk to
# stamp these footprints as obstacles during A* (ARCHITECTURE.md §16). Task
# surfaces (counters/stove/fridge) are deliberately NOT included so the robot can
# still approach them.
_NAV_OBSTACLE_AABBS: List[Tuple[np.ndarray, np.ndarray]] = []
# Categories the robot navigates AROUND (not TO) — free-standing furniture with
# thin legs / open base that the 2-D trav_map under-counts. Substring-matched, so
# "breakfast_table"/"coffee_table"/"side_table"/... qualify via "table". NOTE we
# deliberately do NOT include the primary task surfaces ("countertop", "stove",
# "oven", "fridge", "sink", cabinets) — the robot must be able to approach those.
# "bar"/"island" are free-standing kitchen islands the robot only walks around.
_NAV_AROUND_SUBSTR = ("table", "desk", "ottoman", "bench", "piano", "pool_table",
                      "bar", "island")


def _collect_nav_obstacles(env: Any) -> None:
    """Snapshot go-around furniture AABBs into _NAV_OBSTACLE_AABBS (once)."""
    _NAV_OBSTACLE_AABBS.clear()
    for obj in env.scene.objects:
        cat = (getattr(obj, "category", "") or "").lower()
        if not any(s in cat for s in _NAV_AROUND_SUBSTR):
            continue
        try:
            lo, hi = obj.aabb
            lo = to_numpy(lo).astype(np.float32)
            hi = to_numpy(hi).astype(np.float32)
            _NAV_OBSTACLE_AABBS.append((lo, hi))
            print(f"  [nav-obstacle] {getattr(obj,'name','?')} cat={cat} "
                  f"xy=[{lo[0]:.2f},{lo[1]:.2f}]..[{hi[0]:.2f},{hi[1]:.2f}]", flush=True)
        except Exception:
            pass
    print(f"[multitask] Collected {len(_NAV_OBSTACLE_AABBS)} go-around furniture "
          f"footprints for robot/human path routing", flush=True)


def _restore_pinned_objects(env: Any) -> None:
    """Re-set pinned scene objects (baskets etc.) to their captured poses."""
    for name, (pos, quat) in _pinned_scene_objects.items():
        obj = env.scene.object_registry("name", name)
        if obj is not None:
            obj.set_position_orientation(pos, quat)
            obj.keep_still()


def run_zero_phase_pinned(env: Any, recorder: DemoRecorder, phase: str, steps: int,
                          round_id: int, human_actor: Any = None) -> None:
    """Like run_zero_phase but re-sets robot position every frame so physics
    can't drift it away during wait phases."""
    robot = env.robots[0]
    zero_action = np.zeros(robot.action_dim, dtype=np.float32)
    pin_pos = to_numpy(robot.get_position_orientation()[0]).copy()
    pin_quat = to_numpy(robot.get_position_orientation()[1]).copy()
    for _ in range(max(steps, 1)):
        robot.set_position_orientation(pin_pos, pin_quat)
        robot.keep_still()
        _restore_pinned_objects(env)
        if human_actor is not None:
            human_actor.step(phase, robot)
        obs, rew, term, trunc, info = env.step(zero_action, n_render_iterations=1)
        info["reward"] = float(rew) if rew is not None else 0.0
        info["terminated"] = bool(term) if term is not None else False
        info["truncated"] = bool(trunc) if trunc is not None else False
        recorder.record(phase, obs, info, round_id=round_id)


# --------------------------------------------------------------------------- #
# Path interpolation helpers (used by run_relocation_phase)
# --------------------------------------------------------------------------- #

def _interp_path(path: List[np.ndarray], cum_dist: List[float], d: float) -> np.ndarray:
    """Interpolate a 3-D position at cumulative distance *d* along a polyline."""
    if d <= 0.0:
        return path[0].copy()
    if d >= cum_dist[-1]:
        return path[-1].copy()
    for i in range(1, len(cum_dist)):
        if cum_dist[i] >= d:
            seg_len = cum_dist[i] - cum_dist[i - 1]
            t = (d - cum_dist[i - 1]) / max(seg_len, 1e-6)
            return path[i - 1] + (path[i] - path[i - 1]) * t
    return path[-1].copy()


def _yaw_at_dist(path: List[np.ndarray], cum_dist: List[float], d: float) -> float:
    """Heading direction (yaw) at cumulative distance *d* along a polyline."""
    for i in range(1, len(cum_dist)):
        if cum_dist[i] >= d:
            delta = path[i][:2] - path[i - 1][:2]
            norm = float(np.linalg.norm(delta))
            if norm > 1e-6:
                return math.atan2(float(delta[1]), float(delta[0]))
            break
    if len(path) >= 2:
        delta = path[-1][:2] - path[-2][:2]
        return math.atan2(float(delta[1]), float(delta[0]))
    return 0.0


# --------------------------------------------------------------------------- #
# Scripted radio-toggle phase
# --------------------------------------------------------------------------- #

def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------- #
# Smooth-turning phase functions (local copies of hrc_minimal_demo's that
# spread the rotation over a much larger fraction of the phase, so the robot
# doesn't snap-turn in 5 frames).
# --------------------------------------------------------------------------- #

def send_human_home(human_actor: Any, fps: int, return_steps: int,
                    phase: str = "ReturnToHuman",
                    min_speed: float = 0.4, max_speed: float = 1.5,
                    margin: float = 0.85) -> None:
    """Tell the human to walk back to its spawn anchor, timed so that it
    arrives at spawn at approximately the same time as the robot finishes its
    ReturnToHuman phase.

    Implementation:
      1. Install a single-destination ``waypoint_once`` path ``[spawn_anchor]``
         for the given phase. A* expansion turns this into a navigable polyline
         that routes around furniture / through doorways.
      2. Read the expanded polyline's total length and set the human's walking
         speed so that ``length / speed`` matches ``return_steps / fps`` times
         ``margin`` (slight lead so the human is *already* at spawn when the
         robot arrives, rather than trailing the robot into frame).
      3. Speed is clamped to ``[min_speed, max_speed]`` so we don't get
         teleport-slow or teleport-fast motion if the path length is extreme.
    """
    if human_actor is None:
        return

    spawn_xyz = human_actor.anchor_pos.astype(np.float32).copy()
    cur_xy = human_actor.base_pos.astype(np.float32)[:2]
    spawn_xy = spawn_xyz[:2]

    # If we're already essentially at spawn, just switch to a hold phase.
    if float(np.linalg.norm(cur_xy - spawn_xy)) < 0.05:
        human_actor.active_phase = phase
        human_actor.phase_step = 0
        human_actor.phase_waypoint_idx = 0
        human_actor.motion_state = "hold"
        print(f"[send_home] already at spawn, just holding pose", flush=True)
        return

    # 1. Install the walk-home waypoint path (A* expanded).
    human_actor.set_waypoints(
        [[float(spawn_xyz[0]), float(spawn_xyz[1]), float(spawn_xyz[2])]],
        mode="waypoint_once",
        phase=phase,
        loop=False,
        expand_with_astar=True,
        face_robot=True,  # once home, turn to face the robot coming in
    )

    # Force the actor into this phase now (set_waypoints resolves waypoints for
    # the named phase but doesn't switch active_phase if it's currently something
    # else like ExecuteBehaviorTask).
    human_actor.active_phase = phase
    human_actor.phase_step = 0
    human_actor.phase_waypoint_idx = 0

    # 2. Compute required speed from expanded polyline length.
    pts: List[np.ndarray] = list(human_actor.phase_waypoints)
    path_nodes = [cur_xy] + [np.asarray(p, dtype=np.float32)[:2] for p in pts]
    total_len = 0.0
    for i in range(len(path_nodes) - 1):
        total_len += float(np.linalg.norm(path_nodes[i + 1] - path_nodes[i]))
    if total_len < 1e-4:
        print("[send_home] path length ~0, nothing to walk", flush=True)
        return

    target_time = max(return_steps, 1) / max(float(fps), 1.0)
    required_speed = float(total_len / max(target_time * margin, 1e-3))
    clamped = float(max(min_speed, min(required_speed, max_speed)))

    spec = human_actor.phase_motion.get(phase)
    if spec is not None:
        spec.speed = clamped
    print(
        f"[send_home] path_len={total_len:.2f} m, "
        f"target_time={target_time:.2f} s (margin {margin:.2f}), "
        f"required_speed={required_speed:.2f} m/s -> "
        f"set speed={clamped:.2f} m/s (n_waypoints={len(pts)})",
        flush=True,
    )


def _simplify_path(path: List[np.ndarray], epsilon: float = 0.15) -> List[np.ndarray]:
    """Ramer-Douglas-Peucker path simplification.

    Collapses grid-staircase A* paths into a few straight segments by
    removing points within *epsilon* metres of the simplified line.
    """
    if len(path) <= 2:
        return path
    start = path[0][:2].astype(np.float64)
    end = path[-1][:2].astype(np.float64)
    line_vec = end - start
    line_len = float(np.linalg.norm(line_vec))
    if line_len < 1e-6:
        return [path[0], path[-1]]
    line_dir = line_vec / line_len

    max_dist = 0.0
    max_idx = 0
    for i in range(1, len(path) - 1):
        pt = path[i][:2].astype(np.float64)
        t = float(np.dot(pt - start, line_dir))
        t = max(0.0, min(line_len, t))
        proj = start + line_dir * t
        dist = float(np.linalg.norm(pt - proj))
        if dist > max_dist:
            max_dist = dist
            max_idx = i

    if max_dist > epsilon:
        left = _simplify_path(path[: max_idx + 1], epsilon)
        right = _simplify_path(path[max_idx:], epsilon)
        return left[:-1] + right
    else:
        return [path[0], path[-1]]


def _deflect_path_around_point(
    path: List[np.ndarray],
    avoid_xy: np.ndarray,
    clearance: float = 0.6,
) -> List[np.ndarray]:
    """Post-process a polyline so no segment passes within *clearance* of
    *avoid_xy*.  Inserts a single detour waypoint pushed away from the obstacle.
    """
    if len(path) < 2:
        return path
    avoid_2d = np.asarray(avoid_xy, dtype=np.float32)[:2]
    result: List[np.ndarray] = [path[0]]
    for i in range(1, len(path)):
        a = path[i - 1][:2].astype(np.float32)
        b = path[i][:2].astype(np.float32)
        z = float(path[i][2])
        ab = b - a
        seg_len_sq = float(np.dot(ab, ab))
        if seg_len_sq < 1e-12:
            result.append(path[i])
            continue
        t = float(np.dot(avoid_2d - a, ab)) / seg_len_sq
        t = max(0.0, min(1.0, t))
        closest = a + ab * t
        dist = float(np.linalg.norm(closest - avoid_2d))
        if dist < clearance:
            human_to_seg = closest - avoid_2d
            h2s_norm = float(np.linalg.norm(human_to_seg))
            if h2s_norm > 1e-6:
                away = human_to_seg / h2s_norm
            else:
                seg_dir = ab / math.sqrt(seg_len_sq)
                away = np.array([-seg_dir[1], seg_dir[0]], dtype=np.float32)
            detour_2d = avoid_2d + away * clearance * 2.0
            result.append(np.array([detour_2d[0], detour_2d[1], z], dtype=np.float32))
        result.append(path[i])
    return result


def _smooth_path(path: List[np.ndarray], iterations: int = 3) -> List[np.ndarray]:
    """Chaikin corner-cutting: turns sharp V-shaped detours into smooth curves
    while preserving start and end points."""
    if len(path) <= 2:
        return path
    pts = path
    for _ in range(iterations):
        if len(pts) <= 2:
            break
        smoothed: List[np.ndarray] = [pts[0]]
        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            smoothed.append(q)
            smoothed.append(r)
        smoothed.append(pts[-1])
        pts = smoothed
    return pts


def _plan_robot_path(human_actor: Any, src_xyz: np.ndarray, dst_xy: np.ndarray,
                     z: float, avoid_objects: Optional[List[Any]] = None) -> List[np.ndarray]:
    """Plan an A* path for the robot, then deflect around the human and any
    extra scene objects (baskets, etc.) that aren't in the trav_map.
    """
    if human_actor is None:
        return [np.array([float(dst_xy[0]), float(dst_xy[1]), z], dtype=np.float32)]
    trav_map = human_actor._try_get_trav_map()
    if trav_map is None:
        print("[robot-plan] no trav_map; straight-line fallback", flush=True)
        return [np.array([float(dst_xy[0]), float(dst_xy[1]), z], dtype=np.float32)]
    src_xy = np.asarray(src_xyz, dtype=np.float32)[:2]
    dst_xy_f = np.asarray(dst_xy, dtype=np.float32)[:2]
    # Route around free-standing furniture (tables) that the 2-D trav_map misses.
    # pad ≈ robot body radius so the path keeps real clearance (the trav_map
    # erosion alone under-delivers — it left only ~0.07 m, well inside the base).
    path = human_actor._nav_path_avoiding(src_xy, dst_xy_f, float(z), trav_map,
                                          _NAV_OBSTACLE_AABBS, floor=0, pad=0.35)

    if len(path) < 2:
        # Direct A* found no route. Instead of repeating many (slow) full-map
        # failed-A* searches, compute the robot's reachable region ONCE and route
        # only to a REACHABLE target — A* then always succeeds (fast) and we never
        # straight-line through obstacles (ARCHITECTURE.md §18).
        print("[robot-plan] direct A* failed; computing reachable region", flush=True)
        try:
            reachable, nearest_reachable = human_actor._build_reachability(
                src_xy, trav_map, _NAV_OBSTACLE_AABBS, pad=0.35)
        except Exception as exc:
            reachable, nearest_reachable = (lambda xy: False), (lambda xy: None)
            print(f"[robot-plan] reachability build failed ({exc})", flush=True)
        target = dst_xy_f
        if not reachable(dst_xy_f):
            alt = nearest_reachable(dst_xy_f)
            if alt is not None:
                print(f"[robot-plan] target ({dst_xy_f[0]:.2f},{dst_xy_f[1]:.2f}) unreachable; "
                      f"routing to nearest reachable ({alt[0]:.2f},{alt[1]:.2f}) "
                      f"(no straight-line through obstacles)", flush=True)
                target = np.asarray(alt, dtype=np.float32)
        # one A* to the (reachable) target, snapping the start if needed
        src3 = np.array([src_xy[0], src_xy[1], z], dtype=np.float32)
        starts = [src_xy]
        ss = nearest_reachable(src_xy)
        if ss is not None and float(np.linalg.norm(np.asarray(ss) - src_xy)) > 0.05:
            starts.append(np.asarray(ss, dtype=np.float32))
        for s in starts:
            trial = human_actor._nav_path_avoiding(s, target, float(z), trav_map,
                                                   _NAV_OBSTACLE_AABBS, floor=0, pad=0.35)
            if len(trial) >= 2:
                path = list(trial) if np.allclose(s, src_xy) else [src3] + list(trial)
                break
        if len(path) < 2:
            # Nowhere reachable to go — hold position rather than walk through walls.
            print("[robot-plan] no reachable path; robot holds position", flush=True)
            path = [src3]

    raw_count = len(path)
    path = _simplify_path(path)
    path = _deflect_path_around_point(path, human_actor.base_pos[:2], clearance=0.4)
    if avoid_objects:
        for obj in avoid_objects:
            try:
                obj_pos = to_numpy(obj.get_position_orientation()[0]).astype(np.float32)[:2]
                cat_r = (getattr(obj, "category", "") or "").lower()
                is_large = "chair" in cat_r or "sofa" in cat_r or "stool" in cat_r
                is_basket = "basket" in cat_r or "wicker" in cat_r
                cl = 0.5 if is_large else (0.45 if is_basket else 0.25)
                path = _deflect_path_around_point(path, obj_pos, clearance=cl)
            except Exception:
                pass
    if len(path) > 2:
        path = _smooth_path(path, iterations=1)
    print(f"[robot-plan] A* {src_xy.tolist()} -> {np.asarray(dst_xy)[:2].tolist()}: "
          f"{raw_count} -> {len(path)} nav points (simplified+deflected+smoothed)", flush=True)
    return path


def _walk_robot_along_path(env: Any, recorder: DemoRecorder, human: Any, total_steps: int,
                           round_id: int, phase_label: str, planned_path: List[np.ndarray],
                           human_actor: Any = None, turn_fraction: float = 0.5,
                           track_human_yaw: bool = False) -> None:
    """Walk the robot along ``planned_path`` using cumulative-distance
    interpolation so motion is smooth regardless of waypoint count.
    """
    robot = env.robots[0]
    zero_action = np.zeros(robot.action_dim, dtype=np.float32)
    cur_pos = to_numpy(robot.get_position_orientation()[0]).astype(np.float32)
    start_yaw = _yaw_from_quat(to_numpy(robot.get_position_orientation()[1]))

    nodes: List[np.ndarray] = [cur_pos]
    for p in planned_path:
        nodes.append(np.asarray(p, dtype=np.float32))
    deduped: List[np.ndarray] = [nodes[0]]
    for p in nodes[1:]:
        if float(np.linalg.norm(p[:2] - deduped[-1][:2])) > 1e-3:
            deduped.append(p)
    nodes = deduped

    if len(nodes) < 2:
        for _ in range(max(total_steps, 1)):
            obs, rew, term, trunc, info = env.step(zero_action, n_render_iterations=1)
            info["reward"] = float(rew) if rew is not None else 0.0
            info["terminated"] = bool(term) if term is not None else False
            info["truncated"] = bool(trunc) if trunc is not None else False
            if human_actor is not None:
                human_actor.step(phase_label, robot)
            recorder.record(phase_label, obs, info, round_id=round_id)
        return

    cum_dist = [0.0]
    for i in range(1, len(nodes)):
        cum_dist.append(cum_dist[-1] + float(np.linalg.norm(nodes[i][:2] - nodes[i - 1][:2])))
    total_len = cum_dist[-1]
    if total_len < 1e-6:
        return

    n_total = max(total_steps, 1)
    turn_frames = max(int(n_total * float(turn_fraction)), 1)
    prev_yaw = start_yaw

    for frame_idx in range(n_total):
        _restore_pinned_objects(env)
        t = (frame_idx + 1) / n_total
        d = t * total_len
        pos = _interp_path(nodes, cum_dist, d)

        if track_human_yaw and human_actor is not None:
            live_human = to_numpy(human.get_position_orientation()[0])
            target_yaw = math.atan2(float(live_human[1] - pos[1]),
                                    float(live_human[0] - pos[0]))
        else:
            target_yaw = _yaw_at_dist(nodes, cum_dist, d)

        if frame_idx < turn_frames:
            alpha = _smoothstep((frame_idx + 1) / turn_frames)
            yaw = start_yaw + _wrap_to_pi(target_yaw - start_yaw) * alpha
        else:
            yaw = prev_yaw + 0.25 * _wrap_to_pi(target_yaw - prev_yaw)
        prev_yaw = yaw

        robot.set_position_orientation(pos.astype(np.float32),
                                       quat_from_yaw(float(yaw)))
        if human_actor is not None:
            human_actor.step(phase_label, robot)
        obs, rew, term, trunc, info = env.step(zero_action, n_render_iterations=1)
        info["reward"] = float(rew) if rew is not None else 0.0
        info["terminated"] = bool(term) if term is not None else False
        info["truncated"] = bool(trunc) if trunc is not None else False
        recorder.record(phase_label, obs, info, round_id=round_id)


def run_go_to_task_phase_smooth(env: Any, recorder: DemoRecorder, human: Any, steps: int,
                                round_id: int, target_pos: np.ndarray,
                                human_actor: Any = None, turn_fraction: float = 0.5,
                                avoid_objects: Optional[List[Any]] = None) -> None:
    """A*-planned go-to-task with obstacle avoidance for floor objects."""
    robot = env.robots[0]
    robot_pos = to_numpy(robot.get_position_orientation()[0]).astype(np.float32)
    z = float(robot_pos[2])
    planned = _plan_robot_path(human_actor, robot_pos,
                               np.asarray(target_pos, dtype=np.float32)[:2], z=z,
                               avoid_objects=avoid_objects)
    exact_end = np.array([float(target_pos[0]), float(target_pos[1]), z], dtype=np.float32)
    if len(planned) > 0:
        planned[-1] = exact_end
    else:
        planned = [exact_end]
    _walk_robot_along_path(env, recorder, human, steps, round_id, "GoToTask",
                           planned_path=planned, human_actor=human_actor,
                           turn_fraction=turn_fraction, track_human_yaw=False)


def _find_clear_standoff(human_xy: np.ndarray, preferred_dir: np.ndarray,
                         stand_off: float, avoid_objects: Optional[List[Any]],
                         min_clearance: float = 0.55,
                         human_actor: Any = None) -> np.ndarray:
    """Find a stand-off position around the human that the robot can actually
    stand at: clear of avoid_objects, ON navigable floor (rejects counters/walls,
    which are non-navigable), and clear of go-around furniture footprints (tables).
    Tries the preferred direction first, then rotates; also tries a few distances.
    (Reachability from the robot is then guaranteed by _plan_robot_path's
    nearest-reachable fallback — ARCHITECTURE.md §18.)"""
    base_angle = math.atan2(float(preferred_dir[1]), float(preferred_dir[0]))
    trav_map = human_actor._try_get_trav_map() if human_actor is not None else None

    def _navigable(c):
        if trav_map is None:
            return True
        try:
            snap = human_actor._snap_to_navigable(np.asarray(c, dtype=np.float32), trav_map, 0)
            if snap is None:
                return False
            s = to_numpy(snap) if not isinstance(snap, np.ndarray) else snap
            return float(np.hypot(float(s[0]) - float(c[0]), float(s[1]) - float(c[1]))) < 0.12
        except Exception:
            return True

    def _furniture_clear(c):
        for lo, hi in _NAV_OBSTACLE_AABBS:
            if (lo[0] - 0.30 <= c[0] <= hi[0] + 0.30) and (lo[1] - 0.30 <= c[1] <= hi[1] + 0.30):
                return False
        return True

    for dist in (float(stand_off), float(stand_off) + 0.4, float(stand_off) + 0.8,
                 max(float(stand_off) - 0.25, 0.7)):
        for delta_deg in [0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180]:
            angle = base_angle + math.radians(delta_deg)
            cand = human_xy + np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * dist
            if avoid_objects:
                if any(
                    _safe_obj_dist(obj, cand) < min_clearance for obj in avoid_objects
                ):
                    continue
            if not _navigable(cand) or not _furniture_clear(cand):
                continue
            if delta_deg != 0 or abs(dist - stand_off) > 1e-3:
                print(f"[standoff] adjusted (Δ{delta_deg}°, d={dist:.2f}) -> "
                      f"({cand[0]:.2f},{cand[1]:.2f}) navigable+furniture-clear", flush=True)
            return cand
    return human_xy + preferred_dir * float(stand_off)


def _safe_obj_dist(obj: Any, cand: np.ndarray) -> float:
    try:
        obj_pos = to_numpy(obj.get_position_orientation()[0]).astype(np.float32)[:2]
        return float(np.linalg.norm(cand - obj_pos))
    except Exception:
        return 1e9


def run_return_phase_smooth(env: Any, recorder: DemoRecorder, human: Any, steps: int,
                            round_id: int, stand_off: float,
                            human_actor: Any = None, turn_fraction: float = 0.5,
                            avoid_objects: Optional[List[Any]] = None) -> None:
    """A*-planned return-to-human with obstacle avoidance."""
    robot = env.robots[0]
    robot_pos = to_numpy(robot.get_position_orientation()[0]).astype(np.float32)
    z = float(robot_pos[2])

    if human_actor is not None:
        human_target_xy = human_actor.base_pos.astype(np.float32)[:2]
    else:
        human_target_xy = to_numpy(human.get_position_orientation()[0]).astype(np.float32)[:2]
    direction = robot_pos[:2] - human_target_xy
    norm = float(np.linalg.norm(direction))
    direction = direction / norm if norm > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)
    target_xy = _find_clear_standoff(human_target_xy, direction, stand_off,
                                     avoid_objects, min_clearance=0.55,
                                     human_actor=human_actor)

    planned = _plan_robot_path(human_actor, robot_pos, target_xy, z=z,
                               avoid_objects=avoid_objects)
    _walk_robot_along_path(env, recorder, human, steps, round_id, "ReturnToHuman",
                           planned_path=planned, human_actor=human_actor,
                           turn_fraction=turn_fraction, track_human_yaw=True)


def run_face_human_phase_smooth(env: Any, recorder: DemoRecorder, human: Any, steps: int,
                                round_id: int, human_actor: Any = None) -> None:
    robot = env.robots[0]
    robot_pos, robot_quat = robot.get_position_orientation()
    robot_pos_np = to_numpy(robot_pos)
    # Freeze the base pose (incl. z) for the whole turn-in-place; re-reading the
    # post-physics position each frame let the base drift UPWARD ("floating" when
    # the robot stops next to the human). Pin it like run_zero_phase_pinned does.
    hold_pos = robot_pos_np.astype(np.float32).copy()
    start_yaw = _yaw_from_quat(to_numpy(robot_quat))
    human_pos_np = to_numpy(human.get_position_orientation()[0])
    target_yaw = math.atan2(float(human_pos_np[1] - robot_pos_np[1]),
                            float(human_pos_np[0] - robot_pos_np[0]))
    zero_action = np.zeros(robot.action_dim, dtype=np.float32)
    n = max(steps, 1)
    for i in range(n):
        _restore_pinned_objects(env)
        alpha = _smoothstep((i + 1) / n)
        yaw = start_yaw + _wrap_to_pi(target_yaw - start_yaw) * alpha
        robot.set_position_orientation(hold_pos, quat_from_yaw(float(yaw)))
        robot.keep_still()
        if human_actor is not None:
            human_actor.step("FaceHuman", robot)
        obs, rew, term, trunc, info = env.step(zero_action, n_render_iterations=1)
        info["reward"] = float(rew) if rew is not None else 0.0
        info["terminated"] = bool(term) if term is not None else False
        info["truncated"] = bool(trunc) if trunc is not None else False
        recorder.record("FaceHuman", obs, info, round_id=round_id)


def run_face_yaw_phase(env: Any, recorder: DemoRecorder,
                       pin_pos: np.ndarray, target_yaw: float,
                       steps: int, round_id: int,
                       human_actor: Any = None) -> None:
    """Smoothly rotate the robot from its current yaw to *target_yaw* while
    keeping it pinned at *pin_pos*. Used to align with the replay orientation."""
    robot = env.robots[0]
    start_yaw = _yaw_from_quat(to_numpy(robot.get_position_orientation()[1]))
    zero_action = np.zeros(robot.action_dim, dtype=np.float32)
    n = max(steps, 1)
    pos = np.asarray(pin_pos, dtype=np.float32)
    for i in range(n):
        _restore_pinned_objects(env)
        alpha = _smoothstep((i + 1) / n)
        yaw = start_yaw + _wrap_to_pi(target_yaw - start_yaw) * alpha
        robot.set_position_orientation(pos, quat_from_yaw(float(yaw)))
        robot.keep_still()
        if human_actor is not None:
            human_actor.step("GoToTask", robot)
        obs, rew, term, trunc, info = env.step(zero_action, n_render_iterations=1)
        info["reward"] = float(rew) if rew is not None else 0.0
        info["terminated"] = bool(term) if term is not None else False
        info["truncated"] = bool(trunc) if trunc is not None else False
        recorder.record("GoToTask", obs, info, round_id=round_id)


# --------------------------------------------------------------------------- #
# Between-round relocation: human walks to a new position, robot follows
# --------------------------------------------------------------------------- #

def _find_staircase_xy(env: Any) -> Optional[np.ndarray]:
    """Return (x, y) of the first staircase in the scene, or None."""
    for obj in env.scene.objects:
        cat = (getattr(obj, "category", "") or "").lower()
        if "stair" in cat:
            pos = to_numpy(obj.get_position_orientation()[0])
            print(f"[relocation] found staircase {obj.name} at ({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})", flush=True)
            return np.array([float(pos[0]), float(pos[1])], dtype=np.float32)
    return None


def run_relocation_phase(
    env: Any,
    recorder: DemoRecorder,
    human_actor: Any,
    total_steps: int,
    round_id: int,
    stand_off: float = 1.0,
    min_dist: float = 1.0,
    max_dist: float = 5.0,
    task_center: Optional[np.ndarray] = None,
    walk_speed: float = 1.0,
    fps: int = 30,
    avoid_objects: Optional[List[Any]] = None,
    toward_staircase: bool = False,
) -> None:
    """Human walks a short distance; robot follows behind at realistic speed.

    When ``toward_staircase`` is True the human walks toward the scene's
    staircase (or a random navigable point near it) instead of a fully
    random destination, producing a natural directed walk.
    """
    if human_actor is None:
        return

    robot = env.robots[0]
    zero_action = np.zeros(robot.action_dim, dtype=np.float32)

    human_start = human_actor.base_pos.copy()
    robot_start = to_numpy(robot.get_position_orientation()[0]).astype(np.float32)
    z_human = float(human_start[2])
    z_robot = float(robot_start[2])

    trav_map = human_actor._try_get_trav_map()
    if trav_map is None:
        print("[relocation] no trav_map; skipping", flush=True)
        return

    task_xy = None
    if task_center is not None:
        task_xy = np.asarray(task_center, dtype=np.float32)[:2]

    stair_xy = _find_staircase_xy(env) if toward_staircase else None

    # Snapshot wall/furniture AABBs once so we can reject any destination whose
    # HELD pose or whose A*-routed WALK PATH would clip a wall (ARCHITECTURE.md
    # §16). Routing alone is not enough: the 0-erosion trav_map + path corner-
    # cutting let the human graze walls at doorways, so we validate the real
    # body box along the whole path against geometry.
    reloc_aabbs = human_actor._collect_obstacle_aabbs(human_start[:2].astype(np.float32), max_range=12.0)
    # Walls/structure subset for the WALK path: floor-standing furniture (counters,
    # fridge, sofas...) is already avoided by the A* trav_map routing, so for the
    # transient walk we only forbid crossing walls/railings (the actual complaint).
    # The HELD destination, by contrast, is validated against ALL geometry below.
    wall_aabbs = [a for a in reloc_aabbs if ("wall" in a[1] or a[1] in ("railing", "fence", "baseboard"))]
    body_r, body_h, base_lift = 0.33, 1.80, 0.10

    def _spot_clear(xy, min_clear, aabbs=None):
        ok, _ = human_actor._human_box_clear(
            np.asarray(xy, dtype=np.float32), z_human,
            reloc_aabbs if aabbs is None else aabbs,
            body_r, body_h, base_lift, min_clear)
        return ok

    def _path_clear(path, min_clear, step=0.06):
        for i in range(len(path) - 1):
            a = np.asarray(path[i], dtype=np.float32)[:2]
            b = np.asarray(path[i + 1], dtype=np.float32)[:2]
            L = float(np.linalg.norm(b - a))
            n = max(int(L / step), 1)
            for k in range(n + 1):
                if not _spot_clear(a + (b - a) * (k / n), min_clear, aabbs=wall_aabbs):
                    return False
        return True

    def _build_walk_path(start, goal):
        # A* with extra erosion so the centre-line stays clear of walls, AND with
        # table footprints stamped as obstacles so the walk routes AROUND tables
        # (not under them); then gentle simplify (small epsilon avoids cutting wall
        # corners), then the existing furniture deflection.
        routed = human_actor._nav_path_avoiding(
            start, goal, z_human, trav_map, _NAV_OBSTACLE_AABBS, floor=0,
            erosion_radius=0.45, pad=0.30)
        if len(routed) < 2:
            routed = human_actor._nav_path_avoiding(
                start, goal, z_human, trav_map, _NAV_OBSTACLE_AABBS, floor=0, pad=0.30)
        if len(routed) < 2:
            return None
        path = [start.copy()] + [np.asarray(p, dtype=np.float32) for p in routed]
        if avoid_objects:
            for obj in avoid_objects:
                try:
                    obj_pos = to_numpy(obj.get_position_orientation()[0]).astype(np.float32)[:2]
                    cat_h = (getattr(obj, "category", "") or "").lower()
                    cl = 0.5 if ("chair" in cat_h or "sofa" in cat_h or "stool" in cat_h) else 0.35
                    path_pts = np.array([p[:2] for p in path], dtype=np.float32)
                    if float(np.min(np.linalg.norm(path_pts - obj_pos[None, :], axis=1))) > 1.2:
                        continue
                    path = _deflect_path_around_point(path, obj_pos, clearance=cl)
                except Exception:
                    pass
        return _simplify_path(path, epsilon=0.15)

    # Build an ordered candidate list. For the nearby case (no staircase) use a
    # DETERMINISTIC nearest-first fan around the human (radii outward × angles):
    # random sampling almost never hits the small clean spots in a cluttered
    # kitchen (left the human stuck "stays put"), so we sweep deterministically and
    # take the first clean + reachable + wall-clear-path spot. A per-round seeded
    # angle offset gives cross-round variety yet stays reproducible (§0/§16).
    candidates: List[np.ndarray] = []
    rng = np.random.RandomState(int(round_id) & 0x7FFFFFFF)
    if stair_xy is not None:
        for _ in range(200):
            off = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            candidates.append(np.array([stair_xy[0] + off[0], stair_xy[1] + off[1]], dtype=np.float32))
    else:
        ang0 = float(rng.uniform(-math.pi, math.pi))
        r = float(min_dist)
        while r <= max_dist + 1e-6:
            for deg in range(0, 360, 15):
                a = ang0 + math.radians(deg)
                candidates.append(np.array([human_start[0] + math.cos(a) * r,
                                            human_start[1] + math.sin(a) * r], dtype=np.float32))
            r += 0.2

    dest = None
    full_path: Optional[List[np.ndarray]] = None
    for cand_xy in candidates:
        try:
            candidate = np.array([float(cand_xy[0]), float(cand_xy[1]), z_human], dtype=np.float32)
            dist = float(np.linalg.norm(candidate[:2] - human_start[:2]))
            # Keep the relocation nearby: between min_dist and max_dist straight-line
            # (so the human stays in view / in-room rather than wandering across the
            # house). max_dist is yaml-configurable per combo (relocate_max_dist).
            if dist < min_dist or dist > max_dist:
                continue
            # (1) the held destination must clear walls (human stands here a whole round)
            if not _spot_clear(candidate[:2], 0.25):
                continue
            # (2) a wall-clear walk path to it must exist
            cand_path = _build_walk_path(human_start, candidate)
            if cand_path is None:
                continue
            path_len = sum(float(np.linalg.norm(cand_path[i+1][:2] - cand_path[i][:2]))
                           for i in range(len(cand_path) - 1))
            if path_len > max_dist * 2.0:   # allow some routing detour, but not a trek
                continue
            if not _path_clear(cand_path, 0.02):
                continue
            task_dist = float(np.linalg.norm(candidate[:2] - task_xy)) if task_xy is not None else -1.0
            dest = candidate
            full_path = cand_path
            print(f"[relocation] dest=({candidate[0]:.2f},{candidate[1]:.2f}) "
                  f"dist_from_start={dist:.2f} dist_from_task={task_dist:.2f} "
                  f"attempt={attempt} (dest+path wall-clear)", flush=True)
            break
        except Exception:
            pass

    if dest is None or full_path is None:
        # No collision-free relocation found — staying put is correct (and never
        # clips) rather than walking through a wall.
        print("[relocation] no wall-clear destination found; human stays put", flush=True)
        return

    cum_dist = [0.0]
    for i in range(1, len(full_path)):
        cum_dist.append(
            cum_dist[-1] + float(np.linalg.norm(full_path[i][:2] - full_path[i - 1][:2]))
        )
    total_len = cum_dist[-1]
    if total_len < 0.3:
        print("[relocation] path too short; skipping", flush=True)
        return

    walk_frames = max(int(total_len / walk_speed * fps), 30)

    print(
        f"[relocation] human walking {total_len:.2f}m over {walk_frames} frames "
        f"(speed={walk_speed:.1f} m/s, {len(full_path)} path points)",
        flush=True,
    )

    start_yaw_robot = _yaw_from_quat(to_numpy(robot.get_position_orientation()[1]))
    turn_frames = max(int(walk_frames * 0.3), 1)

    robot_home = robot_start.copy()
    robot_path = [np.array([robot_home[0], robot_home[1], z_robot], dtype=np.float32)]
    for p in full_path:
        robot_path.append(np.array([float(p[0]), float(p[1]), z_robot], dtype=np.float32))
    robot_cum = [0.0]
    for i in range(1, len(robot_path)):
        robot_cum.append(robot_cum[-1] + float(np.linalg.norm(robot_path[i][:2] - robot_path[i - 1][:2])))
    robot_total = robot_cum[-1]
    home_to_start = robot_cum[1] if len(robot_cum) > 1 else 0.0

    for step_i in range(walk_frames):
        _restore_pinned_objects(env)
        t = (step_i + 1) / walk_frames

        human_d = t * total_len
        human_pos = _interp_path(full_path, cum_dist, human_d)
        human_pos[2] = z_human
        human_yaw = _yaw_at_dist(full_path, cum_dist, human_d)

        robot_d = human_d - stand_off + home_to_start
        if robot_d <= 0.0:
            robot_pos = robot_home.copy()
            robot_pos[2] = z_robot
        else:
            robot_pos = _interp_path(robot_path, robot_cum, min(robot_d, robot_total))
            robot_pos[2] = z_robot

        target_robot_yaw = math.atan2(
            float(human_pos[1] - robot_pos[1]),
            float(human_pos[0] - robot_pos[0]),
        )
        if step_i < turn_frames:
            alpha = _smoothstep((step_i + 1) / turn_frames)
            robot_yaw = start_yaw_robot + _wrap_to_pi(target_robot_yaw - start_yaw_robot) * alpha
        else:
            robot_yaw = target_robot_yaw

        human_actor.base_pos = human_pos.copy()
        human_actor.base_yaw = float(human_yaw)
        per_frame_disp = total_len / max(walk_frames, 1)
        human_actor.linear_speed = walk_speed
        human_actor.motion_state = "walk"
        human_actor._last_displacement_m = per_frame_disp
        human_actor.gait_phase += per_frame_disp * 2.0
        human_actor._advance_animation()
        human_actor.apply_pose()

        robot.set_position_orientation(
            robot_pos.astype(np.float32), quat_from_yaw(float(robot_yaw))
        )

        obs, rew, term, trunc, info = env.step(zero_action, n_render_iterations=1)
        info["reward"] = 0.0
        info["terminated"] = False
        info["truncated"] = False
        recorder.record("Relocate", obs, info, round_id=round_id)

    human_actor.anchor_pos = human_actor.base_pos.copy()
    human_actor.motion_state = "hold"
    human_actor.linear_speed = 0.0
    print(
        f"[relocation] done — human now at ({human_actor.base_pos[0]:.2f}, "
        f"{human_actor.base_pos[1]:.2f})",
        flush=True,
    )


# --------------------------------------------------------------------------- #
# Door-based human walking path (ported from whdxmk_whd_backup_20260421's
# hrc_minimal_demo: human walks once to the most-reachable door's approach
# point and back to spawn, then stops. Picks the door whose A*-expanded path
# from spawn has the largest number of nav points -- proxy for "actually
# reachable, not just clamped to the nearest grid cell".)
# --------------------------------------------------------------------------- #

def install_door_human_waypoints(env: Any, human_actor: Any,
                                 phase: str = "ExecuteBehaviorTask") -> bool:
    """Pick the best door in the scene and install ``[approach, spawn]`` as the
    human's walking path. Returns True on success.
    """
    spawn = human_actor.base_pos.astype(np.float32).copy()
    z = float(spawn[2])
    sp = spawn[:2].astype(np.float32)

    doors: List[Tuple[str, float, float]] = []
    for obj in env.scene.objects:
        cat = (getattr(obj, "category", "") or "").lower()
        if "door" in cat:
            p = obj.get_position_orientation()[0]
            p = p.cpu().numpy() if hasattr(p, "cpu") else p
            doors.append((obj.name, float(p[0]), float(p[1])))

    print(f"[waypoint] spawn={spawn.tolist()}", flush=True)
    print(f"[waypoint] door candidates ({len(doors)}):", flush=True)
    for name, dx, dy in doors:
        dist = float(np.hypot(dx - sp[0], dy - sp[1]))
        print(f"  - {name}: xy=({dx:.2f}, {dy:.2f}) dist={dist:.2f}m", flush=True)

    if not doors:
        print("[waypoint] no door found in scene -- skip trajectory", flush=True)
        return False

    trav_map = human_actor._try_get_trav_map()
    best: Optional[Tuple[int, str, float, float, np.ndarray]] = None
    for name, dx, dy in doors:
        dr = np.array([dx, dy], dtype=np.float32)
        v = dr - sp
        v_n = v / (float(np.linalg.norm(v)) + 1e-6)
        approach = dr - v_n * 0.8  # 0.8 m before the door
        if trav_map is None:
            score = 0
        else:
            seg = human_actor._nav_path_between(
                np.array([sp[0], sp[1], z], dtype=np.float32),
                np.array([approach[0], approach[1], z], dtype=np.float32),
                z, trav_map, 0,
            )
            score = len(seg)
        print(f"  - probe {name}: A* nav points={score}", flush=True)
        if best is None or score > best[0]:
            best = (score, name, dx, dy, approach)

    score, door_name, dx, dy, approach = best
    print(f"[waypoint] picked door: {door_name} xy=({dx:.2f}, {dy:.2f}) score={score}", flush=True)
    # NOTE: backup demo used mode='waypoint_once', loop=False (walk to door, back
    # to spawn, then stop). That was fine for a 30s single-round demo. Our multi-
    # round replays are 45s + 240s long, so use waypoint_loop + loop=True so the
    # human keeps pacing door<->spawn for the entire replay duration.
    human_actor.set_waypoints(
        [
            [float(approach[0]), float(approach[1]), z],
            [float(spawn[0]), float(spawn[1]), z],
        ],
        mode="waypoint_loop",
        phase=phase,
        loop=True,
        expand_with_astar=True,
        face_robot=False,
    )
    print(f"[waypoint] door_approach={approach.tolist()}", flush=True)
    return True


def run_radio_toggle_phase(
    env: Any,
    recorder: DemoRecorder,
    human: Any,
    radio_obj: Any,
    dwell_steps: int,
    toggle_at_step: int,
    round_id: int,
    human_actor: Any = None,
) -> bool:
    """Robot leans toward the radio, extends its right arm to "press" the button,
    then toggles ``ToggledOn`` at the peak of the gesture and slowly retracts.

    The motion is implemented by directly setting joint positions each frame
    (``robot.set_joint_positions``). This works because ``disable_robot_control``
    has put every joint in EFFORT mode with kp=kd=None, so the controller exerts
    no torque — re-applying the kinematic pose every frame keeps the arm steady.

    Logged frames use the FSM phase ``ExecuteBehaviorTask`` so the recorded HDF5
    keeps the same schema as single-task demos.
    """
    robot = env.robots[0]
    zero_action = np.zeros(robot.action_dim, dtype=np.float32)

    # ---- Resolve joint indices (R1Pro: 7 right-arm + 4 torso joints).
    joint_names = list(robot.dof_names_ordered)

    def _idx(name: str) -> Optional[int]:
        try:
            return joint_names.index(name)
        except ValueError:
            return None

    right_arm_idx = [_idx(f"right_arm_joint{i}") for i in range(1, 8)]
    trunk_idx = [_idx(f"torso_joint{i}") for i in range(1, 5)]

    base_qpos = to_numpy(robot.get_joint_positions()).astype(np.float32).copy()

    # Target pose for the right arm: shoulder rotates the arm forward+down so the
    # end-effector reaches in front of the chest (where the radio sits on the table).
    # Joint signs are best-effort defaults; if the visual is off we can flip signs
    # via config later. R1Pro right-arm joint convention (radians):
    #   j1 yaw, j2 pitch, j3 roll, j4 elbow, j5 wrist roll, j6 wrist pitch, j7 wrist yaw.
    arm_reach_pose = np.array([0.0, -0.9, 0.0, 1.4, 0.0, 0.6, 0.0], dtype=np.float32)

    # Trunk lean: nudge torso joint 2 forward by ~0.25 rad. Falls back gracefully
    # if the joint isn't present (kept relative to baseline so reset is identity).
    trunk_lean_delta = np.array([0.0, -0.25, 0.0, 0.0], dtype=np.float32)

    # ---- Resolve ToggledOn state class lazily.
    toggled_on_cls = None
    try:
        from omnigibson.object_states import ToggledOn as toggled_on_cls  # type: ignore
    except Exception:
        pass

    toggle_succeeded = False
    radio_pos = to_numpy(radio_obj.get_position_orientation()[0])
    peak_step = max(int(toggle_at_step), 1)
    total = max(int(dwell_steps), 1)

    for step_i in range(total):
        # Smooth motion profile: 0 -> 1 over [0, peak], 1 -> ~0.4 over (peak, end].
        # We hold near the peak for a few extra frames before retracting so the
        # "press" is clearly visible in the rendered video.
        if step_i <= peak_step:
            t = _smoothstep(step_i / peak_step)
        else:
            tail = (step_i - peak_step) / max(total - peak_step, 1)
            t = 1.0 - 0.6 * _smoothstep(tail)  # retract to 0.4 amplitude (still leaning a bit)

        target_qpos = base_qpos.copy()
        for idx, val in zip(right_arm_idx, arm_reach_pose):
            if idx is not None:
                target_qpos[idx] = base_qpos[idx] + (float(val) - base_qpos[idx]) * t
        for idx, delta in zip(trunk_idx, trunk_lean_delta):
            if idx is not None:
                target_qpos[idx] = base_qpos[idx] + float(delta) * t

        # Keep robot facing the radio.
        cur_pos = to_numpy(robot.get_position_orientation()[0])
        face_yaw = math.atan2(float(radio_pos[1] - cur_pos[1]), float(radio_pos[0] - cur_pos[0]))
        robot.set_position_orientation(cur_pos, quat_from_yaw(face_yaw))
        robot.set_joint_positions(th.tensor(target_qpos, dtype=th.float32))
        robot.keep_still()

        # Toggle the radio ON at the peak of the reach.
        if step_i == peak_step and toggled_on_cls is not None and not toggle_succeeded:
            try:
                radio_obj.states[toggled_on_cls].set_value(True)
                toggle_succeeded = True
                print(f"[radio] Toggled ON at dwell step {step_i}", flush=True)
            except Exception as exc:
                print(f"[radio] WARN: could not toggle ToggledOn: {exc}", flush=True)

        if human_actor is not None:
            human_actor.step("ExecuteBehaviorTask", robot)
        obs, rew, term, trunc, info = env.step(zero_action, n_render_iterations=1)
        info["reward"] = float(rew) if rew is not None else 0.0
        info["terminated"] = bool(term) if term is not None else False
        info["truncated"] = bool(trunc) if trunc is not None else False
        if toggle_succeeded:
            done_block = info.get("done")
            if not isinstance(done_block, dict):
                done_block = {}
            done_block["success"] = True
            info["done"] = done_block
        recorder.record("ExecuteBehaviorTask", obs, info, round_id=round_id, action=zero_action)

    # Restore arm/trunk to baseline so subsequent phases (ReturnToHuman) start clean.
    try:
        robot.set_joint_positions(th.tensor(base_qpos, dtype=th.float32))
        robot.keep_still()
    except Exception as exc:
        print(f"[radio] WARN: could not restore baseline pose: {exc}", flush=True)

    return toggle_succeeded


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #

def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _resolve(cli_val: Any, cfg: Dict[str, Any], key: str, default: Any) -> Any:
    return cli_val if cli_val is not None else cfg.get(key, default)


def _build_phase_motion() -> Dict[str, HumanPhaseMotion]:
    """Per-phase human motion: human faces robot only during initial wait &
    command phases (natural for giving instructions).  During task execution,
    return, and relocation the human keeps whatever orientation they have —
    just like a real person who doesn't track the robot with their gaze.
    """
    hold_face = HumanPhaseMotion(mode="hold", face_robot=True, speed=0.0, patrol_radius=0.8, loop=True)
    hold_free = HumanPhaseMotion(mode="hold", face_robot=False, speed=0.0, patrol_radius=0.8, loop=True)
    walk = HumanPhaseMotion(mode="waypoint_once", face_robot=False, speed=0.4, patrol_radius=0.9, loop=False)
    return {
        "WaitNearHuman": hold_face,
        "ReceiveCommand": hold_face,
        "GoToTask": hold_free,
        "ExecuteBehaviorTask": hold_free,
        "TaskSegmentDone": hold_free,
        "ReturnToHuman": hold_free,
        "FaceHuman": hold_free,
        "WaitNextCommand": hold_face,
        "AllTasksDone": hold_free,
        "Relocate": walk,
    }


def _compute_radio_approach(radio_obj: Any, approach_distance: float, robot: Any) -> np.ndarray:
    """Pick a stand point ``approach_distance`` away from the radio, on the line
    between the radio and the robot's current position."""
    radio_pos = to_numpy(radio_obj.get_position_orientation()[0]).astype(np.float32)
    robot_pos = to_numpy(robot.get_position_orientation()[0]).astype(np.float32)
    direction = robot_pos[:2] - radio_pos[:2]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.array([1.0, 0.0], dtype=np.float32)
    else:
        direction = direction / norm
    target = radio_pos.copy()
    target[0] += direction[0] * float(approach_distance)
    target[1] += direction[1] * float(approach_distance)
    target[2] = robot_pos[2]
    return target


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hrc_multitask_v01.yaml")
    parser.add_argument("--raw-hdf5", default=None,
                        help="Override the base task HDF5 (yaml: raw_hdf5).")
    parser.add_argument("--output-dir", default=None,
                        help="Override the output directory (yaml: output_dir).")
    parser.add_argument("--episode-id", type=int, default=None,
                        help="Override the base task episode_id (yaml: episode_id). "
                             "Note: each rawdata HDF5 file only contains demo_0, so this "
                             "is almost always 0; use --raw-hdf5 to switch episode files.")
    parser.add_argument("--max-playback-steps", type=int, default=None)
    parser.add_argument(
        "--external-hdf5", action="append", default=None, metavar="ROUND_IDX:PATH",
        help="Override yaml task_rounds[ROUND_IDX].external_hdf5. "
             "Can be repeated, e.g. --external-hdf5 0:/path/a.hdf5 --external-hdf5 2:/path/b.hdf5.",
    )
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    raw_hdf5 = _resolve(args.raw_hdf5, cfg, "raw_hdf5", None)
    if not raw_hdf5:
        raise ValueError("raw_hdf5 must be set in config or via --raw-hdf5")
    output_dir = _resolve(args.output_dir, cfg, "output_dir", "outputs/hrc_multitask_v01")
    human_usd = cfg.get("human_usd", DEFAULT_HUMAN_USD)
    resolution = int(cfg.get("resolution", 256))
    fps = int(cfg.get("fps", 30))
    return_steps = int(cfg.get("return_steps", 60))
    go_to_task_steps = int(cfg.get("go_to_task_steps", 60))
    face_human_steps = int(cfg.get("face_human_steps", 10))
    wait_steps = int(cfg.get("wait_steps", 5))
    relocate_steps = int(cfg.get("relocate_steps", 120))
    human_distance = float(cfg.get("human_distance", 1.8))
    stand_off = float(cfg.get("stand_off", 1.0))
    # Fraction of the ExecuteBehaviorTask replay at which the human starts
    # walking back toward spawn. 0.90 = human starts going home during the last
    # 10% of the task, arriving at spawn right as the replay ends (matches the
    # natural pattern seen in task-0026 reference demos).
    home_trigger_fraction = float(cfg.get("home_trigger_fraction", 0.90))
    episode_id = int(_resolve(args.episode_id, cfg, "episode_id", 0))
    max_playback_steps = _resolve(args.max_playback_steps, cfg, "max_playback_steps", None)
    max_playback_steps = None if max_playback_steps is None else int(max_playback_steps)
    extra_objects: List[Dict[str, Any]] = list(cfg.get("extra_objects", []))
    hidden_categories: List[str] = [
        str(c).strip().lower() for c in cfg.get("hidden_categories", []) if str(c).strip()
    ]
    hidden_object_names: List[str] = [
        str(n).strip() for n in cfg.get("hidden_object_names", []) if str(n).strip()
    ]
    task_rounds: List[Dict[str, Any]] = list(cfg.get("task_rounds", []))
    if not task_rounds:
        raise ValueError("task_rounds must be defined in config (got empty list).")

    # CLI overrides for external HDF5 paths, e.g. --external-hdf5 0:/foo/bar.hdf5
    if args.external_hdf5:
        for spec in args.external_hdf5:
            if ":" not in spec:
                raise ValueError(
                    f"--external-hdf5 expects 'ROUND_IDX:PATH', got: {spec!r}"
                )
            idx_str, path_str = spec.split(":", 1)
            try:
                idx = int(idx_str)
            except ValueError as exc:
                raise ValueError(
                    f"--external-hdf5: ROUND_IDX must be int, got {idx_str!r}"
                ) from exc
            if idx < 0 or idx >= len(task_rounds):
                raise ValueError(
                    f"--external-hdf5: ROUND_IDX={idx} is out of range "
                    f"(task_rounds has {len(task_rounds)} entries)."
                )
            if task_rounds[idx].get("kind") != "replay_external_segment":
                raise ValueError(
                    f"--external-hdf5: round {idx} kind="
                    f"{task_rounds[idx].get('kind')!r}, expected replay_external_segment."
                )
            if not os.path.isfile(path_str):
                raise FileNotFoundError(
                    f"--external-hdf5: file does not exist: {path_str}"
                )
            old = task_rounds[idx].get("external_hdf5")
            task_rounds[idx] = {**task_rounds[idx], "external_hdf5": path_str}
            print(
                f"[multitask] CLI override: task_rounds[{idx}].external_hdf5 "
                f"{old!r} -> {path_str!r}",
                flush=True,
            )

    ensure_parent(os.path.join(output_dir, "placeholder.txt"))
    print(f"[multitask] Config: {os.path.abspath(args.config)}", flush=True)
    print(f"[multitask] Output dir: {os.path.abspath(output_dir)}", flush=True)
    print(f"[multitask] Base HDF5: {raw_hdf5}", flush=True)

    base_activity = get_activity_from_hdf5(raw_hdf5)
    base_instruction = resolve_instruction(base_activity)

    playback_cache = os.path.join(output_dir, f"playback_cache_{os.getpid()}.hdf5")
    env = build_playback_env(
        raw_hdf5,
        resolution=resolution,
        output_hdf5_path=playback_cache,
        robot_modalities=["rgb"],
    )

    try:
        env.reset()

        # ---- Pitch the onboard ZED camera down so tabletop objects are visible.
        # Opt-in via config: head_camera_pitch_deg (positive = nose-down).
        head_cam_pitch = float(cfg.get("head_camera_pitch_deg", 0.0))
        head_cam_axis = str(cfg.get("head_camera_pitch_axis", "x"))
        tilt_head_camera(env, head_cam_pitch, axis=head_cam_axis)

        # ---- Widen the onboard ZED camera FOV so the robot can see floor-level
        # baskets, tabletop radio, and the human all at once during navigation.
        # 40mm matches BEHAVIOR-1K's official data-collection aperture (HFOV ≈ 99°).
        head_cam_aperture = float(cfg.get("head_camera_horizontal_aperture", 0.0))
        set_head_camera_aperture(env, head_cam_aperture)

        # ---- Spawn extra objects (radio, etc.) BEFORE adding human / starting rounds.
        spawned: Dict[str, Any] = {}
        for spec in extra_objects:
            name = spec["name"]
            print(f"[multitask] Spawning extra object: {name} ({spec.get('category')}/{spec.get('model')})", flush=True)
            obj = spawn_extra_object(
                env,
                name=name,
                category=spec["category"],
                model=spec["model"],
                position=spec["position"],
                orientation=spec.get("orientation", [0.0, 0.0, 0.0, 1.0]),
                in_rooms=spec.get("in_rooms"),
            )
            spawned[name] = obj
            print(f"[multitask]   -> spawned at {to_numpy(obj.get_position_orientation()[0]).tolist()}", flush=True)

        # Robot control must run in EFFORT mode so recorded actions replay deterministically.
        # ``playback_episode_with_callback`` only does this when skip_init=False, but we always
        # use skip_init=True (to keep the radio alive), so call it once here.
        disable_robot_control(env)

        # Load the base-task HDF5 step-0 state so that ALL objects (baskets
        # etc.) are at their HDF5-recorded positions, not the USD-scene
        # defaults.  env.reset() loads from the scene USD, which can differ
        # from the recording.  This ensures Round 1's replay start matches
        # what's already on screen — no basket "pop-in".
        # Save robot pose first — load_state would move it to the task start.
        _robot_save_pos = to_numpy(env.robots[0].get_position_orientation()[0]).copy()
        _robot_save_quat = to_numpy(env.robots[0].get_position_orientation()[1]).copy()
        _init_grp = env.input_hdf5["data"][f"demo_{episode_id}"]
        _init_state = _init_grp["state"][0]
        _init_sz = int(_init_grp["state_size"][0])
        og.sim.load_state(th.tensor(_init_state[:_init_sz]), serialized=True)
        env.robots[0].set_position_orientation(_robot_save_pos, _robot_save_quat)
        env.robots[0].keep_still()
        print("[multitask] Loaded HDF5 step-0 state to align object positions (robot pose preserved)", flush=True)

        # ---- Hide task-irrelevant props (cookies / cheese / bows / candles ...).
        # The objects still exist physically — only their visibility flag is
        # toggled, so reward / state restore semantics are unchanged.
        # NOTE: og.sim.load_state(serialized=True) does not touch USD
        # visibility, so a single set here covers the whole episode.
        if hidden_categories or hidden_object_names:
            wanted_cats = set(hidden_categories)
            wanted_names = set(hidden_object_names)
            hidden_count = 0
            hidden_by_cat: Dict[str, int] = {}
            for obj in env.scene.objects:
                obj_name = getattr(obj, "name", "") or ""
                obj_cat = (getattr(obj, "category", "") or "").lower()
                should_hide = (obj_name in wanted_names) or (obj_cat in wanted_cats)
                if not should_hide:
                    continue
                try:
                    obj.visible = False
                    hidden_count += 1
                    hidden_by_cat[obj_cat] = hidden_by_cat.get(obj_cat, 0) + 1
                except Exception as exc:
                    print(f"[multitask] WARN: could not hide {obj_name} "
                          f"(cat={obj_cat}): {exc}", flush=True)
            if hidden_count > 0:
                summary = ", ".join(f"{c}:{n}" for c, n in sorted(hidden_by_cat.items()))
                print(f"[multitask] Hidden {hidden_count} task-irrelevant objects "
                      f"({summary})", flush=True)
            else:
                print(f"[multitask] No objects matched hidden_categories="
                      f"{hidden_categories} / hidden_object_names="
                      f"{hidden_object_names}", flush=True)

        # Align extra objects (radio etc.) to their external-replay positions so
        # they don't flash/teleport when the replay starts.  For each round that
        # uses an external HDF5, peek step 0 to read the correct object poses.
        for round_cfg in task_rounds:
            if round_cfg.get("kind") != "replay_external_segment":
                continue
            ext_path = round_cfg.get("external_hdf5")
            if not ext_path or not spawned:
                continue
            ext_episode = int(round_cfg.get("episode_id", 0))
            ext_start = int(round_cfg.get("start_step", 0))
            _snap = og.sim.dump_state(serialized=True)
            try:
                with h5py.File(ext_path, "r") as f:
                    traj = f[f"data/demo_{ext_episode}"]
                    st = traj["state"][ext_start, : int(traj["state_size"][ext_start])]
                    og.sim.load_state(th.tensor(st, dtype=th.float32), serialized=True)
                for obj_name, obj in spawned.items():
                    pos = to_numpy(obj.get_position_orientation()[0]).copy()
                    quat = to_numpy(obj.get_position_orientation()[1]).copy()
                    print(f"[multitask] Peeked {obj_name} from {os.path.basename(ext_path)} "
                          f"step {ext_start}: ({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})", flush=True)
                    spawned[obj_name]._ext_init_pos = pos
                    spawned[obj_name]._ext_init_quat = quat
            except Exception as exc:
                print(f"[multitask] WARN: failed to peek extra objects from {ext_path}: {exc}", flush=True)
            finally:
                og.sim.load_state(_snap, serialized=True)
                del _snap
            for obj_name, obj in spawned.items():
                if hasattr(obj, "_ext_init_pos"):
                    obj.set_position_orientation(obj._ext_init_pos, obj._ext_init_quat)
                    obj.keep_still()
                    print(f"[multitask] Set {obj_name} to external-replay position", flush=True)

        # ---- Add walking human (Habitat humanoid in walk mode).
        # Human spawn placement (ARCHITECTURE.md §0/§14): all knobs are opt-in.
        # combo A's yaml sets none of them → byte-for-byte legacy geometric spawn.
        # New / narrow scenes set human_spawn_collision_free: true → a deterministic
        # candidate search validated against the human's real 3-D body volume picks
        # a clear spot automatically, with NO per-scene anchor tuning (the general
        # "一劳永逸" fix for clipping). human_spawn (if given) is just a hint.
        human_actor = KinematicHumanoidActor(
            env=env,
            usd_path=human_usd,
            name="human_npc",
            distance=human_distance,
            collision_proxy=None,
            phase_motion=_build_phase_motion(),
            step_dt=1.0 / max(fps, 1),
            spawn_anchor=cfg.get("human_spawn"),
            spawn_jitter=cfg.get("human_spawn_jitter"),
            spawn_snap=bool(cfg.get("human_spawn_snap", False)),
            spawn_seed=int(episode_id),
            spawn_collision_free=bool(cfg.get("human_spawn_collision_free", False)),
            spawn_search=cfg.get("human_spawn_search"),
        )
        # Collision proxy disabled: OmniGibson's PrimitiveObject asserts on
        # kinematic_only=True for objects added while simulation is playing.
        # Anti-clipping is handled by path-level avoidance instead:
        #   - A* trav_map erosion (0.3m clearance from furniture/walls)
        #   - _deflect_path_around_point (robot detours around human)
        human_actor.collision_proxy.enabled = False
        human_actor.spawn_near_robot()
        human = human_actor
        print(f"[multitask] Human spawned at {human_actor.base_pos.tolist()}", flush=True)

        # NOTE: install_door_human_waypoints(...) is called PER ROUND just before
        # each ExecuteBehaviorTask replay (since the path uses mode='waypoint_once'
        # and exhausts itself after one walk-to-door-and-back trip).

        # ---- Recorder.
        recorder = DemoRecorder(
            env=env,
            human=human,
            activity=base_activity,
            instruction=base_instruction,
            output_dir=output_dir,
            fps=fps,
            input_hdf5_path=raw_hdf5,
        )
        recorder._human_actor = human_actor

        # ---- Install phase-conditioned head-camera gaze hook.
        # If `head_camera_phase_pitch` is provided in the YAML, the head ZED
        # camera is automatically re-pitched whenever the FSM phase changes
        # (e.g. level for FaceHuman / WaitNearHuman, look down during
        # ExecuteBehaviorTask). The static `head_camera_pitch_deg` value is used
        # as the fallback for any phase not listed in the map.
        # When `head_camera_pitch_smooth_deg_per_frame` > 0 (default 1.5°/frame
        # ≈ 0.5s for a 25° transition at 30 fps), pitch changes are interpolated
        # over multiple frames instead of snapping.
        phase_pitch_cfg = cfg.get("head_camera_phase_pitch", None)
        smooth_step = float(cfg.get("head_camera_pitch_smooth_deg_per_frame", 1.5))
        if isinstance(phase_pitch_cfg, dict) and phase_pitch_cfg:
            phase_pitch_map = {str(k): float(v) for k, v in phase_pitch_cfg.items()}
            recorder.phase_change_hook = make_phase_pitch_hook(
                env=env,
                phase_pitch_map=phase_pitch_map,
                axis=head_cam_axis,
                default_pitch=head_cam_pitch if head_cam_pitch else None,
                smooth=(smooth_step > 0),
            )
            if smooth_step > 0:
                recorder.pre_record_hook = lambda: step_head_camera_pitch_smooth(env, max_step_deg=smooth_step)
                print(f"[multitask] Phase-conditioned head-camera gaze enabled (smooth, "
                      f"{smooth_step:.2f}°/frame). Map: {phase_pitch_map}, "
                      f"default fallback: {head_cam_pitch:+.1f}°", flush=True)
            else:
                print(f"[multitask] Phase-conditioned head-camera gaze enabled (snap). "
                      f"Map: {phase_pitch_map}, default fallback: {head_cam_pitch:+.1f}°", flush=True)

        # ---- Per-frame head-camera state provider (writes pitch + extrinsics
        # into HDF5 obs/) and dataset-level reproducibility metadata.
        # Without this, the camera changes (FOV, pitch, dynamic gaze) live only
        # in the YAML and are NOT reflected in the recorded HDF5 — making the
        # dataset non-reproducible. With this provider, every frame's HDF5
        # entry contains the actual pitch / world-pose used to render that
        # frame's RGB.
        recorder.camera_state_provider = lambda: get_head_camera_state(env)
        head_cam_intrinsics = get_head_camera_intrinsics(env)
        recorder.dataset_metadata = {
            "head_camera_intrinsics": head_cam_intrinsics,
            "head_camera_static_pitch_deg": head_cam_pitch,
            "head_camera_pitch_axis": head_cam_axis,
            "head_camera_phase_pitch": (
                {str(k): float(v) for k, v in (phase_pitch_cfg or {}).items()}
                if isinstance(phase_pitch_cfg, dict) else None
            ),
            "head_camera_pitch_smooth_deg_per_frame": smooth_step,
            "human_spawn": getattr(human_actor, "spawn_record", {}),
            "env_args": {
                "robot": type(env.robots[0]).__name__,
                "fps": fps,
                "resolution": resolution,
                "raw_hdf5": raw_hdf5,
                "episode_id": int(episode_id),
            },
        }
        print(f"[multitask] Head-camera state will be recorded per-frame in HDF5 obs/. "
              f"Intrinsics: HFOV={head_cam_intrinsics.get('hfov_deg', float('nan')):.1f}°, "
              f"VFOV={head_cam_intrinsics.get('vfov_deg', float('nan')):.1f}°, "
              f"focal={head_cam_intrinsics.get('focal_length', float('nan')):.2f}mm, "
              f"aperture={head_cam_intrinsics.get('horizontal_aperture', float('nan')):.2f}mm", flush=True)

        # Collect objects for path avoidance:
        #  - Small ground objects (baskets, boxes) not in trav_map
        #  - Chairs (trav_map erosion often insufficient for body width)
        avoid_objects: List[Any] = []
        _skip_cats = {"robot", "agent", "floor", "wall", "ceiling",
                      "door", "window", "shelf", "table",
                      "bed", "cabinet", "counter", "stove", "oven",
                      "fridge", "refrigerator", "dishwasher", "toilet",
                      "sink", "bathtub", "shower", "mirror", "picture",
                      "carpet", "rug", "curtain", "light", "lamp"}
        for obj in env.scene.objects:
            if obj == env.robots[0]:
                continue
            cat = (getattr(obj, "category", "") or "").lower()
            if cat in _skip_cats:
                continue
            try:
                p = to_numpy(obj.get_position_orientation()[0])
                is_chair = "chair" in cat or "sofa" in cat or "stool" in cat
                if is_chair:
                    if p[2] > 1.0:
                        continue
                else:
                    if p[2] > 0.35:
                        continue
                    aabb = to_numpy(obj.aabb_extent) if hasattr(obj, "aabb_extent") else None
                    if aabb is not None and (float(aabb[0]) > 0.8 or float(aabb[1]) > 0.8):
                        continue
                avoid_objects.append(obj)
                print(f"  [avoid-obj] {getattr(obj, 'name', '?')} cat={cat} "
                      f"pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})", flush=True)
            except Exception:
                pass
        print(f"[multitask] Collected {len(avoid_objects)} avoid-objects (chairs+baskets) for path avoidance", flush=True)

        # Collect go-around furniture (tables/desks) footprints so robot & human
        # paths route AROUND them instead of under the overhanging top (the 2-D
        # trav_map marks the floor between thin legs navigable). ARCHITECTURE.md §16.
        _collect_nav_obstacles(env)

        # Capture poses of task-specific objects (baskets etc.) so they can be
        # pinned during external replays whose recordings don't include them.
        _pinned_scene_objects.clear()
        for obj in env.scene.objects:
            cat = (getattr(obj, "category", "") or "").lower()
            if "basket" in cat or "wicker" in cat or "bow" in cat:
                pos = to_numpy(obj.get_position_orientation()[0]).copy()
                quat = to_numpy(obj.get_position_orientation()[1]).copy()
                _pinned_scene_objects[obj.name] = (pos, quat)
                print(f"  [pinned] {obj.name} pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})", flush=True)
        print(f"[multitask] Pinned {len(_pinned_scene_objects)} task-specific objects for cross-round visibility", flush=True)

        # Reward-segments of the base task — used by replay rounds.
        segments = discover_reward_segments(raw_hdf5, episode_id=episode_id)
        recorder._segments = segments
        print(f"[multitask] Discovered {len(segments)} reward-segments in base task.", flush=True)
        for si, (s, e) in enumerate(segments):
            print(f"  base seg {si}: [{s},{e}) ({e-s} steps)", flush=True)

        # ---- Position robot next to human before Round 0.
        # Try 12 directions around the human, pick the first that is
        # navigable (on trav_map) and far enough from all scene furniture.
        robot = env.robots[0]
        human_pos = human_actor.base_pos.astype(np.float32)
        robot_init_z = float(to_numpy(robot.get_position_orientation()[0])[2])

        furniture_xys_init: List[np.ndarray] = []
        for obj in env.scene.objects:
            if obj == robot:
                continue
            cat_i = (getattr(obj, "category", "") or "").lower()
            if cat_i in ("floor", "wall", "ceiling", "agent", "robot"):
                continue
            try:
                op = to_numpy(obj.get_position_orientation()[0])
                if 0.2 < op[2] < 1.5:
                    furniture_xys_init.append(np.array([float(op[0]), float(op[1])], dtype=np.float32))
            except Exception:
                pass

        trav_map_init = human_actor._try_get_trav_map()
        robot_init_pos = None
        for angle_offset in np.linspace(0, 2 * math.pi, 12, endpoint=False):
            cand_dir = np.array([math.cos(angle_offset), math.sin(angle_offset)], dtype=np.float32)
            cand_xy = human_pos[:2] + cand_dir * float(stand_off)
            too_close = False
            for fxy in furniture_xys_init:
                if float(np.linalg.norm(cand_xy - fxy)) < 0.5:
                    too_close = True
                    break
            if too_close:
                continue
            if trav_map_init is not None:
                try:
                    cand_3d = th.tensor([cand_xy[0], cand_xy[1], robot_init_z], dtype=th.float32)
                    on_map = trav_map_init.point_to_map(cand_3d)
                    if on_map is not None:
                        mx, my = int(on_map[0]), int(on_map[1])
                        grid = trav_map_init.floor_map[0]
                        if 0 <= mx < grid.shape[0] and 0 <= my < grid.shape[1]:
                            if grid[mx, my] < 128:
                                continue
                except Exception:
                    pass
            robot_init_pos = np.array([cand_xy[0], cand_xy[1], robot_init_z], dtype=np.float32)
            print(f"[multitask] Robot init candidate angle={math.degrees(angle_offset):.0f}° "
                  f"pos=({cand_xy[0]:.2f},{cand_xy[1]:.2f}) OK", flush=True)
            break

        if robot_init_pos is None:
            cand_xy = human_pos[:2] + np.array([0.0, -float(stand_off)], dtype=np.float32)
            robot_init_pos = np.array([cand_xy[0], cand_xy[1], robot_init_z], dtype=np.float32)
            print(f"[multitask] Robot init fallback pos=({cand_xy[0]:.2f},{cand_xy[1]:.2f})", flush=True)

        face_human_yaw = math.atan2(float(human_pos[1] - robot_init_pos[1]),
                                    float(human_pos[0] - robot_init_pos[0]))
        robot.set_position_orientation(robot_init_pos, quat_from_yaw(face_human_yaw))
        robot.keep_still()
        print(f"[multitask] Robot initialized next to human at {robot_init_pos[:2].tolist()}, "
              f"facing human (yaw={math.degrees(face_human_yaw):.1f}°)", flush=True)

        # ---- Idle 1.5 seconds (robot faces human, both stay still).
        idle_frames = int(1.5 * fps)
        zero_action = np.zeros(robot.action_dim, dtype=np.float32)
        for _ in range(idle_frames):
            robot.set_position_orientation(robot_init_pos, quat_from_yaw(face_human_yaw))
            robot.keep_still()
            if human_actor is not None:
                human_actor.step("WaitNearHuman", robot)
            obs, rew, term, trunc, info = env.step(zero_action, n_render_iterations=1)
            info["reward"] = 0.0
            info["terminated"] = False
            info["truncated"] = False
            recorder.record("WaitNearHuman", obs, info, round_id=0)
        print(f"[multitask] Idle phase done ({idle_frames} frames, {idle_frames/fps:.1f}s)", flush=True)

        # ---- Run rounds.
        num_rounds = len(task_rounds)
        round_summaries: List[Dict[str, Any]] = []

        for round_id, round_cfg in enumerate(task_rounds):
            kind = round_cfg.get("kind")
            instr = str(round_cfg.get("instruction", ""))
            is_last = round_id == num_rounds - 1
            print("\n" + "=" * 60, flush=True)
            print(f"[multitask] === ROUND {round_id}/{num_rounds-1}  kind={kind} ===", flush=True)
            print(f"[multitask] instruction: {instr}", flush=True)
            print("=" * 60, flush=True)

            # Per-round command + instruction override.
            recorder.instruction = instr or recorder.instruction
            recorder.add_command(round_id=round_id)

            # Phase 1-2: WaitNearHuman + ReceiveCommand
            run_zero_phase_pinned(env, recorder, "WaitNearHuman", wait_steps, round_id=round_id, human_actor=human_actor)
            run_zero_phase_pinned(env, recorder, "ReceiveCommand", wait_steps, round_id=round_id, human_actor=human_actor)

            stats: Dict[str, Any] = {"steps": 0, "saw_done": False, "success": False, "kind": kind}

            # Dict populated during replay to capture extra objects' rest poses.
            # For external replays, the first load_state() places extras at their
            # physics-settled position in that task's scene — we capture that and
            # restore after the replay to undo any later disturbance (e.g. robot
            # arm pushing the radio). For non-external replays, extras aren't
            # touched by load_state() so no capture/restore is needed.
            extra_rest_poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

            if kind == "scripted_radio":
                # Phase 3: GoToTask — walk near the radio.
                target_name = round_cfg["target_object"]
                radio_obj = spawned.get(target_name) or env.scene.object_registry("name", target_name)
                if radio_obj is None:
                    raise RuntimeError(f"Round {round_id} target_object '{target_name}' not found in scene.")
                approach_dist = float(round_cfg.get("approach_distance", 0.9))
                target_pos = _compute_radio_approach(radio_obj, approach_dist, env.robots[0])
                print(f"[multitask] GoToTask: walking to {target_pos[:2].tolist()} (radio approach)", flush=True)
                run_go_to_task_phase_smooth(env, recorder, human, go_to_task_steps, round_id=round_id,
                                            target_pos=target_pos, human_actor=human_actor,
                                            avoid_objects=avoid_objects)

                # Phase 4: ExecuteBehaviorTask — scripted dwell + toggle.
                dwell = int(round_cfg.get("dwell_steps", 30))
                toggle_at = int(round_cfg.get("toggle_at_step", dwell // 2))
                ok = run_radio_toggle_phase(env, recorder, human, radio_obj,
                                            dwell_steps=dwell, toggle_at_step=toggle_at,
                                            round_id=round_id, human_actor=human_actor)
                stats["steps"] = dwell
                stats["success"] = bool(ok)

            elif kind == "replay_external_segment":
                ext_path = round_cfg["external_hdf5"]
                ext_episode = int(round_cfg.get("episode_id", 0))
                ext_start = int(round_cfg.get("start_step", 0))
                ext_end_raw = round_cfg.get("end_step", None)
                ext_end = None if ext_end_raw is None else int(ext_end_raw)
                print(f"[multitask] External replay: {os.path.basename(ext_path)} demo_{ext_episode} "
                      f"[{ext_start},{ext_end if ext_end is not None else 'end'})", flush=True)

                # Phase 3: GoToTask — walk to recorded start (peek into external HDF5).
                saved_pos = to_numpy(env.robots[0].get_position_orientation()[0]).copy()
                saved_quat = to_numpy(env.robots[0].get_position_orientation()[1]).copy()
                try:
                    task_start_pos, task_start_quat = peek_robot_pos_from_external_hdf5(
                        ext_path, ext_episode, ext_start, return_quat=True)
                    env.robots[0].set_position_orientation(saved_pos, saved_quat)
                    print(f"[multitask] GoToTask: {saved_pos[:2].tolist()} -> {task_start_pos[:2].tolist()}", flush=True)
                    run_go_to_task_phase_smooth(env, recorder, human, go_to_task_steps, round_id=round_id,
                                                target_pos=task_start_pos, human_actor=human_actor,
                                                avoid_objects=avoid_objects)
                    replay_yaw = _yaw_from_quat(task_start_quat)
                    run_face_yaw_phase(env, recorder, task_start_pos, replay_yaw,
                                       steps=face_human_steps, round_id=round_id,
                                       human_actor=human_actor)
                except Exception as exc:
                    print(f"[multitask] WARN: external peek failed ({exc}); skipping GoToTask interp.", flush=True)

                # Phase 4: ExecuteBehaviorTask — drive sim from external HDF5.
                def _make_ext_cb(rid: int, st: Dict[str, Any]):
                    def _cb(phase: str, step_idx: int, obs: Dict[str, Any], info: Dict[str, Any], action: Any = None) -> None:
                        st["steps"] = max(st["steps"], int(step_idx) + 1 - ext_start)
                        if isinstance(info, dict) and isinstance(info.get("done"), dict):
                            d = info["done"]
                            st["saw_done"] = st["saw_done"] or bool(d.get("done", False))
                            st["success"] = st["success"] or bool(d.get("success", False))
                        recorder.record(phase, obs, info, round_id=rid, action=action)
                    return _cb

                ext_round_key = str(round_cfg.get("human_round_key", "round_0"))
                ext_traj = load_recorded_human_trajectory(ext_path, round_key=ext_round_key)
                if ext_traj is not None:
                    print(f"[multitask] Using recorded human trajectory from "
                          f"{os.path.basename(ext_path)}:hrc/{ext_round_key} "
                          f"({len(ext_traj[0])} frames).", flush=True)
                else:
                    print(f"[multitask] WARN: no recorded human trajectory in "
                          f"{ext_path}:hrc/{ext_round_key} — human will stay put.", flush=True)
                ext_effective_end = ext_end if ext_end is not None else 10**9
                ext_replay_span = max(ext_effective_end - ext_start, 1)

                _ext_first_frame_done = [False]

                def _ext_pre_step(step_idx: int) -> None:
                    if not _ext_first_frame_done[0]:
                        _ext_first_frame_done[0] = True
                        for spec in extra_objects:
                            obj = spawned.get(spec["name"])
                            if obj is not None:
                                extra_rest_poses[spec["name"]] = (
                                    to_numpy(obj.get_position_orientation()[0]).copy(),
                                    to_numpy(obj.get_position_orientation()[1]).copy(),
                                )
                                print(f"[multitask] Captured {spec['name']} rest pose at ext step {step_idx}: "
                                      f"{extra_rest_poses[spec['name']][0].tolist()}", flush=True)
                    _restore_pinned_objects(env)
                    if human_actor is None:
                        return
                    human_actor.sync_after_replay_load()
                    if ext_traj is not None:
                        progress = (step_idx - ext_start) / float(ext_replay_span)
                        drive_human_from_recording(human_actor, ext_traj[0], ext_traj[1], progress)
                    else:
                        human_actor.step("ExecuteBehaviorTask", env.robots[0])
                    # --- Layer 3: real-time dodge ---
                    dodge_robot_if_needed(human_actor, env.robots[0])

                playback_external_segment(
                    env=env,
                    ext_hdf5_path=ext_path,
                    episode_id=ext_episode,
                    callback=_make_ext_cb(round_id, stats),
                    start_step=ext_start,
                    end_step=ext_end,
                    pre_step_callback=_ext_pre_step,
                    max_playback_steps=max_playback_steps,
                )

            elif kind == "replay_segment":
                replay_all = round_cfg.get("replay_all_segments", False)
                seg_idx = int(round_cfg.get("segment_index", 0))
                if seg_idx >= len(segments):
                    raise IndexError(f"Round {round_id} segment_index {seg_idx} out of range (have {len(segments)} segments).")
                if replay_all:
                    seg_start = segments[0][0]
                    seg_end = segments[-1][1]
                else:
                    seg_start, seg_end = segments[seg_idx]
                override_end = round_cfg.get("end_step", None)
                if override_end is not None:
                    seg_end = int(override_end)
                print(f"[multitask] Replay steps [{seg_start},{seg_end})", flush=True)

                # Phase 3: GoToTask — walk to the recorded robot start position.
                # Peek both position AND orientation so we can smoothly rotate
                # into the replay's starting yaw (avoids snap-turn).
                _snap = og.sim.dump_state(serialized=True)
                task_start_pos, task_start_quat = peek_robot_pose_at_step(env, episode_id, seg_start)
                og.sim.load_state(_snap, serialized=True)
                del _snap
                print(f"[multitask] GoToTask: "
                      f"{to_numpy(env.robots[0].get_position_orientation()[0])[:2].tolist()} "
                      f"-> {task_start_pos[:2].tolist()}", flush=True)
                run_go_to_task_phase_smooth(env, recorder, human, go_to_task_steps, round_id=round_id,
                                            target_pos=task_start_pos, human_actor=human_actor,
                                            avoid_objects=avoid_objects)
                # Smooth rotation to match the replay's starting orientation.
                replay_yaw = _yaw_from_quat(task_start_quat)
                run_face_yaw_phase(env, recorder, task_start_pos, replay_yaw,
                                   steps=face_human_steps, round_id=round_id,
                                   human_actor=human_actor)

                def _make_cb(rid: int, st: Dict[str, Any]):
                    def _cb(phase: str, step_idx: int, obs: Dict[str, Any], info: Dict[str, Any], action: Any = None) -> None:
                        st["steps"] = max(st["steps"], int(step_idx) + 1)
                        if isinstance(info, dict) and isinstance(info.get("done"), dict):
                            d = info["done"]
                            st["saw_done"] = st["saw_done"] or bool(d.get("done", False))
                            st["success"] = st["success"] or bool(d.get("success", False))
                        recorder.record(phase, obs, info, round_id=rid, action=action)
                    return _cb

                seg_round_key = str(round_cfg.get("human_round_key", f"round_{seg_idx}"))
                seg_traj = load_recorded_human_trajectory(raw_hdf5, round_key=seg_round_key)
                if seg_traj is None and seg_round_key != "round_0":
                    seg_traj = load_recorded_human_trajectory(raw_hdf5, round_key="round_0")
                    seg_round_key = "round_0"
                if seg_traj is not None:
                    print(f"[multitask] Using recorded human trajectory from "
                          f"{os.path.basename(raw_hdf5)}:hrc/{seg_round_key} "
                          f"({len(seg_traj[0])} frames).", flush=True)
                else:
                    print(f"[multitask] WARN: no recorded human trajectory found for "
                          f"segment {seg_idx} — human will stay put.", flush=True)
                seg_replay_span = max(seg_end - seg_start, 1)

                def _pre_step(step_idx: int) -> None:
                    if human_actor is None:
                        return
                    human_actor.sync_after_replay_load()
                    if seg_traj is not None:
                        progress = (step_idx - seg_start) / float(seg_replay_span)
                        drive_human_from_recording(human_actor, seg_traj[0], seg_traj[1], progress)
                    else:
                        human_actor.step("ExecuteBehaviorTask", env.robots[0])
                    # --- Layer 3: real-time dodge ---
                    dodge_robot_if_needed(human_actor, env.robots[0])

                # Stop pinning baskets — this replay's own load_state will
                # manage them (they're part of this task's recording).
                _pinned_scene_objects.clear()

                # skip_init=True ALWAYS: env.scene.restore() rebuilds the scene from the
                # recorded scene_file (which doesn't include the radio) and would drop our
                # extra objects. The base scene was already initialised by env.reset().
                playback_episode_with_callback(
                    env, episode_id=episode_id,
                    callback=_make_cb(round_id, stats),
                    max_playback_steps=max_playback_steps,
                    pre_step_callback=_pre_step,
                    start_step=seg_start,
                    end_step=seg_end,
                    skip_init=True,
                )
            else:
                raise ValueError(f"Unknown round kind: {kind!r}")

            _restore_pinned_objects(env)

            # Phase 5-7: TaskSegmentDone + ReturnToHuman + FaceHuman
            run_zero_phase_pinned(env, recorder, "TaskSegmentDone", 15, round_id=round_id, human_actor=human_actor)

            run_return_phase_smooth(env, recorder, human, return_steps, round_id=round_id,
                                    stand_off=stand_off, human_actor=human_actor,
                                    avoid_objects=avoid_objects)
            run_face_human_phase_smooth(env, recorder, human, face_human_steps,
                                        round_id=round_id, human_actor=human_actor)

            # Phase 8: WaitNextCommand or AllTasksDone
            tail_phase = "AllTasksDone" if is_last else "WaitNextCommand"
            run_zero_phase_pinned(env, recorder, tail_phase, wait_steps, round_id=round_id, human_actor=human_actor)

            # Between rounds: human walks to a new nearby position, robot follows
            # behind on the same A*-planned path (no wall clipping).
            # Peek at the next round's task area so the human doesn't relocate
            # into the robot's replay trajectory.
            if not is_last:
                next_cfg = task_rounds[round_id + 1]
                next_task_center = None
                try:
                    next_kind = next_cfg.get("kind")
                    if next_kind == "replay_segment":
                        next_seg_idx = int(next_cfg.get("segment_index", 0))
                        if next_seg_idx < len(segments):
                            next_seg_start = segments[next_seg_idx][0]
                            _snap = og.sim.dump_state(serialized=True)
                            next_task_center = peek_robot_position_at_step(env, episode_id, next_seg_start)
                            og.sim.load_state(_snap, serialized=True)
                    elif next_kind == "replay_external_segment":
                        next_task_center = peek_robot_pos_from_external_hdf5(
                            next_cfg["external_hdf5"],
                            int(next_cfg.get("episode_id", 0)),
                            int(next_cfg.get("start_step", 0)),
                        )
                except Exception as exc:
                    print(f"[relocation] WARN: could not peek next task center: {exc}", flush=True)
                run_relocation_phase(
                    env, recorder, human_actor,
                    total_steps=relocate_steps,
                    round_id=round_id + 1,
                    stand_off=stand_off,
                    max_dist=float(cfg.get("relocate_max_dist", 5.0)),
                    task_center=next_task_center,
                    fps=fps,
                    avoid_objects=avoid_objects,
                    toward_staircase=bool(cfg.get("relocate_toward_staircase", True)),
                )

            recorder._round_exec_stats[round_id] = stats
            round_summaries.append({"round_id": round_id, "kind": kind, "instruction": instr, **stats})
            print(f"[multitask] Round {round_id} done: {stats}. Total frames so far: {recorder.frame_idx}", flush=True)

        recorder.close()

        # ---- Summary
        print("\n[multitask] SUCCESS — multi-task HRC demo finished", flush=True)
        print(f"  Output dir: {os.path.abspath(output_dir)}")
        print(f"  Total frames: {recorder.frame_idx}")
        for rs in round_summaries:
            print(f"  Round {rs['round_id']} ({rs['kind']}): success={rs.get('success', False)}, "
                  f"steps={rs.get('steps', 0)}, instr={rs['instruction']!r}")
        print(f"  Head video: {recorder.head_path}")
        print(f"  Third video: {recorder.third_path}")

    except Exception:
        print("[multitask] ERROR", flush=True)
        traceback.print_exc()
        raise
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
