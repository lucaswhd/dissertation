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
from pathlib import Path
import sys
import time
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
    extract_first_rgb,
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
from nav_online_driver import (
    build_contacts_on_playback_wrapper,
    OnlineNavDriver,
)

import h5py

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.evaluation.hrc_rollout_core import build_state23
from scripts.evaluation.hrc_rollout_core import clamp_r1pro_joint_targets
from scripts.evaluation.hrc_rollout_core import navigation_arrived
from scripts.evaluation.pi05_policy_client import Pi05PolicyClient

FOLLOW_POLICY_INSTRUCTION = "Follow the person and maintain a safe distance."


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
    # load_state(serialized=True) writes the state but does NOT flush the
    # physics/fabric read views, so get_position_orientation() would return a
    # STALE pose. Step once so the read reflects the just-loaded state — exactly
    # how the op-replay establishes it (load_state + env.step). Without this the
    # GoToTask target != the replay's true first-frame pose, causing a teleport
    # snap at the GoToTask -> ExecuteBehaviorTask boundary.
    # render_on_step(False): flush physics/pose read views WITHOUT rendering.
    # A normal og.sim.step() would draw this just-loaded FOREIGN task state into
    # the shared camera buffer; the dump/restore restores sim state but not the
    # render buffer, so that frame leaks into the recorded video (the trash/radio
    # 1-frame "flash"). PoseAPI is still invalidated by the physics step, so the
    # pose read below stays correct.
    with og.sim.render_on_step(False):
        og.sim.step()
    pos = to_numpy(env.robots[0].get_position_orientation()[0]).copy()
    quat = to_numpy(env.robots[0].get_position_orientation()[1]).copy()
    return pos, quat


def _peek_first_task_robot_pose(env: Any, episode_id: int,
                                task_rounds: List[Dict[str, Any]], raw_hdf5: str):
    """Peek the robot pose at the FIRST round's task-execution start so the robot
    can be spawned there — the episode opens at the task area instead of trekking
    from an arbitrary reset pose (⑦). Returns (pos, quat) or None.

    Snapshot/restore is handled here (peek_robot_pose_at_step perturbs the sim).
    Only replay rounds have a well-defined task-start pose; other kinds -> None
    (caller keeps the default spawn)."""
    if not task_rounds:
        return None
    rc = task_rounds[0]
    kind = rc.get("kind")
    try:
        if kind == "replay_segment":
            segs = discover_reward_segments(raw_hdf5, episode_id=episode_id)
            if rc.get("replay_all_segments", False):
                seg_start = segs[0][0]
            else:
                seg_start = segs[int(rc.get("segment_index", 0))][0]
            snap = og.sim.dump_state(serialized=True)
            try:
                pos, quat = peek_robot_pose_at_step(env, episode_id, int(seg_start))
            finally:
                og.sim.load_state(snap, serialized=True)
            return np.asarray(pos, dtype=np.float32), np.asarray(quat, dtype=np.float32)
        if kind == "replay_external_segment":
            pos, quat, _ = peek_robot_pos_from_external_hdf5(
                rc["external_hdf5"], int(rc.get("episode_id", 0)),
                int(rc.get("start_step", 0)), return_quat=True, return_joints=True)
            return np.asarray(pos, dtype=np.float32), np.asarray(quat, dtype=np.float32)
    except Exception as exc:
        print(f"[spawn-anchor] peek first-task pose failed: {exc}", flush=True)
    return None


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
    """If the robot is within min_dist of the human, nudge the human away — but
    NEVER into furniture.

    The old version stepped STRAIGHT away from the robot with no map check, which
    shoved the human into the coffee table (the proxy then knocked the radio off).
    Now each candidate dodge target is rejected if it lands within the human's
    body/proxy clearance of a baked furniture rect (``human_actor._furniture_rects``,
    set at trav_map-bake time): we try straight-back first, then progressively more
    lateral slides, and HOLD (don't move) if every direction is blocked — a brief
    robot-human near-miss is preferable to knocking a table/radio over."""
    robot_xy = to_numpy(robot.get_position_orientation()[0]).astype(np.float32)[:2]
    human_xy = human_actor.base_pos[:2].astype(np.float32)
    diff = human_xy - robot_xy
    dist = float(np.linalg.norm(diff))
    if dist >= min_dist:
        return
    base = (diff / dist) if dist > 1e-4 else np.array([1.0, 0.0], dtype=np.float32)
    # D1: prefer dodging AWAY FROM THE TASK CENTRE (away from the cans/trash the robot
    # is manipulating), not just away from the robot — so the human never backs into a
    # task object and corrupts the replay (ep0056). Only override when that direction
    # doesn't head back INTO the robot (dot with toward-robot < 0.3); else keep the
    # away-from-robot base. The furniture-clearance rotation below still applies.
    tc = getattr(human_actor, "_task_center", None)
    if tc is not None:
        v = human_xy - np.asarray(tc, dtype=np.float32)[:2]
        vn = float(np.linalg.norm(v))
        if vn > 1e-4 and float(np.dot(v / vn, -base)) < 0.3:   # -base = toward the robot
            base = (v / vn).astype(np.float32)
    rects = getattr(human_actor, "_furniture_rects", None) or []
    clear_m = 0.30  # human body/proxy half-width margin from furniture edges

    def _into_furniture(xy: np.ndarray) -> bool:
        for r in rects:
            if (r[0] - clear_m) <= xy[0] <= (r[2] + clear_m) and (r[1] - clear_m) <= xy[1] <= (r[3] + clear_m):
                return True
        return False

    direction = None
    new_xy = None
    for deg in (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0):
        a = math.radians(deg)
        ca, sa = math.cos(a), math.sin(a)
        d = np.array([base[0] * ca - base[1] * sa, base[0] * sa + base[1] * ca], dtype=np.float32)
        cand = human_xy + d * dodge_step
        if not _into_furniture(cand):
            direction, new_xy = d, cand
            break
    if new_xy is None:
        human_actor.motion_state = "hold"   # cornered by furniture -> don't push into it
        human_actor.linear_speed = 0.0
        return
    new_pos = human_actor.base_pos.copy()
    new_pos[0] = float(new_xy[0])
    new_pos[1] = float(new_xy[1])
    human_actor.base_pos = new_pos
    # Step ASIDE with a real walking motion (turn toward the dodge direction and
    # advance the walk animation) instead of a T-pose slide.
    step_dt = float(getattr(human_actor, "step_dt", 1.0 / 30.0))
    human_actor.base_yaw = _limited_turn(
        float(human_actor.base_yaw),
        math.atan2(float(direction[1]), float(direction[0])))
    human_actor.linear_speed = float(dodge_step) / max(step_dt, 1e-6)
    human_actor.motion_state = "walk"
    human_actor._last_displacement_m = float(dodge_step)
    human_actor.gait_phase = float(getattr(human_actor, "gait_phase", 0.0)) + float(dodge_step) * 2.0
    human_actor._advance_animation()
    human_actor.apply_pose()


# --------------------------------------------------------------------------- #
# Position-pinning version of run_zero_phase
# --------------------------------------------------------------------------- #

# Two separate pin buckets:
#  - _pinned_scene_objects: TASK PROPS (e.g. baskets). Dropped per-round right
#    before that round's op-replay (the replay's own load_state then manages
#    them) -- see the .clear() inside the replay_segment branch.
#  - _pinned_furniture: chairs/tables/sofas. Pinned ONCE at init and must
#    SURVIVE the per-round task-prop clear(), otherwise physics-nav in later
#    rounds (Relocate / GoToTask) would push furniture around.
_pinned_scene_objects: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
_pinned_furniture: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

# Set in main() when --online-nav is active. When non-None, navigation phases
# (GoToTask / ReturnToHuman) are driven by TRUE physics via this driver, which
# records real base-velocity actions instead of kinematic teleport + action==0.
_online_nav_driver: Any = None

# Opt-in π0.5 rollout controller.  These remain None/empty on every existing
# collection invocation, so the original scripted/expert path is unchanged.
_policy_client: Optional[Pi05PolicyClient] = None
_policy_previous_action: Optional[np.ndarray] = None
_policy_task_instruction: str = ""
_policy_rollout_metrics: Optional[Dict[str, Any]] = None

# Joint configuration captured during the episode's initial idle (the video-
# 0:02 stance). Every CUSTOM phase (ReturnToHuman / FaceHuman / Relocate
# follow / waits) returns the arms+trunk to this one canonical posture;
# GoToTask instead settles toward the task segment's first frame (unchanged).
_HOME_POSTURE_Q: Any = None


def _policy_observation(env: Any) -> Dict[str, Any]:
    """Return a fresh observation without advancing simulation time."""
    obs = getattr(env, "current_obs", None)
    if isinstance(obs, dict):
        return obs
    get_obs = getattr(getattr(env, "env", env), "get_obs", None)
    if get_obs is None:
        raise RuntimeError("policy rollout cannot obtain the current observation")
    obs = get_obs()
    if isinstance(obs, tuple):
        obs = obs[0]
    if not isinstance(obs, dict):
        raise RuntimeError(f"policy rollout expected dict observation, got {type(obs).__name__}")
    return obs


def _policy_step(
    env: Any,
    recorder: DemoRecorder,
    phase: str,
    round_id: int,
    *,
    human_actor: Any = None,
    advance_human: bool = True,
    task_instruction: Optional[str] = None,
) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, Any]]:
    """Query π0.5 once, execute its action through ``env.step``, and record it."""
    global _policy_previous_action
    if _policy_client is None:
        raise RuntimeError("_policy_step called without --policy-rollout")

    robot = env.robots[0]
    current_obs = _policy_observation(env)
    head, head_key = extract_first_rgb(current_obs)
    if head is None:
        raise RuntimeError("π0.5 rollout found no head RGB observation")
    joints = to_numpy(robot.get_joint_positions()).astype(np.float32)
    finger_parts = [
        to_numpy(robot.get_joint_positions())[robot.gripper_control_idx[arm]]
        for arm in robot.arm_names
    ]
    fingers = np.concatenate(finger_parts).astype(np.float32)
    state = build_state23(joints, fingers, _policy_previous_action)
    action = _policy_client.act(
        state=state,
        head=head,
        task=_policy_task_instruction if task_instruction is None else task_instruction,
        phase=phase,
    )
    # Saturate torso/arm targets to the robot limits instead of aborting the
    # rollout: a marginally out-of-range policy prediction should clamp (like a
    # real robot), not crash a 600-step closed-loop episode.
    action = clamp_r1pro_joint_targets(
        action,
        to_numpy(robot.joint_lower_limits),
        to_numpy(robot.joint_upper_limits),
    )

    _restore_pinned_objects(env)
    if human_actor is not None and advance_human:
        human_actor.step(phase, robot)
    obs, reward, terminated, truncated, info = env.step(action, n_render_iterations=1)
    if not isinstance(info, dict):
        info = {}
    info["reward"] = float(reward) if reward is not None else 0.0
    info["terminated"] = bool(terminated) if terminated is not None else False
    info["truncated"] = bool(truncated) if truncated is not None else False
    info["policy_rollout"] = True
    info["policy_head_key"] = head_key
    _update_policy_contact_metrics(env, phase, round_id)
    _record_policy_gripper_command(phase, round_id, action)
    recorder.record(phase, obs, info, round_id=round_id, action=action)
    _policy_previous_action = action.copy()
    return obs, action, info


def _contact_paths(entity: Any) -> set[str]:
    paths: set[str] = set()
    try:
        for contact in entity.contact_list():
            paths.add(str(contact.body0))
            paths.add(str(contact.body1))
    except Exception:
        pass
    return paths


def _update_policy_contact_metrics(env: Any, phase: str, round_id: int) -> None:
    """Collect live contacts without feeding privileged values to the policy."""
    metrics = _policy_rollout_metrics
    if metrics is None:
        return
    metrics["phase_frames"][phase] = metrics["phase_frames"].get(phase, 0) + 1
    robot = env.robots[0]
    robot_path = str(getattr(robot, "prim_path", ""))
    task_names = {
        "pickup": ("can_of_soda_113", "can_of_soda_114", "can_of_soda_115"),
        "radio": ("radio_89",),
    }
    active_task = "pickup" if int(round_id) == 0 else "radio"
    active_paths = set()
    if phase == "ExecuteBehaviorTask":
        touched = False
        # Distance from either hand to the nearest task object. "contact frames = 0"
        # cannot distinguish "reached toward the can and missed" from "never reached",
        # which is the whole question; this is diagnostic only and never reaches pi0.5.
        hand_positions = []
        for arm in getattr(robot, "arm_names", []) or []:
            try:
                link = robot.eef_links[arm]
            except (AttributeError, KeyError, TypeError):
                continue
            if link is not None:
                hand_positions.append(to_numpy(link.get_position_orientation()[0])[:3])
        frame_index = metrics["phase_frames"][phase]
        for name in task_names[active_task]:
            obj = env.scene.object_registry("name", name)
            if obj is None:
                continue
            obj_path = str(getattr(obj, "prim_path", ""))
            active_paths.add(obj_path)
            if hand_positions:
                obj_xyz = to_numpy(obj.get_position_orientation()[0])[:3]
                closest = float(min(float(np.linalg.norm(h - obj_xyz)) for h in hand_positions))
                best = metrics["hand_object_min_distance_m"].get(name)
                if best is None or closest < best:
                    metrics["hand_object_min_distance_m"][name] = closest
                    metrics["hand_object_min_distance_frame"][name] = int(frame_index)
            paths = _contact_paths(obj)
            if robot_path and any(path.startswith(robot_path) for path in paths):
                touched = True
                if metrics["first_contact_frame"].get(active_task) is None:
                    metrics["first_contact_frame"][active_task] = int(frame_index)
            if active_task == "pickup":
                z = float(to_numpy(obj.get_position_orientation()[0])[2])
                initial = metrics["pickup_initial_z"].setdefault(name, z)
                previous_max = metrics["pickup_max_z"].get(name, z)
                metrics["pickup_max_z"][name] = max(previous_max, z)
                if z >= initial + 0.08:
                    metrics["pickup_lifted"][name] = True
                if metrics["pickup_lifted"].get(name) and z <= metrics["pickup_max_z"][name] - 0.08:
                    metrics["pickup_drop_detected"][name] = True
        if touched:
            metrics["task_contact_frames"][active_task] += 1

    robot_contacts = _contact_paths(robot)
    unexpected = set()
    for path in robot_contacts:
        lowered = path.lower()
        if robot_path and path.startswith(robot_path):
            continue
        if any(token in lowered for token in ("floor", "ground_plane")):
            continue
        if any(task_path and path.startswith(task_path) for task_path in active_paths):
            continue
        unexpected.add(path)
    if unexpected:
        metrics["collision_frames"] += 1
        metrics["collision_bodies"].update(unexpected)



def _record_policy_gripper_command(phase: str, round_id: int, action: np.ndarray) -> None:
    """Log every gripper open/close the policy commands, with the hand-object gap.

    Separates "right moment, wrong place" from "opening and closing at random":
    the offline audit measured radio-round flip precision at 0.05, so a flip alone
    is not evidence the policy is doing something meaningful.
    """
    metrics = _policy_rollout_metrics
    if metrics is None or phase != "ExecuteBehaviorTask":
        return
    current = np.sign(np.asarray(action, dtype=np.float32)[[14, 22]])
    previous = (
        None if _policy_previous_action is None
        else np.sign(np.asarray(_policy_previous_action, dtype=np.float32)[[14, 22]])
    )
    if previous is None:
        return
    for column, side in enumerate(("left", "right")):
        if current[column] == previous[column]:
            continue
        distances = metrics["hand_object_min_distance_m"]
        metrics["gripper_commands"].append(
            {
                "frame": int(metrics["phase_frames"].get(phase, 0)),
                "round_id": int(round_id),
                "gripper": side,
                "action": "close" if current[column] < 0 else "open",
                "nearest_task_object_m": (min(distances.values()) if distances else None),
            }
        )


def _record_policy_arrival(round_id: int, phase: str, distance_m: float) -> None:
    """Record one frozen endpoint-distance diagnostic without feeding it to π0.5."""
    if _policy_rollout_metrics is None:
        return
    arrived = navigation_arrived(phase, distance_m)
    key = f"round{int(round_id)}_{phase}"
    _policy_rollout_metrics["navigation_arrival"][key] = {
        "phase": phase,
        "round_id": int(round_id),
        "distance_m": float(distance_m),
        "arrived": bool(arrived),
    }
    print(f"[policy-arrival] {key}: distance={distance_m:.3f}m arrived={arrived}", flush=True)


def _parse_execute_budget(value: Any) -> List[int]:
    """Parse the ExecuteBehaviorTask budget into a per-round list of positive ints.

    Accepts an int, or a string with one value per round ("3000,2400" = round 0
    gets 3000, round 1 gets 2400). The last value covers any further round.
    """
    if isinstance(value, (list, tuple)):
        parts = [str(item) for item in value]
    else:
        parts = [part for part in str(value).split(",") if part.strip()]
    if not parts:
        raise ValueError("policy_execute_steps must not be empty")
    budget = [int(float(part)) for part in parts]
    if any(steps <= 0 for steps in budget):
        raise ValueError(f"policy_execute_steps must all be positive, got {budget}")
    return budget


def _round_execute_steps(budget: List[int], round_id: int) -> int:
    """Budget for this round; rounds past the end reuse the last value."""
    return budget[min(int(round_id), len(budget) - 1)]


def _policy_budget_steps(steps: int, max_seconds: Optional[float], label: str):
    """Yield step indices under both the step budget and an optional wall-clock cap.

    The cap only guards against a policy that stalls the run (each sim step plus
    inference is far slower than real time); it never shortens a phase that is
    progressing within its wall-clock allowance.
    """
    total = max(int(steps), 0)
    started = time.time()
    for index in range(total):
        if max_seconds is not None and (time.time() - started) > max_seconds:
            print(f"[policy] {label}: wall-clock cap {max_seconds:.0f}s hit at "
                  f"step {index}/{total}, ending phase early", flush=True)
            return
        yield index



class SubInstructionSchedule:
    """Walk the measured sub-goal sequence during a policy-controlled Execute phase.

    Group B trained with 7 sub-instructions on Execute frames; the round's long
    instruction never appears there, so replaying it would be out of distribution.
    This advances through the frozen schedule instead.

    A segment ends when the policy's OWN gripper command performs the flip that
    segment ends with (never object poses -- that would feed privileged state into
    the measurement), or on a timeout taken from the 90th percentile of the 45
    demos. The timeout is generous on purpose: advancing early truncates a reaching
    motion and corrupts exactly the signal being measured, while advancing late only
    costs budget. ``min_frames`` (10th percentile) stops a gripper-spamming policy
    from racing through the schedule faster than the fastest real demonstration.
    """

    _EVENT_DIM = {"L": 14, "R": 22}

    def __init__(self, segments: List[Dict[str, Any]], *, stop_at_terminal: bool = True):
        if not segments:
            raise ValueError("sub-instruction schedule needs at least one segment")
        self._segments = segments
        self._stop_at_terminal = stop_at_terminal
        self._index = 0
        self._frames_in_segment = 0
        self._prev_gripper: Optional[np.ndarray] = None
        self.transitions: List[Dict[str, Any]] = []

    @property
    def instruction(self) -> str:
        return str(self._segments[self._index]["label"])

    @property
    def index(self) -> int:
        return self._index

    @property
    def finished(self) -> bool:
        """True once the terminal ('Stand by') segment is reached.

        That segment keeps only ~21% anchor coverage in group B (idle tails were
        excluded from training), so its behaviour is not meaningful to score.
        """
        return self._stop_at_terminal and self._segments[self._index]["ends_with"] is None

    def _fired(self, action: Optional[np.ndarray]) -> bool:
        event = self._segments[self._index]["ends_with"]
        if event is None or action is None:
            return False
        side, direction = event[0], event[1:]
        dim = self._EVENT_DIM[side]
        current = np.sign(np.asarray(action, dtype=np.float32)[[14, 22]])
        if self._prev_gripper is None:
            return False
        column = 0 if dim == 14 else 1
        if current[column] == self._prev_gripper[column]:
            return False
        return (current[column] < 0) if direction == "close" else (current[column] > 0)

    def update(self, action: Optional[np.ndarray], frame: int) -> None:
        """Advance the schedule after one executed policy action."""
        segment = self._segments[self._index]
        self._frames_in_segment += 1
        triggered = (
            self._fired(action) and self._frames_in_segment >= int(segment["min_frames"])
        )
        timed_out = self._frames_in_segment >= int(segment["advance_frames"])
        if action is not None:
            self._prev_gripper = np.sign(np.asarray(action, dtype=np.float32)[[14, 22]])
        if not (triggered or timed_out):
            return
        if self._index + 1 >= len(self._segments):
            return
        self.transitions.append(
            {
                "frame": int(frame),
                "from_index": self._index,
                "from_label": segment["label"],
                "frames_in_segment": self._frames_in_segment,
                "reason": "gripper_event" if triggered else "timeout",
            }
        )
        print(f"[subinstr] frame {frame}: segment {self._index} -> {self._index + 1} "
              f"after {self._frames_in_segment} frames ({'event' if triggered else 'timeout'})",
              flush=True)
        self._index += 1
        self._frames_in_segment = 0


def load_subinstruction_schedule(path: Any, round_id: int) -> "SubInstructionSchedule":
    with open(str(path), encoding="utf-8") as stream:
        data = json.load(stream)
    segments = data["rounds"][str(int(round_id))]
    return SubInstructionSchedule(segments)


def run_policy_phase(
    env: Any,
    recorder: DemoRecorder,
    phase: str,
    steps: int,
    round_id: int,
    *,
    human_actor: Any = None,
    max_seconds: Optional[float] = None,
    schedule: Optional["SubInstructionSchedule"] = None,
) -> int:
    """Run a budgeted active phase entirely under π0.5 control; returns steps run."""
    executed = 0
    for step_index in _policy_budget_steps(steps, max_seconds, f"{phase}#r{round_id}"):
        _, action, _ = _policy_step(
            env,
            recorder,
            phase,
            round_id,
            human_actor=human_actor,
            advance_human=True,
            task_instruction=None if schedule is None else schedule.instruction,
        )
        executed += 1
        if schedule is not None:
            schedule.update(action, step_index)
            if schedule.finished:
                print(f"[subinstr] terminal segment reached at frame {step_index}; "
                      f"ending {phase} (its labels are only ~21% anchor-covered in training)",
                      flush=True)
                break
    return executed


def _restore_pinned_objects(env: Any) -> None:
    """Re-set pinned objects (task props + furniture) to their captured poses."""
    for bucket in (_pinned_scene_objects, _pinned_furniture):
        for name, (pos, quat) in bucket.items():
            obj = env.scene.object_registry("name", name)
            if obj is not None:
                obj.set_position_orientation(pos, quat)
                obj.keep_still()


def _build_hold_action(robot: Any) -> th.Tensor:
    """Build an action that keeps the robot at its CURRENT pose: every position
    controller commands its current joint positions (absolute hold), velocity
    controllers command 0, grippers no-op. Uses each controller's own
    compute_no_op_action so this is robot/controller agnostic.

    Used by the wait phases (TaskSegmentDone / WaitNearHuman / etc.) and the
    end-of-relocation human-turn: under the arms' absolute position control,
    recording action=0 would mean "command arms to the ZERO pose" at deployment
    (the P2 bug + the visible reverse arm bounce after Execute). Recording the
    hold action instead means deployment just keeps the arms where they are."""
    action = th.zeros(robot.action_dim, dtype=th.float32)
    ctrl_dict = robot.get_control_dict()
    for name, controller in robot._controllers.items():
        partial = controller.compute_no_op_action(ctrl_dict)
        action[robot.controller_action_idx[name]] = partial
    return action


def _wait_robot_base_stable(env: Any, recorder: DemoRecorder, human_actor: Any,
                           round_id: int, phase: str = "Relocate",
                           stable_frames: int = 12, max_frames: int = 90,
                           xy_eps: float = 0.01, yaw_eps_deg: float = 1.0) -> None:
    """Patch 2: hold the robot base still until it is physically settled (base xy
    AND yaw delta below eps for ``stable_frames`` consecutive frames) before the
    human starts leading the Relocate walk — fixes "robot hasn't settled, human
    already walking". Feeds a hold-action through ``env.step`` (red line 1 intact);
    the human is kept put (``apply_pose``). On timeout it WARNs and CONTINUES —
    never raises (batch collection must not be interrupted by physics jitter).
    Records frames under ``phase``."""
    robot = env.robots[0]
    hold = _build_hold_action(robot)
    hold_np = to_numpy(hold).astype(np.float32)
    prev_xy = to_numpy(robot.get_position_orientation()[0])[:2].astype(np.float64)
    prev_yaw = _yaw_from_quat(to_numpy(robot.get_position_orientation()[1]))
    yaw_eps = math.radians(float(yaw_eps_deg))
    stable = 0
    f = 0
    for f in range(max(max_frames, 1)):
        robot.keep_still()
        _restore_pinned_objects(env)
        if human_actor is not None:
            human_actor.apply_pose()
        obs, rew, term, trunc, info = env.step(hold, n_render_iterations=1)
        rinfo = {"reward": float(rew) if rew is not None else 0.0,
                 "terminated": False, "truncated": False}
        recorder.record(phase, obs, rinfo, round_id=round_id, action=hold_np)
        cur_xy = to_numpy(robot.get_position_orientation()[0])[:2].astype(np.float64)
        cur_yaw = _yaw_from_quat(to_numpy(robot.get_position_orientation()[1]))
        d_xy = float(np.linalg.norm(cur_xy - prev_xy))
        d_yaw = abs(_wrap_to_pi(cur_yaw - prev_yaw))
        prev_xy, prev_yaw = cur_xy, cur_yaw
        if d_xy < xy_eps and d_yaw < yaw_eps:
            stable += 1
            if stable >= stable_frames:
                print(f"[relocation] robot base stable gate passed after {f + 1} frames", flush=True)
                return
        else:
            stable = 0
    print(f"[relocation] WARN: robot base stable gate TIMEOUT after {f + 1} frames "
          f"(continuing anyway)", flush=True)


# Robot<->human follow/stand-off distance band, per Habitat 3.0 SocialNavReward
# (safe_dis_min=1.0, safe_dis_max=2.0). We keep the robot near the band midpoint.
HABITAT_SAFE_DIS_MIN = 1.0
HABITAT_SAFE_DIS_MAX = 2.0
# Habitat 3.0 SocialNavStats.robot_face_human_threshold: dot(robot_forward,
# robot->human) > 0.5 (~60° cone) counts as "facing the human".
HABITAT_FACE_DOT_THRESH = 0.5
# RC2: hard floor on the robot-human distance the reach-gate will drive to. Above the QC
# hit-human threshold (0.55m) so a returning robot never walks INTO the human even when the
# standoff sits on the human's far side (ep0035 0.40m / ep0052 0.46m).
REACH_MIN_HUMAN_GAP = 0.65


def _drive_until_reached_facing(env: Any, recorder: DemoRecorder, human_actor: Any,
                                round_id: int, human_xy: np.ndarray,
                                standoff_target: np.ndarray, phase: str,
                                max_frames: int = 60,
                                face_thresh: float = HABITAT_FACE_DOT_THRESH) -> bool:
    """Reactively drive the robot with ``follow_step`` until it is back at the
    human in the Habitat ``_reach_human`` sense — distance inside
    [SAFE_DIS_MIN, SAFE_DIS_MAX] AND facing (dot > face_thresh) — or the frame
    budget runs out. Records every frame under *phase*. Returns True if it
    converged. Online only (no-op offline). When already reached+facing it
    returns in 0 frames, so it's cheap to call as a post-condition.

    All motion is a real velocity action through ``env.step`` (red line 1); the
    kinematic human is held in place each frame."""
    if _online_nav_driver is None:
        return False
    robot = env.robots[0]
    human_2d = np.asarray(human_xy, dtype=np.float64)[:2]
    stand_2d = np.asarray(standoff_target, dtype=np.float64)[:2]
    # RC2: the trav_map (walls + baked furniture) lets us clamp the drive so the corrective
    # follow can't ram a wall reaching the standoff, and a human-distance floor stops the
    # robot over-approaching the human (ep0035 0.40m / ep0052 0.46m were driven here).
    trav_map = human_actor._try_get_trav_map() if human_actor is not None else None
    # Re-pin arms at the CURRENT pose so follow_step's _empty_action holds them.
    _online_nav_driver.pin_arm_hold()
    _prev_xy = None
    _stuck = 0
    for f in range(max(max_frames, 1)):
        robot_xy = to_numpy(robot.get_position_orientation()[0]).astype(np.float64)[:2]
        d = float(np.linalg.norm(robot_xy - human_2d))
        dot = _robot_human_dot(robot, human_2d)
        # Never approach closer than REACH_MIN_HUMAN_GAP: below the band's 1.0m floor the
        # robot is already close enough to face; driving on just walks into the human.
        too_close = d < REACH_MIN_HUMAN_GAP
        reached = (HABITAT_SAFE_DIS_MIN <= d <= HABITAT_SAFE_DIS_MAX) or too_close
        if reached and dot > face_thresh:
            print(f"[{phase}] reached+facing (d={d:.2f}m dot={dot:.2f}) after "
                  f"{f} corrective frames", flush=True)
            return True
        # STUCK GUARD (user-approved, shared with the RC1 mechanism): if the robot has stopped
        # moving while wedged against furniture, do NOT keep pushing / hard-rotating to face —
        # rotating in place while wedged is exactly what tips it (ep0031 84°). Translate to
        # clearer space first (face_xy=None -> no hard face-rotation), then resume facing.
        _stuck = _stuck + 1 if (_prev_xy is not None
                                and float(np.linalg.norm(robot_xy - _prev_xy)) < 0.01) else 0
        _prev_xy = robot_xy.copy()
        _tilt_deg = _quat_tilt_deg(robot.get_position_orientation()[1])
        _wedged = (trav_map is not None
                   and not _trav_xy_ok(robot_xy, trav_map, floor=0, clearance=0.30))
        # A LEANING base must NOT rotate: rotating a tilted/wedged base is what drives it from
        # ~9° to 80°+ (ep0031). Whenever it is leaning past normal (>8°) — regardless of the
        # instantaneous clearance, which flip-flops around the threshold — OR wedged-and-stuck,
        # translate to clear space with vyaw≡0 (or hold if boxed) until it is upright again.
        if _tilt_deg > 8.0 or (_wedged and _stuck >= 4):
            esc = _escape_from_obstacle(robot_xy, trav_map) if trav_map is not None else None
            if human_actor is not None:
                human_actor.apply_pose()
            _restore_pinned_objects(env)
            obs, action = _online_nav_driver.escape_step(nudge_xy=esc)
            _restore_pinned_objects(env)
            recorder.record(phase, obs, {"reward": 0.0, "terminated": False,
                                         "truncated": False}, round_id=round_id, action=action)
            _stuck = 0
            continue
        # Out of band -> drive to the clear standoff (clamped so we don't ram a wall);
        # in band / too close but not facing -> target self so follow_step rotates in
        # place toward the human.
        if not reached:
            tgt = (_clamp_target_to_wall_clear(robot_xy, stand_2d, trav_map, clearance=0.25)
                   if trav_map is not None else stand_2d)
        else:
            tgt = robot_xy
        if human_actor is not None:
            human_actor.apply_pose()
        _restore_pinned_objects(env)
        obs, action = _online_nav_driver.follow_step(tgt, face_xy=human_2d)
        _restore_pinned_objects(env)
        info = {"reward": 0.0, "terminated": False, "truncated": False}
        recorder.record(phase, obs, info, round_id=round_id, action=action)
    print(f"[{phase}] WARN: reached+facing not achieved in {max_frames} corrective "
          f"frames (continuing anyway)", flush=True)
    return False


def _settle_and_face_human(env: Any, recorder: DemoRecorder, human_actor: Any,
                           round_id: int, human_xy: np.ndarray, standoff_target: np.ndarray,
                           phase: str = "Relocate",
                           face_thresh: float = HABITAT_FACE_DOT_THRESH,
                           max_correct_frames: int = 60, stable_frames: int = 12,
                           max_frames: int = 90, xy_eps: float = 0.01,
                           yaw_eps_deg: float = 1.0) -> None:
    """Habitat ``_reach_human`` style RELOCATE gate (supersedes the bare base-
    stability wait): ACTIVELY drive the robot to reached+facing the human, THEN
    settle the base, before the human starts leading. ``ReturnToHuman``/
    ``FaceHuman`` normally leave it already reached+facing so the corrective loop
    passes in 0 frames; it only kicks in when the geometric standoff landed too
    far / blocked or the facing drifted (HANDOFF §5 Patch2 ⚠: the old gate only
    proved "stopped", not "in position"). Never raises (batch-safe)."""
    _drive_until_reached_facing(env, recorder, human_actor, round_id, human_xy,
                                standoff_target, phase=phase,
                                max_frames=max_correct_frames, face_thresh=face_thresh)
    # Velocity residual from the corrective follow_step must die before the walk.
    _wait_robot_base_stable(env, recorder, human_actor, round_id, phase=phase,
                            stable_frames=stable_frames, max_frames=max_frames,
                            xy_eps=xy_eps, yaw_eps_deg=yaw_eps_deg)


def run_zero_phase_pinned(env: Any, recorder: DemoRecorder, phase: str, steps: int,
                          round_id: int, human_actor: Any = None) -> None:
    """Like run_zero_phase but re-sets robot position every frame so physics
    can't drift it away during wait phases.

    Records an arm-HOLD action (current joint positions) instead of zeros so the
    BC policy doesn't learn "during wait phases command the arms to zero pose"
    (the P2 trap + the visible reverse arm bounce right after Execute)."""
    global _policy_previous_action
    robot = env.robots[0]
    pin_pos = to_numpy(robot.get_position_orientation()[0]).copy()
    pin_quat = to_numpy(robot.get_position_orientation()[1]).copy()
    hold_action = _build_hold_action(robot)
    hold_action_np = to_numpy(hold_action).astype(np.float32)
    for _ in range(max(steps, 1)):
        robot.set_position_orientation(pin_pos, pin_quat)
        robot.keep_still()
        _restore_pinned_objects(env)
        if human_actor is not None:
            human_actor.step(phase, robot)
        obs, rew, term, trunc, info = env.step(hold_action, n_render_iterations=1)
        info["reward"] = float(rew) if rew is not None else 0.0
        info["terminated"] = bool(term) if term is not None else False
        info["truncated"] = bool(trunc) if trunc is not None else False
        recorder.record(phase, obs, info, round_id=round_id, action=hold_action_np)
        if _policy_client is not None:
            _policy_previous_action = hold_action_np.copy()


def _settle_arms_to_home(env: Any, recorder: DemoRecorder, phase: str,
                         round_id: int, human_actor: Any = None,
                         ramp_frames: int = 45) -> None:
    """Smoothly ramp arms+trunk back to the HOME posture (_HOME_POSTURE_Q, the
    video-0:02 stance) via the driver's settle_to_config (real absolute-position
    actions through env.step), recording the ramp under *phase*. No-op when
    already at home, so it's cheap to call at every custom-phase start.
    GoToTask must NOT call this: its arms settle toward the upcoming task
    segment's FIRST FRAME instead (existing settle_to_config with the replay
    target, unchanged)."""
    if _online_nav_driver is None or _HOME_POSTURE_Q is None:
        return
    robot = env.robots[0]
    q_now = to_numpy(robot.get_joint_positions())
    q_home = to_numpy(_HOME_POSTURE_Q)
    dofs: List[int] = []
    for nm, ctrl in robot.controllers.items():
        if nm == "base" or nm.startswith("gripper"):
            continue  # base untouched; grippers keep their carry state
        dofs.extend(int(i) for i in to_numpy(ctrl.dof_idx).reshape(-1))
    if not dofs:
        return
    dev = float(np.abs(q_now[dofs] - q_home[dofs]).max())
    if dev < math.radians(3.0):
        return  # already home (e.g. FaceHuman right after ReturnToHuman)
    print(f"[home-posture] {phase}: arms/trunk {math.degrees(dev):.0f}° off the "
          f"0:02 home stance -> ramping back over {ramp_frames} frames", flush=True)

    def _on_frame(obs: Any, action: np.ndarray) -> None:
        _restore_pinned_objects(env)
        if human_actor is not None:
            human_actor.step(phase, robot)
        info = {"reward": 0.0, "terminated": False, "truncated": False}
        recorder.record(phase, obs, info, round_id=round_id, action=action)

    _online_nav_driver.settle_to_config(_HOME_POSTURE_Q, ramp_frames=ramp_frames,
                                        hold_frames=0, on_frame=_on_frame)


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


def _project_point_to_path_arc(path: List[np.ndarray], cum_dist: List[float],
                               xy: np.ndarray) -> float:
    """Return the arc-length (cumulative distance) of the closest point on the
    polyline @path to the 2-D point @xy. Used to track how far the (physics-
    driven) robot has progressed along a planned path."""
    p = np.asarray(xy, dtype=np.float64)[:2]
    best_arc = 0.0
    best_dist = float("inf")
    for i in range(len(path) - 1):
        a = np.asarray(path[i], dtype=np.float64)[:2]
        b = np.asarray(path[i + 1], dtype=np.float64)[:2]
        ab = b - a
        seg_len2 = float(ab @ ab)
        if seg_len2 < 1e-9:
            continue
        t = float(np.clip((p - a) @ ab / seg_len2, 0.0, 1.0))
        proj = a + t * ab
        dist = float(np.linalg.norm(p - proj))
        if dist < best_dist:
            best_dist = dist
            best_arc = cum_dist[i] + t * math.sqrt(seg_len2)
    return best_arc


# --------------------------------------------------------------------------- #
# 2-D scene obstacle map (inflated AABB rectangles) + straight-leg corner
# routing. Used by the relocation follow so the robot tracks the HUMAN'S
# route in straight legs with discrete turns and never cuts a corner into
# furniture (center+radius clearance is wrong for large tables; the AABB is
# the actual footprint).
# --------------------------------------------------------------------------- #

_ROBOT_INFLATE_M = 0.55  # robot half-footprint (~0.3m) + safety margin
_ROBOT_STANDOFF_CLEAR = 0.55  # min clearance when picking a stand-off near the human
_HUMAN_EROSION_M = 0.35  # human body-width trav_map erosion (all human nav phases)
# When the human LEADS the leash-following robot during Relocate, it must route on
# the ROBOT's body width, not its own — else it slips through gaps the trailing
# robot (0.55m inflated footprint) can't follow, wedging it (ep0058: human squeezed
# 0.30m past a mirror, robot stuck 14.7s at 0.09m). Plan the lead path with a wider
# erosion so every gap the human takes is one the robot can trail through.
_RELOC_LEAD_EROSION_M = 0.50  # human LEAD path plans on ~robot body-width so it USUALLY
# avoids leading the trailing robot into gaps it can't follow. NOT a hard guarantee — a
# sub-body-width gap (e.g. the mirror's south side) still gets taken when it's the
# shortest route; the robot then jams briefly and the C safety-net collapses it in ~5s.
# Relocation navigation precision is intentionally NOT chased further: the bar is
# data-correctness (no object knocked, no NaN, no infinite freeze), not nav polish.
# 58c verified 0.50 does not harm normal follow (robot tracked 82%, gap stable).


def _clamp_target_to_wall_clear(a, b, trav_map, clearance: float = 0.35, step: float = 0.1):
    """Walk from robot pos *a* toward leash target *b*; return the FARTHEST point along
    the segment that keeps the robot >= *clearance* from walls+baked-furniture on the
    trav_map. The leash's furniture-rect routing (scene_rects) does NOT cover WALLS or
    skipped furniture like the porch mirror, so the reactive follow drove the robot
    straight into a wall while chasing the human (ep0058 wall-hit, ep0063 mirror-fall).
    Clamping the target to the last wall-clear point stops that; if the robot is
    already wall-tight (e.g. mid-doorway) we return *b* unchanged so it never freezes."""
    if trav_map is None:
        return b
    a2 = np.asarray(a, dtype=np.float64)[:2]
    b2 = np.asarray(b, dtype=np.float64)[:2]
    try:
        if not _trav_xy_ok(a2, trav_map, floor=0, clearance=clearance):
            return b2                      # already tight -> don't freeze, let leash proceed
        d = b2 - a2
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            return b2
        u = d / L
        last = a2.copy()
        k = float(step)
        while k <= L:
            p = a2 + u * k
            if not _trav_xy_ok(p, trav_map, floor=0, clearance=clearance):
                return last
            last = p
            k += step
    except Exception:
        return b2
    return b2


def _quat_tilt_deg(q) -> float:
    """max(|roll|, |pitch|) in degrees from an xyzw quaternion — how far the base is
    leaning off level (used by the stuck guard to catch an in-progress tip)."""
    q = to_numpy(q).astype(np.float64)
    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    roll = math.degrees(math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx)))))
    return max(abs(roll), abs(pitch))


def _escape_from_obstacle(robot_xy, trav_map, step: float = 0.35):
    """Return a nearby point in CLEARER space to un-wedge a robot stuck against furniture
    BEFORE a hard in-place rotation tips it (ep0031 84°, and the RC1 follow falls share this
    'wedged + hard turn -> tip' mechanism). Scans 16 directions; returns the first that is
    trav-clear (walls + baked furniture) at 0.30m, else at 0.22m. None if fully boxed in."""
    if trav_map is None:
        return None
    rxy = np.asarray(robot_xy, dtype=np.float64)[:2]
    for clr in (0.30, 0.22):
        for a in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False):
            cand = rxy + np.array([math.cos(a), math.sin(a)], dtype=np.float64) * float(step)
            if _trav_xy_ok(cand, trav_map, floor=0, clearance=clr):
                return cand
    return None


def _build_scene_rects(avoid_objects, inflate: float = _ROBOT_INFLATE_M,
                       tag: str = "scene-map") -> list:
    """World-frame 2-D obstacle map: each avoid object's AABB inflated by the
    robot half-footprint, as (xmin, ymin, xmax, ymax). Logged so every run
    records the exact map it navigated against."""
    rects = []
    for obj in (avoid_objects or []):
        try:
            lo, hi = obj.aabb
            lo = to_numpy(lo).astype(np.float64)
            hi = to_numpy(hi).astype(np.float64)
        except Exception:
            try:
                c = to_numpy(obj.get_position_orientation()[0]).astype(np.float64)
                lo, hi = c - 0.3, c + 0.3
            except Exception:
                continue
        rect = (float(lo[0]) - inflate, float(lo[1]) - inflate,
                float(hi[0]) + inflate, float(hi[1]) + inflate)
        rects.append(rect)
        print(f"[{tag}] {getattr(obj, 'name', '?')}: "
              f"aabb x[{lo[0]:.2f},{hi[0]:.2f}] y[{lo[1]:.2f},{hi[1]:.2f}] "
              f"-> inflated x[{rect[0]:.2f},{rect[2]:.2f}] y[{rect[1]:.2f},{rect[3]:.2f}]",
              flush=True)
    return rects


def _point_in_rect(p, rect) -> bool:
    return rect[0] <= float(p[0]) <= rect[2] and rect[1] <= float(p[1]) <= rect[3]


def _push_out_of_rects(p, rects) -> np.ndarray:
    """If *p* sits inside an inflated rect, move it out through the nearest
    edge (safety net for waypoints planned with the HUMAN's smaller
    clearance, e.g. a path point 0.79m from a chair center)."""
    p = np.asarray(p, dtype=np.float64)[:2].copy()
    for rect in rects:
        if _point_in_rect(p, rect):
            cand = [(p[0] - rect[0], np.array([rect[0] - 1e-3, p[1]])),
                    (rect[2] - p[0], np.array([rect[2] + 1e-3, p[1]])),
                    (p[1] - rect[1], np.array([p[0], rect[1] - 1e-3])),
                    (rect[3] - p[1], np.array([p[0], rect[3] + 1e-3]))]
            p = min(cand, key=lambda c: c[0])[1]
    return p


def _seg_hits_rect(a, b, rect) -> bool:
    """Liang-Barsky clip test: does segment a->b pass through the rect?"""
    x0, y0 = float(a[0]), float(a[1])
    dx, dy = float(b[0]) - x0, float(b[1]) - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - rect[0]), (dx, rect[2] - x0),
                 (-dy, y0 - rect[1]), (dy, rect[3] - y0)):
        if abs(p) < 1e-12:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


def _route_around_rects(a, b, rects, depth: int = 6) -> list:
    """Waypoints (excluding *a*, ending at *b*) for a straight-leg route a->b
    detouring around inflated obstacle rects via their corners — straight
    lines with discrete turns, no arcs. Rects containing an endpoint are
    ignored (the per-frame _push_out_of_rects safety net covers those)."""
    a = np.asarray(a, dtype=np.float64)[:2]
    b = np.asarray(b, dtype=np.float64)[:2]
    if depth <= 0:
        return [b]
    hit = None
    for rect in rects:
        if (not _point_in_rect(a, rect) and not _point_in_rect(b, rect)
                and _seg_hits_rect(a, b, rect)):
            hit = rect
            break
    if hit is None:
        return [b]
    eps = 0.08
    corners = [np.array([hit[0] - eps, hit[1] - eps]),
               np.array([hit[2] + eps, hit[1] - eps]),
               np.array([hit[2] + eps, hit[3] + eps]),
               np.array([hit[0] - eps, hit[3] + eps])]
    best = None
    for c in corners:
        if any(_point_in_rect(c, r) for r in rects):
            continue  # corner sits inside an overlapping rect (chair under table)
        if _seg_hits_rect(a, c, hit):
            continue  # this corner doesn't clear the rect from our side
        cost = float(np.linalg.norm(c - a) + np.linalg.norm(c - b))
        if best is None or cost < best[0]:
            best = (cost, c)
    if best is None:
        return [b]  # boxed in: fall back to the straight leg
    c = best[1]
    return (_route_around_rects(a, c, rects, depth - 1)
            + _route_around_rects(c, b, rects, depth - 1))


def _los_shortcut(pts, rects):
    """Greedy line-of-sight string-pulling on the inflated-AABB obstacle map.

    Collapses the grid-A* staircase the robot path inherits into long straight
    legs: keep a point, then jump to the FARTHEST later point whose connecting
    segment crosses no rect (``_seg_hits_rect``). Because the test uses the same
    inflated rects the corner-routing used, every kept segment is guaranteed
    obstacle-clear — shortcutting can only shorten, never re-introduce clipping."""
    if pts is None or len(pts) <= 2:
        return list(pts) if pts is not None else []
    p2 = [np.asarray(p, dtype=np.float64)[:2] for p in pts]
    out = [p2[0]]
    i, n = 0, len(p2)
    while i < n - 1:
        j = n - 1
        while j > i + 1 and any(_seg_hits_rect(p2[i], p2[j], r) for r in rects):
            j -= 1
        out.append(p2[j])
        i = j
    return out


def _reloc_avoid_list(env, avoid_objects):
    """avoid_objects (chairs/sofas/baskets) PLUS the scene's low TABLES — the
    obstacle set the robot must route its AABB-rect path around. Tables are baked
    into trav_map, but its erosion margin is thinner than the robot footprint, so
    they need an explicit inflated rect (matches run_relocation_phase's reloc_avoid)."""
    out = list(avoid_objects) if avoid_objects else []
    for obj in env.scene.objects:
        cat = (getattr(obj, "category", "") or "").lower()
        if "table" in cat:
            try:
                if float(to_numpy(obj.get_position_orientation()[0])[2]) < 1.0:
                    out.append(obj)
            except Exception:
                pass
    return out


def _rect_routed_robot_path(human_actor, start_xy, end_xy, z, rects):
    """Robot path start->end with the SAME AABB-rect avoidance Relocate uses:
    trav_map A* (walls + baked furniture) -> push each point out of the inflated
    rects -> corner-route straight legs around rects -> LOS shortcut. Returns a
    list of 3D waypoints EXCLUDING the start (the caller re-prepends the robot's
    live pose), or a straight-line fallback if no trav_map."""
    trav_map = human_actor._try_get_trav_map() if human_actor is not None else None
    start2 = np.asarray(start_xy, dtype=np.float64)[:2]
    end2 = np.asarray(end_xy, dtype=np.float64)[:2]
    if trav_map is None:
        return [np.array([float(end2[0]), float(end2[1]), z], dtype=np.float32)]
    base = human_actor._nav_path_between(start2, end2, z, trav_map, 0)
    trail = [_push_out_of_rects(np.asarray(p, dtype=np.float64)[:2], rects) for p in base]
    pts = [start2]
    for tp in trail:
        pts.extend(_route_around_rects(pts[-1], tp, rects))
    pts = _los_shortcut(pts, rects)
    body = pts[1:] if len(pts) > 1 else pts
    return [np.array([float(q[0]), float(q[1]), z], dtype=np.float32) for q in body]


def _dump_trav_map_debug(human_actor: Any, output_dir: str, floor: int = 0,
                         tag: str = "debug",
                         rects: Optional[List[Tuple[float, float, float, float]]] = None,
                         points: Optional[List[Tuple[str, Tuple[float, float], str]]] = None,
                         margin_m: float = 1.0) -> None:
    """Render the runtime trav_map (the array A* navigates on) with our extra
    avoid_objects rects + key actor positions overlaid, on world-metre axes.

    This is the layer the OmniGibson scene baked in (walls + static furniture
    incl. tables/beds — those are NOT in avoid_objects, see _skip_cats) PLUS the
    inflated AABBs we add at runtime for chairs/sofas/baskets/etc.

    Best-effort: never raises (a debug-viz failure must not kill a collection
    run), and matplotlib is imported lazily so normal runs pay nothing.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        trav_map = human_actor._try_get_trav_map()
        if trav_map is None:
            print("[trav-map] no trav_map available", flush=True)
            return

        fmap = to_numpy(trav_map.floor_map[floor]).astype(np.float32)
        free = fmap > 0  # matches A* test: int(fmap[i, j]) > 0
        H, W = fmap.shape
        res = float(trav_map.map_resolution)
        C = W / 2.0  # map is centered on the world origin (map_size / 2)

        # Crop to the traversable bbox + margin so the house fills the figure.
        rows = np.where(free.any(axis=1))[0]
        cols = np.where(free.any(axis=0))[0]
        if len(rows) and len(cols):
            mpx = int(round(margin_m / res))
            r0, r1 = max(0, int(rows.min()) - mpx), min(H - 1, int(rows.max()) + mpx)
            c0, c1 = max(0, int(cols.min()) - mpx), min(W - 1, int(cols.max()) + mpx)
        else:
            r0, r1, c0, c1 = 0, H - 1, 0, W - 1

        # row <-> world Y, col <-> world X (OmniGibson world_to_map flips axes).
        def _wx(col):
            return (col - C) * res

        def _wy(row):
            return (row - C) * res

        extent = (_wx(c0), _wx(c1 + 1), _wy(r0), _wy(r1 + 1))

        fig, ax = plt.subplots(figsize=(11, 9))
        ax.imshow(free[r0:r1 + 1, c0:c1 + 1], origin="lower", extent=extent,
                  cmap="gray", interpolation="nearest", vmin=0, vmax=1)

        for rect in (rects or []):
            xmin, ymin, xmax, ymax = rect
            ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=True, fc=(1.0, 0.5, 0.1, 0.35),
                                   ec=(1.0, 0.4, 0.0, 0.95), lw=1.2))

        for label, (px, py), color in (points or []):
            ax.plot([px], [py], marker="*", ms=16, color=color,
                    mec="black", mew=0.6, ls="none", label=label)

        ax.set_title(
            f"runtime trav_map + avoid_objects  (floor {floor}, tag={tag})\n"
            f"white=traversable  black=obstacle(wall/furniture)  "
            f"orange=avoid_objects (inflated {_ROBOT_INFLATE_M}m)")
        ax.set_xlabel("world X (m)")
        ax.set_ylabel("world Y (m)")
        ax.axhline(0, color="cyan", lw=0.5, alpha=0.6)
        ax.axvline(0, color="cyan", lw=0.5, alpha=0.6)
        ax.grid(True, color="dodgerblue", lw=0.3, alpha=0.35)
        if points:
            ax.legend(loc="upper right", fontsize=8)

        ensure_parent(os.path.join(output_dir, "placeholder.txt"))
        out_path = os.path.join(output_dir, f"trav_map_floor{floor}_{tag}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)

        print(
            f"[trav-map] dumped {out_path} "
            f"shape={fmap.shape} res={res}m/px "
            f"free={int(free.sum())} blocked={int((~free).sum())} "
            f"rects={len(rects or [])}",
            flush=True,
        )
    except Exception as exc:  # never let a debug dump break a collection run
        print(f"[trav-map] dump failed (tag={tag}): {exc}", flush=True)
        traceback.print_exc()


# --------------------------------------------------------------------------- #
# Scripted radio-toggle phase
# --------------------------------------------------------------------------- #

def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


# Human turning knobs (mirror HumanPhaseMotion defaults / Habitat
# HumanoidRearrangeController: turning_step_amount, min_angle_turn). Used by the
# relocation follow + dodge so the human turns gradually (Habitat §6) instead of
# snapping its heading.
_HUMAN_TURN_STEP_DEG = 20.0    # max yaw change per frame (= turning_step_amount)
_HUMAN_TURN_SNAP_DEG = 5.0     # within this error, snap to target (= min_angle_turn)
_HUMAN_TURN_THRESH_DEG = 10.0  # above this error -> "turn-in-place" state
# HRC_DEBUG_RELOC=1 -> per-frame leash-loop trace (turn-lock / freeze diagnosis).
_RELOC_DEBUG = os.environ.get("HRC_DEBUG_RELOC", "").strip() == "1"

def _robot_human_dot(robot: Any, human_xy: np.ndarray) -> float:
    """Port of Habitat ``robot_human_vec_dot_product`` to this project's frame
    (xy-plane + yaw about z, NOT Habitat's xz/y-up): unit robot->human vector
    dotted with the robot forward [cos(yaw), sin(yaw)]. Range [-1, 1]; > 0.5
    means the robot is facing the human."""
    pos, quat = robot.get_position_orientation()
    rxy = to_numpy(pos)[:2].astype(np.float64)
    ryaw = _yaw_from_quat(to_numpy(quat))
    v = np.asarray(human_xy, dtype=np.float64)[:2] - rxy
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return 1.0
    v /= n
    fwd = np.array([math.cos(ryaw), math.sin(ryaw)], dtype=np.float64)
    return float(np.dot(fwd, v))


def _limited_turn(cur_yaw: float, desired_yaw: float,
                  max_step_deg: float = _HUMAN_TURN_STEP_DEG,
                  snap_deg: float = _HUMAN_TURN_SNAP_DEG) -> float:
    """Rotate *cur_yaw* toward *desired_yaw* by at most max_step_deg this frame
    (Habitat-style limited turn). Snaps when within snap_deg."""
    err = _wrap_to_pi(float(desired_yaw) - float(cur_yaw))
    if abs(err) <= math.radians(snap_deg):
        return float(desired_yaw)
    return _wrap_to_pi(float(cur_yaw)
                       + math.copysign(min(abs(err), math.radians(max_step_deg)), err))


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


# --------------------------------------------------------------------------- #
# Patch 1: trav-aware human path smoothing (de-staircase + resample), with a
# clearance-preserving (~0.3m neighborhood) shortcut validation so straightening
# never steals clearance at convex corners.
# --------------------------------------------------------------------------- #

def _path_cum_dist(path) -> List[float]:
    cum = [0.0]
    for i in range(1, len(path)):
        cum.append(cum[-1] + float(np.linalg.norm(
            np.asarray(path[i], dtype=np.float64)[:2]
            - np.asarray(path[i - 1], dtype=np.float64)[:2])))
    return cum


def _resample_path(path, spacing: float = 0.08) -> List[np.ndarray]:
    """Densify a polyline to ~``spacing`` m spacing (uniform walk speed). NOT
    geometric smoothing — just resampling; point count may INCREASE."""
    if len(path) < 2:
        return [np.asarray(p, dtype=np.float32) for p in path]
    z = float(np.asarray(path[0], dtype=np.float64)[2]) if np.asarray(path[0]).shape[0] > 2 else 0.0
    cum = _path_cum_dist(path)
    total = cum[-1]
    if total < 1e-6:
        return [np.asarray(path[0], dtype=np.float32)]
    n = max(int(math.ceil(total / max(spacing, 1e-3))), 1)
    out: List[np.ndarray] = []
    for k in range(n + 1):
        d = min(total * k / n, total)
        j = 1
        while j < len(cum) and cum[j] < d:
            j += 1
        j = min(j, len(path) - 1)
        seg = max(cum[j] - cum[j - 1], 1e-9)
        t = (d - cum[j - 1]) / seg
        a = np.asarray(path[j - 1], dtype=np.float64)[:2]
        b = np.asarray(path[j], dtype=np.float64)[:2]
        xy = a + (b - a) * t
        out.append(np.array([xy[0], xy[1], z], dtype=np.float32))
    return out


def _trav_xy_ok(xy, trav_map, floor: int = 0, clearance: float = 0.3) -> bool:
    """True iff the world (x,y) pixel AND its ~``clearance`` m neighborhood are
    all free on trav_map (equivalent erosion). A bare 'is this cell free' check
    would let a straightened diagonal steal clearance at convex corners."""
    try:
        res = float(getattr(trav_map, "map_resolution", 0.1))
        ij = trav_map.world_to_map(
            th.tensor([float(xy[0]), float(xy[1])], dtype=th.float32)).tolist()
        fmap = trav_map.floor_map[floor]
        H, W = int(fmap.shape[0]), int(fmap.shape[1])
        i0, j0 = int(ij[0]), int(ij[1])
        r = int(math.ceil(clearance / max(res, 1e-3)))
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                i, j = i0 + di, j0 + dj
                if not (0 <= i < H and 0 <= j < W):
                    return False  # near map edge -> conservative block
                if int(fmap[i, j]) <= 0:
                    return False
        return True
    except Exception:
        return False


def _segment_trav_ok(a, b, trav_map, floor: int = 0, step: float = 0.05,
                     clearance: float = 0.3) -> bool:
    """Sample a->b every ``step`` m; every sample must pass _trav_xy_ok."""
    a = np.asarray(a, dtype=np.float64)[:2]
    b = np.asarray(b, dtype=np.float64)[:2]
    d = float(np.linalg.norm(b - a))
    n = max(int(math.ceil(d / max(step, 1e-3))), 1)
    for k in range(n + 1):
        xy = a + (b - a) * (k / n)
        if not _trav_xy_ok(xy, trav_map, floor, clearance):
            return False
    return True


def _shortcut_path_on_trav_map(path, trav_map, floor: int = 0,
                               clearance: float = 0.3) -> List[np.ndarray]:
    """Greedy line-of-sight straightening WITH trav_map clearance validation:
    keep a point, jump to the farthest later point whose connecting segment stays
    clearance-free. Removes the grid staircase without stealing clearance; a
    segment that can't be straightened keeps the original A* grid points."""
    if len(path) <= 2:
        return [np.asarray(p, dtype=np.float32) for p in path]
    p2 = [np.asarray(p, dtype=np.float64) for p in path]
    out = [p2[0]]
    i, n = 0, len(p2)
    while i < n - 1:
        j = n - 1
        while j > i + 1 and not _segment_trav_ok(p2[i], p2[j], trav_map, floor, clearance=clearance):
            j -= 1
        out.append(p2[j])
        i = j
    return [np.asarray(p, dtype=np.float32) for p in out]


def _prepare_human_walk_path(raw_path, trav_map, floor: int = 0,
                             clearance: float = 0.3) -> List[np.ndarray]:
    """Patch 1: trav-aware shortcut (de-staircase, clearance-preserving) then
    0.08m resample (uniform walk speed). Logs raw -> shortcut -> motion counts.
    `clearance` = the shortcut's wall/furniture band; pass the LEAD erosion so the
    de-staircase can't straighten the route back through a gap too tight for the
    trailing robot (A-fix)."""
    raw = [np.asarray(p, dtype=np.float32) for p in raw_path]
    if trav_map is None or len(raw) < 3:
        return raw
    sc = _shortcut_path_on_trav_map(raw, trav_map, floor=floor, clearance=clearance)
    motion = _resample_path(sc, spacing=0.08)
    print(f"[relocation] human path prepared: raw {len(raw)} -> shortcut {len(sc)} "
          f"-> motion {len(motion)} pts", flush=True)
    return motion


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
    path = human_actor._nav_path_between(src_xy, dst_xy_f, z=float(z), trav_map=trav_map, floor=0)

    if len(path) < 2:
        print("[robot-plan] A* failed from source (likely non-navigable); searching nearby start", flush=True)
        best_path = path
        for offset_angle in np.linspace(0, 2 * math.pi, 16, endpoint=False):
            for offset_r in [0.5, 1.0, 1.5]:
                shifted = src_xy + np.array([math.cos(offset_angle) * offset_r,
                                             math.sin(offset_angle) * offset_r], dtype=np.float32)
                trial = human_actor._nav_path_between(shifted, dst_xy_f, z=float(z), trav_map=trav_map, floor=0)
                if len(trial) >= 2:
                    prepend = np.array([src_xy[0], src_xy[1], z], dtype=np.float32)
                    best_path = [prepend] + list(trial)
                    print(f"[robot-plan] found navigable start at offset r={offset_r:.1f} "
                          f"angle={math.degrees(offset_angle):.0f}°", flush=True)
                    break
            if len(best_path) >= 2:
                break
        path = best_path

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
                           track_human_yaw: bool = False,
                           smooth_drive: bool = False) -> None:
    """Walk the robot along ``planned_path`` using cumulative-distance
    interpolation so motion is smooth regardless of waypoint count.

    *smooth_drive* (GoToTask, RC3): use the continuous turn-while-move controller
    (``drive_path_smooth``) so the robot walks toward the task AS it turns (no big
    in-place stop-pivot). The single shortest in-place turn to the replay yaw is done
    by the caller's run_face_yaw_phase afterwards.
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

    # Zero-distance nav (e.g. round-0 GoToTask when the robot already spawned at
    # the task pose): hold the CURRENT pose with a real arm-HOLD action instead of
    # recording action=0 (which would teach the BC policy "command arms to zero").
    short_hold = min(max(total_steps, 1), 15)
    if len(nodes) < 2:
        run_zero_phase_pinned(env, recorder, phase_label, short_hold,
                              round_id, human_actor=human_actor)
        print(f"[online-nav] {phase_label}: zero-distance path; held current pose", flush=True)
        return

    cum_dist = [0.0]
    for i in range(1, len(nodes)):
        cum_dist.append(cum_dist[-1] + float(np.linalg.norm(nodes[i][:2] - nodes[i - 1][:2])))
    total_len = cum_dist[-1]
    if total_len < 0.05:
        # Sub-5cm residual: don't hand a centimetre-scale goal to OnlineNav (it
        # crawls ~200 dead frames). Hold the pose, same as the len<2 case.
        run_zero_phase_pinned(env, recorder, phase_label, short_hold,
                              round_id, human_actor=human_actor)
        print(f"[online-nav] {phase_label}: <5cm path; held current pose", flush=True)
        return

    # ---- Physics-online navigation (--online-nav): drive the base with TRUE
    # velocity actions via OnlineNavDriver and record the REAL action each frame
    # (no more action==0 nav frames). Furniture is pinned every frame so the
    # physics-driven robot can't displace it (keeps the world continuous with the
    # op-replay segments). The frame count is determined by physics, not
    # @total_steps.
    if _online_nav_driver is not None:
        if phase_label != "GoToTask":
            # Custom phases travel in the canonical 0:02 home posture.
            # GoToTask keeps its current arms and settles toward the task's
            # first frame at arrival (existing logic, unchanged).
            _settle_arms_to_home(env, recorder, phase_label, round_id,
                                 human_actor=human_actor)
        waypoints = [np.asarray(n, dtype=np.float64)[:2] for n in nodes[1:]]
        final_yaw = None
        if track_human_yaw and human_actor is not None:
            live_human = to_numpy(human.get_position_orientation()[0])
            end = nodes[-1]
            final_yaw = math.atan2(float(live_human[1] - end[1]),
                                   float(live_human[0] - end[0]))

        def _on_frame(obs: Any, action: np.ndarray) -> None:
            _restore_pinned_objects(env)
            if human_actor is not None:
                human_actor.step(phase_label, robot)
            info = {"reward": 0.0, "terminated": False, "truncated": False}
            recorder.record(phase_label, obs, info, round_id=round_id, action=action)

        # If the robot is basically already at the goal (e.g. ReturnToHuman when it
        # finished the task right next to the human), skip the drive primitive —
        # its rotate winds the LONG way (ep000: spun 315° in place to face a human
        # only 34° to its right) — and just face the target via shortest-direction.
        if total_len < 0.5 and final_yaw is not None:
            _online_nav_driver.rotate_to_yaw_shortest(final_yaw, on_frame=_on_frame)
            return
        # RC3: GoToTask drives turn-while-move (no big stop-pivot); the shortest in-place
        # turn to the replay yaw is a SEPARATE run_face_yaw_phase after this returns.
        if smooth_drive:
            _online_nav_driver.drive_path_smooth(waypoints, on_frame=_on_frame)
            return
        # CLEARANCE-ABORT for the primitive walk (ReturnToHuman): stop the base BEFORE it
        # drives into furniture (ep0014 drove a chair -> 33°). Firing on tilt is too late —
        # a wedged 12° base cannot pull back out — so abort while there is still clearance.
        _wtrav = human_actor._try_get_trav_map() if human_actor is not None else None

        def _drive_abort() -> bool:
            if _wtrav is None:
                return False
            _rp = to_numpy(robot.get_position_orientation()[0]).astype(np.float64)[:2]
            return not _trav_xy_ok(_rp, _wtrav, floor=0, clearance=0.28)
        ok = _online_nav_driver.drive_along_path(
            waypoints, final_yaw=final_yaw, on_frame=_on_frame, abort_fn=_drive_abort)
        if not ok:
            print(f"[online-nav] {phase_label}: primitive ended early "
                  f"(robot left at current pose)", flush=True)
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
    if _policy_client is not None:
        run_policy_phase(env, recorder, "GoToTask", steps, round_id,
                         human_actor=human_actor)
        robot_xy = to_numpy(env.robots[0].get_position_orientation()[0])[:2]
        distance = float(np.linalg.norm(robot_xy - np.asarray(target_pos)[:2]))
        _record_policy_arrival(round_id, "GoToTask", distance)
        return
    robot = env.robots[0]
    robot_pos = to_numpy(robot.get_position_orientation()[0]).astype(np.float32)
    z = float(robot_pos[2])
    # Unified AABB-rect avoidance (same model as ReturnToHuman / Relocate). Task
    # targets are already excluded from avoid_objects via avoid_exclude_categories,
    # so the robot approaches them instead of detouring around them.
    rects = _build_scene_rects(_reloc_avoid_list(env, avoid_objects), tag="gototask-map")
    planned = _rect_routed_robot_path(human_actor, robot_pos[:2],
                                      np.asarray(target_pos, dtype=np.float32)[:2], z, rects)
    if not planned:
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
                           turn_fraction=turn_fraction, track_human_yaw=False,
                           smooth_drive=True)


def _find_clear_standoff(human_xy: np.ndarray, preferred_dir: np.ndarray,
                         stand_off: float, avoid_objects: Optional[List[Any]],
                         min_clearance: float = 0.55,
                         avoid_rects: Optional[List[Tuple[float, float, float, float]]] = None,
                         trav_map: Any = None, trav_clear: float = 0.30) -> np.ndarray:
    """Find a stand-off position around the human that doesn't overlap any
    avoid_object.  Tries the preferred direction first, then rotates in 30°
    increments until a clear spot is found.

    ``avoid_rects`` (inflated AABBs) are checked IN ADDITION to the center-
    distance test: center distance alone misjudged a point INSIDE an extended
    footprint as clear (v4: a standoff 0.64m from the coffee-table CENTRE >
    min_clearance, yet inside its inflated rect -> drive_to_pose EXECUTION_ERROR).

    RC2 (2026-07-17): the object/rect tests do NOT cover WALLS/STAIRS, so the standoff
    could land 0.20m from a wall — the robot then rotated in place there and tipped
    (ep0031 21°, ep0064 12°), and drove into the pocket over-approaching the human
    (ep0035 0.40m). Now every candidate is also checked against the trav_map
    (walls + baked furniture) via ``_trav_xy_ok``. Candidates are tried at decreasing
    clearance and, if the human is boxed in, a shorter stand-off — the OLD fallback of
    returning a raw unchecked point is gone (that is exactly what put the robot in the
    wall pocket)."""
    base_angle = math.atan2(float(preferred_dir[1]), float(preferred_dir[0]))
    angles = [0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180]

    def _obj_rect_ok(cand: np.ndarray) -> bool:
        if avoid_objects:
            for obj in avoid_objects:
                try:
                    obj_pos = to_numpy(obj.get_position_orientation()[0]).astype(np.float32)[:2]
                    if float(np.linalg.norm(cand - obj_pos)) < min_clearance:
                        return False
                except Exception:
                    pass
        if avoid_rects and any(_point_in_rect(cand, r) for r in avoid_rects):
            return False
        return True

    # Progressive tiers: full trav clearance -> relaxed -> shorter stand-off. Return the
    # FIRST candidate that clears objects/rects AND the trav_map at that tier.
    for tc, so_scale in ((float(trav_clear), 1.0), (0.25, 1.0), (0.25, 0.8), (0.20, 0.8)):
        so = float(stand_off) * so_scale
        for delta_deg in angles:
            angle = base_angle + math.radians(delta_deg)
            cand = human_xy + np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * so
            if not _obj_rect_ok(cand):
                continue
            if trav_map is not None and not _trav_xy_ok(cand, trav_map, floor=0, clearance=tc):
                continue
            if delta_deg != 0 or so_scale != 1.0 or tc != float(trav_clear):
                print(f"[standoff] shifted {delta_deg}° clr={tc:.2f} so={so:.2f} "
                      f"-> ({cand[0]:.2f},{cand[1]:.2f})", flush=True)
            return cand
    # Human fully boxed in (rare): give the least-bad point that is at least not IN a
    # wall/furniture cell, preferring the requested direction.
    for delta_deg in angles:
        angle = base_angle + math.radians(delta_deg)
        cand = human_xy + np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * float(stand_off)
        if trav_map is None or _trav_xy_ok(cand, trav_map, floor=0, clearance=0.10):
            print(f"[standoff] WARN boxed-in human; best-effort {delta_deg}° "
                  f"-> ({cand[0]:.2f},{cand[1]:.2f})", flush=True)
            return cand
    return human_xy + preferred_dir * float(stand_off)


def run_return_phase_smooth(env: Any, recorder: DemoRecorder, human: Any, steps: int,
                            round_id: int, stand_off: float,
                            human_actor: Any = None, turn_fraction: float = 0.5,
                            avoid_objects: Optional[List[Any]] = None) -> None:
    """A*-planned return-to-human with obstacle avoidance."""
    if _policy_client is not None:
        run_policy_phase(env, recorder, "ReturnToHuman", steps, round_id,
                         human_actor=human_actor)
        robot_xy = to_numpy(env.robots[0].get_position_orientation()[0])[:2]
        human_xy = (
            np.asarray(human_actor.base_pos)[:2]
            if human_actor is not None
            else to_numpy(human.get_position_orientation()[0])[:2]
        )
        _record_policy_arrival(
            round_id, "ReturnToHuman", float(np.linalg.norm(robot_xy - human_xy))
        )
        return
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

    # AABB-rect avoidance — same model run_relocation_phase uses. The old
    # _plan_robot_path deflected around object CENTERS + a radius, which is wrong
    # for extended furniture (a circle can't capture a chair/table footprint), so
    # the robot grazed a chair during ReturnToHuman. Route around inflated AABB
    # rects (chairs/sofas/baskets + low tables) with straight legs + corner turns.
    # Built BEFORE the standoff so the standoff can reject rect-interior points
    # (the human birth stays near the fridge, so this is the prime crash site).
    return_rects = _build_scene_rects(_reloc_avoid_list(env, avoid_objects),
                                      tag="return-map")
    _return_trav = human_actor._try_get_trav_map() if human_actor is not None else None
    # RC2: keep the return PATH clear of the (static) human — a keep-out box so the robot
    # routes AROUND the human instead of grazing 0.53m past it while looping to a far-side
    # standoff (ep0031/0035/0052 hit_human were the return path, not the reach-gate). The
    # standoff itself is at stand_off (~1.5m) > the box, so it is not rejected.
    _hk = REACH_MIN_HUMAN_GAP  # 0.65m keep-out half-width (> QC hit-human 0.55m)
    human_rect = (float(human_target_xy[0] - _hk), float(human_target_xy[1] - _hk),
                  float(human_target_xy[0] + _hk), float(human_target_xy[1] + _hk))
    path_rects = list(return_rects) + [human_rect]
    target_xy = _find_clear_standoff(human_target_xy, direction, stand_off,
                                     avoid_objects, min_clearance=_ROBOT_STANDOFF_CLEAR,
                                     avoid_rects=return_rects, trav_map=_return_trav)
    planned = _rect_routed_robot_path(human_actor, robot_pos[:2], target_xy, z,
                                      path_rects)
    if not planned:
        planned = _plan_robot_path(human_actor, robot_pos, target_xy, z=z,
                                   avoid_objects=avoid_objects)
    _walk_robot_along_path(env, recorder, human, steps, round_id, "ReturnToHuman",
                           planned_path=planned, human_actor=human_actor,
                           turn_fraction=turn_fraction, track_human_yaw=True)
    # Habitat _reach_human post-condition: the fixed-step walk to a geometric
    # standoff doesn't prove arrival (standoff blocked / path stopped short). Drive
    # reactively until actually back at the human (in-band distance + facing), so
    # "return to human" is judged by STATE, not by a step budget.
    _drive_until_reached_facing(env, recorder, human_actor, round_id,
                                human_target_xy, target_xy, phase="ReturnToHuman",
                                max_frames=60)


def run_face_human_phase_smooth(env: Any, recorder: DemoRecorder, human: Any, steps: int,
                                round_id: int, human_actor: Any = None) -> None:
    if _policy_client is not None:
        run_policy_phase(env, recorder, "FaceHuman", steps, round_id,
                         human_actor=human_actor)
        return
    robot = env.robots[0]
    robot_pos, robot_quat = robot.get_position_orientation()
    robot_pos_np = to_numpy(robot_pos)
    start_yaw = _yaw_from_quat(to_numpy(robot_quat))
    human_pos_np = to_numpy(human.get_position_orientation()[0])
    target_yaw = math.atan2(float(human_pos_np[1] - robot_pos_np[1]),
                            float(human_pos_np[0] - robot_pos_np[0]))

    # --online-nav: physically rotate in place to face the human with TRUE base
    # velocity actions (BEHAVIOR-1K _rotate_in_place), recording the real action
    # each frame instead of a kinematic set_position_orientation + action==0.
    if _online_nav_driver is not None:
        _settle_arms_to_home(env, recorder, "FaceHuman", round_id,
                             human_actor=human_actor)

        def _on_frame(obs: Any, action: np.ndarray) -> None:
            _restore_pinned_objects(env)
            if human_actor is not None:
                human_actor.step("FaceHuman", robot)
            info = {"reward": 0.0, "terminated": False, "truncated": False}
            recorder.record("FaceHuman", obs, info, round_id=round_id, action=action)

        # STUCK GUARD: rotating in place to face the human while the base is wedged/leaning
        # is what tips it (ep0031). Un-wedge to clear space FIRST (vyaw≡0) until upright AND
        # clear; only then rotate. Give up escaping after a few frames if truly boxed in (then
        # a hold has at least kept the lean small, below the fall threshold).
        _face_trav = human_actor._try_get_trav_map() if human_actor is not None else None
        for _ in range(20):
            _rxy = to_numpy(robot.get_position_orientation()[0]).astype(np.float64)[:2]
            _lean = _quat_tilt_deg(robot.get_position_orientation()[1])
            _wedge = _face_trav is None or not _trav_xy_ok(_rxy, _face_trav, floor=0, clearance=0.30)
            if _lean <= 8.0 and not _wedge:
                break
            _e = _escape_from_obstacle(_rxy, _face_trav) if _face_trav is not None else None
            _obs, _a = _online_nav_driver.escape_step(nudge_xy=_e)
            _on_frame(_obs, _a)
        # Skip the face-rotation entirely if still leaning (boxed pocket) — never tip to face.
        if _quat_tilt_deg(robot.get_position_orientation()[1]) <= 8.0:
            ok = _online_nav_driver.rotate_to_yaw_shortest(target_yaw, on_frame=_on_frame)
        else:
            ok = True
            print("[online-nav] FaceHuman: base still leaning in a tight spot; "
                  "skipped face-rotation to avoid a tip", flush=True)
        if not ok:
            print("[online-nav] FaceHuman: rotate primitive ended early", flush=True)
        # Habitat _reach_human facing guarantee: FaceHuman must ALWAYS end looking
        # at the human (the primitive can stop early). If still not facing,
        # recompute the heading from the live human pose and rotate once more.
        live_human_xy = to_numpy(human.get_position_orientation()[0])[:2]
        if _robot_human_dot(robot, live_human_xy) <= HABITAT_FACE_DOT_THRESH:
            rxy = to_numpy(robot.get_position_orientation()[0])[:2]
            re_yaw = math.atan2(float(live_human_xy[1] - rxy[1]),
                                float(live_human_xy[0] - rxy[0]))
            _online_nav_driver.rotate_to_yaw_shortest(re_yaw, on_frame=_on_frame)
            print(f"[online-nav] FaceHuman: re-aligned to face human "
                  f"(dot {_robot_human_dot(robot, live_human_xy):.2f})", flush=True)
        return

    zero_action = np.zeros(robot.action_dim, dtype=np.float32)
    n = max(steps, 1)
    for i in range(n):
        _restore_pinned_objects(env)
        alpha = _smoothstep((i + 1) / n)
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


def run_face_yaw_phase(env: Any, recorder: DemoRecorder,
                       pin_pos: np.ndarray, target_yaw: float,
                       steps: int, round_id: int,
                       human_actor: Any = None) -> None:
    """Smoothly rotate the robot from its current yaw to *target_yaw* while
    keeping it pinned at *pin_pos*. Used to align with the replay orientation."""
    if _policy_client is not None:
        run_policy_phase(env, recorder, "GoToTask", steps, round_id,
                         human_actor=human_actor)
        return
    robot = env.robots[0]

    # --online-nav: physically rotate in place to the replay's start yaw with TRUE
    # base velocity actions. Use rotate_to_yaw_shortest, NOT the BEHAVIOR-1K
    # _rotate_in_place primitive: the latter winds the LONG way, and this GoToTask
    # yaw-align is the SECOND RC3 spin source (the back-loaded ~100° of the round-1
    # GoToTask spin — DIAGNOSIS §6 P1). Shortest-direction can never over-wind.
    # Labeled "GoToTask".
    if _online_nav_driver is not None:
        def _on_frame(obs: Any, action: np.ndarray) -> None:
            _restore_pinned_objects(env)
            if human_actor is not None:
                human_actor.step("GoToTask", robot)
            info = {"reward": 0.0, "terminated": False, "truncated": False}
            recorder.record("GoToTask", obs, info, round_id=round_id, action=action)
        ok = _online_nav_driver.rotate_to_yaw_shortest(float(target_yaw), on_frame=_on_frame)
        if not ok:
            print("[online-nav] GoToTask yaw-align: rotate primitive ended early", flush=True)
        return

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


def run_settle_to_task_config(env: Any, recorder: DemoRecorder, q_target: Optional[np.ndarray],
                              base_pos: Optional[np.ndarray] = None,
                              base_quat: Optional[np.ndarray] = None,
                              round_id: int = 0, human_actor: Any = None,
                              phase_label: str = "GoToTask") -> None:
    """--online-nav: settle the robot onto the op-replay segment's first frame
    before the replay starts -> no arm snap AND no base slide at the GoToTask ->
    ExecuteBehaviorTask handoff (the visible "瞬变"). Arms+trunk ramp to *q_target*
    (recorded first-frame joints); if *base_pos*/*base_quat* are given the base
    glides to that exact recorded pose; then a ~0.5s hold ("arrived, pausing")
    before the task. Records as GoToTask.

    No-op when the online driver is off (offline mode uses load_state replay that
    overwrites the pose anyway) or when q_target is unavailable."""
    if _policy_client is not None:
        return
    if _online_nav_driver is None or q_target is None:
        return
    robot = env.robots[0]

    def _on_frame(obs: Any, action: np.ndarray) -> None:
        _restore_pinned_objects(env)
        if human_actor is not None:
            human_actor.step(phase_label, robot)
        info = {"reward": 0.0, "terminated": False, "truncated": False}
        recorder.record(phase_label, obs, info, round_id=round_id, action=action)

    base_target = ((base_pos, base_quat)
                   if (base_pos is not None and base_quat is not None) else None)
    _online_nav_driver.settle_to_config(q_target, base_target=base_target, on_frame=_on_frame)


# --------------------------------------------------------------------------- #
# Between-round relocation: human walks to a new position, robot follows
# --------------------------------------------------------------------------- #

def _human_turn_to_face_robot(env: Any, recorder: DemoRecorder, human_actor: Any,
                              round_id: int, max_frames: int = 20) -> None:
    """At the end of relocation, smoothly turn the (stationary) human in place to
    face the robot, so the next round's WaitNearHuman doesn't snap its heading.
    The robot is held still; records Relocate frames."""
    if human_actor is None:
        return
    robot = env.robots[0]
    pin_pos = to_numpy(robot.get_position_orientation()[0]).copy()
    pin_quat = to_numpy(robot.get_position_orientation()[1]).copy()
    hold_action = _build_hold_action(robot)
    hold_action_np = to_numpy(hold_action).astype(np.float32)
    for _ in range(max(max_frames, 0)):
        robot_xy = to_numpy(robot.get_position_orientation()[0]).astype(np.float32)
        face_yaw = math.atan2(float(robot_xy[1] - human_actor.base_pos[1]),
                              float(robot_xy[0] - human_actor.base_pos[0]))
        if abs(_wrap_to_pi(face_yaw - float(human_actor.base_yaw))) <= math.radians(_HUMAN_TURN_SNAP_DEG):
            break
        human_actor.base_yaw = _limited_turn(float(human_actor.base_yaw), face_yaw)
        human_actor.linear_speed = 0.0
        human_actor.motion_state = "turn"
        human_actor._last_displacement_m = 0.0
        human_actor._advance_animation()
        human_actor.apply_pose()
        robot.set_position_orientation(pin_pos, pin_quat)
        robot.keep_still()
        _restore_pinned_objects(env)
        obs, _rew, _term, _trunc, _info = env.step(hold_action, n_render_iterations=1)
        info = {"reward": 0.0, "terminated": False, "truncated": False}
        recorder.record("Relocate", obs, info, round_id=round_id, action=hold_action_np)


def _find_staircase_xy(env: Any) -> Optional[np.ndarray]:
    """Return (x, y) of the first staircase in the scene, or None."""
    for obj in env.scene.objects:
        cat = (getattr(obj, "category", "") or "").lower()
        if "stair" in cat:
            pos = to_numpy(obj.get_position_orientation()[0])
            print(f"[relocation] found staircase {obj.name} at ({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})", flush=True)
            return np.array([float(pos[0]), float(pos[1])], dtype=np.float32)
    return None


def _normalize_relocation_dest_candidates(dest_override) -> List[np.ndarray]:
    """Normalize a config ``relocation_dest`` into a list of [x,y] candidates.
    Accepts a single point [x,y], a list of points [[x,y],...], None, or an empty
    list (-> []). Guards against the bare ``dest_override[0]`` IndexError."""
    if not isinstance(dest_override, (list, tuple)) or len(dest_override) == 0:
        return []
    first = dest_override[0]
    cands = list(dest_override) if isinstance(first, (list, tuple)) else [dest_override]
    return [np.asarray(c, dtype=np.float32)[:2] for c in cands]


def _leg_robot_clear(a: np.ndarray, b: np.ndarray, z: float, trav_map: Any,
                     rects: List[Tuple[float, float, float, float]], human_actor: Any,
                     clearance: float, erosion: float, skip: float = 0.6) -> bool:
    """True iff a trav_map A* leg a->b exists AND its INTERIOR clears the ROBOT
    footprint (0.55m), so the leash-following robot can trail it without wedging.
    Per-point re-verification (NOT the erosion arg): _nav_path_between silently
    falls back to 0-erosion when the robot-eroded map is impassable, which would
    otherwise pass a human-only gap. Points within `skip` of either endpoint are
    ignored — there the robot trails at a stand-off, off the human's exact path,
    so a furniture-adjacent start/end (e.g. the human's fridge-side birth) must
    not false-reject the whole leg."""
    try:
        path = [np.asarray(p, dtype=np.float32)[:2] for p in
                human_actor._nav_path_between(np.asarray(a, dtype=np.float32),
                                              np.asarray(b, dtype=np.float32), z,
                                              trav_map, erosion)]
    except Exception:
        return False
    if len(path) < 2:
        return False
    cd = [0.0]
    for i in range(1, len(path)):
        cd.append(cd[-1] + float(np.linalg.norm(path[i] - path[i - 1])))
    total = cd[-1]
    for q, c in zip(path, cd):
        if c < skip or (total - c) < skip:
            continue                                   # near an endpoint: robot trails off-path
        if not _trav_xy_ok(q, trav_map, floor=0, clearance=clearance):
            return False
        if any(_point_in_rect(q, r) for r in rects):
            return False
    return True


def _sample_walk_waypoints(human_actor: Any, trav_map: Any, start: np.ndarray,
                           dest: np.ndarray, z: float,
                           dest_rects: List[Tuple[float, float, float, float]],
                           n_min: int = 1, n_max: int = 3,
                           min_leg: float = 0.8, max_leg: float = 3.0,
                           clearance: float = _HUMAN_EROSION_M,
                           erosion: float = _HUMAN_EROSION_M,
                           robot_clearance: float = _ROBOT_INFLATE_M,
                           robot_erosion: float = _ROBOT_INFLATE_M,
                           attempts: int = 30) -> List[np.ndarray]:
    """Sample 1..n_max validated random navigable WAYPOINTS for the human to
    wander through before the final leg to ``dest`` (Habitat
    OracleNavRandCoordAction-style random navigable points). Candidates are drawn
    in a [min_leg, max_leg] annulus around the PREVIOUS point — like
    ``_sample_dest``'s proven task-annulus branch — so the meander stays local and
    still finishes at ``dest`` (whole-floor get_random_point would scatter across
    the house and the leg cap would reject almost everything). Each candidate
    reuses the ``_sample_dest`` validation chain: snap to navigable
    (_snap_to_navigable) -> reject if inside a furniture rect (_point_in_rect) or
    under trav clearance (_trav_xy_ok) -> reachable from the previous point with a
    sane A* leg length -> still able to reach ``dest`` (no dead-end). Returns the
    validated ordered list of 3-vecs (possibly fewer than requested, or empty ->
    caller routes straight to ``dest``). Task-agnostic (trav_map only)."""
    k = int(np.random.randint(int(n_min), int(n_max) + 1))
    dest3 = np.array([float(dest[0]), float(dest[1]), float(z)], dtype=np.float32)
    prev = np.array([float(start[0]), float(start[1]), float(z)], dtype=np.float32)
    pts: List[np.ndarray] = []
    for _ in range(max(k, 0)):
        chosen = None
        for _a in range(max(attempts, 1)):
            try:
                ang = float(np.random.uniform(0.0, 2.0 * math.pi))
                rr = float(np.random.uniform(min_leg, max_leg))
                cand_xy = prev[:2] + rr * np.array([math.cos(ang), math.sin(ang)],
                                                   dtype=np.float32)
                snapped = human_actor._snap_to_navigable(cand_xy, trav_map, 0)
                if snapped is None:
                    continue
                cand2 = np.array([float(snapped[0]), float(snapped[1])], dtype=np.float32)
                if any(_point_in_rect(cand2, r) for r in dest_rects):
                    continue
                if not _trav_xy_ok(cand2, trav_map, floor=0, clearance=clearance):
                    continue
                # Robot-reachability FIRST (user pick): the human erodes 0.35m but the
                # leash robot is 0.55m, so a human-only waypoint/leg wedges the robot
                # (HANDOFF v5 mismatch). Require the waypoint AND both legs to clear the
                # ROBOT footprint; in tight scenes this rejects most points and the human
                # just routes straight to dest (accepted degrade), but it never leads the
                # robot somewhere it can't follow.
                if not _trav_xy_ok(cand2, trav_map, floor=0, clearance=robot_clearance):
                    continue
                cand3 = np.array([cand2[0], cand2[1], float(z)], dtype=np.float32)
                # erosion is the LEAD body-width, NOT the floor: pass it by keyword.
                # (Positional put it in `floor` -> floor_map[float] TypeError inside
                # _snap_to_navigable -> every wander waypoint silently failed, so this
                # random-detour diversity never actually worked until now.)
                leg = human_actor._nav_path_between(prev, cand3, z, trav_map,
                                                    erosion_radius=erosion)
                if len(leg) < 2:
                    continue
                plen = sum(float(np.linalg.norm(np.asarray(leg[i + 1])[:2] - np.asarray(leg[i])[:2]))
                           for i in range(len(leg) - 1))
                if not (min_leg <= plen <= max_leg * 1.5):   # allow A* detour slack
                    continue
                # Robot can trail prev->cand AND cand->dest (per-point 0.55m verified,
                # immune to _nav_path_between's 0-erosion fallback). Robot-clear =>
                # human-clear, so this subsumes the human dead-end-to-dest check.
                if not _leg_robot_clear(prev, cand3, z, trav_map, dest_rects,
                                        human_actor, robot_clearance, robot_erosion):
                    continue
                if not _leg_robot_clear(cand3, dest3, z, trav_map, dest_rects,
                                        human_actor, robot_clearance, robot_erosion):
                    continue
                chosen = cand3
                break
            except Exception:
                continue
        if chosen is None:
            break   # couldn't validate one -> walk fewer points (safe degrade)
        pts.append(chosen)
        prev = chosen
    return pts


def _pick_yield_target(robot_xy: np.ndarray, hr: np.ndarray, hdir: np.ndarray,
                       trav_map: Any, scene_rects: List[Tuple[float, float, float, float]],
                       human_xy: Optional[np.ndarray] = None,
                       min_gap: float = 0.0,
                       steps: Tuple[float, ...] = (0.30, 0.50, 0.75)) -> Optional[np.ndarray]:
    """Pick where the robot should step to get OUT of the walking human's way.

    Ordered by preference:
      1. PERPENDICULAR to the human's heading (clear the lane — the human then walks
         past and the gap opens on its own).
      2. Straight away from the human (`hr`). Backing up along the approach line just
         gets the robot pushed along ahead of the human.
      3. FORWARD along the human's heading (`hdir`): in a corridor both perpendiculars
         are walls and "away" is behind the human, so the only relief is to scoot ahead
         down the lane. The robot leads until the leash pulls it back.
      4. Diagonals off the away-vector.

    A candidate must keep the robot footprint off walls/furniture (`_trav_xy_ok`) and
    outside the scene rects, AND — critically — it must actually RELIEVE the freeze:
    the human halts whenever it is within `hard_collision_r` of the robot, so a step
    that leaves the gap unchanged (a perpendicular slide keeps `d` roughly constant)
    buys nothing and the human stays frozen forever. Require the candidate to reach at
    least `min_gap` from the human and to strictly increase the current gap.

    Clearance is relaxed 0.40 -> 0.30 -> 0.25 across tiers (0.25 is already precedented
    by `_clamp_target_to_wall_clear`) before giving up: a robot wedged in a corridor
    with nowhere "clean" to go must still move, because standing still deadlocks the
    human against it (ep0002/0003/0004/0007: `d` pinned at 0.59m for the whole follow).
    Returns None only when every direction at every tier is blocked."""
    perp = np.array([-hdir[1], hdir[0]], dtype=np.float64)
    cands: List[np.ndarray] = [perp, -perp, hr, hdir]
    for deg in (45.0, -45.0, 90.0, -90.0):
        a = math.radians(deg)
        ca, sa = math.cos(a), math.sin(a)
        cands.append(np.array([hr[0] * ca - hr[1] * sa, hr[0] * sa + hr[1] * ca],
                              dtype=np.float64))
    cur_gap = (float(np.linalg.norm(robot_xy - human_xy))
               if human_xy is not None else float("inf"))
    for clearance in (0.40, 0.30, 0.25):
        for step in steps:
            for dv in cands:
                cand = robot_xy + dv * float(step)
                if not _trav_xy_ok(cand, trav_map, floor=0, clearance=clearance):
                    continue
                if any(_point_in_rect(cand, r) for r in scene_rects):
                    continue
                if human_xy is not None:
                    cand_gap = float(np.linalg.norm(cand - human_xy))
                    if cand_gap < min_gap or cand_gap <= cur_gap + 0.02:
                        continue        # does not relieve the human's freeze
                return cand
    return None


def _pick_yield_escape(robot_xy: np.ndarray, human_xy: np.ndarray, trav_map: Any,
                       scene_rects: List[Tuple[float, float, float, float]],
                       min_gap: float) -> Optional[np.ndarray]:
    """Last-resort yield: sweep a ring around the robot and take the reachable point
    that puts the most distance between it and the human.

    `_pick_yield_target` only probes 8 directions at 3 step sizes; in a tight corridor
    all of them can fail. Standing still is NOT an acceptable fallback — the human's
    collision freeze then never releases and the whole follow segment is lost. Widen
    the search instead: 24 headings x radii out to 1.2m, clearance >= 0.25."""
    best, best_gap = None, float(np.linalg.norm(robot_xy - human_xy))
    for radius in (0.30, 0.45, 0.60, 0.80, 1.00, 1.20):
        for k in range(24):
            a = 2.0 * math.pi * k / 24.0
            cand = robot_xy + np.array([math.cos(a), math.sin(a)], dtype=np.float64) * radius
            if not _trav_xy_ok(cand, trav_map, floor=0, clearance=0.25):
                continue
            if any(_point_in_rect(cand, r) for r in scene_rects):
                continue
            g = float(np.linalg.norm(cand - human_xy))
            if g > best_gap + 0.02:
                best, best_gap = cand, g
        if best is not None and best_gap >= min_gap:
            break                       # nearest ring that clears the freeze wins
    return best


def _yield_robot_off_path(env: Any, recorder: DemoRecorder, robot: Any, human_actor: Any,
                          full_path: List[np.ndarray], round_id: int,
                          clear: float = 0.6, trav_map: Any = None,
                          rects: Optional[List[Tuple[float, float, float, float]]] = None) -> bool:
    """Drive the ROBOT off the human's planned path (真 action, recorded frames).

    Why: the human is KINEMATIC and — since the no-taper change (§13.3) — walks its
    path at constant speed without slowing for the robot. If the path runs through the
    robot's pose, the human simply shoves the dynamic robot over: ep0004 robot tilt
    180° (flat on the floor, z 0.49->0.00), ep0005 tilt 22°, both with min_gap exactly
    0.49m (the 0.5m hard-collision guard fires too late to prevent contact). Plan-time
    deflection of the HUMAN path handles most cases, but when the detour would clip a
    wall it used to give up and keep the original path — i.e. walk through the robot.

    Here the ROBOT steps aside instead: pick the nearest pose that (a) clears the whole
    path by >= clear, (b) is traversable, (c) is outside furniture rects; then physically
    drive there facing the human. Returns True if it found and drove to such a pose."""
    if _online_nav_driver is None or not full_path:
        return False
    try:
        pxy = np.array([np.asarray(p, dtype=np.float64)[:2] for p in full_path])
        rxy = to_numpy(robot.get_position_orientation()[0]).astype(np.float64)[:2]
        # Step AWAY from the nearest path point (i.e. roughly perpendicular to travel).
        i = int(np.argmin(np.linalg.norm(pxy - rxy[None, :], axis=1)))
        away = rxy - pxy[i]
        n = float(np.linalg.norm(away))
        away = (away / n) if n > 1e-6 else np.array([1.0, 0.0], dtype=np.float64)

        best = None
        for deg in (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0):
            a = math.radians(deg)
            ca, sa = math.cos(a), math.sin(a)
            v = np.array([away[0] * ca - away[1] * sa, away[0] * sa + away[1] * ca])
            for d in (0.5, 0.8, 1.1):
                cand = rxy + v * d
                if trav_map is not None and not _trav_xy_ok(cand, trav_map, floor=0, clearance=0.35):
                    continue
                if rects and any(_point_in_rect(cand, r) for r in rects):
                    continue
                if float(np.min(np.linalg.norm(pxy - cand[None, :], axis=1))) < clear + 0.15:
                    continue
                best = cand
                break
            if best is not None:
                break
        if best is None:
            return False

        def _on_frame(obs: Any, action: np.ndarray) -> None:
            _restore_pinned_objects(env)
            info = {"reward": 0.0, "terminated": False, "truncated": False}
            recorder.record("Relocate", obs, info, round_id=round_id, action=action)

        hxy = np.asarray(human_actor.base_pos, dtype=np.float64)[:2]
        face = math.atan2(float(hxy[1] - best[1]), float(hxy[0] - best[0]))
        _online_nav_driver.drive_to_pose((float(best[0]), float(best[1]), float(face)),
                                         on_frame=_on_frame, low_precision=True)
        print(f"[relocation] robot YIELDED off the human's path -> "
              f"({best[0]:.2f},{best[1]:.2f})", flush=True)
        return True
    except Exception as exc:
        print(f"[relocation] robot yield failed: {exc}", flush=True)
        return False


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
    dest_override: Optional[Any] = None,
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

    stair_xy = _find_staircase_xy(env) if (toward_staircase and dest_override is None) else None

    # Relocation obstacle set (chairs/sofas passed in PLUS scene TABLES + the
    # appliances now carried in avoid_objects). Built HERE — before destination
    # selection — so a config waypoint can be AABB-rejected if it lands inside any
    # inflated rect, and reused for the robot follow path further down.
    reloc_avoid = list(avoid_objects) if avoid_objects else []
    for obj in env.scene.objects:
        cat_t = (getattr(obj, "category", "") or "").lower()
        if "table" in cat_t:
            try:
                if float(to_numpy(obj.get_position_orientation()[0])[2]) < 1.0:
                    reloc_avoid.append(obj)
            except Exception:
                pass
    dest_rects = _build_scene_rects(reloc_avoid, tag="reloc-dest-map")

    # ---- Destination selection.
    # PRIORITY: walk toward the NEXT round's task area (task_xy, peeked by the
    # caller) so the human leads the robot to where the next task happens. Sample
    # in an annulus around task_xy, snap onto a navigable pixel, validate
    # reachability, and KEEP the validated A* path for reuse as the walked path.
    # Two annulus passes (params pinned), then fall back to staircase / random.
    def _sample_dest(annulus, attempts):
        """Return (dest_xyz, astar_path) or (None, None)."""
        for _ in range(attempts):
            try:
                if task_xy is not None:
                    ang = float(np.random.uniform(0.0, 2.0 * math.pi))
                    rr = float(np.random.uniform(annulus[0], annulus[1]))
                    cand_xy = task_xy + rr * np.array([math.cos(ang), math.sin(ang)],
                                                      dtype=np.float32)
                elif stair_xy is not None:
                    cand_xy = (stair_xy + np.random.uniform(-1.0, 1.0, size=2).astype(np.float32))[:2]
                else:
                    _, rand_pt = trav_map.get_random_point(
                        floor=0, reference_point=th.tensor(human_start, dtype=th.float32))
                    cand_xy = np.asarray(rand_pt, dtype=np.float32)[:2]
                # Snap onto a navigable pixel (annulus points may land in a wall
                # / furniture footprint).
                snapped = human_actor._snap_to_navigable(
                    np.asarray(cand_xy, dtype=np.float32), trav_map, 0)
                if snapped is None:
                    continue
                cand = np.array([float(snapped[0]), float(snapped[1]), z_human], dtype=np.float32)
                d = float(np.linalg.norm(cand[:2] - human_start[:2]))
                if d < min_dist:
                    continue
                p = human_actor._nav_path_between(human_start, cand, z_human, trav_map, 0)
                if len(p) >= 2:
                    plen = sum(float(np.linalg.norm(
                        np.asarray(p[i + 1])[:2] - np.asarray(p[i])[:2]))
                        for i in range(len(p) - 1))
                    if plen <= max_dist * 3.0:
                        return cand, p
            except Exception:
                pass
        return None, None

    dest = dest_astar_path = None
    dest_mode = None
    # DEBUG hook: HRC_FORCE_RELOC_DEST="x,y" forces the relocation destination to a
    # specific point (bypassing BOTH samplers) so a particular path — e.g. the
    # mirror-crossing walk — can be reproduced deterministically. Unset -> no effect.
    # NOTE: it must NOT null out dest_override (that falls into the short-sampler
    # `else` branch below, which overwrites the forced dest) — use its own branch.
    _forced_dest = os.environ.get("HRC_FORCE_RELOC_DEST", "").strip()
    _use_forced = bool(_forced_dest) and dest_override is not None
    if _use_forced:
        try:
            _fx, _fy = (float(v) for v in _forced_dest.split(","))
            dest = np.array([_fx, _fy, z_human], dtype=np.float32)
            dest_mode = "forced-debug"
            print(f"[relocation] FORCED debug dest=({_fx:.2f},{_fy:.2f}) "
                  f"(bypass both samplers)", flush=True)
            # 0-erosion path as a fallback so full_path has a basis even if the
            # 0.50m LEAD plan finds no route (avoids the straight-line-through-
            # walls last resort).
            _fp0 = human_actor._nav_path_between(human_start, dest, z_human, trav_map, 0)
            if _fp0 and len(_fp0) >= 2:
                dest_astar_path = list(_fp0)
        except Exception as _exc:
            print(f"[relocation] FORCED dest parse failed ({_forced_dest!r}): {_exc}", flush=True)
            _use_forced = False
    if _use_forced:
        pass   # dest already set by the debug hook -> skip banded + short samplers
    elif isinstance(dest_override, str) and str(dest_override).strip().lower() == "auto":
        # PER-PAIR AUTO endpoint (2026-07-08). One fixed endpoint cannot serve spawns
        # scattered over 1-7m: it landed 0.25m/0.39m away for ep0002/0003 (relocate
        # skipped, `path too short`) and 5.8m away for ep0004 (robot fell behind,
        # maxgap 3.15m). Search the scene's traversability map ON THE ROBOT'S BODY-WIDTH
        # erosion for the straightest endpoint in the medium band, masking out the
        # robot's own pose so the human's path can never run through it (the root cause
        # of the knock-overs). Offline geometry has no rooms, so a straight-but-through-
        # a-doorway endpoint is filtered out below with the RUNTIME seg_map.
        _cands: List[Dict[str, float]] = []
        try:
            from endpoint_search import search_endpoints as _search_eps
            _rb_xy = to_numpy(env.robots[0].get_position_orientation()[0]).astype(np.float64)[:2]
            _cands = _search_eps(np.asarray(human_start, dtype=np.float64)[:2],
                                 robot_xy=_rb_xy, top_k=6)
        except Exception as _exc:
            print(f"[relocation] AUTO endpoint search failed: {_exc}", flush=True)
        # The searcher already masks its planning map to the human's own room (via
        # floor_insseg_0.png, same transform as the trav png), so the whole A* path
        # stays inside it — no runtime seg_map re-check needed. room="any" means the
        # room was too small to hold a band-length walk and the cross-room fallback
        # was taken; that is intentional (better than skipping Relocate).
        if not _cands:
            print("[relocation] AUTO endpoint: search returned no candidate "
                  "— SKIP relocation", flush=True)
            return
        _pick = _cands[0]
        dest = np.array([float(_pick["x"]), float(_pick["y"]), z_human], dtype=np.float32)
        dest_mode = "auto"
        print(f"[relocation] AUTO endpoint ({_pick['x']:.2f},{_pick['y']:.2f}) "
              f"dist={_pick['dist']:.1f}m detour={_pick['detour']:.2f}x "
              f"clear={_pick['clear']:.2f}m room={_pick.get('room', '?')} "
              f"(of {len(_cands)} candidates)", flush=True)
    elif _normalize_relocation_dest_candidates(dest_override):
        # PREDEFINED endpoint set (2026-07-07 plan): config relocation_dest is a point
        # or list of points -> pick ONE at random and A* to it. No random sampling:
        # every endpoint is a hand-picked, pre-validated OPEN location the robot can
        # trail to, so the relocation is deterministically correct. Diversity comes
        # from (endpoint set) x (pair-varying human start) x (leash dynamics), not from
        # sampling the whole floor. Combined with NO-taper (human walks the full path),
        # this eliminates the mid-path idle/jam that random dests caused.
        _cands = _normalize_relocation_dest_candidates(dest_override)
        _ep = _cands[np.random.randint(len(_cands))]
        dest = np.array([float(_ep[0]), float(_ep[1]), z_human], dtype=np.float32)
        dest_mode = "predefined"
        try:
            _fp0 = human_actor._nav_path_between(human_start, dest, z_human, trav_map, 0)
            if _fp0 and len(_fp0) >= 2:
                dest_astar_path = list(_fp0)   # 0-erosion fallback; main A* re-plans at 0.50
        except Exception:
            pass
        print(f"[relocation] PREDEFINED endpoint ({float(_ep[0]):.2f},{float(_ep[1]):.2f}) "
              f"chosen from {len(_cands)} candidate(s)", flush=True)
    elif dest_override is not None:
        # BANDED navigable-first sampler (approved plan 2026-07-01). The relocation
        # walk is DIVERSITY-driven — short/medium/long bands by A* path length — for
        # π0 training on full episodes; the follow segment must vary, not cluster.
        #   Why not the old random-annulus+snap: after Fix-1 baked furniture into the
        #   trav_map, 60-90% of annulus candidates fell in the <0.35m wall band
        #   (ep0042/49/50 fails={trav:73-93}) -> 0 valid -> fixed config waypoint ->
        #   SHORT walk whose path AIMED AT the robot -> deadlock / knock-over / hit.
        #   get_random_point() samples navigable cells of the human's connected
        #   region directly, so candidates start on walkable floor.
        #   The fixed relocation_dest waypoint fallback is REMOVED: if nothing
        #   validates, SKIP relocation entirely (log + QC catches the missing
        #   follow) — never a short robot-aimed walk again.
        # Hard filters kept: dest outside inflated furniture rects; >=0.35m trav
        # clearance; A* reachable (0.35m human erosion, same as the walk); WHOLE
        # path >=0.7m from the robot's current pose (root fix for the 0.49m
        # near-touch). Softened: _leg_robot_clear(0.40m) is a PREFERENCE, not a
        # kill — the two-sided leash + batch gap-QC arbitrate when it fails.
        _BAND_NAMES = ("short", "medium", "long")
        _BANDS = ((1.5, 3.0), (3.0, 6.0), (6.0, 9.0))   # A* path-length bands (m)
        # A2 (user: scene is small -> MEDIUM-primary): weight the band pick toward
        # medium instead of uniform; long is rarely reachable in this house anyway.
        _band_pick = int(np.random.choice(3, p=[0.25, 0.5, 0.25]))
        _MIN_WALK = 1.0   # A: accept a slightly-short walk (closest-to-band) over skipping
        _rb = to_numpy(env.robots[0].get_position_orientation()[0]).astype(np.float64)[:2]
        _fail = {"outdoor": 0, "rect": 0, "trav": 0, "nopath": 0, "tooshort": 0, "nearrobot": 0}
        _cands = []   # (in_band, band_idx, band_dist, robotleg_ok, plen, dest_xyz, astar_path)
        _t_ref = th.tensor([float(human_start[0]), float(human_start[1]), float(z_human)],
                           dtype=th.float32)
        # INDOOR POOL (User A方案 2026-07-02): this scene's trav_map covers the whole
        # YARD, and get_random_point(reference_point) samples the ENTIRE connected
        # component — indoors AND the outdoors reachable through a door — so ~90% of
        # picks landed OUTDOORS (ep0004 fails={outdoor:54/60}) -> band-starve -> most
        # ep0052+ SKIPPED relocation entirely. Fix: build the connected INDOOR
        # navigable pool ONCE (trav_map navigable cells whose seg_map room instance is
        # not None = inside a loaded room) and draw all dests from it, so every
        # candidate is already in the in-house walkable region. get_random_point +
        # per-pick outdoor-reject is the fallback when the scene has no seg_map.
        try:
            _seg = getattr(og.sim.scenes[0], "seg_map", None) if og.sim.scenes else None
            if _seg is not None and getattr(_seg, "room_ins_map", None) is None:
                _seg = None
        except Exception:
            _seg = None
        _pool = None
        if _seg is not None:
            try:
                _ij = np.argwhere(to_numpy(trav_map.floor_map[0]) > 0)   # navigable (row,col)
                if len(_ij) > 3000:
                    _ij = _ij[np.random.choice(len(_ij), 3000, replace=False)]
                _wxy = to_numpy(trav_map.map_to_world(th.tensor(_ij, dtype=th.float32)))
                _pool = np.array([w[:2] for w in _wxy
                                  if _seg.get_room_instance_by_point(
                                      np.asarray(w[:2], dtype=np.float32)) is not None],
                                 dtype=np.float32)
                if len(_pool) < 20:
                    _pool = None                                # too few indoor cells -> fallback
            except Exception:
                _pool = None
        print(f"[relocation] indoor pool = {0 if _pool is None else len(_pool)} navigable cells; "
              f"band pick={_BAND_NAMES[_band_pick]}", flush=True)
        for _try in range(90):
            if _pool is not None:
                cand_xy = _pool[np.random.randint(len(_pool))].astype(np.float32).copy()
            else:
                try:
                    _, _pt = trav_map.get_random_point(floor=0, reference_point=_t_ref)
                    cand_xy = np.array([float(_pt[0]), float(_pt[1])], dtype=np.float32)
                    if _seg is not None and _seg.get_room_instance_by_point(cand_xy) is None:
                        _fail["outdoor"] += 1; continue
                except Exception:
                    continue
            if any(_point_in_rect(cand_xy, r) for r in dest_rects):
                _fail["rect"] += 1; continue
            if not _trav_xy_ok(cand_xy, trav_map, floor=0, clearance=0.35):
                _fail["trav"] += 1; continue
            cand = np.array([cand_xy[0], cand_xy[1], z_human], dtype=np.float32)
            p = human_actor._nav_path_between(human_start, cand, z_human, trav_map)
            if len(p) < 2:
                _fail["nopath"] += 1; continue
            plen = sum(float(np.linalg.norm(np.asarray(p[i + 1])[:2] - np.asarray(p[i])[:2]))
                       for i in range(len(p) - 1))
            if plen < _MIN_WALK:
                _fail["tooshort"] += 1; continue
            if any(float(np.linalg.norm(np.asarray(q, dtype=np.float64)[:2] - _rb)) < 0.7
                   for q in p):
                _fail["nearrobot"] += 1; continue
            # A: classify into a band, but KEEP out-of-band candidates too (tagged with
            # the gap to the nearest band) so we accept the CLOSEST-TO-BAND walk rather
            # than SKIP when the picked band can't be filled in a tight scene.
            _bi = next((i for i, b in enumerate(_BANDS) if b[0] <= plen <= b[1]), None)
            if _bi is None:
                _nb = min(range(3), key=lambda i: min(abs(plen - _BANDS[i][0]), abs(plen - _BANDS[i][1])))
                _band_dist = min(abs(plen - _BANDS[_nb][0]), abs(plen - _BANDS[_nb][1]))
                _in_band = False
            else:
                _nb, _band_dist, _in_band = _bi, 0.0, True
            _rleg = _leg_robot_clear(human_start, cand, z_human, trav_map, dest_rects,
                                     human_actor, 0.40, 0.40)
            _cands.append((_in_band, _nb, _band_dist, _rleg, plen, cand, p))
            if _in_band and _nb == _band_pick and _rleg:
                break                        # ideal hit: picked band + robot-followable
        if _cands:
            # Prefer: in-picked-band+robotleg > in-picked-band > in-any-band > closest-
            # to-a-band; robotleg-ok and closeness to the picked-band midpoint tie-break.
            _mid = sum(_BANDS[_band_pick]) / 2.0
            _cands.sort(key=lambda c: (not (c[0] and c[1] == _band_pick),
                                       not c[0], not c[3], c[2], abs(c[4] - _mid)))
            _in_band, _nb, _band_dist, _rleg, _plen, dest, dest_astar_path = _cands[0]
            dest_mode = f"banded-{_BAND_NAMES[_nb]}" + ("" if _in_band else "~")
            _bc = {n: sum(1 for c in _cands if c[1] == i) for i, n in enumerate(_BAND_NAMES)}
            print(f"[relocation] dest = BANDED {_BAND_NAMES[_nb]}{'' if _in_band else '~closest'} "
                  f"(pick={_BAND_NAMES[_band_pick]}) ({dest[0]:.2f},{dest[1]:.2f}) "
                  f"pathlen={_plen:.2f}m robotleg={'ok' if _rleg else 'soft-fail'} "
                  f"cands={_bc} fails={_fail}", flush=True)
        else:
            # A3 (user "走不了远就向前再向后兜个小圈"): tight pocket, no valid dest ->
            # walk OUT ~loop_r in a clear direction then BACK, so there is ALWAYS a
            # follow segment (never a bare SKIP). The dest returns to the human start.
            _loop_r = 2.0
            for _ in range(40):
                _a = float(np.random.uniform(0.0, 2.0 * math.pi))
                _fwd = human_start[:2].astype(np.float32) + _loop_r * np.array(
                    [math.cos(_a), math.sin(_a)], dtype=np.float32)
                _fs = human_actor._snap_to_navigable(_fwd, trav_map, 0)
                if _fs is None:
                    continue
                _fc = np.array([float(_fs[0]), float(_fs[1]), z_human], dtype=np.float32)
                if (float(np.linalg.norm(_fc[:2] - human_start[:2])) < 1.0
                        or any(_point_in_rect(_fc[:2], r) for r in dest_rects)
                        or not _trav_xy_ok(_fc[:2], trav_map, floor=0, clearance=0.35)):
                    continue
                _op = human_actor._nav_path_between(human_start, _fc, z_human, trav_map,
                                                    erosion_radius=_RELOC_LEAD_EROSION_M)
                if len(_op) < 2 or any(
                        float(np.linalg.norm(np.asarray(q, dtype=np.float64)[:2] - _rb)) < 0.7
                        for q in _op):
                    continue
                dest_astar_path = list(_op) + list(reversed(_op))[1:]   # out and back
                dest = np.array([human_start[0], human_start[1], z_human], dtype=np.float32)
                dest_mode = "small-loop"
                print(f"[relocation] dest = SMALL-LOOP out {_loop_r:.1f}m to "
                      f"({_fc[0]:.2f},{_fc[1]:.2f}) and back (tight scene; fails={_fail})", flush=True)
                break
            if dest is None:
                print(f"[relocation] BANDED + small-loop found none (fails={_fail}) — SKIP "
                      f"relocation", flush=True)
                return
    else:
        dest, dest_astar_path = _sample_dest((0.5, 2.5), 200)
        if dest is None:
            dest, dest_astar_path = _sample_dest((0.5, 4.0), 100)
        if dest is None:
            print("[relocation] could not find short navigable destination; skipping", flush=True)
            return
    task_dist = float(np.linalg.norm(dest[:2] - task_xy)) if task_xy is not None else -1.0
    mode = dest_mode or ("task" if task_xy is not None else ("stair" if stair_xy is not None else "random"))
    print(f"[relocation] dest=({dest[0]:.2f},{dest[1]:.2f}) "
          f"dist_from_start={float(np.linalg.norm(dest[:2] - human_start[:2])):.2f} "
          f"dist_from_task={task_dist:.2f} mode={mode}", flush=True)

    # (reloc_avoid + dest_rects were built above, before destination selection.)

    # ---- Human walked path: PREFER the trav_map A* route. The kinematic human
    # has no collision body, so its path must stay on navigable space to avoid
    # clipping. `_nav_path_between` routes around walls + furniture footprints
    # (trav_map_with_objects=True) with a body-width erosion margin.
    # IMPORTANT: do NOT run _deflect_path_around_point / _simplify_path on an A*
    # path — those are pure geometry (no trav_map) and can re-introduce clipping
    # by cutting corners. They are used ONLY on the straight-line fallback.
    # LEAD path uses the ROBOT's body width (not the human's 0.35m) so the human
    # never routes through a gap the trailing robot can't follow (A-fix).
    HUMAN_EROSION = _RELOC_LEAD_EROSION_M   # m body-width band for the human LEAD path
    full_path: Optional[List[np.ndarray]] = None
    if dest_mode == "small-loop" and dest_astar_path is not None and len(dest_astar_path) >= 2:
        # The small-loop path (start -> fwd -> back to start) must NOT be shortcut-
        # straightened: _shortcut_path_on_trav_map would collapse start->...->start
        # back to the start point (total_len ~= 0 -> the human never moves; ep0058).
        # Resample-only preserves the out-and-back so the human actually walks it.
        full_path = _resample_path(
            [np.asarray(p, dtype=np.float32) for p in dest_astar_path], spacing=0.08)
        print(f"[relocation] human path = SMALL-LOOP out-and-back "
              f"({len(dest_astar_path)}->{len(full_path)} pts, no shortcut)", flush=True)
    # Random WANDER (Habitat OracleNavRandCoordAction-style): route the human
    # through 1-3 validated random navigable waypoints before the final leg to
    # `dest`, so the follow trajectory varies instead of one straight A*. The last
    # leg ALWAYS ends at `dest` -> the next round still starts in place. If any leg
    # is unreachable, fall back to the direct single-leg A* (legacy behavior).
    try:
        # Predefined / forced endpoints are hand-picked & pre-validated -> NO random
        # wander waypoints (those turn the deterministic straight path into an
        # unvalidated detour, e.g. 58g's 5m endpoint became an 8.3m mirror-grazing
        # loop). Random-sampled (banded) dests still wander for trajectory diversity.
        wander = ([] if dest_mode in ("predefined", "forced-debug", "auto")
                  else _sample_walk_waypoints(human_actor, trav_map, human_start, dest,
                                              z_human, dest_rects, n_min=1, n_max=3))
        legs = [np.asarray(human_start, dtype=np.float32), *wander,
                np.asarray(dest, dtype=np.float32)]
        astar = []
        for i in range(len(legs) - 1):
            seg = human_actor._nav_path_between(legs[i], legs[i + 1], z_human,
                                                trav_map, erosion_radius=HUMAN_EROSION)
            if len(seg) < 2:
                astar = None
                break
            astar.extend(seg if i == 0 else seg[1:])   # drop the duplicated joint
        if astar and wander:
            print(f"[relocation] human wander: {len(wander)} random waypoint(s) "
                  f"{[f'({p[0]:.2f},{p[1]:.2f})' for p in wander]} -> dest "
                  f"({len(astar)} A* pts)", flush=True)
    except Exception as exc:
        print(f"[relocation] wander path failed: {exc}", flush=True)
        astar = None
    if not astar:
        try:
            astar = human_actor._nav_path_between(
                human_start, dest, z_human, trav_map, erosion_radius=HUMAN_EROSION)
        except Exception as exc:
            print(f"[relocation] A* path failed: {exc}", flush=True)
            astar = None
    if full_path is None and astar is not None and len(astar) >= 2:
        # Patch 1: trav-aware shortcut (de-staircase, clearance-preserving) +
        # 0.08m resample, instead of using the raw grid-staircase A* points.
        full_path = _prepare_human_walk_path(astar, trav_map, floor=0,
                                             clearance=_RELOC_LEAD_EROSION_M)
    elif dest_astar_path is not None and len(dest_astar_path) >= 2:
        # Eroded plan had no path; reuse the 0-erosion A* validated during dest
        # selection, also trav-aware shortcut + resample it (Patch 1).
        full_path = _prepare_human_walk_path(dest_astar_path, trav_map, floor=0)
        print("[relocation] human path = trav_map A* (0-erosion fallback)", flush=True)

    if full_path is None:
        # Last-resort fallback (A* fully failed): straight line + geometric
        # furniture deflection. Legacy behavior; can clip walls not in reloc_avoid.
        print("[relocation] WARN: A* unavailable — straight-line + deflect fallback", flush=True)
        full_path = [
            human_start.copy(),
            np.array([float(dest[0]), float(dest[1]), z_human], dtype=np.float32),
        ]
        for obj in reloc_avoid:
            try:
                obj_pos = to_numpy(obj.get_position_orientation()[0]).astype(np.float32)[:2]
                cat_h = (getattr(obj, "category", "") or "").lower()
                if "table" in cat_h:
                    cl = 0.7
                elif "chair" in cat_h or "sofa" in cat_h or "stool" in cat_h:
                    cl = 0.6
                else:
                    cl = 0.4
                path_pts = np.array([p[:2] for p in full_path], dtype=np.float32)
                dists = np.linalg.norm(path_pts - obj_pos[None, :], axis=1)
                if float(np.min(dists)) > 1.2:
                    continue
                full_path = _deflect_path_around_point(full_path, obj_pos, clearance=cl)
            except Exception:
                pass
        full_path = _simplify_path(full_path, epsilon=0.3)

    # Point 1: keep the human's path clear of the ROBOT (a circular obstacle at
    # its CURRENT pose) so the collision-less kinematic human doesn't walk THROUGH
    # the robot. Deflect around the robot point, then trav-validate; revert to the
    # original if the detour broke navigability (don't trade a robot-clip for a
    # wall-clip). Static: the robot stays put while slack and only moves once the
    # human is already past, so plan-time avoidance of its current pose suffices.
    try:
        robot_now_xy = to_numpy(robot.get_position_orientation()[0]).astype(np.float32)[:2]
        # human half-width (~0.25) + robot radius (~0.30) = 0.55m of ACTUAL contact, so
        # 0.6 left no margin: ep0004/0005 both touched at exactly 0.49m and the human
        # shoved the robot over. Plan the human's path 0.75m clear, and keep the runtime
        # freeze guard (hard_collision_r) just below that so it never fires spuriously.
        _HUMAN_ROBOT_CLEAR = 0.75
        _pxy = np.array([np.asarray(p, dtype=np.float32)[:2] for p in full_path], dtype=np.float32)
        _d_robot = float(np.min(np.linalg.norm(_pxy - robot_now_xy[None, :], axis=1))) if len(_pxy) else 9.9
        if _d_robot < _HUMAN_ROBOT_CLEAR:
            _defl = _deflect_path_around_point(full_path, robot_now_xy, clearance=_HUMAN_ROBOT_CLEAR)
            if len(_defl) >= 2 and all(
                    _segment_trav_ok(_defl[i][:2], _defl[i + 1][:2], trav_map, floor=0)
                    for i in range(len(_defl) - 1)):
                full_path = _resample_path(_defl, spacing=0.08)
                print(f"[relocation] human path deflected around robot "
                      f"@({robot_now_xy[0]:.2f},{robot_now_xy[1]:.2f}) (was {_d_robot:.2f}m clear)", flush=True)
            else:
                # The detour would clip a wall. Do NOT keep the original path: with the
                # no-taper human that means walking straight through the robot and
                # knocking it over (ep0004 tilt 180°, ep0005 22°, both min_gap 0.49m).
                # Make the ROBOT step aside instead.
                if not _yield_robot_off_path(env, recorder, robot, human_actor, full_path,
                                             round_id, clear=_HUMAN_ROBOT_CLEAR,
                                             trav_map=trav_map):
                    print(f"[relocation] WARN: deflection broke navigability AND robot could "
                          f"not yield — human passes {_d_robot:.2f}m from robot", flush=True)
    except Exception as _exc:
        print(f"[relocation] robot-avoid deflect skipped: {_exc}", flush=True)

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

    # Policy rollout keeps the existing, validated human path and animation,
    # but never asks OnlineNavDriver (or the kinematic fallback) for a robot
    # action.  Every robot frame below comes from π0.5 and passes through
    # _policy_step -> env.step.
    if _policy_client is not None:
        for step_i in range(walk_frames):
            human_d = ((step_i + 1) / walk_frames) * total_len
            human_pos = _interp_path(full_path, cum_dist, human_d)
            human_pos[2] = z_human
            human_yaw = _yaw_at_dist(full_path, cum_dist, human_d)
            per_frame_disp = total_len / max(walk_frames, 1)
            human_actor.base_pos = human_pos.copy()
            human_actor.base_yaw = float(human_yaw)
            human_actor.linear_speed = float(walk_speed)
            human_actor.motion_state = "walk"
            human_actor._last_displacement_m = per_frame_disp
            human_actor.gait_phase += per_frame_disp * 2.0
            human_actor._advance_animation()
            human_actor.apply_pose()
            _policy_step(
                env,
                recorder,
                "Relocate",
                round_id,
                human_actor=human_actor,
                advance_human=False,
                task_instruction=FOLLOW_POLICY_INSTRUCTION,
            )

        robot_xy = to_numpy(robot.get_position_orientation()[0]).astype(np.float32)[:2]
        delta = robot_xy - human_actor.base_pos[:2]
        if float(np.linalg.norm(delta)) > 1e-6:
            human_actor.base_yaw = math.atan2(float(delta[1]), float(delta[0]))
        human_actor.anchor_pos = human_actor.base_pos.copy()
        human_actor.motion_state = "hold"
        human_actor.linear_speed = 0.0
        human_actor.apply_pose()
        print(f"[relocation] done (π0.5 policy) — human now at "
              f"({human_actor.base_pos[0]:.2f}, {human_actor.base_pos[1]:.2f})", flush=True)
        return

    start_yaw_robot = _yaw_from_quat(to_numpy(robot.get_position_orientation()[1]))
    turn_frames = max(int(walk_frames * 0.3), 1)

    # ---- Physics-online relocation (--online-nav): the robot PHYSICALLY follows
    # the human on a LEASH instead of being teleported or sliding along a
    # precomputed route. Records real base-velocity actions every frame.
    if _online_nav_driver is not None:
        # ---- LEASH follow (Habitat-style smooth co-motion). A virtual rope of
        # rest length L=stand_off ties the human to the robot, but instead of the
        # old BINARY slack(freeze)/taut(lurch) + binary human tether (which caused
        # the v6 stop-and-go "frog hop"), the human SPEED and the robot TRAIL are
        # both DISTANCE-MODULATED so the pair moves together (HANDOFF §9.4 残留①;
        # mirrors Habitat set_framerate_for_linspeed + "maintain the band"):
        #   - Human: full speed while close (d <= taper_relax), eases smoothly to 0
        #     as the rope stretches to taper_soft_cap (no hard stop until then).
        #   - Robot: trails CONTINUOUSLY toward the human's CURRENT position once
        #     d > follow_far (= L + deadband); inside the deadband it only rotates
        #     to face the human (never translates / reverses). The target is ALWAYS
        #     anchored to the human's LIVE pose -> still never chases the vacated
        #     spot (the v5 guard) and the straight-pull core is unchanged (no
        #     precomputed route -> no v5 12m detour).
        # taper_soft_cap > follow_far gives an overlap window [follow_far,
        # taper_soft_cap] where the human creeps AND the robot closes at the same
        # time -> smooth, no lurch.
        scene_rects = _build_scene_rects(reloc_avoid, tag="scene-map")
        leash_L = float(stand_off)            # 1.5m rest length
        leash_pull_to = leash_L - 0.5         # 1.0m: robot pull target (in-band)
        leash_deadband = 0.15                 # hysteresis above the pull target
        # FIX (v9 leash regression — robot only rotated, didn't walk): old
        # follow_far = leash_L + deadband = 1.65m, but the tethered + speed-tapered
        # human self-limits the gap to <=~1.65m, so d never EXCEEDED follow_far and
        # the robot stayed permanently in the rotate-only deadband (measured: during
        # Relocate the robot translated 0.06m while rotating 50deg). Anchor follow_far
        # to the PULL TARGET so the robot starts closing as soon as it falls >0.15m
        # behind its 1.0m follow distance -> it co-walks with the human. Target is
        # still the human's LIVE pose, so the v5 'chase the vacated spot' guard holds.
        follow_far = leash_pull_to + leash_deadband  # 1.15m: robot CLOSES IN above this
        # Two-sided leash (ep0004 fix): the robot must also BACK OFF when the human
        # approaches it. Fix 2's co-walk broke the old "robot stays put while slack"
        # premise the plan-time robot-deflection relied on, so a human path that grazes
        # the robot's pose no longer self-resolves -> the human walked INTO the robot
        # and crashed the episode. Below leash_near the robot reverses out to
        # leash_pull_to; hard_collision_r is the last-resort human freeze.
        leash_near = leash_pull_to - 0.15     # 0.85m: robot backs off below this
        hard_collision_r = 0.60               # m: human freezes if this close to robot.
        #                                      Was 0.5 — but human half-width + robot
        #                                      radius ~= 0.55m, so contact (and the
        #                                      knock-over) happened at 0.49m before the
        #                                      guard could stop the human (ep0004/0005).
        taper_relax = leash_L - 0.1           # 1.4m: human full speed at/under this
        taper_soft_cap = leash_L + 0.4        # 1.9m: human eased to ~0 here
        goal_eps = 0.12                       # m: Fix 1+2 — human force-finishes the
                                              #    last leg + done-check tolerance, so
                                              #    the taper can't strand it short of
                                              #    dest and idle out max_frames.

        human_speed = min(float(walk_speed), 0.5)
        human_step = human_speed / float(max(fps, 1))
        human_d = 0.0
        max_frames = int(total_len / max(human_speed, 1e-3) * fps) + int(12 * fps)

        print(f"[relocation] --online-nav: LEASH follow co-motion (L={leash_L:.2f}m, "
              f"deadband={leash_deadband:.2f}, taper[{taper_relax:.1f},{taper_soft_cap:.1f}]m, "
              f"human_speed={human_speed:.2f}m/s, path={total_len:.2f}m)", flush=True)

        # Follow in the canonical 0:02 home posture (human_actor=None: the
        # human is driven manually by this loop, and "Relocate" has no
        # human phase spec — it just stands holding until the lead-walk).
        _settle_arms_to_home(env, recorder, "Relocate", round_id, human_actor=None)
        # ② Habitat _reach_human gate (supersedes the bare base-stability wait):
        # ACTIVELY drive the robot to reached+facing the human, THEN settle, before
        # the human starts leading (HANDOFF §5 Patch2 ⚠: the gate must prove "in
        # position + facing", not just "stopped"). ReturnToHuman/FaceHuman usually
        # leave it satisfied -> 0 corrective frames. Standoff = clear point 1.5m
        # from the (stationary) human on the robot's side.
        gate_human_xy = np.asarray(human_start, dtype=np.float64)[:2]
        gate_robot_xy = to_numpy(robot.get_position_orientation()[0]).astype(np.float64)[:2]
        gate_dir = gate_robot_xy - gate_human_xy
        gn = float(np.linalg.norm(gate_dir))
        gate_dir = gate_dir / gn if gn > 1e-6 else np.array([1.0, 0.0], dtype=np.float64)
        gate_standoff = _find_clear_standoff(
            gate_human_xy.astype(np.float32), gate_dir.astype(np.float32),
            float(stand_off), reloc_avoid, min_clearance=_ROBOT_STANDOFF_CLEAR,
            avoid_rects=scene_rects, trav_map=trav_map)
        _settle_and_face_human(env, recorder, human_actor, round_id,
                               gate_human_xy, gate_standoff, phase="Relocate")
        _online_nav_driver.pin_arm_hold()  # hold arms at home for the whole follow

        # Deadlock safety net (two detectors):
        #  (both-frozen) human isn't advancing AND robot isn't translating. When the
        #    robot wedges (target clamped at a wall/obstacle -> target~=robot_xy), the
        #    gap stays wide, the human tapers to 0 waiting on it, and the loop would
        #    idle out its whole max_frames budget as a multi-second FREEZE (ep0058
        #    mirror: 35s). End the follow gracefully after >~4s.
        #  (chase-stuck / C-fix) the robot is BEHIND its follow band (gap>follow_far,
        #    so it SHOULD be closing) yet isn't translating. This catches the blind
        #    spot of both-frozen: the robot jams while the human keeps LEADING away
        #    (ep0058: robot pinned at the mirror 14.7s but both-frozen only fired for
        #    the last 4s once the human also stopped). Fires after >~5s. In-band holds
        #    never trip it (there gap<=follow_far).
        _stuck_frames = 0
        _chase_stuck_frames = 0
        _prev_robot_xy = to_numpy(robot.get_position_orientation()[0]).astype(np.float64)[:2]
        _prev_human_d = human_d
        _STUCK_LIMIT = int(4.0 * fps)
        _CHASE_STUCK_LIMIT = int(5.0 * fps)

        # Frames in which the human is frozen by the collision guard buy no progress, so
        # charging them to `max_frames` silently truncates the walk (ep0002 covered 1.27m
        # of 4.21m and still "finished"). Refund them, up to a cap so a genuine wedge
        # still terminates.
        _frozen_frames = 0
        _MAX_FROZEN_REFUND = int(20.0 * fps)
        _frame = -1
        while _frame + 1 < max_frames + min(_frozen_frames, _MAX_FROZEN_REFUND):
            _frame += 1
            # --- 0. Measure rope length d (robot <-> human's CURRENT position) ---
            robot_xy = to_numpy(robot.get_position_orientation()[0]).astype(np.float64)[:2]
            human_xy_cur = np.asarray(_interp_path(full_path, cum_dist, human_d),
                                      dtype=np.float64)[:2]
            d = float(np.linalg.norm(robot_xy - human_xy_cur))

            # --- 1. Human leads with DISTANCE-MODULATED speed (taper): full speed
            # while the robot is close (d <= taper_relax), easing smoothly to 0 as
            # the rope stretches to taper_soft_cap, so it co-moves with the robot
            # instead of the old binary full-speed/hard-stop tether.
            # NO taper (2026-07-07 predefined-path plan): the human walks the whole
            # pre-validated path at CONSTANT speed and never stops to wait for the
            # robot. The robot trails as best it can (gap may grow — still a valid
            # follow demo). This removes the mid-path idle the taper caused (human
            # stopping at the gap cap, then old code idling to timeout). The
            # hard-collision guard below still stops the human walking INTO the robot.
            speed_factor = 1.0
            # Fix 1: within goal_eps of the path end, OVERRIDE the taper to full speed
            # so the human always reaches dest — even if the robot is stuck far (e.g.
            # wedged in a furniture pocket) and the taper would otherwise zero the
            # human's speed a few cm short, which deadlocks the done-check below and
            # idles out the whole ~12*fps catch-up budget (the v8 5:30-5:40 freeze).
            if (total_len - human_d) <= goal_eps:
                speed_factor = 1.0
            # Collision guard (ep0004): if already within hard-collision range of the
            # robot, HOLD this frame so the kinematic human never walks THROUGH / shoves
            # it. The two-sided leash below actively backs the robot off, so this clears
            # within a few frames (no deadlock). Overrides goal_eps -> never ram the robot.
            if d < hard_collision_r:
                speed_factor = 0.0
                _frozen_frames += 1
            step_eff = human_step * speed_factor
            _dbg_target = float("nan")
            _dbg_yaw_err = float("nan")
            _dbg_branch = "else"
            _dbg_yaw_before = float(human_actor.base_yaw)
            if step_eff > 1e-9 and human_d < total_len - 1e-9:
                look_d = min(human_d + step_eff, total_len)
                human_yaw_target = _yaw_at_dist(full_path, cum_dist, look_d)
                yaw_err = abs(_wrap_to_pi(human_yaw_target - float(human_actor.base_yaw)))
                _dbg_target, _dbg_yaw_err = human_yaw_target, yaw_err
                human_actor.base_yaw = _limited_turn(float(human_actor.base_yaw), human_yaw_target)
                if yaw_err > math.radians(_HUMAN_TURN_THRESH_DEG):
                    disp = 0.0                                   # turn in place first
                    _dbg_branch = "turn"
                    human_actor.motion_state = "turn"
                    human_actor.linear_speed = 0.0
                else:
                    new_d = min(human_d + step_eff, total_len)
                    disp = max(new_d - human_d, 0.0)
                    human_d = new_d
                    _dbg_branch = "walk"
                    human_actor.motion_state = "walk" if disp > 1e-5 else "hold"
                    human_actor.linear_speed = (human_speed * speed_factor) if disp > 1e-5 else 0.0
            else:
                disp = 0.0                                       # rope over-stretched -> wait
                human_actor.motion_state = "hold"
                human_actor.linear_speed = 0.0
            # HRC_DEBUG_RELOC=1: per-frame trace of the freeze/yield interaction. This is
            # what proved the human halts at d<hard_collision_r and never resumes because
            # the robot creeps IN (0.589 -> 0.495m) instead of clearing.
            if _RELOC_DEBUG and (_frame % 10 == 0 or _frame < 20):
                print(f"[reloc-dbg] f={_frame:4d} branch={_dbg_branch:4s} d={d:.3f} "
                      f"sf={speed_factor:.2f} step_eff={step_eff:.5f} "
                      f"human_d={human_d:.3f}/{total_len:.3f} "
                      f"yaw_before={math.degrees(_dbg_yaw_before):7.2f} "
                      f"yaw_after={math.degrees(float(human_actor.base_yaw)):7.2f} "
                      f"target={math.degrees(_dbg_target):7.2f} "
                      f"yaw_err={math.degrees(_dbg_yaw_err):6.2f} disp={disp:.5f} "
                      f"ms={getattr(human_actor, 'motion_state', '?')}", flush=True)
            human_pos = _interp_path(full_path, cum_dist, human_d)
            human_pos[2] = z_human
            human_actor.base_pos = human_pos.copy()
            human_actor._last_displacement_m = disp
            human_actor.gait_phase += disp * 2.0
            human_actor._advance_animation()
            human_actor.apply_pose()
            _restore_pinned_objects(env)

            # --- 2. Robot trails with a TWO-SIDED deadband (continuous), and YIELDS
            # to a CLEAR direction when crowded (not a blind straight-back that could
            # reverse into a wall). Target always anchored to the human's LIVE pose ->
            # never chases the vacated spot (v5 guard).
            human_xy2 = human_pos[:2].astype(np.float64)
            d_robot = float(np.linalg.norm(robot_xy - human_xy2))
            hr = robot_xy - human_xy2
            hrn = float(np.linalg.norm(hr))
            hr = hr / hrn if hrn > 1e-6 else np.array([1.0, 0.0], dtype=np.float64)
            # Which way is the human walking, and is it heading straight AT the robot?
            _hy = float(getattr(human_actor, "base_yaw", 0.0))
            _hdir = np.array([math.cos(_hy), math.sin(_hy)], dtype=np.float64)
            _closing = float(np.dot(_hdir, hr))   # +1 = walking right at the robot

            if _closing > 0.5 and d_robot < leash_near + 0.45:
                # PREDICTIVE YIELD (highest priority). Real contact is at ~0.55m (human
                # half-width 0.25 + robot radius 0.30). At 0.5 m/s the old 0.85m trigger
                # left only 0.6s — not enough for the base to accelerate clear, so the
                # human simply walked over the robot (ep0004 tilt 180°, ep0005 22°, both
                # min_gap exactly 0.49m). Start moving aside at 1.30m (~1.4s of margin),
                # and step ACROSS the human's lane rather than backing up along it.
                target_xy = _pick_yield_target(robot_xy, hr, _hdir, trav_map, scene_rects,
                                               human_xy=human_xy2,
                                               min_gap=hard_collision_r + 0.10)
                if target_xy is None:
                    target_xy = _pick_yield_escape(robot_xy, human_xy2, trav_map,
                                                   scene_rects, hard_collision_r + 0.10)
                if target_xy is None:
                    # Truly boxed in. NEVER fall back to `robot_xy`: the base then creeps
                    # toward the frozen human (0.589 -> 0.495m measured) and eventually
                    # tips over it. Back straight off instead, wall-clamped.
                    target_xy = _clamp_target_to_wall_clear(
                        robot_xy, robot_xy + hr * 0.30, trav_map, clearance=0.25)
            elif d_robot > follow_far:
                # TOO FAR -> TRAIL the human ALONG ITS OWN path instead of direct-pulling
                # toward its live pose. RC1 (2026-07-17, user-approved): the old direct pull
                # + reactive _route_around_rects drove the robot 0.66-1.29m OFF the
                # followable path into the dining/fridge furniture and tipped it
                # (ep7/14/27/28/51 fell there). The human's full_path is already 0.50m-eroded
                # (~robot body-width, `_RELOC_LEAD_EROSION_M`), so a point one pull-length
                # BACK along it is guaranteed followable. This is NOT the removed pre-computed
                # 12m robot path (§9.2): the trail point is anchored to the human's CURRENT
                # path arclength and moves forward with it, never a vacated old spot (v5).
                trail_d = max(0.0, float(human_d) - float(leash_pull_to))
                trail_pt = np.asarray(_interp_path(full_path, cum_dist, trail_d),
                                      dtype=np.float64)[:2]
                # If the robot drifted off-path, route back around any furniture rect between
                # it and the trailing point; wall-clamp (walls + baked furniture) as a backstop.
                routed = _route_around_rects(robot_xy, trail_pt, scene_rects)
                target_xy = (np.asarray(routed[0], dtype=np.float64)[:2] if routed else trail_pt)
                target_xy = _clamp_target_to_wall_clear(robot_xy, target_xy, trav_map,
                                                        clearance=0.25)
            elif d_robot < leash_near:
                # TOO CLOSE, but the human isn't aimed at us (e.g. we drifted in while
                # turning, or the human is passing by). Same lane-clearing yield.
                target_xy = _pick_yield_target(robot_xy, hr, _hdir, trav_map, scene_rects,
                                               human_xy=human_xy2,
                                               min_gap=hard_collision_r + 0.10)
                if target_xy is None:
                    target_xy = _pick_yield_escape(robot_xy, human_xy2, trav_map,
                                                   scene_rects, hard_collision_r + 0.10)
                if target_xy is None:
                    target_xy = _clamp_target_to_wall_clear(
                        robot_xy, robot_xy + hr * 0.30, trav_map, clearance=0.25)
            else:
                target_xy = robot_xy   # in band: hold, just rotate to face human

            # --- 3. ONE reactive step (Habitat), ALWAYS facing the human ---
            obs, action = _online_nav_driver.follow_step(target_xy, face_xy=human_pos[:2])
            _restore_pinned_objects(env)
            info = {"reward": 0.0, "terminated": False, "truncated": False}
            recorder.record("Relocate", obs, info, round_id=round_id, action=action)

            # --- 4. Done once the human reached the path end AND the robot is back
            # at it in the Habitat _reach_human sense (d <= SAFE_DIS_MAX AND facing;
            # upper-bound only, so a slightly-too-close end still terminates).
            robot_xy2 = to_numpy(robot.get_position_orientation()[0]).astype(np.float64)[:2]
            gap = float(np.linalg.norm(robot_xy2 - human_xy2))
            if (human_d >= total_len - goal_eps and gap <= HABITAT_SAFE_DIS_MAX
                    and _robot_human_dot(robot, human_xy2) > HABITAT_FACE_DOT_THRESH):
                break

            # --- 5. Deadlock safety nets (see init above). Measure the robot's
            # per-frame translation once, feed both detectors.
            _robot_moved = float(np.linalg.norm(robot_xy2 - _prev_robot_xy))
            # (both-frozen) human not advancing AND robot not translating -> wedged.
            if _robot_moved < 0.003 and abs(human_d - _prev_human_d) < 0.003:
                _stuck_frames += 1
            else:
                _stuck_frames = 0
            # (chase-stuck / C-fix) robot is BEHIND its follow band (gap>follow_far,
            # should be closing) yet isn't translating -> single-actor jam while the
            # human keeps leading. An in-band hold has gap<=follow_far, so it never
            # trips this. Catches the wedge BEFORE the human stops (the 14.7s blind
            # spot of both-frozen).
            if gap > follow_far + 0.05 and _robot_moved < 0.003:
                _chase_stuck_frames += 1
            else:
                _chase_stuck_frames = 0
            _prev_robot_xy = robot_xy2
            _prev_human_d = human_d
            if _stuck_frames >= _STUCK_LIMIT or _chase_stuck_frames >= _CHASE_STUCK_LIMIT:
                _why = ("robot+human both frozen" if _stuck_frames >= _STUCK_LIMIT
                        else "robot stuck chasing, human still leading")
                _dur = (_stuck_frames if _stuck_frames >= _STUCK_LIMIT
                        else _chase_stuck_frames) / float(max(fps, 1))
                print(f"[relocation] WARN: leash DEADLOCK ({_why} {_dur:.1f}s at "
                      f"human_d={human_d:.2f}/{total_len:.2f}m gap={gap:.2f}m) "
                      f"-> ending follow early", flush=True)
                break

        # ---- End-of-follow: drive the robot to a FURNITURE-CLEAR standoff
        # around the human's end (not the blind "1.5m straight behind" geometric
        # point, which can land on a chair -- that's the 2:06 chair near-miss
        # we saw) and PHYSICALLY ROTATE TO FACE THE HUMAN, so when the next
        # round's WaitNearHuman / GoToTask starts the robot is stopped, clear of
        # furniture, and already looking at the human.
        # Use the human's ACTUAL stop position, not full_path[-1] (= dest): a
        # deadlock-net break ends the follow MID-PATH, so full_path[-1] is where the
        # human was HEADED, not where it IS. Settling the robot around the dest would
        # strand it far from the human (58e: robot->(-5.41,3.53), human at
        # (-2.20,1.92)) and record a bogus "robot leaves human toward empty dest" leg.
        human_end_xy = np.asarray(human_actor.base_pos, dtype=np.float32)[:2]
        robot_xy_end = to_numpy(robot.get_position_orientation()[0]).astype(np.float32)[:2]
        approach_dir = robot_xy_end - human_end_xy
        an = float(np.linalg.norm(approach_dir))
        approach_dir = approach_dir / an if an > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)

        # If the follow already ended furniture-clear and inside the Habitat
        # band, STAY THERE: asking _navigate_to_pose_direct to reach a
        # geometric point centimetres away made it crawl for ~200 recorded
        # dead frames in step7 (goal was 3cm from the robot).
        end_clear = not any(_point_in_rect(robot_xy_end, r) for r in scene_rects)
        if end_clear and HABITAT_SAFE_DIS_MIN <= an <= HABITAT_SAFE_DIS_MAX:
            clear_xy = robot_xy_end
        else:
            clear_xy = _find_clear_standoff(human_end_xy, approach_dir, float(stand_off),
                                            reloc_avoid, min_clearance=_ROBOT_STANDOFF_CLEAR,
                                            avoid_rects=scene_rects, trav_map=trav_map)

        def _end_on_frame(obs: Any, action: np.ndarray) -> None:
            _restore_pinned_objects(env)
            info = {"reward": 0.0, "terminated": False, "truncated": False}
            recorder.record("Relocate", obs, info, round_id=round_id, action=action)

        if float(np.linalg.norm(clear_xy - robot_xy_end)) > 0.3:
            face_yaw = math.atan2(float(human_end_xy[1] - clear_xy[1]),
                                  float(human_end_xy[0] - clear_xy[0]))
            _online_nav_driver.drive_to_pose(
                (float(clear_xy[0]), float(clear_xy[1]), float(face_yaw)),
                on_frame=_end_on_frame, low_precision=True)

        # Face the human from wherever the robot actually stopped; skip when
        # already aligned (follow_step keeps facing the human while trailing).
        robot_xy_now = to_numpy(robot.get_position_orientation()[0]).astype(np.float64)[:2]
        cur_yaw = _yaw_from_quat(to_numpy(robot.get_position_orientation()[1]))
        face_yaw = math.atan2(float(human_end_xy[1] - robot_xy_now[1]),
                              float(human_end_xy[0] - robot_xy_now[0]))
        if abs(_wrap_to_pi(face_yaw - cur_yaw)) > math.radians(8.0):
            _online_nav_driver.rotate_in_place(face_yaw, on_frame=_end_on_frame,
                                               angle_threshold=math.radians(5.0))
        print(f"[relocation] robot settled at clear standoff "
              f"({float(clear_xy[0]):.2f},{float(clear_xy[1]):.2f}) facing human "
              f"(yaw={math.degrees(face_yaw):.0f}°)", flush=True)

        # Smoothly turn the human to face the robot for the next round's
        # WaitNearHuman (instead of snapping its heading there).
        _human_turn_to_face_robot(env, recorder, human_actor, round_id=round_id)
        human_actor.anchor_pos = human_actor.base_pos.copy()
        human_actor.motion_state = "hold"
        human_actor.linear_speed = 0.0
        print(f"[relocation] done (physics) — human now at "
              f"({human_actor.base_pos[0]:.2f}, {human_actor.base_pos[1]:.2f})", flush=True)
        return

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
# Per-round success predicates (P5 fix)
# --------------------------------------------------------------------------- #

def _eval_round_success(
    env: Any,
    spawned: Optional[Dict[str, Any]],
    pred_cfg: Optional[Dict[str, Any]],
) -> Tuple[Optional[bool], str]:
    """Evaluate one round's END-OF-ROUND success predicate on the live sim.

    The env's own reward/done comes from the base task's FULL BDDL, which is
    misaligned with HRC round semantics (radio isn't in task-0026's BDDL;
    "basket on coffee table" isn't its goal predicate). So each round declares
    an explicit predicate in the YAML (``success_predicate``), evaluated once
    when the round's task segment finishes. Doubles as the rollout evaluator
    and the scale-up QA gate.

    Supported configs:
      {type: toggled_on, object: NAME}
      {type: on_top, object: NAME, surface: NAME}
      {type: on_top, objects: [NAME, ...], surface: NAME}   # success if ANY

    Returns (verdict, detail): verdict None = could not evaluate (missing
    config/object/state) — recorded as -1, never silently coerced to False.
    """
    if not pred_cfg:
        return None, "no success_predicate configured"

    def _find(name: str) -> Any:
        obj = (spawned or {}).get(name)
        if obj is None:
            obj = env.scene.object_registry("name", name)
        return obj

    ptype = str(pred_cfg.get("type", "")).strip().lower()
    try:
        if ptype == "toggled_on":
            from omnigibson.object_states import ToggledOn  # type: ignore
            target = str(pred_cfg["object"])
            obj = _find(target)
            if obj is None:
                return None, f"object {target!r} not found in scene"
            if ToggledOn not in obj.states:
                return None, f"{target!r} has no ToggledOn state"
            val = bool(obj.states[ToggledOn].get_value())
            return val, f"ToggledOn({target})={val}"

        if ptype == "on_top":
            from omnigibson.object_states import OnTop  # type: ignore
            surface_name = str(pred_cfg["surface"])
            surface = _find(surface_name)
            if surface is None:
                return None, f"surface {surface_name!r} not found in scene"
            names = pred_cfg.get("objects") or [pred_cfg.get("object")]
            names = [str(n) for n in names if n]
            if not names:
                return None, "on_top predicate has no object(s)"
            hits: List[str] = []
            missing: List[str] = []
            for nm in names:
                obj = _find(nm)
                if obj is None:
                    missing.append(nm)
                    continue
                try:
                    if bool(obj.states[OnTop].get_value(surface)):
                        hits.append(nm)
                except Exception as exc:
                    print(f"[round-success] WARN: OnTop({nm}, {surface_name}) "
                          f"raised: {exc}", flush=True)
            if missing and not hits and len(missing) == len(names):
                return None, f"none of {names} found in scene"
            ok = len(hits) > 0
            return ok, (f"OnTop(any of {names}, {surface_name}) -> hits={hits}"
                        + (f", missing={missing}" if missing else ""))

        if ptype == "inside":
            from omnigibson.object_states import Inside  # type: ignore
            container_name = str(pred_cfg["container"])
            container = _find(container_name)
            if container is None:
                return None, f"container {container_name!r} not found in scene"
            names = pred_cfg.get("objects") or [pred_cfg.get("object")]
            names = [str(n) for n in names if n]
            if not names:
                return None, "inside predicate has no object(s)"
            hits: List[str] = []
            missing: List[str] = []
            for nm in names:
                obj = _find(nm)
                if obj is None:
                    missing.append(nm)
                    continue
                try:
                    if bool(obj.states[Inside].get_value(container)):
                        hits.append(nm)
                except Exception as exc:
                    print(f"[round-success] WARN: Inside({nm}, {container_name}) "
                          f"raised: {exc}", flush=True)
            if missing and not hits and len(missing) == len(names):
                return None, f"none of {names} found in scene"
            ok = len(hits) > 0
            return ok, (f"Inside(any of {names}, {container_name}) -> hits={hits}"
                        + (f", missing={missing}" if missing else ""))

        return None, f"unknown predicate type {ptype!r}"
    except Exception as exc:
        return None, f"predicate evaluation failed: {exc}"


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
    global _online_nav_driver, _policy_client, _policy_previous_action, _policy_task_instruction
    global _policy_rollout_metrics
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
    parser.add_argument(
        "--online-nav", action="store_true",
        help="Drive navigation phases (GoToTask/ReturnToHuman/...) with TRUE "
             "physics (OnlineNavDriver on a contacts-ON env) instead of "
             "kinematic set_position_orientation teleport. This records real "
             "base-velocity actions into the dataset (no more action==0 nav "
             "frames). Op-replay segments still use og.sim.load_state.",
    )
    parser.add_argument(
        "--policy-rollout", action="store_true",
        help="Opt in to full π0.5 robot control for GoToTask, ExecuteBehaviorTask, "
             "ReturnToHuman, FaceHuman, and Relocate. Implies contacts-on physics; "
             "waiting phases remain safe hold. The default collection path is unchanged.",
    )
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=18765)
    parser.add_argument(
        "--policy-timeout", type=float, default=300.0,
        help="TCP timeout in seconds; 300 allows the first JAX inference to compile.",
    )
    parser.add_argument("--rollout-seed", type=int, default=0,
                        help="Environment and π0.5 sampling seed in policy-rollout mode.")
    parser.add_argument(
        "--policy-execute-steps", type=str, default=None,
        help="π0.5 ExecuteBehaviorTask budget. One int for every round, or one int "
             "per round separated by commas (e.g. '3000,2400' = pickup,radio); the "
             "last value covers any further round. Defaults to YAML "
             "policy_execute_steps or 600. Use a small value only for Gate 2 smoke. "
             "The expert needs ~2505 (pickup) / ~1956 (radio) steps, so a budget "
             "below that caps the achievable success rate regardless of the policy.",
    )
    parser.add_argument(
        "--policy-subinstruction-schedule", type=str, default=None,
        help="frozen sub-instruction schedule JSON. Required for models trained with "
             "per-segment Execute prompts (group B): replaying the round's long "
             "instruction would feed a prompt that never appears on an Execute frame "
             "in their training data, making the result uninterpretable.",
    )
    parser.add_argument(
        "--policy-execute-max-seconds", type=float, default=None,
        help="Wall-clock cap per ExecuteBehaviorTask phase; the phase ends early when "
             "exceeded, so a stuck policy cannot stall a run for hours. Defaults to "
             "YAML policy_execute_max_seconds or no cap.",
    )
    args = parser.parse_args()
    if args.policy_rollout:
        args.online_nav = True
        np.random.seed(args.rollout_seed)
        th.manual_seed(args.rollout_seed)

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
    policy_execute_steps = _parse_execute_budget(
        _resolve(args.policy_execute_steps, cfg, "policy_execute_steps", 600))
    policy_execute_max_seconds = _resolve(
        args.policy_execute_max_seconds, cfg, "policy_execute_max_seconds", None)
    policy_execute_max_seconds = (
        None if policy_execute_max_seconds is None else float(policy_execute_max_seconds))
    extra_objects: List[Dict[str, Any]] = list(cfg.get("extra_objects", []))
    hidden_categories: List[str] = [
        str(c).strip().lower() for c in cfg.get("hidden_categories", []) if str(c).strip()
    ]
    hidden_object_names: List[str] = [
        str(n).strip() for n in cfg.get("hidden_object_names", []) if str(n).strip()
    ]
    # Categories that are the base task's MANIPULATION TARGETS (e.g. the soda
    # cans + trash can for picking_up_trash). These are excluded from path
    # avoidance — the robot must approach them, not detour around them. Match is
    # lowercase substring, so list aliases generously to survive category drift.
    avoid_exclude_categories: List[str] = [
        str(c).strip().lower() for c in cfg.get("avoid_exclude_categories", []) if str(c).strip()
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
    if args.online_nav:
        # Physics-online navigation: contacts ON @ 30/120 Hz so the robot stands
        # on the floor and its base can be velocity-driven by OnlineNavDriver.
        # Op-replay segments still work via og.sim.load_state (verified in
        # test_online_driver.py: ~7.45 mm/step drift).
        print("[multitask] --online-nav: building contacts-ON DataPlaybackWrapper "
              "(physics navigation enabled)", flush=True)
        env = build_contacts_on_playback_wrapper(
            raw_hdf5,
            output_hdf5_path=playback_cache,
            resolution=resolution,
            robot_modalities=["rgb"],
        )
    else:
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
        #
        # In --online-nav mode the OnlineNavDriver drives the base with velocity
        # actions, so control must stay ENABLED. Op-replay segments still force
        # the full state via og.sim.load_state every frame, so determinism is
        # preserved without the global EFFORT switch (verified in the smoke test).
        if not args.online_nav:
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
        human_actor = KinematicHumanoidActor(
            env=env,
            usd_path=human_usd,
            name="human_npc",
            distance=human_distance,
            collision_proxy=None,
            phase_motion=_build_phase_motion(),
            step_dt=1.0 / max(fps, 1),
            nav_erosion_radius=_HUMAN_EROSION_M,  # unified human erosion (all phases)
        )
        # Collision proxy ENABLED (re-enabled for scale-up, user decision): 3
        # kinematic cylinders + a head sphere track the human every frame
        # (_sync_proxies) as a physical body the dynamic robot cannot pass through —
        # a SAFETY NET *under* the geometric avoidance (trav_map A* erosion +
        # _deflect_path_around_point), not a replacement. In nominal runs the leash
        # keeps the robot 1.0-1.5m off, well outside the ~0.2m proxy radius, so it
        # rarely contacts; it only bites if geometric avoidance ever fails.
        # The old "PrimitiveObject asserts while playing" note was a MISDIAGNOSIS:
        # object_base._post_load only allows kinematic_only=True when fixed_base=True
        # (else "Kinematic only decision does not match!"), so the proxy is now
        # created with fixed_base=True (hrc_humanoid._create_proxy_object) — no
        # stop/play needed.
        human_actor.collision_proxy.enabled = True

        # ⑦ Anchor the robot at the FIRST round's task-execution pose BEFORE the
        # human spawns (the human spawns relative to the robot). This makes the
        # episode open at the task area — human next to robot next to the task —
        # so round-0 GoToTask is minimal, instead of the robot trekking from an
        # arbitrary reset pose to the task start. Replay rounds only; else no-op.
        _first_task_pose = _peek_first_task_robot_pose(env, episode_id, task_rounds, raw_hdf5)
        if _first_task_pose is not None:
            env.robots[0].set_position_orientation(_first_task_pose[0], _first_task_pose[1])
            env.robots[0].keep_still()
            print(f"[multitask] ⑦ Robot spawn anchored at first-task pose "
                  f"{_first_task_pose[0][:2].tolist()}", flush=True)

        human_actor.spawn_near_robot()
        human = human_actor
        print(f"[multitask] Human spawned at {human_actor.base_pos.tolist()}", flush=True)

        # DEBUG: dump the raw trav_map (the array A* navigates on) + actor poses.
        try:
            _robot_xy = to_numpy(env.robots[0].get_position_orientation()[0])[:2]
            _human_xy = to_numpy(human_actor.base_pos)[:2]
            _dump_trav_map_debug(
                human_actor, output_dir, floor=0, tag="raw",
                points=[("human", (float(_human_xy[0]), float(_human_xy[1])), "magenta"),
                        ("robot", (float(_robot_xy[0]), float(_robot_xy[1])), "lime")])
        except Exception as _exc:
            print(f"[trav-map] raw dump skipped: {_exc}", flush=True)

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
            "policy_rollout": bool(args.policy_rollout),
            "rollout_seed": int(args.rollout_seed) if args.policy_rollout else None,
            "policy_server": (
                {"host": args.policy_host, "port": int(args.policy_port),
                 "execute_horizon": 8, "action_horizon": 32,
                 "execute_steps_per_round": list(policy_execute_steps),
                 "execute_max_seconds": policy_execute_max_seconds}
                if args.policy_rollout else None
            ),
            "head_camera_intrinsics": head_cam_intrinsics,
            "head_camera_static_pitch_deg": head_cam_pitch,
            "head_camera_pitch_axis": head_cam_axis,
            "head_camera_phase_pitch": (
                {str(k): float(v) for k, v in (phase_pitch_cfg or {}).items()}
                if isinstance(phase_pitch_cfg, dict) else None
            ),
            "head_camera_pitch_smooth_deg_per_frame": smooth_step,
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
        #  - Small ground objects (boxes, props) not in the trav_map
        #  - Chairs / sofas (trav_map erosion often insufficient for body width)
        # EXCLUDING the base task's manipulation targets (avoid_exclude_categories,
        # e.g. soda cans + the trash can): the robot must APPROACH those, so they
        # must NOT be treated as obstacles (else it detours around its own task).
        avoid_objects: List[Any] = []
        # Floor-standing big appliances / cabinets: the robot's reactive follow +
        # standoff selection don't read trav_map, so these must be EXPLICIT AABB
        # rects (else it knocks the fridge over — v4). Substring-matched; wall-
        # mounted units (top_cabinet z~1.9) are filtered by the AABB floor gate.
        _APPLIANCE_CATS = ("fridge", "refrigerator", "oven", "stove", "range",
                           "dishwasher", "washer", "dryer", "bottom_cabinet",
                           "cabinet", "counter", "countertop",
                           # Tall FLOOR-STANDING decor whose ORIGIN sits high at its
                           # center (~0.9m): branch (4)'s p[2]>0.35 gate wrongly drops
                           # these, so a standing mirror was invisible to BOTH nav
                           # models -> human A* routed through it, robot bumped it and
                           # the leash deadlocked (ep0058). Route them through the AABB
                           # *floor* gate below, which keeps floor-standing units
                           # (lo[2]<0.35) and drops wall-hung mirrors.
                           "standing_mirror", "mirror")
        _skip_cats = {"robot", "agent", "floor", "wall", "ceiling",
                      "door", "window", "shelf", "table",
                      "bed", "cabinet", "counter", "stove", "oven",
                      "fridge", "refrigerator", "dishwasher", "toilet",
                      "sink", "bathtub", "shower", "mirror", "picture",
                      "carpet", "rug", "curtain", "light", "lamp",
                      # wall/ceiling-mounted fixtures — never floor obstacles
                      # (a wall socket at z=0.18 was wrongly inflated to a 1.1m rect)
                      "wall_socket", "socket", "outlet", "switch", "light_switch",
                      "vent", "thermostat", "clock", "fire_alarm", "smoke_detector"}
        n_excluded = 0
        for obj in env.scene.objects:
            if obj == env.robots[0]:
                continue
            cat = (getattr(obj, "category", "") or "").lower()
            # (1) Task manipulation targets FIRST (substring): the robot must
            # approach these, so they're never obstacles — even if a future task
            # target is a cabinet/fridge (then in avoid_exclude_categories).
            if avoid_exclude_categories and any(ex in cat for ex in avoid_exclude_categories):
                n_excluded += 1
                # Track these task targets (cans + the trash receptacle) per frame so
                # QC can catch a knocked-over receptacle / replay-state pollution.
                try:
                    _onm = getattr(obj, "name", None)
                    if _onm:
                        recorder.track_objects[str(_onm)] = obj
                except Exception:
                    pass
                print(f"  [avoid-exclude] {getattr(obj, 'name', '?')} cat={cat} "
                      f"(task target — not avoided)", flush=True)
                continue
            # (2) Floor-standing appliances/cabinets -> explicit avoid (AABB rect).
            if any(a in cat for a in _APPLIANCE_CATS):
                try:
                    lo, hi = obj.aabb
                    lo = to_numpy(lo).astype(np.float32)
                    hi = to_numpy(hi).astype(np.float32)
                    footprint = max(float(hi[0] - lo[0]), float(hi[1] - lo[1]))
                    height = float(hi[2] - lo[2])
                    # On the floor, or a base counter whose top is the worktop.
                    low_or_floor = float(lo[2]) < 0.35 or ("counter" in cat and float(lo[2]) < 1.10)
                    if low_or_floor and footprint >= 0.25 and height >= 0.20:
                        avoid_objects.append(obj)
                        print(f"  [avoid-obj] {getattr(obj, 'name', '?')} cat={cat} "
                              f"pos=({float((lo[0]+hi[0])/2):.2f},{float((lo[1]+hi[1])/2):.2f}) "
                              f"appliance z[{float(lo[2]):.2f},{float(hi[2]):.2f}] fp={footprint:.2f}", flush=True)
                    else:
                        print(f"  [avoid-skip] {getattr(obj, 'name', '?')} cat={cat} "
                              f"appliance gated (z_lo={float(lo[2]):.2f} fp={footprint:.2f} "
                              f"h={height:.2f})", flush=True)
                except Exception:
                    pass
                continue
            # (3) Other always-skip categories (walls/fixtures/sink/toilet/tables…).
            if cat in _skip_cats:
                continue
            # (4) Chairs/sofas/low props (unchanged).
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
                    # Min-footprint gate: a tiny footprint = fixture/detail, not a
                    # nav obstacle. Without it, 0.55m inflation turns a 1cm prop
                    # (e.g. a wall socket) into a 1.1m blocking rect.
                    if aabb is not None and max(float(aabb[0]), float(aabb[1])) < 0.10:
                        continue
                avoid_objects.append(obj)
                print(f"  [avoid-obj] {getattr(obj, 'name', '?')} cat={cat} "
                      f"pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})", flush=True)
            except Exception:
                pass
        print(f"[multitask] Collected {len(avoid_objects)} avoid-objects "
              f"(excluded {n_excluded} task-target objects via avoid_exclude_categories)", flush=True)
        # Freeze the QC object-tracking order before any recording starts (all task
        # targets exist from scene load; the record() loop reads this fixed list).
        recorder._track_names = sorted(recorder.track_objects.keys())
        if recorder._track_names:
            print(f"[multitask] QC tracking {len(recorder._track_names)} task objects: "
                  f"{recorder._track_names}", flush=True)

        # ---- Unify the HUMAN's obstacle model with the ROBOT's.
        # The base trav_map is floor-level and OMITS table/cabinet BODIES, so the
        # human (which navigates purely on trav_map) used to spawn / stand / walk
        # INSIDE furniture the robot avoids via AABB rects (observed: human in the
        # coffee table -> proxy knocked the radio; human spawned in a bottom
        # cabinet -> leg clipped through it). Bake the robot's SAME furniture set
        # (avoid_objects + the scene's low tables, via _reloc_avoid_list) into the
        # human trav_map, then re-snap the human if it spawned inside one. RAW AABB
        # (tiny 0.05m margin); the human's own ~0.35m path erosion adds the rest.
        try:
            _human_bake_rects = _build_scene_rects(
                _reloc_avoid_list(env, avoid_objects), inflate=0.05, tag="human-trav-bake")
            human_actor.bake_rects_into_trav_map(_human_bake_rects, floor=0)
            # Also give the per-frame robot-dodge the same furniture set, so it
            # never nudges the human into a table/cabinet (dodge ignores trav_map).
            human_actor._furniture_rects = _human_bake_rects
            human_actor.respawn_if_in_obstacle(floor=0)
        except Exception as _exc:
            print(f"[trav-bake] human obstacle-map unification skipped: {_exc}", flush=True)

        # Boot-time ENDPOINT PRE-CHECK (HRC_ENDPOINT_PRECHECK="x,y;x,y;..."): for each
        # candidate relocation endpoint, test the 0.55m robot-body-width A* from a set
        # of representative human starts (Round-0-end positions seen across trash
        # pairs) and report path length + robot-clear. Lets us pick OPEN endpoints in
        # ONE boot instead of a 40-min run each. Exits after printing (no episode).
        _precheck = os.environ.get("HRC_ENDPOINT_PRECHECK", "").strip()
        if _precheck:
            _cands = []
            for _p in _precheck.split(";"):
                _p = _p.strip()
                if _p:
                    try:
                        _xx, _yy = (float(v) for v in _p.split(","))
                        _cands.append((_xx, _yy))
                    except Exception:
                        pass
            _starts = [(2.02, 3.48), (5.65, 5.78), (6.41, 3.20),
                       (3.95, 0.45), (7.65, 0.55), (1.17, 2.19)]
            _tm = human_actor._try_get_trav_map()
            _zc = float(to_numpy(human_actor.base_pos)[2])
            _prects = _build_scene_rects(_reloc_avoid_list(env, avoid_objects), tag="precheck")
            print(f"[precheck] {len(_cands)} endpoint(s) x {len(_starts)} start(s), "
                  f"0.55m robot-body-width (len m + clear ✓/✗):", flush=True)
            for _ex, _ey in _cands:
                _cells, _nclear = [], 0
                for _sx, _sy in _starts:
                    try:
                        _pa = human_actor._nav_path_between(
                            np.array([_sx, _sy], dtype=np.float32),
                            np.array([_ex, _ey], dtype=np.float32),
                            _zc, _tm, 0, erosion_radius=_RELOC_LEAD_EROSION_M)
                        if _pa and len(_pa) >= 2:
                            _ln = sum(float(np.linalg.norm(np.asarray(_pa[i + 1])[:2] - np.asarray(_pa[i])[:2]))
                                      for i in range(len(_pa) - 1))
                            _dir = float(np.linalg.norm(np.array([_ex - _sx, _ey - _sy])))
                            _ok = _leg_robot_clear(
                                np.array([_sx, _sy], dtype=np.float32),
                                np.array([_ex, _ey], dtype=np.float32),
                                _zc, _tm, _prects, human_actor, clearance=0.3, erosion=_RELOC_LEAD_EROSION_M)
                            _good = bool(_ok and _ln <= _dir * 1.4)
                            _nclear += int(_good)
                            _cells.append(f"{_ln:.1f}{'✓' if _good else '✗'}")
                        else:
                            _cells.append("--")
                    except Exception:
                        _cells.append("er")
                print(f"  ({_ex:5.1f},{_ey:5.1f}) clear {_nclear}/{len(_starts)}: {_cells}", flush=True)
            print("[precheck] done — exiting (no episode run)", flush=True)
            import sys as _sys
            _sys.stdout.flush()
            os._exit(0)

        # DEBUG: dump trav_map + our extra avoid_objects rects (the full 2-D map
        # the robot actually navigates against). Tables/beds won't show as orange
        # rects here (they're in _skip_cats) — they're already black in the base
        # trav_map; orange = chairs/sofas/baskets/etc. we add at runtime.
        try:
            _debug_rects = _build_scene_rects(avoid_objects, tag="debug-scene-map")
            _robot_xy = to_numpy(env.robots[0].get_position_orientation()[0])[:2]
            _human_xy = to_numpy(human_actor.base_pos)[:2]
            _dump_trav_map_debug(
                human_actor, output_dir, floor=0, tag="with_avoid_rects",
                rects=_debug_rects,
                points=[("human", (float(_human_xy[0]), float(_human_xy[1])), "magenta"),
                        ("robot", (float(_robot_xy[0]), float(_robot_xy[1])), "lime")])
        except Exception as _exc:
            print(f"[trav-map] with_avoid_rects dump skipped: {_exc}", flush=True)

        # Task-prop pin bucket (_pinned_scene_objects): would pin task-specific
        # props (e.g. baskets) that an EXTERNAL round's recording doesn't include.
        # DISABLED for now — the old rule was hardcoded to task-0026 props
        # (basket/wicker/bow), and the correct task-agnostic version must pin by
        # each round's ACTUAL start pose, not the step-0 pose (plan TODO). Left
        # empty; the furniture pin below is a SEPARATE bucket and always on.
        _pinned_scene_objects.clear()
        print(f"[multitask] task-prop pin: {len(_pinned_scene_objects)} (disabled — see TODO)", flush=True)

        # --online-nav: physics navigation lets the robot collide with furniture.
        # Without pinning, the robot pushes chairs/tables away during nav, then the
        # next op-replay frame snaps them back via load_state -> visible "knock then
        # restore" discontinuity. Pin all furniture at their (recorded) poses so the
        # physics-driven robot never displaces them; op-replay's load_state then sets
        # them to the SAME pose, keeping the world continuous across all phases.
        # These poses were captured right after the step-0 load_state above, so they
        # equal the recorded poses the replay will use.
        if args.online_nav:
            # Furniture goes in its OWN bucket (_pinned_furniture) so it SURVIVES
            # the per-round task-prop _pinned_scene_objects.clear() and stays
            # pinned for later rounds' physics-nav (Relocate / GoToTask).
            _pinned_furniture.clear()
            _furniture_substrings = ("chair", "table", "sofa", "couch", "stool",
                                     "bench", "armchair", "ottoman")
            n_furniture = 0
            for obj in env.scene.objects:
                if obj == env.robots[0]:
                    continue
                cat = (getattr(obj, "category", "") or "").lower()
                if obj.name in _pinned_scene_objects:
                    continue
                if any(sub in cat for sub in _furniture_substrings):
                    pos = to_numpy(obj.get_position_orientation()[0]).copy()
                    quat = to_numpy(obj.get_position_orientation()[1]).copy()
                    _pinned_furniture[obj.name] = (pos, quat)
                    n_furniture += 1
                    print(f"  [pinned-furniture] {obj.name} cat={cat} "
                          f"pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})", flush=True)
            print(f"[multitask] --online-nav: pinned {n_furniture} furniture objects "
                  f"(separate bucket; survives per-round task-prop clear)", flush=True)

        # Construct the physics navigation driver once (it initializes the
        # action primitives, which is relatively heavy). Navigation phases will
        # use it via the module-level _online_nav_driver.
        if args.online_nav and not args.policy_rollout:
            _online_nav_driver = OnlineNavDriver(env, env.robots[0])
            print("[multitask] --online-nav: OnlineNavDriver constructed "
                  "(navigation phases will be physics-driven with real actions)", flush=True)
        if args.policy_rollout:
            _online_nav_driver = None
            _policy_previous_action = None
            _policy_rollout_metrics = {
                "phase_frames": {},
                "task_contact_frames": {"pickup": 0, "radio": 0},
                "navigation_arrival": {},
                "collision_frames": 0,
                "collision_bodies": set(),
                "pickup_initial_z": {},
                "pickup_max_z": {},
                "pickup_lifted": {},
                "pickup_drop_detected": {},
                # Diagnostics added for the B-group readout: "contact frames = 0" alone
                # cannot separate "reached and missed" from "never reached".
                "hand_object_min_distance_m": {},
                "hand_object_min_distance_frame": {},
                "first_contact_frame": {"pickup": None, "radio": None},
                "gripper_commands": [],
                "subinstruction_transitions": {},
            }
            _policy_client = Pi05PolicyClient(
                host=args.policy_host,
                port=args.policy_port,
                timeout_s=args.policy_timeout,
            )
            handshake = _policy_client.connect()
            _policy_client.reset(seed=args.rollout_seed)
            recorder.dataset_metadata["policy_server"].update(
                {"config": handshake.get("config"),
                 "checkpoint": handshake.get("checkpoint")}
            )
            print(f"[multitask] --policy-rollout: connected to π0.5 at "
                  f"{args.policy_host}:{args.policy_port}; protocol="
                  f"32/{handshake['action_dim']}/{handshake['execute_horizon']}", flush=True)

        # Reward-segments of the base task — used by replay rounds.
        segments = discover_reward_segments(raw_hdf5, episode_id=episode_id)
        recorder._segments = segments
        print(f"[multitask] Discovered {len(segments)} reward-segments in base task.", flush=True)
        for si, (s, e) in enumerate(segments):
            print(f"  base seg {si}: [{s},{e}) ({e-s} steps)", flush=True)

        # ---- Robot start pose before Round 0.
        # When the robot is already anchored at the FIRST round's task pose
        # (_first_task_pose, set above for replay rounds), KEEP it there: the
        # episode opens AT the task area so round-0 GoToTask is ~no-op — no
        # pointless walk-over. Only rotate to face the human for the
        # WaitNearHuman/ReceiveCommand beat (GoToTask->face_yaw->settle re-aligns
        # orientation before the replay, so facing the human now is cosmetic/safe).
        robot = env.robots[0]
        human_pos = human_actor.base_pos.astype(np.float32)
        robot_init_z = float(to_numpy(robot.get_position_orientation()[0])[2])

        if _first_task_pose is not None:
            robot_init_pos = to_numpy(robot.get_position_orientation()[0]).astype(np.float32)
            face_human_yaw = math.atan2(float(human_pos[1] - robot_init_pos[1]),
                                        float(human_pos[0] - robot_init_pos[0]))
            robot.set_position_orientation(robot_init_pos, quat_from_yaw(face_human_yaw))
            robot.keep_still()
            print(f"[multitask] Robot stays at first-task pose (no walk-over) "
                  f"{robot_init_pos[:2].tolist()}, facing human "
                  f"(yaw={math.degrees(face_human_yaw):.1f}°)", flush=True)
        else:
            # Fallback (non-replay episodes): stand the robot next to the human.
            # Try 12 directions around the human, pick the first navigable (on
            # trav_map) and far enough from all scene furniture.
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

        # Capture the HOME posture (= what the video shows at ~0:02): the one
        # canonical arms+trunk stance every custom phase returns to via
        # _settle_arms_to_home. GoToTask is exempt (settles to the task's
        # first frame instead).
        global _HOME_POSTURE_Q
        _HOME_POSTURE_Q = robot.get_joint_positions().clone()
        print(f"[home-posture] captured at episode start "
              f"({int(_HOME_POSTURE_Q.numel())} dofs)", flush=True)

        # ---- Idle 1.5 seconds (robot faces human, both stay still).
        # Same hold-action path as every other wait phase: recording action=0
        # here would teach the BC policy "command the arms to the zero pose"
        # for the episode's very first 45 frames (the step7 leftover).
        idle_frames = int(1.5 * fps)
        run_zero_phase_pinned(env, recorder, "WaitNearHuman", idle_frames,
                              round_id=0, human_actor=human_actor)
        print(f"[multitask] Idle phase done ({idle_frames} frames, {idle_frames/fps:.1f}s)", flush=True)

        # ---- Run rounds.
        num_rounds = len(task_rounds)
        round_summaries: List[Dict[str, Any]] = []

        for round_id, round_cfg in enumerate(task_rounds):
            kind = round_cfg.get("kind")
            instr = str(round_cfg.get("instruction", ""))
            _policy_task_instruction = instr
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

                # Phase 4: in rollout mode the policy must make physical button
                # contact itself.  The legacy scripted toggle remains untouched
                # when --policy-rollout is off.
                if _policy_client is not None:
                    _sched = (None if args.policy_subinstruction_schedule is None
                              else load_subinstruction_schedule(
                                  args.policy_subinstruction_schedule, round_id))
                    stats["steps"] = run_policy_phase(
                        env, recorder, "ExecuteBehaviorTask",
                        _round_execute_steps(policy_execute_steps, round_id),
                        round_id, human_actor=human_actor,
                        max_seconds=policy_execute_max_seconds, schedule=_sched)
                    if _sched is not None and _policy_rollout_metrics is not None:
                        _policy_rollout_metrics["subinstruction_transitions"][str(round_id)] = _sched.transitions
                else:
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
                    task_start_pos, task_start_quat, q_task_start = peek_robot_pos_from_external_hdf5(
                        ext_path, ext_episode, ext_start, return_quat=True, return_joints=True)
                    env.robots[0].set_position_orientation(saved_pos, saved_quat)
                    print(f"[multitask] GoToTask: {saved_pos[:2].tolist()} -> {task_start_pos[:2].tolist()}", flush=True)
                    run_go_to_task_phase_smooth(env, recorder, human, go_to_task_steps, round_id=round_id,
                                                target_pos=task_start_pos, human_actor=human_actor,
                                                avoid_objects=avoid_objects)
                    replay_yaw = _yaw_from_quat(task_start_quat)
                    run_face_yaw_phase(env, recorder, task_start_pos, replay_yaw,
                                       steps=face_human_steps, round_id=round_id,
                                       human_actor=human_actor)
                    run_settle_to_task_config(env, recorder, q_task_start,
                                              base_pos=task_start_pos, base_quat=task_start_quat,
                                              round_id=round_id, human_actor=human_actor)
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

                # D1/D2: park the human's proxies + steer it away from the task
                # object (the radio) for the whole external replay, so it can't
                # disturb the replayed state.
                if human_actor is not None:
                    human_actor._proxies_parked = True
                    _tc_obj = spawned.get("radio_89")
                    if _tc_obj is not None:
                        human_actor._task_center = to_numpy(
                            _tc_obj.get_position_orientation()[0]).astype(np.float32)[:2]
                    human_actor._sync_proxies()
                if _policy_client is not None:
                    round_steps = _round_execute_steps(policy_execute_steps, round_id)
                    _sched = (None if args.policy_subinstruction_schedule is None
                              else load_subinstruction_schedule(
                                  args.policy_subinstruction_schedule, round_id))
                    executed = 0
                    for policy_step in _policy_budget_steps(
                            round_steps, policy_execute_max_seconds,
                            f"ExecuteBehaviorTask#r{round_id}"):
                        if human_actor is not None:
                            if ext_traj is not None:
                                progress = policy_step / float(max(round_steps - 1, 1))
                                drive_human_from_recording(
                                    human_actor, ext_traj[0], ext_traj[1], progress)
                            else:
                                human_actor.step("ExecuteBehaviorTask", env.robots[0])
                            dodge_robot_if_needed(human_actor, env.robots[0])
                        _, _act, _ = _policy_step(
                            env, recorder, "ExecuteBehaviorTask", round_id,
                            human_actor=human_actor, advance_human=False,
                            task_instruction=None if _sched is None else _sched.instruction)
                        executed += 1
                        if _sched is not None:
                            _sched.update(_act, policy_step)
                            if _sched.finished:
                                print(f"[subinstr] terminal segment reached at frame "
                                      f"{policy_step}; ending Execute", flush=True)
                                break
                    stats["steps"] = executed
                    if _sched is not None and _policy_rollout_metrics is not None:
                        _policy_rollout_metrics["subinstruction_transitions"][str(round_id)] = _sched.transitions
                else:
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
                if human_actor is not None:
                    human_actor._proxies_parked = False
                    human_actor._task_center = None
                    human_actor.apply_pose()

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
                # Capture the manip first-frame joint config (the sim is at that
                # loaded state right now) so we can settle the arms into it before
                # the replay starts -> no arm snap at the handoff.
                q_task_start = to_numpy(env.robots[0].get_joint_positions()).copy()
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
                # Settle arms+trunk AND base onto the replay's exact first frame,
                # then ~0.5s pause, so the replay's first load_state is seamless.
                run_settle_to_task_config(env, recorder, q_task_start,
                                          base_pos=task_start_pos, base_quat=task_start_quat,
                                          round_id=round_id, human_actor=human_actor)

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

                # Stop pinning this task's PROPS — the replay's own load_state
                # manages them. Only clears the task-prop bucket; _pinned_furniture
                # is a separate bucket and stays pinned for later rounds' nav.
                _pinned_scene_objects.clear()

                # skip_init=True ALWAYS: env.scene.restore() rebuilds the scene from the
                # recorded scene_file (which doesn't include the radio) and would drop our
                # extra objects. The base scene was already initialised by env.reset().
                # Patch 3: replay_all wants the FULL segment incl. the post-subgoal
                # "put down" tail. task-0001 sets terminated=True at step 6737, so
                # without this the replay stops at 6738 and the robot freezes
                # holding the can. continue_after_done keeps stepping past it.
                continue_after_done = bool(round_cfg.get("continue_after_done", replay_all))
                print(f"[multitask] replay_segment: replay_all={replay_all} "
                      f"continue_after_done={continue_after_done} seg=[{seg_start},{seg_end}) "
                      f"max_playback_steps={max_playback_steps} "
                      f"stop_on_done={not continue_after_done}", flush=True)
                # D1/D2: for the whole replay, steer the human AWAY from the task centre
                # and PARK its collision proxies, so it can never disturb the task
                # objects being replayed (ep0056: a dodge knocked a soda can, corrupting
                # the load_state replay). task_start_pos ~= the manipulation area.
                if human_actor is not None:
                    human_actor._task_center = np.asarray(task_start_pos, dtype=np.float32)[:2]
                    human_actor._proxies_parked = True
                    human_actor._sync_proxies()
                if _policy_client is not None:
                    round_steps = _round_execute_steps(policy_execute_steps, round_id)
                    _sched = (None if args.policy_subinstruction_schedule is None
                              else load_subinstruction_schedule(
                                  args.policy_subinstruction_schedule, round_id))
                    executed = 0
                    for policy_step in _policy_budget_steps(
                            round_steps, policy_execute_max_seconds,
                            f"ExecuteBehaviorTask#r{round_id}"):
                        if human_actor is not None:
                            if seg_traj is not None:
                                progress = policy_step / float(max(round_steps - 1, 1))
                                drive_human_from_recording(
                                    human_actor, seg_traj[0], seg_traj[1], progress)
                            else:
                                human_actor.step("ExecuteBehaviorTask", env.robots[0])
                            dodge_robot_if_needed(human_actor, env.robots[0])
                        _, _act, _ = _policy_step(
                            env, recorder, "ExecuteBehaviorTask", round_id,
                            human_actor=human_actor, advance_human=False,
                            task_instruction=None if _sched is None else _sched.instruction)
                        executed += 1
                        if _sched is not None:
                            _sched.update(_act, policy_step)
                            if _sched.finished:
                                print(f"[subinstr] terminal segment reached at frame "
                                      f"{policy_step}; ending Execute", flush=True)
                                break
                    stats["steps"] = executed
                    if _sched is not None and _policy_rollout_metrics is not None:
                        _policy_rollout_metrics["subinstruction_transitions"][str(round_id)] = _sched.transitions
                else:
                    playback_episode_with_callback(
                        env, episode_id=episode_id,
                        callback=_make_cb(round_id, stats),
                        max_playback_steps=max_playback_steps,
                        pre_step_callback=_pre_step,
                        start_step=seg_start,
                        end_step=seg_end,
                        skip_init=True,
                        stop_on_done=not continue_after_done,
                        mask_done_when_continuing=continue_after_done,
                    )
                if human_actor is not None:
                    human_actor._proxies_parked = False
                    human_actor._task_center = None
                    human_actor.apply_pose()   # bring proxies back to the human
            else:
                raise ValueError(f"Unknown round kind: {kind!r}")

            _restore_pinned_objects(env)

            # ---- End-of-round success predicate (P5 fix, HANDOFF §12.2).
            verdict, verdict_detail = _eval_round_success(
                env, spawned, round_cfg.get("success_predicate"))
            if verdict is None:
                print(f"[round-success] round {round_id}: UNKNOWN ({verdict_detail})", flush=True)
            else:
                print(f"[round-success] round {round_id}: "
                      f"{'SUCCESS' if verdict else 'FAIL'} ({verdict_detail})", flush=True)
                stats["predicate_success"] = bool(verdict)
                recorder.set_round_success(round_id, verdict, verdict_detail)

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
                    task_center=next_task_center,
                    fps=fps,
                    avoid_objects=avoid_objects,
                    # When the next round sets `relocation_dest`, the human walks
                    # to that fixed OPEN-area waypoint (validated in-phase) and the
                    # annulus-around-task / staircase sampling is bypassed entirely.
                    dest_override=next_cfg.get("relocation_dest"),
                    # toward_staircase only selects the FALLBACK when there is NO
                    # relocation_dest AND task_center is None.
                    toward_staircase=True,
                )

            recorder._round_exec_stats[round_id] = stats
            round_summaries.append({"round_id": round_id, "kind": kind, "instruction": instr, **stats})
            print(f"[multitask] Round {round_id} done: {stats}. Total frames so far: {recorder.frame_idx}", flush=True)

        recorder.close()

        if args.policy_rollout and _policy_rollout_metrics is not None:
            contact_report = dict(_policy_rollout_metrics)
            contact_report["collision_bodies"] = sorted(contact_report["collision_bodies"])
            contact_report["checkpoint"] = recorder.dataset_metadata["policy_server"].get("checkpoint")
            contact_report["config"] = recorder.dataset_metadata["policy_server"].get("config")
            contact_report["seed"] = int(args.rollout_seed)
            contact_report["head_video"] = recorder.head_path
            contact_report["third_person_video"] = (
                recorder.third_path if recorder.record_third else None)
            with open(os.path.join(output_dir, "policy_rollout_metrics.json"),
                      "w", encoding="utf-8") as stream:
                json.dump(contact_report, stream, indent=2, ensure_ascii=False)
            print(f"[multitask] Policy contact metrics: "
                  f"{os.path.join(output_dir, 'policy_rollout_metrics.json')}", flush=True)

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
        if _policy_client is not None:
            _policy_client.close()
            _policy_client = None
        _policy_previous_action = None
        _policy_task_instruction = ""
        _policy_rollout_metrics = None
        og.shutdown()


if __name__ == "__main__":
    main()
