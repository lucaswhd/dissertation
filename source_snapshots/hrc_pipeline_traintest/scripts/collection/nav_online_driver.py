#!/usr/bin/env python
# Path A (TRUE physics-online) navigation driver — reusable module.
#
# Validated in nav_physics_sandbox.py:
#   * A contacts-ON 30/120 Hz og.Environment (BEHAVIOR-1K physics) keeps R1Pro
#     upright while its HolonomicBase is velocity-driven (roll/pitch < 0.5 deg).
#   * BEHAVIOR-1K's StarterSemanticActionPrimitives._navigate_to_pose_direct /
#     _rotate_in_place drive the base; _empty_action(follow_arm_targets=True)
#     holds the arms (drift ~0.001 rad) when _arm_targets is pinned.
#   * Operation-segment load_state replay stays faithful on this env
#     (~7 mm/step, corrected every frame by the next load_state).
#
# This module exposes:
#   - build_contacts_on_env(raw_hdf5, ...)        -> og.Environment (physics ON)
#   - OnlineNavDriver(env, robot)                 -> wraps the primitives
#       .pin_arm_hold()                           snapshot current arm pose as hold target
#       .drive_to_pose((x,y,yaw), on_frame=...)   physics-drive, calling on_frame(obs, action) per step
#       .drive_along_path([(x,y), ...], final_yaw, on_frame=...)
#
# It does NOT depend on curobo (skip_curobo_initilization=True) and reuses the
# pipeline's own A* (caller supplies waypoints) for obstacle avoidance.

import json
import math
from typing import Any, Callable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.envs import DataPlaybackWrapper

from hrc_utils import to_numpy

# Base velocity output limits (R1Pro recorded config; verified by probe).
BASE_OUT_LIMITS = np.array([0.75, 0.75, 1.0], dtype=np.float32)


def yaw_from_quat(quat) -> float:
    x, y, z, w = [float(v) for v in to_numpy(quat)]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def build_contacts_on_env(
    raw_hdf5: str,
    resolution: int = 256,
    robot_modalities: Optional[Sequence[str]] = None,
    action_frequency: float = 30.0,
    physics_frequency: float = 120.0,
):
    """Build a normal physics og.Environment from the recorded config, with
    COLLISIONS ENABLED (no gm.VISUAL_ONLY) at BEHAVIOR-1K's 30/120 Hz cadence.

    Unlike build_playback_env (contacts OFF -> robot can't stand), this env lets
    the robot rest on the floor so its base can be velocity-driven by physics.
    Operation segments still replay via og.sim.load_state on this same env.
    """
    robot_modalities = list(robot_modalities or ["rgb"])
    with h5py.File(raw_hdf5, "r") as f:
        config = json.loads(f["data"].attrs["config"])
        scene_file = json.loads(f["data"].attrs["scene_file"])

    config["env"]["action_frequency"] = action_frequency
    config["env"]["rendering_frequency"] = action_frequency
    config["env"]["physics_frequency"] = physics_frequency
    config["env"]["flatten_action_space"] = True
    config["env"]["flatten_obs_space"] = False
    config["env"]["external_sensors"] = []

    config["scene"]["scene_file"] = scene_file
    config["objects"] = []
    config["task"] = {"type": "DummyTask"}

    w = (int(resolution * 4 / 3) + 15) // 16 * 16
    h = (resolution + 15) // 16 * 16
    for rc in config["robots"]:
        rc["obs_modalities"] = robot_modalities
        rc["sensor_config"] = {
            "VisionSensor": {"sensor_kwargs": {"image_height": h, "image_width": w}}
        }

    gm.ENABLE_TRANSITION_RULES = False
    # IMPORTANT: do NOT set gm.VISUAL_ONLY=True -> collisions stay enabled.
    return og.Environment(configs=config)


def build_contacts_on_playback_wrapper(
    raw_hdf5: str,
    output_hdf5_path: str,
    resolution: int = 256,
    robot_modalities: Optional[Sequence[str]] = None,
    action_frequency: float = 30.0,
    physics_frequency: float = 120.0,
    n_render_iterations: int = 1,
):
    """Like hrc_utils.build_playback_env, but builds a DataPlaybackWrapper with
    CONTACTS ENABLED at 30/120 Hz (BEHAVIOR-1K cadence).

    The stock DataPlaybackWrapper.create_from_hdf5 couples include_contacts=True
    to a forced 1000 Hz cadence (unusable for 30 fps recording) and
    include_contacts=False to gm.VISUAL_ONLY=True (robot can't stand). We need
    contacts ON *and* 30/120, so we replicate the builder here.

    Returns a DataPlaybackWrapper exposing the SAME interface the rest of the
    pipeline relies on (env.input_hdf5, env.scene_file, env.env.step,
    env.include_contacts, ...), so operation-segment replay code is unchanged;
    only navigation segments switch to physics driving.
    """
    robot_modalities = list(robot_modalities or ["rgb"])
    gm.ENABLE_TRANSITION_RULES = False

    with h5py.File(raw_hdf5, "r") as f:
        config = json.loads(f["data"].attrs["config"])
        scene_file = json.loads(f["data"].attrs["scene_file"])

    # Contacts ON + BEHAVIOR-1K 30/120 cadence (NOT the stock 1000 Hz path).
    config["env"]["action_frequency"] = action_frequency
    config["env"]["rendering_frequency"] = action_frequency
    config["env"]["physics_frequency"] = physics_frequency
    config["env"]["flatten_obs_space"] = True
    config["env"]["external_sensors"] = []

    config["scene"]["scene_file"] = scene_file
    config["objects"] = []
    config["task"]["include_obs"] = False
    # The recorded BehaviorTask has use_presampled_robot_pose=True, but the
    # scene_file carries no "robot_poses" task metadata, so BehaviorTask.reset()
    # would hit `robot.model_name in None` -> TypeError. We replay recorded
    # robot poses anyway, so presampled poses are unnecessary here.
    config["task"]["use_presampled_robot_pose"] = False

    width = (int(resolution * 4 / 3) + 15) // 16 * 16
    height = (resolution + 15) // 16 * 16
    for rc in config["robots"]:
        rc["obs_modalities"] = robot_modalities
        rc["include_sensor_names"] = None
        rc["exclude_sensor_names"] = None
        rc["sensor_config"] = {
            "VisionSensor": {"sensor_kwargs": {"image_height": height, "image_width": width}}
        }

    # Do NOT set gm.VISUAL_ONLY -> collisions stay on so the robot stands.
    env = og.Environment(configs=config)

    return DataPlaybackWrapper(
        env=env,
        input_path=raw_hdf5,
        output_path=output_hdf5_path,
        n_render_iterations=n_render_iterations,
        overwrite=True,
        only_successes=False,
        include_robot_control=True,   # control active for physics navigation
        include_contacts=True,        # collisions on; robot rests on floor
    )


class OnlineNavDriver:
    """Physics-online base navigation by reusing BEHAVIOR-1K's action
    primitives. Drives the HolonomicBase via velocity actions, holds the arms
    at a pinned pose, and reports the true (obs, action) of every frame so the
    caller can record a BEHAVIOR-1K-standard navigation segment."""

    def __init__(self, env, robot, enable_head_tracking: bool = False):
        from omnigibson.action_primitives.starter_semantic_action_primitives import (
            StarterSemanticActionPrimitives,
        )
        self.env = env
        self.robot = robot
        self.prims = StarterSemanticActionPrimitives(
            env, robot,
            enable_head_tracking=enable_head_tracking,
            skip_curobo_initilization=True,
        )
        self.pin_arm_hold()

    def pin_arm_hold(self) -> None:
        """Freeze the arm targets at the CURRENT joint pose so the primitive's
        _empty_action keeps the arms still during navigation (no drift)."""
        jp = self.robot.get_joint_positions()
        for arm in self.robot.arm_names:
            name = f"arm_{arm}"
            if name in self.robot.controllers:
                self.prims._arm_targets[name] = jp[self.robot.controllers[name].dof_idx]

    def _run(self, gen, on_frame: Optional[Callable[[Any, np.ndarray], None]] = None,
             max_base_lin: Optional[float] = None,
             abort_fn: Optional[Callable[[], bool]] = None) -> bool:
        """Iterate a primitive action-generator, stepping the env and reporting
        each (obs, action). Returns True if it completed without raising.

        C (ep0058): when *max_base_lin* is given, the base LINEAR action magnitude
        ([vx,vy]) is clamped to it BEFORE env.step — the BEHAVIOR-1K nav primitive
        otherwise drove GoToTask at ~3.5 m/s with a SATURATED action (vy=-1.0), an
        unnaturally fast motion whose recorded action didn't match its speed. The
        CLAMPED action is what gets recorded, so action and motion stay consistent
        for pi0/BC training. Rotation (rotate_in_place) passes None -> no clamp."""
        from omnigibson.action_primitives.action_primitive_set_base import (
            ActionPrimitiveError,
        )
        base_idx = None
        if max_base_lin is not None:
            try:
                base_idx = [int(x) for x in
                            to_numpy(self.robot.controller_action_idx["base"]).tolist()]
            except Exception:
                base_idx = None
        try:
            for action in gen:
                if base_idx is not None and len(base_idx) >= 2:
                    if hasattr(action, "clone"):
                        action = action.clone()
                    vx = float(action[base_idx[0]])
                    vy = float(action[base_idx[1]])
                    mag = math.hypot(vx, vy)
                    if mag > float(max_base_lin) and mag > 1e-6:
                        s = float(max_base_lin) / mag
                        action[base_idx[0]] = action[base_idx[0]] * s
                        action[base_idx[1]] = action[base_idx[1]] * s
                obs, _rew, _term, _trunc, _info = self.env.step(action)
                if on_frame is not None:
                    on_frame(obs, to_numpy(action).astype(np.float32))
                # CLEARANCE-ABORT: stop driving BEFORE contact (caller-supplied check on the
                # robot's clearance to furniture/walls). A tilt-abort alone is too late — once
                # the base leans ~12° into a chair it cannot pull back out (ep0014 reached 33°).
                if abort_fn is not None and abort_fn():
                    print("[OnlineNavDriver] CLEARANCE-ABORT — stopped before driving into an "
                          "obstacle", flush=True)
                    return False
                # TILT-ABORT: stop the instant the base starts tipping (>12°) so a primitive
                # can't keep pushing the robot into furniture PAST the point of a fall
                # (ep0014 ReturnToHuman drove vx=0.30 straight into a dining chair -> 30°;
                # normal tilt is 1-3°). The caller returns and the next phase's stuck guard
                # (escape_step, vyaw≡0) un-wedges it before it can rotate.
                _q = to_numpy(self.robot.get_position_orientation()[1])
                _qx, _qy, _qz, _qw = float(_q[0]), float(_q[1]), float(_q[2]), float(_q[3])
                _roll = math.degrees(math.atan2(2 * (_qw * _qx + _qy * _qz),
                                                1 - 2 * (_qx * _qx + _qy * _qy)))
                _pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (_qw * _qy - _qz * _qx)))))
                if max(abs(_roll), abs(_pitch)) > 12.0:
                    print(f"[OnlineNavDriver] TILT-ABORT at "
                          f"{max(abs(_roll), abs(_pitch)):.0f}° — stopped driving to avoid a tip",
                          flush=True)
                    return False
            return True
        except ActionPrimitiveError as exc:
            print(f"[OnlineNavDriver] primitive ended: {exc}", flush=True)
            return False

    def drive_to_pose(self, pose_2d: Tuple[float, float, float],
                      on_frame: Optional[Callable[[Any, np.ndarray], None]] = None,
                      low_precision: bool = True,
                      max_base_lin: float = 0.8,
                      abort_fn: Optional[Callable[[], bool]] = None) -> bool:
        """Drive straight to a single (x, y, yaw) via _navigate_to_pose_direct
        (rotate-to-face + constant-velocity drive + rotate-to-final).

        C: max_base_lin caps the base linear action magnitude (~0.7 m/s walking pace
        at the measured gain) so GoToTask no longer glides at ~3.5 m/s (ep0058)."""
        self.pin_arm_hold()
        gen = self.prims._navigate_to_pose_direct(
            th.tensor([float(pose_2d[0]), float(pose_2d[1]), float(pose_2d[2])], dtype=th.float32),
            low_precision=low_precision,
        )
        return self._run(gen, on_frame, max_base_lin=max_base_lin, abort_fn=abort_fn)

    def rotate_in_place(self, target_yaw: float,
                        on_frame: Optional[Callable[[Any, np.ndarray], None]] = None,
                        angle_threshold: Optional[float] = None) -> bool:
        """Physically rotate the base IN PLACE to world-frame *target_yaw* using
        BEHAVIOR-1K's own ``_rotate_in_place`` primitive (base action =
        [0, 0, +/-ang_vel], arms held at the pinned pose via _empty_action).

        Unlike a kinematic set_position_orientation turn, the robot moves
        *because of* the recorded action, so every frame reports a real,
        physically-executable action (no zero-action / teleport frames). Used
        for FaceHuman and the GoToTask final yaw-alignment so that ALL robot
        motion is online, BEHAVIOR-1K-standard control."""
        self.pin_arm_hold()
        cur_xy = to_numpy(self.robot.get_position_orientation()[0]).astype(np.float64)[:2]
        end_pose = self.prims._get_robot_pose_from_2d_pose(
            th.tensor([float(cur_xy[0]), float(cur_xy[1]), float(target_yaw)], dtype=th.float32)
        )
        if angle_threshold is None:
            gen = self.prims._rotate_in_place(end_pose)
        else:
            gen = self.prims._rotate_in_place(end_pose, angle_threshold=float(angle_threshold))
        return self._run(gen, on_frame)

    def rotate_to_yaw_shortest(self, target_yaw: float,
                               on_frame: Optional[Callable[[Any, np.ndarray], None]] = None,
                               k_ang: float = 1.5, tol_deg: float = 4.0,
                               max_frames: int = 150) -> bool:
        """Rotate the base IN PLACE to world-frame *target_yaw*, ALWAYS via the
        SHORTEST direction. BEHAVIOR-1K's ``_rotate_in_place`` / ``_navigate_to_pose_direct``
        primitives sometimes wind the LONG way (ep000 ReturnToHuman: robot spun ~315°
        in place to face a human only 34° to its right). This uses the SAME wrapped
        vyaw as follow_step, so the error is always in [-pi, pi] and it can never
        over-wind. Records a real base-velocity action each frame."""
        self.pin_arm_hold()
        r = self.robot
        bidx = r.controller_action_idx["base"]
        tol = math.radians(tol_deg)
        for _ in range(int(max_frames)):
            ryaw = yaw_from_quat(r.get_position_orientation()[1])
            err = math.atan2(math.sin(target_yaw - ryaw), math.cos(target_yaw - ryaw))
            if abs(err) <= tol:
                break
            vyaw = max(-1.0, min(1.0, k_ang * err))     # shortest-direction turn
            action = self.prims._empty_action()          # arms held
            action[bidx] = th.tensor([0.0, 0.0, float(vyaw)], dtype=th.float32)
            obs, _rew, _term, _trunc, _info = self.env.step(action)
            if on_frame is not None:
                on_frame(obs, to_numpy(action).astype(np.float32))
        return True

    @staticmethod
    def _smoothstep(t: float) -> float:
        t = max(0.0, min(1.0, float(t)))
        return t * t * (3.0 - 2.0 * t)

    def settle_to_config(self, q_target, base_target=None,
                         ramp_frames: int = 40, hold_frames: int = 15,
                         action_dt: float = 1.0 / 30.0,
                         on_frame: Optional[Callable[[Any, np.ndarray], None]] = None) -> bool:
        """Settle the robot onto the op-replay segment's first frame so the
        custom-nav -> BEHAVIOR-1K-task handoff is seamless (no arm snap AND no
        base slide), then HOLD for hold_frames (the ~0.5s pause) at that exact
        pose before the task starts.

        * Arms + trunk ramp from the current pose to *q_target* (the recorded
          first-frame joint config) via the primitive's _arm_targets / absolute
          position commands.
        * If *base_target*=(pos, quat) is given, the base GLIDES to that exact
          recorded pose (the ~residual nav gap, e.g. 9 cm, spread over the ramp so
          it is imperceptible) instead of snapping when the replay's first
          og.sim.load_state runs. The glide is kinematic (set_position_orientation,
          held by a 0-velocity step) but the recorded action carries the matching
          body-frame base velocity, so the label stays consistent with the motion.
        * hold_frames extra steps at the exact target give the 0.5s "arrived,
          pausing" beat and let the arm controllers converge.

        Every frame is exposed via on_frame(obs, action)."""
        r = self.robot
        q_target = th.as_tensor(to_numpy(q_target), dtype=th.float32)
        q_start = r.get_joint_positions().to(th.float32)

        glide = base_target is not None
        if glide:
            bt_pos = np.asarray(to_numpy(base_target[0]), dtype=np.float64)[:3]
            bt_quat = th.as_tensor(to_numpy(base_target[1]), dtype=th.float32)
            b_start = to_numpy(r.get_position_orientation()[0]).astype(np.float64)[:3]
            yaw_t = yaw_from_quat(bt_quat)
            cyaw, syaw = math.cos(yaw_t), math.sin(yaw_t)
            prev_bpos = b_start.copy()
            base_aidx = r.controller_action_idx["base"]

        arm_ctrl_names = [f"arm_{a}" for a in r.arm_names if f"arm_{a}" in r.controllers]
        # Absolute position controllers other than base / arms / grippers (trunk).
        extra_pos_names = [
            nm for nm, ctrl in r.controllers.items()
            if nm != "base" and nm not in arm_ctrl_names and not nm.startswith("gripper")
        ]

        ramp = max(ramp_frames, 1)
        total = ramp + max(hold_frames, 0)
        for k in range(total):
            alpha = self._smoothstep((k + 1) / ramp)
            q_now = q_start + (q_target - q_start) * alpha
            # Arms: ramp the saved targets so _empty_action drives them there.
            for nm in arm_ctrl_names:
                self.prims._arm_targets[nm] = q_now[r.controllers[nm].dof_idx].to(th.float32)
            if glide:
                bpos = b_start + (bt_pos - b_start) * float(alpha)
                r.set_position_orientation(th.tensor(bpos, dtype=th.float32), bt_quat)
            action = self.prims._empty_action()  # arms ramped; base/trunk/grippers no-op
            # Trunk (etc.): override the no-op with the ramped absolute target.
            for nm in extra_pos_names:
                ctrl = r.controllers[nm]
                aidx = r.controller_action_idx[nm]
                tgt = q_now[ctrl.dof_idx].to(th.float32)
                if getattr(ctrl, "use_delta_commands", False):
                    action[aidx] = tgt - r.get_joint_positions()[ctrl.dof_idx].to(th.float32)
                else:
                    action[aidx] = tgt
            # Step with base velocity 0 so physics holds the just-set base pose.
            obs, _rew, _term, _trunc, _info = self.env.step(action)
            rec_action = to_numpy(action).astype(np.float32)
            if glide:
                # Record the body-frame velocity that matches this frame's glide,
                # so the nav label isn't a zero-action causal break.
                dxy = bpos[:2] - prev_bpos[:2]
                prev_bpos = bpos.copy()
                vx = (cyaw * dxy[0] + syaw * dxy[1]) / action_dt
                vy = (-syaw * dxy[0] + cyaw * dxy[1]) / action_dt
                rec_action[base_aidx] = np.clip(
                    np.array([vx, vy, 0.0], dtype=np.float32) / BASE_OUT_LIMITS,
                    -1.0, 1.0).astype(np.float32)
            if on_frame is not None:
                on_frame(obs, rec_action)
        return True

    def follow_step(self, target_xy, face_xy=None,
                    stop_dist: float = 0.20, max_speed: float = 0.55,
                    k_ang: float = 1.5, k_fwd: float = 1.5):
        """ONE reactive follow step toward *target_xy*, Habitat OracleNavAction
        style ("align-then-move"): turn toward the target, and once roughly
        aligned drive forward; when already at the target, rotate in place to
        face *face_xy* (the human). Keeps the robot facing where it is going /
        the human, so the human stays in its first-person view. The holonomic
        base is driven with a REAL, limit-capped velocity action ([-1,1] ->
        +/-0.75/0.75/1.0); arms are held. Returns (obs, action).

        Re-call every frame with the UPDATED target to follow a MOVING human
        reactively (the controller is stateless: it reads the current robot pose
        and target each call)."""
        r = self.robot
        pos, quat = r.get_position_orientation()
        rxy = to_numpy(pos)[:2].astype(np.float64)
        ryaw = yaw_from_quat(quat)
        tgt = np.asarray(to_numpy(target_xy), dtype=np.float64).reshape(-1)[:2]
        to_t = tgt - rxy
        dist = float(np.linalg.norm(to_t))

        def _wrap(a: float) -> float:
            return math.atan2(math.sin(a), math.cos(a))

        vy = 0.0
        if dist < stop_dist:
            # Arrived at the follow point: rotate in place to face the human.
            f = (np.asarray(to_numpy(face_xy), dtype=np.float64).reshape(-1)[:2]
                 if face_xy is not None else tgt)
            vx = 0.0
            vyaw = max(-1.0, min(1.0, k_ang * _wrap(math.atan2(f[1] - rxy[1], f[0] - rxy[0]) - ryaw)))
        elif face_xy is not None:
            # HOLONOMIC follow (R1Pro base is holonomic — [vx, vy, vyaw] in the body
            # frame). Yaw tracks the HUMAN every frame while the base TRANSLATES toward
            # the target via strafe, so a sideways/backward yield no longer swings the
            # camera off the human. The old code faced the target (= the yield point,
            # up to 90° off the human), which blanked the human from first-person view
            # for ~4.5s on long relocates (v4 follow visibility 60% — the human left the
            # 45° head-cam FOV while the base pointed at its yield target).
            f = np.asarray(to_numpy(face_xy), dtype=np.float64).reshape(-1)[:2]
            vyaw = max(-1.0, min(1.0,
                       k_ang * _wrap(math.atan2(f[1] - rxy[1], f[0] - rxy[0]) - ryaw)))
            speed = min(k_fwd * dist, float(max_speed))         # m/s toward target
            vwx, vwy = (to_t / dist) * speed                    # world-frame velocity
            c, s = math.cos(ryaw), math.sin(ryaw)               # world -> body = R(-yaw)
            vx = max(-1.0, min(1.0, (c * vwx + s * vwy) / float(BASE_OUT_LIMITS[0])))
            vy = max(-1.0, min(1.0, (-s * vwx + c * vwy) / float(BASE_OUT_LIMITS[1])))
        else:
            yaw_err = _wrap(math.atan2(to_t[1], to_t[0]) - ryaw)
            vyaw = max(-1.0, min(1.0, k_ang * yaw_err))
            # Turn WHILE moving (no hard stop-and-turn): forward speed scales with
            # how much we already face the target (cos) and is capped at max_speed,
            # so the robot follows at ~the human's pace and curves smoothly instead
            # of stopping to rotate then rushing at the base limit. Facing away
            # (cos < 0) -> no forward, pure turn.
            face_factor = max(0.0, math.cos(yaw_err))
            vx = (min(k_fwd * dist, float(max_speed)) * face_factor) / float(BASE_OUT_LIMITS[0])
        action = self.prims._empty_action()                    # arms held
        bidx = r.controller_action_idx["base"]
        action[bidx] = th.tensor([float(vx), float(vy), float(vyaw)], dtype=th.float32)
        obs, _rew, _term, _trunc, _info = self.env.step(action)
        return obs, to_numpy(action).astype(np.float32)

    def drive_along_path(self, waypoints_xy: Sequence[Sequence[float]],
                         final_yaw: Optional[float] = None,
                         on_frame: Optional[Callable[[Any, np.ndarray], None]] = None,
                         low_precision: bool = True,
                         abort_fn: Optional[Callable[[], bool]] = None) -> bool:
        """Drive through an obstacle-aware A* path (caller-supplied xy
        waypoints), facing each leg's travel direction; optionally end at
        final_yaw. Returns True if the last leg completed."""
        pts = [np.asarray(p, dtype=np.float64)[:2] for p in waypoints_xy]
        ok = True
        for i, pt in enumerate(pts):
            cur = to_numpy(self.robot.get_position_orientation()[0]).astype(np.float64)[:2]
            seg_yaw = math.atan2(float(pt[1] - cur[1]), float(pt[0] - cur[0]))
            # Face the LEG travel direction while driving; do the FINAL facing after
            # arrival via shortest-direction rotate (the primitive's own final rotate
            # can wind the long way — ep000 ReturnToHuman spun 315°).
            ok = self.drive_to_pose((float(pt[0]), float(pt[1]), seg_yaw),
                                    on_frame=on_frame, low_precision=low_precision,
                                    abort_fn=abort_fn)
            if not ok:
                break
        if ok and final_yaw is not None:
            self.rotate_to_yaw_shortest(float(final_yaw), on_frame=on_frame)
        return ok

    def escape_step(self, nudge_xy=None, max_speed: float = 0.30):
        """ONE base step that NEVER rotates (vyaw ≡ 0): translate holonomically toward
        *nudge_xy* to un-wedge, or hold in place if None. The stuck guard uses this because
        a HARD in-place rotation while the base is wedged against furniture is exactly what
        tips it (ep0031: 9°→83° once follow_step applied vyaw=1.0 in the pocket). Arms held;
        records the real action."""
        r = self.robot
        bidx = r.controller_action_idx["base"]
        pos, quat = r.get_position_orientation()
        rxy = to_numpy(pos)[:2].astype(np.float64)
        ryaw = yaw_from_quat(quat)
        vx = vy = 0.0
        if nudge_xy is not None:
            to_t = np.asarray(to_numpy(nudge_xy), dtype=np.float64).reshape(-1)[:2] - rxy
            d = float(np.linalg.norm(to_t))
            if d > 1e-6:
                sp = min(1.5 * d, float(max_speed))
                vwx, vwy = (to_t / d) * sp
                c, s = math.cos(ryaw), math.sin(ryaw)
                vx = max(-1.0, min(1.0, (c * vwx + s * vwy) / float(BASE_OUT_LIMITS[0])))
                vy = max(-1.0, min(1.0, (-s * vwx + c * vwy) / float(BASE_OUT_LIMITS[1])))
        action = self.prims._empty_action()
        action[bidx] = th.tensor([float(vx), float(vy), 0.0], dtype=th.float32)
        obs, _rew, _term, _trunc, _info = self.env.step(action)
        return obs, to_numpy(action).astype(np.float32)

    def drive_path_smooth(self, waypoints_xy: Sequence[Sequence[float]],
                          max_speed: float = 0.5, k_ang: float = 1.6, k_fwd: float = 1.2,
                          pos_tol: float = 0.12,
                          on_frame: Optional[Callable[[Any, np.ndarray], None]] = None,
                          max_frames_per_leg: int = 400) -> bool:
        """Drive through waypoints with a CONTINUOUS turn-while-move controller: the
        heading tracks the bearing to the current waypoint and the forward speed scales
        with cos(heading error), so the robot starts walking toward the task AS it turns
        instead of a full in-place stop-pivot (RC3: the dominant spin was the ~146° median
        stop-turn to face the travel direction — user: 'leave the human and walk directly
        to the task'). NO terminal yaw blend — that livelocked (drive_path_curved: the last
        stretch needed to face ~170° off travel, so cos-gated forward speed hit 0 and it
        stuck 0.7m short spinning). The caller does ONE shortest in-place turn to the replay
        yaw afterwards (run_face_yaw_phase), which the user explicitly allows. Forward speed
        tapers with distance so it does not overshoot/orbit a waypoint. Forward only (vy=0);
        records a real base velocity each frame."""
        r = self.robot
        bidx = r.controller_action_idx["base"]

        def _wrap(a: float) -> float:
            return math.atan2(math.sin(a), math.cos(a))

        for pt in [np.asarray(p, dtype=np.float64)[:2] for p in waypoints_xy]:
            for _ in range(int(max_frames_per_leg)):
                pos, quat = r.get_position_orientation()
                rxy = to_numpy(pos)[:2].astype(np.float64)
                ryaw = yaw_from_quat(quat)
                to_t = pt - rxy
                d = float(np.linalg.norm(to_t))
                if d < pos_tol:
                    break
                yaw_err = _wrap(math.atan2(float(to_t[1]), float(to_t[0])) - ryaw)
                vyaw = max(-1.0, min(1.0, k_ang * yaw_err))
                v = min(k_fwd * d, max_speed) * max(0.0, math.cos(yaw_err))
                vx = max(-1.0, min(1.0, v / float(BASE_OUT_LIMITS[0])))
                action = self.prims._empty_action()
                action[bidx] = th.tensor([float(vx), 0.0, float(vyaw)], dtype=th.float32)
                obs, _rew, _term, _trunc, _info = self.env.step(action)
                if on_frame is not None:
                    on_frame(obs, to_numpy(action).astype(np.float32))
                # TILT-ABORT (same guard as _run): stop before pushing past a fall.
                _q = to_numpy(r.get_position_orientation()[1])
                _rr = math.degrees(math.atan2(2 * (_q[3] * _q[0] + _q[1] * _q[2]),
                                              1 - 2 * (_q[0] * _q[0] + _q[1] * _q[1])))
                _pp = math.degrees(math.asin(max(-1.0, min(1.0,
                                    2 * (_q[3] * _q[1] - _q[2] * _q[0])))))
                if max(abs(_rr), abs(_pp)) > 12.0:
                    print(f"[OnlineNavDriver] drive_path_smooth TILT-ABORT at "
                          f"{max(abs(_rr), abs(_pp)):.0f}°", flush=True)
                    return True
        return True
