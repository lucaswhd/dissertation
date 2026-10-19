# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
# See LICENSE in the project root for full terms.

import math
import json
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    import omnigibson as og
    import omnigibson.lazy as lazy
except ModuleNotFoundError:
    for local_pkg_dir in ("OmniGibson", "bddl3"):
        candidate = REPO_ROOT / local_pkg_dir
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    import omnigibson as og
    import omnigibson.lazy as lazy
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.objects.usd_object import USDObject
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.utils.usd_utils import array_to_vtarray

from hrc_utils import DEFAULT_HUMAN_USD, download_human_usd, forward_xy_from_quat, quat_from_yaw, to_numpy, yaw_from_quat


HUMAN_MOTION_STATE_IDS = {
    "hold": 0,
    "walk": 1,
    "turn": 2,
}


def _wrap_angle(x: float) -> float:
    return math.atan2(math.sin(x), math.cos(x))


@dataclass
class HumanBodyProfile:
    pelvis_height: float = 0.98
    spine_height: float = 1.14
    neck_height: float = 1.42
    head_height: float = 1.65
    shoulder_width: float = 0.42
    hip_width: float = 0.28
    upper_arm_length: float = 0.30
    lower_arm_length: float = 0.28
    upper_leg_length: float = 0.46
    lower_leg_length: float = 0.45
    foot_length: float = 0.16
    step_swing_amplitude: float = 0.08
    arm_swing_amplitude: float = 0.12


@dataclass
class CollisionProxyConfig:
    enabled: bool = True
    lower_body_radius: float = 0.18
    lower_body_height: float = 0.96
    torso_radius: float = 0.20
    torso_height: float = 0.58
    head_radius: float = 0.12
    mass_kg: float = 70.0


@dataclass
class HumanPhaseMotion:
    mode: str = "hold"
    face_robot: bool = True
    speed: float = 0.35
    patrol_radius: float = 0.8
    waypoints: List[List[float]] = field(default_factory=list)
    loop: bool = True
    turning_step_amount_deg: float = 20.0
    min_angle_turn_deg: float = 5.0
    distance_multiplier: float = 1.0
    dist_thresh: float = 0.15
    turn_thresh_deg: float = 10.0
    expand_with_astar: bool = True


@dataclass
class HumanActorSnapshot:
    phase: str
    motion_state: str
    motion_state_id: int
    base_pos: np.ndarray
    base_quat: np.ndarray
    linear_speed: float
    keypoints_world: Dict[str, np.ndarray]


@dataclass
class HumanoidAnimationRuntime:
    visual_usd_path: str
    metadata_path: str
    npz_path: str
    joint_paths: List[str]
    walk_rotations_xyzw: np.ndarray
    stop_rotations_xyzw: np.ndarray
    translations: np.ndarray
    scales: np.ndarray
    stop_translations: np.ndarray
    stop_scales: np.ndarray
    walk_fps: float
    frame_count: int
    nominal_speed_mps: float
    visual_correction_quat_xyzw: np.ndarray
    is_kit_usd: bool = False
    meters_per_unit: float = 1.0


class ReferencedHumanVisual(XFormPrim):
    """
    Lightweight referenced USD visual mounted under a controllable parent Xform.

    We use this for Habitat humanoid animation packages because they are pure visual
    USD / UsdSkel assets, not articulated OG objects with links and joints.
    """

    def __init__(
        self,
        relative_prim_path: str,
        name: str,
        usd_path: str,
        asset_prim_name: str = "VisualAsset",
        load_config: Optional[Dict[str, Any]] = None,
    ):
        self._usd_path = str(usd_path)
        self._asset_prim_name = str(asset_prim_name)
        super().__init__(relative_prim_path=relative_prim_path, name=name, load_config=load_config)

    @property
    def asset_prim_path(self) -> str:
        return f"{self.prim_path}/{self._asset_prim_name}"

    def _load(self):
        prim = super()._load()
        # Use "" (typeless) so the SkelRoot type from walk.usda propagates through the
        # reference into this prim.  Using "Xform" here would override it in the session
        # layer (highest-strength layer) and prevent UsdSkel from processing the skinned
        # mesh, causing the character to slide as a static shape without deformation.
        asset_prim = og.sim.stage.DefinePrim(self.asset_prim_path, "")
        assert asset_prim.GetReferences().AddReference(self._usd_path)
        # --- diagnostic: dump prim tree after reference ---
        try:
            from pxr import UsdGeom, Usd
            ref_stage = Usd.Stage.Open(self._usd_path)
            if ref_stage:
                dp = ref_stage.GetDefaultPrim()
                mpu = UsdGeom.GetStageMetersPerUnit(ref_stage)
                up = UsdGeom.GetStageUpAxis(ref_stage)
                print(f"[ReferencedHumanVisual] Kit USD defaultPrim={dp.GetPath() if dp else 'NONE'} "
                      f"metersPerUnit={mpu} upAxis={up}", flush=True)
            def _dump(p, depth=0):
                if depth > 4:
                    return
                vis = ""
                vattr = p.GetAttribute("visibility")
                if vattr and vattr.IsValid():
                    vis = f" vis={vattr.Get()}"
                ext = ""
                eattr = p.GetAttribute("extent")
                if eattr and eattr.IsValid():
                    ext = f" extent={eattr.Get()}"
                print(f"[ReferencedHumanVisual]  {'  '*depth}{p.GetPath()} ({p.GetTypeName()}){vis}{ext}", flush=True)
                for c in p.GetChildren():
                    _dump(c, depth+1)
            _dump(asset_prim)
        except Exception as e:
            print(f"[ReferencedHumanVisual] diag error: {e}", flush=True)
        return prim


class KinematicHumanoidActor:
    """
    Lightweight humanoid actor used for closed-loop HRC collection.

    The actor intentionally separates:
    - a visual mesh loaded from USD
    - optional invisible collision proxies for coarse occupancy
    - a simple phase-driven motion controller
    - synthetic 3D keypoint export for dataset logging

    This keeps the implementation stable during BEHAVIOR playback, where the
    robot and scene are frequently restored from recorded simulator states.
    """

    KEYPOINT_NAMES = [
        "pelvis",
        "spine",
        "neck",
        "head",
        "left_shoulder",
        "left_elbow",
        "left_hand",
        "right_shoulder",
        "right_elbow",
        "right_hand",
        "left_hip",
        "left_knee",
        "left_foot",
        "right_hip",
        "right_knee",
        "right_foot",
    ]

    def __init__(
        self,
        env: Any,
        usd_path: str,
        name: str = "human_npc",
        distance: float = 1.8,
        scale: Optional[float] = None,
        body_profile: Optional[HumanBodyProfile] = None,
        collision_proxy: Optional[CollisionProxyConfig] = None,
        phase_motion: Optional[Dict[str, HumanPhaseMotion]] = None,
        step_dt: float = 1.0 / 30.0,
        nav_erosion_radius: float = 0.3,
    ):
        self.env = env
        self.usd_path = usd_path
        self.name = name
        self.distance = float(distance)
        self.scale = None if scale is None else float(scale)
        self.body_profile = body_profile or HumanBodyProfile()
        self.collision_proxy = collision_proxy or CollisionProxyConfig()
        self.phase_motion = phase_motion or self._default_phase_motion()
        self.step_dt = float(step_dt)
        self.nav_erosion_radius = float(nav_erosion_radius)

        self.visual: Optional[Any] = None
        self.proxy_objects: Dict[str, Any] = {}
        # Furniture AABB rects (xmin,ymin,xmax,ymax), set by the demo at trav_map
        # bake time; the robot-dodge consults these so it never nudges the human
        # into a table/cabinet (the dodge doesn't read trav_map).
        self._furniture_rects: List[Any] = []
        # D2: when True, _sync_proxies parks the collision proxies far away (set
        # during ExecuteBehaviorTask so the human can't disturb the task replay).
        self._proxies_parked: bool = False
        # D1: current round's task-area centre (xy), set by the demo; the dodge
        # steers the human AWAY from it so it never backs into task objects.
        self._task_center: Optional[np.ndarray] = None
        self.visual_asset_path = str(usd_path)
        self.visual_relative_prim_path = f"/{self.name}"
        self.animation_runtime = self._load_animation_runtime(str(usd_path))
        if self.animation_runtime is not None:
            self.visual_asset_path = self.animation_runtime.visual_usd_path
            # Auto-compensate Kit USD metersPerUnit (e.g. 0.01 for cm) when user
            # hasn't set an explicit scale.
            if self.scale is None and self.animation_runtime.is_kit_usd:
                mpu = self.animation_runtime.meters_per_unit
                if abs(mpu - 1.0) > 1e-6 and mpu > 0:
                    self.scale = float(mpu)
                    print(f"[hrc_humanoid] Auto-scale from Kit USD metersPerUnit: {self.scale}", flush=True)
        self._animation_root_attr = None
        self._animation_rotations_attr = None
        self._animation_translations_attr = None
        self._animation_scales_attr = None
        self._animation_handles_bound = False

        self.base_pos = np.zeros(3, dtype=np.float32)
        self.base_yaw = 0.0
        self.anchor_pos = np.zeros(3, dtype=np.float32)
        self.anchor_yaw = 0.0

        self.active_phase = "WaitNearHuman"
        self.motion_state = "hold"
        self.linear_speed = 0.0
        self._last_displacement_m = 0.0
        self.gait_phase = 0.0
        self.phase_step = 0
        self.phase_waypoints: List[np.ndarray] = []
        self.phase_waypoint_idx = 0
        self.animation_frame_cursor = 0.0
        self.animation_state = "hold"
        self._kit_joint_remap_indices: Optional[np.ndarray] = None

    @classmethod
    def from_config(cls, env: Any, cfg: Dict[str, Any], fps: int) -> "KinematicHumanoidActor":
        human_cfg = dict(cfg.get("human", {}))
        body_cfg = dict(human_cfg.get("body_profile", {}))
        proxy_cfg = dict(human_cfg.get("collision_proxy", {}))
        phase_cfg_raw = dict(human_cfg.get("phase_motion", {}))
        phase_motion: Dict[str, HumanPhaseMotion] = {}
        for phase_name, phase_cfg in phase_cfg_raw.items():
            phase_motion[phase_name] = HumanPhaseMotion(**phase_cfg)
        return cls(
            env=env,
            usd_path=human_cfg.get("usd_path", cfg.get("human_usd", DEFAULT_HUMAN_USD)),
            name=human_cfg.get("name", "human_npc"),
            distance=float(human_cfg.get("distance", cfg.get("human_distance", 1.8))),
            scale=human_cfg.get("scale", cfg.get("human_scale")),
            body_profile=HumanBodyProfile(**body_cfg),
            collision_proxy=CollisionProxyConfig(**proxy_cfg),
            phase_motion=phase_motion or None,
            step_dt=1.0 / max(int(fps), 1),
        )

    def _default_phase_motion(self) -> Dict[str, HumanPhaseMotion]:
        return {
            "default": HumanPhaseMotion(mode="hold", face_robot=True, speed=0.35, patrol_radius=0.8, loop=True),
            "ExecuteBehaviorTask": HumanPhaseMotion(
                mode="waypoint_loop",
                face_robot=False,
                speed=0.40,
                patrol_radius=0.9,
                loop=True,
            ),
            "ReturnToHuman": HumanPhaseMotion(mode="hold", face_robot=True, speed=0.30, patrol_radius=0.8, loop=True),
            "WaitNextCommand": HumanPhaseMotion(mode="hold", face_robot=True, speed=0.30, patrol_radius=0.8, loop=True),
        }

    def _get_phase_spec(self, phase: str) -> HumanPhaseMotion:
        return self.phase_motion.get(phase, self.phase_motion.get("default", HumanPhaseMotion()))

    def _compute_spawn_pose(self, distance: float) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        robot = self.env.robots[0]
        robot_pos, robot_quat = robot.get_position_orientation()
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
        right = -left
        back = -forward
        robot_xy = to_numpy(robot_pos).astype(np.float32)[:2]
        base_z = float(max(to_numpy(robot_pos)[2], 0.0))

        # Spawn-point selection. The human is preferably placed `distance` in
        # FRONT of the robot (so it faces the robot for the opening shot). The
        # old code placed it there *blindly*; in scenes where the robot's
        # step-0 pose faces a wall/appliance (e.g. task-0001: robot faces the
        # fridge) the human spawned INSIDE furniture and clipped through it.
        # We now validate each candidate against the traversable map and fall
        # back to other directions / a nearest-navigable snap. Scenes where the
        # old in-front point was already navigable (e.g. task-0026) pick the
        # exact same point, so there is no regression.
        trav_map = self._try_get_trav_map()

        def _good_spawn_xy(xy: np.ndarray, min_clearance: float = 0.0) -> Optional[np.ndarray]:
            """Return a navigable, CONNECTED spawn (x,y) for `xy`, else None.

            Two checks, both needed (task-0001 bug): (1) the point is navigable
            and the snap barely moved it (else it sits inside furniture, e.g.
            the fridge); (2) it has an A* path to the robot — i.e. it lies in
            the main navigable region, not an isolated pocket. The original
            spawn [5.51,-0.31] passed (1) but failed (2): it snapped to an
            isolated cell from which the human's A* reached nothing, so Relocate
            (human walks, robot follows) collapsed.
            """
            if trav_map is None:
                return xy.astype(np.float32)  # no map -> cannot validate, accept
            snapped = self._snap_to_navigable(xy, trav_map)
            if snapped is None:
                return None
            sn = to_numpy(snapped).astype(np.float32)
            if float(np.linalg.norm(sn - xy.astype(np.float32))) > 0.5:
                return None  # snap jumped far -> original point inside an obstacle
            # Breathing room: a cell can be "navigable" while hugging a stove/table/
            # stairs (ep0005 spawned 3cm from the stove -> feet clipped). Every BAD
            # episode in the 8-batch had spawn clearance <= 0.24m; good ones 0.61m+.
            if not self._clearance_ok(sn, trav_map, 0, min_clearance):
                return None
            path = self._nav_path_between(
                np.array([sn[0], sn[1], base_z], dtype=np.float32),
                np.array([robot_xy[0], robot_xy[1], base_z], dtype=np.float32),
                base_z, trav_map, 0,
            )
            if path is None or len(path) < 2:
                return None  # navigable but isolated from the robot's region
            # Visibility: require the spawn be in the SAME ROOM as the robot, so the
            # robot's onboard camera actually SEES the human — no door/wall between.
            # The geometric human_visible flag is FOV+distance only (not occlusion-
            # aware), so a human just past a door reads "visible" but is occluded
            # (ep0030: spawned 1.9m straight ahead, but through a doorway). Same-room
            # is the robust occlusion proxy. Skipped if the scene has no seg map, or
            # if either point falls on a room boundary (room=None).
            try:
                _scene = og.sim.scenes[0] if og.sim.scenes else None
                _seg = getattr(_scene, "seg_map", None)
                if _seg is not None and getattr(_seg, "room_ins_map", None) is not None:
                    _r_room = _seg.get_room_instance_by_point(np.asarray(robot_xy, dtype=np.float32))
                    _h_room = _seg.get_room_instance_by_point(np.asarray(sn, dtype=np.float32))
                    if _r_room is not None and _h_room is not None and _r_room != _h_room:
                        return None  # different room -> door/wall between -> occluded
            except Exception:
                pass
            return sn

        # (direction, lateral) candidates in priority order, tried at a few
        # distances: front first, then sides, then behind.
        dir_lateral = [
            (forward, 0.45), (forward, -0.45), (forward, 0.0),
            (left, 0.0), (right, 0.0), (back, 0.0),
        ]
        dist_candidates = [float(distance), float(distance) * 0.7, 1.2]

        human_pos = None
        chosen_lateral = 0.45
        # Direction OUTER, distance INNER: fully exhaust the FRONT directions (which
        # are in the camera FOV) across all distances before falling to the sides/back
        # (out of view). So if the far-front spawn lands past a doorway (different room
        # -> rejected by _good_spawn_xy's same-room check), a CLOSER front spot (still
        # in this room, in FOV) is tried before any out-of-FOV side spawn. (ep0030.)
        # Prefer ROOMY spawns, then relax. Measured on the 8-episode batch: every
        # episode the user judged BAD spawned <=0.24m from furniture/stairs (ep0005
        # 0.03m -> feet clipped the stove), while the good ones had 0.61-1.19m.
        # Relaxing keeps tight scenes spawnable instead of hitting the blind fallback.
        for _min_clr in (0.50, 0.30, 0.0):
            for direction, lateral in dir_lateral:
                for dist in dist_candidates:
                    cand = spawn_pos + direction * float(dist) + left * float(lateral)
                    cand_xy = cand[:2].astype(np.float32)
                    if float(np.linalg.norm(cand_xy - robot_xy)) < 1.1:
                        continue
                    good = _good_spawn_xy(cand_xy, _min_clr)
                    if good is not None:
                        human_pos = np.array([good[0], good[1], base_z], dtype=np.float32)
                        chosen_lateral = float(lateral)
                        break
                if human_pos is not None:
                    break
            if human_pos is not None:
                print(f"[hrc_humanoid] spawn clearance tier = {_min_clr:.2f}m", flush=True)
                break

        if human_pos is None and trav_map is not None:
            # Last resort: sample navigable points near the robot and accept the
            # first one CONNECTED to the robot (so the human can actually walk
            # during Relocate). Avoids the isolated-pocket trap that the simple
            # nearest-navigable snap would fall back into.
            ref = th.tensor([robot_xy[0], robot_xy[1], base_z], dtype=th.float32)
            for _ in range(60):
                try:
                    _, rand_pt = trav_map.get_random_point(floor=0, reference_point=ref)
                    cand_xy = np.array([float(rand_pt[0]), float(rand_pt[1])], dtype=np.float32)
                except Exception:
                    continue
                d = float(np.linalg.norm(cand_xy - robot_xy))
                if d < 1.1 or d > 4.0:
                    continue
                good = _good_spawn_xy(cand_xy, 0.30)   # keep breathing room here too
                if good is not None:
                    human_pos = np.array([good[0], good[1], base_z], dtype=np.float32)
                    chosen_lateral = 0.0
                    print(f"[hrc_humanoid] spawn fell back to sampled navigable "
                          f"({good[0]:.2f}, {good[1]:.2f}), {d:.2f}m from robot", flush=True)
                    break

        if human_pos is None:
            # Absolute fallback: keep the blind in-front point (no map, or no
            # connected point found). Logged so video review can catch it.
            pref_xy = (spawn_pos + forward * float(distance) + left * 0.45)[:2].astype(np.float32)
            human_pos = np.array([pref_xy[0], pref_xy[1], base_z], dtype=np.float32)
            print("[hrc_humanoid] WARN: no navigable/connected spawn found, using "
                  "blind in-front point (may clip furniture)", flush=True)

        human_yaw = math.atan2(float(robot_xy[1] - human_pos[1]), float(robot_xy[0] - human_pos[0]))
        debug = {
            "spawn_sensor_name": evidence_sensor_name,
            "spawn_distance": float(distance),
            "chosen_lateral": float(chosen_lateral),
            "spawn_pos": human_pos.tolist(),
            "spawn_yaw": float(human_yaw),
        }
        return human_pos, human_yaw, debug

    def _load_animation_runtime(self, asset_path: str) -> Optional[HumanoidAnimationRuntime]:
        metadata_path = Path(asset_path).expanduser().resolve().parent / "source_asset.json"
        if not metadata_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        if not bool(metadata.get("walk_clip_contains_skel_animation", False)):
            return None
        package_files = metadata.get("package_files", {})
        walk_usd_path = package_files.get("walk_clip_usda")
        npz_path = package_files.get("walk_motion_metadata")
        if not walk_usd_path or not npz_path:
            return None
        walk_usd = Path(walk_usd_path).expanduser().resolve()
        npz_file = Path(npz_path).expanduser().resolve()
        if not npz_file.exists():
            return None

        # Prefer Kit-converted USD for renderer-native materials (avoids pink skin).
        kit_usd = walk_usd.parent / "_kit_converted.usd"
        using_kit_usd = False
        kit_meters_per_unit = 1.0
        if kit_usd.exists() and kit_usd.stat().st_size > 0:
            visual_usd = kit_usd
            using_kit_usd = True
            # Read metersPerUnit so caller can auto-compensate scale.
            try:
                from pxr import Usd, UsdGeom
                tmp_stage = Usd.Stage.Open(str(kit_usd))
                if tmp_stage:
                    kit_meters_per_unit = UsdGeom.GetStageMetersPerUnit(tmp_stage)
                    print(f"[hrc_humanoid] Using Kit-converted USD: {kit_usd} "
                          f"(metersPerUnit={kit_meters_per_unit})", flush=True)
                del tmp_stage
            except Exception as _e:
                print(f"[hrc_humanoid] Using Kit-converted USD: {kit_usd} (mpu read failed: {_e})", flush=True)
        elif walk_usd.exists():
            visual_usd = walk_usd
            print(f"[hrc_humanoid] Kit USD not found, falling back to walk.usda", flush=True)
        else:
            return None

        try:
            runtime_npz = np.load(npz_file, allow_pickle=False)
        except Exception:
            return None

        required = {
            "skel_joint_paths",
            "skel_translations",
            "skel_scales",
            "skel_rotations_time_samples",
            "stop_rotations",
            "fps",
        }
        if not required.issubset(set(runtime_npz.files)):
            return None

        walk_rotations_xyzw = np.asarray(runtime_npz["skel_rotations_time_samples"], dtype=np.float32)
        stop_rotations_xyzw = np.asarray(runtime_npz["stop_rotations"], dtype=np.float32)
        translations = np.asarray(runtime_npz["skel_translations"], dtype=np.float32)
        scales = np.asarray(runtime_npz["skel_scales"], dtype=np.float32)
        stop_translations = np.asarray(
            runtime_npz["stop_translations"] if "stop_translations" in runtime_npz.files else runtime_npz["skel_translations"],
            dtype=np.float32,
        )
        stop_scales = np.asarray(
            runtime_npz["stop_scales"] if "stop_scales" in runtime_npz.files else runtime_npz["skel_scales"],
            dtype=np.float32,
        )
        joint_paths = [str(v) for v in np.asarray(runtime_npz["skel_joint_paths"]).tolist()]
        walk_fps_arr = np.asarray(runtime_npz["fps"], dtype=np.float32).reshape(-1)
        walk_fps = float(walk_fps_arr[0]) if walk_fps_arr.size > 0 else 0.0
        frame_count = int(walk_rotations_xyzw.shape[0]) if walk_rotations_xyzw.ndim >= 1 else 0
        duration_s = float(frame_count / walk_fps) if walk_fps > 1e-6 else 0.0
        nominal_speed = 0.0
        path_length = float(metadata.get("walk_root_motion_preview_path_length_m", 0.0) or 0.0)
        if duration_s > 1e-6 and path_length > 1e-6:
            nominal_speed = path_length / duration_s
        # Kit converter's convert_stage_up_z only sets stage metadata (upAxis=Z);
        # vertex / skeleton data stays in Y-up local space (confirmed by mesh
        # extent: height is along Y).  Both Kit USD and legacy walk.usda need
        # the 90-deg X rotation to stand upright in OmniGibson's Z-up stage.
        visual_correction_quat_xyzw = np.asarray(
            metadata.get(
                "runtime_visual_correction_quat_xyzw",
                [0.70710677, 0.0, 0.0, 0.70710677]
                if metadata.get("authored_up_axis") == "Y" and metadata.get("target_stage_up_axis") == "Z"
                else [0.0, 0.0, 0.0, 1.0],
            ),
            dtype=np.float32,
        ).reshape(4)

        return HumanoidAnimationRuntime(
            visual_usd_path=str(visual_usd),
            is_kit_usd=using_kit_usd,
            meters_per_unit=kit_meters_per_unit,
            metadata_path=str(metadata_path),
            npz_path=str(npz_file),
            joint_paths=joint_paths,
            walk_rotations_xyzw=walk_rotations_xyzw,
            stop_rotations_xyzw=stop_rotations_xyzw,
            translations=translations,
            scales=scales,
            stop_translations=stop_translations,
            stop_scales=stop_scales,
            walk_fps=walk_fps,
            frame_count=frame_count,
            nominal_speed_mps=nominal_speed,
            visual_correction_quat_xyzw=visual_correction_quat_xyzw,
        )

    def _quat_multiply_xyzw(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        lx, ly, lz, lw = [float(v) for v in np.asarray(lhs, dtype=np.float32).reshape(4)]
        rx, ry, rz, rw = [float(v) for v in np.asarray(rhs, dtype=np.float32).reshape(4)]
        quat = np.asarray(
            [
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
                lw * rw - lx * rx - ly * ry - lz * rz,
            ],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(quat))
        if norm <= 1e-8:
            return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return quat / np.float32(norm)

    def _body_forward_yaw_offset(self) -> float:
        """Yaw offset (rad) that aligns mesh body-forward with motion +X.

        For Habitat SMPL-X assets converted Y-up→Z-up (correction quat
        ≈ [0.707, 0, 0, 0.707]), the mesh body-forward axis (empirically
        +Z body) lands on world -Y after the correction. Adding +pi/2 in
        yaw rotates -Y to +X, matching the motion vector (cos yaw, sin yaw).

        An earlier version of this method used -pi/2 which produced a 180°
        mismatch (body faced opposite to motion, i.e. "walking backward").
        """
        if self.animation_runtime is None:
            return 0.0
        correction = np.asarray(self.animation_runtime.visual_correction_quat_xyzw, dtype=np.float32).reshape(4)
        y_up_to_z_up = np.array([0.70710677, 0.0, 0.0, 0.70710677], dtype=np.float32)
        if np.allclose(correction, y_up_to_z_up, atol=1e-3):
            return math.pi / 2.0
        return 0.0

    def _visual_orientation_from_yaw(self, yaw: float) -> np.ndarray:
        quat = quat_from_yaw(float(yaw) + self._body_forward_yaw_offset())
        if self.animation_runtime is None:
            return quat
        correction = np.asarray(self.animation_runtime.visual_correction_quat_xyzw, dtype=np.float32).reshape(4)
        if np.allclose(correction, np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), atol=1e-5):
            return quat
        return self._quat_multiply_xyzw(quat, correction)

    def _animation_handles_ready(self) -> bool:
        required_attrs = (
            self._animation_rotations_attr,
            self._animation_translations_attr,
            self._animation_scales_attr,
        )
        return bool(self._animation_handles_bound) and all(attr is not None and attr.IsValid() for attr in required_attrs)

    def _get_current_animation_payload(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self.animation_runtime is None:
            return None
        if self.animation_state == "walk" and self.animation_runtime.frame_count > 0:
            frame_idx = int(self.animation_frame_cursor) % self.animation_runtime.frame_count
            rotations_xyzw = self.animation_runtime.walk_rotations_xyzw[frame_idx]
            translations = self.animation_runtime.translations
            scales = self.animation_runtime.scales
        else:
            rotations_xyzw = self.animation_runtime.stop_rotations_xyzw
            translations = self.animation_runtime.stop_translations
            scales = self.animation_runtime.stop_scales
        # Apply joint remapping if Kit USD skeleton has different joint set
        if self._kit_joint_remap_indices is not None:
            idx = self._kit_joint_remap_indices
            rotations_xyzw = rotations_xyzw[idx]
            translations = translations[idx]
            scales = scales[idx]
        # Kit-converted USD stores skeleton data in stage units (e.g. cm when
        # metersPerUnit=0.01).  The bridge npz translations are in metres, so
        # we must convert: stage_units = metres / metersPerUnit.
        translations = np.asarray(translations, dtype=np.float32)
        if self.animation_runtime.is_kit_usd:
            mpu = self.animation_runtime.meters_per_unit
            if mpu > 0 and abs(mpu - 1.0) > 1e-6:
                translations = translations / np.float32(mpu)
        rotations_wxyz = self._quat_array_xyzw_to_wxyz(rotations_xyzw)
        return rotations_wxyz, translations, np.asarray(scales, dtype=np.float32)

    def _bind_animation_handles(self, force: bool = False) -> None:
        """Bind USD attribute handles for skeletal animation.

        Supports two USD layouts:
        - **Kit-converted USD** (from omni.kit.asset_converter): skeleton at
          ``root/pelvis/Skeleton``, mesh at ``root/pelvis/SMPLX_shapes_*``.
          A UsdSkel.Animation prim is created at runtime and bound to the
          skeleton & mesh, similar to the senior's HumanoidPoseController.
        - **Legacy walk.usda**: pre-baked ``Rig/Skel/WalkAnimation`` prim.

        Once bound, handles are cached and reused every frame.
        Pass *force=True* only after the visual prim is re-created.
        """
        if self._animation_handles_ready() and not force:
            return

        # Reset handles.
        self._animation_root_attr = None
        self._animation_rotations_attr = None
        self._animation_translations_attr = None
        self._animation_scales_attr = None
        self._animation_handles_bound = False

        if self.visual is None or self.animation_runtime is None:
            return

        asset_root_path = getattr(self.visual, "asset_prim_path", None) or self.visual.prim_path
        root_prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(asset_root_path)
        if not root_prim or not root_prim.IsValid():
            print(f"[hrc_humanoid] _bind: root prim invalid: {asset_root_path}", flush=True)
            return

        animation_prim = None
        is_kit_usd = False

        # --- Strategy A: Kit-converted USD (skeleton at root/pelvis/Skeleton) ---
        skel_prim = None
        mesh_prim = None
        for skel_suffix in ("root/pelvis/Skeleton",):
            candidate = lazy.isaacsim.core.utils.prims.get_prim_at_path(f"{asset_root_path}/{skel_suffix}")
            if candidate and candidate.IsValid():
                skel_prim = candidate
                break
        if skel_prim is not None:
            # Find mesh prim (SMPLX_shapes_*)
            pelvis_path = f"{asset_root_path}/root/pelvis"
            pelvis_prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(pelvis_path)
            if pelvis_prim and pelvis_prim.IsValid():
                for child in pelvis_prim.GetChildren():
                    if child.GetName().startswith("SMPLX_shapes") or child.GetName().startswith("SMPLX_male") or child.GetName().startswith("SMPLX_female") or child.GetName().startswith("SMPLX_neutral"):
                        mesh_prim = child
                        break
            if mesh_prim is None:
                # Fallback: search more broadly
                for child in pelvis_prim.GetChildren() if pelvis_prim else []:
                    if "Mesh" in str(child.GetTypeName()) or "Mesh" in child.GetName():
                        mesh_prim = child
                        break
            if mesh_prim is not None:
                is_kit_usd = True
                # Create UsdSkel.Animation prim (like senior's HumanoidPoseController)
                anim_path_str = f"{asset_root_path}/WalkAnim"
                anim = lazy.pxr.UsdSkel.Animation.Define(og.sim.stage, anim_path_str)

                # --- Joint path remapping ---
                # Kit USD skeleton may use different joint paths than our npz.
                # e.g. skel: ['pelvis', 'pelvis/left_hip', ...] (55)
                #      npz:  ['root', 'root/pelvis', 'root/pelvis/left_hip', ...] (56)
                # We detect and remap by stripping the common prefix ('root/').
                skel_obj = lazy.pxr.UsdSkel.Skeleton(skel_prim)
                skel_joints_attr = skel_obj.GetJointsAttr().Get() if skel_obj else None
                skel_joints_set = set(skel_joints_attr) if skel_joints_attr else set()
                anim_joints = list(self.animation_runtime.joint_paths)
                direct_match = sum(1 for a in anim_joints if a in skel_joints_set)

                remap_indices = None
                effective_joints = anim_joints
                if direct_match < len(anim_joints) // 2:
                    # Try stripping common root prefix (e.g. 'root/')
                    prefix = anim_joints[0] + "/" if len(anim_joints) > 0 else ""
                    stripped = []
                    indices = []
                    for i, jp in enumerate(anim_joints):
                        if jp == anim_joints[0]:
                            continue  # skip the root joint itself
                        s = jp[len(prefix):] if jp.startswith(prefix) else jp
                        if s in skel_joints_set:
                            stripped.append(s)
                            indices.append(i)
                    if len(stripped) >= len(skel_joints_set) * 0.8:
                        remap_indices = np.array(indices, dtype=np.int64)
                        effective_joints = stripped
                        self._kit_joint_remap_indices = remap_indices
                        print(
                            f"[hrc_humanoid] joint remap: stripped '{prefix}' prefix, "
                            f"{len(stripped)}/{len(skel_joints_set)} skel joints matched",
                            flush=True,
                        )

                joint_tokens = lazy.pxr.Vt.TokenArray(effective_joints)
                anim.CreateJointsAttr().Set(joint_tokens)

                # Use remapped subset for translations/scales if needed
                trans_arr = np.asarray(self.animation_runtime.translations, dtype=np.float32)
                scales_arr = self.animation_runtime.scales
                if remap_indices is not None:
                    trans_arr = trans_arr[remap_indices]
                    scales_arr = scales_arr[remap_indices]
                # Convert metre-scale bridge translations to Kit USD stage
                # units (e.g. centimetres when metersPerUnit=0.01).
                mpu = self.animation_runtime.meters_per_unit
                if mpu > 0 and abs(mpu - 1.0) > 1e-6:
                    trans_arr = trans_arr / np.float32(mpu)
                translations_vt = array_to_vtarray(
                    trans_arr.tolist(), lazy.pxr.Gf.Vec3f
                )
                scales_vt = array_to_vtarray(
                    scales_arr.tolist(), lazy.pxr.Gf.Vec3h
                )
                anim.CreateTranslationsAttr().Set(translations_vt)
                anim.CreateScalesAttr().Set(scales_vt)
                # Bind animation source to skeleton and mesh
                lazy.pxr.UsdSkel.BindingAPI.Apply(skel_prim)
                lazy.pxr.UsdSkel.BindingAPI.Apply(mesh_prim)
                skel_binding = lazy.pxr.UsdSkel.BindingAPI(skel_prim)
                mesh_binding = lazy.pxr.UsdSkel.BindingAPI(mesh_prim)
                anim_sdf_path = lazy.pxr.Sdf.Path(anim_path_str)
                skel_binding.CreateAnimationSourceRel().SetTargets([anim_sdf_path])
                mesh_binding.CreateAnimationSourceRel().SetTargets([anim_sdf_path])
                animation_prim = anim.GetPrim()
                print(
                    f"[hrc_humanoid] Kit USD: skel={skel_prim.GetPath()} "
                    f"mesh={mesh_prim.GetPath()} anim={anim_path_str} "
                    f"joints={len(effective_joints)} remap={'yes' if remap_indices is not None else 'no'}",
                    flush=True,
                )

        # --- Strategy B: Legacy walk.usda (Rig/Skel/WalkAnimation) ---
        if animation_prim is None:
            for root_path in (asset_root_path, self.visual.prim_path, f"{self.visual.prim_path}/Character"):
                candidate_root = lazy.isaacsim.core.utils.prims.get_prim_at_path(root_path)
                if not candidate_root or not candidate_root.IsValid():
                    continue
                anim_path = f"{root_path}/Rig/Skel/WalkAnimation"
                candidate_anim = lazy.isaacsim.core.utils.prims.get_prim_at_path(anim_path)
                if candidate_anim and candidate_anim.IsValid():
                    root_prim = candidate_root
                    animation_prim = candidate_anim
                    break

        if animation_prim is None:
            print(
                f"[hrc_humanoid] _bind: FAILED (no skeleton or WalkAnimation found) "
                f"asset_root={asset_root_path}",
                flush=True,
            )
            return

        # Cache attribute handles.
        self._animation_root_attr = root_prim.GetAttribute("xformOp:transform")
        self._animation_rotations_attr = animation_prim.GetAttribute("rotations")
        self._animation_translations_attr = animation_prim.GetAttribute("translations")
        self._animation_scales_attr = animation_prim.GetAttribute("scales")
        if any(
            attr is None or not attr.IsValid()
            for attr in (self._animation_rotations_attr, self._animation_translations_attr, self._animation_scales_attr)
        ):
            self._animation_root_attr = None
            self._animation_rotations_attr = None
            self._animation_translations_attr = None
            self._animation_scales_attr = None
            print(f"[hrc_humanoid] _bind: animation attributes invalid", flush=True)
            return

        # For legacy walk.usda: clear timeSamples once so we drive
        # the skeleton programmatically.
        if not is_kit_usd:
            if self._animation_root_attr and self._animation_root_attr.IsValid():
                self._animation_root_attr.Clear()
                self._animation_root_attr.Set(lazy.pxr.Gf.Matrix4d(1.0))
            for attr in (self._animation_rotations_attr, self._animation_translations_attr, self._animation_scales_attr):
                if attr and attr.IsValid():
                    attr.Clear()

        # Write initial pose.
        payload = self._get_current_animation_payload()
        if payload is not None:
            rotations_wxyz, translations, scales = payload
            self._animation_rotations_attr.Set(array_to_vtarray(rotations_wxyz.tolist(), lazy.pxr.Gf.Quatf))
            self._animation_translations_attr.Set(array_to_vtarray(translations.tolist(), lazy.pxr.Gf.Vec3f))
            self._animation_scales_attr.Set(array_to_vtarray(scales.tolist(), lazy.pxr.Gf.Vec3h))

        self._animation_handles_bound = True
        print(
            f"[hrc_humanoid] animation handles bound (once) at "
            f"{animation_prim.GetPath()} (kit_usd={is_kit_usd})",
            flush=True,
        )

    def _quat_array_xyzw_to_wxyz(self, quat_array: np.ndarray) -> np.ndarray:
        quat_xyzw = np.asarray(quat_array, dtype=np.float32)
        if quat_xyzw.ndim != 2 or quat_xyzw.shape[1] != 4:
            raise ValueError(f"Unexpected quaternion array shape: {quat_xyzw.shape}")
        return quat_xyzw[:, [3, 0, 1, 2]]

    def _apply_animation_pose(self) -> None:
        """Write the current animation frame to the cached USD attributes.

        This method is called every simulation step.  It NEVER triggers a
        rebind -- if handles are not yet bound (or were invalidated), it
        silently skips.  Binding is handled exclusively by _create_visual /
        refresh_from_scene.
        """
        if self.animation_runtime is None:
            return
        # Skip if handles have not been bound yet (will be bound on the next
        # explicit _create_visual / refresh_from_scene call).
        if not self._animation_handles_ready():
            self._animation_handles_bound = False
            return

        payload = self._get_current_animation_payload()
        if payload is None:
            return
        rotations_wxyz, translations, scales = payload
        self._animation_rotations_attr.Set(array_to_vtarray(rotations_wxyz.tolist(), lazy.pxr.Gf.Quatf))
        self._animation_translations_attr.Set(array_to_vtarray(translations.tolist(), lazy.pxr.Gf.Vec3f))
        self._animation_scales_attr.Set(array_to_vtarray(scales.tolist(), lazy.pxr.Gf.Vec3h))
        if self._animation_root_attr is not None and self._animation_root_attr.IsValid():
            self._animation_root_attr.Set(lazy.pxr.Gf.Matrix4d(1.0))

    def _advance_animation(self) -> None:
        if self.animation_runtime is None or self.animation_runtime.frame_count <= 0:
            return
        if self.linear_speed <= 1e-4 or self.motion_state != "walk":
            self.animation_state = "hold"
            return
        nominal_speed = max(self.animation_runtime.nominal_speed_mps, 1e-3)
        walk_fps = max(self.animation_runtime.walk_fps, 1e-3)
        displacement_per_frame = nominal_speed / walk_fps
        displacement_this_step = float(
            getattr(self, "_last_displacement_m", self.linear_speed * self.step_dt)
        )
        frames_advanced = displacement_this_step / max(displacement_per_frame, 1e-6)
        self.animation_frame_cursor = (self.animation_frame_cursor + frames_advanced) % float(
            self.animation_runtime.frame_count
        )
        self.animation_state = "walk"

    def refresh_from_scene(self) -> None:
        visual_recreated = False
        if self.animation_runtime is None:
            self.visual = self.env.scene.object_registry("name", self.name)
        else:
            existing_visual = self.visual
            if existing_visual is not None and hasattr(existing_visual, "is_valid") and existing_visual.is_valid():
                self.visual = existing_visual
            else:
                parent_prim_path = f"{self.env.scene._scene_prim.prim_path}{self.visual_relative_prim_path}"
                if lazy.isaacsim.core.utils.prims.is_prim_path_valid(parent_prim_path):
                    visual = ReferencedHumanVisual(
                        relative_prim_path=self.visual_relative_prim_path,
                        name=self.name,
                        usd_path=download_human_usd(self.visual_asset_path),
                        load_config=None if self.scale is None else {"scale": self.scale},
                    )
                    visual.load(self.env.scene)
                    visual.initialize()
                    self._ensure_kit_usd()
                    self.visual = visual
                    visual_recreated = True
                else:
                    self.visual = None
                    self._animation_handles_bound = False
        refreshed: Dict[str, Any] = {}
        for proxy_name in list(self.proxy_objects.keys()):
            obj = self.env.scene.object_registry("name", proxy_name)
            if obj is not None:
                refreshed[proxy_name] = obj
        self.proxy_objects = refreshed
        # Only force-rebind animation handles when the visual prim was actually
        # destroyed and re-created.  For the common case (visual still alive
        # after load_state), this is a no-op, which eliminates the per-frame
        # Clear() that caused the "soft body" collapse.
        if self.animation_runtime is not None and self.visual is not None and (visual_recreated or not self._animation_handles_ready()):
            self._bind_animation_handles(force=True)

    def _ensure_kit_usd(self) -> Optional[Path]:
        """Ensure a Kit-converted USD exists for the current human asset.

        If ``_kit_converted.usd`` already exists next to the source files,
        return its path.  Otherwise, convert the source GLB on-the-fly using
        ``omni.kit.asset_converter`` (requires Isaac Sim to be running).

        Returns the path to the Kit-converted USD, or None on failure.
        """
        usd_dir = Path(self.visual_asset_path).resolve().parent
        kit_usd = usd_dir / "_kit_converted.usd"

        if kit_usd.exists() and kit_usd.stat().st_size > 0:
            return kit_usd

        # Locate source GLB via metadata.
        source_json_path = usd_dir / "source_asset.json"
        if not source_json_path.exists():
            print("[_ensure_kit_usd] WARNING: source_asset.json not found", flush=True)
            return None
        meta = json.loads(source_json_path.read_text(encoding="utf-8"))
        glb_path = meta.get("source_glb")
        if not glb_path or not Path(glb_path).exists():
            print(f"[_ensure_kit_usd] WARNING: source GLB not found: {glb_path}", flush=True)
            return None

        # Convert GLB → USD via Kit converter.
        print(f"[_ensure_kit_usd] Converting GLB: {glb_path}", flush=True)
        import omni.kit.app
        import omni.kit.asset_converter as asset_converter
        from omni.kit.async_engine import run_coroutine

        ctx = asset_converter.AssetConverterContext()
        ctx.ignore_camera = True
        ctx.ignore_light = True
        ctx.ignore_animations = True
        ctx.keep_all_materials = True
        ctx.merge_all_meshes = False
        ctx.convert_stage_up_z = True

        mgr = asset_converter.get_instance()
        if mgr is None:
            print("[_ensure_kit_usd] WARNING: asset_converter unavailable", flush=True)
            return None
        task = mgr.create_converter_task(str(glb_path), str(kit_usd), None, ctx)
        future = run_coroutine(task.wait_until_finished())
        app = omni.kit.app.get_app()
        while not future.done():
            app.update()
        if not future.result() or not kit_usd.exists():
            err = task.get_error_message() if hasattr(task, "get_error_message") else "unknown"
            print(f"[_ensure_kit_usd] WARNING: conversion failed: {err}", flush=True)
            return None
        print(f"[_ensure_kit_usd] Done: {kit_usd}", flush=True)
        return kit_usd

    def _create_visual(self, position: np.ndarray, yaw: float) -> Any:
        quat = self._visual_orientation_from_yaw(float(yaw))
        if self.animation_runtime is not None:
            existing = self.visual
            if existing is None or not hasattr(existing, "is_valid") or not existing.is_valid():
                self.refresh_from_scene()
                existing = self.visual
            if existing is None:
                # Ensure Kit-converted USD exists (run converter if needed).
                self._ensure_kit_usd()
                human = ReferencedHumanVisual(
                    relative_prim_path=self.visual_relative_prim_path,
                    name=self.name,
                    usd_path=download_human_usd(self.visual_asset_path),
                    load_config=None if self.scale is None else {"scale": self.scale},
                )
                human.load(self.env.scene)
                human.initialize()
                self.visual = human
                existing = human
            existing.set_position_orientation(position=position, orientation=quat)
            self.visual = existing
            # --- diagnostic: verify visual prim state ---
            try:
                p = existing.prim
                actual_pos = existing.get_position_orientation()
                sc = existing.scale
                print(f"[_create_visual] prim={p.GetPath()} type={p.GetTypeName()} "
                      f"pos={actual_pos[0]} scale={sc}", flush=True)
                asset_path = getattr(existing, "asset_prim_path", None)
                if asset_path:
                    from pxr import UsdGeom
                    ap = lazy.isaacsim.core.utils.prims.get_prim_at_path(asset_path)
                    if ap and ap.IsValid():
                        img = UsdGeom.Imageable(ap)
                        vis = img.ComputeVisibility()
                        purpose = img.GetPurposeAttr().Get() if img.GetPurposeAttr().IsValid() else "N/A"
                        print(f"[_create_visual] asset_prim={asset_path} type={ap.GetTypeName()} "
                              f"vis={vis} purpose={purpose}", flush=True)
            except Exception as diag_e:
                print(f"[_create_visual] diag error: {diag_e}", flush=True)
            if not self._animation_handles_ready():
                self._bind_animation_handles(force=True)
            return existing

        existing = self.env.scene.object_registry("name", self.name)
        if existing is not None:
            existing.set_position_orientation(position=position, orientation=quat)
            self.visual = existing
            self._bind_animation_handles(force=True)
            return existing
        human = USDObject(
            name=self.name,
            usd_path=download_human_usd(self.visual_asset_path),
            category="human",
            scale=self.scale,
            visual_only=True,
            position=position,
            orientation=quat,
        )
        self.env.scene.add_object(human)
        human.set_position_orientation(position=position, orientation=quat)
        self.visual = human
        self._bind_animation_handles(force=True)
        return human

    def _create_proxy_object(
        self,
        proxy_name: str,
        primitive_type: str,
        radius: float,
        height: Optional[float],
        position: np.ndarray,
    ) -> Any:
        obj = self.env.scene.object_registry("name", proxy_name)
        if obj is None:
            obj = PrimitiveObject(
                name=proxy_name,
                primitive_type=primitive_type,
                category="human_proxy",
                visible=False,
                # fixed_base MUST be True for a kinematic_only collider: object_base
                # ._post_load only resolves kinematic_only=True under fixed_base=True
                # (else asserts "Kinematic only decision does not match!"). Mirrors
                # OmniGibson's own kinematic_only particle templates (system_base.py).
                # fixed_base + kinematic_only skips the world rootJoint, so the
                # per-frame set_position_orientation in _sync_proxies still moves it.
                fixed_base=True,
                visual_only=False,
                kinematic_only=True,
                position=position,
                orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                radius=radius,
                height=height,
            )
            self.env.scene.add_object(obj)
        obj.set_position_orientation(position=position, orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        # Force-hide the collision-only proxy. It has no PBR material, so if it
        # renders it shows as a magenta/purple blob on the human (log warns "Could
        # not find PBR material for ...human_proxy"). `obj.visible = False` proved
        # INSUFFICIENT for a kinematic_only PrimitiveObject (still rendered), so
        # also force USD-level invisibility on the prim subtree (hierarchical ->
        # hides the visual mesh). load_state doesn't touch USD visibility, so once holds.
        try:
            obj.visible = False
        except Exception:
            pass
        try:
            _pp = getattr(obj, "prim_path", None)
            _prim = og.sim.stage.GetPrimAtPath(_pp) if _pp else None
            if _prim is not None and _prim.IsValid():
                lazy.pxr.UsdGeom.Imageable(_prim).MakeInvisible()
        except Exception as _exc:
            print(f"[proxy] USD MakeInvisible failed for {proxy_name}: {_exc}", flush=True)
        try:
            obj.mass = self.collision_proxy.mass_kg / 3.0
        except Exception:
            pass
        self.proxy_objects[proxy_name] = obj
        return obj

    def _ensure_collision_proxies(self) -> None:
        if not self.collision_proxy.enabled:
            return
        lower_center, torso_center, head_center = self._proxy_centers()
        self._create_proxy_object(
            proxy_name=f"{self.name}_proxy_lower",
            primitive_type="Cylinder",
            radius=self.collision_proxy.lower_body_radius,
            height=self.collision_proxy.lower_body_height,
            position=lower_center,
        )
        self._create_proxy_object(
            proxy_name=f"{self.name}_proxy_torso",
            primitive_type="Cylinder",
            radius=self.collision_proxy.torso_radius,
            height=self.collision_proxy.torso_height,
            position=torso_center,
        )
        self._create_proxy_object(
            proxy_name=f"{self.name}_proxy_head",
            primitive_type="Sphere",
            radius=self.collision_proxy.head_radius,
            height=None,
            position=head_center,
        )

    def _proxy_centers(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lower_center = self.base_pos + np.array([0.0, 0.0, self.collision_proxy.lower_body_height / 2.0], dtype=np.float32)
        torso_bottom = self.collision_proxy.lower_body_height
        torso_center = self.base_pos + np.array(
            [0.0, 0.0, torso_bottom + self.collision_proxy.torso_height / 2.0],
            dtype=np.float32,
        )
        head_center = self.base_pos + np.array(
            [
                0.0,
                0.0,
                torso_bottom + self.collision_proxy.torso_height + self.collision_proxy.head_radius * 1.2,
            ],
            dtype=np.float32,
        )
        return lower_center, torso_center, head_center

    def _sync_proxies(self) -> None:
        if not self.proxy_objects:
            return
        # D2 (protect task replay): while PARKED (set during ExecuteBehaviorTask), sink
        # the proxies far below the floor so the kinematic human can never shove a task
        # object (soda can / trash can) mid-replay and corrupt the recorded state. The
        # replay load_state is authoritative there (robot included), so the proxy has no
        # useful job during the task — only the risk of a bad contact.
        if getattr(self, "_proxies_parked", False):
            for obj in self.proxy_objects.values():
                if obj is not None:
                    obj.set_position_orientation(
                        position=np.array([0.0, 0.0, -50.0], dtype=np.float32),
                        orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            return
        lower_center, torso_center, head_center = self._proxy_centers()
        placements = {
            f"{self.name}_proxy_lower": lower_center,
            f"{self.name}_proxy_torso": torso_center,
            f"{self.name}_proxy_head": head_center,
        }
        for proxy_name, position in placements.items():
            obj = self.proxy_objects.get(proxy_name)
            if obj is not None:
                obj.set_position_orientation(
                    position=position,
                    orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                )

    def spawn_near_robot(self) -> Any:
        position, yaw, spawn_debug = self._compute_spawn_pose(distance=self.distance)
        self.base_pos = position.astype(np.float32)
        self.base_yaw = float(yaw)
        self.anchor_pos = self.base_pos.copy()
        self.anchor_yaw = self.base_yaw
        visual = self._create_visual(position=self.base_pos, yaw=self.base_yaw)
        setattr(visual, "_hrc_humanoid_spawn_debug", spawn_debug)
        self._ensure_collision_proxies()
        self.apply_pose()
        self.set_phase("WaitNearHuman")
        return visual

    def set_phase(self, phase: str) -> None:
        phase = str(phase)
        if phase == self.active_phase:
            # DIAG: log early-return path at low frequency so we can detect
            # unexpected re-entries in steady state (should never mutate state)
            if getattr(self, "_diag_set_phase_calls", 0) < 5 or self.phase_step % 30 == 0:
                print(
                    f"[diag] set_phase({phase}) EARLY-RETURN "
                    f"active={self.active_phase} phase_step={self.phase_step} "
                    f"idx={self.phase_waypoint_idx}/{len(self.phase_waypoints)}",
                    flush=True,
                )
            self._diag_set_phase_calls = getattr(self, "_diag_set_phase_calls", 0) + 1
            return
        print(
            f"[diag] set_phase({phase}) RESET old_active={self.active_phase} "
            f"old_idx={self.phase_waypoint_idx}/{len(self.phase_waypoints)} "
            f"old_phase_step={self.phase_step}",
            flush=True,
        )
        self.active_phase = phase
        self.phase_step = 0
        self.phase_waypoint_idx = 0
        self.phase_waypoints = self._resolve_phase_waypoints(self._get_phase_spec(phase))
        print(
            f"[diag] set_phase({phase}) POST-RESET waypoints_len={len(self.phase_waypoints)}",
            flush=True,
        )

    def _try_get_trav_map(self):
        """Return the scene's TraversableMap if available, else None."""
        try:
            scene = og.sim.scenes[0] if og.sim.scenes else None
            if scene is not None and hasattr(scene, "trav_map"):
                tmap = scene.trav_map
                if tmap is not None and tmap.floor_map is not None and len(tmap.floor_map) > 0:
                    return tmap
        except Exception:
            pass
        return None

    def bake_rects_into_trav_map(self, rects, floor: int = 0, margin_m: float = 0.0) -> int:
        """Burn world-frame furniture AABB rects (xmin,ymin,xmax,ymax) into the
        trav_map ``floor_map`` as NON-traversable.

        The base trav_map is FLOOR-level and omits table/cabinet BODIES (the floor
        under/around them reads 'walkable'), so the human — which navigates purely
        on trav_map (spawn snap + A*) — used to spawn / stand / walk INSIDE
        furniture that the ROBOT correctly avoids via inflated AABB rects. Baking
        the robot's same furniture set here unifies the two obstacle models. The
        per-path erosion (``nav_erosion_radius`` ~0.35m) still applies on top, so
        the human keeps a body-width margin from the newly-blocked cells. Returns
        the number of cells blocked."""
        trav_map = self._try_get_trav_map()
        if trav_map is None or not rects:
            return 0
        try:
            fmap = trav_map.floor_map[floor]
            H, W = int(fmap.shape[0]), int(fmap.shape[1])
        except Exception:
            return 0
        n_cells = 0
        for rect in rects:
            x0, y0, x1, y1 = (float(rect[0]) - margin_m, float(rect[1]) - margin_m,
                              float(rect[2]) + margin_m, float(rect[3]) + margin_m)
            iis, jjs = [], []
            for cx, cy in ((x0, y0), (x0, y1), (x1, y0), (x1, y1)):
                try:
                    ij = trav_map.world_to_map(th.tensor([cx, cy], dtype=th.float32)).tolist()
                    iis.append(int(ij[0]))
                    jjs.append(int(ij[1]))
                except Exception:
                    pass
            if len(iis) < 4:
                continue
            i_lo, i_hi = max(min(iis), 0), min(max(iis), H - 1)
            j_lo, j_hi = max(min(jjs), 0), min(max(jjs), W - 1)
            if i_lo > i_hi or j_lo > j_hi:
                continue
            fmap[i_lo:i_hi + 1, j_lo:j_hi + 1] = 0
            n_cells += (i_hi - i_lo + 1) * (j_hi - j_lo + 1)
        print(f"[trav-bake] baked {len(rects)} furniture rects into human trav_map "
              f"({n_cells} cells non-traversable)", flush=True)
        return n_cells

    def respawn_if_in_obstacle(self, floor: int = 0) -> bool:
        """If the human's current base sits in a (newly-baked) non-traversable
        cell, re-run the validated spawn so it lands on clear floor connected to
        the robot. Call AFTER :meth:`bake_rects_into_trav_map`. Re-applies pose
        (which re-syncs the collision proxies). Returns True if the human moved."""
        trav_map = self._try_get_trav_map()
        if trav_map is None:
            return False
        try:
            xy = self.base_pos[:2].astype(np.float32)
            ij = trav_map.world_to_map(th.tensor([float(xy[0]), float(xy[1])], dtype=th.float32)).tolist()
            fmap = trav_map.floor_map[floor]
            H, W = int(fmap.shape[0]), int(fmap.shape[1])
            i, j = int(ij[0]), int(ij[1])
            if 0 <= i < H and 0 <= j < W and int(fmap[i, j]) > 0:
                return False  # already on clear floor
        except Exception:
            return False
        try:
            pos, yaw, _dbg = self._compute_spawn_pose(self.distance)
        except Exception as exc:
            print(f"[spawn-fix] respawn failed: {exc}", flush=True)
            return False
        old = self.base_pos[:2].copy()
        self.base_pos = pos.astype(np.float32)
        self.base_yaw = float(yaw)
        self.anchor_pos = self.base_pos.copy()
        self.anchor_yaw = self.base_yaw
        self.apply_pose()
        print(f"[spawn-fix] human spawned inside furniture ({old[0]:.2f},{old[1]:.2f}); "
              f"re-snapped to clear floor ({self.base_pos[0]:.2f},{self.base_pos[1]:.2f})", flush=True)
        return True

    def _clearance_ok(self, xy: np.ndarray, trav_map, floor: int = 0,
                      min_clearance: float = 0.5) -> bool:
        """True iff every trav_map cell within *min_clearance* metres of (x,y) is
        traversable.

        The existing spawn checks (snap-didn't-jump + connected-to-robot + same-room)
        all PASS for a human standing 3cm from a stove — a cell can be "navigable"
        while hugging furniture (ep0005: feet clipped into the stove; ep0002/3/4 spawned
        0.10-0.24m from tables/stairs). Measured on the 8-episode batch: every episode
        the user judged BAD had spawn clearance <= 0.24m, the good ones 0.61-1.19m.
        This adds the missing breathing room, and also stops Relocate from starting
        inside a pocket the trailing robot can't follow out of."""
        if trav_map is None or min_clearance <= 0:
            return True
        try:
            fmap = trav_map.floor_map[int(floor)]
            H, W = int(fmap.shape[0]), int(fmap.shape[1])
            res = float(getattr(trav_map, "map_resolution", 0.1)) or 0.1
            ij = trav_map.world_to_map(
                th.tensor([float(xy[0]), float(xy[1])], dtype=th.float32)).tolist()
            i0, j0 = int(ij[0]), int(ij[1])
            rad = max(int(round(float(min_clearance) / res)), 1)
            r2 = rad * rad
            for di in range(-rad, rad + 1):
                for dj in range(-rad, rad + 1):
                    if di * di + dj * dj > r2:
                        continue
                    i, j = i0 + di, j0 + dj
                    if not (0 <= i < H and 0 <= j < W):
                        return False
                    if int(fmap[i, j]) <= 0:
                        return False
            return True
        except Exception:
            return True   # cannot validate -> don't block the spawn

    def _snap_to_navigable(self, xy: np.ndarray, trav_map, floor: int = 0, max_radius_px: int = 40):
        """Snap a world (x,y) point to the nearest navigable cell on trav_map.

        Returns a torch.Tensor world-frame (x,y) that is guaranteed to lie in a
        navigable pixel, or None if no navigable pixel is found within
        max_radius_px of the requested location.
        """
        try:
            floor = int(floor)   # defensive: a float here -> floor_map[float] TypeError
            xy_2d = th.tensor([float(xy[0]), float(xy[1])], dtype=th.float32)
            map_ij = trav_map.world_to_map(xy_2d).tolist()
            fmap = trav_map.floor_map[floor]
            H, W = int(fmap.shape[0]), int(fmap.shape[1])
            i0, j0 = int(map_ij[0]), int(map_ij[1])
            if 0 <= i0 < H and 0 <= j0 < W and int(fmap[i0, j0]) > 0:
                return xy_2d
            best = None
            best_d2 = None
            for r in range(1, max_radius_px + 1):
                found = False
                for di in range(-r, r + 1):
                    for dj in range(-r, r + 1):
                        if max(abs(di), abs(dj)) != r:
                            continue
                        i, j = i0 + di, j0 + dj
                        if not (0 <= i < H and 0 <= j < W):
                            continue
                        if int(fmap[i, j]) <= 0:
                            continue
                        d2 = di * di + dj * dj
                        if best_d2 is None or d2 < best_d2:
                            best = (i, j)
                            best_d2 = d2
                        found = True
                if found:
                    break
            if best is None:
                return None
            # float32 map_to_world convention (matches the banded sampler ~line 2448).
            snapped = to_numpy(trav_map.map_to_world(
                th.tensor([[float(best[0]), float(best[1])]], dtype=th.float32)))[0]
            return th.tensor([float(snapped[0]), float(snapped[1])], dtype=th.float32)
        except Exception as exc:
            print(f"[hrc_humanoid] snap_to_navigable failed: {exc}", flush=True)
            return None

    def _nav_path_between(
        self, src_xy: np.ndarray, dst_xy: np.ndarray, z: float, trav_map, floor: int = 0,
        erosion_radius: Optional[float] = None,
    ) -> List[np.ndarray]:
        """Compute A* path between two 2D world points using the trav map.

        Endpoints are first snapped to the nearest navigable pixel so that points
        lying on walls / in doorways / eroded cells still produce a valid path.
        Returns a list of 3D waypoints (including the destination).  If no
        navigable path is found, returns a single-element list with just the
        destination (straight-line fallback).

        Args:
            erosion_radius: extra erosion (meters) applied to the trav map before
                planning, giving the human a safety margin from furniture / walls.
                Defaults to ``self.nav_erosion_radius`` (0.3 m).
                Falls back to 0 erosion if the eroded map makes the path impossible.
        """
        if erosion_radius is None:
            erosion_radius = self.nav_erosion_radius
        try:
            src_snap = self._snap_to_navigable(np.asarray(src_xy, dtype=np.float32), trav_map, floor)
            dst_snap = self._snap_to_navigable(np.asarray(dst_xy, dtype=np.float32), trav_map, floor)
            if src_snap is None or dst_snap is None:
                print(
                    f"[hrc_humanoid] A* endpoint not navigable: "
                    f"src={src_xy[:2].tolist()} (snap={src_snap}) "
                    f"dst={dst_xy[:2].tolist()} (snap={dst_snap})",
                    flush=True,
                )
                return [np.array([float(dst_xy[0]), float(dst_xy[1]), z], dtype=np.float32)]

            path_world = None

            # Try with extra erosion first (wider clearance from obstacles).
            if erosion_radius > 0:
                saved_radius = trav_map.default_erosion_radius
                try:
                    trav_map.default_erosion_radius = erosion_radius
                    path_world, _ = trav_map.get_shortest_path(
                        floor=floor, source_world=src_snap, target_world=dst_snap,
                        entire_path=True,
                    )
                finally:
                    trav_map.default_erosion_radius = saved_radius

            # Fallback: plan without extra erosion if eroded map had no path.
            if path_world is None or len(path_world) == 0:
                path_world, _ = trav_map.get_shortest_path(
                    floor=floor, source_world=src_snap, target_world=dst_snap,
                    entire_path=True,
                )

            if path_world is not None and len(path_world) > 0:
                pts = []
                for pt in path_world:
                    pts.append(np.array([float(pt[0]), float(pt[1]), z], dtype=np.float32))
                return pts
            print(
                f"[hrc_humanoid] A* returned no path: "
                f"src_snap={src_snap.tolist()} dst_snap={dst_snap.tolist()}",
                flush=True,
            )
        except Exception as exc:
            print(f"[hrc_humanoid] A* path failed: {exc}", flush=True)
        return [np.array([float(dst_xy[0]), float(dst_xy[1]), z], dtype=np.float32)]

    def _resolve_phase_waypoints(self, spec: HumanPhaseMotion) -> List[np.ndarray]:
        if spec.waypoints:
            raw = [np.asarray(p, dtype=np.float32) for p in spec.waypoints]
            if not spec.expand_with_astar:
                return raw
            trav_map = self._try_get_trav_map()
            if trav_map is None:
                return raw
            floor = 0
            z = float(self.base_pos[2])
            expanded: List[np.ndarray] = []
            src = self.base_pos.copy()
            for dst in raw:
                seg = self._nav_path_between(src, dst, z, trav_map, floor)
                expanded.extend(seg)
                src = dst
            if spec.loop and len(raw) >= 2:
                seg = self._nav_path_between(raw[-1], raw[0], z, trav_map, floor)
                expanded.extend(seg)
            if expanded:
                print(
                    f"[hrc_humanoid] user waypoints expanded via A*: "
                    f"{len(raw)} raw -> {len(expanded)} nav points",
                    flush=True,
                )
                return expanded
            return raw
        if spec.mode not in {"waypoint_loop", "waypoint_once"}:
            return []

        radius = float(spec.patrol_radius)
        cx, cy, cz = self.base_pos.tolist()

        # Fallback diamond waypoints (no nav awareness)
        fallback = [
            np.array([cx + radius, cy, cz], dtype=np.float32),
            np.array([cx, cy + radius, cz], dtype=np.float32),
            np.array([cx - radius, cy, cz], dtype=np.float32),
            np.array([cx, cy - radius, cz], dtype=np.float32),
        ]

        trav_map = self._try_get_trav_map()
        if trav_map is None:
            return fallback

        # --- Sample navigable patrol targets ---
        floor = 0
        ref_point = th.tensor([cx, cy, cz], dtype=th.float32)
        patrol_targets: List[np.ndarray] = []
        n_targets = 4
        for _ in range(n_targets):
            try:
                _, rand_pt = trav_map.get_random_point(
                    floor=floor, reference_point=ref_point,
                )
                rp = np.array([float(rand_pt[0]), float(rand_pt[1]), cz], dtype=np.float32)
                # Clamp distance to patrol_radius
                delta = rp[:2] - np.array([cx, cy], dtype=np.float32)
                dist = float(np.linalg.norm(delta))
                if dist > radius:
                    rp[:2] = np.array([cx, cy], dtype=np.float32) + delta / dist * radius
                patrol_targets.append(rp)
            except Exception:
                pass
        if len(patrol_targets) < 2:
            return fallback

        # --- Build full nav-aware path through all patrol targets ---
        nav_waypoints: List[np.ndarray] = []
        src = self.base_pos.copy()
        for target in patrol_targets:
            segment = self._nav_path_between(src, target, cz, trav_map, floor)
            nav_waypoints.extend(segment)
            src = target
        # Close the loop: path from last target back to first
        if spec.loop and len(patrol_targets) >= 2:
            segment = self._nav_path_between(patrol_targets[-1], patrol_targets[0], cz, trav_map, floor)
            nav_waypoints.extend(segment)

        if len(nav_waypoints) < 2:
            return fallback

        print(
            f"[hrc_humanoid] nav waypoints: {len(nav_waypoints)} points "
            f"({len(patrol_targets)} patrol targets, trav_map avail)",
            flush=True,
        )
        return nav_waypoints

    def reset_anchor_to_current_pose(self) -> None:
        self.anchor_pos = self.base_pos.copy()
        self.anchor_yaw = float(self.base_yaw)

    def set_waypoints(
        self,
        points: Sequence[Sequence[float]],
        mode: str = "waypoint_once",
        phase: Optional[str] = None,
        loop: bool = False,
        expand_with_astar: bool = True,
        face_robot: bool = False,
    ) -> None:
        phase = phase or self.active_phase
        base = self._get_phase_spec(phase)
        new_spec = replace(
            base,
            mode=mode,
            loop=loop,
            expand_with_astar=expand_with_astar,
            face_robot=face_robot,
            waypoints=[list(map(float, p)) for p in points],
        )
        self.phase_motion[phase] = new_spec
        self.phase_waypoints = self._resolve_phase_waypoints(new_spec)
        self.phase_waypoint_idx = 0
        print(
            f"[diag] set_waypoints phase={phase} mode={mode} loop={loop} "
            f"n_raw={len(points)} n_resolved={len(self.phase_waypoints)} "
            f"active_phase={self.active_phase} phase_step={self.phase_step}",
            flush=True,
        )

    def goto(self, x: float, y: float, phase: Optional[str] = None) -> None:
        self.set_waypoints(
            [[float(x), float(y), float(self.base_pos[2])]],
            mode="waypoint_once",
            phase=phase,
            loop=False,
        )

    def apply_pose(self) -> None:
        if self.visual is None or (self.animation_runtime is not None and not self._animation_handles_ready()):
            self.refresh_from_scene()
        if self.animation_runtime is not None and self.visual is not None and not self._animation_handles_ready():
            self._bind_animation_handles(force=True)
        if self.visual is None:
            raise RuntimeError("Humanoid visual asset is not available in the scene.")
        quat = self._visual_orientation_from_yaw(float(self.base_yaw))
        self.visual.set_position_orientation(position=self.base_pos, orientation=quat)
        self._apply_animation_pose()
        self._sync_proxies()

    def sync_after_replay_load(self) -> None:
        # Replay restores robot / scene state, so the humanoid pose must be re-applied after each load_state call.
        # DIAG: snapshot base_pos/yaw/idx before+after to detect any accidental mutation in this path
        _diag = self.phase_step < 5 or self.phase_step % 10 == 0
        if _diag:
            print(
                f"[diag] sync_after_replay_load BEFORE phase_step={self.phase_step} "
                f"idx={self.phase_waypoint_idx}/{len(self.phase_waypoints)} "
                f"base_pos=({float(self.base_pos[0]):.3f},{float(self.base_pos[1]):.3f}) "
                f"base_yaw={math.degrees(float(self.base_yaw)):.1f}deg "
                f"active_phase={self.active_phase}",
                flush=True,
            )
        self.refresh_from_scene()
        self.apply_pose()
        if _diag:
            print(
                f"[diag] sync_after_replay_load AFTER  phase_step={self.phase_step} "
                f"idx={self.phase_waypoint_idx}/{len(self.phase_waypoints)} "
                f"base_pos=({float(self.base_pos[0]):.3f},{float(self.base_pos[1]):.3f}) "
                f"base_yaw={math.degrees(float(self.base_yaw)):.1f}deg "
                f"active_phase={self.active_phase}",
                flush=True,
            )

    def _advance_towards_waypoint(self, spec: HumanPhaseMotion) -> None:
        self._last_displacement_m = 0.0
        # DIAG: capture entry state for steps where we'll also print summary
        diag_enabled = self.phase_step < 5 or self.phase_step % 10 == 0
        idx_before = int(self.phase_waypoint_idx)
        if diag_enabled:
            print(
                f"[diag] _advance ENTRY phase_step={self.phase_step} "
                f"idx={idx_before}/{len(self.phase_waypoints)} "
                f"base_pos=({self.base_pos[0]:.3f},{self.base_pos[1]:.3f}) "
                f"base_yaw={math.degrees(float(self.base_yaw)):.1f}deg "
                f"dist_thresh={float(spec.dist_thresh):.3f} turn_thresh={float(spec.turn_thresh_deg):.1f}",
                flush=True,
            )
        if not self.phase_waypoints:
            self.linear_speed = 0.0
            self.motion_state = "hold"
            if diag_enabled:
                print("[diag] _advance EXIT empty-waypoints", flush=True)
            return

        target = self.phase_waypoints[self.phase_waypoint_idx]
        delta = target - self.base_pos
        delta[2] = 0.0
        dist = float(np.linalg.norm(delta[:2]))

        desired_yaw = (
            math.atan2(float(delta[1]), float(delta[0])) if dist > 1e-6 else float(self.base_yaw)
        )
        yaw_err = _wrap_angle(desired_yaw - float(self.base_yaw))
        yaw_err_deg = math.degrees(yaw_err)

        is_last = self.phase_waypoint_idx + 1 >= len(self.phase_waypoints)
        arrived = dist < float(spec.dist_thresh)
        outer_cond = arrived and (not is_last or abs(yaw_err_deg) < float(spec.turn_thresh_deg))
        if diag_enabled:
            print(
                f"[diag] _advance CHECK target=({float(target[0]):.3f},{float(target[1]):.3f}) "
                f"dist={dist:.4f} yaw_err_deg={yaw_err_deg:.1f} "
                f"is_last={is_last} arrived={arrived} outer_cond={outer_cond}",
                flush=True,
            )
        if outer_cond:
            if not is_last:
                self.phase_waypoint_idx += 1
                if diag_enabled:
                    print(
                        f"[diag] _advance INCREMENTED idx {idx_before} -> {self.phase_waypoint_idx}",
                        flush=True,
                    )
            elif spec.loop:
                self.phase_waypoint_idx = 0
                if diag_enabled:
                    print(
                        f"[diag] _advance LOOP-RESET idx {idx_before} -> 0",
                        flush=True,
                    )
            else:
                self.linear_speed = 0.0
                self.motion_state = "hold"
                if diag_enabled:
                    print(
                        f"[diag] _advance DONE (last, no loop) idx={self.phase_waypoint_idx}",
                        flush=True,
                    )
                return

        max_step_rad = math.radians(float(spec.turning_step_amount_deg))
        if abs(yaw_err) <= math.radians(float(spec.min_angle_turn_deg)):
            self.base_yaw = desired_yaw
        else:
            self.base_yaw = _wrap_angle(
                float(self.base_yaw) + math.copysign(min(abs(yaw_err), max_step_rad), yaw_err)
            )

        forward = np.array(
            [math.cos(float(self.base_yaw)), math.sin(float(self.base_yaw)), 0.0],
            dtype=np.float32,
        )
        step_len = min(
            float(spec.speed) * float(spec.distance_multiplier) * self.step_dt,
            dist,
        )
        self.base_pos = self.base_pos + forward * step_len
        self.linear_speed = step_len / max(self.step_dt, 1e-6)
        self._last_displacement_m = float(step_len)
        self.motion_state = "walk"
        if diag_enabled:
            print(
                f"[diag] _advance EXIT idx_final={self.phase_waypoint_idx} "
                f"base_pos=({self.base_pos[0]:.3f},{self.base_pos[1]:.3f}) "
                f"base_yaw={math.degrees(float(self.base_yaw)):.1f}deg "
                f"step_len={step_len:.4f}",
                flush=True,
            )
        if self.phase_step % 30 == 0:
            print(
                f"[hrc_humanoid] step={self.phase_step} idx={self.phase_waypoint_idx}/"
                f"{len(self.phase_waypoints)} pos=({self.base_pos[0]:.2f},{self.base_pos[1]:.2f}) "
                f"yaw={math.degrees(float(self.base_yaw)):.1f}deg dist={dist:.2f} "
                f"err={yaw_err_deg:.1f}deg step_len={step_len:.3f}",
                flush=True,
            )

    def _face_robot(self, robot: Any) -> None:
        robot_pos, _ = robot.get_position_orientation()
        robot_pos_np = to_numpy(robot_pos).astype(np.float32)
        delta = robot_pos_np - self.base_pos
        self.base_yaw = math.atan2(float(delta[1]), float(delta[0]))
        if self.motion_state != "walk":
            self.motion_state = "turn"

    def step(self, phase: str, robot: Any) -> HumanActorSnapshot:
        # DIAG: tag each high-level step() entry so we can correlate the log with pre_step_callback order
        if self.phase_step < 5 or self.phase_step % 10 == 0:
            print(
                f"[diag] step() ENTRY requested_phase={phase} active_phase={self.active_phase} "
                f"phase_step={self.phase_step} idx={self.phase_waypoint_idx}/{len(self.phase_waypoints)}",
                flush=True,
            )
        self.set_phase(phase)
        spec = self._get_phase_spec(phase)
        self.linear_speed = 0.0
        self.motion_state = "hold"

        if spec.mode in {"waypoint_loop", "waypoint_once"}:
            self._advance_towards_waypoint(spec)
        else:
            # Keep the current base pose and only update orientation if needed.
            self.motion_state = "hold"

        if spec.face_robot:
            self._face_robot(robot)
            if self.linear_speed <= 1e-6:
                self.linear_speed = 0.0

        self._advance_animation()
        # Tie synthetic keypoint swing to actual ground displacement so arm/leg
        # swing stays in lockstep with feet regardless of distance_multiplier.
        self.gait_phase += max(float(getattr(self, "_last_displacement_m", 0.0)), 0.0) * 2.0
        self.phase_step += 1
        self.apply_pose()
        return self.snapshot(phase)

    def snapshot(self, phase: Optional[str] = None) -> HumanActorSnapshot:
        phase = phase or self.active_phase
        quat = quat_from_yaw(float(self.base_yaw))
        keypoints = self.get_keypoints_world()
        motion_state = "walk" if self.linear_speed > 1e-3 else self.motion_state
        if motion_state not in HUMAN_MOTION_STATE_IDS:
            motion_state = "hold"
        return HumanActorSnapshot(
            phase=phase,
            motion_state=motion_state,
            motion_state_id=HUMAN_MOTION_STATE_IDS[motion_state],
            base_pos=self.base_pos.copy(),
            base_quat=quat.copy(),
            linear_speed=float(self.linear_speed),
            keypoints_world=keypoints,
        )

    def get_position_orientation(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.base_pos.copy(), quat_from_yaw(float(self.base_yaw))

    def get_keypoints_world(self) -> Dict[str, np.ndarray]:
        profile = self.body_profile
        gait = self.gait_phase
        forward = np.array([math.cos(self.base_yaw), math.sin(self.base_yaw), 0.0], dtype=np.float32)
        left = np.array([-forward[1], forward[0], 0.0], dtype=np.float32)

        def _world(local: Sequence[float]) -> np.ndarray:
            local = np.asarray(local, dtype=np.float32)
            return self.base_pos + forward * local[0] + left * local[1] + np.array([0.0, 0.0, local[2]], dtype=np.float32)

        pelvis = _world([0.0, 0.0, profile.pelvis_height])
        spine = _world([0.0, 0.0, profile.spine_height])
        neck = _world([0.0, 0.0, profile.neck_height])
        head = _world([0.0, 0.0, profile.head_height])

        arm_swing = math.sin(gait) * profile.arm_swing_amplitude
        leg_swing = math.sin(gait) * profile.step_swing_amplitude

        left_shoulder = _world([0.0, profile.shoulder_width / 2.0, profile.neck_height - 0.02])
        right_shoulder = _world([0.0, -profile.shoulder_width / 2.0, profile.neck_height - 0.02])
        left_elbow = _world([arm_swing * 0.5, profile.shoulder_width / 2.0 + 0.03, profile.neck_height - profile.upper_arm_length])
        right_elbow = _world([-arm_swing * 0.5, -profile.shoulder_width / 2.0 - 0.03, profile.neck_height - profile.upper_arm_length])
        left_hand = _world([arm_swing, profile.shoulder_width / 2.0 + 0.05, profile.neck_height - profile.upper_arm_length - profile.lower_arm_length])
        right_hand = _world([-arm_swing, -profile.shoulder_width / 2.0 - 0.05, profile.neck_height - profile.upper_arm_length - profile.lower_arm_length])

        left_hip = _world([0.0, profile.hip_width / 2.0, profile.pelvis_height])
        right_hip = _world([0.0, -profile.hip_width / 2.0, profile.pelvis_height])
        left_knee = _world([leg_swing * 0.45, profile.hip_width / 2.0, profile.pelvis_height - profile.upper_leg_length])
        right_knee = _world([-leg_swing * 0.45, -profile.hip_width / 2.0, profile.pelvis_height - profile.upper_leg_length])
        left_foot = _world([leg_swing, profile.hip_width / 2.0, max(0.02, profile.pelvis_height - profile.upper_leg_length - profile.lower_leg_length)])
        right_foot = _world([-leg_swing, -profile.hip_width / 2.0, max(0.02, profile.pelvis_height - profile.upper_leg_length - profile.lower_leg_length)])

        return {
            "pelvis": pelvis,
            "spine": spine,
            "neck": neck,
            "head": head,
            "left_shoulder": left_shoulder,
            "left_elbow": left_elbow,
            "left_hand": left_hand,
            "right_shoulder": right_shoulder,
            "right_elbow": right_elbow,
            "right_hand": right_hand,
            "left_hip": left_hip,
            "left_knee": left_knee,
            "left_foot": left_foot,
            "right_hip": right_hip,
            "right_knee": right_knee,
            "right_foot": right_foot,
        }

    def dump_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "usd_path": self.usd_path,
            "visual_asset_path": self.visual_asset_path,
            "distance": self.distance,
            "step_dt": self.step_dt,
            "body_profile": self.body_profile.__dict__,
            "collision_proxy": self.collision_proxy.__dict__,
            "phase_motion": {k: v.__dict__ for k, v in self.phase_motion.items()},
            "animation_runtime": None
            if self.animation_runtime is None
            else {
                "visual_usd_path": self.animation_runtime.visual_usd_path,
                "metadata_path": self.animation_runtime.metadata_path,
                "npz_path": self.animation_runtime.npz_path,
                "frame_count": self.animation_runtime.frame_count,
                "walk_fps": self.animation_runtime.walk_fps,
                "nominal_speed_mps": self.animation_runtime.nominal_speed_mps,
                "visual_correction_quat_xyzw": self.animation_runtime.visual_correction_quat_xyzw.tolist(),
            },
        }
