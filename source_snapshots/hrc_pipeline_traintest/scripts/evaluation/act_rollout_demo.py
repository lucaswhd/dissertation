#!/usr/bin/env python
# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
# See LICENSE in the project root for full terms.

"""Closed-loop ACT rollout in OmniGibson（HANDOFF §12.6.1 仿真 rollout 目检）.

跑在 **behavior-venv**。场景搭建与 hrc_multitask_demo.main() 的 --online-nav 分支
一致（contacts-ON wrapper、radio 外援道具、隐藏无关道具、家具钉固、Habitat 人物），
区别在于：机器人不走 FSM/回放，而是每帧把 obs（state 44 维 + head/third RGB）发给
act_policy_server（lerobot-venv 进程），用返回的 22 维 ACT 动作经 env.step() 驱动。

环境侧时间轴（人物位姿、头相机俯仰、FSM 相位标签、round 边界、指令文本）按
step10 快照逐帧回放，保证 obs 不出训练分布；机器人本体完全由策略闭环驱动
（无 load_state、无瞬移——红线不破）。

输出：独立目录（默认 /root/autodl-tmp/outputs/hrc_rollout_act_v01/）下的
head/third 视频 + rollout HDF5（同 schema，可跑 §9 验证）+ rollout_meta.json。
轮末用与采集侧相同的 success_predicate 谓词给出客观判定（[rollout-success] 日志）。

用法（先起 act_policy_server，再）:
  /root/behavior-venv/bin/python scripts/act_rollout_demo.py \
      --config configs/hrc_multitask_v01.yaml \
      --snapshot /root/autodl-tmp/hrc_snapshots/hrc_multitask_v01_step10 \
      --output-dir /root/autodl-tmp/outputs/hrc_rollout_act_v01
  --dry-run 只校验时间轴/配置/服务连通，不起仿真。
"""
import argparse
import json
import os
import pickle
import socket
import struct
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 22 维 adapter 动作 -> 23 维原生动作（HANDOFF §12.1：dim6 训练时被丢，部署重插 0.0）
ACTION_KEEP = [i for i in range(23) if i != 6]
DIM6_DEPLOY_VALUE = 0.0
GRIPPER_DIMS_23 = (14, 22)  # 夹爪二值 ±1（+1=张开 / -1=闭合）

FSM_ID_TO_NAME = ["WaitNearHuman", "ReceiveCommand", "GoToTask", "ExecuteBehaviorTask",
                  "TaskSegmentDone", "ReturnToHuman", "FaceHuman", "WaitNextCommand",
                  "AllTasksDone", "Relocate"]


# --------------------------------------------------------------------------- #
# Policy client (length-prefixed pickle over TCP, see act_policy_server.py)
# --------------------------------------------------------------------------- #

# numpy-version-agnostic wire codec (must match act_policy_server.py). server
# is numpy 2.x, this client is numpy 1.x; pickling ndarrays directly embeds the
# numpy-2 internal module path that numpy-1 can't import. So arrays cross the
# wire as {raw bytes + dtype + shape}, which pickles identically on any numpy.
def _pack(obj):
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {"__nd__": True, "data": arr.tobytes(),
                "dtype": str(arr.dtype), "shape": list(arr.shape)}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _pack(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_pack(v) for v in obj]
    return obj


def _unpack(obj):
    if isinstance(obj, dict):
        if obj.get("__nd__"):
            return np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"]).copy()
        return {k: _unpack(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unpack(v) for v in obj]
    return obj


class ActPolicyClient:
    def __init__(self, host: str, port: int, timeout_s: float = 60.0):
        self.sock = socket.create_connection((host, port), timeout=timeout_s)

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("act_policy_server closed connection")
            buf += chunk
        return buf

    def call(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        payload = pickle.dumps(_pack(msg), protocol=pickle.HIGHEST_PROTOCOL)
        self.sock.sendall(struct.pack(">I", len(payload)) + payload)
        (length,) = struct.unpack(">I", self._recv_exact(4))
        reply = _unpack(pickle.loads(self._recv_exact(length)))
        if not reply.get("ok", False):
            raise RuntimeError(f"act_policy_server error: {reply.get('error')}")
        return reply

    def ping(self) -> Dict[str, Any]:
        return self.call({"type": "ping"})

    def reset(self) -> None:
        self.call({"type": "reset"})

    def act(self, state: np.ndarray, head: np.ndarray, task: str,
            third: Optional[np.ndarray] = None) -> np.ndarray:
        msg = {"type": "act", "state": state, "head": head, "task": task}
        if third is not None:
            msg["third"] = third
        reply = self.call(msg)
        return np.asarray(reply["action"], dtype=np.float32)

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Snapshot timeline (environment-side script replayed per frame)
# --------------------------------------------------------------------------- #

def load_timeline(snapshot_dir: str) -> Dict[str, Any]:
    """Concatenate the snapshot's demos (in round order) into one per-frame
    environment timeline + per-round instructions (P4: commands.json 为唯一可信源)."""
    import h5py  # local import keeps --dry-run light

    h5_path = os.path.join(snapshot_dir, "hrc_dataset.hdf5")
    parts: List[Dict[str, np.ndarray]] = []
    with h5py.File(h5_path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        for dn in demos:
            d = f["data"][dn]
            parts.append({
                # 23 维原生专家动作（顶层 action），用于 --action-source expert 开环对照。
                "action": np.asarray(d["action"][:], dtype=np.float32),
                "human_pos": np.asarray(d["obs/human_pos"][:], dtype=np.float32),
                "human_quat": np.asarray(d["obs/human_quat"][:], dtype=np.float32),
                "fsm": np.asarray(d["obs/fsm_state_id"][:], dtype=np.int32).ravel(),
                "pitch": np.asarray(d["obs/head_camera_pitch_deg"][:], dtype=np.float32).ravel(),
                "round_id": np.asarray(d["round_id"][:], dtype=np.int32).ravel(),
                "robot_pos": np.asarray(d["obs/robot_pos"][:], dtype=np.float32),
                "robot_quat": np.asarray(d["obs/robot_quat"][:], dtype=np.float32),
                # joint_positions needed to seat the robot at a MID-episode start
                # frame (--start-frame), e.g. the execute-phase pose in front of
                # the radio (no full state load until Step B).
                "joint_positions": np.asarray(d["obs/joint_positions"][:], dtype=np.float32),
            })
    tl = {k: np.concatenate([p[k] for p in parts], axis=0)
          for k in parts[0].keys()}
    tl["num_frames"] = int(len(tl["fsm"]))
    tl["robot_pos0"] = parts[0]["robot_pos"][0].copy()
    tl["robot_quat0"] = parts[0]["robot_quat"][0].copy()

    with open(os.path.join(snapshot_dir, "annotations", "commands.json"), encoding="utf-8") as f:
        commands = json.load(f)
    tl["instruction_by_round"] = {int(c["round_id"]): str(c["instruction"]) for c in commands}
    return tl


def _resize_half(img: np.ndarray) -> np.ndarray:
    """1280x720 -> 640x360 area-average downsample (≈ 训练侧 pyav reformat)."""
    try:
        import cv2
        return cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2),
                          interpolation=cv2.INTER_AREA)
    except Exception:
        h, w = img.shape[0] // 2, img.shape[1] // 2
        return img[: 2 * h, : 2 * w].reshape(h, 2, w, 2, 3).mean(axis=(1, 3)).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hrc_multitask_v01.yaml")
    ap.add_argument("--snapshot", default="/root/autodl-tmp/hrc_snapshots/hrc_multitask_v01_step10")
    ap.add_argument("--output-dir", default="/root/autodl-tmp/outputs/hrc_rollout_act_v01")
    ap.add_argument("--server-host", default="127.0.0.1")
    ap.add_argument("--server-port", type=int, default=18765)
    ap.add_argument("--ckpt-label", default="act_overfit_01/020000",
                    help="写入 rollout_meta.json 的 checkpoint 标识（推理在 server 侧）。")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="只滚前 N 帧（默认整条时间轴）。配合 --start-frame 即起点后 N 帧。")
    ap.add_argument("--start-frame", type=int, default=0,
                    help="从时间轴第 N 帧起步（默认 0=开头）。用于段策略：机器人直接坐到"
                         "该帧的位姿+关节构型再闭环（如执行段=站在收音机前）。无全状态 load，"
                         "靠 robot_pos/quat + joint_positions 重建，足够操作段用。")
    ap.add_argument("--start-phase", type=int, default=None,
                    help="便捷：自动取某 round 内首个该 fsm 相位帧作为 --start-frame。"
                         "例 --start-phase 3 = ExecuteBehaviorTask 首帧。")
    ap.add_argument("--dry-run", action="store_true",
                    help="只校验时间轴/配置/服务连通性，不启动仿真。")
    ap.add_argument("--action-source", choices=["policy", "expert"], default="policy",
                    help="policy=查 act_policy_server（默认）；expert=直接喂快照里录下的 23 维"
                         "专家动作（开环、无 load_state）。expert 模式是「完美动作能否在 rollout "
                         "物理环境里真正打开 radio」的对照实验，不需要 policy server。")
    ap.add_argument("--debug-radio", action="store_true",
                    help="逐帧打印左右夹爪 eef 到 radio 的距离 + ToggledOn 状态，并在收尾汇总"
                         "全程最近距离/是否曾 toggle。用于定位 radio 失败是「没接近」还是"
                         "「接近了但开关语义没触发」。")
    ap.add_argument("--eef-verify", action="store_true",
                    help="额外打印 base 世界坐标 + get_eef_position(API) vs eef-link 真实世界坐标，"
                         "核 eef 计算是否世界系、手臂物理上够不够得到 radio。需配 --debug-radio。")
    args = ap.parse_args()
    expert_mode = (args.action_source == "expert")

    import yaml
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    tl = load_timeline(args.snapshot)
    T_tl = tl["num_frames"]

    # Resolve the start frame (explicit --start-frame, or first frame of --start-phase).
    start = int(args.start_frame)
    if args.start_phase is not None:
        cand = np.where(tl["fsm"] == int(args.start_phase))[0]
        if len(cand) == 0:
            raise ValueError(f"--start-phase {args.start_phase} 不在时间轴里")
        start = int(cand[0])
        print(f"[rollout] --start-phase {args.start_phase} -> start_frame {start}", flush=True)
    start = max(0, min(start, T_tl - 1))
    end = T_tl if args.max_frames is None else min(start + int(args.max_frames), T_tl)
    T = end - start
    # Robot seats at the START frame's pose + joint config (not frame 0).
    tl["robot_pos0"] = tl["robot_pos"][start].copy()
    tl["robot_quat0"] = tl["robot_quat"][start].copy()
    tl["robot_jp0"] = tl["joint_positions"][start].copy()
    print(f"[rollout] timeline: {T_tl} frames; roll [{start},{end}) = {T} 帧 "
          f"(start round={int(tl['round_id'][start])} phase={int(tl['fsm'][start])})", flush=True)
    print(f"[rollout] instructions: {tl['instruction_by_round']}", flush=True)
    print(f"[rollout] robot start pose: pos={tl['robot_pos0'].tolist()}", flush=True)

    client: Optional[ActPolicyClient] = None
    use_third = True
    if expert_mode:
        # 专家开环对照：不查 policy server。第三人称仍渲染以便目检视频。
        print("[rollout] action-source=EXPERT：开环回放快照录下的 23 维专家动作"
              "（无 load_state、无 policy server）。", flush=True)
        if args.dry_run:
            print("[rollout] --dry-run OK (expert mode)", flush=True)
            return
    else:
        try:
            client = ActPolicyClient(args.server_host, args.server_port)
            info = client.ping()
            # The checkpoint declares which cameras it consumes; for a head-only
            # policy we skip the (expensive) third-person viewer render entirely.
            img_inputs = info.get("image_inputs", ["observation.images.head",
                                                   "observation.images.third"])
            use_third = "observation.images.third" in img_inputs
            print(f"[rollout] act_policy_server OK: {info}", flush=True)
            print(f"[rollout] policy cameras={img_inputs}  use_third={use_third}", flush=True)
        except Exception as exc:
            print(f"[rollout] act_policy_server NOT reachable "
                  f"({args.server_host}:{args.server_port}): {exc}", flush=True)
            if not args.dry_run:
                raise

    if args.dry_run:
        rounds = cfg.get("task_rounds", [])
        for i, rc in enumerate(rounds):
            print(f"[rollout] round {i}: kind={rc.get('kind')} "
                  f"success_predicate={rc.get('success_predicate')}", flush=True)
        print("[rollout] --dry-run OK", flush=True)
        if client:
            client.close()
        return

    # ---- Heavy imports (OmniGibson) only past this point.
    import torch as th
    import omnigibson as og
    from hrc_utils import (
        compute_human_visible, compute_relative_pose, drive_human_from_recording,
        ensure_parent, extract_first_rgb, get_activity_from_hdf5, normalize_rgb,
        spawn_extra_object, to_numpy, update_viewer_camera,
    )
    from hrc_humanoid import KinematicHumanoidActor
    from hrc_minimal_demo import DemoRecorder
    from nav_online_driver import build_contacts_on_playback_wrapper
    import hrc_multitask_demo as mt
    import h5py

    raw_hdf5 = str(cfg["raw_hdf5"])
    output_dir = os.path.abspath(args.output_dir)
    resolution = int(cfg.get("resolution", 256))
    fps = int(cfg.get("fps", 30))
    human_usd = cfg.get("human_usd", None)
    human_distance = float(cfg.get("human_distance", 1.8))
    pitch_axis = str(cfg.get("head_camera_pitch_axis", "x"))
    aperture = float(cfg.get("head_camera_horizontal_aperture", 0.0))
    extra_objects: List[Dict[str, Any]] = list(cfg.get("extra_objects", []))
    hidden_categories = [str(c).strip().lower() for c in cfg.get("hidden_categories", [])]
    task_rounds: List[Dict[str, Any]] = list(cfg.get("task_rounds", []))
    episode_id = int(cfg.get("episode_id", 0))

    ensure_parent(os.path.join(output_dir, "placeholder.txt"))
    base_activity = get_activity_from_hdf5(raw_hdf5)
    playback_cache = os.path.join(output_dir, f"playback_cache_{os.getpid()}.hdf5")

    print("[rollout] building contacts-ON DataPlaybackWrapper ...", flush=True)
    env = build_contacts_on_playback_wrapper(
        raw_hdf5, output_hdf5_path=playback_cache,
        resolution=resolution, robot_modalities=["rgb"],
    )

    recorder = None
    closed = False
    try:
        env.reset()
        robot = env.robots[0]

        # Head camera: capture rest pose (pitch=0 短路只缓存)，随后每帧按快照
        # 录得的绝对俯仰回放；FOV 与采集一致。
        mt.tilt_head_camera(env, 0.0, axis=pitch_axis)
        mt.set_head_camera_aperture(env, aperture)

        # Extra objects (radio …) — 与采集一致。
        spawned: Dict[str, Any] = {}
        for spec in extra_objects:
            obj = spawn_extra_object(
                env, name=spec["name"], category=spec["category"], model=spec["model"],
                position=spec["position"],
                orientation=spec.get("orientation", [0.0, 0.0, 0.0, 1.0]),
                in_rooms=spec.get("in_rooms"),
            )
            spawned[spec["name"]] = obj
            print(f"[rollout] spawned {spec['name']}", flush=True)

        # Step-0 state load（对齐道具初始位姿），随后把机器人放到快照 frame-0 位姿。
        _grp = env.input_hdf5["data"][f"demo_{episode_id}"]
        _state0 = _grp["state"][0]
        _sz0 = int(_grp["state_size"][0])
        og.sim.load_state(th.tensor(_state0[:_sz0]), serialized=True)
        # Seat arms/trunk/gripper at the START frame's joint config, but SKIP the 6
        # holonomic base virtual joints (base_footprint_{x,y,z,rx,ry,rz} = robot.base_idx).
        # 这些虚拟关节编码的是底盘相对 world_base_joint 原点的累计位移；snapshot 里录的是
        # task-0000 原点下的绝对值。若整段 set_joint_positions 会把它们一并写入，覆盖掉
        # set_position_orientation 刚摆好的底盘 → 位移被算两遍。实测 mid-episode 起步因此
        # 落偏 ~1.4m、离 radio ~2.8m（2026-06-18 诊断）。底盘位姿只交给 set_position_orientation。
        _jp0 = th.tensor(tl["robot_jp0"], dtype=th.float32)
        try:
            _base_idx = set(int(i) for i in robot.base_idx.tolist())
        except Exception as _exc:
            _base_idx = set()
            print(f"[rollout] WARN: robot.base_idx 不可用（{_exc}）；退回整段 set_joint_positions", flush=True)
        _non_base = [i for i in range(int(_jp0.shape[0])) if i not in _base_idx]
        if _base_idx:
            robot.set_joint_positions(_jp0[_non_base], indices=th.tensor(_non_base, dtype=th.long))
        else:
            robot.set_joint_positions(_jp0)
        # 底盘位姿放在关节之后设，独占 base_idx，避免被覆盖。
        robot.set_position_orientation(tl["robot_pos0"], tl["robot_quat0"])
        robot.keep_still()
        print(f"[rollout] seated: base via set_pose xy={np.round(tl['robot_pos0'][:2],3).tolist()}; "
              f"{len(_non_base)} non-base joints from snapshot "
              f"(skipped {len(_base_idx)} base virtual joints)", flush=True)

        # Hide task-irrelevant props (visual only) — 与采集一致。
        if hidden_categories:
            n_hidden = 0
            for obj in env.scene.objects:
                if (getattr(obj, "category", "") or "").lower() in hidden_categories:
                    try:
                        obj.visible = False
                        n_hidden += 1
                    except Exception:
                        pass
            print(f"[rollout] hidden {n_hidden} task-irrelevant objects", flush=True)

        # Align extras to their external-replay rest pose（与采集的 peek 逻辑一致）。
        for round_cfg in task_rounds:
            if round_cfg.get("kind") != "replay_external_segment" or not spawned:
                continue
            ext_path = round_cfg.get("external_hdf5")
            if not ext_path:
                continue
            ext_ep = int(round_cfg.get("episode_id", 0))
            ext_start = int(round_cfg.get("start_step", 0))
            _snap = og.sim.dump_state(serialized=True)
            try:
                with h5py.File(ext_path, "r") as f:
                    traj = f[f"data/demo_{ext_ep}"]
                    st = traj["state"][ext_start, : int(traj["state_size"][ext_start])]
                    og.sim.load_state(th.tensor(st, dtype=th.float32), serialized=True)
                for name, obj in spawned.items():
                    pos = to_numpy(obj.get_position_orientation()[0]).copy()
                    quat = to_numpy(obj.get_position_orientation()[1]).copy()
                    obj._ext_init_pos, obj._ext_init_quat = pos, quat
            except Exception as exc:
                print(f"[rollout] WARN: extras peek failed: {exc}", flush=True)
            finally:
                og.sim.load_state(_snap, serialized=True)
                del _snap
            for name, obj in spawned.items():
                if hasattr(obj, "_ext_init_pos"):
                    obj.set_position_orientation(obj._ext_init_pos, obj._ext_init_quat)
                    obj.keep_still()
                    print(f"[rollout] {name} -> external-replay position", flush=True)

        # Human NPC：与采集一致的 actor，但逐帧由快照轨迹驱动。
        human_actor = KinematicHumanoidActor(
            env=env, usd_path=human_usd, name="human_npc",
            distance=human_distance, collision_proxy=None,
            phase_motion=mt._build_phase_motion(), step_dt=1.0 / max(fps, 1),
        )
        human_actor.collision_proxy.enabled = False
        human_actor.spawn_near_robot()
        drive_human_from_recording(human_actor, tl["human_pos"], tl["human_quat"], 0.0)
        print(f"[rollout] human spawned; snapped to timeline frame 0 "
              f"{tl['human_pos'][0][:2].tolist()}", flush=True)

        # 家具钉固（同采集 --online-nav）：策略驱动的物理底盘撞不动家具。
        # 注意：篮子等任务物 **不** 钉——策略必须能真实搬动它们。
        mt._pinned_scene_objects.clear()
        _furniture_substrings = ("chair", "table", "sofa", "couch", "stool",
                                 "bench", "armchair", "ottoman")
        n_furniture = 0
        for obj in env.scene.objects:
            if obj == robot:
                continue
            cat = (getattr(obj, "category", "") or "").lower()
            if any(sub in cat for sub in _furniture_substrings):
                pos = to_numpy(obj.get_position_orientation()[0]).copy()
                quat = to_numpy(obj.get_position_orientation()[1]).copy()
                mt._pinned_scene_objects[obj.name] = (pos, quat)
                n_furniture += 1
        print(f"[rollout] pinned {n_furniture} furniture objects (baskets NOT pinned)", flush=True)

        # Recorder（rollout 输出与训练数据同 schema，可直接跑 §9 验证脚本）。
        recorder = DemoRecorder(
            env=env, human=human_actor, activity=base_activity,
            instruction=tl["instruction_by_round"].get(0, ""),
            output_dir=output_dir, fps=fps, input_hdf5_path="",
        )
        recorder._human_actor = human_actor
        recorder.camera_state_provider = lambda: mt.get_head_camera_state(env)
        recorder.dataset_metadata = {
            "rollout": True,
            "policy_checkpoint": args.ckpt_label,
            "timeline_snapshot": args.snapshot,
            "head_camera_intrinsics": mt.get_head_camera_intrinsics(env),
            "env_args": {"robot": type(robot).__name__, "fps": fps,
                         "resolution": resolution, "raw_hdf5": raw_hdf5},
        }

        # ---- Closed loop.
        hold_np = to_numpy(mt._build_hold_action(robot)).astype(np.float32)

        def assemble_state() -> np.ndarray:
            """44 维 state，布局与 export_lerobot_act.py 完全一致。"""
            rp, rq = robot.get_position_orientation()
            hp, _hq = human_actor.get_position_orientation()
            rp, rq, hp = to_numpy(rp), to_numpy(rq), to_numpy(hp)
            jp = to_numpy(robot.get_joint_positions()).astype(np.float32)          # 28
            grip_parts = [to_numpy(robot.get_joint_positions())[robot.gripper_control_idx[a]]
                          for a in robot.arm_names]                                 # 4
            grip = np.concatenate(grip_parts).astype(np.float32)
            rel = compute_relative_pose(rp, hp).astype(np.float32)                  # 3
            vis = float(compute_human_visible(rp, rq, hp))                          # 1
            pitch = float(mt.get_head_camera_state(env).get("pitch_deg", 0.0))      # 1
            state = np.concatenate([jp, grip, rp.astype(np.float32),
                                    rq.astype(np.float32), rel,
                                    np.array([vis, pitch], dtype=np.float32)])
            assert state.shape == (44,), state.shape
            return state

        def to_native_action(a22: np.ndarray) -> np.ndarray:
            a23 = np.zeros(23, dtype=np.float32)
            a23[ACTION_KEEP] = a22
            a23[6] = DIM6_DEPLOY_VALUE
            a23[0:3] = np.clip(a23[0:3], -1.0, 1.0)            # base 速度指令限幅
            for g in GRIPPER_DIMS_23:                          # 夹爪二值化（离散语义）
                a23[g] = 1.0 if a23[g] >= 0.0 else -1.0
            return a23

        # ---- radio toggle diagnostics (--debug-radio)
        radio_obj = spawned.get("radio_89") or (next(iter(spawned.values())) if spawned else None)
        toggled_on_cls = None
        try:
            from omnigibson.object_states import ToggledOn as toggled_on_cls  # type: ignore
        except Exception:
            pass
        dbg = {"min_eef_to_radio": float("inf"), "min_eef_to_marker": float("inf"),
               "ever_toggled": False, "prev_toggle": None}

        def _marker_pos() -> Optional[np.ndarray]:
            if radio_obj is None or toggled_on_cls is None:
                return None
            try:
                st = radio_obj.states[toggled_on_cls]
                link = getattr(st, "link", None)
                if link is not None:
                    return to_numpy(link.get_position_orientation()[0])
            except Exception:
                pass
            return None

        def radio_debug(gf: int, force: bool = False) -> None:
            if not args.debug_radio or radio_obj is None:
                return
            try:
                rad = to_numpy(radio_obj.get_position_orientation()[0])
                mk = _marker_pos()
                toggled = None
                if toggled_on_cls is not None:
                    try:
                        toggled = bool(radio_obj.states[toggled_on_cls].get_value())
                    except Exception:
                        toggled = None
                base_w = to_numpy(robot.get_position_orientation()[0])
                dists = {}
                eef_verify = []
                for arm in robot.arm_names:
                    try:
                        eef = to_numpy(robot.get_eef_position(arm))
                    except Exception:
                        continue
                    d_center = float(np.linalg.norm(eef - rad))
                    dists[arm] = d_center
                    dbg["min_eef_to_radio"] = min(dbg["min_eef_to_radio"], d_center)
                    if mk is not None:
                        d_mk = float(np.linalg.norm(eef - mk))
                        dbg["min_eef_to_marker"] = min(dbg["min_eef_to_marker"], d_mk)
                    # --eef-verify: 对比 API eef vs eef-link 真实世界坐标 vs 底盘
                    if getattr(args, "eef_verify", False):
                        link_pos = None
                        try:
                            link_pos = to_numpy(robot.eef_links[arm].get_position_orientation()[0])
                        except Exception:
                            pass
                        eef_verify.append(
                            f"{arm}: api_eef={np.round(eef,2).tolist()} "
                            f"link_eef={None if link_pos is None else np.round(link_pos,2).tolist()} "
                            f"|api-base|={np.linalg.norm(eef-base_w):.2f}")
                if eef_verify:
                    print(f"[eef-verify] f{gf} base={np.round(base_w,2).tolist()} "
                          f"radio={np.round(rad,2).tolist()} | " + " || ".join(eef_verify), flush=True)
                if toggled:
                    dbg["ever_toggled"] = True
                changed = (toggled is not None and toggled != dbg["prev_toggle"])
                dbg["prev_toggle"] = toggled
                if force or changed or (gf % 100 == 0):
                    dstr = " ".join(f"{a}={d:.3f}m" for a, d in dists.items())
                    mkstr = "" if mk is None else f" marker_pos={np.round(mk, 3).tolist()}"
                    print(f"[radio-dbg] f{gf} eef->radio_center: {dstr} "
                          f"ToggledOn={toggled}{mkstr}", flush=True)
            except Exception as exc:
                if force:
                    print(f"[radio-dbg] WARN: {exc}", flush=True)

        if client is not None:
            client.reset()
        verdicts: List[Dict[str, Any]] = []

        def eval_round(rid: int) -> None:
            pred = task_rounds[rid].get("success_predicate") if rid < len(task_rounds) else None
            ok, detail = mt._eval_round_success(env, spawned, pred)
            tag = "UNKNOWN" if ok is None else ("SUCCESS" if ok else "FAIL")
            print(f"[rollout-success] round {rid}: {tag} ({detail})", flush=True)
            if ok is not None:
                recorder.set_round_success(rid, ok, detail)
            verdicts.append({"round_id": rid, "success": ok, "detail": detail})

        obs = None
        prev_rid = int(tl["round_id"][start])
        recorder.instruction = tl["instruction_by_round"].get(prev_rid, recorder.instruction)
        recorder.add_command(round_id=prev_rid)
        t_wall0 = time.time()
        for t in range(T):
            gf = start + t                    # global timeline frame index
            rid = int(tl["round_id"][gf])
            if rid != prev_rid:
                eval_round(prev_rid)          # 上一轮的轮末判定（demo 边界）
                if client is not None:
                    client.reset()            # episode 边界：清 ACT action 队列
                recorder.instruction = tl["instruction_by_round"].get(rid, recorder.instruction)
                recorder.add_command(round_id=rid)
                prev_rid = rid

            # ---- environment-side timeline replay
            mt.set_head_camera_pitch_absolute(env, float(tl["pitch"][gf]),
                                              axis=pitch_axis, verbose=False)
            drive_human_from_recording(human_actor, tl["human_pos"], tl["human_quat"],
                                       gf / max(T_tl - 1, 1))
            mt._restore_pinned_objects(env)

            # ---- action: expert open-loop replay 或 policy 闭环
            if expert_mode:
                # 原生 23 维专家动作，原样喂入（不二值化、不限幅）——最忠实的「完美动作」。
                action23 = tl["action"][gf].astype(np.float32).copy()
                # 第三人称仍渲染进 obs 供 recorder 写视频（人物跟拍视角目检）。
                if use_third and obs is not None:
                    rp = to_numpy(robot.get_position_orientation()[0])
                    hp = to_numpy(human_actor.get_position_orientation()[0])
                    update_viewer_camera(rp, hp)
                    og.sim.viewer_camera.get_obs()
            elif obs is None:
                action23 = hold_np.copy()     # 帧 0：与训练开局 idle hold 一致
            else:
                head_frame, _ = extract_first_rgb(obs)
                if head_frame is None:
                    raise RuntimeError("no robot RGB in obs")
                third_frame = None
                if use_third:
                    rp = to_numpy(robot.get_position_orientation()[0])
                    hp = to_numpy(human_actor.get_position_orientation()[0])
                    update_viewer_camera(rp, hp)
                    viewer_obs, _ = og.sim.viewer_camera.get_obs()
                    third_frame = _resize_half(normalize_rgb(to_numpy(viewer_obs["rgb"])))
                a22 = client.act(assemble_state(), head_frame,
                                 task=tl["instruction_by_round"].get(rid, ""),
                                 third=third_frame)
                action23 = to_native_action(a22)

            obs, rew, term, trunc, info = env.step(action23, n_render_iterations=1)
            radio_debug(gf)
            info = info if isinstance(info, dict) else {}
            info["reward"] = float(rew) if rew is not None else 0.0
            info["terminated"] = bool(term) if term is not None else False
            info["truncated"] = bool(trunc) if trunc is not None else False
            phase_name = FSM_ID_TO_NAME[int(tl["fsm"][gf])]
            recorder.record(phase_name, obs, info, round_id=rid, action=action23)

            if (t + 1) % 500 == 0:
                el = time.time() - t_wall0
                print(f"[rollout] {t + 1}/{T} frames ({el:.0f}s, "
                      f"{(t + 1) / max(el, 1e-9):.1f} fps)", flush=True)

        eval_round(prev_rid)                  # 最后一轮判定
        if args.debug_radio:
            mk_str = ("n/a" if dbg["min_eef_to_marker"] == float("inf")
                      else f"{dbg['min_eef_to_marker']:.3f}m")
            ctr_str = ("n/a" if dbg["min_eef_to_radio"] == float("inf")
                       else f"{dbg['min_eef_to_radio']:.3f}m")
            print(f"[radio-dbg] SUMMARY action_source={args.action_source} "
                  f"min_eef_to_radio_center={ctr_str} min_eef_to_toggle_marker={mk_str} "
                  f"ever_toggled={dbg['ever_toggled']}", flush=True)
        recorder.close()
        closed = True

        with open(os.path.join(output_dir, "meta", "rollout_meta.json"), "w",
                  encoding="utf-8") as f:
            json.dump({
                "policy_checkpoint": args.ckpt_label,
                "server": f"{args.server_host}:{args.server_port}",
                "timeline_snapshot": args.snapshot,
                "frames_rolled": T,
                "timeline_frames": T_tl,
                "action_source": args.action_source,
                "round_verdicts": verdicts,
                "radio_debug": ({
                    "min_eef_to_radio_center": (None if dbg["min_eef_to_radio"] == float("inf")
                                                else dbg["min_eef_to_radio"]),
                    "min_eef_to_toggle_marker": (None if dbg["min_eef_to_marker"] == float("inf")
                                                 else dbg["min_eef_to_marker"]),
                    "ever_toggled": dbg["ever_toggled"],
                } if args.debug_radio else None),
            }, f, indent=2, ensure_ascii=False)

        print("\n[rollout] DONE", flush=True)
        print(f"  output dir : {output_dir}")
        print(f"  frames     : {T}")
        for v in verdicts:
            print(f"  round {v['round_id']}: success={v['success']} ({v['detail']})")
        print(f"  head video : {recorder.head_path}")
        print(f"  third video: {recorder.third_path}")

    except Exception:
        print("[rollout] ERROR", flush=True)
        traceback.print_exc()
        raise
    finally:
        # Finalize the recorder even on crash/interrupt so a partial rollout
        # still yields a playable (multi-frame) video + HDF5 for diagnosis,
        # instead of a 1-frame stub from un-flushed writers.
        if recorder is not None and not closed:
            try:
                recorder.close()
                print(f"[rollout] partial output finalized "
                      f"({recorder.frame_idx} frames)", flush=True)
            except Exception as exc:
                print(f"[rollout] WARN: recorder.close() failed: {exc}", flush=True)
        if client:
            client.close()
        try:
            og.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
