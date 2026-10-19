#!/usr/bin/env python
"""Export the model-agnostic canonical dataset from a step10-style snapshot.

Canonical 层原则（与 HANDOFF §10 红线一致）：
  - 所有 obs/action 字段**原样拷贝**：不归一化、不丢维（dim6 保留）、不重采样、不改夹爪。
  - 只新增派生索引列（frame_id/demo_id/timestamp/done/round_id 等）与元数据。
  - 语言一律取自 annotations/commands.json（demo attrs 的 instruction 有 P4 错误，仅作 provenance 保留）。
  - reward/task_success/terminated/truncated 原样保留但标记 deprecated（P5 不可信）。

用法:
  python scripts/export_canonical.py \
      --snapshot /root/autodl-tmp/hrc_snapshots/hrc_multitask_v01_step10 \
      --out /root/autodl-tmp/hrc_canonical/hrc_multitask_v01_step10
"""
import argparse
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

import h5py
import numpy as np

FPS = 30.0

OBS_FIELDS = [
    "robot_pos", "robot_quat", "joint_positions", "gripper_state",
    "human_pos", "human_quat", "relative_human_pose", "human_visible",
    "human_motion_state_id", "fsm_state_id", "head_camera_pitch_deg",
    "head_camera_local_pos", "head_camera_local_quat",
    "head_camera_world_pos", "head_camera_world_quat",
]
DEPRECATED_FIELDS = ["reward", "terminated", "truncated", "task_success"]

ACTION_SPACE_DESCRIPTION = {
    "action_dim": 23,
    "control_frequency_hz": 30.0,
    "physics_frequency_hz": 120.0,
    "action_normalize": False,
    "groups": [
        {"dims": [0, 3], "name": "base", "controller": "HolonomicBaseJointController",
         "type": "velocity_command", "frame": "robot_base_link",
         "meaning": "[vx, vy, vyaw]，归一化输入 [-1,1] -> [±0.75 m/s, ±0.75 m/s, ±1.0 rad/s]",
         "no_op": [0.0, 0.0, 0.0]},
        {"dims": [3, 7], "name": "trunk", "controller": "JointController(position)",
         "type": "absolute_joint_position_rad",
         "joint_positions_cols": [6, 7, 8, 9],
         "note": "dim6 准常数（step10 实测 std=4.65e-4，中位数 0.0）；训练建议丢弃、部署时重插 0.0"},
        {"dims": [7, 14], "name": "arm_left", "controller": "JointController(position)",
         "type": "absolute_joint_position_rad",
         "joint_positions_cols": [10, 12, 14, 16, 18, 20, 22]},
        {"dims": [14, 15], "name": "gripper_left", "controller": "MultiFingerGripper",
         "type": "binary", "values": [-1.0, 1.0],
         "note": "step10 实测严格二值（距 ±1 最大偏差 4.5e-4，无中间值）；极性见 polarity 字段"},
        {"dims": [15, 22], "name": "arm_right", "controller": "JointController(position)",
         "type": "absolute_joint_position_rad",
         "joint_positions_cols": [11, 13, 15, 17, 19, 21, 23]},
        {"dims": [22, 23], "name": "gripper_right", "controller": "MultiFingerGripper",
         "type": "binary", "values": [-1.0, 1.0]},
    ],
    "absolute_position_semantics": "action=0 表示『打向零位』而非『保持』；非任务段录的是 hold-action（当前关节角）",
    "joint_positions_layout": {
        "cols_0_5": "base 虚拟关节（推断）",
        "cols_6_9": "trunk（等待帧实测 |action-jp| = 0.0000 rad）",
        "cols_10_23": "左右臂交错：左=偶数列，右=奇数列（等待帧实测）",
        "cols_24_27": "夹爪指关节（推断，量程 0–0.05m 与 gripper_state 一致）",
    },
    "quaternion_convention": "xyzw (scalar-last, OmniGibson)",
}


def load_commands(snapshot: Path):
    with open(snapshot / "annotations" / "commands.json") as f:
        return json.load(f)


def measure_gripper_polarity(actions, gripper_state, act_dim, gs_cols):
    """实测夹爪极性：action 在 act_dim 上发生符号翻转后，gs_cols 指关节的移动方向。"""
    a = actions[:, act_dim]
    flips = np.where(np.sign(a[1:]) != np.sign(a[:-1]))[0]
    results = []
    for t in flips:
        lo, hi = t + 1, min(t + 1 + 45, len(a))  # 翻转后 1.5s 窗口
        if hi - lo < 10:
            continue
        delta = gripper_state[hi - 1, gs_cols].mean() - gripper_state[lo, gs_cols].mean()
        if abs(delta) > 1e-3:
            results.append((float(np.sign(a[t + 1])), float(delta)))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    snap, out = args.snapshot, args.out

    out.mkdir(parents=True, exist_ok=True)
    (out / "videos").mkdir(exist_ok=True)

    commands = load_commands(snap)
    with open(snap / "meta" / "episode_meta.json") as f:
        meta = json.load(f)

    src = h5py.File(snap / "hrc_dataset.hdf5", "r")
    demos = sorted(src["data"].keys())

    # ---- 拼接全局帧序列 ----
    per_demo = []
    for di, dn in enumerate(demos):
        d = src["data"][dn]
        n = d["action"].shape[0]
        per_demo.append((di, dn, d, n))
    total = sum(n for _, _, _, n in per_demo)

    cn = h5py.File(out / "canonical.hdf5", "w")
    fr = cn.create_group("frames")

    def concat(getter, dtype=None):
        arrs = [np.asarray(getter(d)) for _, _, d, _ in per_demo]
        a = np.concatenate(arrs, axis=0)
        return a.astype(dtype) if dtype else a

    # 原样字段
    fr.create_dataset("action", data=concat(lambda d: d["action"][:]))
    for k in OBS_FIELDS:
        fr.create_dataset(f"obs/{k}", data=concat(lambda d, k=k: d["obs"][k][:]))
    for k in DEPRECATED_FIELDS:
        ds = fr.create_dataset(f"deprecated/{k}", data=concat(lambda d, k=k: d[k][:]))
        ds.attrs["deprecated"] = "P5: 环境加载的是 task-0026 完整 BDDL，与 HRC 轮语义错位；勿用于训练/评测"
    fr.create_dataset("round_id", data=concat(lambda d: d["round_id"][:]))

    # 派生索引列
    frame_id = np.arange(total, dtype=np.int64)
    demo_id = np.concatenate([np.full(n, di, dtype=np.int32) for di, _, _, n in per_demo])
    frame_in_demo = np.concatenate([np.arange(n, dtype=np.int64) for _, _, _, n in per_demo])
    done = np.zeros(total, dtype=bool)
    done[np.cumsum([n for _, _, _, n in per_demo]) - 1] = True
    fr.create_dataset("frame_id", data=frame_id)
    fr.create_dataset("demo_id", data=demo_id)
    fr.create_dataset("frame_in_demo", data=frame_in_demo)
    fr.create_dataset("timestamp_s", data=(frame_id / FPS).astype(np.float64))
    fr.create_dataset("done", data=done)

    # ---- 夹爪极性实测 ----
    A = np.asarray(fr["action"][:])
    GS = np.asarray(fr["obs/gripper_state"][:])
    pol = {}
    for nm, ad, cols in [("gripper_left", 14, [0, 1]), ("gripper_right", 22, [2, 3])]:
        evidence = measure_gripper_polarity(A, GS, ad, cols)
        if evidence:
            sign_open = [s for s, dlt in evidence if dlt > 0]
            pol[nm] = {"+1_means": "open" if (sign_open and sign_open[0] > 0) else "close",
                       "evidence_flips": evidence}
        else:
            pol[nm] = {"+1_means": "unverified（本 episode 翻转样本不足）", "evidence_flips": []}
    asd = dict(ACTION_SPACE_DESCRIPTION)
    asd["gripper_polarity_measured"] = pol

    # ---- 指令表（语言唯一可信来源 + 人工目检 success）----
    demo_starts = np.concatenate([[0], np.cumsum([n for _, _, _, n in per_demo])[:-1]])
    instructions = []
    for c in commands:
        rid = c["round_id"]
        instructions.append({
            **c,
            "demo_id": rid,
            "demo_name": demos[rid],
            "issued_frame_global": c["issued_frame"],
            "issued_frame_in_demo": int(c["issued_frame"] - demo_starts[rid]),
            "success": True,
            "success_source": "human_video_review_2026-06-11（HDF5 内 task_success 不可信，见 deprecated）",
        })

    # ---- 顶层元数据 ----
    prov = {dn: {k: (v.item() if hasattr(v, "item") else str(v)) for k, v in src["data"][dn].attrs.items()}
            for dn in demos}
    cn.attrs["schema_version"] = "1.0"
    cn.attrs["episode_id"] = "hrc_multitask_v01_step10"
    cn.attrs["created"] = str(date.today())
    cn.attrs["source_snapshot"] = str(snap)
    cn.attrs["robot"] = "Galaxea R1Pro (OmniGibson robot_r1)"
    cn.attrs["simulator"] = "OmniGibson / BEHAVIOR-1K"
    cn.attrs["base_task"] = "task-0026 assembling_gift_baskets（底座场景；radio 道具借自 task-0000）"
    cn.attrs["control_frequency_hz"] = FPS
    cn.attrs["num_frames"] = total
    cn.attrs["action_space_description"] = json.dumps(asd, ensure_ascii=False, indent=1)
    cn.attrs["instructions"] = json.dumps(instructions, ensure_ascii=False, indent=1)
    cn.attrs["demo_attrs_provenance"] = json.dumps(prov, ensure_ascii=False, indent=1)
    cn.attrs["demo_attrs_warning"] = "P4: demo attrs 的 instruction 不可信（demo_0 实测错写为 round1 指令）；语言以 instructions 为准"
    cn.attrs["videos"] = json.dumps({
        "head_rgb": {"file": "videos/head_rgb.mp4", "resolution": [352, 256], "fps": 30,
                     "view": "robot head ZED first-person, 相位条件化俯仰（逐帧角度在 obs/head_camera_pitch_deg）"},
        "third_person_rgb": {"file": "videos/third_person_rgb.mp4", "resolution": [1280, 720], "fps": 30,
                             "view": "global third-person"},
        "alignment": "视频第 i 帧 == frames/frame_id==i（实测两路均 10239 帧，与 HDF5 严格 1:1）",
    }, ensure_ascii=False, indent=1)
    cn.attrs["known_gaps"] = "无深度/分割/点云；无人物骨骼关键点；reward/success 走 deprecated"
    cn.close()

    # ---- 视频拷贝（canonical 自包含）----
    for v in ["head_rgb.mp4", "third_person_rgb.mp4"]:
        shutil.copy2(snap / "videos" / v, out / "videos" / v)

    # ================= 验证 =================
    print("=== 验证：逐字节回比 ===")
    cn = h5py.File(out / "canonical.hdf5", "r")
    ok = True
    off = 0
    for di, dn, d, n in per_demo:
        sl = slice(off, off + n)
        if not np.array_equal(cn["frames/action"][sl], d["action"][:]):
            ok = False; print(f"  MISMATCH action {dn}")
        for k in OBS_FIELDS:
            if not np.array_equal(cn[f"frames/obs/{k}"][sl], d["obs"][k][:]):
                ok = False; print(f"  MISMATCH obs/{k} {dn}")
        for k in DEPRECATED_FIELDS:
            if not np.array_equal(cn[f"frames/deprecated/{k}"][sl], d[k][:]):
                ok = False; print(f"  MISMATCH {k} {dn}")
        off += n
    print(f"  action + 15 obs + 4 deprecated 全字段: {'一致' if ok else '有差异!'}")

    nf = cn.attrs["num_frames"]
    assert nf == meta["num_frames"], f"num_frames {nf} != meta {meta['num_frames']}"
    print(f"  num_frames={nf} 与 episode_meta 一致")
    dn_frames = int(np.asarray(cn["frames/done"]).sum())
    print(f"  done 帧数={dn_frames}（应={len(demos)}）")

    for v in ["head_rgb.mp4", "third_person_rgb.mp4"]:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
                            str(out / "videos" / v)], capture_output=True, text=True)
        vf = int(r.stdout.strip())
        print(f"  {v}: {vf} 帧 {'== num_frames ✓' if vf == nf else '!= num_frames ✗'}")
        assert vf == nf

    print("  夹爪极性实测:", json.dumps(pol, ensure_ascii=False))
    print("  指令表:", [(i['round_id'], i['instruction'][:40], i['success']) for i in instructions])
    cn.close(); src.close()
    print(f"=== 完成: {out / 'canonical.hdf5'} ===")


if __name__ == "__main__":
    main()
