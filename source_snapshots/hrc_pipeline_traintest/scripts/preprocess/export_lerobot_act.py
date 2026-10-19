#!/usr/bin/env python
"""ACT adapter: canonical dataset -> LeRobot v3 数据集（本地，不推 hub）。

加工内容（全部可溯源，canonical 不动）：
  - action: 丢 dim6（准常数，std=4.65e-4）→ 22 维；常值 0.0 记录在 meta 供部署重插。
  - 夹爪保持 ±1 原值（本来就是干净二值，阈值化留给部署侧）。
  - state 44 维：joint_positions(28) + gripper_state(4) + robot_pos(3) + robot_quat(4)
                + relative_human_pose(3) + human_visible(1) + head_camera_pitch_deg(1)。
  - 图像：head 352x256 原样；third 1280x720 -> 640x360（训练开销）。RGB 从 canonical mp4 解帧。
  - 每个 demo = 一个 LeRobot episode，task = 该轮指令（canonical attrs.instructions，按 round_id）。

用法:
  python scripts/export_lerobot_act.py \
      --canonical /root/autodl-tmp/hrc_canonical/hrc_multitask_v01_step10 \
      --root /root/autodl-tmp/lerobot_datasets/hrc_v01_step10 \
      --repo-id hrc/hrc_v01_step10
"""
import argparse
import json
import shutil
from pathlib import Path

import av
import h5py
import numpy as np

ACTION_KEEP = [i for i in range(23) if i != 6]  # 丢 dim6
DROPPED_ACTION = {"index": 6, "deploy_value": 0.0, "reason": "trunk 准常数维，std=4.65e-4"}

ACTION_NAMES = (
    ["base_vx", "base_vy", "base_vyaw"]
    + [f"trunk_{i}" for i in [0, 1, 2]]          # 原 3,4,5（原 6 丢弃）
    + [f"arm_left_{i}" for i in range(7)]
    + ["gripper_left"]
    + [f"arm_right_{i}" for i in range(7)]
    + ["gripper_right"]
)
STATE_NAMES = (
    [f"jp_{i}" for i in range(28)]
    + [f"finger_{s}" for s in ["L0", "L1", "R0", "R1"]]
    + ["robot_x", "robot_y", "robot_z"]
    + ["robot_qx", "robot_qy", "robot_qz", "robot_qw"]
    + ["rel_human_x", "rel_human_y", "rel_human_yaw"]
    + ["human_visible", "head_pitch_deg"]
)


def decode_frames(path, resize=None):
    """顺序解码 mp4，yield uint8 HWC RGB。"""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if resize:
                frame = frame.reformat(width=resize[0], height=resize[1], format="rgb24")
                yield frame.to_ndarray()
            else:
                yield frame.to_ndarray(format="rgb24")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--repo-id", default="hrc/hrc_v01_step10")
    ap.add_argument("--third-size", default="640x360")
    ap.add_argument("--cameras", default="head,third",
                    help="逗号分隔的图像输入。部署realistic（真机只有机载相机）用 'head' "
                         "→ 不把上帝视角第三人称喂给策略。")
    ap.add_argument("--rounds", default=None,
                    help="逗号分隔的 round_id（=demo_id）白名单，只导出这些轮。默认全部。"
                         "例: '0' 只要收音机轮。")
    ap.add_argument("--fsm-phases", default=None,
                    help="逗号分隔的 fsm_state_id 白名单，只导出这些相位帧。默认全部。"
                         "例: '3' 只要 ExecuteBehaviorTask（操作段）。相位表见 §5.2。")
    ap.add_argument("--action-shift", type=int, default=0,
                    help="监督动作相对观测向后偏移多少帧。默认 0 = 旧行为 obs[i]->action[i]；"
                         "设为 1 = causal 对照 obs[i]->action[i+1]，会丢弃 demo/phase 边界处无法配对的帧。")
    args = ap.parse_args()
    if args.action_shift < 0:
        raise ValueError("--action-shift 目前只支持非负整数")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    if "head" not in cams:
        raise ValueError("--cameras 必须至少包含 head（机载相机是部署时唯一可用视觉）")
    use_third = "third" in cams
    print(f"[export] cameras={cams} (use_third={use_third})")

    tw, th = (int(x) for x in args.third_size.split("x"))

    cn = h5py.File(args.canonical / "canonical.hdf5", "r")
    fr = cn["frames"]
    instructions = {c["round_id"]: c for c in json.loads(cn.attrs["instructions"])}
    A = np.asarray(fr["action"][:], dtype=np.float32)
    demo_id = np.asarray(fr["demo_id"][:])
    fsm = np.asarray(fr["obs/fsm_state_id"][:]).ravel().astype(int)
    obs = {k: np.asarray(fr[f"obs/{k}"][:]) for k in
           ["joint_positions", "gripper_state", "robot_pos", "robot_quat",
            "relative_human_pose", "human_visible", "head_camera_pitch_deg"]}

    # ---- Frame selection (round + fsm-phase whitelist).
    sel_rounds = (None if args.rounds is None
                  else {int(x) for x in args.rounds.split(",") if x.strip() != ""})
    sel_phases = (None if args.fsm_phases is None
                  else {int(x) for x in args.fsm_phases.split(",") if x.strip() != ""})
    keep = np.ones(len(demo_id), dtype=bool)
    if sel_rounds is not None:
        keep &= np.isin(demo_id.astype(int), list(sel_rounds))
    if sel_phases is not None:
        keep &= np.isin(fsm, list(sel_phases))
    raw_keep = keep.copy()
    kept_idx = np.where(raw_keep)[0]
    if len(kept_idx) == 0:
        raise ValueError(f"selection empty: rounds={sel_rounds} phases={sel_phases}")

    action_src_idx = np.arange(len(demo_id), dtype=np.int64) + int(args.action_shift)
    shift_valid = action_src_idx < len(demo_id)
    if args.action_shift:
        safe_action_src_idx = np.minimum(action_src_idx, len(demo_id) - 1)
        shift_valid &= raw_keep[safe_action_src_idx]
        shift_valid &= demo_id.astype(int) == demo_id[safe_action_src_idx].astype(int)
    keep = raw_keep & shift_valid
    kept_idx = np.where(keep)[0]
    if len(kept_idx) == 0:
        raise ValueError(f"selection empty after action_shift={args.action_shift}: "
                         f"rounds={sel_rounds} phases={sel_phases}")
    dropped_for_shift = int(raw_keep.sum() - keep.sum())
    last_kept = int(kept_idx[-1])
    shift_note = (f" (action_shift={args.action_shift}, dropped_for_shift={dropped_for_shift})"
                  if args.action_shift else "")
    print(f"[export] selection rounds={sel_rounds} phases={sel_phases} -> "
          f"{int(keep.sum())}/{len(keep)} 帧" + shift_note)

    def col(x):  # (T,) -> (T,1)
        x = np.asarray(x, dtype=np.float32)
        return x[:, None] if x.ndim == 1 else x

    STATE = np.concatenate(
        [col(obs["joint_positions"]), col(obs["gripper_state"]), col(obs["robot_pos"]),
         col(obs["robot_quat"]), col(obs["relative_human_pose"]),
         col(obs["human_visible"]), col(obs["head_camera_pitch_deg"])], axis=1
    ).astype(np.float32)
    assert STATE.shape[1] == len(STATE_NAMES), STATE.shape
    ACT = A[:, ACTION_KEEP]
    assert ACT.shape[1] == len(ACTION_NAMES)

    # std 体检（mean_std 归一化的分母健康度）
    for nm, arr in [("state", STATE), ("action", ACT)]:
        std = arr.std(0)
        tiny = np.where(std < 1e-6)[0]
        print(f"[std-check] {nm}: min_std={std.min():.2e}" +
              (f" 恒定维 {list(tiny)}（mean_std 下归一化为 0，无 NaN 风险）" if len(tiny) else " 无恒定维"))

    if args.root.exists():
        shutil.rmtree(args.root)

    features = {
        "observation.images.head": {"dtype": "video", "shape": (256, 352, 3),
                                    "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (len(ACTION_NAMES),), "names": ACTION_NAMES},
    }
    if use_third:
        features["observation.images.third"] = {"dtype": "video", "shape": (th, tw, 3),
                                                "names": ["height", "width", "channels"]}
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=30, root=args.root,
                               features=features, use_videos=True, video_backend="pyav")

    head_it = decode_frames(args.canonical / "videos" / "head_rgb.mp4")
    third_it = (decode_frames(args.canonical / "videos" / "third_person_rgb.mp4", resize=(tw, th))
                if use_third else None)

    # Global sequential decode: must advance the mp4 decoder over EVERY frame in
    # order (so it stays aligned), but only add_frame the selected ones. Kept
    # frames are grouped into one LeRobot episode per demo (contiguous runs).
    n_frames = 0
    cur_demo = None
    ep_counts = {}
    for gi in range(len(demo_id)):
        if gi > last_kept:
            break
        head = next(head_it)
        third = next(third_it) if use_third else None
        if not keep[gi]:
            continue
        di = int(demo_id[gi])
        if cur_demo is not None and di != cur_demo:
            ds.save_episode()
        cur_demo = di
        frame = {
            "observation.images.head": head,
            "observation.state": STATE[gi],
            "action": ACT[action_src_idx[gi]],
            "task": instructions[di]["instruction"],
        }
        if use_third:
            frame["observation.images.third"] = third
        ds.add_frame(frame)
        n_frames += 1
        ep_counts[di] = ep_counts.get(di, 0) + 1
    if cur_demo is not None:
        ds.save_episode()
    print(f"[export] episodes(per demo) 帧数: {ep_counts}")

    # 自定义元数据：动作维映射（部署重插 dim6 必需）
    extra = {
        "source_canonical": str(args.canonical),
        "action_mapping": {"kept_source_dims": ACTION_KEEP, "dropped": DROPPED_ACTION,
                           "names": ACTION_NAMES},
        "state_layout": STATE_NAMES,
        "cameras": cams,
        "action_shift": int(args.action_shift),
        "action_target": "action[source_frame + action_shift]",
        "deploy_realistic": (not use_third),  # head-only = 真机只有机载相机
        "gripper": {"values": "±1 binary", "polarity": "+1=open, -1=close（canonical 实测）"},
        "normalization_note": "base[-1,1] 与 arm(rad) 混量纲，必须 per-dim 归一化（ACT 默认 mean_std 即 per-dim）",
    }
    with open(args.root / "meta" / "hrc_adapter_meta.json", "w") as f:
        json.dump(extra, f, ensure_ascii=False, indent=1)

    cn.close()
    print(f"=== 完成: {n_frames} 帧, {ds.meta.total_episodes} episodes -> {args.root} ===")


if __name__ == "__main__":
    main()
