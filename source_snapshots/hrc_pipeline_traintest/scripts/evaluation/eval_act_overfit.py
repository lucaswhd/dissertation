#!/usr/bin/env python
"""ACT overfit 离线评估：在训练数据上比较 预测 action chunk vs 真值。

指标：
  - 原始单位逐组平均|误差|（base / trunk / arm_L / arm_R，rad 或 [-1,1] 速度单位）
  - 夹爪符号一致率（二值 ±1 语义下唯一有意义的指标）
  - 按 FSM 相位拆解的总体 L1（用 canonical 的 fsm_state_id 映射回去）
  - chunk 内 padding 帧用 action_is_pad 掩掉

判定参考（overfit 验收，不是泛化指标）：
  arms/trunk mean|err| < 0.02 rad（≈1.1°）、base < 0.02、夹爪符号一致率 ≥ 99.5%
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import default_collate

FSM_NAMES = ["WaitNearHuman", "ReceiveCommand", "GoToTask", "ExecuteBehaviorTask",
             "TaskSegmentDone", "ReturnToHuman", "FaceHuman", "WaitNextCommand",
             "AllTasksDone", "Relocate"]

GROUPS = {  # 22 维 adapter 动作空间内的切片
    "base(vel)": slice(0, 3),
    "trunk(rad)": slice(3, 6),
    "arm_L(rad)": slice(6, 13),
    "grip_L": slice(13, 14),
    "arm_R(rad)": slice(14, 21),
    "grip_R": slice(21, 22),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path, help=".../checkpoints/NNNNNN/pretrained_model")
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--repo-id", default="hrc/hrc_v01_step10")
    ap.add_argument("--canonical", required=True, type=Path)
    ap.add_argument("--n-samples", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=12)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = ACTPolicy.from_pretrained(args.ckpt)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(args.ckpt))

    chunk = policy.config.chunk_size
    ds = LeRobotDataset(repo_id=args.repo_id, root=args.root, video_backend="pyav",
                        delta_timestamps={"action": [i / 30.0 for i in range(chunk)]})
    with h5py.File(args.canonical / "canonical.hdf5", "r") as cn:
        fsm = np.asarray(cn["frames/obs/fsm_state_id"][:]).astype(int).ravel()
    N = len(ds)

    idxs = np.unique(np.linspace(0, N - 1, args.n_samples).astype(int))
    rows = []
    for b0 in range(0, len(idxs), args.batch_size):
        sel = idxs[b0:b0 + args.batch_size]
        items = [ds[int(i)] for i in sel]
        gt = torch.stack([it["action"] for it in items])            # (B, chunk, 22) 原始单位
        pad = torch.stack([it["action_is_pad"] for it in items])    # (B, chunk)
        batch = default_collate([{k: v for k, v in it.items()
                                  if not k.startswith("action")} for it in items])
        batch = pre(batch)
        with torch.no_grad():
            pred_n = policy.predict_action_chunk(batch)             # 归一化空间
            pred = post(pred_n)                                     # 原始单位
        pred = pred.cpu()
        for j, gi in enumerate(sel):
            valid = ~pad[j]
            rows.append({"gi": int(gi), "fsm": int(fsm[gi]),
                         "err": (pred[j][valid] - gt[j][valid]).abs().numpy(),
                         "pred": pred[j][valid].numpy(), "gt": gt[j][valid].numpy()})

    E = np.concatenate([r["err"] for r in rows])      # (ΣT, 22)
    P = np.concatenate([r["pred"] for r in rows])
    G = np.concatenate([r["gt"] for r in rows])
    print(f"=== ACT overfit 评估 @ {args.ckpt} ===")
    print(f"样本窗口 {len(rows)} 个（覆盖两 episode），有效 (帧,维) = {E.shape}")
    verdicts = []
    for name, sl in GROUPS.items():
        if name.startswith("grip"):
            agree = float((np.sign(P[:, sl]) == np.sign(G[:, sl])).mean())
            ok = agree >= 0.995
            verdicts.append(ok)
            print(f"  {name}: 符号一致率={agree*100:.2f}% {'✓' if ok else '✗'}")
        else:
            m = float(E[:, sl].mean()); mx = float(E[:, sl].max())
            ok = m < 0.02
            verdicts.append(ok)
            extra = f"（≈{np.degrees(m):.2f}°）" if "rad" in name else ""
            print(f"  {name}: mean|err|={m:.4f}{extra} max={mx:.3f} {'✓' if ok else '✗'}")
    print("  --- 按 FSM 相位（总体 mean|err|，22 维平均）---")
    for ph in sorted(set(r["fsm"] for r in rows)):
        sub = np.concatenate([r["err"] for r in rows if r["fsm"] == ph])
        print(f"  {FSM_NAMES[ph]:>20}: {float(sub.mean()):.4f}  (n_win={sum(1 for r in rows if r['fsm']==ph)})")
    print(f"=== 结论: {'全部通过 — 单 episode overfit 成立' if all(verdicts) else '存在未达标组，见上'} ===")
    print(json.dumps({"per_dim_mean_err": [float(x) for x in E.mean(0)]}, indent=None))


if __name__ == "__main__":
    main()
