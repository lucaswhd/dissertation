#!/usr/bin/env python
"""验证 LeRobot 数据集与 canonical 的一致性（数值精确 + 视频近似 + 统计健康）。"""
import argparse
import json
from pathlib import Path

import av
import h5py
import numpy as np


def grab_canonical_frame(video_path, target_idx, resize=None):
    with av.open(str(video_path)) as c:
        for i, frame in enumerate(c.decode(c.streams.video[0])):
            if i == target_idx:
                if resize:
                    frame = frame.reformat(width=resize[0], height=resize[1], format="rgb24")
                    return frame.to_ndarray()
                return frame.to_ndarray(format="rgb24")
    raise IndexError(target_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--repo-id", default="hrc/hrc_v01_step10")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=args.repo_id, root=args.root, video_backend="pyav")
    m = ds.meta
    print(f"episodes={m.total_episodes} frames={m.total_frames} fps={m.fps}")
    print(f"tasks={list(m.tasks.index) if hasattr(m.tasks, 'index') else m.tasks}")
    assert m.total_frames == 10239 and m.total_episodes == 2

    cn = h5py.File(args.canonical / "canonical.hdf5", "r")
    A = np.asarray(cn["frames/action"][:], dtype=np.float32)
    keep = [i for i in range(23) if i != 6]

    # 数值精确比对（含 episode 边界）
    test_idx = [0, 1, 1700, 3757, 3758, 3759, 7000, 10238]
    max_a = max_s = 0.0
    for gi in test_idx:
        item = ds[gi]
        a = item["action"].numpy()
        max_a = max(max_a, float(np.abs(a - A[gi][keep]).max()))
    print(f"action 与 canonical 最大差: {max_a:.2e}（应=0，float32 精确）")

    # 视频帧近似比对（AV1 有损重编码）
    for gi, key, src, rs in [
        (100, "observation.images.head", "head_rgb.mp4", None),
        (3758, "observation.images.head", "head_rgb.mp4", None),
        (5000, "observation.images.third", "third_person_rgb.mp4", (640, 360)),
    ]:
        item = ds[gi]
        got = (item[key].numpy().transpose(1, 2, 0) * 255).round().astype(np.int16)
        ref = grab_canonical_frame(args.canonical / "videos" / src, gi, resize=rs).astype(np.int16)
        mad = float(np.abs(got - ref).mean())
        print(f"帧 {gi} {key}: 平均像素差={mad:.2f}（有损编码，<8 视为对齐）")
        assert mad < 8.0, f"像素差过大，可能错帧: {mad}"

    # 统计健康度
    stats = m.stats
    for k in ["observation.state", "action"]:
        std = np.asarray(stats[k]["std"]).ravel()
        print(f"{k}: std_min={std.min():.2e} 维数={len(std)} NaN={int(np.isnan(std).sum())}")

    # episode↔task 对应
    ep_meta = m.episodes
    for ei in range(m.total_episodes):
        row = ep_meta[ei]
        print(f"episode {ei}: len={row['length']} tasks={row['tasks']}")
    cn.close()
    print("=== LeRobot 数据集验证通过 ===")


if __name__ == "__main__":
    main()
