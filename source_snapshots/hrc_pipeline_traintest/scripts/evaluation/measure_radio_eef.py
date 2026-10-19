#!/usr/bin/env python
# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# See LICENSE in the project root for full terms.
"""Teacher-forced FK 扫描：把机器人逐帧摆成 step10 收音机执行段的 GT 构型（不动作
驱动、无开环漂移），读真实 eef 世界坐标，定位「演示按压手」最接近 radio_89 的点。

目的（重设计第二步的前置测量）：知道演示手势在我们场景里实际经过哪儿，才能把
radio_89 的按钮/高度对齐到那条轨迹上（option 1：对齐优先）。

跑在 behavior-venv。输出 [eef-scan] 行 + 收尾 SUMMARY。
"""
import argparse
import os
import sys
from pathlib import Path

# 脚本重组后本地模块在 scripts/<category>/，照搬 shim 的 path 设置。
_SCRIPTS = Path(__file__).resolve().parent.parent
for _c in ["collection", "preprocess", "training", "evaluation", "debug", "legacy"]:
    _p = str(_SCRIPTS / _c)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch as th

import omnigibson as og


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hrc_multitask_v01.yaml")
    ap.add_argument("--snapshot", default="/root/autodl-tmp/hrc_snapshots/hrc_multitask_v01_step10")
    ap.add_argument("--stride", type=int, default=20, help="每隔几帧测一次（FK 慢，默认 20）")
    ap.add_argument("--radio-z-offset", type=float, default=0.0,
                    help="对齐后把 radio 抬高这么多米（option 1：补我们桌子矮的 ~0.10m），"
                         "让演示按压手指真接触按钮 → 物理触发 ToggledOn。")
    ap.add_argument("--use-loadstate", action="store_true",
                    help="用 og.sim.load_state 逐帧精确还原 task-0000 全状态(=采集同法)，"
                         "而非 set_joint_positions 近似摆位。量真实采集几何下的指尖→按钮距离。")
    args = ap.parse_args()

    import yaml, h5py
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    raw_hdf5 = str(cfg["raw_hdf5"]); resolution = int(cfg.get("resolution", 256))
    extra_objects = list(cfg.get("extra_objects", []))
    task_rounds = list(cfg.get("task_rounds", []))
    episode_id = int(cfg.get("episode_id", 0))

    from nav_online_driver import build_contacts_on_playback_wrapper
    from hrc_utils import spawn_extra_object, to_numpy

    # GT 执行段构型（demo_0, fsm==3）
    with h5py.File(os.path.join(args.snapshot, "hrc_dataset.hdf5"), "r") as fsnap:
        d = fsnap["data/demo_0"]
        fsm = np.asarray(d["obs/fsm_state_id"][:]).ravel()
        sel = np.where(fsm == 3)[0]
        gtJ = np.asarray(d["obs/joint_positions"][:])[sel]
        gtP = np.asarray(d["obs/robot_pos"][:])[sel]
        gtQ = np.asarray(d["obs/robot_quat"][:])[sel]
    print(f"[eef-scan] GT 执行段 {len(gtJ)} 帧，stride={args.stride}", flush=True)

    out = "/root/autodl-tmp/outputs/hrc_eef_scan_tmp"
    os.makedirs(out, exist_ok=True)
    env = build_contacts_on_playback_wrapper(
        raw_hdf5, output_hdf5_path=os.path.join(out, f"cache_{os.getpid()}.hdf5"),
        resolution=resolution, robot_modalities=["rgb"])
    try:
        env.reset()
        robot = env.robots[0]
        _grp = env.input_hdf5["data"][f"demo_{episode_id}"]
        og.sim.load_state(th.tensor(_grp["state"][0][:int(_grp["state_size"][0])]), serialized=True)

        # spawn + 对齐 radio_89（和 rollout 一致）
        spawned = {}
        for spec in extra_objects:
            spawned[spec["name"]] = spawn_extra_object(
                env, name=spec["name"], category=spec["category"], model=spec["model"],
                position=spec["position"], orientation=spec.get("orientation", [0, 0, 0, 1]),
                in_rooms=spec.get("in_rooms"))
        for rc in task_rounds:
            if rc.get("kind") != "replay_external_segment" or not spawned:
                continue
            ext = rc.get("external_hdf5"); es = int(rc.get("start_step", 0))
            snap = og.sim.dump_state(serialized=True)
            with h5py.File(ext, "r") as f:
                tr = f[f"data/demo_{int(rc.get('episode_id',0))}"]
                og.sim.load_state(th.tensor(tr["state"][es, :int(tr["state_size"][es])], dtype=th.float32), serialized=True)
            for nm, obj in spawned.items():
                obj._p = to_numpy(obj.get_position_orientation()[0]).copy()
                obj._q = to_numpy(obj.get_position_orientation()[1]).copy()
            og.sim.load_state(snap, serialized=True)
            for nm, obj in spawned.items():
                p = obj._p.copy()
                p[2] += float(args.radio_z_offset)   # option 1：抬高对齐
                obj.set_position_orientation(p, obj._q); obj.keep_still()
        radio = spawned.get("radio_89")
        radio_c = to_numpy(radio.get_position_orientation()[0])
        from omnigibson.object_states import ToggledOn as TON
        has_ton = TON in radio.states
        mk = to_numpy(radio.states[TON].link.get_position_orientation()[0]) if has_ton else radio_c
        print(f"[eef-scan] radio_z_offset={args.radio_z_offset} radio 中心={np.round(radio_c,3).tolist()} "
              f"按钮marker={np.round(mk,3).tolist()}", flush=True)
        ever_toggled = [False]; first_toggle = [None]
        # 手指 link（接触判定用的就是这些）
        finger_links = []
        try:
            for a in robot.arm_names:
                finger_links += list(robot.finger_links[a])
        except Exception:
            pass

        try:
            base_idx = set(int(i) for i in robot.base_idx.tolist())
        except Exception:
            base_idx = set()
        non_base = [i for i in range(gtJ.shape[1]) if i not in base_idx]

        # 逐帧把 radio 钉在抬高位（实际采集也钉固；否则悬空会被重力掉回桌面，污染接触测试）
        radio_pin_p = to_numpy(radio.get_position_orientation()[0]).copy()
        radio_pin_q = to_numpy(radio.get_position_orientation()[1]).copy()

        # load_state 模式：打开 task-0000 的 state 记录（采集就是逐帧 load 它）
        ext_state = ext_sz = None
        if args.use_loadstate:
            rc0 = next(r for r in task_rounds if r.get("kind") == "replay_external_segment")
            f0000 = h5py.File(rc0["external_hdf5"], "r")
            ed = f0000[f"data/demo_{int(rc0.get('episode_id',0))}"]
            ext_state = ed["state"]; ext_sz = ed["state_size"]
            print(f"[eef-scan] use_loadstate: 逐帧 og.sim.load_state(task-0000 state[k]) "
                  f"(共 {ext_state.shape[0]} 步)", flush=True)

        best = {a: (1e9, None, None) for a in robot.arm_names}
        for k in range(0, len(gtJ), args.stride):
            if args.use_loadstate and ext_state is not None and k < ext_state.shape[0]:
                # 采集同法：load_state 精确还原 task-0000 全状态(机器人+物体)
                og.sim.load_state(th.tensor(ext_state[k, :int(ext_sz[k])], dtype=th.float32), serialized=True)
                # radio_89 钉回对齐位(采集也钉固 task-specific 物体)
                radio.set_position_orientation(radio_pin_p, radio_pin_q); radio.keep_still()
            else:
                jp = th.tensor(gtJ[k], dtype=th.float32)
                if base_idx:
                    robot.set_joint_positions(jp[non_base], indices=th.tensor(non_base, dtype=th.long))
                else:
                    robot.set_joint_positions(jp)
                robot.set_position_orientation(th.tensor(gtP[k], dtype=th.float32), th.tensor(gtQ[k], dtype=th.float32))
            robot.keep_still()
            # 完整 og.sim.step() 才跑对象状态更新(含 ToggledOn 的 finger-contact 判定)。
            # 手指贴上后多步几次让接触冲量积累；每步前把 radio 钉回抬高位（防掉落）。
            for _ in range(3):
                radio.set_position_orientation(radio_pin_p, radio_pin_q)
                radio.keep_still()
                og.sim.step()
            # 物理接触触发 ToggledOn?（teacher-forced、无 load_state → True 必是真接触）
            if has_ton:
                try:
                    if bool(radio.states[TON].get_value()):
                        ever_toggled[0] = True
                        if first_toggle[0] is None:
                            first_toggle[0] = int(sel[k])
                except Exception:
                    pass
            # 手指到 radio 中心最近距离
            fmin = 1e9
            for fl in finger_links:
                try:
                    fp = to_numpy(fl.get_position_orientation()[0])
                    fmin = min(fmin, float(np.linalg.norm(fp - radio_c)))
                except Exception:
                    pass
            line = []
            for a in robot.arm_names:
                eef = to_numpy(robot.get_eef_position(a))
                dm = float(np.linalg.norm(eef - mk))
                if dm < best[a][0]:
                    best[a] = (dm, int(sel[k]), eef.copy())
                line.append(f"{a}={dm:.2f}")
            if k % (args.stride*5) == 0:
                print(f"[eef-scan] f{int(sel[k])} eef->btn {' '.join(line)} finger->radio={fmin:.3f} "
                      f"ToggledOn={bool(radio.states[TON].get_value()) if has_ton else 'n/a'}", flush=True)
        print(f"\n[eef-scan] SUMMARY radio_z_offset={args.radio_z_offset} 按钮marker={np.round(mk,3).tolist()}", flush=True)
        for a, (dm, fr, eef) in best.items():
            print(f"[eef-scan]   {a}: 最近按钮 {dm:.3f}m @f{fr} eef={np.round(eef,3).tolist()}", flush=True)
        print(f"[eef-scan]   *** 物理接触触发 ToggledOn: ever={ever_toggled[0]} first@f{first_toggle[0]} ***", flush=True)
    finally:
        try: og.shutdown()
        except Exception: pass


if __name__ == "__main__":
    main()
