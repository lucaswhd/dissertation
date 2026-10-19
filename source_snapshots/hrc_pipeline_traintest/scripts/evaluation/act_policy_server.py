#!/usr/bin/env python
# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
# See LICENSE in the project root for full terms.

"""ACT checkpoint inference server — run inside **lerobot-venv**.

跨 venv 方案（HANDOFF §12.6.1）：OmniGibson 闭环在 behavior-venv 里跑，ACT 推理在
lerobot-venv 里以独立进程提供服务，两者通过 localhost TCP 通信，互不污染依赖。

协议：length-prefixed pickle（4 字节大端 uint32 长度 + pickle bytes），单连接串行。
请求:
  {"type": "ping"}                            -> {"ok": True, "msg": "pong", ...}
  {"type": "reset"}                           -> {"ok": True}   # 清空 action 队列（episode 边界）
  {"type": "act", "state": f32[44],
   "head": u8[256,352,3], "third": u8[360,640,3],
   "task": str}                               -> {"ok": True, "action": f32[22]}
  {"type": "shutdown"}                        -> {"ok": True}   # 服务退出

ACT n_action_steps=100：select_action 内部维护队列，每 100 帧才真正前向一次。
图像在客户端以 uint8 HWC RGB 发送，本服务转 float CHW [0,1]（与 LeRobotDataset
训练时的图像约定一致），归一化由 checkpoint 自带的 pre/post processor 完成。

用法（lerobot-venv）:
  /root/autodl-tmp/lerobot-venv/bin/python scripts/act_policy_server.py \
      --ckpt /root/autodl-tmp/act_runs/hrc_v01_overfit_01/checkpoints/020000/pretrained_model
"""
import argparse
import pickle
import socket
import struct
import time
import traceback
from pathlib import Path

import numpy as np
import torch


# --- numpy-version-agnostic wire codec ---------------------------------------
# CRITICAL: server (lerobot-venv) runs numpy 2.x, client (behavior-venv) runs
# numpy 1.x. A raw ``pickle.dumps(np.ndarray)`` embeds the internal module path
# (``numpy._core.*`` on 2.x), which 1.x cannot import -> ModuleNotFoundError on
# the client. So we NEVER pickle ndarrays directly: pack() turns every array
# into {raw bytes + dtype str + shape}, which pickles identically on any numpy.
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


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed connection")
        buf += chunk
    return buf


def recv_msg(conn: socket.socket) -> dict:
    (length,) = struct.unpack(">I", _recv_exact(conn, 4))
    return _unpack(pickle.loads(_recv_exact(conn, length)))


def send_msg(conn: socket.socket, obj: dict) -> None:
    payload = pickle.dumps(_pack(obj), protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack(">I", len(payload)) + payload)


def _img_to_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    """uint8 HWC RGB -> float32 (1, C, H, W) in [0, 1]."""
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    t = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
    return t.permute(2, 0, 1).unsqueeze(0).float() / 255.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path,
                    help=".../checkpoints/NNNNNN/pretrained_model")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18765)
    args = ap.parse_args()

    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[act-server] loading {args.ckpt} on {device} ...", flush=True)
    policy = ACTPolicy.from_pretrained(args.ckpt)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(policy_cfg=policy.config,
                                         pretrained_path=str(args.ckpt))
    chunk = int(policy.config.chunk_size)
    n_act = int(policy.config.n_action_steps)
    img_inputs = [k for k in policy.config.input_features if k.startswith("observation.images")]
    print(f"[act-server] ready: chunk_size={chunk}, n_action_steps={n_act}, "
          f"input_features={list(policy.config.input_features)}, "
          f"image_inputs={img_inputs}", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[act-server] listening on {args.host}:{args.port}", flush=True)

    n_queries = 0
    t_infer = 0.0
    running = True
    while running:
        conn, addr = srv.accept()
        print(f"[act-server] client connected: {addr}", flush=True)
        try:
            while True:
                msg = recv_msg(conn)
                mtype = msg.get("type")
                if mtype == "ping":
                    send_msg(conn, {"ok": True, "msg": "pong",
                                    "chunk_size": chunk, "n_action_steps": n_act,
                                    "image_inputs": img_inputs})
                elif mtype == "reset":
                    policy.reset()
                    send_msg(conn, {"ok": True})
                elif mtype == "shutdown":
                    send_msg(conn, {"ok": True})
                    running = False
                    break
                elif mtype == "act":
                    t0 = time.time()
                    try:
                        # Only build the image keys this checkpoint actually
                        # declares (head-only vs head+third), ignoring extras the
                        # client may have sent. Keeps the server checkpoint-agnostic.
                        batch = {
                            "observation.state": torch.from_numpy(
                                np.asarray(msg["state"], dtype=np.float32)
                            ).unsqueeze(0).to(device),
                            "task": [str(msg.get("task", ""))],
                        }
                        if "observation.images.head" in img_inputs:
                            batch["observation.images.head"] = _img_to_tensor(msg["head"], device)
                        if "observation.images.third" in img_inputs and msg.get("third") is not None:
                            batch["observation.images.third"] = _img_to_tensor(msg["third"], device)
                        batch = pre(batch)
                        with torch.no_grad():
                            action_n = policy.select_action(batch)
                        action = post(action_n)
                        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
                        n_queries += 1
                        t_infer += time.time() - t0
                        if n_queries % 300 == 0:
                            print(f"[act-server] {n_queries} queries, "
                                  f"avg {1e3 * t_infer / n_queries:.1f} ms", flush=True)
                        send_msg(conn, {"ok": True, "action": action_np})
                    except Exception as exc:
                        traceback.print_exc()
                        send_msg(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                else:
                    send_msg(conn, {"ok": False, "error": f"unknown type {mtype!r}"})
        except (ConnectionError, EOFError) as exc:
            print(f"[act-server] client gone: {exc}", flush=True)
        finally:
            conn.close()
    srv.close()
    print(f"[act-server] shutdown after {n_queries} queries", flush=True)


if __name__ == "__main__":
    main()
