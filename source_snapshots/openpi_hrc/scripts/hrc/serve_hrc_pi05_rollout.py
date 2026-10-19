#!/usr/bin/env python
"""pi0.5 HRC policy inference server for closed-loop rollout.

Speaks the SAME length-prefixed pickle TCP protocol as the pipeline's
`scripts/evaluation/act_policy_server.py`, so the existing behavior-venv rollout
client can drive our openpi pi0.5 policy with minimal changes. Runs inside the
openpi_hrc `.venv`; the OmniGibson sim runs in behavior-venv; they talk over
localhost TCP (cross-venv, numpy-version-agnostic wire codec copied verbatim).

Protocol (superset of the ACT server):
  {"type":"ping"}   -> {"ok":True,"msg":"pong","action_horizon":32,
                        "action_dim":23,"execute_horizon":H}
  {"type":"reset"}  -> {"ok":True}   # episode boundary: drop the cached plan
  {"type":"act", "state": f32[23], "head": u8[H,W,3],
   "prompt": str | ("task": str [, "phase": str])} -> {"ok":True,"action": f32[23]}
  {"type":"shutdown"} -> {"ok":True}

Chunking (handoff §22): each `act` returns ONE 23-dim action. The server keeps a
queue of the first `execute_horizon` (default 8) steps of the last 32-step
prediction; when the queue drains it re-runs policy.infer on the CURRENT obs
(= predict 32, execute first 8, re-observe). `reset` clears the queue.

Usage (openpi_hrc/.venv):
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python scripts/hrc/serve_hrc_pi05_rollout.py \
      --config pi05_hrc_e4_main \
      --checkpoint /root/autodl-tmp/openpi_hrc/checkpoints/pi05_hrc_e4_main/hrc_e4_seed42/5000
"""

import argparse
import collections
import pickle
import socket
import struct
import time
import traceback
from pathlib import Path

import numpy as np


ACTION_HORIZON = 32
ACTION_DIM = 23
EXECUTE_HORIZON = 8


def canonical_checkpoint_path(value: Path | str) -> Path:
    """Return one stable checkpoint identity for logs and run-plan matching."""

    return Path(value).expanduser().resolve()


def validate_execute_horizon(value: int) -> int:
    """Steps executed per 32-step plan.

    The frozen HRC protocol is 8 (see HANDOFF 17.2); anything else is a
    deliberate replan-rate experiment, so it is allowed but announced. The
    client re-checks this value in its handshake, so both ends must agree and
    a run can never silently mix rates.
    """
    execute_horizon = int(value)
    if not 1 <= execute_horizon <= ACTION_HORIZON:
        raise ValueError(
            f"execute_horizon must be within [1, {ACTION_HORIZON}], got {execute_horizon}"
        )
    if execute_horizon != EXECUTE_HORIZON:
        print(
            f"[pi05-server] WARNING: execute_horizon={execute_horizon} deviates from the "
            f"frozen HRC protocol ({EXECUTE_HORIZON}); label these runs as a replan-rate "
            f"experiment and do not pool them with protocol-compliant results.",
            flush=True,
        )
    return execute_horizon


def validate_act_request(message: dict) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(message.get("state"), dtype=np.float32)
    if state.shape != (ACTION_DIM,):
        raise ValueError(f"expected state shape (23,), got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("state must be finite")

    head = np.asarray(message.get("head"))
    if head.ndim != 3 or head.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB head image, got {head.shape}")
    if not np.isfinite(head).all():
        raise ValueError("head image must be finite")
    if np.issubdtype(head.dtype, np.floating):
        scale = 255.0 if head.size and float(head.max()) <= 1.0 else 1.0
        head = np.clip(head * scale, 0.0, 255.0).astype(np.uint8)
    else:
        head = np.clip(head, 0, 255).astype(np.uint8)
    return state, head


def validate_action_chunk(value) -> np.ndarray:
    actions = np.asarray(value, dtype=np.float32)
    if actions.shape != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError(f"expected actions shape [32, 23], got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("policy action chunk must be finite")
    return actions

# --- numpy-version-agnostic wire codec (copied verbatim from act_policy_server) --
def _pack(obj):
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {"__nd__": True, "data": arr.tobytes(), "dtype": str(arr.dtype), "shape": list(arr.shape)}
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


def _build_prompt(msg: dict) -> str:
    prompt = msg.get("prompt")
    if prompt:
        return str(prompt)
    task = str(msg.get("task", ""))
    phase = msg.get("phase")
    return f"Task: {task} Phase: {phase}." if phase else task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_hrc_e4_main", help="openpi train config name")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/root/autodl-tmp/openpi_hrc/checkpoints/pi05_hrc_e4_main/hrc_e4_seed42/5000"),
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18765)
    ap.add_argument("--execute-horizon", type=int, default=8, help="steps executed per 32-step plan")
    args = ap.parse_args()
    args.checkpoint = canonical_checkpoint_path(args.checkpoint)

    from openpi.policies.policy_config import create_trained_policy
    from openpi.training.config import get_config

    print(f"[pi05-server] loading {args.config} @ {args.checkpoint} ...", flush=True)
    policy = create_trained_policy(get_config(args.config), args.checkpoint)
    exec_h = validate_execute_horizon(args.execute_horizon)
    plan_queue: collections.deque = collections.deque()
    print(f"[pi05-server] ready: config={args.config}, execute_horizon={exec_h}", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[pi05-server] listening on {args.host}:{args.port}", flush=True)

    n_queries = 0
    n_infer = 0
    t_infer = 0.0
    running = True
    while running:
        conn, addr = srv.accept()
        print(f"[pi05-server] client connected: {addr}", flush=True)
        try:
            while True:
                msg = recv_msg(conn)
                mtype = msg.get("type")
                if mtype == "ping":
                    send_msg(conn, {"ok": True, "msg": "pong", "action_horizon": 32,
                                    "action_dim": 23, "execute_horizon": exec_h,
                                    "config": args.config,
                                    "checkpoint": str(args.checkpoint)})
                elif mtype == "reset":
                    plan_queue.clear()
                    seed = msg.get("seed")
                    if seed is not None:
                        import jax

                        # openpi Policy owns the sampling key. Resetting it at an
                        # episode boundary makes the two Gate-3 policy seeds
                        # reproducible without rebuilding checkpoint weights.
                        if not hasattr(policy, "_rng"):
                            raise RuntimeError("loaded policy does not expose a JAX sampling RNG")
                        policy._rng = jax.random.key(int(seed))
                    send_msg(conn, {"ok": True, "seed": seed})
                elif mtype == "shutdown":
                    send_msg(conn, {"ok": True})
                    running = False
                    break
                elif mtype == "act":
                    try:
                        if not plan_queue:
                            state, head = validate_act_request(msg)
                            t0 = time.time()
                            result = policy.infer(
                                {
                                    "observation/image_0": head,
                                    "observation/state": state,
                                    "prompt": _build_prompt(msg),
                                }
                            )
                            actions = validate_action_chunk(result["actions"])
                            for step in actions[:exec_h]:
                                plan_queue.append(step)
                            n_infer += 1
                            t_infer += time.time() - t0
                        action = plan_queue.popleft()
                        n_queries += 1
                        if n_queries % 200 == 0:
                            avg = 1e3 * t_infer / max(n_infer, 1)
                            print(f"[pi05-server] {n_queries} acts, {n_infer} infers, "
                                  f"avg {avg:.1f} ms/infer", flush=True)
                        send_msg(conn, {"ok": True, "action": action})
                    except Exception as exc:  # noqa: BLE001
                        traceback.print_exc()
                        send_msg(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                else:
                    send_msg(conn, {"ok": False, "error": f"unknown type {mtype!r}"})
        except (ConnectionError, EOFError) as exc:
            print(f"[pi05-server] client gone: {exc}", flush=True)
        finally:
            conn.close()
    srv.close()
    print(f"[pi05-server] shutdown after {n_queries} acts / {n_infer} infers", flush=True)


if __name__ == "__main__":
    main()
