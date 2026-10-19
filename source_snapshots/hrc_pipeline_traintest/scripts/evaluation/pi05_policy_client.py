"""Validated TCP client for the HRC π0.5 rollout server."""

from __future__ import annotations

import os
import pickle
import socket
import struct
from typing import Any

import numpy as np

from scripts.evaluation.hrc_rollout_core import sanitize_r1pro_action


ACTION_HORIZON = 32
ACTION_DIM = 23
EXECUTE_HORIZON = 8
DEFAULT_TIMEOUT_S = 300.0


def _pack(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__nd__": True,
            "data": array.tobytes(),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _pack(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pack(item) for item in value]
    return value


def _unpack(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("__nd__"):
            return np.frombuffer(value["data"], dtype=value["dtype"]).reshape(value["shape"]).copy()
        return {key: _unpack(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack(item) for item in value]
    return value


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("π0.5 server closed the connection")
        chunks.extend(chunk)
    return bytes(chunks)


def send_message(connection: socket.socket, message: dict[str, Any]) -> None:
    payload = pickle.dumps(_pack(message), protocol=pickle.HIGHEST_PROTOCOL)
    connection.sendall(struct.pack(">I", len(payload)) + payload)


def recv_message(connection: socket.socket) -> dict[str, Any]:
    (length,) = struct.unpack(">I", _recv_exact(connection, 4))
    reply = _unpack(pickle.loads(_recv_exact(connection, length)))
    if not isinstance(reply, dict):
        raise RuntimeError(f"π0.5 server returned {type(reply).__name__}, expected dict")
    return reply


class Pi05PolicyClient:
    """One persistent, shape-checked connection to the π0.5 policy service."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 18765,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self._connection: socket.socket | None = None
        self._context: tuple[str, str] | None = None

    @property
    def connected(self) -> bool:
        return self._connection is not None

    def __enter__(self) -> "Pi05PolicyClient":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _request(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._connection is None:
            raise ConnectionError("π0.5 policy client is not connected")
        send_message(self._connection, message)
        reply = recv_message(self._connection)
        if reply.get("ok") is not True:
            raise RuntimeError(f"π0.5 server error: {reply.get('error', reply)}")
        return reply

    def connect(self) -> dict[str, Any]:
        if self._connection is not None:
            return self.ping()
        self._connection = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        self._connection.settimeout(self.timeout_s)
        try:
            return self.ping()
        except Exception:
            self.close()
            raise

    def ping(self) -> dict[str, Any]:
        reply = self._request({"type": "ping"})
        expected = {
            "action_horizon": ACTION_HORIZON,
            "action_dim": ACTION_DIM,
            # Replan rate is an experiment knob (HRC_EXECUTE_HORIZON); the handshake
            # still forces client and server to agree on whatever value is in use.
            "execute_horizon": int(os.environ.get("HRC_EXECUTE_HORIZON", EXECUTE_HORIZON)),
        }
        mismatches = [
            f"{key}={value}" for key, value in expected.items() if int(reply.get(key, -1)) != value
        ]
        if mismatches:
            raise RuntimeError("incompatible π0.5 server; expected " + ", ".join(mismatches))
        return reply

    def reset(self, *, seed: int | None = None) -> None:
        message: dict[str, Any] = {"type": "reset"}
        if seed is not None:
            message["seed"] = int(seed)
        self._request(message)
        self._context = None

    def reconnect(self, *, reset: bool = True, seed: int | None = None) -> None:
        self.close()
        self.connect()
        if reset:
            self.reset(seed=seed)

    def close(self) -> None:
        connection, self._connection = self._connection, None
        self._context = None
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()

    def act(
        self,
        *,
        state: np.ndarray,
        head: np.ndarray,
        task: str,
        phase: str,
    ) -> np.ndarray:
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.shape != (ACTION_DIM,):
            raise ValueError(f"expected state shape (23,), got {state_array.shape}")
        if not np.isfinite(state_array).all():
            raise ValueError("state must be finite")

        head_array = np.asarray(head)
        if head_array.ndim != 3 or head_array.shape[-1] != 3:
            raise ValueError(f"expected HWC RGB head image, got {head_array.shape}")
        if not np.isfinite(head_array).all():
            raise ValueError("head image must be finite")
        if np.issubdtype(head_array.dtype, np.floating):
            scale = 255.0 if head_array.size and float(head_array.max()) <= 1.0 else 1.0
            head_array = np.clip(head_array * scale, 0.0, 255.0).astype(np.uint8)
        else:
            head_array = np.clip(head_array, 0, 255).astype(np.uint8)

        task_text = str(task).strip()
        phase_text = str(phase).strip()
        context = (task_text, phase_text)
        if self._context is not None and context != self._context:
            # The server serves eight actions from each 32-step chunk.  Do not
            # execute queued actions generated for the preceding task/phase.
            self._request({"type": "reset"})
            self._context = None

        prompt = f"Task: {task_text} Phase: {phase_text}."
        reply = self._request(
            {
                "type": "act",
                "state": state_array,
                "head": head_array,
                "prompt": prompt,
            }
        )
        if "action" not in reply:
            raise RuntimeError("π0.5 server reply is missing action")
        action = sanitize_r1pro_action(reply["action"])
        self._context = context
        return action
