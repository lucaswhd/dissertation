import socket
import threading

import numpy as np
import pytest

from scripts.evaluation.pi05_policy_client import Pi05PolicyClient
from scripts.evaluation.pi05_policy_client import recv_message
from scripts.evaluation.pi05_policy_client import send_message


def _serve_once(listener, replies, requests):
    connection, _ = listener.accept()
    try:
        for reply in replies:
            requests.append(recv_message(connection))
            send_message(connection, reply)
    finally:
        connection.close()
        listener.close()


def _fake_server(replies):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    requests = []
    thread = threading.Thread(target=_serve_once, args=(listener, replies, requests), daemon=True)
    thread.start()
    return listener.getsockname()[1], requests, thread


def test_client_validates_handshake_resets_and_sanitizes_action():
    raw_action = np.zeros(23, dtype=np.float32)
    raw_action[:3] = [-2.0, 0.5, 2.0]
    raw_action[14] = -0.2
    raw_action[22] = 0.0
    replies = [
        {"ok": True, "msg": "pong", "action_horizon": 32, "action_dim": 23, "execute_horizon": 8},
        {"ok": True},
        {"ok": True, "action": raw_action},
    ]
    port, requests, thread = _fake_server(replies)

    with Pi05PolicyClient(port=port, timeout_s=1.0) as client:
        client.connect()
        client.reset(seed=123)
        action = client.act(
            state=np.zeros(23, dtype=np.float32),
            head=np.zeros((224, 224, 3), dtype=np.uint8),
            task="Pick up the can.",
            phase="ExecuteBehaviorTask",
        )

    thread.join(timeout=1.0)
    assert [request["type"] for request in requests] == ["ping", "reset", "act"]
    assert requests[1]["seed"] == 123
    assert requests[2]["prompt"] == "Task: Pick up the can. Phase: ExecuteBehaviorTask."
    np.testing.assert_allclose(action[:3], [-1.0, 0.5, 1.0])
    np.testing.assert_array_equal(action[[14, 22]], [-1.0, 1.0])


def test_client_clears_server_plan_when_task_or_phase_changes():
    action = np.zeros(23, dtype=np.float32)
    replies = [
        {"ok": True, "msg": "pong", "action_horizon": 32, "action_dim": 23, "execute_horizon": 8},
        {"ok": True},
        {"ok": True, "action": action},
        {"ok": True},
        {"ok": True, "action": action},
        {"ok": True},
        {"ok": True, "action": action},
    ]
    port, requests, thread = _fake_server(replies)

    with Pi05PolicyClient(port=port, timeout_s=1.0) as client:
        client.connect()
        client.reset(seed=123)
        observation = {
            "state": np.zeros(23, dtype=np.float32),
            "head": np.zeros((16, 16, 3), dtype=np.uint8),
        }
        client.act(**observation, task="Pick up the can.", phase="GoToTask")
        client.act(**observation, task="Pick up the can.", phase="ExecuteBehaviorTask")
        client.act(**observation, task="Turn on the radio.", phase="ExecuteBehaviorTask")

    thread.join(timeout=1.0)
    assert [request["type"] for request in requests] == [
        "ping",
        "reset",
        "act",
        "reset",
        "act",
        "reset",
        "act",
    ]
    assert "seed" not in requests[3]
    assert "seed" not in requests[5]


def test_client_rejects_incompatible_chunk_protocol():
    port, _, thread = _fake_server(
        [{"ok": True, "msg": "pong", "action_horizon": 32, "action_dim": 23, "execute_horizon": 7}]
    )

    client = Pi05PolicyClient(port=port, timeout_s=1.0)
    with pytest.raises(RuntimeError, match="execute_horizon=8"):
        client.connect()
    client.close()
    thread.join(timeout=1.0)


@pytest.mark.parametrize(
    ("state", "head", "message"),
    [
        (np.zeros(22), np.zeros((8, 8, 3), dtype=np.uint8), "state shape"),
        (np.zeros(23), np.zeros((8, 8), dtype=np.uint8), "head image"),
        (np.full(23, np.nan), np.zeros((8, 8, 3), dtype=np.uint8), "finite"),
    ],
)
def test_client_validates_observation_before_network(state, head, message):
    client = Pi05PolicyClient()
    with pytest.raises(ValueError, match=message):
        client.act(state=state, head=head, task="task", phase="GoToTask")
