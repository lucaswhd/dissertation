from typing import Any

import numpy as np

from openpi.models.model import Observation


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value.shape)


def validate_model_batch(observation: Observation, actions: Any) -> dict[str, object]:
    """Fail fast unless a transformed batch exactly matches the HRC pi0.5 contract."""
    state_shape = _shape(observation.state)
    if len(state_shape) != 2 or state_shape[1] != 32:
        raise ValueError(f"expected state shape [B,32], got {state_shape}")
    batch_size = state_shape[0]

    expected_image_shape = (batch_size, 224, 224, 3)
    expected_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    if tuple(observation.images) != expected_keys:
        raise ValueError(f"unexpected image keys: {tuple(observation.images)}")
    for key in expected_keys:
        if _shape(observation.images[key]) != expected_image_shape:
            raise ValueError(f"expected image shape {expected_image_shape} for {key}, got {_shape(observation.images[key])}")
        if not np.isfinite(np.asarray(observation.images[key])).all():
            raise ValueError(f"non-finite image values in {key}")

    if not np.asarray(observation.image_masks["base_0_rgb"]).all():
        raise ValueError("head camera mask must be true")
    for key in ("left_wrist_0_rgb", "right_wrist_0_rgb"):
        if np.asarray(observation.image_masks[key]).any():
            raise ValueError("padded wrist camera masks must be false")

    action_shape = _shape(actions)
    if action_shape != (batch_size, 32, 32):
        raise ValueError(f"expected action shape [B,32,32], got {action_shape}")
    if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
        raise ValueError("tokenized prompt and mask are required")
    prompt_shape = _shape(observation.tokenized_prompt)
    if prompt_shape != (batch_size, 200):
        raise ValueError(f"expected prompt shape [B,200], got {prompt_shape}")
    if _shape(observation.tokenized_prompt_mask) != prompt_shape:
        raise ValueError("prompt token and mask shapes differ")
    if not np.isfinite(np.asarray(observation.state)).all() or not np.isfinite(np.asarray(actions)).all():
        raise ValueError("state/actions contain non-finite values")

    return {
        "batch_size": batch_size,
        "image_shape": list(expected_image_shape),
        "state_shape": list(state_shape),
        "action_shape": list(action_shape),
        "prompt_shape": list(prompt_shape),
    }
