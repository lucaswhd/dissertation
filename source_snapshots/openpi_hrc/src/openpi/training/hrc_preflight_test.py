import jax.numpy as jnp
import numpy as np

from openpi.models.model import Observation
from openpi.training.hrc_preflight import validate_model_batch


def _valid_batch(batch_size=2):
    observation = Observation(
        images={
            "base_0_rgb": jnp.zeros((batch_size, 224, 224, 3), dtype=jnp.float32),
            "left_wrist_0_rgb": jnp.zeros((batch_size, 224, 224, 3), dtype=jnp.float32),
            "right_wrist_0_rgb": jnp.zeros((batch_size, 224, 224, 3), dtype=jnp.float32),
        },
        image_masks={
            "base_0_rgb": jnp.ones((batch_size,), dtype=bool),
            "left_wrist_0_rgb": jnp.zeros((batch_size,), dtype=bool),
            "right_wrist_0_rgb": jnp.zeros((batch_size,), dtype=bool),
        },
        state=jnp.zeros((batch_size, 32), dtype=jnp.float32),
        tokenized_prompt=jnp.zeros((batch_size, 200), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((batch_size, 200), dtype=bool),
    )
    return observation, jnp.zeros((batch_size, 32, 32), dtype=jnp.float32)


def test_validate_model_batch_accepts_required_pi05_shapes():
    report = validate_model_batch(*_valid_batch())
    assert report == {
        "batch_size": 2,
        "image_shape": [2, 224, 224, 3],
        "state_shape": [2, 32],
        "action_shape": [2, 32, 32],
        "prompt_shape": [2, 200],
    }


def test_validate_model_batch_rejects_wrong_shape_mask_and_nonfinite():
    observation, actions = _valid_batch()
    bad_state = observation.replace(state=jnp.zeros((2, 23)))
    with np.testing.assert_raises_regex(ValueError, "state shape"):
        validate_model_batch(bad_state, actions)

    bad_mask = observation.replace(
        image_masks={**observation.image_masks, "left_wrist_0_rgb": jnp.ones((2,), dtype=bool)}
    )
    with np.testing.assert_raises_regex(ValueError, "wrist camera"):
        validate_model_batch(bad_mask, actions)

    bad_actions = actions.at[0, 0, 0].set(jnp.nan)
    with np.testing.assert_raises_regex(ValueError, "non-finite"):
        validate_model_batch(observation, bad_actions)
