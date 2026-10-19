import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.training.gradient_accumulation import accumulate_gradients


def test_accumulated_gradient_matches_equivalent_large_batch_update():
    params = {"w": jnp.array([0.5, -0.25]), "b": jnp.array(0.1)}
    microbatches = {
        "x": jnp.array(
            [
                [[1.0, 2.0], [3.0, -1.0]],
                [[-2.0, 0.5], [1.5, 2.5]],
            ]
        ),
        "y": jnp.array([[0.2, -0.5], [1.0, 0.7]]),
    }

    def loss_fn(current_params, batch):
        prediction = batch["x"] @ current_params["w"] + current_params["b"]
        return jnp.mean((prediction - batch["y"]) ** 2)

    def micro_grad_fn(index, batch):
        del index
        return jax.value_and_grad(loss_fn)(params, batch)

    accumulated_loss, accumulated_grads = accumulate_gradients(
        micro_grad_fn,
        microbatches,
        jax.tree.map(jnp.zeros_like, params),
    )
    large_batch = jax.tree.map(lambda value: value.reshape((-1, *value.shape[2:])), microbatches)
    large_loss, large_grads = jax.value_and_grad(loss_fn)(params, large_batch)

    np.testing.assert_allclose(accumulated_loss, large_loss, rtol=1e-6, atol=1e-6)
    for accumulated, large in zip(jax.tree.leaves(accumulated_grads), jax.tree.leaves(large_grads), strict=True):
        np.testing.assert_allclose(accumulated, large, rtol=1e-6, atol=1e-6)

    optimizer = optax.adamw(1e-3)
    opt_state = optimizer.init(params)
    accumulated_updates, _ = optimizer.update(accumulated_grads, opt_state, params)
    large_updates, _ = optimizer.update(large_grads, opt_state, params)
    accumulated_params = optax.apply_updates(params, accumulated_updates)
    large_params = optax.apply_updates(params, large_updates)
    for accumulated, large in zip(jax.tree.leaves(accumulated_params), jax.tree.leaves(large_params), strict=True):
        np.testing.assert_allclose(accumulated, large, rtol=1e-6, atol=1e-6)


def test_accumulation_rejects_empty_or_mismatched_microbatch_axes():
    def grad_fn(index, batch):
        del index, batch
        return jnp.array(0.0), {"w": jnp.array(0.0)}

    with np.testing.assert_raises_regex(ValueError, "at least one"):
        accumulate_gradients(grad_fn, {"x": jnp.empty((0, 2))}, {"w": jnp.array(0.0)})
    with np.testing.assert_raises_regex(ValueError, "leading dimension"):
        accumulate_gradients(
            grad_fn,
            {"x": jnp.ones((2, 1)), "y": jnp.ones((3, 1))},
            {"w": jnp.array(0.0)},
        )
