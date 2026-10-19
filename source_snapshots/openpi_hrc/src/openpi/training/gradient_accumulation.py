from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp


def accumulate_gradients(
    grad_fn: Callable[[jax.Array, Any], tuple[jax.Array, Any]],
    microbatches: Any,
    initial_grads: Any,
) -> tuple[jax.Array, Any]:
    """Average loss and gradients across a leading microbatch axis with `lax.scan`."""
    leaves = jax.tree.leaves(microbatches)
    if not leaves:
        raise ValueError("microbatches must contain at least one array")
    microbatch_count = int(leaves[0].shape[0])
    if microbatch_count == 0:
        raise ValueError("microbatches must contain at least one item")
    if any(leaf.ndim == 0 or leaf.shape[0] != microbatch_count for leaf in leaves):
        raise ValueError("all microbatch leaves must have the same leading dimension")

    def scan_step(carry, inputs):
        loss_sum, grad_sum = carry
        index, microbatch = inputs
        loss, grads = grad_fn(index, microbatch)
        return (
            loss_sum + loss,
            jax.tree.map(lambda total, grad: total + grad, grad_sum, grads),
        ), None

    (loss_sum, grad_sum), _ = jax.lax.scan(
        scan_step,
        (jnp.array(0.0, dtype=jnp.float32), initial_grads),
        (jnp.arange(microbatch_count), microbatches),
    )
    scale = jnp.asarray(microbatch_count, dtype=loss_sum.dtype)
    return loss_sum / scale, jax.tree.map(lambda grad: grad / scale, grad_sum)
