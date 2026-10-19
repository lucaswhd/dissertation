import types

import flax.nnx as nnx
import jax.numpy as jnp

from scripts.hrc.run_training_preflight import _audit_trainable_tree


class _TinyTrainableTree(nnx.Module):
    def __init__(self):
        self.lora_adapter = nnx.Param(jnp.zeros((2, 3)))
        self.action_in_proj = nnx.Param(jnp.zeros((3, 4)))
        self.action_out_proj = nnx.Param(jnp.zeros((4, 3)))
        self.time_mlp_in = nnx.Param(jnp.zeros((2, 2)))
        self.time_mlp_out = nnx.Param(jnp.zeros((2, 2)))


def test_audit_trainable_tree_handles_nnx_flat_state_mapping():
    params = nnx.state(_TinyTrainableTree())
    state = types.SimpleNamespace(params=params)
    config = types.SimpleNamespace(trainable_filter=nnx.Param)
    report = _audit_trainable_tree(config, state)
    assert report == {"trainable_leaves": 5, "trainable_parameters": 38}
