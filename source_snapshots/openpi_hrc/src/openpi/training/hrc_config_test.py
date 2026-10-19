
import flax.nnx as nnx
import numpy as np

from openpi import transforms
from openpi.models import pi0_config
from openpi.training import config as _config


def test_hrc_factory_repack_and_delta_mapping(monkeypatch, tmp_path):
    monkeypatch.setattr(
        _config.ModelTransformFactory,
        "__call__",
        lambda self, model_config: transforms.Group(),
    )
    factory = _config.LeRobotHrcDataConfig(
        repo_id="hrc/hrc_pi05_v0_train",
        dataset_root="/data/hrc_pi05_v0/train",
        manifest_path="/data/hrc_pi05_v0/manifest.json",
        split="train",
        state_key="observation.state",
        task_weights=(0.4, 0.2, 0.4),
        include_phase=True,
    )
    model = pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=32)

    data_config = factory.create(tmp_path, model)

    assert isinstance(data_config, _config.HrcDataConfig)
    assert data_config.dataset_root == "/data/hrc_pi05_v0/train"
    assert data_config.manifest_path == "/data/hrc_pi05_v0/manifest.json"
    assert data_config.state_key == "observation.state"
    assert data_config.action_sequence_keys == ("action",)
    assert data_config.prompt_from_task

    raw = {
        "observation.rgb.zed_link_camera_0": np.zeros((3, 8, 10), dtype=np.float32),
        "observation.state": np.arange(23, dtype=np.float32),
        "action": np.tile(np.arange(23, dtype=np.float32), (32, 1)),
        "prompt": "Task: Move the cup. Phase: PICKUP.",
    }
    repacked = transforms.compose(data_config.repack_transforms.inputs)(raw)
    output = transforms.compose(data_config.data_transforms.inputs)(repacked)
    assert output["state"].shape == (23,)
    assert output["actions"].shape == (32, 23)
    assert output["image_mask"]["base_0_rgb"]
    assert not output["image_mask"]["left_wrist_0_rgb"]
    np.testing.assert_array_equal(output["actions"][:, :3], raw["action"][:, :3])
    np.testing.assert_array_equal(output["actions"][:, 3:14], 0.0)
    np.testing.assert_array_equal(output["actions"][:, 15:22], 0.0)


def test_hrc_experiment_matrix_and_lora_freezing():
    expected = {
        "pi05_hrc_e0_overfit": (23, None, True, 1_000, 128),
        "pi05_hrc_e1_privileged": (28, (0.4, 0.2, 0.4), True, 1_000, None),
        "pi05_hrc_e2_vanilla": (23, None, False, 5_000, None),
        "pi05_hrc_e3_balanced": (23, (0.4, 0.2, 0.4), False, 5_000, None),
        "pi05_hrc_e4_main": (23, (0.4, 0.2, 0.4), True, 5_000, None),
        "pi05_hrc_e5_continuous_e0": (23, None, True, 1_000, 128),
        "pi05_hrc_e5_continuous_main": (23, None, True, 5_000, None),
        "pi05_hrc_e5_continuous_20k": (23, None, True, 20_000, None),
    }

    for name, (state_dim, weights, include_phase, steps, max_anchors) in expected.items():
        train_config = _config.get_config(name)
        assert train_config.model.pi05
        assert train_config.model.action_dim == 32
        assert train_config.model.action_horizon == 32
        assert train_config.model.max_token_len == 200
        assert train_config.model.discrete_state_input
        assert train_config.model.paligemma_variant == "gemma_2b_lora"
        assert train_config.model.action_expert_variant == "gemma_300m_lora"
        assert train_config.num_train_steps == steps
        assert train_config.batch_size == 2
        assert train_config.grad_accum_steps == 4
        assert train_config.ema_decay is None
        assert train_config.data.state_dim == state_dim
        assert train_config.data.task_weights == weights
        assert train_config.data.include_phase == include_phase
        assert train_config.data.max_valid_anchors == max_anchors


def test_e5_continuous_configs_point_at_continuous_dataset_and_stratify_e0():
    base = "/root/autodl-tmp/lerobot_datasets/hrc_pi05_continuous_clean_v0"

    e0 = _config.get_config("pi05_hrc_e5_continuous_e0")
    assert e0.data.dataset_root == f"{base}/train"
    assert e0.data.manifest_path == f"{base}/manifest.json"
    assert e0.data.repo_id == "hrc/hrc_pi05_continuous_clean_v0_train"
    assert e0.val_repo_id == "hrc/hrc_pi05_continuous_clean_v0_val"
    assert e0.data.diag_subset_seed == 42  # E0 uses the stratified diagnostic subset
    assert e0.data.max_valid_anchors == 128
    assert e0.data.task_weights is None

    main = _config.get_config("pi05_hrc_e5_continuous_main")
    assert main.data.dataset_root == f"{base}/train"
    assert main.data.diag_subset_seed is None  # main uses the full natural distribution
    assert main.data.max_valid_anchors is None
    assert main.num_train_steps == 5_000

    # The base three-segment configs must be untouched by the new parameters.
    e4 = _config.get_config("pi05_hrc_e4_main")
    assert e4.data.dataset_root == "/root/autodl-tmp/lerobot_datasets/hrc_pi05_v0/train"
    assert e4.data.diag_subset_seed is None

    predicate = nnx.filterlib.to_predicate(_config.get_config("pi05_hrc_e4_main").freeze_filter)
    param = nnx.Param(np.array(0.0))
    assert predicate(("PaliGemma", "img", "kernel"), param)
    assert predicate(("PaliGemma", "llm", "kernel"), param)
    assert not predicate(("PaliGemma", "llm", "lora", "a"), param)
    assert not predicate(("action_in_proj", "kernel"), param)
    assert not predicate(("action_out_proj", "kernel"), param)
    assert not predicate(("time_mlp_in", "kernel"), param)
