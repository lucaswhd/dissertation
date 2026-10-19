# ruff: noqa: SLF001

from pathlib import Path

from openpi.training import config as _config
from scripts.b1k import train_b1k


def test_interval_gate_uses_completed_optimizer_steps():
    assert not train_b1k._is_interval_step(999, 1_000)
    assert train_b1k._is_interval_step(1_000, 1_000)
    assert not train_b1k._is_interval_step(1_000, None)


def test_prepare_validation_config_switches_hrc_root_and_disables_weighting(monkeypatch):
    captured = {}

    def fake_create(self, assets_dirs, model_config):
        del assets_dirs, model_config
        captured["factory"] = self
        return _config.HrcDataConfig(
            repo_id=self.repo_id,
            dataset_root=self.dataset_root,
            split=self.split,
            state_key=self.state_key,
            state_dim=self.state_dim,
            task_weights=self.task_weights,
            include_phase=self.include_phase,
        )

    monkeypatch.setattr(_config.LeRobotHrcDataConfig, "create", fake_create)
    config = _config.get_config("pi05_hrc_e4_main")
    training_stats = {"state": object(), "actions": object()}

    val_config, repo_id, use_stats, actual = train_b1k._prepare_validation_config(config, training_stats)

    factory = captured["factory"]
    assert repo_id == "hrc/hrc_pi05_v0_val"
    assert factory.dataset_root == str(Path(config.data.dataset_root).parent / "val")
    assert factory.split == "val"
    assert factory.task_weights is None
    assert factory.max_valid_anchors is None
    assert val_config.batch_size == 2
    assert use_stats
    assert actual.norm_stats is training_stats
