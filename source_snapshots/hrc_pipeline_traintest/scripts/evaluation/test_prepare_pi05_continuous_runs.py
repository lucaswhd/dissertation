import argparse
import json
from pathlib import Path

from scripts.evaluation.prepare_pi05_continuous_runs import build_plan


def _fixture_args(tmp_path: Path, *, smoke: bool = False) -> argparse.Namespace:
    continuous_manifest = tmp_path / "continuous_manifest.json"
    continuous_manifest.write_text(
        json.dumps({"source_splits": {"train": [3, 4], "val": [15, 18], "test": []}}),
        encoding="utf-8",
    )
    batch_manifest = tmp_path / "batch_manifest.json"
    batch_manifest.write_text(
        json.dumps(
            {
                "episodes": [
                    {"idx": 15, "trash_hdf5": "/data/ep15/trash.hdf5", "radio_hdf5": "/data/ep15/radio.hdf5"},
                    {"idx": 18, "trash_hdf5": "/data/ep18/trash.hdf5", "radio_hdf5": "/data/ep18/radio.hdf5"},
                ]
            }
        ),
        encoding="utf-8",
    )
    return argparse.Namespace(
        batch_manifest=batch_manifest,
        continuous_manifest=continuous_manifest,
        openpi_root=Path("/openpi"),
        config=Path("/cfg/hrc.yaml"),
        output_root=tmp_path / "closed_loop",
        init_checkpoint=Path("/ckpt/init/0"),
        candidate_checkpoint=Path("/ckpt/5000"),
        execute_steps="3000,2400",
        execute_max_seconds=1800.0,
        smoke_execute_steps="32",
        smoke=smoke,
        policy_host="127.0.0.1",
        policy_port=18765,
        dry_run=True,
    )


def test_full_plan_has_eight_runs_two_models_and_comparison(tmp_path):
    plan = build_plan(_fixture_args(tmp_path))

    assert plan["source_ids"] == [15, 18]
    assert plan["policy_seeds"] == [0, 1]
    assert plan["config_name"] == "pi05_hrc_e5_continuous_main"
    assert plan["comparison"] == {"baseline": "continuous_init", "candidate": "e5_continuous_5k"}

    labels = [group["model"] for group in plan["model_groups"]]
    assert labels == ["continuous_init", "e5_continuous_5k"]
    all_runs = [run for group in plan["model_groups"] for run in group["runs"]]
    assert len(all_runs) == 8  # 2 models x 2 sources x 2 seeds

    ids = {(run["model"], run["source_id"], run["seed"]) for run in all_runs}
    assert len(ids) == 8
    for group in plan["model_groups"]:
        assert "serve_hrc_pi05_rollout.py" in group["server_command"]
        assert "pi05_hrc_e5_continuous_main" in group["server_command"]
    # the expert needs ~2505 (pickup) / ~1956 (radio) steps; a flat 600 capped both
    assert plan["execute_steps_per_round"] == "3000,2400"
    assert plan["execute_max_seconds"] == 1800.0
    for run in all_runs:
        assert "--policy-execute-steps 3000,2400" in run["command"]
        assert "--policy-execute-max-seconds 1800.0" in run["command"]
        assert "--policy-rollout" in run["command"]
        assert run["command"].startswith("HRC_RECORD_THIRD=1 ")
        assert run["expected_third_video"].endswith("third_person_rgb.mp4")


def test_smoke_plan_is_single_run_trained_model_only(tmp_path):
    plan = build_plan(_fixture_args(tmp_path, smoke=True))
    all_runs = [run for group in plan["model_groups"] for run in group["runs"]]
    assert len(all_runs) == 1
    assert all_runs[0]["model"] == "e5_continuous_5k"
    assert plan["source_ids"] == [15]
    assert plan["execute_steps_per_round"] == "32"


def test_rejects_manifest_whose_val_is_not_15_18(tmp_path):
    args = _fixture_args(tmp_path)
    args.continuous_manifest.write_text(
        json.dumps({"source_splits": {"train": [], "val": [15, 99], "test": []}}), encoding="utf-8"
    )
    try:
        build_plan(args)
    except ValueError:
        return
    raise AssertionError("expected ValueError when val sources are not [15, 18]")
