from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.hrc.run_dissertation_final_audit import COMPARISON_FIELDS
from scripts.hrc.run_dissertation_final_audit import MODEL_SPECS
from scripts.hrc.run_dissertation_final_audit import build_comparison_rows
from scripts.hrc.run_dissertation_final_audit import validate_complete_seed_records
from scripts.hrc.run_dissertation_final_audit import validate_model_binding
from scripts.hrc.run_dissertation_final_audit import validate_saved_prediction_bundle


def test_incomplete_seed_records_cannot_be_finalized() -> None:
    records = {
        0: [{"anchor_id": "a"}, {"anchor_id": "b"}],
        1: [{"anchor_id": "a"}],
    }

    with pytest.raises(ValueError, match="seed 1.*missing.*b"):
        validate_complete_seed_records(records, expected_anchor_ids=["a", "b"], expected_seeds=[0, 1])


def test_model_binding_rejects_the_wrong_dataset_version() -> None:
    with pytest.raises(ValueError, match="dataset version"):
        validate_model_binding(
            MODEL_SPECS["A20"],
            config_name="pi05_hrc_e6_rerun_20k",
            checkpoint=Path("checkpoints/pi05_hrc_e6_rerun_20k/hrc_e6_rerun_seed42/20000"),
            dataset_version="hrc_subinstr_rerun_v1",
        )


def test_saved_predictions_require_exact_manifest_order() -> None:
    predictions = np.zeros((2, 32, 23), dtype=np.float32)

    validate_saved_prediction_bundle(predictions, ["a", "b"], ["a", "b"])
    with pytest.raises(ValueError, match="anchor order"):
        validate_saved_prediction_bundle(predictions, ["b", "a"], ["a", "b"])


def test_comparison_rows_have_fixed_paper_schema() -> None:
    summaries = {
        "A20": {
            "by_context": {
                "pickup": {
                    "num_anchors": 200,
                    "metrics": {
                        "arms.raw_mae": {"mean": 0.2, "std": 0.01, "n": 5},
                        "arms.null_mae": {"mean": 0.4, "std": 0.0, "n": 5},
                        "arms.skill": {"mean": 0.5, "std": 0.02, "n": 5},
                        "arm_motion_ratio": {"mean": 0.7, "std": 0.03, "n": 5},
                        "grippers.f1": {"mean": 0.25, "std": 0.1, "n": 5},
                    },
                    "dispersion": {
                        "arms": {"mean": 0.04, "std": 0.005, "n": 200},
                    },
                }
            }
        }
    }

    rows = build_comparison_rows(summaries)

    assert rows
    assert list(rows[0]) == COMPARISON_FIELDS
    assert {row["metric"] for row in rows} == {
        "raw_mae",
        "null_mae",
        "skill",
        "arm_motion_ratio",
        "gripper_f1",
        "output_dispersion",
    }
    assert all(row["baseline"] == "null" for row in rows)
