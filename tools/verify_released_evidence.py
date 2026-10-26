#!/usr/bin/env python3
"""Check internal consistency of the released, non-restricted evidence."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def read_csv(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    mapping = read_csv("evidence/provenance/source_mapping_sanitized.csv")
    require(len(mapping) == 45, "expected 45 retained HRC sources")
    counts = {split: sum(row["split"] == split for row in mapping)
              for split in ("train", "val", "test")}
    require(counts == {"train": 37, "val": 4, "test": 4}, "source split mismatch")
    require(len({row["hrc_source_id"] for row in mapping}) == 45, "duplicate source ID")
    for stage in ("round0", "round1"):
        for suffix in ("raw_sha256", "action_sha256", "state_sha256"):
            require(len({row[f"{stage}_{suffix}"] for row in mapping}) == 45,
                    f"duplicate {stage} {suffix}")
        require(all(row[f"{stage}_replay_exact_actions"] == "True" for row in mapping),
                f"unverified replay action match in {stage}")
    require(len({row[f"round{stage}_raw_sha256"] for row in mapping
                 for stage in (0, 1)}) == 90, "upstream raw file reuse")
    require(all(row["qc_status"] == "CLEAN" and row["status"] == "Verified"
                for row in mapping), "unverified retained source")
    provenance = read_json("evidence/provenance/verification_summary.json")
    require(provenance["split_counts"] == counts, "provenance summary mismatch")
    require(provenance["exact_replay_action_matches"] == 90, "replay summary mismatch")

    split_rows = read_csv("tables/split_summary.csv")
    require({row["split"]: int(row["sources"]) for row in split_rows} == counts,
            "released table source split mismatch")
    require(sum(int(row["frames"]) for row in split_rows) == 409670,
            "released frame total mismatch")

    manifest_path = ROOT / "evidence/frozen_audit/anchor_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    require(manifest_hash == "d5553026605163a74820a865db80e90e81249d3fcceba9cb19655a43fa2bd6a8",
            "frozen anchor manifest hash changed")
    require(manifest["num_anchors"] == len(manifest["anchors"]) == 652,
            "frozen anchor count mismatch")
    require(len({row["anchor_id"] for row in manifest["anchors"]}) == 652,
            "duplicate audit anchor")
    require(manifest["flow_seeds"] == [0, 1, 2, 3, 4],
            "inference seed protocol mismatch")
    summaries = {}
    for model in ("A20", "B20", "C20"):
        summary = read_json(f"evidence/frozen_audit/{model}/summary.json")
        summaries[model] = summary
        require(summary["model"] == model and summary["num_anchors"] == 652,
                f"{model} summary mismatch")
        require(summary["manifest_sha256"] == manifest_hash and
                summary["flow_seeds"] == manifest["flow_seeds"],
                f"{model} protocol mismatch")

    for row in read_csv("tables/v8_gripper_event_summary.csv"):
        context = summaries[row["model"]]["by_context"][row["context"]]
        metric = context["metrics"]["grippers." + row["metric"]]
        require(abs(float(row["mean"]) - metric["mean"]) < 1e-12 and
                abs(float(row["sample_sd"]) - metric["std"]) < 1e-12,
                "released gripper metric differs from frozen summary")
        require(int(row["matched_context_anchors"]) == context["num_anchors"] and
                int(row["inference_flow_seeds"]) == 5 and
                int(row["training_runs"]) == 1 and int(row["training_seed"]) == 42,
                "released metric population or seed labels differ from frozen protocol")

    prediction_manifest = read_json("evidence/predictions/manifest.json")
    files = prediction_manifest["prediction_files"]
    require(len(files) == 15, "expected five archived predictions per model")
    require({(row["model"], row["inference_flow_seed"]) for row in files} ==
            {(model, seed) for model in ("A20", "B20", "C20") for seed in range(5)},
            "prediction file coverage mismatch")
    require(all(row["size_bytes"] > 0 and len(row["sha256"]) == 64 and
                {a["name"]: a["shape"] for a in row["arrays"]} ==
                {"predictions.npy": [652, 32, 23], "anchor_ids.npy": [652]}
                for row in files), "prediction archive schema mismatch")

    checkpoints = read_json("evidence/checkpoints/verification_summary.json")["models"]
    require(checkpoints["A20"]["inference_parameter_leaves_restored"] == 71 and
            checkpoints["A20"]["architecture_bind"] == "Verified", "A20 restore mismatch")
    require(all(not checkpoints[name]["current_weight_payload_located"]
                for name in ("B20", "C20")), "B20/C20 bounded search mismatch")
    require(all(row["fresh_matched_inference"] == "not_verified"
                for row in checkpoints.values()), "fresh inference claim mismatch")
    print("PASS: 45 sources, 90 unique upstream files, 409670 frames, 652 anchors, "
          "18 frozen gripper metrics, 15 prediction manifests, and bounded checkpoint conclusions")


if __name__ == "__main__":
    main()
