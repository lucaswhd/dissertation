#!/usr/bin/env python
"""Freeze the closed-loop sub-instruction schedule from the measured demo timings.

Group B rewrote every Execute frame's prompt into one of 7 sub-goals, so replaying
it with the old single long instruction would feed the policy a prompt that never
appears on an Execute frame in its training set -- an out-of-distribution input whose
result says nothing about the policy. The rollout therefore has to walk the same
sub-goal sequence.

Who advances the sequence is a design choice with consequences:

  * timed schedule (this file) -- reproducible, uses no privileged state, and keeps
    the high-level plan fixed so the measurement reads out low-level execution.
  * geometric FSM on hand-object distance -- "smarter", but reads privileged object
    poses the policy never sees and makes the switching logic a confounder.

So the durations come from the 45 CLEAN demos: a segment ends at its median duration,
or earlier if the policy's OWN gripper command produces the flip that segment ends
with. ``min_frames`` is the 10th percentile of the measured distribution, so a policy
that spams the gripper (the radio-round failure mode: precision 0.05) still cannot
race through the schedule faster than the fastest real demonstration.

Usage (openpi_hrc/.venv):
  .venv/bin/python -m scripts.hrc.build_rollout_subinstruction_schedule \
      --subinstructions /root/autodl-tmp/lerobot_datasets/hrc_subinstructions_rerun_v1.json \
      --output /root/autodl-tmp/hrc_pipeline_traintest/configs/rollout_subinstruction_schedule.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def build_schedule(plan: dict, *, timeout_percentile: float, timeout_scale: float, min_percentile: float) -> dict:
    sources = list(plan["sources"].values())
    if not sources:
        raise ValueError("sub-instruction plan contains no sources")

    rounds: dict[str, list[dict]] = {}
    for round_id in ("0", "1"):
        segment_count = len(sources[0][round_id]["segments"])
        if any(len(s[round_id]["segments"]) != segment_count for s in sources):
            raise ValueError(f"round {round_id}: sources disagree on segment count")

        segments = []
        for index in range(segment_count):
            durations, labels, ends = [], set(), set()
            for source in sources:
                segment = source[round_id]["segments"][index]
                durations.append(segment["stop"] - segment["start"])
                labels.add(segment["label"])
                ends.add(segment["ends_with"])
            if len(labels) != 1 or len(ends) != 1:
                raise ValueError(f"round {round_id} segment {index}: inconsistent label/event")
            samples = np.asarray(durations)
            segments.append(
                {
                    "index": index,
                    "label": labels.pop(),
                    "ends_with": ends.pop(),
                    "median_frames": int(np.median(samples)),
                    "advance_frames": int(round(float(np.percentile(samples, timeout_percentile)) * timeout_scale)),
                    "min_frames": int(np.percentile(samples, min_percentile)),
                    "observed": {
                        "n": len(samples),
                        "p10": int(np.percentile(samples, 10)),
                        "p90": int(np.percentile(samples, 90)),
                        "max": int(samples.max()),
                    },
                }
            )
        rounds[round_id] = segments
    return {
        "source_plan": plan.get("manifest"),
        "num_sources": len(sources),
        "timeout_percentile": timeout_percentile,
        "timeout_scale": timeout_scale,
        "min_percentile": min_percentile,
        "advance_rule": (
            "advance when frames_in_segment >= advance_frames, or earlier when the "
            "policy's own gripper command performs `ends_with` and "
            "frames_in_segment >= min_frames"
        ),
        "rounds": rounds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subinstructions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout-percentile", type=float, default=90.0,
                        help="percentile of measured durations used as the forced-advance timeout; "
                             "advancing early truncates a reaching motion and corrupts the readout, "
                             "while advancing late only costs budget, so this is deliberately generous")
    parser.add_argument("--timeout-scale", type=float, default=1.0)
    parser.add_argument("--min-percentile", type=float, default=10.0,
                        help="percentile of measured durations used as the early-advance floor")
    args = parser.parse_args()

    plan = json.loads(args.subinstructions.read_text(encoding="utf-8"))
    schedule = build_schedule(
        plan,
        timeout_percentile=args.timeout_percentile,
        timeout_scale=args.timeout_scale,
        min_percentile=args.min_percentile,
    )
    args.output.write_text(json.dumps(schedule, indent=2), encoding="utf-8")

    for round_id, segments in schedule["rounds"].items():
        total = sum(s["advance_frames"] for s in segments)
        print(f"round {round_id}: {len(segments)} segments, {total} frames if never triggered early")
        for segment in segments:
            print(f"  {segment['index']:>2} advance@{segment['advance_frames']:>5} "
                  f"min{segment['min_frames']:>5}  on={str(segment['ends_with']):<7} "
                  f"{segment['label']}")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
