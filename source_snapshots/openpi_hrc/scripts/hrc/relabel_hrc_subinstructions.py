#!/usr/bin/env python
"""Re-label the Execute phases of every CLEAN source with measured sub-instructions.

The single long instruction ("Pick up the soda cans and put them in the trash can.")
spans ~138 s and 6+ sub-goals, so the same observation maps to different correct
actions and the model converges on the mean. The Step-2 probe showed that splitting
this into sub-goals plus oversampling the gripper-flip frames lifts arms skill from
-0.004 to 0.832 and gripper flip recall from 0.000 to 0.559.

Segment boundaries are the expert's own gripper sign changes -- physical events, not
authored labels. All 45 CLEAN episodes share one pattern, verified frame-by-frame:

    round 0 (pickup): Rclose Lclose Lopen Lclose Lopen Lclose Lopen Ropen
    round 1 (radio):  Rclose Lclose Ropen

so the robot carries the trash can in its right gripper the whole time and drops cans
in with its left. The sub-instructions say exactly that. Describing an idealized
"walk back to the trash can" would attach language to actions that never occur, which
is the error that invalidated task-0026.

Frames after the last flip where the base does not move (median 361, 5.6% of Execute)
are idle settle-out. They keep the terminal instruction but are marked invalid as
anchors: training on them teaches the policy to stand still, which is the exact
failure being fixed.

Outputs a per-source JSON of segment boundaries, labels and anchor masks. It writes no
dataset -- the exporter consumes this to build one.

Usage (openpi_hrc/.venv):
  .venv/bin/python -m scripts.hrc.relabel_hrc_subinstructions \
      --manifest /root/autodl-tmp/lerobot_datasets/hrc_pi05_continuous_rerun_v1/manifest.json \
      --output /root/autodl-tmp/lerobot_datasets/hrc_subinstructions_rerun_v1.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from openpi.training.hrc_continuous_export import load_continuous_rows

ACTION_HORIZON = 32
GRIPPER_DIMS = [14, 22]
LEFT, RIGHT = 0, 1

# Expected flip pattern per round; a source that disagrees is not silently relabelled.
EXPECTED_PATTERN = {
    0: ("Rclose", "Lclose", "Lopen", "Lclose", "Lopen", "Lclose", "Lopen", "Ropen"),
    1: ("Rclose", "Lclose", "Ropen"),
}

# Label for the segment ENDING in each flip: what the robot is working toward during it.
SEGMENT_GOAL = {
    "Rclose": "Walk to the trash can and pick it up.",
    "Lclose": "Pick up the soda can.",
    "Lopen": "Drop the can into the trash can you are holding.",
    "Ropen": "Put the trash can down.",
}
TERMINAL_LABEL = {
    0: "The trash is cleaned up. Stand by.",
    1: "The radio is on. Stand by.",
}
RADIO_OVERRIDE = {
    # In the radio round the left-gripper close is the button press, not a grasp.
    "Lclose": "Turn on the radio receiver.",
}


def flip_events(action: np.ndarray) -> tuple[list[int], list[str]]:
    """Indices and names of gripper sign changes, in order."""
    signs = np.sign(action[:, GRIPPER_DIMS])
    frames, names = [], []
    for index in range(1, len(signs)):
        for side, tag in ((LEFT, "L"), (RIGHT, "R")):
            if signs[index, side] != signs[index - 1, side]:
                frames.append(index)
                names.append(f"{tag}{'close' if signs[index, side] < 0 else 'open'}")
    return frames, names


def idle_tail_start(base_velocity: np.ndarray, last_flip: int, *, move_threshold: float, fps: int = 30) -> int:
    """First frame of the stationary settle-out tail, or len(...) if there is none.

    Distance is integrated from the commanded base velocity rather than a recorded
    world pose: the pose is privileged information the policy never sees, and the
    command stream is what the label has to be consistent with.
    """
    tail = base_velocity[last_flip:]
    if len(tail) < 2:
        return len(base_velocity)
    travelled = float(np.linalg.norm(tail[:, :2], axis=1).sum() / fps)
    return last_flip if travelled < move_threshold else len(base_velocity)


def label_round(action: np.ndarray, round_id: int, *, move_threshold: float) -> dict:
    """Segment one Execute round into sub-instructions plus an idle-tail mask."""
    frames, names = flip_events(action)
    expected = EXPECTED_PATTERN[round_id]
    if tuple(names) != expected:
        raise ValueError(
            f"round {round_id} flip pattern {tuple(names)} != expected {expected}; "
            "refusing to relabel a source whose demo structure differs"
        )

    labels = np.empty(len(action), dtype=object)
    boundaries = []
    cursor = 0
    for frame, name in zip(frames, names, strict=True):
        goal = RADIO_OVERRIDE.get(name) if round_id == 1 else None
        goal = goal or SEGMENT_GOAL[name]
        labels[cursor:frame] = goal
        boundaries.append({"start": cursor, "stop": frame, "ends_with": name, "label": goal})
        cursor = frame

    terminal = TERMINAL_LABEL[round_id]
    labels[cursor:] = terminal
    boundaries.append({"start": cursor, "stop": len(action), "ends_with": None, "label": terminal})

    idle_from = idle_tail_start(action[:, :3], frames[-1], move_threshold=move_threshold)
    anchor_ok = np.ones(len(action), dtype=bool)
    anchor_ok[idle_from:] = False
    # A chunk must stay inside the round.
    anchor_ok[max(0, len(action) - ACTION_HORIZON) :] = False

    return {
        "flip_frames": frames,
        "flip_names": names,
        "segments": boundaries,
        "idle_tail_from": int(idle_from),
        "idle_tail_frames": int(len(action) - idle_from),
        "labels": labels,
        "anchor_ok": anchor_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--move-threshold", type=float, default=0.05,
                        help="metres of base travel below which a tail counts as idle")
    parser.add_argument("--flip-radius", type=int, default=16,
                        help="frames each side of a flip that count as decision frames")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = {
        "manifest": str(args.manifest),
        "action_horizon": ACTION_HORIZON,
        "move_threshold_m": args.move_threshold,
        "flip_radius": args.flip_radius,
        "expected_pattern": {str(k): list(v) for k, v in EXPECTED_PATTERN.items()},
        "sources": {},
    }
    totals = {"frames": 0, "anchors": 0, "event_anchors": 0, "idle": 0}
    label_counts: dict[str, int] = {}

    for source in manifest["sources"]:
        source_id = int(source["source_episode_id"])
        rows = load_continuous_rows(source, action_horizon=ACTION_HORIZON)
        demo0_len = int(source["demo0_length"])

        # Execute rounds are located by FSM state, per round, on the continuous timeline.
        fsm = rows["fsm_state_id"]
        per_round = {}
        for round_id, (lo, hi) in {0: (0, demo0_len), 1: (demo0_len, len(fsm))}.items():
            block = slice(lo, hi)
            # ExecuteBehaviorTask is fsm id 3 in the frozen HRC state list.
            mask = np.flatnonzero(fsm[block] == 3) + lo
            if len(mask) < ACTION_HORIZON:
                raise ValueError(f"source {source_id} round {round_id}: no Execute frames")
            start, stop = int(mask[0]), int(mask[-1]) + 1
            if stop - start != len(mask):
                raise ValueError(f"source {source_id} round {round_id}: Execute frames not contiguous")

            labelled = label_round(
                rows["action"][start:stop], round_id, move_threshold=args.move_threshold
            )
            event = np.zeros(stop - start, dtype=bool)
            for flip in labelled["flip_frames"]:
                event[max(0, flip - args.flip_radius) : flip + args.flip_radius + 1] = True

            for segment in labelled["segments"]:
                label_counts[segment["label"]] = label_counts.get(segment["label"], 0) + (
                    segment["stop"] - segment["start"]
                )
            totals["frames"] += stop - start
            totals["anchors"] += int(labelled["anchor_ok"].sum())
            totals["event_anchors"] += int((event & labelled["anchor_ok"]).sum())
            totals["idle"] += labelled["idle_tail_frames"]

            per_round[str(round_id)] = {
                "execute_start": start,
                "execute_stop": stop,
                "flip_frames": [start + f for f in labelled["flip_frames"]],
                "flip_names": labelled["flip_names"],
                "segments": [
                    {**s, "start": start + s["start"], "stop": start + s["stop"]}
                    for s in labelled["segments"]
                ],
                "idle_tail_from": start + labelled["idle_tail_from"],
                "idle_tail_frames": labelled["idle_tail_frames"],
                "num_anchors": int(labelled["anchor_ok"].sum()),
                "num_event_anchors": int((event & labelled["anchor_ok"]).sum()),
            }
        report["sources"][str(source_id)] = per_round
        print(f"[relabel] source {source_id:>3}: "
              f"r0 {per_round['0']['num_anchors']:>5} anchors "
              f"({per_round['0']['num_event_anchors']} event, {per_round['0']['idle_tail_frames']} idle) | "
              f"r1 {per_round['1']['num_anchors']:>5} anchors "
              f"({per_round['1']['num_event_anchors']} event, {per_round['1']['idle_tail_frames']} idle)",
              flush=True)

    report["totals"] = totals
    report["label_frame_counts"] = dict(sorted(label_counts.items(), key=lambda kv: -kv[1]))
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{len(report['sources'])} sources relabelled")
    print(f"  Execute frames      {totals['frames']:,}")
    print(f"  usable anchors      {totals['anchors']:,}")
    print(f"  decision anchors    {totals['event_anchors']:,} "
          f"({100 * totals['event_anchors'] / max(totals['anchors'], 1):.1f}%)")
    print(f"  idle tail dropped   {totals['idle']:,} frames")
    print("\n每条子指令覆盖的帧数:")
    for label, count in report["label_frame_counts"].items():
        print(f"  {count:>8,}  {label}")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
