# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
# See LICENSE in the project root for full terms.

"""
Verify BehaviorTask + Human NPC integration.

Loads a real BEHAVIOR task scene with a Fetch robot,
adds the human NPC, captures robot-view and third-person snapshots.
"""
import argparse
import os
import traceback

import imageio.v2 as imageio
import numpy as np

import omnigibson as og
from omnigibson.macros import gm

gm.HEADLESS = True
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = False

from hrc_utils import (
    DEFAULT_HUMAN_USD,
    add_human_to_scene,
    download_human_usd,
    ensure_parent,
    extract_first_rgb,
    normalize_rgb,
    orient_robot_towards_target,
    to_numpy,
    update_viewer_camera,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activity", default="assembling_gift_baskets")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--human-usd", default=DEFAULT_HUMAN_USD)
    parser.add_argument("--robot-output", default="outputs/behavior_task_with_human/robot_view.png")
    parser.add_argument("--third-output", default="outputs/behavior_task_with_human/third_view.png")
    args = parser.parse_args()

    ensure_parent(args.robot_output)
    ensure_parent(args.third_output)
    download_human_usd(args.human_usd)

    img_h = (args.resolution + 15) // 16 * 16
    img_w = (int(args.resolution * 4 / 3) + 15) // 16 * 16

    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "house_double_floor_lower",
            "trav_map_resolution": 0.1,
            "default_erosion_radius": 0.0,
            "trav_map_with_objects": True,
            "scene_source": "OG",
            "include_robots": True,
        },
        "robots": [
            {
                "model": "fetch",
                "obs_modalities": ["rgb"],
                "action_type": "continuous",
                "action_normalize": True,
                "sensor_config": {
                    "VisionSensor": {
                        "sensor_kwargs": {
                            "image_height": img_h,
                            "image_width": img_w,
                        }
                    }
                },
            }
        ],
        "objects": [],
        "task": {
            "type": "BehaviorTask",
            "activity_name": args.activity,
            "activity_definition_id": 0,
            "activity_instance_id": 0,
            "online_object_sampling": False,
            "use_presampled_robot_pose": False,
            "termination_config": {"max_steps": 500},
            "reward_config": {"r_potential": 1.0},
        },
    }

    print(f"[Behavior+Human] Loading scene with activity={args.activity}, res={img_w}x{img_h}", flush=True)
    env = og.Environment(configs=cfg)
    try:
        obs, _ = env.reset()
        print("[Behavior+Human] Scene reset done, adding human NPC...", flush=True)

        human = add_human_to_scene(env, args.human_usd)
        orient_robot_towards_target(env.robots[0], to_numpy(human.get_position_orientation()[0]))
        print("[Behavior+Human] Human added, stepping physics...", flush=True)

        zero_action = np.zeros(env.robots[0].action_dim, dtype=np.float32)
        for i in range(5):
            obs, _, _, _, _ = env.step(zero_action, n_render_iterations=1)
            print(f"  step {i+1}/5 done", flush=True)

        # Robot head camera view
        robot_frame, cam_key = extract_first_rgb(obs)
        if robot_frame is None:
            print("[Behavior+Human] WARNING: No robot RGB frame found!", flush=True)
            # Debug: dump obs structure
            def dump_obs(d, prefix=""):
                if isinstance(d, dict):
                    for k, v in d.items():
                        dump_obs(v, f"{prefix}.{k}" if prefix else k)
                elif hasattr(d, 'shape'):
                    print(f"  obs key={prefix}, shape={d.shape}, dtype={d.dtype}", flush=True)
            dump_obs(obs)
        else:
            imageio.imwrite(args.robot_output, robot_frame)
            print(f"[Behavior+Human] Saved robot view ({robot_frame.shape}): {os.path.abspath(args.robot_output)}", flush=True)
            print(f"  Camera key: {cam_key}", flush=True)

        # Third-person view
        robot_pos, _ = env.robots[0].get_position_orientation()
        human_pos, _ = human.get_position_orientation()
        update_viewer_camera(to_numpy(robot_pos), to_numpy(human_pos))
        viewer_obs, _ = og.sim.viewer_camera.get_obs()
        third_frame = normalize_rgb(to_numpy(viewer_obs["rgb"]))
        if third_frame is None:
            print("[Behavior+Human] WARNING: No third-person RGB frame!", flush=True)
        else:
            imageio.imwrite(args.third_output, third_frame)
            print(f"[Behavior+Human] Saved third view ({third_frame.shape}): {os.path.abspath(args.third_output)}", flush=True)

        print("[Behavior+Human] SUCCESS", flush=True)

    except Exception:
        traceback.print_exc()
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
