# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
# See LICENSE in the project root for full terms.

"""
Batch HRC data collection — runs each episode in a separate subprocess
to avoid Isaac Sim singleton re-init issues.
"""
import argparse
import os
import subprocess
import sys

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hrc_task.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Global defaults from the config (everything except 'episodes' and 'schema')
    global_keys = {k: v for k, v in config.items() if k not in ("episodes", "schema", "episode_template", "dataset_name")}

    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "collect_hrc_data.py")

    for idx, episode in enumerate(config.get("episodes", [])):
        merged = {**global_keys, **episode}
        cmd = [sys.executable, script_path]
        for key, value in merged.items():
            cli_key = f"--{key.replace('_', '-')}"
            cmd.extend([cli_key, str(value)])

        print(f"\n{'='*60}")
        print(f"[Batch HRC] Episode {idx+1}/{len(config['episodes'])}: {episode.get('episode_name', '?')}")
        print(f"  cmd: {' '.join(cmd)}")
        print(f"{'='*60}")

        if args.dry_run:
            continue

        result = subprocess.run(cmd, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"})
        if result.returncode != 0:
            print(f"[Batch HRC] WARNING: episode {episode.get('episode_name', idx)} exited with code {result.returncode}")

    print("\n[Batch HRC] All episodes processed.")


if __name__ == "__main__":
    main()
