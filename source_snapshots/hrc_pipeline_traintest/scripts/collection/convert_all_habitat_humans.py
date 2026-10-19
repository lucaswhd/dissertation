# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
# See LICENSE in the project root for full terms.

#!/usr/bin/env python3
"""Batch-convert all 12 Habitat humanoid GLB assets to USD packages.

Usage:
    python scripts/convert_all_habitat_humans.py [--cache-root DIR]

Each humanoid is converted via habitat_humanoid_bridge.convert_habitat_humanoid_to_usda().
Already-cached packages are skipped automatically (cache-hit).
"""
import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from habitat_humanoid_bridge import convert_habitat_humanoid_to_usda, is_habitat_humanoid_asset

HUMANOIDS_DIR = (
    REPO_ROOT
    / "outputs"
    / "external"
    / "habitat3_0"
    / "habitat_assets"
    / "versioned_data"
    / "habitat_humanoids"
)

HUMANOID_NAMES = [
    "male_0", "male_1", "male_2", "male_3",
    "female_0", "female_1", "female_2", "female_3",
    "neutral_0", "neutral_1", "neutral_2", "neutral_3",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch convert Habitat humanoids GLB -> USD")
    parser.add_argument(
        "--cache-root",
        default=None,
        help="Override cache root directory (default: outputs/external/converted_humans)",
    )
    args = parser.parse_args()

    results = {"success": [], "skipped": [], "failed": []}
    t0 = time.time()

    for name in HUMANOID_NAMES:
        glb_path = HUMANOIDS_DIR / name / f"{name}.glb"
        if not glb_path.exists():
            print(f"[SKIP] {name}: GLB not found at {glb_path}")
            results["skipped"].append(name)
            continue
        if not is_habitat_humanoid_asset(str(glb_path)):
            print(f"[SKIP] {name}: missing companion *_motion_data_smplx.pkl")
            results["skipped"].append(name)
            continue

        print(f"\n{'='*60}")
        print(f"[CONVERT] {name}: {glb_path}")
        print(f"{'='*60}")
        try:
            usda_path = convert_habitat_humanoid_to_usda(
                asset_path=str(glb_path),
                cache_root=args.cache_root,
            )
            print(f"[OK] {name} -> {usda_path}")
            results["success"].append((name, usda_path))
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")
            results["failed"].append((name, str(exc)))

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Batch conversion finished in {elapsed:.1f}s")
    print(f"  Success: {len(results['success'])}")
    print(f"  Skipped: {len(results['skipped'])}")
    print(f"  Failed:  {len(results['failed'])}")
    for name, path in results["success"]:
        print(f"    [OK]   {name} -> {path}")
    for name in results["skipped"]:
        print(f"    [SKIP] {name}")
    for name, err in results["failed"]:
        print(f"    [FAIL] {name}: {err}")


if __name__ == "__main__":
    main()
