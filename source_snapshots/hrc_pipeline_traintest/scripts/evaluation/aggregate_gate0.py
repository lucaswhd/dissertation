#!/usr/bin/env python3
"""Aggregate per-process Gate 0 case results into one pickup/radio verdict.

Each isolated case writes ``<output_root>/<UTC>/source_XXXX_task/result.json``.
This tool collects them all and reports the Gate 0 pass condition
(pickup 5/5 and radio 5/5) plus a plumbing view (how many cases ran without
raising). For a --max-steps smoke run, every case is partial so successes stay
0; the smoke is healthy when ``num_results == 10`` and ``cases_with_error == 0``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    for result_path in sorted(args.output_root.glob("*/source_*/result.json")):
        results.append(json.loads(result_path.read_text(encoding="utf-8")))

    per_task: dict[str, dict[str, int]] = {}
    for task in ("pickup", "radio"):
        rows = [r for r in results if r.get("task") == task]
        successes = sum(
            r.get("success") is True and not r.get("partial_smoke") for r in rows
        )
        errored = sum(1 for r in rows if r.get("error"))
        per_task[task] = {
            "attempts": len(rows),
            "successes": successes,
            "errored": errored,
        }

    errors = [
        {
            "source_id": r.get("source_id"),
            "task": r.get("task"),
            "error": r.get("error"),
        }
        for r in results
        if r.get("error")
    ]

    passed_full = all(
        per_task[t] == {"attempts": 5, "successes": 5, "errored": 0}
        for t in ("pickup", "radio")
    )

    summary = {
        "gate": "Gate 0 (isolated per-process)",
        "num_results": len(results),
        "cases_with_error": len(errors),
        "per_task": per_task,
        "errors": errors,
        "passed_full_gate": passed_full,
        "required": "pickup 5/5 and radio 5/5 with no --max-steps",
    }

    out_path = args.output_root / "gate0_aggregate.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if passed_full else 2


if __name__ == "__main__":
    raise SystemExit(main())
