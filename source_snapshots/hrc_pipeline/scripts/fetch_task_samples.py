#!/usr/bin/env python3
"""Download ONE sample hdf5 per BEHAVIOR-1K task that is not already present.

Network note: this box routes HF traffic through an HTTP proxy (HTTP_PROXY) while
the httpx client used by huggingface_hub trips over the socks5h ALL_PROXY and fails
with SSL EOF. So we use the stdlib ``urllib`` (HTTP-proxy aware) for both the repo
file listing (HF API) and the file download (resolve URL).

- Repo: behavior-1k/2025-challenge-rawdata (public dataset), tasks task-0000..0049.
- Tasks already fully downloaded under DATA_ROOT/2025-challenge-rawdata/task-XXXX/
  are SKIPPED. One sample per remaining task goes to DATA_ROOT/task_samples/task-XXXX/.
- Concurrent, skip-existing (resumable), per-task retries.

Background:
    nohup /root/behavior-venv/bin/python scripts/fetch_task_samples.py \
        > /root/hrc_pipeline/logs/fetch_task_samples.log 2>&1 &
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "behavior-1k/2025-challenge-rawdata"
API_TREE = "https://huggingface.co/api/datasets/{repo}/tree/main/{task}"
RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/root/autodl-tmp/datasets"))
FULL_ROOT = DATA_ROOT / "2025-challenge-rawdata"
SAMPLE_ROOT = DATA_ROOT / "task_samples"
N_TASKS = 50
MAX_WORKERS = 6
MAX_RETRIES = 4
TIMEOUT = 60


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def task_already_full(task_id: str) -> bool:
    d = FULL_ROOT / task_id
    return d.is_dir() and any(d.glob("*.hdf5"))


def sample_already_have(task_id: str) -> Path | None:
    d = SAMPLE_ROOT / task_id
    if d.is_dir():
        for f in sorted(d.glob("*.hdf5")):
            return f
    return None


def http_get_json(url: str) -> list | dict:
    req = urllib.request.Request(url, headers={"User-Agent": "hrc-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def first_hdf5_in_repo(task_id: str) -> str | None:
    url = API_TREE.format(repo=REPO, task=task_id)
    try:
        entries = http_get_json(url)
    except Exception as exc:
        log(f"  {task_id}: list failed: {type(exc).__name__}: {exc}")
        return None
    names = sorted(e["path"] for e in entries
                   if isinstance(e, dict) and e.get("path", "").endswith(".hdf5"))
    return names[0] if names else None


def download(url: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "hrc-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        expected = resp.headers.get("Content-Length")
        expected = int(expected) if expected is not None else None
        with open(tmp, "wb") as fh:
            size = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                size += len(chunk)
    # Guard against silently-truncated downloads (proxy drops mid-stream).
    if expected is not None and size != expected:
        tmp.unlink(missing_ok=True)
        raise IOError(f"truncated: got {size} of {expected} bytes")
    tmp.rename(dest)
    return size


def fetch_one(task_id: str) -> tuple[str, str, str]:
    existing = sample_already_have(task_id)
    if existing is not None:
        return task_id, "skip", f"already have {existing.name}"

    repo_path = first_hdf5_in_repo(task_id)
    if repo_path is None:
        return task_id, "missing", "no hdf5 in repo"

    dest = SAMPLE_ROOT / repo_path  # repo_path already starts with task-XXXX/
    url = RESOLVE.format(repo=REPO, path=repo_path)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            size = download(url, dest)
            return task_id, "ok", f"{Path(repo_path).name} ({size/1e6:.1f} MB)"
        except Exception as exc:
            if attempt < MAX_RETRIES:
                wait = min(30, 5 * attempt)
                log(f"  {task_id}: attempt {attempt} failed ({type(exc).__name__}); retry {wait}s")
                time.sleep(wait)
            else:
                return task_id, "fail", f"{type(exc).__name__}: {exc}"
    return task_id, "fail", "exhausted"


def main() -> int:
    SAMPLE_ROOT.mkdir(parents=True, exist_ok=True)
    all_tasks = [f"task-{i:04d}" for i in range(N_TASKS)]
    full = [t for t in all_tasks if task_already_full(t)]
    todo = [t for t in all_tasks if t not in full]

    log(f"Total tasks: {N_TASKS}")
    log(f"Already full (skip): {len(full)} -> {full}")
    log(f"To fetch sample for: {len(todo)} tasks")
    log(f"Sample dir: {SAMPLE_ROOT}")

    results: dict[str, tuple[str, str]] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_one, t): t for t in todo}
        for fut in as_completed(futs):
            t = futs[fut]
            tid, status, detail = fut.result()
            results[tid] = (status, detail)
            log(f"[{status.upper():7s}] {tid}: {detail}")

    ok = [t for t, (s, _) in results.items() if s == "ok"]
    skip = [t for t, (s, _) in results.items() if s == "skip"]
    bad = [t for t, (s, _) in results.items() if s in ("fail", "missing")]
    log("=" * 60)
    log(f"DONE. ok={len(ok)} skip={len(skip)} failed/missing={len(bad)}")
    if bad:
        log(f"Problem tasks: {bad}")
    (SAMPLE_ROOT / "_fetch_done.txt").write_text(
        f"ok={len(ok)} skip={len(skip)} bad={len(bad)}\nbad={bad}\n", encoding="utf-8")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
