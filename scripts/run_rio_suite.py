#!/usr/bin/env python3
"""Run the registered four-model RIO pilot concurrently with durable status."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
    ).strip()


def write_status(path: Path, status: dict) -> None:
    path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-config", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    suite_config = args.suite_config.resolve()
    config = yaml.safe_load(suite_config.read_text(encoding="utf-8"))
    output_root = (root / config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "suite_provenance.json"
    if status_path.exists():
        prior = json.loads(status_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete":
            raise FileExistsError(f"suite is already complete: {status_path}")

    jobs = []
    handles = []
    runner = root / "scripts" / "run_question_benchmark.py"
    for spec in config["jobs"]:
        model_config = (root / spec["config"]).resolve()
        if not model_config.is_file():
            raise FileNotFoundError(model_config)
        log_path = output_root / f"{spec['name']}.log"
        handle = log_path.open("a", encoding="utf-8", buffering=1)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(spec["gpu"])
        if spec.get("pythonpath"):
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(spec["pythonpath"]) + (os.pathsep + existing if existing else "")
        command = [sys.executable, str(runner), "--config", str(model_config)]
        process = subprocess.Popen(
            command, cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        handles.append(handle)
        jobs.append({
            "name": spec["name"], "gpu": int(spec["gpu"]),
            "config": str(model_config), "config_sha256": sha256(model_config),
            "pythonpath": spec.get("pythonpath"), "command": command,
            "pid": process.pid, "log": str(log_path), "process": process,
            "exit_code": None,
        })

    status = {
        "schema_version": "cta/rio-suite-run-v1",
        "status": "running", "started_at_utc": utc_now(),
        "suite_config": str(suite_config),
        "suite_config_sha256": sha256(suite_config),
        "git_head": git_head(root), "hostname": platform.node(),
        "python": sys.version, "jobs": [],
    }

    def serializable_jobs() -> list[dict]:
        return [
            {key: value for key, value in job.items() if key != "process"}
            for job in jobs
        ]

    status["jobs"] = serializable_jobs()
    write_status(status_path, status)
    try:
        while True:
            running = 0
            for job in jobs:
                if job["exit_code"] is not None:
                    continue
                code = job["process"].poll()
                if code is None:
                    running += 1
                else:
                    job["exit_code"] = int(code)
                    job["finished_at_utc"] = utc_now()
            status["jobs"] = serializable_jobs()
            status["running_jobs"] = running
            write_status(status_path, status)
            if running == 0:
                break
            time.sleep(5)
    finally:
        for handle in handles:
            handle.close()

    status["finished_at_utc"] = utc_now()
    status["jobs"] = serializable_jobs()
    status["status"] = "complete" if all(job["exit_code"] == 0 for job in jobs) else "failed"
    write_status(status_path, status)
    if status["status"] != "complete":
        failed = {job["name"]: job["exit_code"] for job in jobs if job["exit_code"] != 0}
        raise SystemExit(f"RIO suite failed: {failed}")


if __name__ == "__main__":
    main()
