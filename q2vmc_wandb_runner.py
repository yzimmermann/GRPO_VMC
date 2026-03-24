#!/usr/bin/env python3
"""Launch a LapNet run and stream train_stats.csv rows to Weights & Biases."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--launch-script", type=Path, required=True)
    parser.add_argument("--stats-file", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--project", default=os.environ.get("Q2VMC_WANDB_PROJECT", "q2vmc-cluster"))
    parser.add_argument("--entity", default=os.environ.get("Q2VMC_WANDB_ENTITY", ""))
    parser.add_argument("--group", default="")
    parser.add_argument("--mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--tags", nargs="*", default=[])
    return parser.parse_args()


def load_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def coerce_value(value: str):
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


class CsvTailer:
    def __init__(self, path: Path):
        self.path = path
        self.position = 0
        self.header = ""
        self.pending = ""

    def read_rows(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        if size < self.position:
            self.position = 0
            self.header = ""
            self.pending = ""
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            handle.seek(self.position)
            chunk = handle.read()
            self.position = handle.tell()
        if not chunk:
            return []
        text = self.pending + chunk
        if not text.endswith("\n"):
            if "\n" not in text:
                self.pending = text
                return []
            text, self.pending = text.rsplit("\n", 1)
        else:
            self.pending = ""
        lines = text.splitlines()
        if not self.header:
            if not lines:
                return []
            self.header = lines[0]
            lines = lines[1:]
        if not lines:
            return []
        payload = self.header + "\n" + "\n".join(lines) + "\n"
        return list(csv.DictReader(io.StringIO(payload)))


class StdoutTailer:
    def __init__(self, path: Path):
        self.path = path
        self.position = 0
        self.pattern = re.compile(
            r"Step\s+(?P<step>\d+):.*?grpo_obj=(?P<grpo_objective>[-+0-9.eE]+),\s+"
            r"clip_mean=(?P<grpo_clip_mean>[-+0-9.eE]+),\s+"
            r"clip_max=(?P<grpo_clip_max>[-+0-9.eE]+)"
        )

    def read_metrics(self) -> list[tuple[int, dict[str, object]]]:
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        if size < self.position:
            self.position = 0
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(self.position)
            lines = handle.readlines()
            self.position = handle.tell()
        if not lines:
            return []
        results: list[tuple[int, dict[str, object]]] = []
        for line in lines:
            match = self.pattern.search(line)
            if not match:
                continue
            step = int(match.group("step"))
            metrics: dict[str, object] = {
                "iteration": step,
                "lapnet_step": step,
            }
            for key in ("grpo_objective", "grpo_clip_mean", "grpo_clip_max"):
                metrics[key] = coerce_value(match.group(key))
            results.append((step, metrics))
        return results


def row_to_metrics(row: dict[str, str]) -> tuple[int, dict[str, object]]:
    metrics: dict[str, object] = {}
    iteration = 0
    for key, value in row.items():
        if value in ("", None):
            continue
        cast_value = coerce_value(value)
        if key == "t":
            iteration = int(cast_value)
            metrics["iteration"] = iteration
        elif key == "step":
            iteration = int(cast_value)
            metrics["iteration"] = iteration
            metrics["lapnet_step"] = cast_value
        else:
            metrics[key] = cast_value
    if "iteration" not in metrics:
        metrics["iteration"] = iteration
    return iteration, metrics


def init_wandb(args: argparse.Namespace, metadata: dict):
    if args.mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "wandb is not installed in this environment. "
            "Install it or run with --mode=disabled."
        ) from exc

    os.environ.setdefault("WANDB_DIR", str(args.run_dir / "wandb"))
    run = wandb.init(
        project=args.project,
        entity=args.entity or None,
        group=args.group or None,
        name=args.run_id,
        job_type="train",
        mode=args.mode,
        tags=args.tags,
        config=metadata,
    )
    wandb.define_metric("iteration")
    return run


def write_wandb_metadata(run, path: Path) -> None:
    if run is None:
        return
    payload = {
        "id": run.id,
        "name": run.name,
        "project": run.project,
        "entity": getattr(run, "entity", None),
        "group": run.group,
        "url": run.url,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata)
    metadata.setdefault("host", socket.gethostname())
    metadata.setdefault("launch_script", str(args.launch_script))
    metadata.setdefault("stats_file", str(args.stats_file))

    wandb_run = init_wandb(args, metadata)
    write_wandb_metadata(wandb_run, args.run_dir / "wandb_run.json")

    process = subprocess.Popen(["/bin/bash", str(args.launch_script)], cwd=args.run_dir)
    tailer = CsvTailer(args.stats_file)
    stdout_tailer = StdoutTailer(args.run_dir / "stdout.log")
    last_metrics: dict[str, object] = {}
    seen_stdout_steps: set[int] = set()

    try:
        while True:
            rows = tailer.read_rows()
            for row in rows:
                iteration, metrics = row_to_metrics(row)
                last_metrics = metrics
                if wandb_run is not None:
                    wandb_run.log(metrics, step=iteration)
            for iteration, metrics in stdout_tailer.read_metrics():
                if iteration in seen_stdout_steps:
                    continue
                seen_stdout_steps.add(iteration)
                last_metrics.update(metrics)
                if wandb_run is not None:
                    wandb_run.log(metrics, step=iteration)
            if process.poll() is not None:
                break
            time.sleep(args.poll_seconds)

        rows = tailer.read_rows()
        for row in rows:
            iteration, metrics = row_to_metrics(row)
            last_metrics = metrics
            if wandb_run is not None:
                wandb_run.log(metrics, step=iteration)
        for iteration, metrics in stdout_tailer.read_metrics():
            if iteration in seen_stdout_steps:
                continue
            seen_stdout_steps.add(iteration)
            last_metrics.update(metrics)
            if wandb_run is not None:
                wandb_run.log(metrics, step=iteration)

        return_code = process.wait()
        if wandb_run is not None:
            for key, value in last_metrics.items():
                wandb_run.summary[f"final_{key}"] = value
            wandb_run.summary["return_code"] = return_code
            wandb_run.finish(exit_code=return_code)
        return return_code
    except KeyboardInterrupt:
        process.terminate()
        return_code = process.wait()
        if wandb_run is not None:
            wandb_run.summary["return_code"] = return_code
            wandb_run.finish(exit_code=return_code)
        raise


if __name__ == "__main__":
    sys.exit(main())
