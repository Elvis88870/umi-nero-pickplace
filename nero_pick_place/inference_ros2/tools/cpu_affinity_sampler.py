#!/usr/bin/env python3
"""Sample per-core utilization and process affinity for isolation tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def read_cpu():
    result = {}
    with open("/proc/stat", "r", encoding="ascii") as handle:
        for line in handle:
            if not line.startswith("cpu") or line.startswith("cpu "):
                continue
            fields = line.split()
            values = [int(item) for item in fields[1:]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            result[fields[0]] = (sum(values), idle)
    return result


def processes(patterns):
    found = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(
                b"\0", b" ").decode("utf-8", "replace").strip()
            if not any(pattern in command for pattern in patterns):
                continue
            stat = (entry / "stat").read_text().split()
            ticks = int(stat[13]) + int(stat[14])
            found[int(entry.name)] = {
                "command": command,
                "ticks": ticks,
                "affinity_text": (
                    next(
                        (item.split(":", 1)[1].strip()
                         for item in (entry / "task" / entry.name / "status")
                         .read_text().splitlines()
                         if item.startswith("Cpus_allowed_list:")),
                        "",
                    )
                ),
            }
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-sec", type=float, default=310)
    parser.add_argument("--interval-sec", type=float, default=1)
    parser.add_argument(
        "--process-match",
        default="mjpeg_camera_node,d405_shm_writer,eval_nero_ros",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    patterns = [item for item in args.process_match.split(",") if item]
    previous_cpu = read_cpu()
    previous_processes = processes(patterns)
    core_samples = {name: [] for name in previous_cpu}
    process_samples = {
        pid: [] for pid in previous_processes
    }
    started = time.monotonic()
    try:
        while time.monotonic() - started < args.duration_sec:
            time.sleep(args.interval_sec)
            current_cpu = read_cpu()
            current_processes = processes(patterns)
            for name, (total, idle) in current_cpu.items():
                old_total, old_idle = previous_cpu.get(name, (total, idle))
                total_delta = max(1, total - old_total)
                core_samples.setdefault(name, []).append(
                    100 * (total_delta - (idle - old_idle)) / total_delta)
            for pid, item in current_processes.items():
                old = previous_processes.get(pid)
                if old is not None:
                    elapsed_ticks = 100 * args.interval_sec
                    process_samples.setdefault(pid, []).append(
                        100 * (item["ticks"] - old["ticks"]) / elapsed_ticks)
            previous_cpu = current_cpu
            previous_processes = current_processes
    except KeyboardInterrupt:
        pass
    report = {
        "duration_sec": time.monotonic() - started,
        "cores": {
            name: {
                "mean_percent": sum(values) / len(values),
                "max_percent": max(values),
            }
            for name, values in sorted(core_samples.items()) if values
        },
        "processes": {
            str(pid): {
                "command": item["command"],
                "affinity": item["affinity_text"],
                "mean_cpu_percent": (
                    sum(process_samples.get(pid, []))
                    / len(process_samples[pid])
                    if process_samples.get(pid) else None),
                "max_cpu_percent": (
                    max(process_samples[pid])
                    if process_samples.get(pid) else None),
            }
            for pid, item in previous_processes.items()
        },
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
