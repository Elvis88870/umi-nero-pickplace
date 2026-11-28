#!/usr/bin/env python3
"""Measure D405 shared-memory writer availability and timestamp intervals."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import time
from typing import Dict, Optional

import numpy as np

try:
    from .d405_shm_reader import SharedMemoryD405Reader
except ImportError:
    from d405_shm_reader import SharedMemoryD405Reader


def percentile_stats(values: deque[float]) -> Dict[str, Optional[float]]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
    p50, p95, p99 = np.percentile(array, [50.0, 95.0, 99.0])
    return {
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "max_ms": float(array.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shm-name", default="/umi_d405_right")
    parser.add_argument("--warmup-sec", type=float, default=10.0)
    parser.add_argument("--duration-sec", type=float, default=300.0)
    parser.add_argument("--poll-sec", type=float, default=0.005)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reader = SharedMemoryD405Reader(args.shm_name)
    warmup_end = time.monotonic() + max(0.0, args.warmup_sec)
    measurement_start = warmup_end
    deadline = measurement_start + max(0.0, args.duration_sec)
    last_sequence = 0
    previous_stamp: Optional[float] = None
    initial_total: Optional[int] = None
    final_total: Optional[int] = None
    observed_frames = 0
    intervals_ms: deque[float] = deque()
    ages_ms: deque[float] = deque()
    first_observed_stamp: Optional[float] = None
    last_observed_stamp: Optional[float] = None

    try:
        while time.monotonic() < deadline:
            latest = reader.latest_metadata()
            now = time.monotonic()
            if now < measurement_start or latest is None:
                time.sleep(max(0.001, args.poll_sec))
                continue
            if initial_total is None:
                initial_total = latest.sequence
            if latest.sequence > last_sequence:
                last_sequence = latest.sequence
                observed_frames += 1
                if first_observed_stamp is None:
                    first_observed_stamp = latest.timestamp
                if previous_stamp is not None and latest.timestamp > previous_stamp:
                    intervals_ms.append(
                        (latest.timestamp - previous_stamp) * 1000.0)
                previous_stamp = latest.timestamp
                last_observed_stamp = latest.timestamp
                ages_ms.append(
                    (time.monotonic_ns() - latest.capture_monotonic_ns) / 1e6)
            final_total = latest.sequence
            time.sleep(max(0.001, args.poll_sec))
    finally:
        diagnostics = reader.diagnostics()
        reader.close()

    wall_duration = max(0.0, time.monotonic() - measurement_start)
    header_duration = (
        last_observed_stamp - first_observed_stamp
        if first_observed_stamp is not None and last_observed_stamp is not None
        else 0.0
    )
    writer_frames = (
        final_total - initial_total
        if final_total is not None and initial_total is not None
        else 0
    )
    report = {
        "shm_name": args.shm_name,
        "warmup_requested_sec": args.warmup_sec,
        "duration_requested_sec": args.duration_sec,
        "measurement_wall_time_sec": wall_duration,
        "first_to_last_frame_sec": header_duration,
        "writer_frames_in_window": writer_frames,
        "writer_fps": (
            writer_frames / header_duration if header_duration > 0.0 else None),
        "metadata_reader_observed_frames": observed_frames,
        "metadata_reader_fps": (
            observed_frames / header_duration if header_duration > 0.0 else None),
        "frame_interval": percentile_stats(intervals_ms),
        "latest_frame_age": percentile_stats(ages_ms),
        "ring_diagnostics": diagnostics,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
