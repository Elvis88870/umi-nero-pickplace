#!/usr/bin/env python3
"""Preview-only UMI entry with CPU affinity and bounded math threads."""

from __future__ import annotations

import os
from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))


def _parse_cpu_set(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(value) for value in part.split("-", 1))
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    available = os.sched_getaffinity(0)
    selected = cpus & available
    if not selected:
        raise RuntimeError(
            f"requested CPU set {text!r} has no available CPUs")
    return selected


def main() -> int:
    if "--control" in sys.argv:
        print("REFUSING --control: isolated test is preview-only", file=sys.stderr)
        return 2
    if "--preview" not in sys.argv:
        sys.argv.append("--preview")

    cpus = _parse_cpu_set(
        os.environ.get("UMI_INFERENCE_CPUS", "0-11"))
    os.sched_setaffinity(0, cpus)
    thread_count = min(
        len(cpus), max(1, int(os.environ.get("UMI_TORCH_THREADS", "8"))))
    os.environ["OMP_NUM_THREADS"] = str(thread_count)
    os.environ["MKL_NUM_THREADS"] = str(thread_count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(thread_count)
    os.environ["NUMEXPR_NUM_THREADS"] = str(thread_count)

    import cv2
    import torch

    torch.set_num_threads(thread_count)
    cv2.setNumThreads(1)
    print(
        f"[isolation] pid={os.getpid()} affinity="
        f"{sorted(os.sched_getaffinity(0))} torch_threads={thread_count} "
        "opencv_threads=1",
        flush=True,
    )

    from eval_nero_ros import main as eval_main

    return int(eval_main())


if __name__ == "__main__":
    raise SystemExit(main())
