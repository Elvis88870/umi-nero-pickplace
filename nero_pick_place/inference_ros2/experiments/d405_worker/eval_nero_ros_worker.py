#!/usr/bin/env python3
"""Preview-only UMI inference with MJPEG/YUYV decode in a worker process."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT))


def _cpus(text):
    result = set()
    for part in text.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            result.update(range(lo, hi + 1))
        elif part:
            result.add(int(part))
    selected = result & os.sched_getaffinity(0)
    if not selected:
        raise RuntimeError(f"invalid inference CPU set {text}")
    return selected


def main():
    if "--control" in sys.argv:
        print("REFUSING --control: worker preview is preview-only", file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_ROOT / "config" / "deploy.yaml")
    parser.add_argument(
        "--worker-config", type=Path,
        default=EXPERIMENT_ROOT / "config" / "decoder_worker.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inference-steps", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--rerun-save", type=Path)
    parser.add_argument("--diagnostic-json", type=Path)
    parser.add_argument("--no-rerun", action="store_true")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Accepted for compatibility; this experiment is preview-only")
    args = parser.parse_args()

    cpus = _cpus(os.environ.get("UMI_INFERENCE_CPUS", "0-11,13"))
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
        f"[main isolation] pid={os.getpid()} affinity="
        f"{sorted(os.sched_getaffinity(0))} torch_threads={thread_count} "
        "opencv_threads=1",
        flush=True,
    )

    from eval_nero_ros import load_config
    from env.ros_observation_buffer import ObservationUnavailable
    from runtime.control_loop import episode_start_pose
    from runtime.policy_runtime import PolicyRuntime
    from runtime.preview_logger import PreviewLogger
    from experiments.d405_worker.worker_preview_env import WorkerPreviewEnv

    config = load_config(args.config)
    with args.worker_config.open("r", encoding="utf-8") as handle:
        config.update(yaml.safe_load(handle))
    runtime = PolicyRuntime(
        args.checkpoint.expanduser().resolve(),
        config,
        args.device,
        args.num_inference_steps,
    )
    env = WorkerPreviewEnv(config)
    logger = PreviewLogger(args.rerun_save, args.no_rerun)
    env.start()
    plan_interval = float(config["action"]["replan_interval_sec"])
    action_interval = float(config["action"]["action_interval_sec"])
    diagnostic_interval = float(config["observation"]["diagnostic_interval_sec"])
    index = 0
    last_diagnostic = time.monotonic()
    try:
        obs = env.wait_for_obs(args.timeout)
        start_pose = episode_start_pose(obs)
        while True:
            if args.max_cycles and index >= args.max_cycles:
                break
            started = time.monotonic()
            actions, inference_ms = runtime.infer(obs, start_pose)
            timestamps = (
                float(obs["timestamp"][-1])
                + np.arange(actions.shape[0], dtype=np.float64)
                * action_interval)
            logger.log(index, obs, actions, timestamps, inference_ms)
            print(
                f"[worker-infer {index:04d}] obs={obs['timestamp'][-1]:.6f} "
                f"model={inference_ms:.1f}ms "
                f"cycle={(time.monotonic()-started)*1000:.1f}ms",
                flush=True,
            )
            index += 1
            if (
                diagnostic_interval > 0
                and time.monotonic() - last_diagnostic >= diagnostic_interval
            ):
                print(
                    "[worker diagnostics] "
                    + json.dumps(env.diagnostics(), separators=(",", ":")),
                    flush=True)
                last_diagnostic = time.monotonic()
            remaining = plan_interval - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
            try:
                obs = env.get_obs()
            except ObservationUnavailable as exc:
                print(f"[worker observation rejected] {exc}", flush=True)
                obs = env.wait_for_obs(args.timeout)
    except KeyboardInterrupt:
        pass
    finally:
        diagnostics = env.diagnostics()
        print(
            "[worker diagnostics final] "
            + json.dumps(diagnostics, ensure_ascii=False),
            flush=True)
        if args.diagnostic_json:
            path = args.diagnostic_json.expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n")
        env.stop()
    print("[preview only] no robot command was published", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
