#!/usr/bin/env python3
"""NERO ROS2 UMI inference entry point.

Preview is the default. Control is blocked until every hardware fact in
config/deploy.yaml has been explicitly verified.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import yaml

from env.nero_ros_umi_env import (
    NeroRosUmiEnv,
    validate_control_request,
)
from runtime.control_loop import UmiControlLoop
from runtime.policy_runtime import PolicyRuntime
from runtime.preview_logger import PreviewLogger


DEFAULT_CONFIG = Path(__file__).parent / "config" / "deploy.yaml"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inference-steps", type=int, default=16)
    parser.add_argument(
        "--frequency",
        type=float,
        default=10.0,
        help="Original UMI control frequency in Hz")
    parser.add_argument(
        "--steps-per-inference",
        type=int,
        default=6,
        help="Original UMI number of 10 Hz steps between inference calls")
    parser.add_argument(
        "--action-exec-latency",
        type=float,
        default=0.01,
        help="Drop actions that cannot lead the command path by this much")
    parser.add_argument(
        "--frame-latency",
        type=float,
        default=1.0 / 60.0,
        help="Original UMI wait margin before each cycle boundary")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Exit after N inference cycles; 0 runs continuously")
    parser.add_argument(
        "--control-action-limit",
        type=int,
        default=0,
        help=(
            "In --control mode, publish at most N future actions per plan "
            "after expired actions are removed; 0 uses the full horizon"))
    parser.add_argument("--rerun-save", type=Path)
    parser.add_argument(
        "--diagnostic-json",
        type=Path,
        help="Write cumulative camera synchronization diagnostics")
    parser.add_argument(
        "--print-action-diagnostics",
        action="store_true",
        help="Print predicted action XYZ/step ranges without changing control")
    parser.add_argument("--no-rerun", action="store_true")
    parser.add_argument("--check-config", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true")
    mode.add_argument("--control", action="store_true")
    args = parser.parse_args(argv)
    if not args.preview and not args.control:
        args.preview = True
    if args.num_inference_steps < 1:
        parser.error("--num-inference-steps must be positive")
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    if args.frequency <= 0.0:
        parser.error("--frequency must be positive")
    if args.steps_per_inference < 1:
        parser.error("--steps-per-inference must be positive")
    if args.action_exec_latency < 0.0:
        parser.error("--action-exec-latency must be nonnegative")
    if args.frame_latency < 0.0:
        parser.error("--frame-latency must be nonnegative")
    if args.max_cycles < 0:
        parser.error("--max-cycles must be nonnegative")
    if args.control_action_limit < 0:
        parser.error("--control-action-limit must be nonnegative")
    if args.control_action_limit and not args.control:
        parser.error("--control-action-limit requires --control")
    return args


def load_config(path: Path) -> Dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("deployment config must be a YAML mapping")
    return config


def print_config_summary(
        config: Dict[str, Any],
        verification_issues: list[str],
) -> None:
    obs = config["observation"]
    control = config["control_output"]
    include_d405 = bool(obs.get("include_d405", True))
    print("NERO ROS2 UMI configuration:", flush=True)
    print(f"  camera0_rgb (fisheye): {obs['fisheye_rgb_topic']}", flush=True)
    if include_d405:
        print(f"  camera1_rgb (D405):    {obs['d405_rgb_topic']}", flush=True)
    else:
        print("  camera1_rgb (D405):    disabled by checkpoint", flush=True)
    print(f"  TCP feedback:          {obs['tcp_pose_topic']}", flush=True)
    print(f"  gripper feedback:      {obs['gripper_state_topic']}", flush=True)
    print(f"  QP target:             {control['pose_topic']}", flush=True)
    print(
        f"  gripper target:        {control['gripper_control_topic']}",
        flush=True)
    print(
        f"  control verified:      {not verification_issues}",
        flush=True)
    print(
        "  translation step check: "
        + ("enabled" if config.get("safety", {}).get(
            "translation_step_check_enabled", True) else "disabled"),
        flush=True)
    for issue in verification_issues:
        print(f"  CONTROL BLOCKED: {issue}", flush=True)
    print("  rotation step check:    " + (
        "enabled" if config.get("safety", {}).get(
            "rotation_step_check_enabled", True) else "disabled"), flush=True)
    grasp = config.get("action", {}).get("grasp_sync", {})
    if grasp.get("enabled") and grasp.get("model_driven_pose", False):
        print("  grasp motion:          model pose/opening timing; jaw-only compensation",
              flush=True)
        print("  lift grip confirmation: " + (
            "enabled" if grasp.get("lift_guard", {}).get("enabled", False)
            else "disabled"), flush=True)
    verification = grasp.get("verification", {})
    if (grasp.get("enabled") and not grasp.get("model_driven_pose", False)
            and not verification.get("effort_calibrated", False)):
        print("  grasp contact: uncalibrated; lift held until contact is verified",
              flush=True)


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    verification = validate_control_request(config)
    print_config_summary(config, verification.issues)
    if args.check_config:
        return 0 if not args.control or verification.ready else 2
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required outside --check-config")
    if not args.checkpoint.expanduser().is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.control and not verification.ready:
        print(
            "REFUSING --control: hardware verification is incomplete.",
            file=sys.stderr,
            flush=True)
        return 2

    if args.control_action_limit:
        config["action"]["max_actions_per_plan"] = args.control_action_limit
        print(
            f"[slow control] max_actions_per_plan="
            f"{args.control_action_limit}",
            flush=True)
    config["action"]["minimum_command_lead_sec"] = (
        args.action_exec_latency)

    runtime = PolicyRuntime(
        args.checkpoint.expanduser().resolve(),
        config,
        args.device,
        args.num_inference_steps,
    )
    env = NeroRosUmiEnv(config, control=bool(args.control))
    logger = PreviewLogger(args.rerun_save, args.no_rerun or args.control)
    return UmiControlLoop(runtime, env, logger, config, args).run()


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
