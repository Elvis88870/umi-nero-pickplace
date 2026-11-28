#!/usr/bin/env python3
"""Measure right-gripper command-to-feedback response latency.

This probe publishes only to the gripper command topic. It never publishes to
the arm pose, joint, or teleoperation topics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Sample:
    monotonic: float
    stamp: float
    value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--command-topic",
        default="/right/gripper/joint_states_ctrl")
    parser.add_argument(
        "--feedback-topic",
        default="/right/gripper/joint_state")
    parser.add_argument("--joint-name", default="right_gripper_joint")
    parser.add_argument(
        "--amplitude-m",
        type=float,
        default=0.004,
        help="Opening step from the measured initial width; capped at 10 mm")
    parser.add_argument(
        "--cycles",
        type=int,
        default=3,
        help="Number of open-and-return cycles")
    parser.add_argument("--settle-sec", type=float, default=0.35)
    parser.add_argument("--move-window-sec", type=float, default=0.8)
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually publish gripper commands; otherwise run a dry check")
    return parser.parse_args()


def percentile_stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "p50_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    p50, p95 = np.percentile(array, [50.0, 95.0])
    return {
        "count": int(array.size),
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def response_latency(
    samples: list[Sample],
    command_time: float,
    initial: float,
    target: float,
    noise_sigma: float,
) -> Optional[dict]:
    direction = 1.0 if target >= initial else -1.0
    command_size = abs(target - initial)
    if command_size < 1e-6:
        return None
    response = np.asarray(
        [(sample.value - initial) * direction for sample in samples],
        dtype=np.float64,
    )
    times = np.asarray(
        [sample.monotonic for sample in samples], dtype=np.float64)
    thresholds = {
        "onset": max(command_size * 0.10, noise_sigma * 5.0, 1e-5),
        "p10": max(command_size * 0.10, noise_sigma * 5.0, 1e-5),
        "p50": command_size * 0.50,
        "p90": command_size * 0.90,
    }
    result = {
        "command_m": float(target - initial),
        "noise_sigma_m": float(noise_sigma),
        "onset_threshold_m": float(thresholds["onset"]),
    }
    for name, threshold in thresholds.items():
        indices = np.flatnonzero(
            (times >= command_time) & (response >= threshold))
        result[f"{name}_latency_ms"] = (
            float((times[int(indices[0])] - command_time) * 1000.0)
            if indices.size else None)
    return result


def main() -> int:
    args = parse_args()
    amplitude = float(np.clip(args.amplitude_m, 0.001, 0.010))
    if args.cycles < 1:
        raise ValueError("--cycles must be positive")

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from sensor_msgs.msg import JointState

    class GripperProbe(Node):
        def __init__(self) -> None:
            super().__init__("gripper_latency_probe")
            self.samples: list[Sample] = []
            qos = QoSProfile(
                depth=100,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.publisher = self.create_publisher(
                JointState, args.command_topic, 1)
            self.create_subscription(
                JointState,
                args.feedback_topic,
                self._feedback_callback,
                qos,
            )

        def _feedback_callback(self, msg: JointState) -> None:
            by_name = dict(zip(msg.name, msg.position))
            if args.joint_name in by_name:
                value = float(by_name[args.joint_name])
            elif msg.position:
                value = float(msg.position[0])
            else:
                return
            if not math.isfinite(value):
                return
            stamp = (
                float(msg.header.stamp.sec)
                + float(msg.header.stamp.nanosec) * 1e-9)
            self.samples.append(
                Sample(time.monotonic(), stamp, value))

    rclpy.init()
    node = GripperProbe()
    report: dict = {
        "command_topic": args.command_topic,
        "feedback_topic": args.feedback_topic,
        "joint_name": args.joint_name,
        "amplitude_requested_m": args.amplitude_m,
        "amplitude_used_m": amplitude,
        "executed": bool(args.execute),
    }
    try:
        deadline = time.monotonic() + args.timeout_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
            if len(node.samples) >= 10:
                break
        if len(node.samples) < 10:
            raise RuntimeError("did not receive enough gripper feedback")
        if not args.execute:
            report.update({
                "status": "dry_run",
                "feedback_fps_approx": None,
                "initial_width_m": node.samples[-1].value,
                "note": "No command was published. Add --execute to measure.",
            })
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0

        subscriber_deadline = time.monotonic() + args.timeout_sec
        while (
            time.monotonic() < subscriber_deadline
            and node.publisher.get_subscription_count() < 1
        ):
            rclpy.spin_once(node, timeout_sec=0.02)
        if node.publisher.get_subscription_count() < 1:
            raise RuntimeError("gripper command topic has no subscriber")

        baseline_samples: list[Sample] = []
        baseline_end = time.monotonic() + 0.6
        while time.monotonic() < baseline_end:
            rclpy.spin_once(node, timeout_sec=0.01)
            baseline_samples = node.samples[-100:]
        baseline_values = np.asarray(
            [sample.value for sample in baseline_samples[10:]],
            dtype=np.float64,
        )
        baseline = float(np.median(baseline_values))
        noise_sigma = float(np.std(baseline_values - baseline))
        target_open = min(0.098, baseline + amplitude)
        target_return = baseline
        report.update({
            "initial_width_m": baseline,
            "open_target_m": target_open,
            "return_target_m": target_return,
            "baseline_noise_sigma_m": noise_sigma,
        })

        def publish_width(width: float) -> tuple[float, float]:
            command = JointState()
            command.header.stamp = node.get_clock().now().to_msg()
            command.name = [args.joint_name]
            command.position = [float(width)]
            node.publisher.publish(command)
            return (
                time.monotonic(),
                float(command.header.stamp.sec)
                + float(command.header.stamp.nanosec) * 1e-9,
            )

        def collect_after_command(
            command_mono: float,
            duration: float,
        ) -> list[Sample]:
            start_index = len(node.samples)
            end = command_mono + duration
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.01)
            return node.samples[start_index:]

        # Initialize the serial controller to the current measured width.
        publish_width(baseline)
        time.sleep(0.2)
        responses: list[dict] = []
        timestamps: list[float] = []
        command_stamps: list[float] = []
        try:
            for cycle in range(args.cycles):
                for label, target in (
                    ("open", target_open),
                    ("return", target_return),
                ):
                    initial = node.samples[-1].value
                    command_mono, command_stamp = publish_width(target)
                    samples = collect_after_command(
                        command_mono, args.move_window_sec)
                    response = response_latency(
                        samples,
                        command_mono,
                        float(initial),
                        float(target),
                        noise_sigma,
                    )
                    if response is not None:
                        response.update({
                            "cycle": cycle,
                            "direction": label,
                        })
                        responses.append(response)
                    timestamps.append(command_mono)
                    command_stamps.append(command_stamp)
                    settle_end = (
                        command_mono
                        + args.move_window_sec
                        + args.settle_sec
                    )
                    while time.monotonic() < settle_end:
                        rclpy.spin_once(node, timeout_sec=0.01)
        finally:
            publish_width(baseline)
            restore_end = time.monotonic() + 0.5
            while time.monotonic() < restore_end:
                rclpy.spin_once(node, timeout_sec=0.01)

        onset = [
            item["onset_latency_ms"]
            for item in responses
            if item["onset_latency_ms"] is not None
        ]
        p10 = [
            item["p10_latency_ms"]
            for item in responses
            if item["p10_latency_ms"] is not None
        ]
        p50 = [
            item["p50_latency_ms"]
            for item in responses
            if item["p50_latency_ms"] is not None
        ]
        p90 = [
            item["p90_latency_ms"]
            for item in responses
            if item["p90_latency_ms"] is not None
        ]
        report.update({
            "status": "completed",
            "responses": responses,
            "onset": percentile_stats(onset),
            "p10": percentile_stats(p10),
            "p50": percentile_stats(p50),
            "p90": percentile_stats(p90),
            "final_width_m": node.samples[-1].value,
        })
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
