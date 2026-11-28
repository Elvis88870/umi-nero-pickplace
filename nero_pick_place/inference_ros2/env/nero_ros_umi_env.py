"""ROS2-only UMI environment adapter for NERO.

This module does not import or modify UR RTDE, WSG, QP, ZMQ bridge, or driver
code. In preview mode it creates no command publishers.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .ros_action_scheduler import (
    PlanRejected,
    RosActionScheduler,
    ScheduleReport,
)
from .ros_observation_buffer import (
    ObservationUnavailable,
    RosObservationBuffer,
)


@dataclass(frozen=True)
class ControlVerificationReport:
    ready: bool
    issues: List[str]


CONTROL_VERIFICATION_KEYS = (
    "operator_confirmed",
    "tcp_frame_verified",
    "tcp_position_unit_verified",
    "tcp_orientation_representation_verified",
    "tcp_training_semantics_verified",
    "gripper_unit_verified",
    "gripper_direction_verified",
    "downstream_interface_verified",
    "qp_timeout_stop_verified",
    "exclusive_control_path_verified",
)


def validate_control_request(config: Dict[str, Any]) -> ControlVerificationReport:
    """Return every reason why control is not yet allowed."""
    issues: List[str] = []
    verification = config.get("hardware_verification", {})
    for key in CONTROL_VERIFICATION_KEYS:
        if verification.get(key) is not True:
            issues.append(f"hardware_verification.{key} is not confirmed")

    safety = config.get("safety", {})
    safety_keys = [
        "gripper_min_m",
        "gripper_max_m",
    ]
    if bool(safety.get("translation_step_check_enabled", True)):
        safety_keys.append("max_translation_step_m")
    if bool(safety.get("rotation_step_check_enabled", True)):
        safety_keys.append("max_rotation_step_rad")
    if bool(safety.get("workspace_enabled", True)):
        safety_keys = [
            "workspace_min_xyz_m",
            "workspace_max_xyz_m",
            *safety_keys,
        ]
    for key in safety_keys:
        if safety.get(key) is None:
            issues.append(f"safety.{key} is not configured")

    control = config.get("control_output", {})
    if control.get("mode") != "direct_ros_pose":
        issues.append(
            "control_output.mode must be direct_ros_pose for this adapter")
    for key in (
        "pose_topic",
        "gripper_control_topic",
        "teleop_active_topic",
        "expected_pose_frame",
    ):
        if not control.get(key):
            issues.append(f"control_output.{key} is empty")
    return ControlVerificationReport(ready=not issues, issues=issues)


class NeroRosUmiEnv:
    def __init__(self, config: Dict[str, Any], control: bool):
        self.config = config
        self.control = bool(control)
        if self.control:
            report = validate_control_request(config)
            if not report.ready:
                raise RuntimeError(
                    "control verification failed: " + "; ".join(report.issues))

        import rclpy

        self.rclpy = rclpy
        self.owns_rclpy_context = not rclpy.ok()
        if self.owns_rclpy_context:
            rclpy.init()
        self.buffer = RosObservationBuffer(config)
        self.scheduler: Optional[RosActionScheduler] = None
        self.started = False
        self.executor = None
        self.executor_thread: Optional[threading.Thread] = None
        self.last_plan_report: Optional[ScheduleReport] = None
        self.last_plan_error: Optional[str] = None

    def start(self) -> None:
        if self.started:
            return
        from rclpy.executors import SingleThreadedExecutor

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.buffer.node)
        self.executor_thread = threading.Thread(
            target=self._spin_executor,
            name="nero-ros-observation-executor",
            daemon=True,
        )
        self.executor_thread.start()
        self.started = True

    def _spin_executor(self) -> None:
        from rclpy.executors import ExternalShutdownException

        try:
            self.executor.spin()
        except ExternalShutdownException:
            pass

    def stop(self) -> None:
        self.started = False
        if self.scheduler is not None:
            self.scheduler.close()
            self.scheduler = None
        if self.executor is not None:
            self.executor.shutdown()
            self.executor = None
        if self.executor_thread is not None:
            self.executor_thread.join(timeout=2.0)
            self.executor_thread = None
        self.buffer.destroy()
        if self.owns_rclpy_context and self.rclpy.ok():
            self.rclpy.shutdown()
        self.started = False

    def spin_once(self, timeout_sec: float = 0.02) -> None:
        # Callbacks run in the dedicated ROS executor thread. This method keeps
        # the inference loop's timing structure without blocking observation IO.
        time.sleep(max(0.0, timeout_sec))

    def get_obs(self, count_attempt: bool = True) -> Dict[str, np.ndarray]:
        if not self.started:
            raise RuntimeError("environment is not started")
        return self.buffer.get_obs(count_attempt=count_attempt)

    def verify_live_control_facts(
        self,
        timeout_sec: float,
    ) -> Dict[str, Any]:
        """Verify observable facts and only then create control publishers."""
        if not self.control:
            raise RuntimeError(
                "live control verification is only valid in --control mode")
        obs = self.wait_for_obs(timeout_sec)
        del obs
        tcp = self.buffer.latest_tcp
        gripper = self.buffer.latest_gripper
        if tcp is None:
            raise RuntimeError("cannot verify TCP frame without TCP feedback")
        if gripper is None:
            raise RuntimeError("cannot verify gripper range without feedback")
        expected_frame = str(
            self.config["control_output"]["expected_pose_frame"])
        if tcp.frame_id != expected_frame:
            raise RuntimeError(
                f"TCP frame mismatch: observed={tcp.frame_id!r}, "
                f"expected={expected_frame!r}")
        gripper_min = float(self.config["safety"]["gripper_min_m"])
        gripper_max = float(self.config["safety"]["gripper_max_m"])
        if not gripper_min <= gripper.value <= gripper_max:
            raise RuntimeError(
                f"gripper feedback {gripper.value:.6f} is outside configured "
                f"range [{gripper_min:.6f}, {gripper_max:.6f}]")
        if self.scheduler is None:
            self.scheduler = RosActionScheduler(
                self.buffer.node,
                self.config,
                str(self.config["observation"]["gripper_joint_name"]),
            )
        return {
            "tcp_frame": tcp.frame_id,
            "gripper_value": gripper.value,
            "control_publishers_created": True,
        }

    def wait_for_obs(
        self,
        timeout_sec: float,
        poll_period_sec: float = 0.02,
    ) -> Dict[str, np.ndarray]:
        deadline = time.monotonic() + timeout_sec
        last_reason = "waiting for observations"
        while time.monotonic() < deadline:
            time.sleep(poll_period_sec)
            try:
                return self.buffer.get_obs(count_attempt=False)
            except ObservationUnavailable as exc:
                last_reason = str(exc)
        raise TimeoutError(
            f"timed out waiting for model-compatible observations: {last_reason}")

    def exec_actions(
        self,
        actions: np.ndarray,
        timestamps: np.ndarray,
        now: Optional[float] = None,
        observation_timestamp: Optional[float] = None,
    ) -> ScheduleReport:
        if not self.control:
            raise RuntimeError("exec_actions is disabled outside --control")
        if self.scheduler is None:
            raise RuntimeError(
                "exec_actions requires verify_live_control_facts() first")
        try:
            report = self.scheduler.replace_plan(
                actions,
                timestamps,
                now=now,
                observation_timestamp=observation_timestamp,
            )
        except PlanRejected as exc:
            self.last_plan_error = str(exc)
            raise
        self.last_plan_error = None
        self.last_plan_report = report
        return report

    def prime_control_from_feedback(self) -> Dict[str, Any]:
        """Activate QP at the measured state before submitting model actions."""
        if not self.control:
            raise RuntimeError("controller priming is disabled outside --control")
        if self.scheduler is None:
            raise RuntimeError(
                "controller priming requires verify_live_control_facts() first")
        tcp = self.buffer.latest_tcp
        gripper = self.buffer.latest_gripper
        if tcp is None:
            raise RuntimeError("cannot prime controller without TCP feedback")
        if gripper is None:
            raise RuntimeError(
                "cannot prime controller without gripper feedback")

        now_sec = self.buffer.node.get_clock().now().nanoseconds * 1e-9
        max_age = float(
            self.config["observation"]["max_stream_age_ms"]) / 1000.0
        tcp_age = now_sec - tcp.timestamp
        gripper_age = now_sec - gripper.timestamp
        if tcp_age < -0.05 or tcp_age > max_age:
            raise RuntimeError(
                f"TCP feedback age {tcp_age * 1000.0:.1f}ms is invalid "
                "for controller priming")
        if gripper_age < -0.05 or gripper_age > max_age:
            raise RuntimeError(
                f"gripper feedback age {gripper_age * 1000.0:.1f}ms is "
                "invalid for controller priming")
        expected_frame = str(
            self.config["control_output"]["expected_pose_frame"])
        if tcp.frame_id != expected_frame:
            raise RuntimeError(
                f"TCP frame mismatch during priming: observed="
                f"{tcp.frame_id!r}, expected={expected_frame!r}")

        action_config = self.config["action"]
        published = self.scheduler.prime_current_pose(
            xyz=tcp.xyz,
            quat_xyzw=tcp.quat_xyzw,
            gripper=gripper.value,
            hold_sec=float(action_config["controller_prime_hold_sec"]),
            rate_hz=float(action_config["controller_prime_rate_hz"]),
        )
        self.last_plan_error = None
        return {
            "tcp_frame": tcp.frame_id,
            "tcp_xyz": tcp.xyz.tolist(),
            "gripper": gripper.value,
            "published_messages": published,
            "hold_sec": float(
                action_config["controller_prime_hold_sec"]),
        }

    @property
    def control_is_active(self) -> bool:
        return self.scheduler is not None and self.scheduler.was_active

    @property
    def latest_tcp_frame(self) -> Optional[str]:
        item = self.buffer.latest_tcp
        return item.frame_id if item is not None else None

    @property
    def latest_gripper_value(self) -> Optional[float]:
        item = self.buffer.latest_gripper
        return item.value if item is not None else None

    @property
    def observation_rejection_counts(self) -> Dict[str, int]:
        return dict(self.buffer.assembler.rejection_counts)

    def diagnostics(self) -> Dict[str, Any]:
        return self.buffer.diagnostics()
