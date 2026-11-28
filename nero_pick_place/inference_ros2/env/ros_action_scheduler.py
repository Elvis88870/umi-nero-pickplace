"""Absolute-time action scheduling for NERO UMI inference."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .grasp_verifier import GraspVerifier, closing_target_index
from .grasp_lift_guard import GraspLiftGuard


class PlanRejected(RuntimeError):
    """Raised when a plan is unsafe, malformed, or entirely expired."""


@dataclass(frozen=True)
class ScheduleReport:
    input_actions: int
    accepted_actions: int
    dropped_expired: int
    dropped_minimum_lead: int
    replaced_future_actions: int
    scheduler_now: Optional[float] = None
    first_execution_timestamp: Optional[float] = None
    last_execution_timestamp: Optional[float] = None
    translation_limited_actions: int = 0
    input_max_translation_step_m: float = 0.0
    output_max_translation_step_m: float = 0.0
    rotation_limited_actions: int = 0
    input_max_rotation_step_rad: float = 0.0
    output_max_rotation_step_rad: float = 0.0


@dataclass(frozen=True)
class ScheduledAction:
    raw_timestamp: float
    pose_timestamp: float
    gripper_timestamp: float
    xyz: np.ndarray
    quat_xyzw: np.ndarray
    gripper: float


@dataclass(frozen=True)
class SampledAction:
    timestamp: float
    pose_xyz_rotvec: np.ndarray
    gripper: float


def _bracket(
        timestamps: np.ndarray, target: float) -> Tuple[int, int]:
    if timestamps.size == 0:
        raise PlanRejected("empty action plan")
    upper = int(np.searchsorted(timestamps, target, side="left"))
    if upper < timestamps.size and abs(timestamps[upper] - target) <= 1e-9:
        return upper, upper
    if upper == 0:
        return 0, 0
    if upper >= timestamps.size:
        return timestamps.size - 1, timestamps.size - 1
    return upper - 1, upper


def find_grasp_transition(
    actions: np.ndarray,
    initial_gripper: Optional[float],
    close_threshold: float,
) -> Optional[int]:
    """Return the first open-to-closed gripper transition index."""
    previous = initial_gripper
    for index, width in enumerate(actions[:, 6]):
        width = float(width)
        if (
            previous is not None
            and previous >= close_threshold
            and width < close_threshold
        ):
            return index
        previous = width
    return None


def grasp_pose_at_width(
    actions: np.ndarray,
    start: int,
    close_index: int,
    pose_width: float,
) -> Tuple[np.ndarray, bool]:
    """Interpolate the first pose where a closing segment reaches a width."""
    for index in range(start, close_index + 1):
        width = float(actions[index, 6])
        if width > pose_width:
            continue
        if index == 0:
            return actions[index, :6].copy(), True
        previous_width = float(actions[index - 1, 6])
        if previous_width <= pose_width or abs(previous_width - width) < 1e-9:
            return actions[index, :6].copy(), True
        alpha = float(np.clip(
            (previous_width - pose_width) / (previous_width - width),
            0.0,
            1.0,
        ))
        pose = np.empty(6, dtype=np.float64)
        pose[:3] = (
            actions[index - 1, :3]
            + alpha * (actions[index, :3] - actions[index - 1, :3])
        )
        rotations = Rotation.from_rotvec(actions[index - 1:index + 1, 3:6])
        pose[3:6] = Slerp([0.0, 1.0], rotations)([alpha]).as_rotvec()[0]
        return pose, True
    return actions[close_index, :6].copy(), False


def limit_translation_steps(actions: np.ndarray, max_step: float) -> np.ndarray:
    """Slow progress along the source path, keeping pose/gripper in phase.

    Timestamps and output count stay unchanged. Each output advances at most
    max_step metres along the polyline, never past its source time index.
    The tail may be deferred to the next observation-relative replan.
    """
    output = actions.copy()
    segment = 0
    fraction = 0.0
    budget_per_step = max_step * (1.0 - 1e-10)
    for index in range(1, len(actions)):
        budget = budget_per_step
        while segment < index:
            length = float(np.linalg.norm(
                actions[segment + 1, :3] - actions[segment, :3]))
            remaining = length * (1.0 - fraction)
            if remaining <= budget:
                budget -= remaining
                segment += 1
                fraction = 0.0
            else:
                fraction += budget / length
                break
        if fraction == 0.0:
            output[index] = actions[segment]
        else:
            lower, upper = actions[segment], actions[segment + 1]
            output[index] = (1.0 - fraction) * lower + fraction * upper
            output[index, 3:6] = Slerp(
                [0.0, 1.0], Rotation.from_rotvec(
                    np.stack([lower[3:6], upper[3:6]]))
            )([fraction]).as_rotvec()[0]
    return output


def limit_rotation_steps(actions: np.ndarray, max_step: float) -> np.ndarray:
    """Slow modest angular overshoots, keeping XYZ and jaw at the same phase."""
    output = actions.copy()
    rotations = Rotation.from_rotvec(actions[:, 3:6])
    lengths = (rotations[:-1].inv() * rotations[1:]).magnitude()
    segment, fraction = 0, 0.0
    for index in range(1, len(actions)):
        budget = max_step * (1.0 - 1e-10)
        while segment < index:
            remaining = float(lengths[segment]) * (1.0 - fraction)
            if remaining <= budget:
                budget -= remaining
                segment += 1
                fraction = 0.0
            else:
                fraction += budget / float(lengths[segment])
                break
        if fraction == 0.0:
            output[index] = actions[segment]
        else:
            output[index] = ((1.0 - fraction) * actions[segment]
                             + fraction * actions[segment + 1])
            output[index, 3:6] = Slerp(
                [0.0, 1.0], rotations[segment:segment + 2]
            )([fraction]).as_rotvec()[0]
    return output


class ActionScheduler:
    """Pure scheduler used by the ROS timer and unit tests."""

    def __init__(
        self,
        action_config: Dict[str, Any],
        safety_config: Dict[str, Any],
    ) -> None:
        self.action_config = dict(action_config)
        self.safety_config = dict(safety_config)
        compensation = action_config["compensation"]
        common_compensation = sum(float(compensation[key]) for key in (
            "camera_timestamp_sec",
            "inference_latency_sec",
            "ros_transport_latency_sec",
            "zmq_transport_latency_sec",
        ))
        self.pose_compensation = common_compensation + float(
            compensation["qp_latency_sec"])
        self.gripper_compensation = common_compensation + float(
            compensation["gripper_latency_sec"])
        self.minimum_lead = float(action_config["minimum_command_lead_sec"])
        self.max_actions_per_plan = int(
            action_config.get("max_actions_per_plan", 0))
        limiter = action_config.get("translation_step_limiter", {})
        self.translation_step_check_enabled = bool(
            safety_config.get("translation_step_check_enabled", True))
        self.translation_limiter_enabled = (
            self.translation_step_check_enabled
            and bool(limiter.get("enabled", False)))
        self.translation_limiter_max_ratio = float(
            limiter.get("max_input_step_ratio", 1.5))
        rotation_limiter = action_config.get("rotation_step_limiter", {})
        self.rotation_step_check_enabled = bool(
            safety_config.get("rotation_step_check_enabled", True))
        self.rotation_limiter_enabled = (
            self.rotation_step_check_enabled
            and bool(rotation_limiter.get("enabled", False)))
        self.rotation_limiter_max_ratio = float(
            rotation_limiter.get("max_input_step_ratio", 1.5))
        if (not math.isfinite(self.rotation_limiter_max_ratio)
                or self.rotation_limiter_max_ratio < 1.0):
            raise ValueError("rotation limiter max_input_step_ratio must be >= 1")
        if (
            not math.isfinite(self.translation_limiter_max_ratio)
            or self.translation_limiter_max_ratio < 1.0
        ):
            raise ValueError("translation limiter max_input_step_ratio must be >= 1")
        if self.max_actions_per_plan < 0:
            raise ValueError("max_actions_per_plan must be nonnegative")
        self.plan: List[ScheduledAction] = []
        self.last_report: Optional[ScheduleReport] = None
        self.last_violation: Optional[str] = None

    def clear(self) -> None:
        self.plan.clear()

    @property
    def has_plan(self) -> bool:
        return bool(self.plan)

    def _validate_action_batch(
        self, actions: np.ndarray, timestamps: np.ndarray) -> None:
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise PlanRejected(
                f"actions must have shape (T, 7), got {actions.shape}")
        if timestamps.shape != (actions.shape[0],):
            raise PlanRejected("one absolute timestamp per action is required")
        if actions.shape[0] == 0:
            raise PlanRejected("action plan is empty")
        if not np.all(np.isfinite(actions)):
            raise PlanRejected("action plan contains non-finite values")
        if not np.all(np.isfinite(timestamps)):
            raise PlanRejected("action timestamps contain non-finite values")
        if np.any(np.diff(timestamps) <= 0.0):
            raise PlanRejected("action timestamps are not strictly increasing")

        expected_interval = float(self.action_config["action_interval_sec"])
        tolerance = float(
            self.safety_config["required_action_interval_tolerance_sec"])
        if np.any(np.abs(np.diff(timestamps) - expected_interval) > tolerance):
            raise PlanRejected(
                "action interval does not match configured action_interval_sec")

    def _validate_action_geometry(
        self, actions: np.ndarray, translation_limit: Optional[float] = None,
        rotation_limit: Optional[float] = None,
    ) -> None:
        if bool(self.safety_config.get("workspace_enabled", True)):
            lower = np.asarray(
                self.safety_config["workspace_min_xyz_m"], dtype=np.float64)
            upper = np.asarray(
                self.safety_config["workspace_max_xyz_m"], dtype=np.float64)
            if lower.shape != (3,) or upper.shape != (3,):
                raise PlanRejected(
                    "workspace bounds are not configured as xyz")
            if np.any(lower >= upper):
                raise PlanRejected(
                    "workspace minimum must be below maximum")
            outside = np.any(
                (actions[:, :3] < lower)
                | (actions[:, :3] > upper),
                axis=1,
            )
            if np.any(outside):
                index = int(np.flatnonzero(outside)[0])
                xyz = np.array2string(
                    actions[index, :3], precision=4, suppress_small=True)
                raise PlanRejected(
                    f"action {index} xyz={xyz} exceeds configured Cartesian "
                    f"workspace {lower.tolist()}..{upper.tolist()}")

        if self.translation_step_check_enabled:
            max_step = float(self.safety_config["max_translation_step_m"])
            if translation_limit is not None:
                max_step = translation_limit
            if max_step <= 0.0:
                raise PlanRejected("max_translation_step_m is not configured")
            steps = np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=1)
            if steps.size and float(steps.max()) > max_step:
                raise PlanRejected(
                    f"translation step {steps.max():.4f}m exceeds {max_step:.4f}m")

        if self.rotation_step_check_enabled:
            max_rotation = float(self.safety_config["max_rotation_step_rad"])
            if rotation_limit is not None:
                max_rotation = rotation_limit
            if max_rotation <= 0.0:
                raise PlanRejected("max_rotation_step_rad is not configured")
            rotations = Rotation.from_rotvec(actions[:, 3:6])
            if len(rotations) > 1:
                rotation_steps = (
                    rotations[:-1].inv() * rotations[1:]).magnitude()
                if float(rotation_steps.max()) > max_rotation:
                    raise PlanRejected(
                        f"rotation step {rotation_steps.max():.4f}rad exceeds "
                        f"{max_rotation:.4f}rad")

        gripper_min = float(self.safety_config["gripper_min_m"])
        gripper_max = float(self.safety_config["gripper_max_m"])
        if not gripper_min < gripper_max:
            raise PlanRejected("gripper range is not configured")
        if np.any(actions[:, 6] < gripper_min) or np.any(
                actions[:, 6] > gripper_max):
            raise PlanRejected("action exceeds configured gripper range")

    def replace_plan(
        self,
        actions: np.ndarray,
        timestamps: np.ndarray,
        now: Optional[float] = None,
    ) -> ScheduleReport:
        # Action timestamps come from ROS header stamps (Unix epoch seconds).
        # The pure scheduler must use the same clock domain by default.
        now = time.time() if now is None else float(now)
        actions = np.asarray(actions, dtype=np.float64)
        timestamps = np.asarray(timestamps, dtype=np.float64)
        try:
            self._validate_action_batch(actions, timestamps)
        except PlanRejected as exc:
            self.reject_and_clear(str(exc))
            raise

        pose_times = timestamps - self.pose_compensation
        gripper_times = timestamps - self.gripper_compensation
        expired = np.minimum(pose_times, gripper_times) <= now
        too_soon = np.minimum(pose_times, gripper_times) < (
            now + self.minimum_lead)
        keep = ~too_soon
        kept_indices = np.flatnonzero(keep)
        if self.max_actions_per_plan > 0:
            kept_indices = kept_indices[:self.max_actions_per_plan]
        if kept_indices.size == 0:
            self.clear()
            raise PlanRejected("all actions are expired or below minimum lead")
        selected_actions = actions[kept_indices]
        input_steps = np.linalg.norm(
            np.diff(selected_actions[:, :3], axis=0), axis=1)
        input_max_step = float(input_steps.max()) if input_steps.size else 0.0
        limited_count = 0
        rotation_limited_count = 0
        input_rotations = Rotation.from_rotvec(selected_actions[:, 3:6])
        input_rotation_steps = ((
            input_rotations[:-1].inv() * input_rotations[1:]).magnitude()
            if len(input_rotations) > 1 else np.asarray([]))
        input_max_rotation = (float(input_rotation_steps.max())
                              if len(input_rotation_steps) else 0.0)
        try:
            max_rotation = (float(self.safety_config["max_rotation_step_rad"])
                            if self.rotation_step_check_enabled else None)
            rotation_input_limit = (
                max_rotation * self.rotation_limiter_max_ratio
                if self.rotation_limiter_enabled else None)
            if self.translation_limiter_enabled:
                max_step = float(self.safety_config["max_translation_step_m"])
                # Validate original geometry before modifying it. Large jumps,
                # workspace, rotation and gripper violations still clear/stop.
                self._validate_action_geometry(
                    selected_actions,
                    translation_limit=max_step * self.translation_limiter_max_ratio,
                    rotation_limit=rotation_input_limit,
                )
                if input_max_step > max_step:
                    original_actions = selected_actions
                    selected_actions = limit_translation_steps(
                        selected_actions, max_step)
                    limited_count = int(np.count_nonzero(np.any(
                        np.abs(selected_actions - original_actions) > 1e-9,
                        axis=1)))
            if self.rotation_limiter_enabled:
                # Check raw predictions first: interpolation must never hide
                # a large angular jump, bad gripper width, or workspace error.
                self._validate_action_geometry(
                    selected_actions, rotation_limit=rotation_input_limit)
                if input_max_rotation > max_rotation:
                    original_actions = selected_actions
                    selected_actions = limit_rotation_steps(selected_actions, max_rotation)
                    rotation_limited_count = int(np.count_nonzero(np.any(
                        np.abs(selected_actions - original_actions) > 1e-9, axis=1)))
            self._validate_action_geometry(selected_actions)
        except PlanRejected as exc:
            self.reject_and_clear(str(exc))
            raise

        replaced_future = sum(
            min(item.pose_timestamp, item.gripper_timestamp) > now
            for item in self.plan)
        self.plan = []
        rotation = Rotation.from_rotvec(selected_actions[:, 3:6])
        for selected_index, index in enumerate(kept_indices):
            self.plan.append(ScheduledAction(
                raw_timestamp=float(timestamps[index]),
                pose_timestamp=float(pose_times[index]),
                gripper_timestamp=float(gripper_times[index]),
                xyz=selected_actions[selected_index, :3].copy(),
                quat_xyzw=rotation[selected_index].as_quat().copy(),
                gripper=float(selected_actions[selected_index, 6]),
            ))
        self.last_report = ScheduleReport(
            input_actions=int(actions.shape[0]),
            accepted_actions=len(self.plan),
            dropped_expired=int(np.count_nonzero(expired)),
            dropped_minimum_lead=int(np.count_nonzero(too_soon & ~expired)),
            replaced_future_actions=int(replaced_future),
            scheduler_now=float(now),
            first_execution_timestamp=min(
                float(pose_times[kept_indices[0]]),
                float(gripper_times[kept_indices[0]]),
            ),
            last_execution_timestamp=max(
                float(pose_times[kept_indices[-1]]),
                float(gripper_times[kept_indices[-1]]),
            ),
            translation_limited_actions=limited_count,
            input_max_translation_step_m=input_max_step,
            output_max_translation_step_m=(
                float(np.linalg.norm(np.diff(
                    selected_actions[:, :3], axis=0), axis=1).max())
                if len(selected_actions) > 1 else 0.0),
            rotation_limited_actions=rotation_limited_count,
            input_max_rotation_step_rad=input_max_rotation,
            output_max_rotation_step_rad=(float((
                rotation[:-1].inv() * rotation[1:]).magnitude().max())
                if len(rotation) > 1 else 0.0),
        )
        self.last_violation = None
        return self.last_report

    def sample(self, now: float) -> Optional[SampledAction]:
        if not self.plan:
            return None
        now = float(now)
        pose_times = np.asarray(
            [item.pose_timestamp for item in self.plan], dtype=np.float64)
        gripper_times = np.asarray(
            [item.gripper_timestamp for item in self.plan], dtype=np.float64)
        if now < min(float(pose_times[0]), float(gripper_times[0])):
            return None
        plo, phi = _bracket(pose_times, now)
        if plo == phi:
            xyz = self.plan[plo].xyz.copy()
            quat = self.plan[plo].quat_xyzw.copy()
        else:
            alpha = (
                (now - pose_times[plo])
                / (pose_times[phi] - pose_times[plo]))
            xyz = (
                (1.0 - alpha) * self.plan[plo].xyz
                + alpha * self.plan[phi].xyz)
            quat = Slerp(
                [pose_times[plo], pose_times[phi]],
                Rotation.from_quat([
                    self.plan[plo].quat_xyzw,
                    self.plan[phi].quat_xyzw,
                ]),
            )([now]).as_quat()[0]

        glo, ghi = _bracket(gripper_times, now)
        if glo == ghi:
            gripper = self.plan[glo].gripper
        else:
            alpha = (
                (now - gripper_times[glo])
                / (gripper_times[ghi] - gripper_times[glo]))
            gripper = (
                (1.0 - alpha) * self.plan[glo].gripper
                + alpha * self.plan[ghi].gripper)

        pose = np.concatenate([xyz, Rotation.from_quat(quat).as_rotvec()])
        return SampledAction(now, pose, float(gripper))

    def reject_and_clear(self, reason: str) -> None:
        self.last_violation = reason
        self.clear()


class RosActionScheduler:
    """ROS timer wrapper that publishes only when explicitly in control mode."""

    def __init__(
        self,
        node: Any,
        config: Dict[str, Any],
        gripper_joint_name: str,
    ) -> None:
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import (
            DurabilityPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool

        self.node = node
        self.config = config
        self._plan_lock = threading.RLock()
        self.gripper_joint_name = gripper_joint_name
        self.core = ActionScheduler(
            config["action"], config["safety"])
        self.pose_topic = str(config["control_output"]["pose_topic"])
        self.gripper_topic = str(
            config["control_output"]["gripper_control_topic"])
        self.active_topic = str(
            config["control_output"]["teleop_active_topic"])
        self.pose_frame = str(
            config["control_output"]["expected_pose_frame"])
        stable_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        command_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pose_pub = node.create_publisher(
            PoseStamped, self.pose_topic, command_qos)
        self.gripper_pub = node.create_publisher(
            JointState, self.gripper_topic, command_qos)
        self.active_pub = node.create_publisher(
            Bool, self.active_topic, stable_qos)
        self.was_active = False
        self._priming = False
        self._last_published_action = None
        self._rejected_plan_hold = None
        self._actual_tcp_xyz: Optional[np.ndarray] = None
        self._tcp_feedback_received = None
        self._tcp_feedback_stamp = None
        self._actual_gripper: Optional[float] = None
        grasp_sync = dict(config["action"].get("grasp_sync", {}))
        self._grasp_sync_enabled = bool(grasp_sync.get("enabled", False))
        self._grasp_model_driven_pose = bool(
            grasp_sync.get("model_driven_pose", False))
        self._lift_guard = GraspLiftGuard(grasp_sync.get("lift_guard", {}))
        self._lift_guard_last_reason = None
        self._gripper_extra_close = 0.0
        self._gripper_last_tick = None
        self._gripper_previous_model_width = None
        self._gripper_model_widening = False
        self._gripper_close_started = None
        self._gripper_contact_reported = False
        self._grasp_close_threshold = float(
            grasp_sync.get("close_threshold_m", 0.080))
        self._grasp_pose_width = float(
            grasp_sync.get("pose_width_m", 0.070))
        if not (
            float(config["safety"]["gripper_min_m"])
            < self._grasp_pose_width
            < self._grasp_close_threshold
        ):
            raise ValueError("grasp pose width must be inside the closing range")
        self._grasp_max_extra_close = float(
            grasp_sync.get("max_extra_close_m", 0.0))
        self._grasp_tighten_floor = float(
            grasp_sync.get("tighten_floor_m", config["safety"]["gripper_min_m"]))
        self._grasp_tighten_rate = float(
            grasp_sync.get("tighten_rate_mps", 0.005))
        self._grasp_target_width_tolerance = float(
            grasp_sync.get("target_width_tolerance_m", 0.001))
        self._grasp_always_tighten = bool(
            grasp_sync.get("always_tighten_before_verification", False))
        self._grasp_require_fresh_plan = bool(
            grasp_sync.get("require_fresh_post_grasp_plan", True))
        if self._grasp_max_extra_close < 0 or min(
            self._grasp_tighten_rate, self._grasp_target_width_tolerance) <= 0:
            raise ValueError("grasp tightening limits must be positive")
        if not (
            float(config["safety"]["gripper_min_m"])
            <= self._grasp_tighten_floor
            < min(self._grasp_close_threshold,
                  float(config["safety"]["gripper_max_m"]))
        ):
            raise ValueError("grasp tightening floor is outside closing range")
        self._grasp_position_tolerance = float(
            grasp_sync.get("position_tolerance_m", 0.012))
        self._grasp_settle_sec = float(
            grasp_sync.get("settle_sec", 0.6))
        self._grasp_close_rate = float(grasp_sync.get("close_rate_mps", 0.02))
        self._grasp_close_timeout = float(grasp_sync.get("close_timeout_sec", 5.0))
        if min(self._grasp_close_rate, self._grasp_close_timeout) <= 0:
            raise ValueError("grasp close rate and timeout must be positive")
        self._grasp_verifier = GraspVerifier(grasp_sync.get("verification", {}))
        self._grasp_failed = False
        self._grasp_hold_width: Optional[float] = None
        self._grasp_reason = "not started"
        self._grasp_target_pose: Optional[np.ndarray] = None
        self._grasp_pose_width_reached = False
        self._grasp_open_width: Optional[float] = None
        self._grasp_close_width: Optional[float] = None
        self._grasp_settle_started: Optional[float] = None
        self._grasp_tighten_started: Optional[float] = None
        self._grasp_complete = False
        self._grasp_awaiting_fresh_plan = False
        self._grasp_contact_epoch: Optional[float] = None
        rate = float(config["action"]["publish_rate_hz"])
        if rate <= 0.0:
            raise ValueError("publish_rate_hz must be positive")
        self.timer = node.create_timer(1.0 / rate, self._timer_callback)
        if self._grasp_sync_enabled:
            from data_msgs.msg import Gripper

            observation = config["observation"]
            node.create_subscription(
                PoseStamped,
                str(observation["tcp_pose_topic"]),
                self._tcp_feedback_callback,
                command_qos,
            )
            node.create_subscription(
                JointState,
                str(observation["gripper_state_topic"]),
                self._gripper_feedback_callback,
                command_qos,
            )
            node.create_subscription(
                Gripper,
                str(observation["gripper_telemetry_topic"]),
                self._gripper_telemetry_callback,
                command_qos,
            )

    def _tcp_feedback_callback(self, msg: Any) -> None:
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if (not math.isfinite(stamp) or stamp <= 0
                or (self._tcp_feedback_stamp is not None
                    and stamp <= self._tcp_feedback_stamp)):
            return
        self._actual_tcp_xyz = np.asarray([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=np.float64)
        self._tcp_feedback_received = time.monotonic()
        self._tcp_feedback_stamp = stamp

    def _gripper_feedback_callback(self, msg: Any) -> None:
        if msg.position:
            self._actual_gripper = float(msg.position[0])

    def _gripper_telemetry_callback(self, msg: Any) -> None:
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        with self._plan_lock:
            self._grasp_verifier.update(
                time.monotonic(), stamp, float(msg.distance), float(msg.effort),
                bool(msg.enable) and not msg.error)

    def _clock_now_sec(self) -> float:
        """Return the ROS clock in the same epoch domain as message stamps."""
        now_sec = self.node.get_clock().now().nanoseconds * 1e-9
        if not math.isfinite(now_sec) or now_sec <= 0.0:
            raise PlanRejected("ROS clock is not initialized")
        return float(now_sec)

    def _set_inactive(self) -> None:
        from std_msgs.msg import Bool

        self._rejected_plan_hold = None
        if self.was_active:
            self.active_pub.publish(Bool(data=False))
            self.was_active = False

    def replace_plan(
        self,
        actions: np.ndarray,
        timestamps: np.ndarray,
        now: Optional[float] = None,
        observation_timestamp: Optional[float] = None,
    ) -> ScheduleReport:
        with self._plan_lock:
            return self._replace_plan(
                actions,
                timestamps,
                now,
                observation_timestamp,
            )

    def _replace_plan(
        self,
        actions: np.ndarray,
        timestamps: np.ndarray,
        now: Optional[float] = None,
        observation_timestamp: Optional[float] = None,
    ) -> ScheduleReport:
        previous_plan = list(self.core.plan)
        preserve_grasp_hold = (
            self._grasp_sync_enabled
            and self._grasp_target_pose is not None
            and not self._grasp_failed
            and (
                not self._grasp_complete
                or self._grasp_awaiting_fresh_plan
            )
            and bool(previous_plan)
        )
        try:
            effective_now = (
                self._clock_now_sec() if now is None else float(now))
            report = self.core.replace_plan(
                actions, timestamps, now=effective_now)
            self._priming = False
            self._rejected_plan_hold = None
        except Exception as exc:
            self._priming = False
            if (
                isinstance(exc, PlanRejected)
                and self._grasp_model_driven_pose
                and self.was_active
                and self._last_published_action is not None
            ):
                # Keep QP initialized while rejecting this unsafe replacement.
                # Hold the last actually published target, never the old
                # future trajectory or any part of the rejected prediction.
                self._rejected_plan_hold = self._last_published_action
                self.node.get_logger().warning(
                    "[schedule] rejected replacement; holding last published "
                    "TCP/jaw target with control active")
            elif preserve_grasp_hold:
                # The new plan is unsafe, but the last accepted plan is still
                # safe and grasp_sync overrides it to a stationary grasp hold.
                # Preserve that hold so rejection cannot loosen the jaw or
                # hand control back to the GUI between inference cycles.
                self.core.plan = previous_plan
                self.node.get_logger().warning(
                    "[grasp_sync] rejected replacement; preserving the last "
                    "accepted grasp hold")
            else:
                self._set_inactive()
            raise

        # Latch only accepted future waypoints, including any path slowing.
        if self._grasp_model_driven_pose:
            if self._lift_guard.accept_observation(observation_timestamp):
                self.node.get_logger().info("[lift_guard] " + self._lift_guard.reason)
            elif self._lift_guard.enabled and self._lift_guard.holding:
                # The accepted model forecast may correct an unconfirmed grasp.
                # Use only future, validated points, never a stale observation.
                approach = np.asarray([
                    np.r_[item.xyz, Rotation.from_quat(item.quat_xyzw).as_rotvec(),
                          item.gripper] for item in self.core.plan])
                if self._lift_guard.update_approach(
                    approach, observation_timestamp, self._grasp_close_threshold,
                    time.monotonic()):
                    self.node.get_logger().info(
                        "[lift_guard] visual correction: "
                        f"xyz={self._lift_guard.low_pose[:3].round(5).tolist()} "
                        f"width={self._lift_guard.base_width:.5f}")
            # A jaw-opening threshold must not select or freeze a TCP pose.
            # Pose and the base opening keep their original model timestamps.
            return report
        actions_array = np.asarray([
            np.concatenate([
                item.xyz,
                Rotation.from_quat(item.quat_xyzw).as_rotvec(),
                [item.gripper],
            ]) for item in self.core.plan
        ])
        if self._grasp_awaiting_fresh_plan:
            obs_stamp = (
                None if observation_timestamp is None
                else float(observation_timestamp)
            )
            if (
                obs_stamp is not None
                and math.isfinite(obs_stamp)
                and self._grasp_contact_epoch is not None
                and obs_stamp > self._grasp_contact_epoch
            ):
                self._grasp_awaiting_fresh_plan = False
                reference_z = (
                    float(self._actual_tcp_xyz[2])
                    if self._actual_tcp_xyz is not None
                    else float(self._grasp_target_pose[2])
                )
                predicted_lift = float(
                    np.max(actions_array[:, 2]) - reference_z)
                self.node.get_logger().info(
                    "[grasp_sync] accepted fresh post-grasp plan: "
                    f"obs={obs_stamp:.6f} predicted_max_lift="
                    f"{predicted_lift:.4f}m; returning control to model")
        if (
            self._grasp_sync_enabled
            and not self._grasp_complete
            and self._grasp_target_pose is None
        ):
            grasp_index = find_grasp_transition(
                actions_array,
                self._actual_gripper,
                self._grasp_close_threshold,
            )
            if grasp_index is not None:
                previous_width = (
                    float(actions_array[grasp_index - 1, 6])
                    if grasp_index > 0
                    else self._actual_gripper
                )
                if previous_width is None:
                    previous_width = self._grasp_close_threshold
                self._grasp_open_width = max(
                    float(previous_width), self._grasp_close_threshold)
                close_index = closing_target_index(
                    actions_array, grasp_index, self._grasp_close_threshold)
                # The threshold crossing identifies closing too early, while
                # the minimum-width pose may already be lifting. Stop at an
                # interpolated intermediate closing width instead.
                (self._grasp_target_pose,
                 self._grasp_pose_width_reached) = grasp_pose_at_width(
                    actions_array,
                    grasp_index,
                    close_index,
                    self._grasp_pose_width,
                )
                self._grasp_close_width = float(actions_array[close_index, 6])
                self._grasp_settle_started = None
                self._grasp_tighten_started = None
                self.node.get_logger().info(
                    "[grasp_sync] locked grasp target "
                    f"xyz={self._grasp_target_pose[:3].round(5).tolist()} "
                    f"pose_width={self._grasp_pose_width:.5f} "
                    f"open={self._grasp_open_width:.5f} "
                    f"close={self._grasp_close_width:.5f}")
        elif (
            self._grasp_sync_enabled and not self._grasp_complete
            and not self._grasp_failed and self._grasp_target_pose is not None
            and self._grasp_settle_started is None
        ):
            # Replans may expose the rest of the same closing segment. Never
            # tighten beyond the model's predicted opening or after contact.
            index = next((i for i, a in enumerate(actions_array)
                          if a[6] < self._grasp_close_threshold), None)
            if index is not None:
                close_index = closing_target_index(
                    actions_array, index, self._grasp_close_threshold)
                candidate_width = float(actions_array[close_index, 6])
                if candidate_width < self._grasp_close_width:
                    # A short earlier horizon may not yet have exposed the
                    # configured intermediate close pose. Update it once, but
                    # never drift it toward the later minimum-width/lift pose.
                    if not self._grasp_pose_width_reached:
                        (self._grasp_target_pose,
                         self._grasp_pose_width_reached) = grasp_pose_at_width(
                            actions_array,
                            index,
                            close_index,
                            self._grasp_pose_width,
                        )
                    self._grasp_close_width = candidate_width
        return report

    def reject_and_clear(self, reason: str) -> None:
        self._priming = False
        self.core.reject_and_clear(reason)
        self._set_inactive()

    def prime_current_pose(
        self,
        xyz: np.ndarray,
        quat_xyzw: np.ndarray,
        gripper: float,
        hold_sec: float,
        rate_hz: float,
    ) -> int:
        """Activate QP at the measured pose before installing a model plan."""
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool

        xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
        quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
        quat_norm = float(np.linalg.norm(quat_xyzw))
        if not np.all(np.isfinite(xyz)):
            raise PlanRejected("current TCP position is non-finite")
        if not np.all(np.isfinite(quat_xyzw)) or quat_norm < 1e-9:
            raise PlanRejected("current TCP quaternion is invalid")
        if not math.isfinite(float(gripper)):
            raise PlanRejected("current gripper value is non-finite")
        hold_sec = float(hold_sec)
        rate_hz = float(rate_hz)
        if hold_sec <= 0.0 or rate_hz <= 0.0:
            raise PlanRejected("controller priming duration/rate is invalid")

        quat_xyzw = quat_xyzw / quat_norm
        period = 1.0 / rate_hz
        deadline = time.monotonic() + hold_sec
        self.core.clear()
        self._rejected_plan_hold = None
        self._priming = True
        published = 0
        try:
            while True:
                now_sec = self._clock_now_sec()
                stamp_msg = self.node.get_clock().now().to_msg()
                pose_msg = PoseStamped()
                pose_msg.header.stamp = stamp_msg
                pose_msg.header.frame_id = self.pose_frame
                pose_msg.pose.position.x = float(xyz[0])
                pose_msg.pose.position.y = float(xyz[1])
                pose_msg.pose.position.z = float(xyz[2])
                pose_msg.pose.orientation.x = float(quat_xyzw[0])
                pose_msg.pose.orientation.y = float(quat_xyzw[1])
                pose_msg.pose.orientation.z = float(quat_xyzw[2])
                pose_msg.pose.orientation.w = float(quat_xyzw[3])

                gripper_msg = JointState()
                gripper_msg.header.stamp = stamp_msg
                gripper_msg.name = [self.gripper_joint_name]
                gripper_msg.position = [float(gripper)]

                if not self.was_active:
                    self.active_pub.publish(Bool(data=True))
                    self.was_active = True
                self.pose_pub.publish(pose_msg)
                self.gripper_pub.publish(gripper_msg)
                self._last_published_action = SampledAction(
                    now_sec,
                    np.r_[xyz, Rotation.from_quat(quat_xyzw).as_rotvec()],
                    float(gripper))
                published += 1

                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                time.sleep(min(period, remaining))
        except Exception:
            self._priming = False
            self._set_inactive()
            raise
        return published

    def _timer_callback(self) -> None:
        with self._plan_lock:
            self._publish_action()

    def _compensate_model_gripper(self, model_width, now):
        """Apply bounded jaw-only tightening; never select a grasp TCP pose."""
        dt = (0.0 if self._gripper_last_tick is None else
              min(0.1, max(0.0, now - self._gripper_last_tick)))
        self._gripper_last_tick = now
        # The existing open/closed threshold is used only for jaw release.
        # It has no effect on the model's Cartesian trajectory or timing.
        if model_width >= self._grasp_close_threshold:
            self._gripper_extra_close = 0.0
            self._gripper_close_started = None
            self._gripper_contact_reported = False
            self._gripper_previous_model_width = model_width
            self._gripper_model_widening = False
            return model_width
        if self._gripper_close_started is None:
            self._gripper_close_started = now
        if self._gripper_previous_model_width is not None:
            delta = model_width - self._gripper_previous_model_width
            if delta > 1e-6:
                self._gripper_model_widening = True
            elif delta < -1e-6:
                self._gripper_model_widening = False
        self._gripper_previous_model_width = model_width
        if not self._gripper_model_widening:
            self._gripper_extra_close = min(
                self._grasp_max_extra_close,
                self._gripper_extra_close + self._grasp_tighten_rate * dt)
        command = min(model_width, max(
            self._grasp_tighten_floor,
            model_width - self._gripper_extra_close))
        return command

    def _publish_action(self) -> None:
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool

        if self._priming:
            return
        rejected_hold = self._rejected_plan_hold is not None
        sampled = (self._rejected_plan_hold if rejected_hold
                   else self.core.sample(self._clock_now_sec()))
        if sampled is None:
            # A valid future plan is allowed to wait for its first waypoint.
            # Once started, ActionScheduler.sample() clamps to the final
            # waypoint so a one-action test keeps publishing a stable hold
            # target until the plan is replaced or the session closes.
            if not self.core.has_plan:
                self._set_inactive()
            elif self.was_active:
                self.active_pub.publish(Bool(data=True))
            return
        pose_xyz_rotvec = sampled.pose_xyz_rotvec.copy()
        gripper_command = float(sampled.gripper)
        if (self._grasp_sync_enabled and self._grasp_model_driven_pose
                and not rejected_hold):
            now = time.monotonic()
            guard = self._lift_guard
            if guard.enabled:
                was_holding = guard.holding
                guard.observe_model(pose_xyz_rotvec, gripper_command,
                                    self._grasp_close_threshold, now)
                if guard.holding and not was_holding:
                    guard.hold_epoch = self._clock_now_sec()
                if guard.holding:
                    # Follow the latest eligible visual approach correction,
                    # while blocking the unconfirmed transport/lift forecast.
                    pose_xyz_rotvec = guard.low_pose.copy()
                    gripper_command = guard.base_width
            gripper_command = (guard.frozen_command
                if guard.enabled and guard.holding and guard.confirmed_epoch is not None
                else self._compensate_model_gripper(gripper_command, now))
            if guard.enabled and guard.holding:
                epoch_now = self._clock_now_sec()
                fresh = (self._tcp_feedback_received is not None
                         and self._tcp_feedback_stamp is not None
                         and now - self._tcp_feedback_received <= guard.max_age
                         and -0.05 <= epoch_now - self._tcp_feedback_stamp <= guard.max_age)
                final_width = min(guard.base_width, max(
                    self._grasp_tighten_floor,
                    guard.base_width - self._grasp_max_extra_close))
                guard.verify(
                    now, epoch_now, self._actual_tcp_xyz, fresh, gripper_command,
                    gripper_command <= final_width + 1e-6,
                    lambda since: self._grasp_verifier.check(
                        now, epoch_now, since, self._grasp_close_threshold,
                        command_width=gripper_command),
                    self._grasp_verifier.stable_sec)
                if guard.reason != self._lift_guard_last_reason:
                    self._lift_guard_last_reason = guard.reason
                    self.node.get_logger().info("[lift_guard] " + guard.reason)
            if (self._gripper_close_started is not None
                    and not self._gripper_contact_reported):
                contact, reason = self._grasp_verifier.check(
                    now, self._clock_now_sec(), self._gripper_close_started,
                    self._grasp_close_threshold, command_width=gripper_command)
                if contact:
                    self._gripper_contact_reported = True
                    self.node.get_logger().info(
                        "[gripper] contact evidence (diagnostic only): " + reason)
        if (
            self._grasp_sync_enabled and not self._grasp_model_driven_pose
            and self._grasp_complete
            and self._grasp_hold_width is not None
            and gripper_command < self._grasp_close_threshold
        ):
            # Do not loosen a verified grip during transport. A tighter model
            # prediction still passes through, and model reopening releases it.
            gripper_command = min(gripper_command, self._grasp_hold_width)
        if (
            self._grasp_sync_enabled
            and not self._grasp_model_driven_pose
            and not self._grasp_complete
            and self._grasp_target_pose is not None
            and (
                gripper_command < self._grasp_close_threshold
                or self._grasp_settle_started is not None
                or self._grasp_failed
            )
        ):
            # Do not close in mid-air while the QP is still catching up to the
            # model's grasp waypoint. Hold the gripper open at that waypoint,
            # then close and dwell before allowing the lift trajectory.
            pose_xyz_rotvec = self._grasp_target_pose.copy()
            if self._grasp_failed:
                gripper_command = float(self._grasp_hold_width)
            elif self._actual_tcp_xyz is None:
                gripper_command = float(self._grasp_open_width)
            else:
                position_error = float(np.linalg.norm(
                    self._actual_tcp_xyz - self._grasp_target_pose[:3]))
                if position_error > self._grasp_position_tolerance:
                    gripper_command = float(self._grasp_open_width)
                else:
                    now = time.monotonic()
                    if self._grasp_settle_started is None:
                        self._grasp_settle_started = now
                        self._grasp_tighten_started = None
                        self._grasp_hold_width = float(
                            self._actual_gripper if self._actual_gripper is not None
                            else self._grasp_open_width)
                        self.node.get_logger().info(
                            "[grasp_sync] pose reached; closing toward "
                            f"{self._grasp_close_width:.5f}m; awaiting contact evidence")
                    elapsed = now - self._grasp_settle_started
                    # First reach the model width. Only after stable, fresh
                    # no-contact feedback may a bounded correction begin.
                    model_command = max(
                        self._grasp_close_width,
                        self._grasp_hold_width - self._grasp_close_rate * elapsed)
                    gripper_command = model_command
                    correction_target = self._grasp_close_width
                    if self._grasp_close_width > self._grasp_tighten_floor:
                        correction_target = max(
                            self._grasp_tighten_floor,
                            self._grasp_close_width - self._grasp_max_extra_close)
                    if self._grasp_tighten_started is not None:
                        tighten_elapsed = now - self._grasp_tighten_started
                        gripper_command = max(
                            correction_target,
                            self._grasp_close_width
                            - self._grasp_tighten_rate * tighten_elapsed)
                    tightening_required = (
                        correction_target
                        < self._grasp_close_width - 1e-9)
                    verification_allowed = (
                        not self._grasp_always_tighten
                        or not tightening_required
                        or (
                            self._grasp_tighten_started is not None
                            and gripper_command
                            <= correction_target
                            + self._grasp_target_width_tolerance
                        )
                    )
                    if verification_allowed:
                        confirmed, reason = self._grasp_verifier.check(
                            now,
                            self._clock_now_sec(),
                            self._grasp_settle_started,
                            self._grasp_open_width,
                            command_width=gripper_command,
                        )
                    else:
                        confirmed = False
                        reason = "waiting for bounded tightening"
                    self._grasp_reason = reason
                    if (
                        not confirmed
                        and self._grasp_tighten_started is None
                        and correction_target < self._grasp_close_width
                    ):
                        ready, correction_reason = (
                            self._grasp_verifier.correction_ready(
                                now,
                                self._clock_now_sec(),
                                self._grasp_settle_started,
                                self._grasp_open_width,
                                self._grasp_close_width,
                                self._grasp_target_width_tolerance,
                                command_width=model_command,
                                ignore_contact=self._grasp_always_tighten,
                            ))
                        if ready:
                            self._grasp_tighten_started = now
                            self.node.get_logger().warning(
                                "[grasp_sync] no contact at model width; "
                                f"bounded tightening to {correction_target:.5f}m")
                        else:
                            self._grasp_reason = correction_reason
                    if confirmed and elapsed >= self._grasp_settle_sec:
                        self._grasp_complete = True
                        self._grasp_hold_width = float(gripper_command)
                        if self._grasp_require_fresh_plan:
                            self._grasp_awaiting_fresh_plan = True
                            self._grasp_contact_epoch = self._clock_now_sec()
                        self.node.get_logger().info(
                            "[grasp_sync] contact confirmed: "
                            f"{reason}; awaiting a fresh post-grasp model plan")
                    timeout_started = self._grasp_settle_started
                    timeout_phase = "model close"
                    if (
                        self._grasp_always_tighten
                        and tightening_required
                        and self._grasp_tighten_started is not None
                    ):
                        # Rotation-plan rejection or observation gaps may make
                        # the initial close phase exceed the timeout just as
                        # bounded tightening becomes possible. Give the 60 mm
                        # correction its own full timeout window.
                        timeout_started = self._grasp_tighten_started
                        timeout_phase = "bounded tightening"
                    phase_elapsed = now - timeout_started
                    if (
                        not confirmed
                        and phase_elapsed >= self._grasp_close_timeout
                    ):
                        self._grasp_failed = True
                        self._grasp_hold_width = float(
                            self._actual_gripper if self._actual_gripper is not None
                            else gripper_command)
                        gripper_command = self._grasp_hold_width
                        self.node.get_logger().error(
                            "[grasp_sync] grasp NOT confirmed: "
                            f"{reason}; phase={timeout_phase} "
                            f"elapsed={phase_elapsed:.2f}s; holding TCP and jaw; "
                            "restart after inspection")
        if (
            self._grasp_sync_enabled
            and not self._grasp_model_driven_pose
            and self._grasp_complete
            and self._grasp_awaiting_fresh_plan
        ):
            # Plans inferred while the TCP was artificially held low are not
            # valid post-grasp behavior. Keep the secured grasp until a plan
            # whose observation timestamp is newer than contact is accepted.
            pose_xyz_rotvec = self._grasp_target_pose.copy()
            gripper_command = float(self._grasp_hold_width)
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.node.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.pose_frame
        pose_msg.pose.position.x = float(pose_xyz_rotvec[0])
        pose_msg.pose.position.y = float(pose_xyz_rotvec[1])
        pose_msg.pose.position.z = float(pose_xyz_rotvec[2])
        quat = Rotation.from_rotvec(
            pose_xyz_rotvec[3:6]).as_quat()
        pose_msg.pose.orientation.x = float(quat[0])
        pose_msg.pose.orientation.y = float(quat[1])
        pose_msg.pose.orientation.z = float(quat[2])
        pose_msg.pose.orientation.w = float(quat[3])

        gripper_msg = JointState()
        gripper_msg.header.stamp = pose_msg.header.stamp
        gripper_msg.name = [self.gripper_joint_name]
        gripper_msg.position = [gripper_command]

        self.active_pub.publish(Bool(data=True))
        self.was_active = True
        self.pose_pub.publish(pose_msg)
        self.gripper_pub.publish(gripper_msg)
        self._last_published_action = SampledAction(
            self._clock_now_sec(), pose_xyz_rotvec.copy(), float(gripper_command))

    def close(self) -> None:
        self.timer.cancel()
        self._priming = False
        self.core.clear()
        self._rejected_plan_hold = None
        self._set_inactive()
