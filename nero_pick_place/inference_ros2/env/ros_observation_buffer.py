"""Timestamped ROS2 observation buffer for NERO UMI inference.

The buffer is intentionally independent of training and hardware drivers. It
only subscribes to the four observation streams and constructs model input.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Deque, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class ObservationUnavailable(RuntimeError):
    """Raised when a synchronized, model-compatible observation cannot be built."""


def classify_observation_rejection(reason: str) -> str:
    lowered = str(reason).casefold()
    if "shared-memory" in lowered:
        return "camera_desynchronization"
    if "reuse one source image" in lowered:
        return "image_reuse"
    if "camera pair skew" in lowered or "residual" in lowered:
        return "camera_desynchronization"
    if "stream age" in lowered or "in the future" in lowered:
        return "stale_data"
    if "interpolation gap" in lowered or "outside interpolation" in lowered:
        return "state_gap"
    return "other"


def rejection_reason_code(reason: str) -> str:
    lowered = str(reason).casefold()
    if lowered.startswith("fisheye residual"):
        return "fisheye_residual"
    if lowered.startswith("d405 shared-memory residual"):
        return "d405_shm_residual"
    if lowered.startswith("d405 residual"):
        return "d405_residual"
    if lowered.startswith("camera pair skew"):
        return "camera_pair_skew"
    if "reuse one source image" in lowered:
        return "image_reuse"
    if "stream age" in lowered:
        return "stream_age"
    if "shared-memory" in lowered:
        return "d405_shm_unavailable"
    if "interpolation gap" in lowered or "outside interpolation" in lowered:
        return "state_interpolation_gap"
    if "timestamp is in the future" in lowered:
        return "future_timestamp"
    return "other"


@dataclass(frozen=True)
class ImageRecord:
    timestamp: float
    image: np.ndarray


@dataclass(frozen=True)
class PoseRecord:
    timestamp: float
    xyz: np.ndarray
    quat_xyzw: np.ndarray
    frame_id: str


@dataclass(frozen=True)
class ScalarRecord:
    timestamp: float
    value: float


def _message_stamp(msg: Any) -> float:
    stamp = msg.header.stamp
    value = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("message has no valid header timestamp")
    return value


def _nearest_index(timestamps: np.ndarray, target: float) -> int:
    position = int(np.searchsorted(timestamps, target))
    candidates = []
    if position < timestamps.size:
        candidates.append(position)
    if position > 0:
        candidates.append(position - 1)
    if not candidates:
        raise ObservationUnavailable("empty timestamp history")
    return min(candidates, key=lambda idx: abs(float(timestamps[idx]) - target))


def _bracketing_indices(
        timestamps: np.ndarray, target: float) -> Tuple[int, int]:
    upper = int(np.searchsorted(timestamps, target, side="left"))
    if upper < timestamps.size and abs(timestamps[upper] - target) <= 1e-9:
        return upper, upper
    if upper == 0 or upper >= timestamps.size:
        raise ObservationUnavailable(
            f"state target {target:.6f} is outside interpolation history")
    return upper - 1, upper


def interpolate_pose_records(
        records: Sequence[PoseRecord],
        target: float,
        max_gap_sec: float,
) -> Tuple[np.ndarray, float]:
    """Interpolate xyz linearly and orientation with quaternion SLERP."""
    if len(records) < 2:
        raise ObservationUnavailable("insufficient TCP pose history")
    timestamps = np.asarray([item.timestamp for item in records], dtype=np.float64)
    lower, upper = _bracketing_indices(timestamps, target)
    gap = float(timestamps[upper] - timestamps[lower])
    if gap > max_gap_sec:
        raise ObservationUnavailable(
            f"TCP interpolation gap {gap * 1000.0:.1f}ms exceeds "
            f"{max_gap_sec * 1000.0:.1f}ms")
    if lower == upper:
        return records[lower].xyz.copy(), records[lower].quat_xyzw.copy()
    alpha = (target - float(timestamps[lower])) / gap
    xyz = (
        (1.0 - alpha) * records[lower].xyz
        + alpha * records[upper].xyz
    )
    rotations = Rotation.from_quat([
        records[lower].quat_xyzw, records[upper].quat_xyzw])
    quat = Slerp(
        [float(timestamps[lower]), float(timestamps[upper])], rotations
    )([target]).as_quat()[0]
    return xyz, quat


def interpolate_scalar_records(
        records: Sequence[ScalarRecord],
        target: float,
        max_gap_sec: float,
) -> float:
    if len(records) < 2:
        raise ObservationUnavailable("insufficient gripper history")
    timestamps = np.asarray([item.timestamp for item in records], dtype=np.float64)
    lower, upper = _bracketing_indices(timestamps, target)
    gap = float(timestamps[upper] - timestamps[lower])
    if gap > max_gap_sec:
        raise ObservationUnavailable(
            f"gripper interpolation gap {gap * 1000.0:.1f}ms exceeds "
            f"{max_gap_sec * 1000.0:.1f}ms")
    if lower == upper:
        return float(records[lower].value)
    alpha = (target - float(timestamps[lower])) / gap
    return float(
        (1.0 - alpha) * records[lower].value
        + alpha * records[upper].value)


class ObservationAssembler:
    """Pure timestamp-alignment implementation used by ROS and unit tests."""

    BASE_REQUIRED_KEYS = (
        "max_stream_age_sec",
        "max_fisheye_match_residual_sec",
        "max_tcp_interpolation_gap_sec",
        "max_gripper_interpolation_gap_sec",
        "history_horizon",
        "history_interval_sec",
    )
    D405_REQUIRED_KEYS = (
        "max_d405_match_residual_sec",
        "max_camera_pair_skew_sec",
        "d405_to_fisheye_offset_sec",
    )

    def __init__(
        self,
        settings: Dict[str, Any],
        include_d405: Optional[bool] = None,
    ):
        self.include_d405 = (
            bool(settings.get("include_d405", True))
            if include_d405 is None else bool(include_d405)
        )
        required_keys = self.BASE_REQUIRED_KEYS + (
            self.D405_REQUIRED_KEYS if self.include_d405 else ())
        missing = [key for key in required_keys if key not in settings]
        if missing:
            raise ValueError(f"missing observation settings: {missing}")
        self.settings = dict(settings)
        if self.settings["history_horizon"] < 1:
            raise ValueError("history_horizon must be positive")
        if self.settings["history_interval_sec"] <= 0.0:
            raise ValueError("history_interval_sec must be positive")
        for key in required_keys:
            if key in ("history_horizon", "history_interval_sec"):
                continue
            if float(self.settings[key]) < 0.0:
                raise ValueError(f"{key} must be nonnegative")
        self.rejection_counts: Counter[str] = Counter()
        self.last_pair_skews_sec: list[float] = []

    def _reject(self, reason: str) -> None:
        self.rejection_counts[reason] += 1
        raise ObservationUnavailable(reason)

    def assemble(
        self,
        fisheye_records: Sequence[ImageRecord],
        tcp_records: Sequence[PoseRecord],
        gripper_records: Sequence[ScalarRecord],
        now_sec: float,
        d405_records: Sequence[ImageRecord] = (),
    ) -> Dict[str, np.ndarray]:
        if self.include_d405 and not d405_records:
            self._reject("no D405 RGB")
        if not fisheye_records:
            self._reject("no fisheye RGB")
        if not tcp_records:
            self._reject("no TCP pose")
        if not gripper_records:
            self._reject("no gripper state")

        streams = {
            "fisheye": fisheye_records[-1].timestamp,
            "TCP": tcp_records[-1].timestamp,
            "gripper": gripper_records[-1].timestamp,
        }
        if self.include_d405:
            streams["D405"] = d405_records[-1].timestamp
        for name, stamp in streams.items():
            age = now_sec - stamp
            if age < -0.05:
                self._reject(f"{name} timestamp is in the future")
            if age > self.settings["max_stream_age_sec"]:
                self._reject(
                    f"{name} stream age {age * 1000.0:.1f}ms exceeds "
                    f"{self.settings['max_stream_age_sec'] * 1000.0:.1f}ms")

        master_time = float(
            d405_records[-1].timestamp
            if self.include_d405 else fisheye_records[-1].timestamp)
        horizon = int(self.settings["history_horizon"])
        interval = float(self.settings["history_interval_sec"])
        targets = master_time - (
            np.arange(horizon - 1, -1, -1, dtype=np.float64) * interval)

        d405_times = np.asarray(
            [item.timestamp for item in d405_records], dtype=np.float64)
        fisheye_times = np.asarray(
            [item.timestamp for item in fisheye_records], dtype=np.float64)
        offset = float(
            self.settings.get("d405_to_fisheye_offset_sec", 0.0))

        d405_images = []
        fisheye_images = []
        self.last_pair_skews_sec = []
        used_d405 = set()
        used_fisheye = set()
        for target in targets:
            d405_index = -1
            if self.include_d405:
                d405_index = _nearest_index(d405_times, float(target))
                d405_residual = abs(float(d405_times[d405_index]) - target)
                if d405_residual > self.settings[
                        "max_d405_match_residual_sec"]:
                    self._reject(
                        f"D405 residual {d405_residual * 1000.0:.1f}ms "
                        "exceeds configured limit")
                if d405_index in used_d405:
                    self._reject(
                        "D405 history would reuse one source image")
                used_d405.add(d405_index)

            aligned_fisheye_times = fisheye_times + offset
            fisheye_index = _nearest_index(
                aligned_fisheye_times, float(target))
            fisheye_residual = abs(
                float(aligned_fisheye_times[fisheye_index]) - target)
            if fisheye_residual > self.settings[
                    "max_fisheye_match_residual_sec"]:
                self._reject(
                    f"fisheye residual {fisheye_residual * 1000.0:.1f}ms "
                    "exceeds configured limit")
            if fisheye_index in used_fisheye:
                self._reject("fisheye history would reuse one source image")
            used_fisheye.add(fisheye_index)

            if self.include_d405:
                pair_skew = abs(
                    float(aligned_fisheye_times[fisheye_index])
                    - float(d405_times[d405_index]))
                self.last_pair_skews_sec.append(pair_skew)
                if pair_skew > self.settings["max_camera_pair_skew_sec"]:
                    self._reject(
                        f"camera pair skew {pair_skew * 1000.0:.1f}ms "
                        "exceeds configured limit")
                d405_images.append(d405_records[d405_index].image)
            fisheye_images.append(fisheye_records[fisheye_index].image)

        pose_values = []
        gripper_values = []
        for target in targets:
            xyz, quat = interpolate_pose_records(
                tcp_records, float(target),
                self.settings["max_tcp_interpolation_gap_sec"])
            pose_values.append(np.concatenate(
                [xyz, Rotation.from_quat(quat).as_rotvec()]))
            gripper_values.append(interpolate_scalar_records(
                gripper_records, float(target),
                self.settings["max_gripper_interpolation_gap_sec"]))

        fisheye_array = np.stack(fisheye_images)
        poses = np.asarray(pose_values, dtype=np.float64)
        widths = np.asarray(gripper_values, dtype=np.float64)[:, None]
        output = {
            "camera0_rgb": fisheye_array,
            "robot0_eef_pos": poses[:, :3],
            "robot0_eef_rot_axis_angle": poses[:, 3:6],
            "robot0_gripper_width": widths,
            "timestamp": targets,
        }
        if self.include_d405:
            d405_array = np.stack(d405_images)
            if d405_array.shape != fisheye_array.shape:
                self._reject(
                    f"camera shapes differ: D405={d405_array.shape}, "
                    f"fisheye={fisheye_array.shape}")
            output["camera1_rgb"] = d405_array

        return output


class RosObservationBuffer:
    """ROS2 subscriptions and timestamp caches around ObservationAssembler."""

    def __init__(self, config: Dict[str, Any]):
        import cv_bridge
        import cv2
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.node import Node
        from rclpy.qos import (
            DurabilityPolicy,
            QoSHistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
            qos_profile_sensor_data,
        )
        from sensor_msgs.msg import CompressedImage, Image, JointState

        self.rclpy = rclpy
        self.cv2 = cv2
        self.cv_bridge = cv_bridge
        self.cv_bridge_converter = cv_bridge.CvBridge()
        self.Image = Image
        self.lock = threading.Lock()
        self.node = Node(str(config["ros"]["node_name"]))

        obs = config["observation"]
        self.include_d405 = bool(obs.get("include_d405", True))
        self.settings = {
            "history_horizon": int(obs["history_horizon"]),
            "history_interval_sec": float(obs["history_interval_sec"]),
            "max_stream_age_sec": float(obs["max_stream_age_ms"]) / 1000.0,
            "max_fisheye_match_residual_sec": (
                float(obs["max_fisheye_match_residual_ms"]) / 1000.0),
            "max_tcp_interpolation_gap_sec": (
                float(obs["max_tcp_interpolation_gap_ms"]) / 1000.0),
            "max_gripper_interpolation_gap_sec": (
                float(obs["max_gripper_interpolation_gap_ms"]) / 1000.0),
        }
        if self.include_d405:
            self.settings.update({
                "max_d405_match_residual_sec": (
                    float(obs["max_d405_match_residual_ms"]) / 1000.0),
                "max_camera_pair_skew_sec": (
                    float(obs["max_camera_pair_skew_ms"]) / 1000.0),
                "d405_to_fisheye_offset_sec": (
                    float(obs["d405_to_fisheye_offset_ms"]) / 1000.0),
            })
        self.assembler = ObservationAssembler(
            self.settings, include_d405=self.include_d405)
        max_samples = max(100, int(obs["history_horizon"]) * 20)
        self.d405: Deque[ImageRecord] = deque(maxlen=max_samples)
        self.fisheye: Deque[ImageRecord] = deque(maxlen=max_samples)
        self.tcp: Deque[PoseRecord] = deque(maxlen=max_samples)
        self.gripper: Deque[ScalarRecord] = deque(maxlen=max_samples)
        self.last_reason: Optional[str] = None
        self.last_obs_time: Optional[float] = None
        self.get_attempts = 0
        self.get_successes = 0
        self.rejection_categories: Counter[str] = Counter()
        self.detailed_rejection_counts: Counter[str] = Counter()
        self.pair_skew_samples: Deque[float] = deque(maxlen=100000)
        self.stream_stats: Dict[str, Dict[str, Any]] = {
            "fisheye": self._new_stream_stats(),
        }
        if self.include_d405:
            self.stream_stats["d405"] = self._new_stream_stats()
        self.d405_shm_reader = None

        image_qos = qos_profile_sensor_data
        fisheye_qos = QoSProfile(
            depth=max(1, int(config["ros"].get("fisheye_qos_depth", 30))),
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        state_qos = QoSProfile(
            depth=int(config["ros"]["state_qos_depth"]),
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        if self.include_d405:
            d405_source = str(obs.get("d405_source", "ros_image")).casefold()
            if d405_source in ("shared_memory", "shm"):
                from experiments.d405_worker.d405_shm_reader import (
                    SharedMemoryD405Reader,
                )
                self.d405_shm_reader = SharedMemoryD405Reader(
                    str(obs["d405_shm_name"]),
                    float(obs.get("d405_shm_open_timeout_sec", 3.0)),
                )
            elif d405_source in ("ros_image", "image", "ros"):
                self.node.create_subscription(
                    Image, str(obs["d405_rgb_topic"]),
                    self._d405_callback, image_qos)
            else:
                raise ValueError(f"unsupported d405_source: {d405_source}")
        fisheye_message_type = str(
            obs.get("fisheye_message_type", "CompressedImage")).casefold()
        if fisheye_message_type in ("compressedimage", "compressed"):
            self.node.create_subscription(
                CompressedImage, str(obs["fisheye_rgb_topic"]),
                self._fisheye_callback, fisheye_qos)
        elif fisheye_message_type == "image":
            self.node.create_subscription(
                Image, str(obs["fisheye_rgb_topic"]),
                self._fisheye_callback, fisheye_qos)
        else:
            raise ValueError(
                f"unsupported fisheye_message_type: {fisheye_message_type}")
        self.node.create_subscription(
            PoseStamped, str(obs["tcp_pose_topic"]),
            self._tcp_callback, state_qos)
        self.node.create_subscription(
            JointState, str(obs["gripper_state_topic"]),
            self._gripper_callback, state_qos)
        self.gripper_joint_name = str(obs["gripper_joint_name"])

    @staticmethod
    def _new_stream_stats() -> Dict[str, Any]:
        return {
            "frames": 0,
            "decode_rejections": 0,
            "first_stamp": None,
            "last_stamp": None,
            "first_arrival": None,
            "last_arrival": None,
            "intervals": deque(maxlen=100000),
        }

    def _record_stream(self, name: str, record: ImageRecord) -> None:
        stats = self.stream_stats[name]
        arrival = time.monotonic()
        if stats["first_stamp"] is None:
            stats["first_stamp"] = record.timestamp
            stats["first_arrival"] = arrival
        else:
            stats["intervals"].append(record.timestamp - stats["last_stamp"])
        stats["last_stamp"] = record.timestamp
        stats["last_arrival"] = arrival
        stats["frames"] += 1

    def _decode_image(self, msg: Any) -> ImageRecord:
        if not msg.encoding:
            raise ValueError("empty image encoding")
        image = self.cv_bridge_converter.imgmsg_to_cv2(
            msg, desired_encoding="rgb8")
        return ImageRecord(
            timestamp=_message_stamp(msg),
            image=np.ascontiguousarray(image),
        )

    def _decode_compressed_image(self, msg: Any) -> ImageRecord:
        image_format = str(msg.format).casefold()
        if "jpeg" not in image_format and "jpg" not in image_format:
            raise ValueError(
                f"unsupported CompressedImage format: {msg.format!r}")
        encoded = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = self.cv2.imdecode(encoded, self.cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode returned None")
        rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
        return ImageRecord(
            timestamp=_message_stamp(msg),
            image=np.ascontiguousarray(rgb),
        )

    def _d405_callback(self, msg: Any) -> None:
        try:
            record = self._decode_image(msg)
        except Exception as exc:
            self.last_reason = f"rejected D405 frame: {exc}"
            self.stream_stats["d405"]["decode_rejections"] += 1
            return
        with self.lock:
            self.d405.append(record)
            self._record_stream("d405", record)

    def _fisheye_callback(self, msg: Any) -> None:
        try:
            if hasattr(msg, "format"):
                record = self._decode_compressed_image(msg)
            else:
                record = self._decode_image(msg)
        except Exception as exc:
            self.last_reason = f"rejected fisheye frame: {exc}"
            self.stream_stats["fisheye"]["decode_rejections"] += 1
            return
        with self.lock:
            self.fisheye.append(record)
            self._record_stream("fisheye", record)

    def _tcp_callback(self, msg: Any) -> None:
        try:
            stamp = _message_stamp(msg)
            quat = np.asarray([
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ], dtype=np.float64)
            norm = float(np.linalg.norm(quat))
            if not np.all(np.isfinite(quat)) or norm < 1e-9:
                raise ValueError("invalid TCP quaternion")
            record = PoseRecord(
                timestamp=stamp,
                xyz=np.asarray([
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                ], dtype=np.float64),
                quat_xyzw=quat / norm,
                frame_id=str(msg.header.frame_id),
            )
        except Exception as exc:
            self.last_reason = f"rejected TCP pose: {exc}"
            return
        with self.lock:
            self.tcp.append(record)

    def _gripper_callback(self, msg: Any) -> None:
        try:
            stamp = _message_stamp(msg)
            by_name = dict(zip(msg.name, msg.position))
            if self.gripper_joint_name in by_name:
                value = float(by_name[self.gripper_joint_name])
            elif msg.position:
                value = float(msg.position[0])
            else:
                raise ValueError("empty gripper position")
            if not math.isfinite(value):
                raise ValueError("non-finite gripper position")
        except Exception as exc:
            self.last_reason = f"rejected gripper state: {exc}"
            return
        with self.lock:
            self.gripper.append(ScalarRecord(stamp, value))

    def spin_once(self, timeout_sec: float = 0.02) -> None:
        self.rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def _d405_shm_plan_is_synchronized(
        self,
        plan: Any,
        fisheye_records: Sequence[ImageRecord],
    ) -> Tuple[bool, str]:
        if len(plan.frames) != len(plan.targets) or not fisheye_records:
            return False, "D405 shared-memory history frame reuse"
        fisheye_times = np.asarray(
            [item.timestamp for item in fisheye_records], dtype=np.float64)
        aligned_fisheye_times = (
            fisheye_times + self.settings["d405_to_fisheye_offset_sec"])
        used_fisheye = set()
        for target, d405_frame in zip(plan.targets, plan.frames):
            d405_residual = abs(float(d405_frame.timestamp) - target)
            if d405_residual > self.settings[
                    "max_d405_match_residual_sec"]:
                return False, (
                    f"D405 shared-memory residual "
                    f"{d405_residual * 1000.0:.1f}ms exceeds configured limit")
            fisheye_index = _nearest_index(
                aligned_fisheye_times, float(target))
            if fisheye_index in used_fisheye:
                return False, "fisheye history would reuse one source image"
            used_fisheye.add(fisheye_index)
            fisheye_residual = abs(
                float(aligned_fisheye_times[fisheye_index]) - target)
            if fisheye_residual > self.settings[
                    "max_fisheye_match_residual_sec"]:
                return False, (
                    f"fisheye residual {fisheye_residual * 1000.0:.1f}ms "
                    "exceeds configured limit")
            pair_skew = abs(
                float(aligned_fisheye_times[fisheye_index])
                - float(d405_frame.timestamp))
            if pair_skew > self.settings["max_camera_pair_skew_sec"]:
                return False, (
                    f"camera pair skew {pair_skew * 1000.0:.1f}ms exceeds "
                    f"{self.settings['max_camera_pair_skew_sec'] * 1000.0:.1f}ms")
        return True, ""

    def get_obs(
        self,
        count_attempt: bool = True,
    ) -> Dict[str, np.ndarray]:
        if count_attempt:
            self.get_attempts += 1
        with self.lock:
            fisheye = tuple(self.fisheye)
            tcp = tuple(self.tcp)
            gripper = tuple(self.gripper)
            if self.d405_shm_reader is None:
                d405 = tuple(self.d405)
            else:
                d405 = ()
        if self.include_d405 and self.d405_shm_reader is not None:
            try:
                plan = self.d405_shm_reader.plan_history(
                    self.settings["history_horizon"],
                    self.settings["history_interval_sec"],
                )
                d405 = ()
                sync_error = None
                if plan is not None:
                    synchronized, sync_error = (
                        self._d405_shm_plan_is_synchronized(plan, fisheye))
                    if synchronized:
                        d405 = tuple(
                            self.d405_shm_reader.decode_plan(plan))
            except Exception as exc:
                d405 = ()
                shared_memory_error = str(exc)
            else:
                shared_memory_error = None
            if not d405:
                reason = (
                    sync_error
                    if sync_error is not None
                    else "D405 shared-memory returned no complete frame"
                    if shared_memory_error is None
                    else "D405 shared-memory read failed: "
                    + shared_memory_error)
                self.last_reason = reason
                if count_attempt:
                    self.rejection_categories[
                        classify_observation_rejection(reason)] += 1
                    self.detailed_rejection_counts[
                        rejection_reason_code(reason)] += 1
                raise ObservationUnavailable(reason)
        now_sec = self.node.get_clock().now().nanoseconds * 1e-9
        if now_sec <= 0.0:
            now_sec = time.time()
        try:
            obs = self.assembler.assemble(
                fisheye, tcp, gripper, now_sec,
                d405_records=d405)
        except ObservationUnavailable as exc:
            self.last_reason = str(exc)
            if count_attempt:
                with self.lock:
                    self.pair_skew_samples.extend(
                        self.assembler.last_pair_skews_sec)
                self.rejection_categories[
                    classify_observation_rejection(str(exc))] += 1
                self.detailed_rejection_counts[
                    rejection_reason_code(str(exc))] += 1
            raise
        if count_attempt:
            with self.lock:
                self.pair_skew_samples.extend(
                    self.assembler.last_pair_skews_sec)
            self.get_successes += 1
        self.last_reason = None
        self.last_obs_time = now_sec
        return obs

    @property
    def latest_tcp(self) -> Optional[PoseRecord]:
        with self.lock:
            return self.tcp[-1] if self.tcp else None

    @property
    def latest_gripper(self) -> Optional[ScalarRecord]:
        with self.lock:
            return self.gripper[-1] if self.gripper else None

    def destroy(self) -> None:
        if self.d405_shm_reader is not None:
            self.d405_shm_reader.close()
            self.d405_shm_reader = None
        self.node.destroy_node()

    @staticmethod
    def _percentiles(
        values: Iterable[float],
    ) -> Dict[str, Optional[float]]:
        array = np.asarray(list(values), dtype=np.float64)
        array = array[np.isfinite(array)]
        if array.size == 0:
            return {
                "p50_ms": None,
                "p95_ms": None,
                "p99_ms": None,
                "max_ms": None,
            }
        p50, p95, p99 = np.percentile(array, [50.0, 95.0, 99.0])
        return {
            "p50_ms": float(p50) * 1000.0,
            "p95_ms": float(p95) * 1000.0,
            "p99_ms": float(p99) * 1000.0,
            "max_ms": float(array.max()) * 1000.0,
        }

    def diagnostics(self) -> Dict[str, Any]:
        """Return cumulative rates and rejection statistics."""
        with self.lock:
            streams = {}
            for name, stats in self.stream_stats.items():
                frames = int(stats["frames"])
                receive_fps = None
                header_fps = None
                if frames > 1:
                    receive_elapsed = (
                        float(stats["last_arrival"])
                        - float(stats["first_arrival"]))
                    header_elapsed = (
                        float(stats["last_stamp"])
                        - float(stats["first_stamp"]))
                    if receive_elapsed > 0.0:
                        receive_fps = (frames - 1) / receive_elapsed
                    if header_elapsed > 0.0:
                        header_fps = (frames - 1) / header_elapsed
                streams[name] = {
                    "frames": frames,
                    "receive_fps": receive_fps,
                    "header_fps": header_fps,
                    "first_header_stamp": stats["first_stamp"],
                    "last_header_stamp": stats["last_stamp"],
                    "decode_rejections": int(stats["decode_rejections"]),
                    "header_interval": self._percentiles(stats["intervals"]),
                }
            pair_skew = self._percentiles(self.pair_skew_samples)
            rejection_categories = dict(self.rejection_categories)
            detailed_rejection_counts = dict(
                self.detailed_rejection_counts)
        attempts = int(self.get_attempts)
        rejected = attempts - int(self.get_successes)
        rate = 100.0 * rejected / attempts if attempts else 0.0
        category_rates = {
            key: (
                100.0 * value / attempts if attempts else 0.0)
            for key, value in rejection_categories.items()
        }
        result = {
            "attempts": attempts,
            "accepted": int(self.get_successes),
            "rejected": rejected,
            "rejected_ratio_percent": rate,
            "rejection_counts": rejection_categories,
            "rejection_ratios_percent": category_rates,
            "detailed_rejection_counts": detailed_rejection_counts,
            "pair_skew": pair_skew,
            "streams": streams,
        }
        if self.include_d405 and self.d405_shm_reader is not None:
            result["d405_shared_memory"] = self.d405_shm_reader.diagnostics()
        return result
