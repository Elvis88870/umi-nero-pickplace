"""Timestamp alignment with decoding delegated to an isolated worker process."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from .d405_shm_reader import FrameMetadata, SharedMemoryD405Reader
    from .image_decoder_client import ImageDecoderClient
    from env.ros_observation_buffer import (
        ObservationUnavailable,
        PoseRecord,
        ScalarRecord,
        _bracketing_indices,
        _message_stamp,
        classify_observation_rejection,
        interpolate_pose_records,
        interpolate_scalar_records,
        rejection_reason_code,
    )
except ImportError:
    from d405_shm_reader import FrameMetadata, SharedMemoryD405Reader
    from image_decoder_client import ImageDecoderClient
    from env.ros_observation_buffer import (
        ObservationUnavailable,
        PoseRecord,
        ScalarRecord,
        _bracketing_indices,
        _message_stamp,
        classify_observation_rejection,
        interpolate_pose_records,
        interpolate_scalar_records,
        rejection_reason_code,
    )


@dataclass(frozen=True)
class RawFishFrame:
    sequence: int
    timestamp: float
    payload: bytes


@dataclass(frozen=True)
class RawD405Frame:
    sequence: int
    timestamp: float
    width: int
    height: int
    stride: int
    payload: bytes


class WorkerObservationBuffer:
    def __init__(self, config: Dict[str, Any]):
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.node import Node
        from rclpy.qos import (
            DurabilityPolicy,
            QoSProfile,
            ReliabilityPolicy,
            qos_profile_sensor_data,
        )
        from sensor_msgs.msg import CompressedImage, JointState

        self.rclpy = rclpy
        self.node = Node('worker_umi_preview')
        self.lock = threading.Lock()
        obs = config["observation"]
        worker_cfg = config["decoder_worker"]
        self.settings = {
            "history_horizon": int(obs["history_horizon"]),
            "history_interval_sec": float(obs["history_interval_sec"]),
            "max_stream_age_sec": float(obs["max_stream_age_ms"]) / 1000.0,
            "max_d405_match_residual_sec": (
                float(obs["max_d405_match_residual_ms"]) / 1000.0),
            "max_fisheye_match_residual_sec": (
                float(obs["max_fisheye_match_residual_ms"]) / 1000.0),
            "max_camera_pair_skew_sec": (
                float(obs["max_camera_pair_skew_ms"]) / 1000.0),
            "max_tcp_interpolation_gap_sec": (
                float(obs["max_tcp_interpolation_gap_ms"]) / 1000.0),
            "max_gripper_interpolation_gap_sec": (
                float(obs["max_gripper_interpolation_gap_ms"]) / 1000.0),
            "d405_to_fisheye_offset_sec": (
                float(obs["d405_to_fisheye_offset_ms"]) / 1000.0),
        }
        self.d405_reader = SharedMemoryD405Reader(
            str(obs["d405_shm_name"]),
            float(obs.get("d405_shm_open_timeout_sec", 3.0)),
        )
        self.decoder = ImageDecoderClient(
            str(worker_cfg["socket_path"]),
            float(worker_cfg["timeout_sec"]),
        )
        max_samples = max(100, self.settings["history_horizon"] * 20)
        self.fish: Deque[RawFishFrame] = deque(maxlen=max_samples)
        self.tcp: Deque[PoseRecord] = deque(maxlen=max_samples)
        self.gripper: Deque[ScalarRecord] = deque(maxlen=max_samples)
        self._fish_sequence = 0
        self._last_served_d405_sequence: Optional[int] = None
        self.attempts = 0
        self.accepted = 0
        self.consecutive_rejections = 0
        self.longest_consecutive_rejections = 0
        self.rejection_categories: Counter[str] = Counter()
        self.rejection_reasons: Counter[str] = Counter()
        self.fish_arrivals: Deque[float] = deque(maxlen=100000)
        self.fish_intervals: Deque[float] = deque(maxlen=100000)
        self.pair_skews: Deque[float] = deque(maxlen=100000)
        self.last_reason: Optional[str] = None

        state_qos = QoSProfile(
            depth=int(config["ros"]["state_qos_depth"]),
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.node.create_subscription(
            CompressedImage, str(obs["fisheye_rgb_topic"]),
            self._fish_callback, qos_profile_sensor_data)
        self.node.create_subscription(
            PoseStamped, str(obs["tcp_pose_topic"]),
            self._tcp_callback, state_qos)
        self.node.create_subscription(
            JointState, str(obs["gripper_state_topic"]),
            self._gripper_callback, state_qos)
        self.gripper_joint_name = str(obs["gripper_joint_name"])

    def _fish_callback(self, msg: Any) -> None:
        try:
            stamp = _message_stamp(msg)
            payload = bytes(msg.data)
            if not payload:
                raise ValueError("empty fisheye payload")
        except Exception as exc:
            self.last_reason = f"rejected fisheye frame: {exc}"
            return
        with self.lock:
            self._fish_sequence += 1
            if self.fish:
                self.fish_intervals.append(stamp - self.fish[-1].timestamp)
            self.fish_arrivals.append(time.monotonic())
            self.fish.append(RawFishFrame(
                sequence=self._fish_sequence,
                timestamp=stamp,
                payload=payload,
            ))

    def _tcp_callback(self, msg: Any) -> None:
        try:
            stamp = _message_stamp(msg)
            quat = np.asarray([
                msg.pose.orientation.x, msg.pose.orientation.y,
                msg.pose.orientation.z, msg.pose.orientation.w,
            ], dtype=np.float64)
            norm = float(np.linalg.norm(quat))
            if not np.all(np.isfinite(quat)) or norm < 1e-9:
                raise ValueError("invalid TCP quaternion")
            record = PoseRecord(
                timestamp=stamp,
                xyz=np.asarray([
                    msg.pose.position.x, msg.pose.position.y,
                    msg.pose.position.z,
                ]),
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
            value = (
                float(by_name[self.gripper_joint_name])
                if self.gripper_joint_name in by_name
                else float(msg.position[0]))
            if not math.isfinite(value):
                raise ValueError("non-finite gripper value")
        except Exception as exc:
            self.last_reason = f"rejected gripper state: {exc}"
            return
        with self.lock:
            self.gripper.append(ScalarRecord(stamp, value))

    def _reject(self, reason: str, count_attempt: bool) -> None:
        self.last_reason = reason
        if count_attempt:
            self.rejection_categories[
                classify_observation_rejection(reason)] += 1
            self.rejection_reasons[rejection_reason_code(reason)] += 1
            self.consecutive_rejections += 1
            self.longest_consecutive_rejections = max(
                self.longest_consecutive_rejections,
                self.consecutive_rejections)
        raise ObservationUnavailable(reason)

    @staticmethod
    def _nearest(items: Sequence[Any], target: float) -> Any:
        timestamps = np.asarray(
            [item.timestamp for item in items], dtype=np.float64)
        index = int(np.searchsorted(timestamps, target))
        candidates = []
        if index < len(items):
            candidates.append(index)
        if index > 0:
            candidates.append(index - 1)
        return items[min(
            candidates,
            key=lambda idx: abs(timestamps[idx] - target))]

    def _state_precheck(
        self,
        targets: Sequence[float],
        tcp: Sequence[PoseRecord],
        gripper: Sequence[ScalarRecord],
        now_sec: float,
    ) -> Optional[str]:
        if not tcp or not gripper:
            return "missing TCP or gripper state"
        max_age = self.settings["max_stream_age_sec"]
        if now_sec - tcp[-1].timestamp > max_age:
            return "TCP stream exceeds maximum age"
        if now_sec - gripper[-1].timestamp > max_age:
            return "gripper stream exceeds maximum age"
        tcp_times = np.asarray([item.timestamp for item in tcp])
        gripper_times = np.asarray([item.timestamp for item in gripper])
        for target in targets:
            try:
                low, high = _bracketing_indices(tcp_times, target)
                if tcp_times[high] - tcp_times[low] > self.settings[
                        "max_tcp_interpolation_gap_sec"]:
                    return "TCP interpolation gap exceeds configured limit"
                low, high = _bracketing_indices(gripper_times, target)
                if gripper_times[high] - gripper_times[low] > self.settings[
                        "max_gripper_interpolation_gap_sec"]:
                    return "gripper interpolation gap exceeds configured limit"
            except ObservationUnavailable as exc:
                return str(exc)
        return None

    def get_obs(self, count_attempt: bool = True) -> Dict[str, np.ndarray]:
        if count_attempt:
            self.attempts += 1
        with self.lock:
            fish_snapshot = tuple(self.fish)
            tcp = tuple(self.tcp)
            gripper = tuple(self.gripper)
        d405_metadata = self.d405_reader.snapshot_metadata()
        if not fish_snapshot or not d405_metadata:
            self._reject("missing fisheye or D405 frame", count_attempt)
        master = max(d405_metadata, key=lambda item: item.sequence)
        if master.sequence == self._last_served_d405_sequence:
            self._reject("no new D405 master frame", count_attempt)
        targets = master.timestamp - (
            np.arange(
                self.settings["history_horizon"] - 1,
                -1,
                -1,
                dtype=np.float64,
            ) * self.settings["history_interval_sec"])
        fish_selected: List[RawFishFrame] = []
        d405_selected_meta: List[FrameMetadata] = []
        for target in targets:
            fish_frame = self._nearest(fish_snapshot, float(target))
            d405_frame = self._nearest(d405_metadata, float(target))
            fish_selected.append(fish_frame)
            d405_selected_meta.append(d405_frame)
        if len({frame.sequence for frame in fish_selected}) != len(fish_selected):
            self._reject(
                "fisheye history would reuse one source image", count_attempt)
        if len({
                frame.sequence for frame in d405_selected_meta
            }) != len(d405_selected_meta):
            self._reject(
                "D405 history would reuse one source image", count_attempt)

        offset = self.settings["d405_to_fisheye_offset_sec"]
        now_sec = self.node.get_clock().now().nanoseconds * 1e-9
        pair_skews = []
        for target, fish_frame, d405_frame in zip(
                targets, fish_selected, d405_selected_meta):
            d405_residual = abs(d405_frame.timestamp - target)
            if d405_residual > self.settings["max_d405_match_residual_sec"]:
                self._reject(
                    f"D405 residual {d405_residual * 1000:.1f}ms exceeds "
                    f"{self.settings['max_d405_match_residual_sec'] * 1000:.1f}ms",
                    count_attempt)
            aligned_fish = fish_frame.timestamp + offset
            fish_residual = abs(aligned_fish - target)
            if fish_residual > self.settings["max_fisheye_match_residual_sec"]:
                self._reject(
                    f"fisheye residual {fish_residual * 1000:.1f}ms exceeds "
                    "configured limit", count_attempt)
            pair_skew = abs(aligned_fish - d405_frame.timestamp)
            if pair_skew > self.settings["max_camera_pair_skew_sec"]:
                self._reject(
                    f"camera pair skew {pair_skew * 1000:.1f}ms exceeds "
                    f"{self.settings['max_camera_pair_skew_sec'] * 1000:.1f}ms",
                    count_attempt)
            if now_sec - fish_frame.timestamp > self.settings[
                    "max_stream_age_sec"]:
                self._reject("fisheye frame exceeds maximum age", count_attempt)
            if now_sec - d405_frame.timestamp > self.settings[
                    "max_stream_age_sec"]:
                self._reject("D405 frame exceeds maximum age", count_attempt)
            pair_skews.append(pair_skew)

        state_error = self._state_precheck(targets, tcp, gripper, now_sec)
        if state_error:
            self._reject(state_error, count_attempt)

        d405_selected = []
        for metadata in d405_selected_meta:
            payload = self.d405_reader._read_payload(metadata)
            if payload is None:
                self._reject(
                    "D405 shared-memory frame was overwritten before decode",
                    count_attempt)
            d405_selected.append(RawD405Frame(
                sequence=metadata.sequence,
                timestamp=metadata.timestamp,
                width=metadata.width,
                height=metadata.height,
                stride=metadata.stride,
                payload=payload.tobytes(),
            ))
        try:
            response = self.decoder.decode(fish_selected, d405_selected)
        except Exception as exc:
            self._reject(
                f"decoder worker failed: {exc}", count_attempt)
        if now_sec - min(targets) > self.settings["max_stream_age_sec"]:
            self._reject(
                "decoded image exceeded maximum age", count_attempt)

        images = {"fisheye": {}, "d405": {}}
        for item in response["frames"]:
            image = np.frombuffer(
                item["rgb"], dtype=np.uint8
            ).reshape(item["height"], item["width"], 3)
            images[item["role"]][int(item["sequence"])] = image
        fish_images = [
            images["fisheye"][frame.sequence] for frame in fish_selected]
        d405_images = [
            images["d405"][frame.sequence]
            for frame in d405_selected_meta]
        pose_values = []
        gripper_values = []
        for target in targets:
            xyz, quat = interpolate_pose_records(
                tcp, float(target),
                self.settings["max_tcp_interpolation_gap_sec"])
            pose_values.append(np.concatenate(
                [xyz, Rotation.from_quat(quat).as_rotvec()]))
            gripper_values.append(interpolate_scalar_records(
                gripper, float(target),
                self.settings["max_gripper_interpolation_gap_sec"]))
        poses = np.asarray(pose_values)
        widths = np.asarray(gripper_values)[:, None]
        obs = {
            "camera0_rgb": np.stack(fish_images),
            "camera1_rgb": np.stack(d405_images),
            "robot0_eef_pos": poses[:, :3],
            "robot0_eef_rot_axis_angle": poses[:, 3:6],
            "robot0_gripper_width": widths,
            "timestamp": np.asarray(targets, dtype=np.float64),
        }
        self._last_served_d405_sequence = master.sequence
        if count_attempt:
            self.pair_skews.extend(pair_skews)
            self.accepted += 1
            self.consecutive_rejections = 0
        self.last_reason = None
        return obs

    def wait_for_obs(self, timeout_sec: float):
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            time.sleep(0.02)
            try:
                return self.get_obs(count_attempt=False)
            except ObservationUnavailable:
                continue
        raise TimeoutError(
            f"timed out waiting for worker-backed observation: {self.last_reason}")

    @staticmethod
    def _percentiles(values: Sequence[float]):
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        if not array.size:
            return {"p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
        p50, p95, p99 = np.percentile(array * 1000, [50, 95, 99])
        return {
            "p50_ms": float(p50),
            "p95_ms": float(p95),
            "p99_ms": float(p99),
            "max_ms": float(array.max() * 1000),
        }

    def diagnostics(self):
        rejected = self.attempts - self.accepted
        fish_fps = None
        if len(self.fish_arrivals) > 1:
            elapsed = self.fish_arrivals[-1] - self.fish_arrivals[0]
            fish_fps = (len(self.fish_arrivals) - 1) / elapsed
        return {
            **self.decoder.diagnostics(),
            "attempts": self.attempts,
            "accepted": self.accepted,
            "rejected": rejected,
            "rejected_ratio_percent": (
                100 * rejected / self.attempts if self.attempts else 0),
            "longest_consecutive_rejections": (
                self.longest_consecutive_rejections),
            "rejection_categories": dict(self.rejection_categories),
            "rejection_reasons": dict(self.rejection_reasons),
            "fisheye_receive_fps": fish_fps,
            "fisheye_header_dt": self._percentiles(self.fish_intervals),
            "pair_skew": self._percentiles(self.pair_skews),
            "d405_ring": self.d405_reader.diagnostics(),
        }

    def destroy(self):
        self.decoder.close()
        self.d405_reader.close()
        self.node.destroy_node()
