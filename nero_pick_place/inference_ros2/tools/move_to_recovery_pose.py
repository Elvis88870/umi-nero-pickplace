#!/usr/bin/env python3
"""Restore the confirmed right-arm task pose by joints, then verify its TCP."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/deploy.yaml"))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Publish motion commands; otherwise only print target errors")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.config.expanduser().resolve().open(
        "r", encoding="utf-8"
    ) as handle:
        config = yaml.safe_load(handle)
    recovery = config["recovery_pose"]
    target_joint_names = [str(name) for name in recovery["joint_names"]]
    target_joints = np.asarray(
        recovery["joint_positions_rad"], dtype=np.float64)
    if len(target_joint_names) != 7 or target_joints.shape != (7,):
        raise ValueError("recovery joint target must contain seven joints")
    if len(set(target_joint_names)) != 7 or not np.all(np.isfinite(target_joints)):
        raise ValueError("recovery joint target is invalid")
    joint_control_topic = str(recovery["joint_control_topic"])
    joint_tolerance = float(recovery["joint_tolerance_rad"])
    max_joint_velocity = float(recovery["max_joint_velocity_rad_s"])
    target_position = np.asarray(
        recovery["position_xyz_m"], dtype=np.float64)
    target_quaternion = np.asarray(
        recovery["orientation_xyzw"], dtype=np.float64)
    if target_position.shape != (3,):
        raise ValueError("recovery position must contain three values")
    if target_quaternion.shape != (4,):
        raise ValueError("recovery quaternion must contain four values")
    target_quaternion /= np.linalg.norm(target_quaternion)
    target_frame = str(recovery["frame_id"])
    target_gripper = float(recovery["gripper_width_m"])
    position_tolerance = float(recovery["position_tolerance_m"])
    rotation_tolerance = float(recovery["rotation_tolerance_rad"])
    gripper_tolerance = float(recovery["gripper_tolerance_m"])
    settle_sec = float(recovery["settle_sec"])
    timeout_sec = float(recovery["timeout_sec"])
    publish_rate = float(recovery["publish_rate_hz"])
    if min(
        position_tolerance,
        rotation_tolerance,
        joint_tolerance,
        max_joint_velocity,
        gripper_tolerance,
        settle_sec,
        timeout_sec,
        publish_rate,
    ) <= 0.0:
        raise ValueError("recovery tolerances, velocity and timing must be positive")

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from std_msgs.msg import Bool
    from sensor_msgs.msg import JointState

    class RecoveryMover(Node):
        def __init__(self) -> None:
            super().__init__("right_recovery_pose_mover")
            command_qos = QoSProfile(
                depth=1, reliability=ReliabilityPolicy.RELIABLE)
            active_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.joint_pub = self.create_publisher(
                JointState, joint_control_topic, command_qos)
            self.gripper_pub = self.create_publisher(
                JointState,
                str(config["control_output"]["gripper_control_topic"]),
                command_qos,
            )
            self.active_pub = self.create_publisher(
                Bool, "/right/teleop_active", active_qos)
            self.tcp = None
            self.joints = None
            self.gripper = None
            self.create_subscription(
                PoseStamped,
                "/right/feedback/tcp_pose",
                self._tcp_callback,
                command_qos,
            )
            self.create_subscription(
                JointState,
                "/right/feedback/joint_states",
                self._joint_callback,
                command_qos,
            )
            self.create_subscription(
                JointState,
                str(config["observation"]["gripper_state_topic"]),
                self._gripper_callback,
                command_qos,
            )

        def _tcp_callback(self, msg: PoseStamped) -> None:
            self.tcp = (
                np.asarray([
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                ], dtype=np.float64),
                np.asarray([
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ], dtype=np.float64),
                str(msg.header.frame_id),
            )

        def _joint_callback(self, msg: JointState) -> None:
            values = dict(zip(msg.name, msg.position))
            if all(name in values for name in target_joint_names):
                self.joints = np.asarray(
                    [values[name] for name in target_joint_names],
                    dtype=np.float64,
                )

        def _gripper_callback(self, msg: JointState) -> None:
            if msg.position:
                self.gripper = float(msg.position[0])

        def publish_state(
            self,
            joints: np.ndarray,
            gripper: float,
            active: bool,
        ) -> None:
            self.active_pub.publish(Bool(data=active))
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = list(target_joint_names)
            msg.position = [float(value) for value in joints]
            self.joint_pub.publish(msg)
            gripper_msg = JointState()
            gripper_msg.header.stamp = msg.header.stamp
            gripper_msg.name = [
                str(config["observation"]["gripper_joint_name"])]
            gripper_msg.position = [float(gripper)]
            self.gripper_pub.publish(gripper_msg)

    rclpy.init()
    node = RecoveryMover()
    success = False
    try:
        deadline = time.monotonic() + 5.0
        while (
            time.monotonic() < deadline
            and (node.tcp is None or node.joints is None or node.gripper is None)
        ):
            rclpy.spin_once(node, timeout_sec=0.02)
        if node.tcp is None or node.joints is None or node.gripper is None:
            raise RuntimeError("no TCP, joint or gripper feedback received")
        position, quaternion, frame = node.tcp
        joints = node.joints.copy()
        gripper = float(node.gripper)
        if frame != target_frame:
            raise RuntimeError(
                f"TCP frame mismatch: observed={frame!r}, "
                f"target={target_frame!r}")
        position_error = float(np.linalg.norm(target_position - position))
        rotation_error = float(
            (Rotation.from_quat(target_quaternion)
             * Rotation.from_quat(quaternion).inv()).magnitude())
        joint_error = float(np.max(np.abs(target_joints - joints)))
        gripper_error = abs(target_gripper - gripper)
        print(
            f"current_joints={joints.tolist()} "
            f"max_joint_error={joint_error:.5f}rad "
            f"current_position={position.tolist()} "
            f"position_error={position_error:.5f}m "
            f"rotation_error={rotation_error:.5f}rad "
            f"gripper={gripper:.5f}m "
            f"gripper_error={gripper_error:.5f}m",
            flush=True)
        if not args.execute:
            print("dry run: no motion command published", flush=True)
            return 0

        pose_topic = str(config["control_output"]["pose_topic"])
        if node.count_publishers(pose_topic) > 0:
            raise RuntimeError(
                f"refusing recovery while {pose_topic} has an active publisher; "
                "stop inference first")
        subscriber_deadline = time.monotonic() + 5.0
        while (
            time.monotonic() < subscriber_deadline
            and node.joint_pub.get_subscription_count() < 1
        ):
            rclpy.spin_once(node, timeout_sec=0.02)
        if node.joint_pub.get_subscription_count() != 1:
            raise RuntimeError(
                f"expected exactly one arm subscriber on {joint_control_topic}; "
                f"found {node.joint_pub.get_subscription_count()}")

        period = 1.0 / publish_rate
        seed_end = time.monotonic() + 0.4
        while time.monotonic() < seed_end:
            node.publish_state(joints, gripper, True)
            rclpy.spin_once(node, timeout_sec=period)

        started = time.monotonic()
        delta = target_joints - joints
        # Cubic smoothstep has a 1.5x peak slope; include it so the configured
        # value is a true peak command-trajectory velocity bound.
        move_duration = max(
            0.5,
            1.5 * float(np.max(np.abs(delta))) / max_joint_velocity,
        )
        if move_duration + settle_sec >= timeout_sec:
            raise RuntimeError(
                f"configured timeout is too short for the joint move: "
                f"duration={move_duration:.2f}s")
        while True:
            elapsed = time.monotonic() - started
            progress = min(1.0, elapsed / move_duration)
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)
            command = joints + smooth_progress * delta
            node.publish_state(command, target_gripper, True)
            rclpy.spin_once(node, timeout_sec=period)
            if progress >= 1.0:
                break

        within_tolerance_since = None
        last_report = 0.0
        while time.monotonic() - started < timeout_sec:
            node.publish_state(target_joints, target_gripper, True)
            rclpy.spin_once(node, timeout_sec=period)
            if node.tcp is None or node.joints is None or node.gripper is None:
                continue
            position, quaternion, _ = node.tcp
            joints = node.joints.copy()
            gripper = float(node.gripper)
            position_error = float(
                np.linalg.norm(target_position - position))
            rotation_error = float(
                (Rotation.from_quat(target_quaternion)
                 * Rotation.from_quat(quaternion).inv()).magnitude())
            joint_error = float(np.max(np.abs(target_joints - joints)))
            gripper_error = abs(target_gripper - gripper)
            now = time.monotonic()
            if (
                joint_error <= joint_tolerance
                and position_error <= position_tolerance
                and rotation_error <= rotation_tolerance
                and gripper_error <= gripper_tolerance
            ):
                if within_tolerance_since is None:
                    within_tolerance_since = now
                elif now - within_tolerance_since >= settle_sec:
                    success = True
                    break
            else:
                within_tolerance_since = None
            if now - last_report >= 0.5:
                print(
                    f"max_joint_error={joint_error:.5f}rad "
                    f"position_error={position_error:.5f}m "
                    f"rotation_error={rotation_error:.5f}rad "
                    f"gripper_error={gripper_error:.5f}m "
                    f"elapsed={now - started:.2f}s",
                    flush=True)
                last_report = now
        if not success:
            raise TimeoutError(
                f"recovery pose not reached within {timeout_sec:.1f}s")
        print(
            f"recovery pose reached: max_joint_error={joint_error:.5f}rad "
            f"position_error={position_error:.5f}m "
            f"rotation_error={rotation_error:.5f}rad "
            f"gripper_error={gripper_error:.5f}m",
            flush=True)
        return 0
    finally:
        node.active_pub.publish(Bool(data=False))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
