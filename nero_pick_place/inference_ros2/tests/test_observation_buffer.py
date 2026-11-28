import math
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.ros_observation_buffer import (  # noqa: E402
    ImageRecord,
    ObservationAssembler,
    ObservationUnavailable,
    PoseRecord,
    ScalarRecord,
    RosObservationBuffer,
    classify_observation_rejection,
)


def settings(**overrides):
    value = {
        "max_stream_age_sec": 0.2,
        "max_d405_match_residual_sec": 0.03,
        "max_fisheye_match_residual_sec": 0.03,
        "max_camera_pair_skew_sec": 0.05,
        "max_tcp_interpolation_gap_sec": 0.3,
        "max_gripper_interpolation_gap_sec": 0.3,
        "history_horizon": 2,
        "history_interval_sec": 0.1,
        "d405_to_fisheye_offset_sec": 0.0,
    }
    value.update(overrides)
    return value


def image(timestamp, value):
    return ImageRecord(
        timestamp,
        np.full((2, 2, 3), value, dtype=np.uint8),
    )


def pose(timestamp, yaw):
    return PoseRecord(
        timestamp=timestamp,
        xyz=np.asarray([timestamp, 0.0, 0.2], dtype=np.float64),
        quat_xyzw=np.asarray([
            0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        ], dtype=np.float64),
        frame_id="right_base_link",
    )


class ObservationAssemblerTest(unittest.TestCase):
    def test_pose_uses_linear_xyz_and_quaternion_slerp(self):
        assembler = ObservationAssembler(settings())
        obs = assembler.assemble(
            d405_records=[
                image(19.8, 1), image(19.9, 2), image(20.0, 3)],
            fisheye_records=[
                image(19.8, 4), image(19.9, 5), image(20.0, 6)],
            tcp_records=[pose(19.8, 0.0), pose(20.0, 1.0)],
            gripper_records=[
                ScalarRecord(19.8, 0.01), ScalarRecord(20.0, 0.03)],
            now_sec=20.05,
        )
        self.assertEqual(obs["camera0_rgb"].shape, (2, 2, 2, 3))
        self.assertEqual(obs["camera1_rgb"].shape, (2, 2, 2, 3))
        self.assertAlmostEqual(obs["robot0_eef_pos"][0, 0], 19.9)
        self.assertAlmostEqual(
            obs["robot0_eef_rot_axis_angle"][0, 2], 0.5, places=6)
        self.assertAlmostEqual(
            obs["robot0_gripper_width"][0, 0], 0.02, places=7)
        self.assertEqual(len(assembler.last_pair_skews_sec), 2)

    def test_single_fisheye_observation_does_not_require_d405(self):
        assembler = ObservationAssembler(
            settings(), include_d405=False)
        obs = assembler.assemble(
            fisheye_records=[
                image(19.8, 4), image(19.9, 5), image(20.0, 6)],
            tcp_records=[pose(19.8, 0.0), pose(20.0, 1.0)],
            gripper_records=[
                ScalarRecord(19.8, 0.01), ScalarRecord(20.0, 0.03)],
            now_sec=20.05,
        )
        self.assertEqual(obs["camera0_rgb"].shape, (2, 2, 2, 3))
        self.assertNotIn("camera1_rgb", obs)
        self.assertEqual(len(assembler.last_pair_skews_sec), 0)

    def test_sync_limit_rejects_observation(self):
        assembler = ObservationAssembler(settings(max_stream_age_sec=0.6))
        with self.assertRaisesRegex(ObservationUnavailable, "fisheye residual"):
            assembler.assemble(
                d405_records=[image(19.9, 1), image(20.0, 2)],
                fisheye_records=[image(19.5, 3), image(19.6, 4)],
                tcp_records=[pose(19.8, 0.0), pose(20.0, 0.0)],
                gripper_records=[
                    ScalarRecord(19.8, 0.01), ScalarRecord(20.0, 0.01)],
                now_sec=20.05,
            )
        self.assertGreater(sum(assembler.rejection_counts.values()), 0)
        self.assertEqual(len(assembler.last_pair_skews_sec), 0)

    def test_image_reuse_is_rejected(self):
        assembler = ObservationAssembler(settings(
            max_fisheye_match_residual_sec=0.06,
            max_camera_pair_skew_sec=0.1,
        ))
        with self.assertRaisesRegex(ObservationUnavailable, "reuse"):
            assembler.assemble(
                d405_records=[image(19.9, 1), image(20.0, 2)],
                fisheye_records=[image(19.95, 3)],
                tcp_records=[pose(19.8, 0.0), pose(20.0, 0.0)],
                gripper_records=[
                    ScalarRecord(19.8, 0.01), ScalarRecord(20.0, 0.01)],
                now_sec=20.05,
            )

    def test_stale_stream_rejects_observation(self):
        assembler = ObservationAssembler(settings())
        with self.assertRaisesRegex(ObservationUnavailable, "stream age"):
            assembler.assemble(
                d405_records=[image(19.0, 1), image(19.1, 2)],
                fisheye_records=[image(19.0, 3), image(19.1, 4)],
                tcp_records=[pose(19.0, 0.0), pose(19.1, 0.0)],
                gripper_records=[
                    ScalarRecord(19.0, 0.01), ScalarRecord(19.1, 0.01)],
                now_sec=20.0,
            )

    def test_rejection_reason_categories(self):
        self.assertEqual(
            classify_observation_rejection(
                "camera pair skew 40.0ms exceeds 30.0ms"),
            "camera_desynchronization")
        self.assertEqual(
            classify_observation_rejection(
                "D405 stream age 400.0ms exceeds 250.0ms"),
            "stale_data")
        self.assertEqual(
            classify_observation_rejection(
                "fisheye history would reuse one source image"),
            "image_reuse")

    def test_compressed_image_decode_preserves_stamp_and_rgb_order(self):
        bgr = np.zeros((4, 5, 3), dtype=np.uint8)
        bgr[:, :, 2] = 255
        ok, encoded = cv2.imencode(".jpg", bgr)
        self.assertTrue(ok)
        header = SimpleNamespace(
            stamp=SimpleNamespace(sec=123, nanosec=456000000))
        message = SimpleNamespace(
            header=header,
            format="jpeg; compressed mjpeg",
            data=encoded.tobytes(),
        )
        buffer = object.__new__(RosObservationBuffer)
        buffer.cv2 = cv2
        record = buffer._decode_compressed_image(message)
        self.assertAlmostEqual(record.timestamp, 123.456, places=9)
        self.assertEqual(record.image.shape, (4, 5, 3))
        self.assertGreater(int(record.image[..., 0].mean()), 240)
        self.assertLess(int(record.image[..., 2].mean()), 15)


if __name__ == "__main__":
    unittest.main()
