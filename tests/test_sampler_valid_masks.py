import unittest

import numpy as np

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler


def make_buffer():
    count = 100
    zeros = np.zeros((count, 3), dtype=np.float32)
    replay = ReplayBuffer.create_empty_numpy()
    replay.add_episode({
        "camera0_rgb": np.zeros((count, 2, 2, 3), dtype=np.uint8),
        "camera1_rgb": np.zeros((count, 2, 2, 3), dtype=np.uint8),
        "robot0_eef_pos": zeros.copy(),
        "robot0_eef_rot_axis_angle": zeros.copy(),
        "robot0_gripper_width": np.full((count, 1), 0.04, dtype=np.float32),
        "fisheye_valid": np.ones(count, dtype=bool),
        "d405_valid": np.ones(count, dtype=bool),
        "pose_valid": np.ones(count, dtype=bool),
        "gripper_valid": np.ones(count, dtype=bool),
        "frame_valid": np.ones(count, dtype=bool),
    })
    return replay


def make_sampler(replay):
    shape_meta = {
        "obs": {
            "camera0_rgb": {
                "type": "rgb", "horizon": 2, "latency_steps": 0,
                "down_sample_steps": 3,
            },
            "camera1_rgb": {
                "type": "rgb", "horizon": 2, "latency_steps": 0,
                "down_sample_steps": 3,
            },
            "robot0_eef_pos": {
                "type": "low_dim", "horizon": 2, "latency_steps": 0,
                "down_sample_steps": 3,
            },
            "robot0_eef_rot_axis_angle": {
                "type": "low_dim", "horizon": 2, "latency_steps": 0,
                "down_sample_steps": 3,
            },
            "robot0_gripper_width": {
                "type": "low_dim", "horizon": 2, "latency_steps": 0,
                "down_sample_steps": 3,
            },
        },
        "action": {"horizon": 16, "latency_steps": 0, "down_sample_steps": 3},
    }
    rgb_keys = ["camera0_rgb", "camera1_rgb"]
    lowdim_keys = [
        "robot0_eef_pos",
        "robot0_eef_rot_axis_angle",
        "robot0_gripper_width",
    ]
    return SequenceSampler(
        shape_meta=shape_meta,
        replay_buffer=replay,
        rgb_keys=rgb_keys,
        lowdim_keys=lowdim_keys,
        key_horizon={
            **{key: shape_meta["obs"][key]["horizon"]
               for key in shape_meta["obs"]},
            "action": 16,
        },
        key_latency_steps={
            **{key: shape_meta["obs"][key]["latency_steps"]
               for key in shape_meta["obs"]},
            "action": 0,
        },
        key_down_sample_steps={
            **{key: shape_meta["obs"][key]["down_sample_steps"]
               for key in shape_meta["obs"]},
            "action": 3,
        },
    )


class SequenceSamplerValidMaskTest(unittest.TestCase):
    def test_future_image_gap_does_not_reject_current_sample(self):
        replay = make_buffer()
        replay["d405_valid"][21] = False
        sampler = make_sampler(replay)
        sampled_current = {item[0] for item in sampler.indices}
        self.assertIn(3, sampled_current)
        sample = sampler.sample_sequence(
            next(i for i, item in enumerate(sampler.indices)
                 if item[0] == 3))
        self.assertTrue(np.all(np.isfinite(sample["action"])))
        self.assertEqual(sample["camera1_rgb"].shape,
                         (2, 2, 2, 3))

    def test_invalid_observation_rejects_sample(self):
        replay = make_buffer()
        replay["fisheye_valid"][3] = False
        sampler = make_sampler(replay)
        sampled_current = {item[0] for item in sampler.indices}
        self.assertNotIn(3, sampled_current)
        self.assertNotIn(6, sampled_current)
        self.assertIn(7, sampled_current)

    def test_invalid_action_label_rejects_sample(self):
        replay = make_buffer()
        replay["pose_valid"][21] = False
        sampler = make_sampler(replay)
        sampled_current = {item[0] for item in sampler.indices}
        self.assertNotIn(3, sampled_current)
        self.assertIn(4, sampled_current)

    def test_sampled_training_windows_are_finite(self):
        replay = make_buffer()
        sentinel = np.nan
        replay["robot0_eef_pos"][10] = sentinel
        replay["pose_valid"][10] = False
        sampler = make_sampler(replay)
        for index in range(len(sampler)):
            sample = sampler.sample_sequence(index)
            self.assertTrue(np.all(np.isfinite(sample["action"])))
            self.assertTrue(np.all(np.isfinite(
                sample["robot0_eef_pos"])))


if __name__ == "__main__":
    unittest.main()
