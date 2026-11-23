"""Quality filtering must preserve time, label reliability and episode splits."""

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
import zarr
from hydra import compose, initialize_config_dir

from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.umi_quality import build_validity_masks
from diffusion_policy.dataset.umi_dataset import UmiDataset
from nero_pick_place.inference import make_thor_observation


class Buffer(dict):
    episode_ends = np.array([90, 180])


def fixture():
    n = 180
    data = Buffer(timestamp=np.tile(np.arange(90) / 30, 2))
    for key in ('fisheye_valid', 'd405_valid', 'pose_valid', 'gripper_valid', 'frame_valid'):
        data[key] = np.ones(n, bool)
    # Broadcasted arrays keep the unit fixture small.
    for camera in ('camera0_rgb', 'camera1_rgb'):
        data[camera] = np.broadcast_to(np.zeros((1, 224, 224, 3), np.uint8), (n, 224, 224, 3))
    for key, width in [('eef_pos', 3), ('eef_rot_axis_angle', 3), ('gripper_width', 1),
                       ('demo_start_pose', 6), ('demo_end_pose', 6)]:
        data[f'robot0_{key}'] = np.zeros((n, width), np.float32)
    data['robot0_eef_pos'][:, 0] = np.arange(n) * 0.001
    data['robot0_gripper_width'][:] = 0.05
    return data


RGB = ['camera0_rgb', 'camera1_rgb']
LOW = ['robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_gripper_width',
       'robot0_demo_start_pose', 'robot0_demo_end_pose']
CAMERA_MASKS = {'camera0_rgb': 'fisheye_valid', 'camera1_rgb': 'd405_valid'}


def shape_meta():
    obs = {}
    for key in RGB + LOW[:3]:
        is_rgb = key in RGB
        obs[key] = {'shape': [3, 224, 224] if is_rgb else ([1] if 'gripper' in key else [3]),
                    'type': 'rgb' if is_rgb else 'low_dim', 'horizon': 2,
                    'latency_steps': 0, 'down_sample_steps': 3}
    return {'obs': obs, 'action': {'shape': [10], 'horizon': 16,
                                  'latency_steps': 0, 'down_sample_steps': 3}}


def sampler(data):
    masks = build_validity_masks(data, RGB, LOW, CAMERA_MASKS)
    keys = RGB + LOW + ['action']
    return SequenceSampler(shape_meta(), data, RGB, LOW,
                           {k: 16 if k == 'action' else 2 for k in keys},
                           {k: 0 for k in keys}, {k: 3 for k in keys},
                           validity_masks=masks)


class QualityTest(unittest.TestCase):
    def test_single_camera_ignores_d405_and_aggregate_mask(self):
        data = fixture()
        data.pop('camera1_rgb')
        data['d405_valid'][:] = False
        data['frame_valid'][:] = False
        meta = shape_meta()
        meta['obs'].pop('camera1_rgb')
        rgb = ['camera0_rgb']
        masks = build_validity_masks(data, rgb, LOW, {'camera0_rgb': 'fisheye_valid'})
        keys = rgb + LOW + ['action']
        samples = SequenceSampler(meta, data, rgb, LOW,
                                  {k: 16 if k == 'action' else 2 for k in keys},
                                  {k: 0 for k in keys}, {k: 3 for k in keys},
                                  validity_masks=masks)
        self.assertEqual(len(samples), 90)
        raw = samples.sample_sequence(0)
        self.assertNotIn('camera1_rgb', raw)
        obs = make_thor_observation(raw, meta, np.zeros(6))
        self.assertEqual(obs['camera0_rgb'].shape, (2, 3, 224, 224))
        self.assertNotIn('camera1_rgb', obs)
        # D405 fields need not even exist for this config.
        data.pop('d405_valid')
        data.pop('frame_valid')
        build_validity_masks(data, rgb, LOW, {'camera0_rgb': 'fisheye_valid'})
        # Pose and gripper validity still apply to the single-camera model.
        data['robot0_eef_pos'][40] = np.nan
        masks = build_validity_masks(data, rgb, LOW, {'camera0_rgb': 'fisheye_valid'})
        self.assertFalse(masks['action'][40])

    def test_camera_configs_agree_for_dataset_and_policy(self):
        repo = Path(__file__).resolve().parents[1]
        with initialize_config_dir(config_dir=str(repo / 'nero_pick_place/config'), version_base=None):
            for name, cameras in [('pika_nero_pick_place_train', set(RGB)),
                                  ('pika_nero_pick_place_fisheye_train', {'camera0_rgb'})]:
                cfg = compose(config_name=name, overrides=[
                    f'hydra.searchpath=[file://{repo / "diffusion_policy/config"}]'])
                for meta in (cfg.task.shape_meta, cfg.shape_meta, cfg.task.dataset.shape_meta,
                             cfg.policy.shape_meta, cfg.policy.obs_encoder.shape_meta):
                    self.assertEqual({k for k, v in meta.obs.items() if v.get('type') == 'rgb'}, cameras)
                self.assertEqual(set(cfg.task.dataset.quality_filter.camera_masks), cameras)
                self.assertEqual(cfg.task.action_horizon, 16)
                self.assertEqual(cfg.task.obs_down_sample_steps, 3)

    def test_future_camera_missing_does_not_reject_label(self):
        data = fixture()
        data['d405_valid'][30] = False
        data['frame_valid'][30] = False
        samples = sampler(data)
        currents = [index[0] for index in samples.indices]
        self.assertIn(10, currents)  # Future image is unused.
        for current in range(30, 34):
            self.assertNotIn(current, currents)  # Complete history interval.
        self.assertIn(34, currents)

    def test_pose_gap_blocks_even_unsampled_label_and_no_slerp_context(self):
        data = fixture()
        data['pose_valid'][29] = False  # Not a stride-3 label of current=10.
        data['robot0_eef_rot_axis_angle'][29] = np.nan
        samples = sampler(data)
        currents = [index[0] for index in samples.indices]
        self.assertNotIn(10, currents)
        self.assertNotIn(32, currents)  # Gap in history.
        self.assertIn(33, currents)  # NaN outside history; no +/-5 Slerp.
        result = samples.sample_sequence(currents.index(33))
        self.assertTrue(all(np.isfinite(array).all() for array in result.values()))
        np.testing.assert_array_equal(result['action'][:, 0], data['robot0_eef_pos'][33:79:3, 0])

    def test_gripper_mask_and_unmarked_nan_reject_windows(self):
        data = fixture()
        data['gripper_valid'][40] = False
        data['robot0_gripper_width'][40] = 1e9  # Finite placeholder.
        self.assertNotIn(10, [index[0] for index in sampler(data).indices])
        data = fixture()
        data['robot0_eef_pos'][40] = np.nan
        self.assertNotIn(10, [index[0] for index in sampler(data).indices])

    def test_timestamp_gap_rejected(self):
        data = fixture()
        data['timestamp'][50:] += 0.1
        with self.assertRaisesRegex(ValueError, 'not fixed 30 Hz'):
            sampler(data)

    def test_no_usable_windows_rejected(self):
        data = fixture()
        data['pose_valid'][:] = False
        with self.assertRaisesRegex(ValueError, 'No valid windows'):
            sampler(data)

    def test_dataset_split_normalizer_and_thor_dual_rgb(self):
        data = fixture()
        heldout = int(np.flatnonzero(get_val_mask(2, 0.5, seed=42))[0])
        data['robot0_gripper_width'][heldout * 90:(heldout + 1) * 90] = 0.15
        for offset in (0, 90):
            data['gripper_valid'][offset + 40] = False
            data['robot0_gripper_width'][offset + 40] = 1e9
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.zarr.zip'
            with zarr.ZipStore(str(path), mode='w') as store:
                root = zarr.group(store)
                group = root.create_group('data')
                for key, array in data.items():
                    group.array(key, array, chunks=(1,) + array.shape[1:] if key in RGB else array.shape)
                root.create_group('meta').array('episode_ends', data.episode_ends)
            dataset = UmiDataset(shape_meta(), str(path), val_ratio=0.5,
                                 pose_repr={'obs_pose_repr': 'relative', 'action_pose_repr': 'relative'},
                                 quality_filter={'camera_masks': CAMERA_MASKS}, normalizer_num_workers=0)
            validation = dataset.get_validation_dataset()
            train_episodes = {int(np.searchsorted(data.episode_ends, index[0], side='right')) for index in dataset.sampler.indices}
            val_episodes = {int(np.searchsorted(data.episode_ends, index[0], side='right')) for index in validation.sampler.indices}
            self.assertFalse(train_episodes & val_episodes)
            normalizer = dataset.get_normalizer()
            self.assertTrue(all(torch.isfinite(value).all() for value in normalizer.state_dict().values()))
            self.assertAlmostEqual(float(normalizer['action'].get_input_stats()['max'][-1]), 0.05)
            raw = dataset.sampler.sample_sequence(0)
            obs = make_thor_observation(raw, shape_meta(), np.zeros(6))
            for key in RGB:
                self.assertEqual(obs[key].shape, (2, 3, 224, 224))
            raw.pop('camera1_rgb')
            with self.assertRaisesRegex(ValueError, 'camera1_rgb'):
                make_thor_observation(raw, shape_meta(), np.zeros(6))


if __name__ == '__main__':
    unittest.main()
