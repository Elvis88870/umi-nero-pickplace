"""Observation adapter for the separate Thor/NERO hardware gateway.

This module only validates and preprocesses observations; it sends no commands.
The gateway must provide synchronized RGB (not BGR/depth) and robot histories
on the same 10 Hz sampling schedule used in training.
"""

import numpy as np

from umi.real_world.real_inference_util import get_real_umi_obs_dict


def make_thor_observation(env_obs, shape_meta, episode_start_pose):
    """Prepare all RGB keys declared by the checkpoint, failing on missing data.

    Dual-camera checkpoints require camera0_rgb (fisheye) and camera1_rgb
    (D405 color), each THWC uint8. Pose inputs are metres / rotation vectors in
    radians; gripper width is metres. The caller must reject invalid sensor
    histories before invoking this adapter, rather than filling them with zeros.
    """
    required = {'robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_gripper_width'}
    required.update(key for key, value in shape_meta['obs'].items() if value.get('type') == 'rgb')
    missing = required - set(env_obs)
    if missing:
        raise ValueError(f'Missing Thor observations: {sorted(missing)}')
    for key in required:
        array = np.asarray(env_obs[key])
        horizon = int(shape_meta['obs'][key]['horizon'])
        if len(array) != horizon or not np.isfinite(array).all():
            raise ValueError(f'Invalid {key}: expected {horizon} finite history frames')
        if key.endswith('_rgb'):
            if array.dtype != np.uint8 or array.ndim != 4 or array.shape[-1] != 3:
                raise ValueError(f'{key} must be THWC uint8 RGB')
        else:
            width = 1 if key.endswith('gripper_width') else 3
            if array.shape != (horizon, width):
                raise ValueError(f'{key} must have shape {(horizon, width)}')
    start_pose = np.asarray(episode_start_pose)
    if start_pose.shape != (6,) or not np.isfinite(start_pose).all():
        raise ValueError('episode_start_pose must be a finite 6D pose')
    return get_real_umi_obs_dict(
        env_obs, shape_meta, obs_pose_repr='relative',
        episode_start_pose=[start_pose],
    )
