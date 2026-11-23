"""Quality contracts for fixed-rate UMI exports; never delete or retime rows."""

import numpy as np


def check_timeline(replay_buffer, frequency, tolerance=1e-4):
    timestamp = np.asarray(replay_buffer['timestamp'][:])
    ends = np.asarray(replay_buffer.episode_ends, dtype=np.int64)
    if ends.ndim != 1 or not len(ends) or np.any(np.diff(np.r_[0, ends]) < 2):
        raise ValueError('episode_ends must describe episodes with at least two frames')
    if timestamp.shape != (int(ends[-1]),) or not np.isfinite(timestamp).all():
        raise ValueError('timestamp must contain one finite value per stored frame')
    if not np.isfinite(frequency) or frequency <= 0 or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('frequency must be positive and tolerance nonnegative, both finite')
    start = 0
    deltas = []
    for episode, end in enumerate(ends):
        delta = np.diff(timestamp[start:end])
        if np.any(np.abs(delta - 1.0 / frequency) > tolerance):
            raise ValueError(f'episode {episode} is not fixed {frequency:g} Hz (tolerance {tolerance:g} s)')
        deltas.append(delta)
        start = end
    return np.concatenate(deltas)


def build_validity_masks(replay_buffer, rgb_keys, lowdim_keys, camera_masks,
                         frequency=30.0, timestamp_tolerance=1e-4):
    """Separate observed RGB validity from future action-label validity."""
    check_timeline(replay_buffer, frequency, timestamp_tolerance)
    n = int(replay_buffer.episode_ends[-1])

    def mask(name):
        array = np.asarray(replay_buffer[name][:])
        if array.shape != (n,) or array.dtype != np.bool_:
            raise ValueError(f'{name} must be bool [{n}]')
        return array

    def finite(key):
        array = np.asarray(replay_buffer[key][:])
        if len(array) != n:
            raise ValueError(f'{key} frame count mismatch')
        return np.isfinite(array).reshape(n, -1).all(axis=1)

    pose_valid, gripper_valid = mask('pose_valid'), mask('gripper_valid')
    masks = {}
    for key in rgb_keys:
        if key not in camera_masks:
            raise ValueError(f'No camera validity field configured for {key}')
        if replay_buffer[key].shape != (n, 224, 224, 3) or replay_buffer[key].dtype != np.uint8:
            raise ValueError(f'{key} must be uint8 [{n},224,224,3]')
        masks[key] = mask(camera_masks[key])
    for key in lowdim_keys:
        if '_eef_' in key:
            masks[key] = finite(key) & pose_valid
        elif key.endswith('_gripper_width'):
            masks[key] = finite(key) & gripper_valid
        else:
            # Episode reference poses must also be finite in sampled history.
            masks[key] = finite(key)
    action_valid = np.ones(n, dtype=bool)
    robot_keys = [key for key in lowdim_keys if key.endswith('_eef_pos')]
    for key in robot_keys:
        prefix = key.split('_')[0]
        for suffix in ('eef_pos', 'eef_rot_axis_angle', 'gripper_width'):
            action_valid &= masks[f'{prefix}_{suffix}']
    if not robot_keys:
        raise ValueError('No robot state found for action validity')
    if 'action' in replay_buffer:
        action_valid &= finite('action')
    masks['action'] = action_valid
    return masks


def filter_sequence_indices(indices, masks, rgb_keys, lowdim_keys, horizons, strides):
    """Require complete history and continuous reliable labels through the horizon.

    RGB is checked only on the history interval, never on future action frames.
    State is checked even between stride-3 labels, so gaps cannot be bridged.
    Original episode bounds and the original time axis are retained.
    """
    prefixes = {key: np.r_[0, np.cumsum(~value)] for key, value in masks.items()}

    def valid(key, first, last):
        return prefixes[key][last + 1] == prefixes[key][first]

    kept = []
    for index in indices:
        current, start, end, _ = index
        history_start = current
        acceptable = True
        for key in rgb_keys + lowdim_keys:
            first = max(start, current - (horizons[key] - 1) * strides[key])
            history_start = min(history_start, first)
            if not valid(key, first, current):
                acceptable = False
                break
        last = min(end - 1, current + (horizons['action'] - 1) * strides['action'])
        if acceptable and valid('action', history_start, last):
            kept.append(index)
    return kept
