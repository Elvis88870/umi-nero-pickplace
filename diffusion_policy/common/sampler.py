from typing import Optional
import numpy as np
import random
import scipy.spatial.transform as st
from diffusion_policy.common.replay_buffer import ReplayBuffer

def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


class SequenceSampler:
    def __init__(self,
        shape_meta: dict,
        replay_buffer: ReplayBuffer,
        rgb_keys: list,
        lowdim_keys: list,
        key_horizon: dict,
        key_latency_steps: dict,
        key_down_sample_steps: dict,
        episode_mask: Optional[np.ndarray]=None,
        action_padding: bool=False,
        repeat_frame_prob: float=0.0,
        max_duration: Optional[float]=None,
        validity_masks: Optional[dict]=None
    ):
        episode_ends = replay_buffer.episode_ends[:]
        self.valid_masks = {}
        for key in (
            "fisheye_valid", "d405_valid", "pose_valid",
            "gripper_valid", "frame_valid",
        ):
            if key in replay_buffer:
                self.valid_masks[key] = np.asarray(
                    replay_buffer[key][:], dtype=bool)

        # load gripper_width
        gripper_width = replay_buffer['robot0_gripper_width'][:, 0]
        gripper_width_threshold = 0.08
        self.repeat_frame_prob = repeat_frame_prob

        # create indices, including (current_idx, start_idx, end_idx)
        indices = list()
        for i in range(len(episode_ends)):
            before_first_grasp = True # initialize for each episode
            if episode_mask is not None and not episode_mask[i]:
                # skip episode
                continue
            start_idx = 0 if i == 0 else episode_ends[i-1]
            end_idx = episode_ends[i]
            if max_duration is not None:
                end_idx = min(end_idx, max_duration * 60)
            for current_idx in range(start_idx, end_idx):
                if not action_padding and end_idx < current_idx + (key_horizon['action'] - 1) * key_down_sample_steps['action'] + 1:
                    continue
                if not self._training_sample_valid(
                        current_idx=current_idx,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        rgb_keys=rgb_keys,
                        lowdim_keys=lowdim_keys,
                        key_horizon=key_horizon,
                        key_latency_steps=key_latency_steps,
                        key_down_sample_steps=key_down_sample_steps,
                    ):
                    continue
                if gripper_width[current_idx] < gripper_width_threshold:
                    before_first_grasp = False
                indices.append((current_idx, start_idx, end_idx, before_first_grasp))
        
        # load low_dim to memory and keep rgb as compressed zarr array
        self.replay_buffer = dict()
        self.num_robot = 0
        for key in lowdim_keys:
            if key.endswith('eef_pos'):
                self.num_robot += 1

            if key.endswith('pos_abs'):
                axis = shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                self.replay_buffer[key] = replay_buffer[key[:-4]][:, list(axis)]
            elif key.endswith('quat_abs'):
                axis = shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                # HACK for hybrid abs/relative proprioception
                rot_in = replay_buffer[key[:-4]][:]
                rot_out = st.Rotation.from_quat(rot_in).as_euler('XYZ')
                self.replay_buffer[key] = rot_out[:, list(axis)]
            elif key.endswith('axis_angle_abs'):
                axis = shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                rot_in = replay_buffer[key[:-4]][:]
                rot_out = st.Rotation.from_rotvec(rot_in).as_euler('XYZ')
                self.replay_buffer[key] = rot_out[:, list(axis)]
            else:
                self.replay_buffer[key] = replay_buffer[key][:]
        for key in rgb_keys:
            self.replay_buffer[key] = replay_buffer[key]
        
        
        if 'action' in replay_buffer:
            self.replay_buffer['action'] = replay_buffer['action'][:]
        else:
            # construct action (concatenation of [eef_pos, eef_rot, gripper_width])
            actions = list()
            for robot_idx in range(self.num_robot):
                for cat in ['eef_pos', 'eef_rot_axis_angle', 'gripper_width']:
                    key = f'robot{robot_idx}_{cat}'
                    if key in self.replay_buffer:
                        actions.append(self.replay_buffer[key])
            self.replay_buffer['action'] = np.concatenate(actions, axis=-1)

        self.action_padding = action_padding
        self.validity_masks = validity_masks
        if validity_masks is not None:
            from diffusion_policy.common.umi_quality import filter_sequence_indices
            for key in rgb_keys + lowdim_keys + ['action']:
                if key_latency_steps[key] != 0 or int(key_down_sample_steps[key]) != key_down_sample_steps[key]:
                    raise ValueError('Quality filtering requires zero latency and integer frame strides')
            original_count = len(indices)
            indices = filter_sequence_indices(
                indices, validity_masks, rgb_keys, lowdim_keys,
                key_horizon, key_down_sample_steps)
            print(f'Quality windows: kept={len(indices)} rejected={original_count-len(indices)} candidates={original_count}', flush=True)
            if not indices:
                raise ValueError('No valid windows remain in this episode split')
        self.indices = indices
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        
        self.ignore_rgb_is_applied = False # speed up the interation when getting normalizaer

    def _key_valid_mask(self, key):
        if key == "camera0_rgb":
            mask = self.valid_masks.get("fisheye_valid")
        elif key == "camera1_rgb":
            mask = self.valid_masks.get("d405_valid")
        elif "gripper" in key:
            mask = self.valid_masks.get("gripper_valid")
        elif "eef" in key or "pose" in key:
            mask = self.valid_masks.get("pose_valid")
        else:
            mask = self.valid_masks.get("frame_valid")
        return mask

    @staticmethod
    def _mask_query_valid(mask, indices, start_idx, end_idx):
        indices = np.asarray(indices, dtype=np.float64)
        if not np.all(np.isfinite(indices)):
            return False
        lower = np.floor(indices).astype(np.int64)
        upper = np.ceil(indices).astype(np.int64)
        if lower.min() < start_idx or upper.max() >= end_idx:
            return False
        return bool(np.all(mask[lower]) and np.all(mask[upper]))

    def _training_sample_valid(
            self,
            current_idx,
            start_idx,
            end_idx,
            rgb_keys,
            lowdim_keys,
            key_horizon,
            key_latency_steps,
            key_down_sample_steps,
        ):
        if not self.valid_masks:
            return True

        for key in list(rgb_keys) + list(lowdim_keys):
            mask = self._key_valid_mask(key)
            if mask is None:
                continue
            horizon = key_horizon[key]
            latency = key_latency_steps[key]
            step = key_down_sample_steps[key]
            if key in rgb_keys:
                query_indices = np.asarray([
                    current_idx - idx * step for idx in range(horizon)
                ], dtype=np.float64)
            else:
                query_indices = np.asarray([
                    current_idx - idx * step + latency
                    for idx in range(horizon)
                ], dtype=np.float64)
            if not self._mask_query_valid(
                    mask, query_indices, start_idx, end_idx):
                return False

        pose_mask = self._key_valid_mask("robot0_eef_pos")
        gripper_mask = self._key_valid_mask("robot0_gripper_width")
        action_indices = np.asarray([
            current_idx + idx * key_down_sample_steps["action"]
            for idx in range(key_horizon["action"])
        ], dtype=np.int64)
        if action_indices[-1] >= end_idx:
            return False
        if pose_mask is not None and not np.all(pose_mask[action_indices]):
            return False
        if gripper_mask is not None and not np.all(gripper_mask[action_indices]):
            return False
        return True

    def __len__(self):
        return len(self.indices)
    
    def sample_sequence(self, idx):
        current_idx, start_idx, end_idx, before_first_grasp = self.indices[idx]

        result = dict()

        obs_keys = self.rgb_keys + self.lowdim_keys
        if self.ignore_rgb_is_applied:
            obs_keys = self.lowdim_keys

        # observation
        for key in obs_keys:
            input_arr = self.replay_buffer[key]
            this_horizon = self.key_horizon[key]
            this_latency_steps = self.key_latency_steps[key]
            this_downsample_steps = self.key_down_sample_steps[key]
            
            if key in self.rgb_keys:
                assert this_latency_steps == 0
                num_valid = min(this_horizon, (current_idx - start_idx) // this_downsample_steps + 1)
                slice_start = current_idx - (num_valid - 1) * this_downsample_steps

                output = input_arr[slice_start: current_idx + 1: this_downsample_steps]
                assert output.shape[0] == num_valid
                
                # solve padding
                if output.shape[0] < this_horizon:
                    padding = np.repeat(output[:1], this_horizon - output.shape[0], axis=0)
                    output = np.concatenate([padding, output], axis=0)
            elif self.validity_masks is not None:
                # Fixed-rate, zero-latency export: use exactly the stored states.
                # Do not construct Slerp from an unrelated +/-5-frame buffer,
                # which may contain NaNs outside this accepted window.
                frame_indices = np.maximum(
                    start_idx, current_idx - np.arange(this_horizon)[::-1] * this_downsample_steps)
                output = input_arr[frame_indices].copy()
            else:
                idx_with_latency = np.array(
                    [current_idx - idx * this_downsample_steps + this_latency_steps for idx in range(this_horizon)],
                    dtype=np.float32)
                idx_with_latency = idx_with_latency[::-1]
                idx_with_latency = np.clip(idx_with_latency, start_idx, end_idx - 1)
                lower = np.floor(idx_with_latency).astype(np.int64)
                upper = np.ceil(idx_with_latency).astype(np.int64)
                alpha = idx_with_latency - lower

                if 'rot' in key:
                    # rotation
                    rot_preprocess, rot_postprocess = None, None
                    if key.endswith('quat'):
                        rot_preprocess = st.Rotation.from_quat
                        rot_postprocess = st.Rotation.as_quat
                    elif key.endswith('axis_angle'):
                        rot_preprocess = st.Rotation.from_rotvec
                        rot_postprocess = st.Rotation.as_rotvec
                    else:
                        raise NotImplementedError
                    rotations = []
                    for query, lo, hi, weight in zip(
                            idx_with_latency, lower, upper, alpha):
                        if lo == hi:
                            rotations.append(
                                rot_preprocess(input_arr[int(lo)]))
                            continue
                        slerp = st.Slerp(
                            times=np.asarray([lo, hi], dtype=np.float64),
                            rotations=rot_preprocess(input_arr[[lo, hi]]),
                        )
                        rotations.append(slerp([float(query)]))
                    output = rot_postprocess(
                        st.Rotation.concatenate(rotations))
                else:
                    output = (
                        (1.0 - alpha[:, None]) * input_arr[lower]
                        + alpha[:, None] * input_arr[upper]
                    )
                
            result[key] = output

        # repeat frame before first grasp
        if self.repeat_frame_prob != 0.0:
            if before_first_grasp and random.random() < self.repeat_frame_prob:
                for key in obs_keys:
                    result[key][:-1] = result[key][-1:]

        # aciton
        input_arr = self.replay_buffer['action']
        action_horizon = self.key_horizon['action']
        action_latency_steps = self.key_latency_steps['action']
        assert action_latency_steps == 0
        action_down_sample_steps = self.key_down_sample_steps['action']
        slice_end = min(end_idx, current_idx + (action_horizon - 1) * action_down_sample_steps + 1)
        output = input_arr[current_idx: slice_end: action_down_sample_steps]
        # solve padding
        if not self.action_padding:
            assert output.shape[0] == action_horizon
        elif output.shape[0] < action_horizon:
            padding = np.repeat(output[-1:], action_horizon - output.shape[0], axis=0)
            output = np.concatenate([output, padding], axis=0)
        result['action'] = output

        return result
    
    def ignore_rgb(self, apply=True):
        self.ignore_rgb_is_applied = apply
