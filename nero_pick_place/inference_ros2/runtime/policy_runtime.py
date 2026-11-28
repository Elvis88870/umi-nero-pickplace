"""Checkpoint loading and the original UMI observation/action runtime."""

from __future__ import annotations

import copy
from pathlib import Path
import time
from typing import Any, Dict, Tuple

import numpy as np


def deterministic_camera_transforms(encoder):
    """Replace training-only image randomness in this deployment instance."""
    import torch.nn as nn
    from torchvision import transforms as tv

    def convert(transform):
        if isinstance(transform, nn.Sequential):
            return nn.Sequential(*(convert(child) for child in transform.children()))
        if isinstance(transform, tv.RandomCrop):
            if (transform.padding is not None or transform.pad_if_needed):
                raise ValueError("deployment does not support padded random crops")
            return tv.CenterCrop(transform.size)
        if isinstance(transform, tv.ColorJitter):
            return nn.Identity()
        if isinstance(transform, (tv.Resize, tv.CenterCrop, tv.Normalize, nn.Identity)):
            return copy.deepcopy(transform)
        raise ValueError(
            "unsupported deployment image transform: "
            f"{type(transform).__module__}.{type(transform).__name__}; "
            "add an explicit deterministic conversion before deployment")

    descriptions = {}
    for key in encoder.rgb_keys:
        transform = convert(encoder.key_transform_map[key])
        encoder.key_transform_map[key] = transform
        descriptions[key] = str(transform)
    return descriptions


class PolicyRuntime:
    """Load an EMA diffusion policy and expose UMI-compatible inference."""

    def __init__(
        self,
        checkpoint: Path,
        config: Dict[str, Any],
        device_name: str,
        num_inference_steps: int,
    ) -> None:
        import dill
        import hydra
        import torch
        from omegaconf import OmegaConf

        self.torch = torch
        self.device = torch.device(device_name)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        payload = torch.load(
            checkpoint, map_location="cpu", pickle_module=dill)
        self.cfg = copy.deepcopy(payload["cfg"])
        shape_meta = self.cfg.task.shape_meta

        if "camera0_rgb" not in shape_meta.obs:
            raise ValueError("checkpoint does not expect camera0_rgb")
        if shape_meta.obs["camera0_rgb"].get("type") != "rgb":
            raise ValueError("camera0_rgb is not configured as RGB")
        self.camera_keys = ["camera0_rgb"]
        if "camera1_rgb" in shape_meta.obs:
            if shape_meta.obs["camera1_rgb"].get("type") != "rgb":
                raise ValueError("camera1_rgb is not configured as RGB")
            self.camera_keys.append("camera1_rgb")
        config["observation"]["include_d405"] = (
            "camera1_rgb" in self.camera_keys)

        if list(shape_meta.action.shape) != [10]:
            raise ValueError(
                f"expected one 10-D UMI action, got {shape_meta.action.shape}")
        if not bool(self.cfg.training.use_ema):
            raise ValueError("checkpoint was not trained with EMA")
        if "ema_model" not in payload["state_dicts"]:
            raise ValueError("checkpoint has no EMA model weights")

        observation = config["observation"]
        action = config["action"]
        for camera_key in self.camera_keys:
            model_horizon = int(shape_meta.obs[camera_key].horizon)
            if model_horizon != int(observation["history_horizon"]):
                raise ValueError(
                    f"{camera_key} horizon {model_horizon} does not match "
                    f"config history_horizon {observation['history_horizon']}")
        if int(shape_meta.action.horizon) != int(action["action_horizon"]):
            raise ValueError(
                f"checkpoint action horizon {shape_meta.action.horizon} "
                f"does not match config action_horizon "
                f"{action['action_horizon']}")

        down_sample = shape_meta.action.down_sample_steps
        frequency = self.cfg.task.get("dataset_frequency", None)
        if frequency is not None:
            expected_interval = float(down_sample) / float(frequency)
            configured = float(action["action_interval_sec"])
            if abs(expected_interval - configured) > 1e-6:
                raise ValueError(
                    f"action interval {configured} does not match checkpoint "
                    f"stride {down_sample}/{frequency}={expected_interval}")
            for camera_key in self.camera_keys:
                camera_stride = float(
                    shape_meta.obs[camera_key].down_sample_steps)
                expected_observation_interval = (
                    camera_stride / float(frequency))
                configured_observation_interval = float(
                    observation["history_interval_sec"])
                if abs(
                    expected_observation_interval
                    - configured_observation_interval
                ) > 1e-6:
                    raise ValueError(
                        f"observation interval "
                        f"{configured_observation_interval} does not match "
                        f"{camera_key} stride {camera_stride}/{frequency}="
                        f"{expected_observation_interval}")

        policy_cfg = copy.deepcopy(self.cfg.policy)
        if "pretrained" in policy_cfg.obs_encoder:
            policy_cfg.obs_encoder.pretrained = False
        if "frozen" in policy_cfg.obs_encoder:
            policy_cfg.obs_encoder.frozen = False
        self.policy = hydra.utils.instantiate(policy_cfg)
        self.policy.load_state_dict(
            payload["state_dicts"]["ema_model"], strict=True)
        self.camera_transform_descriptions = deterministic_camera_transforms(
            self.policy.obs_encoder)
        self.policy.num_inference_steps = int(num_inference_steps)
        self.policy.eval().to(self.device)
        self.policy.reset()
        self.shape_meta = shape_meta
        self.obs_pose_repr = str(self.cfg.task.pose_repr.obs_pose_repr)
        self.action_pose_repr = str(self.cfg.task.pose_repr.action_pose_repr)
        if self.obs_pose_repr != "relative":
            raise ValueError("this deployment expects relative observations")
        if self.action_pose_repr != "relative":
            raise ValueError("this deployment expects relative actions")
        print(
            f"[model] checkpoint={checkpoint} weights=ema_model "
            f"device={self.device} steps={num_inference_steps}",
            flush=True)
        for key, description in self.camera_transform_descriptions.items():
            print(f"[model] {key} deterministic preprocessing: "
                  f"{description.replace(chr(10), ' ')}", flush=True)

    def infer(
        self,
        env_obs: Dict[str, np.ndarray],
        episode_start_pose: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        from umi.real_world.real_inference_util import (
            get_real_umi_action,
            get_real_umi_obs_dict,
        )

        obs_np = get_real_umi_obs_dict(
            env_obs=env_obs,
            shape_meta=self.shape_meta,
            obs_pose_repr=self.obs_pose_repr,
            episode_start_pose=[episode_start_pose],
        )
        tensors = {
            key: self.torch.as_tensor(
                value, dtype=self.torch.float32, device=self.device
            ).unsqueeze(0)
            for key, value in obs_np.items()
        }
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            raw = self.policy.predict_action(tensors)["action_pred"][0]
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        raw_np = raw.detach().cpu().numpy()
        if raw_np.shape != (int(self.shape_meta.action.horizon), 10):
            raise ValueError(f"unexpected action shape: {raw_np.shape}")
        if not np.all(np.isfinite(raw_np)):
            raise ValueError("policy returned non-finite actions")
        action = get_real_umi_action(
            raw_np, env_obs, self.action_pose_repr)
        if action.shape != (int(self.shape_meta.action.horizon), 7):
            raise ValueError(f"unexpected 7-D action shape: {action.shape}")
        return np.asarray(action, dtype=np.float64), elapsed_ms
