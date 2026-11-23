#!/usr/bin/env python3
"""Evaluate a single-arm relative-pose checkpoint on held-out episodes.

No robot connection, no pretrained download, no Rerun dependency. Outputs can
be converted to Rerun recordings in a separate environment with render_eval_rerun.py.
"""

import argparse
import copy
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

from umi.common.pose_util import mat_to_pose10d, pose10d_to_mat, pose_to_mat
from umi.real_world.real_inference_util import get_real_umi_obs_dict


def action_metrics(predicted, target):
    """Errors per horizon step; rotations use SO(3) geodesic distance."""
    # The upstream rotation helper accepts a single leading dimension.
    shape = predicted.shape[:-1]
    pred_rot = pose10d_to_mat(predicted[..., :9].reshape(-1, 9))[:, :3, :3].reshape(shape + (3, 3))
    target_rot = pose10d_to_mat(target[..., :9].reshape(-1, 9))[:, :3, :3].reshape(shape + (3, 3))
    relative = pred_rot @ np.swapaxes(target_rot, -1, -2)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return {
        "position_mm": np.linalg.norm(predicted[..., :3] - target[..., :3], axis=-1) * 1000,
        "rotation_deg": np.rad2deg(np.arccos(cosine)),
        "gripper_mm": np.abs(predicted[..., 9] - target[..., 9]) * 1000,
    }


def summarize(errors):
    return {
        key: {
            "mean": float(value.mean()),
            "median": float(np.median(value)),
            "p95": float(np.quantile(value, 0.95)),
            "by_horizon_mean": value.mean(axis=0).tolist(),
        }
        for key, value in errors.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path, help="Defaults to the dataset saved in the checkpoint")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples-per-episode", type=int, default=10,
                        help="Evenly spaced valid windows; 0 evaluates every valid window")
    parser.add_argument("--max-episodes", type=int, default=0, help="0 evaluates every held-out episode")
    parser.add_argument("--seed", type=int, default=42, help="Inference randomness only; split comes from checkpoint")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.samples_per_episode < 0 or args.max_episodes < 0 or args.cpu_threads < 1:
        parser.error("sample and episode limits must be nonnegative; cpu-threads must be positive")
    if not args.checkpoint.is_file() or (args.dataset is not None and not args.dataset.is_file()):
        parser.error("checkpoint and dataset must be existing files")
    output = args.output_dir or args.checkpoint.parent.parent / "offline_eval" / args.checkpoint.stem
    # Prevent accidental mixing or overwriting of evaluations from different runs.
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Run on the GPU host, or use --device cpu.")

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    payload = torch.load(args.checkpoint, map_location="cpu", pickle_module=dill)
    cfg = copy.deepcopy(payload["cfg"])
    args.dataset = args.dataset or Path(cfg.task.dataset.dataset_path)
    if not args.dataset.is_file():
        parser.error("Checkpoint dataset not found; pass --dataset explicitly")
    if cfg.task.dataset._target_ != "diffusion_policy.dataset.umi_dataset.UmiDataset":
        raise ValueError("Only UmiDataset is supported")
    if list(cfg.task.shape_meta.action.shape) != [10]:
        raise ValueError("Only single-arm 10D actions are supported")
    if any(cfg.task.pose_repr[key] != "relative" for key in ("obs_pose_repr", "action_pose_repr")):
        raise ValueError("This evaluator requires relative observation and action poses")

    # Construct only the policy, without optimizer, workspace or pretrained fetch.
    policy_cfg = copy.deepcopy(cfg.policy)
    policy_cfg.obs_encoder.pretrained = False
    policy_cfg.obs_encoder.frozen = False
    state_key = "ema_model" if cfg.training.use_ema else "model"
    print(f"Loading {state_key} from {args.checkpoint}", flush=True)
    policy = hydra.utils.instantiate(policy_cfg)
    policy.load_state_dict(payload["state_dicts"][state_key], strict=True)
    policy.eval().to(device)
    del payload

    # Keep the checkpoint's original episode split and sampling configuration.
    dataset_cfg = copy.deepcopy(cfg.task.dataset)
    dataset_cfg.dataset_path = str(args.dataset.resolve())
    dataset_cfg.cache_dir = None
    dataset_cfg.repeat_frame_prob = 0.0
    dataset = hydra.utils.instantiate(dataset_cfg)
    validation = dataset.get_validation_dataset()
    ends = np.asarray(dataset.replay_buffer.episode_ends)
    held_out = np.flatnonzero(dataset.val_mask).tolist()
    selected = held_out[:args.max_episodes] if args.max_episodes else held_out
    by_episode = {episode: [] for episode in selected}
    for sample_index, (current, start, end, _) in enumerate(validation.sampler.indices):
        episode = int(np.searchsorted(ends, current, side="right"))
        if episode in by_episode:
            by_episode[episode].append(sample_index)
    if not selected or any(not indices for indices in by_episode.values()):
        raise ValueError("No valid action windows in one or more selected validation episodes")
    print(f"Held-out episodes (zero-based): {held_out}; evaluating: {selected}", flush=True)

    raw = dataset.replay_buffer
    stride = int(cfg.task.shape_meta.action.down_sample_steps)
    horizon = int(cfg.task.shape_meta.action.horizon)
    episode_reports = []
    for episode, candidates in by_episode.items():
        count = min(args.samples_per_episode or len(candidates), len(candidates))
        indices = [candidates[i] for i in np.linspace(0, len(candidates) - 1, count, dtype=int)]
        start = 0 if episode == 0 else int(ends[episode - 1])
        end = int(ends[episode])
        time_values = np.asarray(raw["timestamp"][start:end]) if "timestamp" in raw else np.arange(end-start) / float(cfg.task.dataset_frequeny)
        time_values = time_values - time_values[0]
        images, predictions, targets, baselines, transforms = [], [], [], [], []
        camera_images = {key: [] for key in dataset.rgb_keys}
        frame_indices, obs_times, future_times, latencies = [], [], [], []
        for sample_index in indices:
            current, _, _, _ = validation.sampler.indices[sample_index]
            sampled = validation.sampler.sample_sequence(sample_index)
            # Match deployment preprocessing rather than UmiDataset.__getitem__,
            # which perturbs the episode start pose even for validation samples.
            env_obs = {key: sampled[key] for key in dataset.rgb_keys + [
                "robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_gripper_width"
            ]}
            start_pose = np.asarray(raw["robot0_demo_start_pose"][start]).copy()
            obs = get_real_umi_obs_dict(
                env_obs, cfg.task.shape_meta, obs_pose_repr="relative",
                episode_start_pose=[start_pose],
            )
            tensors = {key: torch.as_tensor(value, dtype=torch.float32, device=device).unsqueeze(0)
                       for key, value in obs.items()}
            # Sample-specific seeds permit repeatable subset comparisons.
            random.seed(args.seed + sample_index)
            np.random.seed(args.seed + sample_index)
            torch.manual_seed(args.seed + sample_index)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tick = time.perf_counter()
            with torch.inference_mode():
                prediction = policy.predict_action(tensors)["action_pred"][0].cpu().numpy()
            elapsed_ms = (time.perf_counter() - tick) * 1000
            if prediction.shape != (horizon, 10) or not np.isfinite(prediction).all():
                raise ValueError(f"Invalid prediction at validation index {sample_index}")
            base = pose_to_mat(np.concatenate([
                sampled["robot0_eef_pos"][-1], sampled["robot0_eef_rot_axis_angle"][-1]
            ]))
            target_world = pose_to_mat(sampled["action"][..., :6])
            target_relative = np.linalg.inv(base) @ target_world
            target = np.concatenate([mat_to_pose10d(target_relative), sampled["action"][..., 6:7]], axis=-1)
            predicted_world = base @ pose10d_to_mat(prediction[:, :9])
            # The hold-current-pose baseline shares the same frame and horizon.
            baseline = np.concatenate([
                np.tile(mat_to_pose10d(np.eye(4)), (horizon, 1)),
                np.full((horizon, 1), sampled["robot0_gripper_width"][-1, 0]),
            ], axis=-1)
            images.append(sampled["camera0_rgb"][-1])
            for key in camera_images:
                camera_images[key].append(sampled[key][-1])
            predictions.append(prediction)
            targets.append(target)
            baselines.append(baseline)
            transforms.append(predicted_world)
            frame_indices.append(int(current))
            obs_times.append(time_values[current-start])
            future_indices = np.minimum(current - start + np.arange(horizon) * stride, end-start-1)
            future_times.append(time_values[future_indices] - time_values[current-start])
            latencies.append(elapsed_ms)
        predictions, targets, baselines = map(np.stack, (predictions, targets, baselines))
        errors = action_metrics(predictions, targets)
        baseline_errors = action_metrics(baselines, targets)
        gt_world = np.stack([
            pose_to_mat(validation.sampler.sample_sequence(i)["action"][:, :6]) for i in indices
        ])
        np.savez_compressed(
            output / f"episode_{episode:03d}.npz", episode=np.array(episode),
            frame_index=np.array(frame_indices), timestamp=np.array(obs_times),
            future_time_offsets=np.array(future_times), images=np.stack(images),
            prediction=predictions, ground_truth=targets,
            predicted_world=np.stack(transforms), ground_truth_world=gt_world,
            inference_ms=np.array(latencies), **errors,
            **{key: np.stack(value) for key, value in camera_images.items()},
        )
        report = {"episode": episode, "samples": len(indices),
                  "model": summarize(errors), "hold_baseline": summarize(baseline_errors)}
        episode_reports.append(report)
        print(f"Episode {episode}: {len(indices)} windows, "
              f"position={errors['position_mm'].mean():.2f} mm, "
              f"rotation={errors['rotation_deg'].mean():.2f} deg, "
              f"gripper={errors['gripper_mm'].mean():.2f} mm", flush=True)

    # Equal weighting of episodes, not weighting by their lengths.
    overall = {}
    for group in ("model", "hold_baseline"):
        overall[group] = {
            metric: float(np.mean([episode[group][metric]["mean"] for episode in episode_reports]))
            for metric in ("position_mm", "rotation_deg", "gripper_mm")
        }
    summary = {
        "checkpoint": str(args.checkpoint.resolve()), "dataset": str(args.dataset.resolve()),
        "weights": state_key, "device": str(device), "inference_seed": args.seed,
        "split_seed": int(dataset_cfg.seed), "val_ratio": float(dataset_cfg.val_ratio),
        "held_out_episode_ids": held_out, "evaluated_episode_ids": selected,
        "samples": sum(episode["samples"] for episode in episode_reports),
        "sampling": "evenly spaced valid windows per episode; 0-based episode IDs",
        "preprocessing": "deployment observations; no training start-pose noise",
        "aggregation": "episode means weighted equally; horizon step 0 includes current frame",
        "timing": "model inference plus CPU action transfer, includes cold first inference; not end-to-end latency",
        "overall": overall, "episodes": episode_reports,
        "limitation": "teacher-forced offline comparison; does not measure closed-loop task success",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(overall, indent=2))
    print(f"OFFLINE EVALUATION COMPLETE: {output}")


if __name__ == "__main__":
    main()
