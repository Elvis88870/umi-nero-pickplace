#!/usr/bin/env python3
"""Read-only schema, quality-window and timing checks for configured Pika RGB."""

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import zarr


REQUIRED_KEYS = {
    "robot0_eef_pos": (np.dtype("float32"), 2),
    "robot0_eef_rot_axis_angle": (np.dtype("float32"), 2),
    "robot0_gripper_width": (np.dtype("float32"), 2),
    "robot0_demo_start_pose": (np.dtype("float32"), 2),
    "robot0_demo_end_pose": (np.dtype("float32"), 2),
    "pose_valid": (np.dtype("bool"), 1),
    "gripper_valid": (np.dtype("bool"), 1),
}


def fail(message: str) -> None:
    raise SystemExit(f"VALIDATION FAILED: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to dataset.zarr.zip")
    parser.add_argument("--config-name", default=os.environ.get("NERO_TRAIN_CONFIG", "pika_nero_pick_place_train"),
                        help="Camera/task config; defaults to NERO_TRAIN_CONFIG or the dual-camera config")
    parser.add_argument("--check-normalizer", action="store_true", help="Also compute finite statistics over accepted training windows (CPU)")
    args = parser.parse_args()

    umi_root = os.environ.get("UMI_ROOT", str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, umi_root)
    try:
        # Importing the dataset registers the image codecs once.
        from diffusion_policy.dataset.umi_dataset import UmiDataset
    except ModuleNotFoundError as error:
        fail("Set UMI_ROOT to an upstream UMI checkout. " + str(error))

    # Use the same camera contract for validation, dataset and policy.
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    repo = Path(umi_root)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "config"), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=[
            f"hydra.searchpath=[file://{repo / 'diffusion_policy/config'}]"])
    cfg.task.dataset_path = str(Path(args.dataset).resolve())
    rgb_keys = [key for key, value in cfg.task.shape_meta.obs.items() if value.get('type') == 'rgb']
    camera_masks = cfg.task.dataset.quality_filter.camera_masks
    required = dict(REQUIRED_KEYS)
    for key in rgb_keys:
        required[key] = (np.dtype('uint8'), 4)
        required[camera_masks[key]] = (np.dtype('bool'), 1)
    print(f"Config: {args.config_name}; RGB inputs: {rgb_keys}")

    with zarr.ZipStore(args.dataset, mode="r") as store:
        root = zarr.group(store)
        if "data" not in root or "meta" not in root:
            fail("expected top-level data and meta groups")
        data = root["data"]
        meta = root["meta"]
        if "episode_ends" not in meta:
            fail("meta/episode_ends is missing")

        keys = set(data.array_keys())
        missing = set(required) - keys
        if missing:
            fail(f"missing required arrays: {sorted(missing)}")

        n_steps = None
        for key, (expected_dtype, expected_ndim) in required.items():
            array = data[key]
            if array.dtype != expected_dtype:
                fail(f"{key} dtype is {array.dtype}; expected {expected_dtype}")
            if array.ndim != expected_ndim:
                fail(f"{key} ndim is {array.ndim}; expected {expected_ndim}")
            if n_steps is None:
                n_steps = array.shape[0]
            elif array.shape[0] != n_steps:
                fail(f"{key} has {array.shape[0]} steps; expected {n_steps}")
            print(f"{key:34s} shape={array.shape} dtype={array.dtype}")

        expected_shapes = {
            "robot0_eef_pos": (3,),
            "robot0_eef_rot_axis_angle": (3,),
            "robot0_gripper_width": (1,),
            "robot0_demo_start_pose": (6,),
            "robot0_demo_end_pose": (6,),
        }
        for key in rgb_keys:
            channels, height, width = cfg.task.shape_meta.obs[key].shape
            expected_shapes[key] = (height, width, channels)
        for key, expected_shape in expected_shapes.items():
            if data[key].shape[1:] != expected_shape:
                fail(f"{key} trailing shape is {data[key].shape[1:]}; expected {expected_shape}")

        for key in ("robot0_eef_pos", "robot0_eef_rot_axis_angle",
                    "robot0_gripper_width", "robot0_demo_start_pose",
                    "robot0_demo_end_pose"):
            finite = np.isfinite(np.asarray(data[key][:])).all(axis=tuple(range(1, data[key].ndim)))
            print(f"{key} finite={int(finite.sum())}/{n_steps}")
        for key in ("fisheye_valid", "d405_valid", "frame_valid", "pose_valid", "gripper_valid"):
            if key not in keys:
                continue
            mask = np.asarray(data[key][:], dtype=bool)
            print(f"{key}: valid={int(mask.sum())}/{n_steps} invalid={int((~mask).sum())}")

        ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
        if ends.ndim != 1 or len(ends) == 0:
            fail("episode_ends must be a non-empty one-dimensional array")
        if np.any(np.diff(ends) <= 0):
            fail("episode_ends must be strictly increasing")
        if ends[-1] != n_steps:
            fail(f"episode_ends[-1]={ends[-1]} but data has {n_steps} steps")
        lengths = np.diff(np.concatenate(([0], ends)))
        print(f"episodes={len(ends)} steps={n_steps}")
        print(f"episode length: min={lengths.min()} median={np.median(lengths):.1f} max={lengths.max()}")
        if lengths.min() < 24:
            fail("an episode has fewer than 24 stored frames")

        if "timestamp" in keys:
            timestamp = np.asarray(data["timestamp"][:])
            if timestamp.shape[0] != n_steps:
                fail("timestamp length does not match data length")
            # Timestamps may validly restart between independently recorded
            # episodes, so enforce monotonicity within each episode only.
            delta_parts = []
            start = 0
            for episode_index, end in enumerate(ends):
                episode_delta = np.diff(timestamp[start:end])
                if np.any(episode_delta <= 0):
                    fail(f"timestamp is not strictly increasing in episode {episode_index}")
                delta_parts.append(episode_delta)
                start = end
            delta = np.concatenate(delta_parts)
            print(
                "timestamp delta (s): "
                f"median={np.median(delta):.6f} p99={np.quantile(delta, 0.99):.6f} max={delta.max():.6f}"
            )
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    validation = dataset.get_validation_dataset()
    print(f"accepted windows: train={len(dataset)} validation={len(validation)}")
    print(f"held-out original episodes: {np.flatnonzero(dataset.val_mask).tolist()}")
    for split in (dataset, validation):
        batch = split[0]
        for key, value in batch['obs'].items():
            if not np.isfinite(value.numpy()).all():
                fail(f"nonfinite sampled observation {key}")
        if not np.isfinite(batch['action'].numpy()).all():
            fail("nonfinite sampled action")
    if args.check_normalizer:
        normalizer = dataset.get_normalizer()
        if not all(np.isfinite(value.numpy()).all() for value in normalizer.state_dict().values()):
            fail("nonfinite normalizer parameters")
        print("NORMALIZER PASSED (accepted train windows only)")
    print(f"VALIDATION PASSED ({len(rgb_keys)} RGB, quality masks, fixed {cfg.task.dataset_frequeny:g} Hz)")


if __name__ == "__main__":
    main()
