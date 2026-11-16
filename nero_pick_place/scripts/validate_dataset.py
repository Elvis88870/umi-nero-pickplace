#!/usr/bin/env python3
"""Read-only structural validation for a single-arm UMI ReplayBuffer Zarr."""

import argparse
import os
import sys

import numpy as np
import zarr


REQUIRED_KEYS = {
    "camera0_rgb": (np.dtype("uint8"), 4),
    "robot0_eef_pos": (np.dtype("float32"), 2),
    "robot0_eef_rot_axis_angle": (np.dtype("float32"), 2),
    "robot0_gripper_width": (np.dtype("float32"), 2),
    "robot0_demo_start_pose": (np.dtype("float32"), 2),
    "robot0_demo_end_pose": (np.dtype("float32"), 2),
}


def fail(message: str) -> None:
    raise SystemExit(f"VALIDATION FAILED: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to dataset.zarr.zip")
    args = parser.parse_args()

    umi_root = os.environ.get("UMI_ROOT")
    if umi_root:
        sys.path.insert(0, umi_root)
    try:
        from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
    except ModuleNotFoundError as error:
        fail("Set UMI_ROOT to an upstream UMI checkout. " + str(error))
    register_codecs()

    with zarr.ZipStore(args.dataset, mode="r") as store:
        root = zarr.group(store)
        if "data" not in root or "meta" not in root:
            fail("expected top-level data and meta groups")
        data = root["data"]
        meta = root["meta"]
        if "episode_ends" not in meta:
            fail("meta/episode_ends is missing")

        keys = set(data.array_keys())
        missing = set(REQUIRED_KEYS) - keys
        if missing:
            fail(f"missing required arrays: {sorted(missing)}")

        n_steps = None
        for key, (expected_dtype, expected_ndim) in REQUIRED_KEYS.items():
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
            "camera0_rgb": (224, 224, 3),
            "robot0_eef_pos": (3,),
            "robot0_eef_rot_axis_angle": (3,),
            "robot0_gripper_width": (1,),
            "robot0_demo_start_pose": (6,),
            "robot0_demo_end_pose": (6,),
        }
        for key, expected_shape in expected_shapes.items():
            if data[key].shape[1:] != expected_shape:
                fail(f"{key} trailing shape is {data[key].shape[1:]}; expected {expected_shape}")

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
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
