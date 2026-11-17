# NERO Pick-and-Place Workspace

This directory is the project-local integration layer for the single-arm Pika/NERO pick-and-place policy. It does not modify the upstream UMI source tree under `diffusion_policy/` or `umi/`.

It is portable: set `UMI_ROOT` to any checkout of the UMI repository and `NERO_PICK_PLACE_DATASET_PATH` to the exported Zarr file. The directory can be moved beside that checkout without changing its contents.

## Scope

The baseline expects:

```text
camera0_rgb                         uint8 [N, 224, 224, 3]
robot0_eef_pos                      float32 [N, 3], metres
robot0_eef_rot_axis_angle           float32 [N, 3], radians
robot0_gripper_width                float32 [N, 1], metres
robot0_demo_start_pose              float32 [N, 6]
robot0_demo_end_pose                float32 [N, 6]
meta/episode_ends                   int64 [num_episodes]
```

The task is single-arm, uses relative TCP poses, stores data at 30 Hz, and trains at 10 Hz by sampling every third stored frame. It does not use the original UMI GoPro/SLAM/ArUco pipeline.

## Layout

```text
config/pika_nero_pick_place_train.yaml  Hydra override composition
scripts/validate_dataset.py              Read-only Zarr schema validator
scripts/validate_dataset.sh              Linux launcher for the validator
scripts/check_config.sh                  Resolve and print final Hydra config
scripts/train_debug.sh                   Short end-to-end smoke test
scripts/train.sh                         Full training launcher
```

## Server setup

On the 80 GB GPU training server, clone the upstream UMI repository and copy or clone this directory next to it. Create and activate the server's UMI environment first.

```bash
export UMI_ROOT=/workspace/universal_manipulation_interface
export NERO_PICK_PLACE_ROOT=/workspace/nero_pick_place
export NERO_PICK_PLACE_DATASET_PATH=/datasets/pika_pick_place_100.zarr.zip

cd "$NERO_PICK_PLACE_ROOT"
bash scripts/validate_dataset.sh
bash scripts/check_config.sh
bash scripts/train_debug.sh
```

After the smoke test completes, begin full training:

```bash
bash scripts/train.sh
```

Both training launchers default to GPU index `7` (the eighth GPU in
`nvidia-smi`, whose indices start at zero). They set `CUDA_DEVICE_ORDER=PCI_BUS_ID`
and preserve an explicitly supplied `CUDA_VISIBLE_DEVICES`. To select another GPU:

```bash
CUDA_VISIBLE_DEVICES=6 bash scripts/train.sh
```

With a single visible GPU, Accelerate uses it as logical `cuda:0`. Keep
`training.device` at its inherited `cuda:0`; the training workspace delegates
device placement to Accelerate, so overriding `training.device=cuda:7` does not
select physical GPU `7`.

The scripts accept extra Hydra overrides. For example:

```bash
bash scripts/train.sh training.num_epochs=200 dataloader.batch_size=128
```

For distributed training on GPU indices 6 and 7, use the two-process launcher:

```bash
CUDA_VISIBLE_DEVICES=6,7 bash scripts/train_multi_gpu.sh
```

It defaults to this checkout and `data/pika_pick_place_100.zarr.zip`, unless
`UMI_ROOT` or `NERO_PICK_PLACE_DATASET_PATH` is supplied. Both processes share
one output directory under `data/outputs`. Batch size is per GPU: the default
64 gives a global batch size of 128. To retain a global batch size of 64, pass
`dataloader.batch_size=32 val_dataloader.batch_size=32`.
The distributed port defaults to 29667. If occupied, set `NERO_MASTER_PORT`
to another available port. Avoid port 0 with the PyTorch 2.1 static launcher.
The minimal `train_smoke.sh` test remains single-GPU.

To check distributed training first, run the two-GPU smoke test:

```bash
CUDA_VISIBLE_DEVICES=6,7 bash scripts/train_smoke_multi_gpu.sh
```

It validates the local Zarr and runs one optimizer step per rank (batch size 1
per GPU), action sampling and checkpoint saving. Pretrained downloads, EMA and
WandB are disabled. Successful completion prints `DUAL GPU SMOKE TEST PASSED`.

## Baseline timing decision

`dataset_frequeny` is deliberately `30.0`: it describes the Zarr storage timeline, not the final 10 Hz training interval. `obs_down_sample_steps: 3` creates 10 Hz observation and action windows.

The task explicitly fixes both observation and action pose representations to
`relative`. This is compatible with the session-dependent `pika_world` origin
only when each episode is internally rigid and continuous.

All three latency values initially remain `0.0`. Timestamp-alignment residuals are suitable for a baseline, but they are not a measurement of physical camera/tracker/gripper end-to-end latency. Only change these values after a dedicated hardware latency calibration.

## Deployment boundary

This directory does not control NERO. Deployment needs a separate hardware gateway:

```text
Pika observations -> UMI obs_dict -> checkpoint inference -> safety filter -> NERO commands
```

Keep that safety gateway in `AGX_ARM_DUAL_WS`; do not reuse upstream `eval_real.py`, which targets UR/Franka/WSG hardware.
