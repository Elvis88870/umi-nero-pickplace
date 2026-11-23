# NERO Pick-and-Place Workspace

This directory is the project-local integration layer for the single-arm Pika/NERO pick-and-place policy. The UMI dataset/sampler also support optional quality masks for fixed-rate exports.

Set `UMI_ROOT` to this UMI checkout (including the quality-mask and resume changes) and `NERO_PICK_PLACE_DATASET_PATH` to the exported Zarr file.

## Scope

### Single fisheye RGB dataset

`data/pika_dataset_fisheye_0_99.zarr.zip` uses only `camera0_rgb`.
Use `pika_nero_pick_place_fisheye_train.yaml` via the dedicated launcher:

```bash
conda activate umi
cd /root/umi-nero-pickplace
unset NERO_RESUME_DIR NERO_RUN_DIR
export UMI_ROOT="$PWD"
export NERO_PICK_PLACE_DATASET_PATH="$PWD/nero_pick_place/data/pika_dataset_fisheye_0_99.zarr.zip"

# CPU schema, quality-window, timing and normalizer checks:
python nero_pick_place/scripts/validate_dataset.py "$NERO_PICK_PLACE_DATASET_PATH" \
  --config-name pika_nero_pick_place_fisheye_train --check-normalizer

# One camera, two GPUs; one batch per rank without pretrained downloads:
CUDA_VISIBLE_DEVICES=6,7 bash nero_pick_place/scripts/train_fisheye.sh --smoke

# Start a fresh single-camera model, CLIP pretrained + EMA:
CUDA_VISIBLE_DEVICES=6,7 bash nero_pick_place/scripts/train_fisheye.sh
```

Only `fisheye_valid`, finite/valid pose and gripper labels affect eligibility.
For the current file the sampler retains 8,873 train windows and 504 validation
windows (the dual-camera version retained 7,133 and 423). These counts use the
same original-episode split; single-camera training must start a new model.
The stored `frame_valid` may also include missing D405 frames, so it remains
diagnostic and does not reject single-camera samples. The dataset is not
rewritten. Timing (30 Hz, stride 3), horizon 16, original-episode split,
checkpointing and optimizer-state saving are unchanged. The original dual-RGB
entry point `train_multi_gpu.sh` retains its default config and dataset.

All generic train/check scripts accept `NERO_TRAIN_CONFIG` (dual-RGB by default).
For example, to resolve the single-camera config without running training:

```bash
NERO_TRAIN_CONFIG=pika_nero_pick_place_fisheye_train \
  bash nero_pick_place/scripts/check_config.sh
```

To resume, pass this single-camera run's own directory as `NERO_RESUME_DIR` to
`train_fisheye.sh`. A saved dual-camera run must keep its own two-camera config;
its weights cannot be resumed as this single-camera model. Offline evaluation
and Rerun already select RGB keys from the checkpoint, so the same scripts work
with new single-camera checkpoints. Thor's observation adapter likewise follows
the checkpoint: the single-camera model requires only `camera0_rgb` plus states.

The baseline expects:

```text
camera0_rgb                         uint8 [N, 224, 224, 3] fisheye RGB
camera1_rgb                         uint8 [N, 224, 224, 3] D405 RGB
robot0_eef_pos                      float32 [N, 3], metres
robot0_eef_rot_axis_angle           float32 [N, 3], radians
robot0_gripper_width                float32 [N, 1], metres
robot0_demo_start_pose              float32 [N, 6]
robot0_demo_end_pose                float32 [N, 6]
meta/episode_ends                   int64 [num_episodes]
data/fisheye_valid, d405_valid       bool [N]
data/pose_valid, gripper_valid       bool [N]
data/frame_valid                    bool [N] diagnostic aggregate
data/timestamp                      float64 [N] seconds, fixed 30 Hz
```

The task is single-arm, uses relative TCP poses, stores data at 30 Hz, and trains at 10 Hz by sampling every third stored frame. It does not use the original UMI GoPro/SLAM/ArUco pipeline.

Both RGB streams are policy inputs; depth is excluded. Accepted windows require
both cameras and all low-dimensional states throughout observation history,
and finite, valid pose/gripper labels throughout the action interval (including
frames between stride-3 labels). Future missing images do not reject a window.
`frame_valid` is diagnostic; independent component masks determine eligibility.
Rows are never deleted or joined across gaps. At original episode starts,
the existing sampler repeats the earliest valid observation for history padding.
The original episode split is retained, and normalization enumerates only
accepted training windows, excluding every invalid placeholder and validation
episode. Invalid slots remain in the original Zarr file.
Quality filtering supports zero latency and integer strides. Timeline checks
enforce 30 Hz within 0.1 ms; changing timing requires revisiting this contract.
At stride 3, 16 action labels cover offsets 0.0--1.5 s (a 1.6 s chunk convention).

Run the three stages on the server:

```bash
conda activate umi
cd /root/umi-nero-pickplace
export UMI_ROOT="$PWD"
export NERO_PICK_PLACE_DATASET_PATH="$PWD/nero_pick_place/data/pika_dataset_0_99.zarr.zip"
bash nero_pick_place/scripts/validate_dataset.sh --check-normalizer
CUDA_VISIBLE_DEVICES=6,7 bash nero_pick_place/scripts/train_smoke_multi_gpu.sh
# After the smoke test passes; a new run directory is created automatically:
CUDA_VISIBLE_DEVICES=6,7 bash nero_pick_place/scripts/train_multi_gpu.sh
```

The smoke test disables pretrained downloads and EMA, and performs one batch
per rank, validation and backward/optimizer update. Full training enables CLIP
pretraining and EMA. `latest.ckpt` contains resume state; five ranked copies are
selected by validation loss at the checkpoint interval. Compare held-out action
errors and actual robot rollouts as well; loss alone does not establish success.

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
export NERO_PICK_PLACE_DATASET_PATH=/datasets/pika_dataset_0_99.zarr.zip

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

It defaults to this checkout and `data/pika_dataset_0_99.zarr.zip`, unless
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

`nero_pick_place.inference.make_thor_observation` prepares the checkpoint's
declared RGB streams with the existing UMI preprocessing. For this new model,
the Thor gateway must supply `camera0_rgb` and `camera1_rgb` as synchronized
uint8 RGB histories `[2,H,W,3]`, at 0.1 s spacing, plus matching Pika pose and
gripper histories in the training coordinate convention. Map D405 color to
`camera1_rgb`; do not send depth or substitute a blank frame. The adapter rejects
missing camera1, invalid shapes and nonfinite states. The hardware acquisition
and control gateway are separate and have not been integrated or tested here.

## Saving and resuming training

New runs always save the AdamW state, learning-rate scheduler, model and EMA
weights, EMA update counter, completed-epoch progress, and per-rank main-process
RNG/loader generator states. `training.resume=false` means start fresh; it no
longer excludes the optimizer from checkpoints. The latest checkpoint is saved
after every complete epoch, including the final epoch. A temporary file is
completed before replacing `checkpoints/latest.ckpt`. Interrupting mid-epoch
resumes from the last completed epoch, not the interrupted batch.

Named checkpoints are retained at the configured interval (default 10 epochs,
plus the final epoch), with up to 5 selected by validation loss. These now contain
optimizer state and are substantially larger than the old weights-only files.
Evaluation still loads the model/EMA weights directly.

To start a new model from scratch in a new directory:

```bash
conda activate umi
cd /root/umi-nero-pickplace
NERO_RUN_DIR="/root/umi-nero-pickplace/nero_pick_place/data/outputs/$(date +%Y%m%d_%H%M%S)_restart" \
CUDA_VISIBLE_DEVICES=6,7 bash nero_pick_place/scripts/train_multi_gpu.sh
```

Keep the printed output directory. To resume that run later:

```bash
NERO_RESUME_DIR=/absolute/path/to/the/new/run \
CUDA_VISIBLE_DEVICES=6,7 bash nero_pick_place/scripts/train_multi_gpu.sh
```

The launcher reuses the saved Hydra configuration and `latest.ckpt`. A saved
epoch of 21 means epochs 0--20 completed and the next epoch is 21. `num_epochs`
is the total target, not a number of additional epochs. Keep total epochs,
dataset/split, batch size, accumulation, scheduler and process count unchanged
when restoring. Optimizer state is restored after distributed device placement.
Old checkpoints without optimizer/scheduler state are explicitly rejected for
this resume path; they remain valid for evaluation.

Checkpoint recovery preserves optimization state but is not guaranteed bitwise
identical with multiprocessing data loaders: worker RNG and prefetch queues are
not saved. The CPU regression test with zero workers verifies that uninterrupted
and interrupted/resumed runs match exactly, including an incomplete gradient
accumulation group. Run it with:

```bash
python -m unittest discover -s tests -p test_training_resume.py -v
```

## Offline checkpoint evaluation and Rerun

`scripts/eval_checkpoint.py` loads the checkpoint policy and embedded normalizer
(EMA weights if enabled), without creating an optimizer or downloading pretrained
weights. It preserves the checkpoint's episode split, samples 10 evenly spaced
valid windows per held-out episode, and uses deployment observation preprocessing
without the dataset loader's training-time start-pose noise. No robot is accessed.

```bash
conda activate umi
cd /root/umi-nero-pickplace
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 python \
  nero_pick_place/scripts/eval_checkpoint.py \
  'nero_pick_place/data/outputs/2026.11.17-13.10.36_dual_gpu_560720/checkpoints/epoch=0020-train_loss=0.020.ckpt'
```

The default output is `<run>/offline_eval/<checkpoint-stem>/`. The output directory
must not exist; use `--output-dir /path/to/new-directory` for repeat evaluations.
Use `--device cpu` for CPU inference, `--max-episodes 1 --samples-per-episode 1`
for a quick check, or `--samples-per-episode 0` for all valid validation windows.
The inference seed is fixed at 42 by default; it does not change the stored split.

`summary.json` reports TCP position distance in mm, SO(3) rotation error in degrees,
and gripper width error in mm, with per-horizon statistics and a hold-current-pose
baseline. Overall means weight episodes equally. A horizon includes the current
frame as step 0. The reported inference time includes the CPU action transfer and
cold first prediction, but excludes preprocessing; it is not a deployment latency
benchmark. Offline comparison cannot establish closed-loop task success.

Each `episode_XXX.npz` also contains images, predictions and targets, world-frame
TCP transforms, sampled frame indices and actual episode-relative timestamps.
Episodes remain separate because their reference frames may differ.

Install Rerun separately from the training environment, using Python 3.10+:

```bash
python3 -m venv /tmp/umi-rerun-view
/tmp/umi-rerun-view/bin/python -m pip install 'rerun-sdk==0.22.1'
/tmp/umi-rerun-view/bin/python nero_pick_place/scripts/render_eval_rerun.py \
  'nero_pick_place/data/outputs/2026.11.17-13.10.36_dual_gpu_560720/offline_eval/epoch=0020-train_loss=0.020'
```

This creates one `.rrd` per episode, with RGB, world-frame TCP paths and orientation
axes, gripper horizon curves, and error plots. Green is demonstration, red is model
prediction. Use the `sample` or `episode_time` timeline to inspect prediction windows.
The recorded steps are the sampled observations, not a full-rate video.
Copy `.rrd` files to a desktop with a matching Rerun 0.22.1 Viewer and open them there;
the server does not need a GUI. The exporter needs only NumPy and Rerun, not the model
or dataset, so it can also run on the desktop using the generated NPZ files.
