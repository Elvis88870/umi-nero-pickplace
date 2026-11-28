# D405 isolated-decoder experiment

This directory preserves the task-selective image decoder experiment:

```text
fisheye CompressedImage + D405 shared memory
  -> timestamp synchronization
  -> selected fisheye/D405 sequences
  -> isolated decoder process
  -> policy observation
```

It is diagnostic only and is not imported by the production single-fisheye
entrypoint. The production checkpoint currently uses only `camera0_rgb`.

The experiment is preview-only. Do not add `--control`, and do not run it at
the same time as `eval_nero_ros.py`.

Start the decoder worker in one terminal:

```bash
python experiments/d405_worker/image_decoder_worker.py
```

Run preview inference in another terminal from `inference_ros2/`:

```bash
UMI_INFERENCE_CPUS=1-11,13 UMI_TORCH_THREADS=8 \
python experiments/d405_worker/eval_nero_ros_worker.py \
  --checkpoint ../checkpoints/REPLACE_ME.ckpt \
  --preview \
  --no-rerun
```

Both experiments require their external D405 shared-memory writer and fisheye
camera bridge to be running. They are not started by the production launch.
