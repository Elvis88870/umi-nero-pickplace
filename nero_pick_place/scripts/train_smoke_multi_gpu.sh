#!/usr/bin/env bash
set -euo pipefail

NERO_PICK_PLACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UMI_ROOT="${UMI_ROOT:-$(cd "$NERO_PICK_PLACE_ROOT/.." && pwd)}"
export NERO_PICK_PLACE_DATASET_PATH="${NERO_PICK_PLACE_DATASET_PATH:-$NERO_PICK_PLACE_ROOT/data/pika_pick_place_100.zarr.zip}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"

bash "$NERO_PICK_PLACE_ROOT/scripts/validate_dataset.sh"

bash "$NERO_PICK_PLACE_ROOT/scripts/train_multi_gpu.sh" \
  training.debug=false \
  training.num_epochs=1 \
  training.max_train_steps=1 \
  training.max_val_steps=1 \
  training.lr_warmup_steps=0 \
  training.use_ema=false \
  training.sample_every=1 \
  training.checkpoint_every=1 \
  policy.obs_encoder.pretrained=false \
  dataloader.batch_size=1 \
  dataloader.num_workers=0 \
  dataloader.persistent_workers=false \
  val_dataloader.batch_size=1 \
  val_dataloader.num_workers=0 \
  val_dataloader.persistent_workers=false \
  logging.mode=disabled \
  "$@"

echo "DUAL GPU SMOKE TEST PASSED"
