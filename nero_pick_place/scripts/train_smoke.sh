#!/usr/bin/env bash
set -euo pipefail

# Minimal end-to-end smoke test: validate the local Zarr, build the policy,
# load one batch, and run one optimizer step.
NERO_PICK_PLACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UMI_ROOT="${UMI_ROOT:-$(cd "$NERO_PICK_PLACE_ROOT/.." && pwd)}"
NERO_PICK_PLACE_DATASET_PATH="${NERO_PICK_PLACE_DATASET_PATH:-$NERO_PICK_PLACE_ROOT/data/pika_dataset_0_99.zarr.zip}"

export UMI_ROOT
export NERO_PICK_PLACE_DATASET_PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

echo "UMI_ROOT=$UMI_ROOT"
echo "NERO_PICK_PLACE_DATASET_PATH=$NERO_PICK_PLACE_DATASET_PATH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

bash "$NERO_PICK_PLACE_ROOT/scripts/validate_dataset.sh"

python "$UMI_ROOT/train.py" \
  --config-dir "$NERO_PICK_PLACE_ROOT/config" \
  --config-name "${NERO_TRAIN_CONFIG:-pika_nero_pick_place_train}" \
  training.debug=false \
  training.num_epochs=1 \
  training.max_train_steps=1 \
  training.max_val_steps=1 \
  training.rollout_every=1 \
  training.checkpoint_every=1 \
  training.sample_every=1 \
  training.val_every=1 \
  training.use_ema=false \
  policy.obs_encoder.pretrained=false \
  dataloader.batch_size=1 \
  dataloader.num_workers=0 \
  dataloader.persistent_workers=false \
  val_dataloader.batch_size=1 \
  val_dataloader.num_workers=0 \
  val_dataloader.persistent_workers=false \
  logging.mode=disabled \
  "$@"

echo "SMOKE TEST PASSED"
