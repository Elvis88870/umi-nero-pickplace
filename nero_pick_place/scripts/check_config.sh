#!/usr/bin/env bash
set -euo pipefail

# Resolve Hydra composition without creating a model, loading the dataset, or
# starting training. Run this first on the GPU server after setting the two
# environment variables used by all launchers.
NERO_PICK_PLACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${UMI_ROOT:?Set UMI_ROOT to the upstream UMI checkout}"
: "${NERO_PICK_PLACE_DATASET_PATH:?Set NERO_PICK_PLACE_DATASET_PATH to the Zarr zip}"

export NERO_PICK_PLACE_DATASET_PATH

python "$UMI_ROOT/train.py" \
  --config-dir "$NERO_PICK_PLACE_ROOT/config" \
  --config-name "${NERO_TRAIN_CONFIG:-pika_nero_pick_place_train}" \
  --cfg job \
  --resolve \
  "$@"
