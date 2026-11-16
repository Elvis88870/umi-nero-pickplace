#!/usr/bin/env bash
set -euo pipefail

NERO_PICK_PLACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${UMI_ROOT:?Set UMI_ROOT to the upstream UMI checkout}"
: "${NERO_PICK_PLACE_DATASET_PATH:?Set NERO_PICK_PLACE_DATASET_PATH to the Zarr zip}"

export NERO_PICK_PLACE_DATASET_PATH

python "$UMI_ROOT/train.py" \
  --config-dir "$NERO_PICK_PLACE_ROOT/config" \
  --config-name pika_nero_pick_place_train \
  training.debug=true \
  logging.mode=disabled \
  "$@"
