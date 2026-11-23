#!/usr/bin/env bash
set -euo pipefail

# One RGB camera, two training GPUs. Delegate checkpoint/resume behavior to
# the existing distributed launcher; --smoke selects its one-batch variant.
NERO_PICK_PLACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UMI_ROOT="${UMI_ROOT:-$(cd "$NERO_PICK_PLACE_ROOT/.." && pwd)}"
export NERO_TRAIN_CONFIG=pika_nero_pick_place_fisheye_train
export NERO_PICK_PLACE_DATASET_PATH="${NERO_PICK_PLACE_DATASET_PATH:-$NERO_PICK_PLACE_ROOT/data/pika_dataset_fisheye_0_99.zarr.zip}"
launcher=train_multi_gpu.sh
if [[ "${1:-}" == --smoke ]]; then
  launcher=train_smoke_multi_gpu.sh
  shift
  if [[ -n "${NERO_RESUME_DIR:-}" ]]; then
    echo "Smoke tests must use a fresh run; unset NERO_RESUME_DIR." >&2
    exit 1
  fi
fi
bash "$NERO_PICK_PLACE_ROOT/scripts/$launcher" "$@"
