#!/usr/bin/env bash
set -euo pipefail

NERO_PICK_PLACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${UMI_ROOT:?Set UMI_ROOT to the upstream UMI checkout}"
: "${NERO_PICK_PLACE_DATASET_PATH:?Set NERO_PICK_PLACE_DATASET_PATH to the Zarr zip}"

python "$NERO_PICK_PLACE_ROOT/scripts/validate_dataset.py" \
  "$NERO_PICK_PLACE_DATASET_PATH" "$@"
