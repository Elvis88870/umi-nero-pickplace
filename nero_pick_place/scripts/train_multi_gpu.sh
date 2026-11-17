#!/usr/bin/env bash
set -euo pipefail

NERO_PICK_PLACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UMI_ROOT="${UMI_ROOT:-$(cd "$NERO_PICK_PLACE_ROOT/.." && pwd)}"
export NERO_PICK_PLACE_DATASET_PATH="${NERO_PICK_PLACE_DATASET_PATH:-$NERO_PICK_PLACE_ROOT/data/pika_pick_place_100.zarr.zip}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"

IFS=',' read -ra gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#gpu_ids[@]} -ne 2 ]]; then
  echo "Expected two GPU IDs, e.g. CUDA_VISIBLE_DEVICES=6,7" >&2
  exit 1
fi

# Resolve the output directory once so both ranks share normalizer/checkpoints.
run_dir="$NERO_PICK_PLACE_ROOT/data/outputs/$(date +%Y.%m.%d-%H.%M.%S)_dual_gpu_$$"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Output directory: $run_dir"
echo "Distributed port: ${NERO_MASTER_PORT:-29667}"

python -m accelerate.commands.launch \
  --multi_gpu \
  --num_processes 2 \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_port "${NERO_MASTER_PORT:-29667}" \
  --mixed_precision no \
  --dynamo_backend no \
  "$UMI_ROOT/train.py" \
  --config-dir "$NERO_PICK_PLACE_ROOT/config" \
  --config-name pika_nero_pick_place_train \
  "hydra.run.dir=$run_dir" \
  "$@"
