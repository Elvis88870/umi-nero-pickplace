#!/usr/bin/env bash
set -euo pipefail

NERO_PICK_PLACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UMI_ROOT="${UMI_ROOT:-$(cd "$NERO_PICK_PLACE_ROOT/.." && pwd)}"
export NERO_PICK_PLACE_DATASET_PATH="${NERO_PICK_PLACE_DATASET_PATH:-$NERO_PICK_PLACE_ROOT/data/pika_dataset_0_99.zarr.zip}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"

IFS=',' read -ra gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#gpu_ids[@]} -ne 2 ]]; then
  echo "Expected two GPU IDs, e.g. CUDA_VISIBLE_DEVICES=6,7" >&2
  exit 1
fi

# Resolve the output directory once so both ranks share normalizer/checkpoints.
run_dir="${NERO_RUN_DIR:-$NERO_PICK_PLACE_ROOT/data/outputs/$(date +%Y.%m.%d-%H.%M.%S)_dual_gpu_$$}"
config_args=(--config-dir "$NERO_PICK_PLACE_ROOT/config" --config-name "${NERO_TRAIN_CONFIG:-pika_nero_pick_place_train}")
resume_args=(training.resume=false)
if [[ -n "${NERO_RESUME_DIR:-}" ]]; then
  run_dir="$(cd "$NERO_RESUME_DIR" && pwd)"
  if [[ ! -f "$run_dir/checkpoints/latest.ckpt" || ! -f "$run_dir/.hydra/config.yaml" ]]; then
    echo "Resume directory must contain checkpoints/latest.ckpt and .hydra/config.yaml" >&2
    exit 1
  fi
  config_args=(--config-dir "$run_dir/.hydra" --config-name config)
  resume_args=(training.resume=true)
elif [[ -e "$run_dir/checkpoints/latest.ckpt" ]]; then
  echo "Output already contains a checkpoint. Use NERO_RESUME_DIR to resume it." >&2
  exit 1
fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Config: ${config_args[*]}"
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
  "${config_args[@]}" \
  "${resume_args[@]}" \
  "hydra.run.dir=$run_dir" \
  "$@"
