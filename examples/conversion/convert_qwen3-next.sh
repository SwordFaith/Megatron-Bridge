#!/bin/bash

set -euo pipefail

# Convert a large HF model with model-parallelism (TP/PP/EP/ETP) via torchrun.
#
# Notes:
# - This performs HF -> Megatron (distributed) -> HF roundtrip conversion.
# - For large models, increasing TP (and/or PP) reduces per-GPU memory footprint.
# - WORLD_SIZE must match the model-parallel product:
#     WORLD_SIZE == TP * PP * EP
#   (ETP is an additional expert tensor-parallel dimension used by some MoE configs.)
#
# Examples:
#   TP=2 CUDA_VISIBLE_DEVICES=0,1 ./examples/conversion/convert_qwen3-next.sh
#   TP=4 PP=2 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./examples/conversion/convert_qwen3-next.sh
#
# You can also export a Megatron checkpoint:
#   MEGATRON_SAVE_PATH=./megatron_ckpt/qwen3-next TP=4 ./examples/conversion/convert_qwen3-next.sh

HF_MODEL_ID="${HF_MODEL_ID:-Qwen/Qwen3-Next-80B-A3B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./converted_models}"

TP="${TP:-2}"
PP="${PP:-1}"
EP="${EP:-4}"
ETP="${ETP:-1}"

MEGATRON_SAVE_PATH="${MEGATRON_SAVE_PATH:-}"

mkdir -p "${OUTPUT_DIR}"

# Default to one process per TP/PP/EP rank.
WORLD_SIZE="${WORLD_SIZE:-$((TP * PP * EP))}"

EXTRA_ARGS=()
if [[ -n "${MEGATRON_SAVE_PATH}" ]]; then
  EXTRA_ARGS+=(--megatron-save-path "${MEGATRON_SAVE_PATH}")
fi

# CUDA_VISIBLE_DEVICES=1,3,5,7 torchrun \
#   --nproc_per_node="${WORLD_SIZE}" \
#   examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
#   --hf-model-id "${HF_MODEL_ID}" \
#   --output-dir "${OUTPUT_DIR}" \
#   --tp "${TP}" \
#   --pp "${PP}" \
#   --ep "${EP}" \
#   --etp "${ETP}" \
#   "${EXTRA_ARGS[@]}"

CUDA_VISIBLE_DEVICES=1,3,5,7 python -m torch.distributed.run \
  --nproc_per_node="4" \
  examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
  --hf-model-id Qwen/Qwen3-Next-80B-A3B-Instruct \
  --output-dir ./converted_models \
  --tp 2 \
  --pp 1 \
  --ep 4 \
  --etp 1

CUDA_VISIBLE_DEVICES=1,3,5,7 python3 -m torch.distributed.run \
  --nproc_per_node="${WORLD_SIZE}" \
  examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
  --hf-model-id "${HF_MODEL_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --tp "${TP}" \
  --pp "${PP}" \
  --ep "${EP}" \
  --etp "${ETP}" \
  "${EXTRA_ARGS[@]}"
