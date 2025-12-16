#!/bin/bash

set -euo pipefail

# =============================================================================
# Script to convert Qwen3-Next models using Megatron-Bridge.
# Supports HF -> Megatron (distributed) -> HF roundtrip conversion verification.
#
# Usage:
#   [Env Vars] ./examples/conversion/convert_qwen3-next.sh
#
# Environment Variables (with defaults):
#   HF_MODEL_ID        : Qwen/Qwen3-Next-80B-A3B-Instruct
#   OUTPUT_DIR         : ./converted_models
#   TP                 : 2  (Tensor Parallel)
#   PP                 : 1  (Pipeline Parallel)
#   EP                 : 4  (Expert Parallel)
#   ETP                : 1  (Expert Tensor Parallel)
#   MEGATRON_SAVE_PATH : (Optional) Path to save Megatron checkpoint
#   LOG_FILE           : convert_qwen3-next.log
#
# =============================================================================
# Reference Example (Verified Configuration):
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./examples/conversion/convert_qwen3-next.sh
#
#   This maps to the following manual command:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
#     --nproc_per_node="4" \
#     examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
#     --hf-model-id Qwen/Qwen3-Next-80B-A3B-Instruct \
#     --output-dir ./converted_models \
#     --tp 2 \
#     --pp 1 \
#     --ep 4 \
#     --etp 1 |& tee convert_qwen3-next.log
# =============================================================================

# --- Configuration ---

export CUDA_VISIBLE_DEVICES=0,1,2,3
HF_MODEL_ID="${HF_MODEL_ID:-Qwen/Qwen3-Next-80B-A3B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./converted_models}"
LOG_FILE="${LOG_FILE:-convert_qwen3-next.log}"

# Parallelism Defaults (Matches Qwen3-Next-80B-A3B verified config)
TP="${TP:-2}"
PP="${PP:-1}"
EP="${EP:-4}"
ETP="${ETP:-1}"

MEGATRON_SAVE_PATH="${MEGATRON_SAVE_PATH:-}"

# Calculate World Size
# Note: Ensure CUDA_VISIBLE_DEVICES provides enough GPUs for this WORLD_SIZE.
WORLD_SIZE="${WORLD_SIZE:-$(( (TP * PP > EP * ETP) ? TP * PP : EP * ETP ))}"

mkdir -p "${OUTPUT_DIR}"

# --- Arguments Building ---

CMD_ARGS=(
    "--nproc_per_node=${WORLD_SIZE}"
    "examples/conversion/hf_megatron_roundtrip_multi_gpu.py"
    "--hf-model-id" "${HF_MODEL_ID}"
    "--output-dir" "${OUTPUT_DIR}"
    "--tp" "${TP}"
    "--pp" "${PP}"
    "--ep" "${EP}"
    "--etp" "${ETP}"
)

if [[ -n "${MEGATRON_SAVE_PATH}" ]]; then
    CMD_ARGS+=("--megatron-save-path" "${MEGATRON_SAVE_PATH}")
fi

# --- Execution ---

echo "----------------------------------------------------------------"
echo "Starting Conversion Verification"
echo "Model     : ${HF_MODEL_ID}"
echo "Output    : ${OUTPUT_DIR}"
echo "Topology  : TP=${TP}, PP=${PP}, EP=${EP}, ETP=${ETP} (World Size: ${WORLD_SIZE})"
echo "Log File  : ${LOG_FILE}"
echo "Command   : python -m torch.distributed.run ${CMD_ARGS[*]}"
echo "----------------------------------------------------------------"

# Run with pipe to tee for logging
# Note: We use 'python -m torch.distributed.run' instead of 'torchrun' for better python path handling
python -m torch.distributed.run "${CMD_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
