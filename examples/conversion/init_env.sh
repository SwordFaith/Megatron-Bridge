#!/bin/bash

set -euo pipefail

git submodule update --init --recursive

# Megatron-Bridge has several native extensions that import `torch` at build time
# (e.g., nv-grouped-gemm, TransformerEngine). This repo also intentionally
# disables `uv` from automatically installing `torch` (see pyproject.toml) to
# avoid clobbering a system-provided PyTorch (common in NGC containers).
#
# Therefore:
# - If system Python already has torch, create a venv that can reuse it.
# - Otherwise, create an isolated venv and require the user to install torch.
if python3 -c "import torch" >/dev/null 2>&1; then
  python3 -m uv venv --system-site-packages .venv
else
  python3 -m uv venv .venv
fi

source ./.venv/bin/activate

uv pip install -U setuptools wheel

if ! python -c "import torch" >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ERROR: PyTorch (torch) is not available in this environment.

Megatron-Bridge builds several native dependencies that require `import torch`
at build time. Please install PyTorch first (matching your CUDA) and re-run
this script.

Reference: https://pytorch.org/get-started/locally/
EOF
  exit 1
fi

uv pip install -e .