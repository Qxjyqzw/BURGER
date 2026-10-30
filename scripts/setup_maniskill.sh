#!/usr/bin/env bash
# Linux x86_64, NVIDIA CUDA 12.4-capable driver and Vulkan ICD required.
# Usage: bash scripts/setup_maniskill.sh [VENV_DIR]
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:-${ROOT_DIR}/.venv-maniskill}"
export UV_NO_CONFIG=1
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo "ManiSkill GPU setup requires Linux x86_64" >&2; exit 1; }
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --managed-python --python 3.10 "$VENV"
fi
"$VENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use a separate Python 3.10 venv"'
uv pip install --python "$VENV/bin/python" 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
uv pip install --python "$VENV/bin/python" -r "$ROOT_DIR/scripts/requirements-maniskill.txt"
uv pip install --python "$VENV/bin/python" --no-deps -e "$ROOT_DIR"
"$VENV/bin/python" -c 'import torch, mani_skill, sapien; import burger.backends.maniskill.environment; assert torch.cuda.is_available(), "CUDA GPU unavailable"; print("ManiSkill imports and CUDA OK; run the README smoke test to verify Vulkan rendering")'
uv pip freeze --python "$VENV/bin/python" > "$VENV/installed-requirements.txt"
echo "ManiSkill environment ready: $VENV"
