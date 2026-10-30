#!/usr/bin/env bash
# Preprocessing, canonical inspection, and simulator-independent regression tests.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPU_VENV="${1:-$ROOT_DIR/.venv-cpu}"
command -v uv >/dev/null || { echo "Install uv: https://docs.astral.sh/uv/" >&2; exit 1; }
export UV_NO_CONFIG=1
[[ -d "$CPU_VENV" ]] || uv venv --managed-python --python 3.10 "$CPU_VENV"
"$CPU_VENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 10), "CPU tools require Python 3.10"'
uv pip install --python "$CPU_VENV/bin/python" \
  'numpy==1.26.4' 'h5py==3.14.0' 'Pillow==11.3.0' 'scipy==1.15.3' \
  'svgpathtools==1.7.1' 'scikit-image==0.25.2' 'opencv-python-headless==4.11.0.86' \
  'networkx==3.4.2' 'pytest==8.4.2' -e "$ROOT_DIR[preprocess,test]"
uv pip freeze --python "$CPU_VENV/bin/python" > "$CPU_VENV/installed-requirements.txt"
echo "CPU environment ready: $CPU_VENV"
