#!/usr/bin/env bash
# CPU-only format conversion environment, independent of simulator dependencies.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPORT_VENV="${1:-$ROOT_DIR/.venv-lerobot-export}"
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
export UV_NO_CONFIG="${UV_NO_CONFIG:-1}"
[[ -x "$EXPORT_VENV/bin/python" ]] || uv venv --managed-python --python 3.10 "$EXPORT_VENV"
"$EXPORT_VENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Export requires Python 3.10"'
uv pip install --python "$EXPORT_VENV/bin/python" --torch-backend cpu \
  'lerobot==0.3.3' h5py svgpathtools pillow scipy imageio-ffmpeg 'numpy<2' 'pytest>=8,<10'
FFMPEG_PATH="$("$EXPORT_VENV/bin/python" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
ln -sf "$FFMPEG_PATH" "$EXPORT_VENV/bin/ffmpeg"
uv pip install --python "$EXPORT_VENV/bin/python" --no-deps -e "$ROOT_DIR"
uv pip freeze --python "$EXPORT_VENV/bin/python" > "$EXPORT_VENV/installed-requirements.txt"
echo "Export environment ready: $EXPORT_VENV"
