#!/usr/bin/env bash
# Python 3.10 environment for headless LIBERO data generation.
# Usage: bash scripts/setup_libero.sh [VENV_DIR] [LIBERO_DIR]
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:-${ROOT_DIR}/.venv-libero}"
LIBERO_DIR="${2:-${ROOT_DIR}/third_party/LIBERO}"
LIBERO_COMMIT=8f1084e3132a39270c3a13ebe37270a43ece2a01
export UV_NO_CONFIG=1
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/"; exit 1; }
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --managed-python --python 3.10 "$VENV"
fi
"$VENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 10), "LIBERO requires Python 3.10; use a separate venv"'
if [[ ! -d "$LIBERO_DIR/.git" ]]; then
  mkdir -p "$LIBERO_DIR"
  git -C "$LIBERO_DIR" init -q
  git -C "$LIBERO_DIR" remote add origin https://github.com/Lifelong-Robot-Learning/LIBERO.git
  git -C "$LIBERO_DIR" fetch --depth 1 origin "$LIBERO_COMMIT"
  git -C "$LIBERO_DIR" checkout --detach "$LIBERO_COMMIT"
fi
if [[ "$(git -C "$LIBERO_DIR" rev-parse HEAD)" != "$LIBERO_COMMIT" ]] || [[ -n "$(git -C "$LIBERO_DIR" status --porcelain)" ]]; then
  echo "LIBERO checkout must be clean at $LIBERO_COMMIT. Use a new LIBERO_DIR to preserve your changes." >&2
  exit 1
fi
# robosuite 1.4.1 is incompatible with MuJoCo >=3 and NumPy >=2.
# robosuite requires opencv-python. A second headless provider owns the same cv2
# files; remove it from this venv and restore the supported provider on reruns.
OPENCV_FLAGS=()
if "$VENV/bin/python" -c 'from importlib.metadata import distribution; distribution("opencv-python-headless")' >/dev/null 2>&1; then
  uv pip uninstall --python "$VENV/bin/python" opencv-python-headless
  OPENCV_FLAGS+=(--reinstall-package opencv-python)
fi
uv pip install --python "$VENV/bin/python" "${OPENCV_FLAGS[@]}" -r "$ROOT_DIR/scripts/requirements-libero.txt"
SITE="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
# LIBERO is a namespace package; register both source roots for imports from any cwd.
printf '%s\n' "$LIBERO_DIR" > "$SITE/libero_repo_root.pth"
uv pip install --python "$VENV/bin/python" --no-deps -e "$ROOT_DIR"
# Keep LIBERO's configuration in this venv, without interactive prompts or modifying ~/.libero.
"$VENV/bin/python" - "$LIBERO_DIR" "$VENV" "$SITE" <<'PY'
import pathlib, sys, yaml
libero_dir, venv, site = map(pathlib.Path, sys.argv[1:])
config_dir = (venv / 'libero-config').resolve()
config_dir.mkdir(exist_ok=True)
base = (libero_dir / 'libero/libero').resolve()
paths = dict(benchmark_root=str(base), bddl_files=str(base/'bddl_files'), init_states=str(base/'init_files'), datasets=str(base.parent/'datasets'), assets=str(base/'assets'))
(config_dir/'config.yaml').write_text(yaml.safe_dump(paths))
(site/'burger_libero_config.pth').write_text(f'import os; os.environ.setdefault("LIBERO_CONFIG_PATH", {str(config_dir)!r})\n')
PY
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$VENV/bin/python" -c "import libero.libero.envs, burger.backends.libero.environment; print('LIBERO + burger.backends.libero import OK')"
uv pip freeze --python "$VENV/bin/python" > "$VENV/installed-requirements.txt"
echo "LIBERO environment ready: $VENV"
