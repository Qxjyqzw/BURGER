#!/usr/bin/env bash
# Convenience wrapper: runs burger.pipeline.generate with the virtualenv of the requested backend.
#   bash scripts/generate.sh --backend libero --format canonical,lerobot \
#     --svg_dir SVG --output_dir OUT --object_name house --source_desc "draw a house"
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND=""
prev=""
for a in "$@"; do
  if [[ "$prev" == "--backend" ]]; then BACKEND="$a"; fi
  case "$a" in --backend=*) BACKEND="${a#--backend=}";; esac
  prev="$a"
done
case "$BACKEND" in
  maniskill) PY="${PYTHON_MANISKILL:-${ROOT_DIR}/.venv-maniskill/bin/python}";;
  libero)    PY="${PYTHON_LIBERO:-${ROOT_DIR}/.venv-libero/bin/python}";;
  isaac)     PY="${PYTHON_ISAAC:-${ROOT_DIR}/.venv-isaac61/bin/python}";;
  *) echo "Pass --backend maniskill, libero, or isaac." >&2; exit 1;;
esac
[[ -f "$ROOT_DIR/burger/backends/${BACKEND}/backend.py" ]] || {
  echo "backend $BACKEND is not included in this branch" >&2; exit 1;
}
[[ -x "$PY" ]] || { echo "Missing environment: $PY. See README.md for setup." >&2; exit 1; }
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m burger generate "$@"
