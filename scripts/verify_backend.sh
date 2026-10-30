#!/usr/bin/env bash
# Three unequal-length drawings, two vector slots/workers, and real readback.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${1:?Usage: bash scripts/verify_backend.sh maniskill|libero|isaac [output_dir] [parallel]}"
OUT="${2:-$ROOT_DIR/validation/smoke-$(date +%Y%m%d-%H%M%S)}"
PARALLEL="${3:-2}"
cd "$ROOT_DIR"
bash scripts/generate.sh \
  --backend "$BACKEND" --svg_dir examples/svg --output_dir "$OUT" \
  --object_name smoke --source_desc "draw the supplied geometric sketch" \
  --parallel "$PARALLEL" --seed 7 --format canonical,lerobot \
  --lerobot_repo_name smoke-export
export PATH="$ROOT_DIR/.venv-lerobot-export/bin:$PATH"
"$ROOT_DIR/.venv-lerobot-export/bin/python" -m burger check \
  --canonical "$OUT/smoke/$BACKEND/smoke.canonical.h5" \
  --expected_episodes 3 --lerobot "$OUT/smoke/$BACKEND/smoke-export"
