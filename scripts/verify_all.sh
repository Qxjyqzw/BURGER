#!/usr/bin/env bash
# Execute README setup steps independently and retain every result.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
RUN_DIR="${BURGER_VERIFY_DIR:-$ROOT_DIR/validation/verification/run-$(date +%Y%m%d-%H%M%S)-$$}"
mkdir -p "$RUN_DIR"
STEP_TIMEOUT="${BURGER_VERIFY_TIMEOUT_S:-3600}"
read -r -a BACKENDS <<< "${BURGER_VERIFY_BACKENDS:-maniskill libero isaac}"
for backend in "${BACKENDS[@]}"; do
  case "$backend" in
    maniskill|libero|isaac) ;;
    *) echo "Unknown verification backend: $backend" >&2; exit 2 ;;
  esac
done
failed=0
step() {
  local name="$1"
  shift
  echo "[verify] $name"
  if timeout --signal=TERM --kill-after=30s "$STEP_TIMEOUT" "$@" </dev/null >"$RUN_DIR/$name.log" 2>&1; then
    echo "$name PASS" | tee -a "$RUN_DIR/status.txt"
  else
    echo "$name FAIL" | tee -a "$RUN_DIR/status.txt"
    tail -30 "$RUN_DIR/$name.log"
    failed=1
    return 1
  fi
}
{
  uname -a
  cat /etc/os-release
  nvidia-smi
  df -h .
} > "$RUN_DIR/host.txt" 2>&1
step setup-cpu bash scripts/setup_cpu.sh && step tests-cpu .venv-cpu/bin/python -m pytest -q -m 'not export'
step setup-export bash scripts/setup_export.sh &&
  step tests-export .venv-lerobot-export/bin/python -m pytest -q tests/test_data_contract.py tests/test_svg_trajectory.py tests/test_export.py
for backend in "${BACKENDS[@]}"; do
  setup="scripts/setup_$backend.sh"
  if step "setup-$backend" bash "$setup"; then
    for parallel in 1 2; do
      step "$backend-p$parallel" bash scripts/verify_backend.sh "$backend" \
        "$RUN_DIR/$backend-p$parallel" "$parallel"
    done
  fi
done
echo "[verify] Results: $RUN_DIR"
exit "$failed"
