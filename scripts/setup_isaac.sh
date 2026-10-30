#!/usr/bin/env bash
# Supported host: Linux x86_64, glibc >=2.35, NVIDIA RTX-capable GPU/driver.
# Usage: bash scripts/setup_isaac.sh
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT_DIR/.venv-isaac61}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$ROOT_DIR/third_party/IsaacLab}"
ISAACLAB_COMMIT=eae3f3cfcdc61342c6fd5b12d5ed9316e97a92e7
export UV_NO_CONFIG=1
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/"; exit 1; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo "Isaac setup requires Linux x86_64" >&2; exit 1; }
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --managed-python --python 3.12 "$VENV"
fi
"$VENV/bin/python" - "$VENV" <<'PY'
import os, platform, sys
assert sys.version_info[:2] == (3, 12), 'Isaac requires a separate Python 3.12 environment'
# Conda interpreters load their bundled libstdc++, which lacks GLIBCXX_3.4.30 for Kit.
assert not os.path.isdir(os.path.join(sys.base_prefix, 'conda-meta')), \
    f'{sys.argv[1]} uses a conda Python ({sys.base_prefix}); remove it and rerun to create a uv-managed venv'
name, version = platform.libc_ver()
assert name == 'glibc' and tuple(map(int, version.split('.'))) >= (2, 35), 'Isaac 6.1 wheels require glibc >=2.35 (Ubuntu 22.04+)'
PY
UV_FLAGS=(--python "$VENV/bin/python" --index-strategy unsafe-best-match --prerelease allow)
if [[ ! -d "$ISAACLAB_DIR/.git" ]]; then
  mkdir -p "$ISAACLAB_DIR"
  git -C "$ISAACLAB_DIR" init -q
  git -C "$ISAACLAB_DIR" remote add origin https://github.com/isaac-sim/IsaacLab.git
  git -C "$ISAACLAB_DIR" fetch --depth 1 origin "$ISAACLAB_COMMIT"
  git -C "$ISAACLAB_DIR" checkout --detach "$ISAACLAB_COMMIT"
fi
if [[ "$(git -C "$ISAACLAB_DIR" rev-parse HEAD)" != "$ISAACLAB_COMMIT" ]] || [[ -n "$(git -C "$ISAACLAB_DIR" status --porcelain)" ]]; then
  echo "Isaac Lab must be clean at $ISAACLAB_COMMIT. Use a new ISAACLAB_DIR to preserve your changes." >&2
  exit 1
fi
DEPS_FILE="$(mktemp)"
OVERRIDES_FILE="$(mktemp)"
trap 'rm -f "$DEPS_FILE" "$OVERRIDES_FILE"' EXIT
# Upstream explicitly owns overrides for packaging, Newton, MuJoCo and torch.
# Constraints alone cannot replace Isaac Sim's incompatible transitive requirements.
"$VENV/bin/python" - "$ISAACLAB_DIR/pyproject.toml" "$DEPS_FILE" "$OVERRIDES_FILE" <<'PY'
import pathlib, re, sys, tomllib
with open(sys.argv[1], 'rb') as stream:
    metadata = tomllib.load(stream)
torch_names = ('torch', 'torchvision', 'torchaudio')
versions = metadata['tool']['isaaclab']['versions']
requirements = [f"isaacsim[all,extscache]=={versions['isaacsim']}"]
for requirement in metadata['project']['dependencies']:
    name = re.split(r'[<>=!~;\[ ]', requirement.strip())[0].lower().replace('_', '-')
    if not name.startswith('isaaclab'):
        requirements.append(requirement)
overrides = [
    requirement for requirement in metadata['tool']['uv']['override-dependencies']
    if re.split(r'[<>=!~;\[ ]', requirement.strip())[0].lower() not in torch_names
]
# Match upstream's Linux x86_64 CUDA 12.8 index routing. A plain 2.11.0 pin
# would also accept an existing +cu130 wheel because PEP 440 ignores that suffix.
overrides.extend(f'{name}=={versions[name]}+cu128' for name in torch_names)
pathlib.Path(sys.argv[2]).write_text('\n'.join(requirements) + '\n')
pathlib.Path(sys.argv[3]).write_text('\n'.join(overrides) + '\n')
PY
EDITABLES=()
for pkg in isaaclab isaaclab_ppisp isaaclab_contrib isaaclab_newton isaaclab_physx isaaclab_assets; do
  EDITABLES+=(-e "$ISAACLAB_DIR/source/$pkg")
done
echo "[setup] pinned Isaac Sim/Lab stack (large download; allow at least 50 GB free disk)"
# Resolve the whole supported environment together, applying overrides to every requester.
# Prefer public package metadata, as upstream does for omniverseclient/OV libraries.
uv pip install "${UV_FLAGS[@]}" \
  --index https://pypi.org/simple --index https://pypi.nvidia.com \
  --index https://download.pytorch.org/whl/cu128 \
  --overrides "$OVERRIDES_FILE" -r "$DEPS_FILE" "${EDITABLES[@]}" \
  svgpathtools==1.6.1 imageio-ffmpeg
"$VENV/bin/python" - <<'PY'
from importlib.metadata import version
for name in ('torch', 'torchvision', 'torchaudio'):
    assert version(name).endswith('+cu128'), f'{name} must use the pinned CUDA 12.8 wheel'
PY
SITE="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
uv pip install --python "$VENV/bin/python" --no-deps -e "$ROOT_DIR"
uv pip freeze --python "$VENV/bin/python" > "$VENV/installed-requirements.txt"
cp "$OVERRIDES_FILE" "$VENV/isaaclab-overrides.txt"
printf '%s\n' "$ISAACLAB_COMMIT" > "$VENV/isaaclab-commit.txt"
echo "[setup] ready: $VENV (run the documented generation smoke test to check GPU rendering)"
