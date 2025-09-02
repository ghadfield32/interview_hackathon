#!/usr/bin/env bash
set -euo pipefail
echo "## postCreate (idempotent) ##"

PROJECT_DIR=${PROJECT_DIR:-/workspace}
DEV_DIR="${PROJECT_DIR}/.devcontainer"
TARGET_DIR="${PROJECT_DIR}"
if [ ! -f "${TARGET_DIR}/pyproject.toml" ]; then
  TARGET_DIR="${DEV_DIR}"
fi

# Ensure .env exists first
bash "${DEV_DIR}/setup_env.sh"

export UV_PROJECT_ENVIRONMENT=/app/.venv

# Dependency hash sentinel
mkdir -p "${DEV_DIR}"
dep_hash="$(
  { [ -f "${TARGET_DIR}/pyproject.toml" ] && cat "${TARGET_DIR}/pyproject.toml"; } 2>/dev/null
  { [ -f "${TARGET_DIR}/uv.lock" ] && cat "${TARGET_DIR}/uv.lock"; } 2>/dev/null
  ) | sha256sum | awk '{print $1}'
"
sentinel="${DEV_DIR}/.postcreate.hash"
prev_hash="$( [ -f "${sentinel}" ] && cat "${sentinel}" || echo "" )"

uv --version || true

if [ "${dep_hash}" != "${prev_hash}" ]; then
  echo "[postCreate] Dependency hash changed or first run → uv sync"
  (uv sync --frozen || uv sync)
  echo "${dep_hash}" > "${sentinel}"
else
  echo "[postCreate] Dependencies up-to-date; skipping uv sync"
fi

python - <<'PY'
import sys, subprocess
print('[postCreate] Ensuring ipykernel present for', sys.executable)
try:
    import ipykernel  # noqa
except Exception:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ipykernel'])
subprocess.check_call([
    sys.executable, '-m', 'ipykernel', 'install',
    '--name', 'uv-app-venv',
    '--display-name', 'Python (uv /app/.venv)',
    '--user'
])
print('[postCreate] ipykernel installed/updated as uv-app-venv')
PY

echo "[postCreate] Optional PyJAGS"
CPPFLAGS='-include cstdint' uv pip install --no-build-isolation pyjags==1.3.8 || true

python "${DEV_DIR}/verify_env.py" || true
echo "## postCreate DONE ##"
