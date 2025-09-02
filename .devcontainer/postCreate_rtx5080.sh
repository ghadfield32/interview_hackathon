#!/usr/bin/env bash
set -euo pipefail

echo "## postCreate RTX 5080 Optimized (idempotent) ##"

PROJECT_DIR=${PROJECT_DIR:-/workspace}
DEV_DIR="${PROJECT_DIR}/.devcontainer"
TARGET_DIR="${PROJECT_DIR}"
if [ ! -f "${TARGET_DIR}/pyproject.toml" ]; then
  TARGET_DIR="${DEV_DIR}"
fi

# Ensure .env exists first
bash "${DEV_DIR}/setup_env.sh"

export UV_PROJECT_ENVIRONMENT=/app/.venv

# --- CRITICAL: Environment hygiene for JAX ---
echo "[postCreate] Clearing JAX platform forcing for RTX 5080 compatibility..."
unset JAX_PLATFORM_NAME || true
unset JAX_PLATFORMS || true

# --- Memory management for RTX 5080 ---
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25

echo "[postCreate] Using uv=$(uv --version || true)"

# Upgrade pip tooling inside uv venv for reliability
uv pip install --upgrade pip wheel setuptools

# Dependency hash sentinel
mkdir -p "${DEV_DIR}"
dep_hash="$(
  { [ -f "${TARGET_DIR}/pyproject.toml" ] && cat "${TARGET_DIR}/pyproject.toml"; } 2>/dev/null
  { [ -f "${TARGET_DIR}/uv.lock" ] && cat "${TARGET_DIR}/uv.lock"; } 2>/dev/null
  ) | sha256sum | awk '{print $1}'
"
sentinel="${DEV_DIR}/.postcreate.hash"
prev_hash="$( [ -f "${sentinel}" ] && cat "${sentinel}" || echo "" )"

if [ "${dep_hash}" != "${prev_hash}" ]; then
  echo "[postCreate] Dependency hash changed or first run → uv sync"
  (uv sync --frozen || uv sync)
  echo "${dep_hash}" > "${sentinel}"
else
  echo "[postCreate] Dependencies up-to-date; skipping uv sync"
fi

# ---- PyTorch (CUDA 12.8) ----
echo "[postCreate] Installing PyTorch nightly for RTX 5080 (CUDA 12.8)..."
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# ---- JAX (CUDA 12) ----
echo "[postCreate] Installing JAX with CUDA 12 support for RTX 5080..."
# Remove any existing JAX installations to avoid conflicts
uv pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt || true
# Install JAX with proper CUDA 12 support
uv pip install -U "jax[cuda12]"

# ---- TensorFlow removed for efficiency ----
echo "[postCreate] TensorFlow installation skipped (removed for efficiency)"

# ---- Quick version check ----
python - <<'PY'
import os, importlib.util
print("=== RTX 5080 Framework Versions ===")
try:
    import torch; print("torch:", torch.__version__, "cuda:", torch.version.cuda)
except Exception as e: print("torch import failed:", e)
try:
    import jax, jaxlib; print("jax:", jax.__version__, "jaxlib:", jaxlib.__version__)
except Exception as e: print("jax import failed:", e)
# TensorFlow removed for efficiency
print("tf: removed for efficiency")
print("JAX_PLATFORM_NAME =", os.getenv("JAX_PLATFORM_NAME"))
print("===================================")
PY

# ---- Ipykernel setup ----
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

# ---- Framework verification (non-fatal) ----
echo "[postCreate] Running framework verification..."
python "${DEV_DIR}/verify_env.py" || {
    echo "[postCreate] Framework verification had issues - check logs above"
    echo "[postCreate] This is non-fatal - container will continue"
}

echo "## postCreate RTX 5080 DONE ##"


