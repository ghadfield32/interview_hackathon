#!/usr/bin/env bash
set -euxo pipefail

echo "[gpu-bootstrap] BEGIN"

PY="/app/.venv/bin/python"
export UV_PROJECT_ENVIRONMENT="/app/.venv"

handle_error() {
  echo "❌ GPU bootstrap error at line: ${1}"
  exit 1
}
trap 'handle_error ${LINENO}' ERR

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [gpu-bootstrap] $1"; }
_have_cmd() { command -v "$1" >/dev/null 2>&1; }

seed_pip() {
  if ! "$PY" -m pip --version >/dev/null 2>&1; then
    log "Seeding pip (ensurepip)"
    "$PY" -m ensurepip --upgrade || true
  fi
}

PIP() {
  if _have_cmd uv; then uv pip "$@"; else seed_pip; "$PY" -m pip "$@"; fi
}
PIP_SHOW() {
  if _have_cmd uv; then uv pip show "$@" || true; else seed_pip; "$PY" -m pip show "$@" || true; fi
}

pick_torch_index() {
  local tag="${CUDA_TAG:-12.8.0}"
  case "$tag" in
    12.1* ) echo "cu121" ;;
    12.4* ) echo "cu124" ;;
    12.5*|12.6*|12.7*|12.8*|12.9* ) echo "cu128" ;;
    * ) echo "cu128" ;;
  esac
}

log "Env:"
echo "  whoami=$(whoami)"
echo "  PY=$PY"
echo "  UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
echo "  CUDA_TAG=${CUDA_TAG:-12.8.0}"

$PY - <<'PY'
import sys, os
print("[gpu-bootstrap] sys.executable:", sys.executable)
print("[gpu-bootstrap] VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV","<unset>"))
for m in ("torch","jax","jaxlib"):
    try:
        mod = __import__(m)
        print(f"[gpu-bootstrap] pre: import {m}: OK from", getattr(mod,"__file__","?"))
    except Exception as e:
        print(f"[gpu-bootstrap] pre: import {m}: FAIL -> {e.__class__.__name__}: {e}")
PY

# GPU availability
if _have_cmd nvidia-smi; then
  log "nvidia-smi present"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits || true
else
  log "No nvidia-smi; GPU likely not available"
fi

unset JAX_PLATFORM_NAME || true

# --- 1) Ensure PyTorch (GPU) -------------------------------------------------
log "Checking PyTorch CUDA availability..."
if $PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  log "✅ PyTorch CUDA available"
else
  if _have_cmd nvidia-smi; then
    IDX="$(pick_torch_index)"
    log "Installing PyTorch (${IDX})"
    PIP install --no-cache-dir torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${IDX}"
  else
    log "⚠️  Skipping PyTorch GPU install (no nvidia-smi)"
  fi
fi

# --- 2) Enforce single JAX PJRT backend (prefer local plugin) ---------------
log "Reconciling JAX backend packages..."
HAS_PLUGIN=0
HAS_PJRT=0
$PY - <<'PY'
import importlib.util, sys
def have(pkg):
    return importlib.util.find_spec(pkg) is not None
print("has_plugin", have("jax_cuda12_plugin"))
print("has_pjrt", have("jax_cuda12_pjrt"))
PY | tee /tmp/jax_backends.txt >/dev/null

if grep -q "has_plugin True" /tmp/jax_backends.txt; then HAS_PLUGIN=1; fi
if grep -q "has_pjrt True" /tmp/jax_backends.txt; then HAS_PJRT=1; fi

if [ "$HAS_PLUGIN" -eq 1 ] && [ "$HAS_PJRT" -eq 1 ]; then
  log "Both jax-cuda backends present → uninstall pjrt"
  PIP uninstall -y jax-cuda12-pjrt || true
  HAS_PJRT=0
fi

# Policy: keep LOCAL plugin (system CUDA) and remove pip NVIDIA runtime libs
if [ "$HAS_PLUGIN" -eq 0 ]; then
  log "Installing jax[cuda12-local] (PJRT CUDA plugin)"
  PIP install --no-cache-dir "jax[cuda12-local]>=0.4.26" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi

# Remove pip-provided NVIDIA CUDA stacks to avoid mixing with system CUDA
log "Removing pip NVIDIA CUDA stacks (if any) to prevent double-lib loads"
PIP uninstall -y \
  nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 \
  nvidia-cuda-cupti-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
  nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
  || true

# Re-probe JAX devices
log "Probing JAX devices..."
$PY - <<'PY'
import sys
try:
    import jax
    print("[gpu-bootstrap] JAX", jax.__version__, "devices:", jax.devices())
    # sanity: ensure at most one CUDA PJRT present
    import importlib.util as u
    print("[gpu-bootstrap] has plugin:", u.find_spec("jax_cuda12_plugin") is not None)
    print("[gpu-bootstrap] has pjrt:", u.find_spec("jax_cuda12_pjrt") is not None)
except Exception as e:
    print("[gpu-bootstrap] JAX import/probe error:", e)
    sys.exit(1)
PY

log "Package snapshot:"
PIP_SHOW jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt torch torchvision torchaudio || true

log "END"
