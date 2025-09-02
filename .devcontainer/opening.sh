#!/usr/bin/env bash
set -Eeuo pipefail

log() { printf "\n[%s] %s\n" "$(date +'%F %T')" "$*" >&2; }
die() { printf "\n[ERROR] %s\n" "$*" >&2; exit 1; }

# Parse boolean-ish env values: 1/true/yes/on → 1, else 0
parse_env_bool() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) echo "1" ;;
    *) echo "0" ;;
  esac
}

# Load /workspace/.env if present, exporting its variables
read_dotenv_if_present() {
  if [ -f "/workspace/.env" ]; then
    log "Sourcing /workspace/.env into the environment"
    sed -i 's/\r$//' /workspace/.env || true
    set -a; . /workspace/.env; set +a
  fi
}

# Normalize line endings for shell scripts
normalize_line_endings() {
  log "Normalizing CRLF -> LF for .devcontainer/*.sh"
  find .devcontainer -maxdepth 1 -type f -name "*.sh" -print0 | \
    xargs -0 -I{} bash -lc 'sed -i "s/\r$//" "{}"'
}

# Ensure git safe.directory
fix_git_safety() {
  if git rev-parse --show-toplevel >/dev/null 2>&1; then
    local root="$(git rev-parse --show-toplevel || echo /workspace)"
    log "Marking ${root} as a safe.directory for git."
    git config --global --add safe.directory "${root}" || true
  fi
}

# Ensure uv targets the project venv
ensure_uv_env() {
  export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/app/.venv}"
  log "UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
  if [[ ! -d "${UV_PROJECT_ENVIRONMENT}" ]]; then
    log "Creating uv venv at ${UV_PROJECT_ENVIRONMENT}"
    uv venv "${UV_PROJECT_ENVIRONMENT}" --python "3.10" --prompt "cancer_bayes_iris_env"
  fi
  echo ". ${UV_PROJECT_ENVIRONMENT}/bin/activate" > /etc/profile.d/10-uv-activate.sh
}

# Sync core project dependencies (prefer root pyproject.toml)
sync_project_deps() {
  # Prefer root pyproject for universal local+container dev
  if [[ -f "/workspace/pyproject.toml" ]]; then
    log "Syncing project deps from /workspace (frozen)"
    (cd /workspace && uv sync --frozen --no-dev) || {
      log "Lock out-of-date; refreshing from /workspace…"
      (cd /workspace && uv sync --no-dev && uv lock)
    }
  elif [[ -f "/workspace/.devcontainer/pyproject.toml" ]]; then
    log "Syncing project deps from .devcontainer (legacy fallback)"
    (cd /workspace/.devcontainer && uv sync --frozen --no-dev) || {
      log "Lock out-of-date; refreshing from .devcontainer…"
      (cd /workspace/.devcontainer && uv sync --no-dev && uv lock)
    }
  else
    die "No pyproject.toml found at /workspace or .devcontainer"
  fi
}

# Simplified memory management for RTX 5080
setup_memory_management() {
  log "Setting up RTX 5080 memory management..."

  # Verify jemalloc
  if python -c "import ctypes; ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libjemalloc.so.2')" 2>/dev/null; then
    log "jemalloc memory allocator loaded successfully"
  else
    log "jemalloc not available - using system malloc"
    unset LD_PRELOAD
  fi

  # Essential memory settings only (removed complex variables)
  export MALLOC_ARENA_MAX=1
  export MALLOC_TCACHE_MAX=0
  export PYTORCH_NO_CUDA_MEMORY_CACHING=1
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25

  log "Memory management configured"
}

# GPU Framework Installation (JAX + PyTorch only, TensorFlow removed)
install_gpu_frameworks() {
  log "Installing GPU frameworks (PyTorch + JAX only)..."

  read_dotenv_if_present

  # Clear JAX platform forcing
  unset JAX_PLATFORM_NAME || true
  unset JAX_PLATFORMS || true

  # Remove conflicting NVIDIA packages to prevent double loads
  log "Cleaning up conflicting NVIDIA packages..."
  uv pip uninstall -y \
    nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 \
    nvidia-cuda-cupti-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
    nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
    || true

  # PyTorch with CUDA 12.8 support
  if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    log "Installing PyTorch nightly cu128..."
    uv pip install --no-cache-dir --pre torch torchvision torchaudio \
      --index-url "https://download.pytorch.org/whl/nightly/cu128"
  else
    log "PyTorch CUDA already available"
  fi

  # JAX with CUDA support  
  log "Installing JAX with CUDA support..."
  uv pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt || true
  uv pip install --no-cache-dir "jax[cuda12-local]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
}

# Framework initialization (PyTorch + JAX only)
initialize_frameworks() {
  log "Initializing PyTorch and JAX..."

  python3 -c "
import os, gc
print('Initializing JAX...')
try:
    import jax, jax.numpy as jnp
    devs = jax.devices()
    print('   JAX devices:', devs)
    x = jnp.ones((128,128)); y = (x @ x.T).sum(); _ = y.block_until_ready()
    print('   JAX small matmul: OK')
except Exception as e:
    print('   JAX init failed:', e)

print('Initializing PyTorch...')
try:
    import torch
    print('   torch:', torch.__version__, 'CUDA avail:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('   device:', torch.cuda.get_device_name(0))
        a = torch.randn(128,128, device='cuda')
        b = torch.randn(128,128, device='cuda')
        _ = (a@b).sum().item()
        print('   PyTorch small matmul: OK')
except Exception as e:
    print('   PyTorch init failed:', e)

gc.collect()
"
}

# Register Jupyter kernel
register_kernel() {
  log "Registering Jupyter ipykernel..."
  python -c "
import json, sys, subprocess
name = 'cancer_bayes_iris_env'
display = 'Python (cancer_bayes_iris_env)'
try:
    subprocess.check_call([sys.executable, '-m', 'ipykernel', 'install',
                           '--sys-prefix', '--name', name, '--display-name', display])
    print('Kernel registered:', display)
except subprocess.CalledProcessError as e:
    print('Kernel registration failed:', e)
    sys.exit(1)
"
}

# Simplified verification (PyTorch + JAX only)
verify_frameworks() {
  log "Verifying frameworks..."

  local failed=0

  # PyTorch verification
  if python -c "
import torch, gc
assert torch.cuda.is_available(), 'CUDA not available'
for _ in range(3):
    x = torch.randn(1000,1000, device='cuda')
    del x; torch.cuda.empty_cache(); gc.collect()
print('PyTorch CUDA verification passed')
" 2>/dev/null; then
    log "PyTorch verification: PASSED"
  else
    log "PyTorch verification: FAILED"
    failed=$((failed+1))
  fi

  # JAX verification  
  if python -c "
import jax, jax.numpy as jnp, gc
devices = jax.devices()
assert any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices), 'No GPU devices'
for _ in range(3):
    x = jnp.ones((1000,1000)); del x; gc.collect()
print('JAX CUDA verification passed')
" 2>/dev/null; then
    log "JAX verification: PASSED"
  else
    log "JAX verification: FAILED"  
    failed=$((failed+1))
  fi

  if [ "${failed}" -gt 0 ]; then
    log "Some frameworks failed verification but continuing..."
  else
    log "All frameworks verified successfully"
  fi
}

# MAIN EXECUTION
main() {
  normalize_line_endings
  read_dotenv_if_present
  fix_git_safety
  ensure_uv_env
  setup_memory_management

  if [[ "${1:-}" == "--verify-only" ]]; then
    verify_frameworks || true
    return 0
  fi

  sync_project_deps
  install_gpu_frameworks
  initialize_frameworks  
  register_kernel
  verify_frameworks
  log "Setup completed successfully"
}

main "$@"
