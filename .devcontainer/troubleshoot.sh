#!/bin/bash
set -euo pipefail

# DevContainer GPU Troubleshooting Script
# This script provides comprehensive diagnostics for GPU-enabled ML development environments

echo "🔍 DevContainer GPU Troubleshooting Report"
echo "=========================================="
echo "Generated: $(date)"
echo ""

# Function for logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function for section headers
section() {
    echo ""
    echo "📋 $1"
    echo "$(printf '%.0s-' {1..50})"
}

# Function for status indicators
status_ok() { echo "✅ $1"; }
status_warn() { echo "⚠️  $1"; }
status_error() { echo "❌ $1"; }
status_info() { echo "ℹ️  $1"; }

# Phase 1: System Prerequisites
section "System Prerequisites"

# Check if we're in a container
if [ -f /.dockerenv ]; then
    status_ok "Running in Docker container"
else
    status_warn "Not running in Docker container"
fi

# Check OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    status_info "OS: $PRETTY_NAME"
else
    status_warn "Could not determine OS"
fi

# Check Python environment
section "Python Environment"

PYTHON_VERSION=$(python --version 2>&1 || echo "Python not found")
status_info "Python: $PYTHON_VERSION"

if [ -n "${VIRTUAL_ENV:-}" ]; then
    status_ok "Virtual environment active: $VIRTUAL_ENV"
else
    status_warn "No virtual environment detected"
fi

# Check uv
if command -v uv >/dev/null 2>&1; then
    UV_VERSION=$(uv --version)
    status_ok "uv available: $UV_VERSION"
else
    status_error "uv not found"
fi

# Phase 2: GPU Hardware & Drivers
section "GPU Hardware & Drivers"

# Check for NVIDIA drivers
if command -v nvidia-smi >/dev/null 2>&1; then
    status_ok "nvidia-smi available"

    # Try to run nvidia-smi
    if nvidia-smi >/dev/null 2>&1; then
        status_ok "GPU accessible"

        # Get GPU info
        echo ""
        echo "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader,nounits | while IFS=, read -r name memory driver compute; do
            echo "  GPU: $name"
            echo "  Memory: $memory MB"
            echo "  Driver: $driver"
            echo "  Compute Capability: $compute"
        done
    else
        status_error "nvidia-smi failed to run"
    fi
else
    status_error "nvidia-smi not available"
fi

# Check CUDA installation
section "CUDA Installation"

if [ -d "/usr/local/cuda" ]; then
    CUDA_VERSION=$(cat /usr/local/cuda/version.txt 2>/dev/null | grep "CUDA Version" | cut -d' ' -f3 || echo "Unknown")
    status_ok "CUDA installed: $CUDA_VERSION"

    # Check CUDA libraries
    if [ -f "/usr/local/cuda/lib64/libcudart.so" ]; then
        status_ok "CUDA runtime libraries found"
    else
        status_warn "CUDA runtime libraries not found"
    fi

    if [ -f "/usr/local/cuda/lib64/libcudnn.so" ]; then
        status_ok "cuDNN libraries found"
    else
        status_warn "cuDNN libraries not found"
    fi
else
    status_error "CUDA not installed"
fi

# Check environment variables
section "GPU Environment Variables"

ENV_VARS=(
    "NVIDIA_VISIBLE_DEVICES"
    "CUDA_VISIBLE_DEVICES"
    "LD_LIBRARY_PATH"
    "PATH"
    "XLA_PYTHON_CLIENT_PREALLOCATE"
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
    "PYTORCH_CUDA_ALLOC_CONF"
    "TF_FORCE_GPU_ALLOW_GROWTH"
)

for var in "${ENV_VARS[@]}"; do
    if [ -n "${!var:-}" ]; then
        status_ok "$var: ${!var}"
    else
        status_warn "$var: not set"
    fi
done

# Phase 3: ML Framework Installation
section "ML Framework Installation"

# Check PyTorch
if python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
    PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
    status_ok "PyTorch installed: $PYTORCH_VERSION"

    # Check CUDA support
    if python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
        CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
        if [ "$CUDA_AVAILABLE" = "True" ]; then
            status_ok "PyTorch CUDA support: Available"
            CUDA_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
            status_info "CUDA devices: $CUDA_COUNT"
        else
            status_warn "PyTorch CUDA support: Not available"
        fi
    fi
else
    status_error "PyTorch not installed"
fi

# Check JAX
if python -c "import jax; print(f'JAX {jax.__version__}')" 2>/dev/null; then
    JAX_VERSION=$(python -c "import jax; print(jax.__version__)" 2>/dev/null)
    status_ok "JAX installed: $JAX_VERSION"

    # Check JAX devices
    if python -c "import jax; print('JAX devices:', jax.devices())" 2>/dev/null; then
        JAX_DEVICES=$(python -c "import jax; print(jax.devices())" 2>/dev/null)
        status_info "JAX devices: $JAX_DEVICES"

        # Check for GPU devices
        if echo "$JAX_DEVICES" | grep -q "gpu\|cuda"; then
            status_ok "JAX GPU support: Available"
        else
            status_warn "JAX GPU support: CPU only"
        fi
    fi
else
    status_error "JAX not installed"
fi

# TensorFlow removed for efficiency
status_info "TensorFlow: Removed for efficiency (PyTorch + JAX only)"

# Phase 4: Package Manager Status
section "Package Manager Status"

# Check uv sync status
if [ -f "pyproject.toml" ]; then
    status_ok "pyproject.toml found"

    if [ -f "uv.lock" ]; then
        status_ok "uv.lock found"
    else
        status_warn "uv.lock not found (run 'uv sync' to generate)"
    fi

    # Check if dependencies are installed
    if python -c "import pandas, numpy, matplotlib" 2>/dev/null; then
        status_ok "Core dependencies installed"
    else
        status_error "Core dependencies missing"
    fi
else
    status_warn "pyproject.toml not found in current directory"
fi

# Phase 5: Network & Connectivity
section "Network & Connectivity"

# Check internet connectivity
if ping -c 1 8.8.8.8 >/dev/null 2>&1; then
    status_ok "Internet connectivity available"
else
    status_error "No internet connectivity"
fi

# Check PyPI connectivity
if curl -s https://pypi.org/simple/ >/dev/null 2>&1; then
    status_ok "PyPI accessible"
else
    status_error "PyPI not accessible"
fi

# Phase 6: Recommendations
section "Recommendations"

echo "Based on the diagnostics above, here are some recommendations:"
echo ""

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "❌ Install NVIDIA drivers and NVIDIA Container Toolkit"
    echo "   - For Ubuntu: sudo apt-get install nvidia-container-toolkit"
    echo "   - Restart Docker daemon after installation"
fi

if ! python -c "import torch; torch.cuda.is_available()" 2>/dev/null | grep -q "True"; then
    echo "⚠️  PyTorch CUDA support not working"
    echo "   - Check CUDA version compatibility"
    echo "   - Reinstall PyTorch with: uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
fi

if ! python -c "import jax; 'gpu' in str(jax.devices()).lower()" 2>/dev/null; then
    echo "⚠️  JAX GPU support not working"
    echo "   - Install JAX with CUDA: uv pip install 'jax[cuda12-local]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
fi

if [ ! -f "uv.lock" ]; then
    echo "⚠️  Run 'uv sync' to install dependencies"
fi

echo ""
echo "🔍 Troubleshooting completed at $(date)"



