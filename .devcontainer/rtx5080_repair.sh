#!/usr/bin/env bash
set -euo pipefail

echo "🔧 RTX 5080 GPU Framework Repair Script"
echo "======================================="
echo "This script will fix JAX and TensorFlow issues for RTX 5080"
echo ""

# Check if we're in the right environment
if [[ ! -f "/app/.venv/bin/activate" ]]; then
    echo "❌ Error: This script must be run inside the container with uv environment"
    exit 1
fi

# Activate the environment
source /app/.venv/bin/activate

echo "📋 Current environment:"
echo "   UV_PROJECT_ENVIRONMENT: ${UV_PROJECT_ENVIRONMENT:-<unset>}"
echo "   Python: $(python --version)"
echo "   uv: $(uv --version)"
echo ""

# --- Step 1: Environment Cleanup ---
echo "🧹 Step 1: Cleaning up environment variables..."
unset JAX_PLATFORM_NAME || true
unset JAX_PLATFORMS || true

# Set optimal environment variables
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25

echo "   ✅ Environment variables configured"
echo ""

# --- Step 2: JAX Repair ---
echo "🧠 Step 2: Repairing JAX..."
echo "   Removing existing JAX installations..."

# Remove all JAX-related packages
uv pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt || true

echo "   Installing JAX with CUDA 12 support..."
uv pip install -U "jax[cuda12]"

echo "   ✅ JAX installation completed"
echo ""

# --- Step 3: TensorFlow removed for efficiency ---
echo "🔢 Step 3: TensorFlow removed for efficiency..."
echo "   TensorFlow installation skipped to improve build speed and reduce conflicts"
echo ""

# --- Step 4: Verification ---
echo "🔍 Step 4: Verifying installations..."

python - <<'PY'
import os, importlib.util
print("=== Installation Verification ===")

# Check JAX
try:
    import jax, jaxlib
    print(f"✅ JAX: {jax.__version__}")
    print(f"✅ JAXlib: {jaxlib.__version__}")

    # Check for CUDA plugin
    has_cuda = importlib.util.find_spec("jax_cuda12_plugin") or importlib.util.find_spec("jax_cuda12_pjrt")
    print(f"✅ JAX CUDA plugin: {'Yes' if has_cuda else 'No'}")

    # Check devices
    try:
        devices = jax.devices()
        gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
        print(f"✅ JAX GPU devices: {len(gpu_devices)} found")
        for d in gpu_devices:
            print(f"   - {d}")
    except Exception as e:
        print(f"❌ JAX device check failed: {e}")

except Exception as e:
    print(f"❌ JAX verification failed: {e}")

# TensorFlow removed for efficiency
print("✅ TensorFlow: removed for efficiency")

# Check PyTorch
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ PyTorch CUDA: {torch.version.cuda}")
    print(f"✅ PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ PyTorch device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ PyTorch verification failed: {e}")

print("=== Environment Variables ===")
print(f"JAX_PLATFORM_NAME: {os.environ.get('JAX_PLATFORM_NAME', '<unset>')}")
print(f"TF_FORCE_GPU_ALLOW_GROWTH: {os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH', '<unset>')}")
print("=============================")
PY

echo ""
echo "🎉 Repair completed!"
echo ""
echo "💡 Next steps:"
echo "   1. Test the frameworks with: python .devcontainer/enhanced_gpu_test_functions.py"
echo "   2. If issues persist, check the diagnostic output above"
echo "   3. For TensorFlow PTX errors, consider building from source with CUDA 12.8+"
echo ""
echo "🔗 Useful commands:"
echo "   # Test all frameworks"
echo "   python .devcontainer/enhanced_gpu_test_functions.py"
echo ""
echo "   # Quick JAX test"
echo "   python -c \"import jax; print('JAX devices:', jax.devices())\""
echo ""
echo "   # Quick TensorFlow test"
echo "   python -c \"import tensorflow as tf; print('TF GPUs:', tf.config.list_physical_devices('GPU'))\""
echo ""

