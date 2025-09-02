# api/app/utils/env_sanitizer.py
"""
Early‑process clean‑up of env variables that mis‑configure JAX / PyTensor.
Import *before* anything touches JAX / PyMC.
"""

from __future__ import annotations
import os, logging, importlib.util

log = logging.getLogger(__name__)

_VALID_XLA_PREFIXES = ("--xla_", "--mmap_", "--tfrt_")

def _clean_xla_flags() -> None:
    """Remove invalid XLA_FLAGS tokens that cause crashes."""
    val = os.getenv("XLA_FLAGS")
    if not val:
        return
    tokens = [t for t in val.split() if t]
    bad = [t for t in tokens if not t.startswith(_VALID_XLA_PREFIXES)]
    if bad:
        log.warning("🧹 Removing invalid XLA_FLAGS tokens: %s", bad)
        tokens = [t for t in tokens if t not in bad]
    if tokens:
        os.environ["XLA_FLAGS"] = " ".join(tokens)
    else:        # was just '--'
        os.environ.pop("XLA_FLAGS", None)

def _downgrade_jax_backend() -> None:
    """Force JAX to use CPU if GPU is requested but not available."""
    # Check if GPU backend is explicitly requested
    platform_name = os.getenv("JAX_PLATFORM_NAME", "").lower()
    if platform_name in ("gpu", "cuda"):
        # Check if CUDA runtime is actually available
        cuda_spec = importlib.util.find_spec("jaxlib.cuda_extension")
        if cuda_spec is None:
            log.warning("⚠️ No CUDA runtime found – forcing JAX_PLATFORM_NAME=cpu")
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
        else:
            log.info("✅ CUDA runtime detected, keeping GPU backend")

def _force_pytensor_cpu() -> None:
    """Force PyTensor to use CPU device to avoid C++ compilation issues."""
    # Only set if not already configured
    if "PYTENSOR_FLAGS" not in os.environ:
        os.environ["PYTENSOR_FLAGS"] = "device=cpu,floatX=float32"
        log.info("🔧 Set PyTensor to CPU device")

    # Also set legacy config for compatibility
    if "DEVICE" not in os.environ:
        os.environ["DEVICE"] = "cpu"

def _disable_pytensor_compilation() -> None:
    """Completely disable PyTensor C compilation to avoid MSVC issues."""
    # Force PyTensor to use Python backend instead of C compilation
    os.environ["PYTENSOR_FLAGS"] = "device=cpu,floatX=float32"

    # Disable C compilation entirely
    os.environ["PYTENSOR_COMPILE_OPTIMIZER"] = "fast_compile"
    os.environ["PYTENSOR_COMPILE_MODE"] = "FAST_COMPILE"

    # Force Python backend for PyTensor (no C compilation)
    os.environ["PYTENSOR_LINKER"] = "py"

    log.info("🔧 Disabled PyTensor C compilation, using Python backend")

def _check_cuda_environment() -> None:
    """Log CUDA-related environment variables for debugging."""
    cuda_vars = {k: v for k, v in os.environ.items() 
                 if 'CUDA' in k or 'GPU' in k or 'JAX' in k}
    if cuda_vars:
        log.info("🔍 CUDA/JAX environment variables: %s", cuda_vars)

def fix_ml_backends() -> None:
    """
    Comprehensive fix for JAX/PyTensor backend configuration.

    This function should be called **once** at the very top of app.main
    before any JAX or PyMC imports.
    """
    log.info("🔧 Sanitizing ML backend configuration...")

    _check_cuda_environment()
    _clean_xla_flags()
    _downgrade_jax_backend()
    _force_pytensor_cpu()
    _disable_pytensor_compilation()

    log.info("✅ ML backend sanitization complete")

# Legacy function for backward compatibility
def fix_xla_flags() -> None:
    """Legacy function - now calls the comprehensive fix."""
    fix_ml_backends() 
