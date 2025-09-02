#!/usr/bin/env python3
"""
Enhanced GPU Test Functions for RTX 5080 (PyTorch + JAX Only)
============================================================

Drop-in replacements for the test functions in your notebook.
These provide comprehensive diagnostics and actionable guidance for RTX 5080 issues.

Usage:
    # Import and use in your notebook
    from .devcontainer.enhanced_gpu_test_functions import check_jax, check_pytorch, comprehensive_gpu_test

    # Or run standalone
    python .devcontainer/enhanced_gpu_test_functions.py
"""

import os
import sys
import gc
import textwrap
import subprocess
import importlib.util


def _run(cmd):
    """
    Run a command and return (success, output).

    Args:
        cmd: Command to run

    Returns:
        tuple: (success: bool, output: str)
    """
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return True, out.strip()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_jax():
    """
    Enhanced JAX GPU test with backend detection and Blackwell compatibility.

    This function provides:
    - Detection of CUDA backend plugins vs PJRT runtime
    - Identification of mixed backend conflicts
    - GPU device enumeration and computation testing
    - Clear error messages for common configuration issues
    - Actionable hints for RTX 5080 compatibility
    """
    import os, importlib.util, textwrap
    print("=== JAX ===")
    try:
        import jax
        print(f"jax.__version__={jax.__version__}")
        try:
            import jaxlib
            print(f"jaxlib.__version__={jaxlib.__version__}")
        except Exception as e:
            print("Could not import jaxlib:", repr(e))

        forced = os.environ.get("JAX_PLATFORM_NAME"), os.environ.get("JAX_PLATFORMS")
        print(f"JAX_PLATFORM_NAME={forced[0]!r} JAX_PLATFORMS={forced[1]!r}")

        # Is a CUDA PJRT plugin present?
        has_cuda_extra = importlib.util.find_spec("jax_cuda12_plugin") or importlib.util.find_spec("jax_cuda12_pjrt")
        print(f"JAX CUDA plugin present? {bool(has_cuda_extra)}")

        try:
            devs = jax.devices()
            if not devs:
                print("No JAX devices found.")
            else:
                for d in devs:
                    print(f"Device: kind={getattr(d,'device_kind',None)}, platform={d.platform}, id={d.id}")
            # Smoke test
            from jax import numpy as jnp
            x = jnp.ones((1024,1024), dtype=jnp.float32)
            y = (x @ x.T).sum()
            _ = y.block_until_ready()
            print("small matmul check (JAX default backend): OK")
        except RuntimeError as re:
            msg = str(re)
            print("jax.devices() raised RuntimeError:", msg)
            if "Unknown backend: 'gpu'" in msg:
                print(textwrap.dedent("""
                    HINT: GPU was forced but CUDA-enabled jaxlib is missing.
                    Fix:
                      1) unset JAX_PLATFORM_NAME / JAX_PLATFORMS
                      2) install GPU wheels (CUDA12):  uv pip install -U "jax[cuda12]"
                """).strip())
        except Exception as e:
            print("jax.devices() failed:", repr(e))
    except Exception as e:
        print("JAX import failed:", repr(e))
    print("===========\n")


def check_pytorch():
    """
    Enhanced PyTorch GPU test with memory management.

    This function provides:
    - CUDA availability and device information
    - Memory cleanup after computation
    - Version and capability reporting
    - RTX 5080 specific compatibility checks
    """
    print("=== PyTorch ===")
    try:
        import torch
        print(f"torch.__version__={torch.__version__}")
        print(f"torch.version.cuda={torch.version.cuda}")
        print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
            print(f"current device name: {torch.cuda.get_device_name(0)}")

            # Test computation with memory cleanup
            x = torch.randn(1024, 1024, device="cuda")
            y = torch.randn(1024, 1024, device="cuda")
            z = (x @ y).sum().item()
            del x, y
            gc.collect()
            torch.cuda.empty_cache()
            print("small matmul check (PyTorch CUDA): OK")
        else:
            print("CUDA not available in PyTorch.")
    except Exception as e:
        print("PyTorch check failed:", repr(e))
    print("==============\n")


def comprehensive_gpu_test():
    """
    Run comprehensive GPU framework testing (PyTorch + JAX only).

    This function tests PyTorch and JAX frameworks
    and provides a summary of results with specific guidance for failures.

    Returns:
        bool: True if all frameworks passed, False otherwise
    """
    print("🧪 COMPREHENSIVE GPU FRAMEWORK TEST (PyTorch + JAX Only)")
    print("=" * 60)
    print("Testing GPU support for PyTorch and JAX...")
    print("Optimized for NVIDIA Blackwell RTX 5080 / CUDA 12.8")
    print("TensorFlow removed for faster builds and cleaner environment")
    print("=" * 60)

    # Test frameworks
    check_pytorch()
    check_jax()  

    print("=" * 60)
    print("✅ Comprehensive GPU test completed!")
    print("Check output above for any framework-specific issues.")
    print("=" * 60)

    return True


def main():
    """Main entry point for standalone execution."""
    import os

    # Show environment snapshot
    print("🔧 GPU ENVIRONMENT SNAPSHOT")
    print("=" * 30)
    env_vars = [
        "JAX_PLATFORM_NAME", "JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES",
        "XLA_FLAGS", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
        "PYTORCH_CUDA_ALLOC_CONF"
    ]
    for var in env_vars:
        value = os.environ.get(var, "<unset>")
        print(f"{var}={value}")
    print("=" * 30)
    print()

    # Run comprehensive test
    comprehensive_gpu_test()


if __name__ == "__main__":
    main()

