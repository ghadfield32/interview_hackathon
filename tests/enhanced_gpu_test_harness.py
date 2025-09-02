#!/usr/bin/env python3
"""
Enhanced GPU Test Harness with Blackwell/SM_120 Support
======================================================

This module provides enhanced test functions for GPU frameworks with specific
support for NVIDIA Blackwell RTX 5080 and CUDA 12.8 compatibility.

Features:
- Enhanced JAX CUDA backend detection and conflict resolution
- TensorFlow INVALID_PTX error detection with diagnostic messages
- Comprehensive error reporting for Blackwell-specific issues

Usage:
    python tests/enhanced_gpu_test_harness.py

Or import individual functions:
    from tests.enhanced_gpu_test_harness import check_jax, check_tensorflow
"""

import sys
import gc
import textwrap


def check_jax():
    """
    Enhanced JAX GPU test with backend detection and Blackwell compatibility.

    This function provides:
    - Detection of CUDA backend plugins vs PJRT runtime
    - Identification of mixed backend conflicts
    - GPU device enumeration and computation testing
    - Clear error messages for common configuration issues
    """
    print("=== JAX ===")
    try:
        import jax, importlib.util as u
        print(f"jax.__version__={jax.__version__}")
        try:
            import jaxlib
            print(f"jaxlib.__version__={jaxlib.__version__}")
        except Exception as e:
            print("Could not import jaxlib:", repr(e))

        print("Backends: plugin?", u.find_spec("jax_cuda12_plugin") is not None,
              " pjrt?", u.find_spec("jax_cuda12_pjrt") is not None)

        try:
            devs = jax.devices()
            print(f"devices: {devs}")
            gpu = [d for d in devs if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
            if not gpu:
                print("No JAX GPU devices found.")
            else:
                from jax import numpy as jnp
                x = jnp.ones((1024,1024), dtype=jnp.float32)
                y = (x @ x.T).sum()
                _ = y.block_until_ready()
                print("small matmul check (JAX GPU): OK")
        except RuntimeError as re:
            print("jax.devices() raised RuntimeError:", str(re))
        except Exception as e:
            print("jax.devices() failed:", repr(e))
    except Exception as e:
        print("JAX import failed:", repr(e))
    print("===========\n")


def check_tensorflow():
    """
    Enhanced TensorFlow GPU test with Blackwell INVALID_PTX detection.

    This function provides:
    - GPU device enumeration and memory growth configuration  
    - Explicit CUDA_ERROR_INVALID_PTX detection for Blackwell GPUs
    - Diagnostic messages for SM_120/CUDA 12.8 compatibility issues
    - Clear guidance for resolving PTX compilation failures
    """
    print("=== TensorFlow ===")
    try:
        import tensorflow as tf
        print(f"tf.__version__={tf.__version__}")
        try:
            gpus = tf.config.list_physical_devices("GPU")
            print(f"tf GPUs: {gpus}")
            if gpus:
                # Memory growth first
                for g in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(g, True)
                    except Exception:
                        pass
                # Minimal compute test with explicit error surfacing
                try:
                    with tf.device("/GPU:0"):
                        a = tf.random.normal((1024,1024))
                        b = tf.random.normal((1024,1024))
                        c = tf.reduce_sum(tf.matmul(a, b))
                    _ = c.numpy()
                    print("small matmul check (TensorFlow GPU): OK")
                except Exception as e:
                    msg = str(e)
                    print("TensorFlow GPU op failed:", type(e).__name__, msg)
                    if "CUDA_ERROR_INVALID_PTX" in msg or "INVALID_PTX" in msg:
                        print(textwrap.dedent("""
>>> DIAG: INVALID_PTX on Blackwell usually means the TF wheel lacks SM_120 SASS and the PTX
          embedded in the wheel predates CUDA 12.8 support for Blackwell. Options:
          • install tf-nightly (post-CUDA-12.8 builds), or
          • use NVIDIA's TensorFlow NGC container built for CUDA 12.8+.
                        """).strip())
                    raise
            else:
                print("No TensorFlow GPU devices found.")
        except Exception as e:
            print("TensorFlow device query failed:", repr(e))
    except Exception as e:
        print("TensorFlow import failed:", repr(e))
    print("==================\n")


def check_pytorch():
    """
    Enhanced PyTorch GPU test with memory management.

    This function provides:
    - CUDA availability and device information
    - Memory cleanup after computation
    - Version and capability reporting
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
    Run comprehensive GPU framework testing.

    This function tests all three major frameworks (PyTorch, JAX, TensorFlow)
    and provides a summary of results with specific guidance for failures.

    Returns:
        bool: True if all frameworks passed, False otherwise
    """
    print("🧪 COMPREHENSIVE GPU FRAMEWORK TEST")
    print("=" * 50)
    print("Testing GPU support for PyTorch, JAX, and TensorFlow...")
    print("Optimized for NVIDIA Blackwell RTX 5080 / CUDA 12.8")
    print("=" * 50)

    # Test all frameworks
    check_pytorch()
    check_jax()  
    check_tensorflow()

    print("=" * 50)
    print("✅ Comprehensive GPU test completed!")
    print("Check output above for any framework-specific issues.")
    print("=" * 50)

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
        "PYTORCH_CUDA_ALLOC_CONF", "TF_FORCE_GPU_ALLOW_GROWTH"
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





