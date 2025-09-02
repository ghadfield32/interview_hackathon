#!/usr/bin/env python3
"""
Test script to verify that PyTorch and JAX can access the GPU,
and that PyJAGS is working correctly.
"""

import sys


def test_pytorch_gpu():
    """Test PyTorch GPU availability and basic operations."""
    print("\n=== Testing PyTorch GPU ===")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if not torch.cuda.is_available():
            print("❌ PyTorch CUDA not available!")
            return False

        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

        # Run a simple test computation
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        z = torch.matmul(x, y)
        end.record()

        # Wait for GPU computation to finish
        torch.cuda.synchronize()
        print(f"Matrix multiplication time: {start.elapsed_time(end):.2f} ms")
        print(f"Result shape: {z.shape}")
        print("✅ PyTorch GPU test passed!")
        return True

    except ImportError:
        print("❌ PyTorch not found!")
        return False
    except Exception as e:
        print(f"❌ Error during PyTorch GPU test: {e}")
        return False


def test_jax_gpu():
    """Test JAX GPU availability and basic operations."""
    print("\n=== Testing JAX GPU ===")
    try:
        import jax, jax.numpy as jnp
        print(f"JAX version: {jax.__version__}")

        # Probe both: this won't crash if GPU plugin isn't present
        gpu_devs = []
        try:
            gpu_devs = jax.devices("gpu")
        except Exception:
            # Some builds don't register 'gpu' backend explicitly
            pass

        if not gpu_devs:
            # Fallback: inspect all devices for cuda/gpu strings
            devs = jax.devices()
            gpu_devs = [d for d in devs if "gpu" in str(d).lower() or "cuda" in str(d).lower()]

        if not gpu_devs:
            print("❌ No GPU devices found by JAX!")
            return False

        print(f"Available JAX GPU devices: {gpu_devs}")
        @jax.jit
        def matmul(a, b): return jnp.matmul(a, b)
        x = jnp.ones((1024, 1024))
        y = jnp.ones((1024, 1024))
        result = matmul(x, y)
        print(f"Result shape: {result.shape}")
        print("✅ JAX GPU test passed!")
        return True

    except ImportError:
        print("❌ JAX not found!")
        return False
    except Exception as e:
        print(f"❌ Error during JAX GPU test: {e}")
        return False



def test_pyjags():
    """Test PyJAGS installation and basic functionality."""
    print("\n=== Testing PyJAGS ===")
    try:
        import pyjags
        print(f"PyJAGS version: {pyjags.__version__}")

        # Create a simple model to verify that PyJAGS works
        code = """
        model {
            # Likelihood
            y ~ dnorm(mu, 1/sigma^2)

            # Priors
            mu ~ dnorm(0, 0.001)
            sigma ~ dunif(0, 100)
        }
        """

        # Sample data
        data = {'y': 0.5}

        # Initialize model with data
        model = pyjags.Model(code, data=data, chains=1, adapt=100)
        print("JAGS model initialized successfully!")

        # Sample from the model
        samples = model.sample(200, vars=['mu', 'sigma'])
        print("JAGS sampling completed successfully!")

        # Verify the samples
        mu_samples = samples['mu']
        sigma_samples = samples['sigma']
        print(f"mu mean: {mu_samples.mean():.4f}")
        print(f"sigma mean: {sigma_samples.mean():.4f}")

        print("✅ PyJAGS test passed!")
        return True

    except ImportError:
        print("❌ PyJAGS not found!")
        return False
    except Exception as e:
        print(f"❌ Error during PyJAGS test: {e}")
        return False


if __name__ == "__main__":
    print("Running GPU and PyJAGS verification tests...")

    pytorch_success = test_pytorch_gpu()
    jax_success = test_jax_gpu()
    pyjags_success = test_pyjags()

    print("\n=== Test Summary ===")
    print(f"PyTorch GPU: {'✅ PASS' if pytorch_success else '❌ FAIL'}")
    print(f"JAX GPU: {'✅ PASS' if jax_success else '❌ FAIL'}")
    print(f"PyJAGS: {'✅ PASS' if pyjags_success else '❌ FAIL'}")

    if pytorch_success and jax_success and pyjags_success:
        print("\n🎉 All tests passed! The container is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the output for details.")
        sys.exit(1)


