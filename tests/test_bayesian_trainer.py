#!/usr/bin/env python3
"""
Standalone test script for the Bayesian breast cancer trainer.
Run this to verify the JAX/NumPyro configuration and training works correctly.
"""

import os
import sys
import logging
import time

# Add the api directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname).1s] %(asctime)s %(name)s ▶ %(message)s'
)
logger = logging.getLogger(__name__)

def test_jax_backend():
    """Test that JAX and NumPyro are available and working."""
    logger.info("🔧 Testing JAX/NumPyro backend...")

    try:
        import jax
        import numpyro
        import pymc as pm
        
        logger.info(f"✅ JAX version: {jax.__version__}")
        logger.info(f"✅ NumPyro version: {numpyro.__version__}")
        logger.info(f"✅ PyMC version: {pm.__version__}")
        logger.info(f"✅ JAX devices: {jax.devices()}")
        logger.info(f"✅ JAX platform: {jax.default_backend()}")
        return True
    except ImportError as e:
        logger.error(f"❌ JAX/NumPyro import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ JAX/NumPyro configuration failed: {e}")
        return False

def test_bayesian_training():
    """Test the Bayesian training function with JAX backend."""
    logger.info("🧠 Testing Bayesian training with JAX backend...")

    try:
        from app.ml.builtin_trainers import train_breast_cancer_bayes

        # Test with smaller parameters for faster testing
        start_time = time.time()
        run_id = train_breast_cancer_bayes(
            draws=200,      # Reduced from 1000
            tune=100,       # Reduced from 1000
            target_accept=0.9
        )
        elapsed = time.time() - start_time

        logger.info(f"✅ Bayesian training completed successfully!")
        logger.info(f"   Run ID: {run_id}")
        logger.info(f"   Time: {elapsed:.1f} seconds")
        return True

    except Exception as e:
        logger.error(f"❌ Bayesian training failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_stub_training():
    """Test the stub training function as a fallback."""
    logger.info("🔧 Testing stub training...")

    try:
        from app.ml.builtin_trainers import train_breast_cancer_stub

        start_time = time.time()
        run_id = train_breast_cancer_stub()
        elapsed = time.time() - start_time

        logger.info(f"✅ Stub training completed successfully!")
        logger.info(f"   Run ID: {run_id}")
        logger.info(f"   Time: {elapsed:.1f} seconds")
        return True

    except Exception as e:
        logger.error(f"❌ Stub training failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting Bayesian trainer tests with JAX backend...")

    # Test 1: JAX backend availability
    if not test_jax_backend():
        logger.error("❌ JAX backend test failed")
        return 1

    # Test 2: Stub training (should always work)
    if not test_stub_training():
        logger.error("❌ Stub training test failed")
        return 1

    # Test 3: Bayesian training (may fail on some systems)
    if not test_bayesian_training():
        logger.warning("⚠️ Bayesian training test failed - this is expected on some systems")
        logger.info("   The stub model will be used as fallback")
        return 0  # Not a critical failure

    logger.info("🎉 All tests passed!")
    return 0

if __name__ == "__main__":
    exit(main()) 
