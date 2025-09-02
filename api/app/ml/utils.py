# api/app/ml/utils.py

def configure_pytensor_compiler(*_, **__):
    """
    Stub kept for backward‑compatibility.

    The project now uses the **JAX backend**, so PyTensor never calls a C
    compiler.  This function therefore does nothing and always returns True.
    """
    return True

# ─── LEGACY ALIAS ──────────────────────────────────────────────────────────
# Some early-boot modules import "find_compiler", so we alias it here
find_compiler = configure_pytensor_compiler

