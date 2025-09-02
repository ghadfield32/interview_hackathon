#!/usr/bin/env python3
import sys, importlib, os, textwrap

CRIT = []
WARN = []

def _have(mod: str) -> bool:
    try:
        importlib.import_module(mod); return True
    except Exception:
        return False

def _msg_box(title: str, body: str) -> None:
    line = "=" * 80
    print(f"\n{line}\n{title}\n{line}\n{body}\n")

def _probe_jax() -> None:
    """
    Probe JAX availability and devices; detect duplicate PJRT CUDA packages and
    mixed CUDA stacks (system + pip nvidia-*), which commonly trigger allocator
    double-frees inside XLA/PJRT.
    """
    import importlib.util as u
    import importlib, os, textwrap, subprocess

    jp = os.environ.get("JAX_PLATFORM_NAME", "<unset>")
    print(f"   JAX_PLATFORM_NAME: {jp}")

    try:
        jax = importlib.import_module("jax")
    except Exception as e:
        WARN.append(f"jax not importable: {e!r}")
        _msg_box(
            "Action: JAX not importable",
            "• Install JAX into /app/.venv:\n"
            "    uv pip install 'jax[cuda12-local]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html\n"
            "• Avoid bare 'pip'; prefer 'uv pip' with UV_PROJECT_ENVIRONMENT=/app/.venv ."
        )
        return

    try:
        jaxlib = importlib.import_module("jaxlib")
        print(f"   jax: {getattr(jax,'__version__','?')} @ {getattr(jax,'__file__','?')}")
        print(f"   jaxlib @ {getattr(jaxlib,'__file__','?')}")
    except Exception as e:
        WARN.append(f"jaxlib import failed: {e!r}")

    # Detect PJRT packages
    has_plugin = u.find_spec("jax_cuda12_plugin") is not None
    has_pjrt   = u.find_spec("jax_cuda12_pjrt") is not None
    if has_plugin and has_pjrt:
        WARN.append("Both jax-cuda12-plugin and jax-cuda12-pjrt are installed (conflict).")
        _msg_box(
            "Conflict: Two JAX PJRT CUDA backends found",
            textwrap.dedent("""\
                You must keep exactly one of these:
                  • jax-cuda12-plugin  (for LOCAL /usr/local/cuda)
                  • jax-cuda12-pjrt    (bundled runtime)
                Recommended (in this image): keep the plugin and remove pjrt:
                  uv pip uninstall -y jax-cuda12-pjrt
            """),
        )

    # Detect pip NVIDIA CUDA stacks if using local plugin
    def _pip_freeze_contains(prefix: str) -> bool:
        try:
            out = subprocess.check_output([os.environ.get("UV_BIN","uv"), "pip", "freeze"], text=True)
        except Exception:
            try:
                out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
            except Exception:
                return False
        return any(line.strip().startswith(prefix) for line in out.splitlines())

    if has_plugin:
        offending = [p for p in (
            "nvidia-cuda-runtime-cu12", "nvidia-cudnn-cu12", "nvidia-cublas-cu12",
            "nvidia-cusolver-cu12", "nvidia-cusparse-cu12", "nvidia-cufft-cu12",
            "nvidia-curand-cu12", "nvidia-nvtx-cu12", "nvidia-nvjitlink-cu12", "nvidia-cuda-cupti-cu12"
        ) if _pip_freeze_contains(p)]
        if offending:
            WARN.append(f"Local CUDA policy but pip NVIDIA libs present: {', '.join(offending)}")
            _msg_box(
                "Mixed CUDA stacks detected",
                "You're using the local PJRT plugin, but pip-installed NVIDIA CUDA libs are present.\n"
                "Remove them to avoid duplicate allocators:\n"
                "  uv pip uninstall -y " + " ".join(offending)
            )

    # Device probe
    try:
        devs = jax.devices()
        print(f"   jax {getattr(jax,'__version__','?')} devices: {devs}")
    except Exception as e:
        WARN.append(f"jax.devices() raised: {e!r}")
        _msg_box(
            "Action: Fix JAX GPU backend",
            textwrap.dedent("""\
                • Install the PJRT CUDA plugin (local CUDA policy):
                  uv pip install "jax[cuda12-local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
                • Ensure only one backend (plugin XOR pjrt) is installed.
                • Avoid mixing pip 'nvidia-*' CUDA stacks with the system CUDA.
            """),
        )
        return

    # CPU-only hints
    gpu = [d for d in devs if "gpu" in str(d).lower() or "cuda" in str(d).lower()]
    if not gpu:
        WARN.append("JAX imported but reports CPU-only devices.")
        _msg_box(
            "Info: JAX is CPU-only right now",
            textwrap.dedent("""\
                Likely causes (check in order):
                1) CUDA plugin not installed in this venv:
                   uv pip install "jax[cuda12-local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
                2) Backend conflict (both plugin and pjrt installed) — remove one.
                3) Mixed CUDA stacks (remove pip 'nvidia-*' libs when using local CUDA).
            """),
        )


def main():
    print("## Python & library diagnostics ##")
    print("Python:", sys.version.split()[0])
    print("sys.executable:", sys.executable)
    print("sys.prefix:", sys.prefix)
    print("VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV","<unset>"))
    print("PATH head:", os.environ.get("PATH","").split(":")[:3])

    if not sys.executable.startswith("/app/.venv/"):
        CRIT.append("Interpreter is not /app/.venv — uv env not active for this process")

    jlab_ok = _have("jupyterlab")
    print(f" - jupyterlab: {'OK' if jlab_ok else 'MISSING'}")

    torch_ok = _have("torch")
    print(f" - torch: {'OK' if torch_ok else 'MISSING'}")
    if torch_ok:
        try:
            import torch
            print("   torch", torch.__version__, "CUDA available:", torch.cuda.is_available())
        except Exception as e:
            WARN.append(f"torch import ok but CUDA probe errored: {e}")

    print(f" - jax: {'OK' if _have('jax') else 'MISSING'}")
    _probe_jax()

    if CRIT:
        _msg_box("Critical failures", "\n".join(f"• {m}" for m in CRIT))
        sys.exit(1)

    if WARN:
        _msg_box("Warnings (non-blocking)", "\n".join(f"• {m}" for m in WARN))

    print("✅ verify_env completed (warnings above are informational).")
    sys.exit(0)

if __name__ == "__main__":
    main()
