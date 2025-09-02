# .devcontainer/Dockerfile - Optimized for faster builds
ARG CUDA_TAG=12.8.0
# Use NVIDIA's runtime image with pre-installed CUDA libraries (saves 1+ GB downloads)
FROM nvidia/cuda:${CUDA_TAG}-runtime-ubuntu22.04 as runtime-base
FROM nvidia/cuda:${CUDA_TAG}-devel-ubuntu22.04

ARG PYTHON_VER=3.10
ARG ENV_NAME=cancer_bayes_iris_env
ENV DEBIAN_FRONTEND=noninteractive

# Single RUN layer for system packages with optimized cache mounts
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        # Essential tools
        bash curl ca-certificates git procps htop util-linux \
        python3 python3-venv python3-pip python3-dev \
        # Build tools (minimal set)
        build-essential cmake pkg-config \
        # Memory management 
        libjemalloc2 libjemalloc-dev \
        # Network tools for diagnostics
        iproute2 net-tools lsof \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Node.js in single layer with cache mount
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager (pinned version for reproducibility)
COPY --from=ghcr.io/astral-sh/uv:0.7.12 /uv /uvx /bin/

WORKDIR /app

# Create virtual environment early
RUN uv venv .venv --python "${PYTHON_VER}" --prompt "${ENV_NAME}"

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:${PATH}" \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Essential memory management (simplified from original complex setup)
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
    MALLOC_ARENA_MAX=1 \
    MALLOC_TCACHE_MAX=0 \
    PYTORCH_NO_CUDA_MEMORY_CACHING=1

# GPU framework environment (JAX & PyTorch only, TensorFlow removed)
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
    XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True \
    JAX_PREALLOCATION_SIZE_LIMIT_BYTES=8589934592

# Install core Python packages first (better caching)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-cache-dir jupyterlab ipykernel

# CUDA symlink setup (simplified)
RUN ln -sf $(find /usr/local -name "cuda-*" -type d | head -n1) /usr/local/cuda || \
    ln -sf /usr/local/cuda /usr/local/cuda

# Copy project files for dependency installation (from repo root)
COPY pyproject.toml /workspace/
COPY uv.lock* /workspace/

# Install project dependencies with better error handling
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /workspace && \
    (uv sync --frozen --no-dev || (echo "🔁 Lock out-of-date; refreshing…" && uv sync --no-dev && uv lock))

# Essential CUDA libraries path
ENV LD_LIBRARY_PATH="/app/.venv/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Workspace setup
WORKDIR /workspace
RUN echo 'cd /workspace' > /etc/profile.d/99-workspace-cd.sh && \
    mkdir -p /root/.ipython/profile_default/startup && \
    printf "import os, sys\nos.chdir('/workspace')\nsys.path.append('/workspace')\n" \
      > /root/.ipython/profile_default/startup/00-cd-workspace.py && \
    echo '. /app/.venv/bin/activate' > /etc/profile.d/10-uv-activate.sh

CMD ["bash"]

