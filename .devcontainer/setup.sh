#!/bin/bash
set -euo pipefail

# Error handling
handle_error() {
    echo "❌ Error at line: ${1}"
    exit 1
}
trap 'handle_error ${LINENO}' ERR

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "🚀 Starting DevContainer setup..."

# Verify environment
log "📋 Environment check..."
echo "Python: $(python --version)"
echo "uv: $(uv --version)"
echo "Working directory: $(pwd)"
echo "VIRTUAL_ENV: ${VIRTUAL_ENV:-<unset>}"

# Ensure we're in workspace
cd /workspace

# Check if dependencies need to be installed
if ! python -c "import pymc" 2>/dev/null; then
    log "📦 Installing project dependencies..."

    # Prefer root pyproject.toml for universal local+container dev
    if [ -f "pyproject.toml" ]; then
        log "Using workspace pyproject.toml"
        uv sync --frozen || uv sync
    elif [ -f ".devcontainer/pyproject.toml" ]; then
        log "Using .devcontainer/pyproject.toml (fallback)"
        cd .devcontainer
        uv sync --frozen || uv sync
        cd ..
    else
        log "❌ No pyproject.toml found in /workspace or .devcontainer"
        exit 1
    fi
else
    log "✅ Dependencies already installed"
fi

# Ensure ipykernel is installed and registered
log "🔧 Setting up Jupyter kernel..."
python -m ipykernel install --name uv-app-venv --display-name "Python (uv /app/.venv)" --user || true

# Install PyJAGS (optional)
log "📊 Installing PyJAGS (optional)..."
CPPFLAGS='-include cstdint' uv pip install --no-build-isolation pyjags==1.3.8 || {
    log "⚠️  PyJAGS installation failed (optional)"
}

# Verify GPU access
log "🔍 Checking GPU..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits || true
fi

# Run verification
log "🔬 Running environment verification..."
python .devcontainer/verify_env.py || true

log "✅ Setup completed"




