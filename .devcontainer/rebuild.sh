#!/bin/bash
set -euo pipefail

# DevContainer GPU Setup Rebuild Script
# This script helps rebuild the DevContainer with all the latest fixes

echo "🔧 DevContainer GPU Setup Rebuild Script"
echo "========================================"
echo ""

# Function for logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function for status indicators
status_ok() { echo "✅ $1"; }
status_warn() { echo "⚠️  $1"; }
status_error() { echo "❌ $1"; }

# Check if we're in the right directory
if [ ! -f ".devcontainer/devcontainer.json" ]; then
    status_error "This script must be run from the project root directory"
    echo "Please run: cd /path/to/your/project && bash .devcontainer/rebuild.sh"
    exit 1
fi

log "Starting DevContainer rebuild process..."

# Step 1: Stop existing containers
log "Step 1: Stopping existing containers..."
if docker compose -p cancer_bayes_iris_env ps -q >/dev/null 2>&1; then
    docker compose -p cancer_bayes_iris_env down
    status_ok "Existing containers stopped"
else
    status_warn "No existing containers found"
fi

# Step 2: Clean up Docker resources
log "Step 2: Cleaning up Docker resources..."
docker system prune -f --volumes
status_ok "Docker resources cleaned"

# Step 3: Rebuild with no cache
log "Step 3: Rebuilding DevContainer with no cache..."
docker compose -p cancer_bayes_iris_env build --no-cache datascience
status_ok "DevContainer rebuilt successfully"

# Step 4: Start the container
log "Step 4: Starting DevContainer..."
docker compose -p cancer_bayes_iris_env up -d
status_ok "DevContainer started"

# Step 5: Wait for container to be ready
log "Step 5: Waiting for container to be ready..."
sleep 15

# Step 6: Check container status
log "Step 6: Checking container status..."
if docker compose -p cancer_bayes_iris_env ps | grep -q "Up"; then
    status_ok "Container is running"
else
    status_error "Container failed to start"
    echo "Check logs with: docker compose -p cancer_bayes_iris_env logs -f"
    exit 1
fi

# Step 7: Run setup script inside container
log "Step 7: Running setup script inside container..."
CONTAINER_NAME=$(docker compose -p cancer_bayes_iris_env ps -q datascience)
if [ -n "$CONTAINER_NAME" ]; then
    docker exec -it "$CONTAINER_NAME" bash -c "cd /workspace && bash .devcontainer/setup.sh"
    status_ok "Setup script completed"
else
    status_error "Could not find container"
    exit 1
fi

# Step 8: Verify GPU access
log "Step 8: Verifying GPU access..."
docker exec -it "$CONTAINER_NAME" bash -c "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits" || {
    status_warn "GPU not accessible (this may be expected if no GPU is available)"
}

# Step 9: Test Python imports
log "Step 9: Testing Python imports..."
docker exec -it "$CONTAINER_NAME" bash -c "python - <<'PY'
import importlib, os
mods = ['pymc','torch','jax']
ok = True
for m in mods:
    try: importlib.import_module(m); print('✅', m, 'import OK')
    except Exception as e: print('❌', m, 'failed:', e); ok = False
if not ok: raise SystemExit(1)
print('✅ All key packages imported successfully')
PY" || { status_error "Some packages failed to import"; }

# Step 10: Run basic GPU verification
log "Step 10: Running basic GPU verification..."
docker exec -it "$CONTAINER_NAME" bash -c "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits" || {
    status_warn "GPU verification failed - check the output above"
}

# Step 11: Test framework imports (TensorFlow optional)
log "Step 11: Testing framework imports..."
docker exec -it "$CONTAINER_NAME" bash -c "python - <<'PY'
import importlib, os, sys
install_tf = os.getenv('INSTALL_TF','0').lower() in ('1','true','yes','on')
to_check = ['torch','jax'] + (['tensorflow'] if install_tf else [])
ok = True
for m in to_check:
    try: importlib.import_module(m); print('✅', m, 'import OK')
    except Exception as e: print('❌', m, 'failed:', e); ok = False
if not ok: raise SystemExit(1)
print('✅ Requested frameworks imported successfully')
PY" || { status_warn "A requested framework failed to import - check logs"; }

# Step 12: Check memory management
log "Step 12: Checking memory management setup..."
docker exec -it "$CONTAINER_NAME" bash -c "
echo '🔍 Checking jemalloc installation...'
if [ -f '/usr/lib/x86_64-linux-gnu/libjemalloc.so.2' ]; then
    echo '✅ jemalloc library found'
else
    echo '❌ jemalloc library not found'
fi

echo '🔍 Checking memory environment variables...'
env | grep -E '(LD_PRELOAD|MALLOC|PYTORCH|XLA|TF_FORCE)' || echo 'No memory management variables found'

echo '🔍 Checking memory allocator in use...'
python -c \"
import ctypes
try:
    ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libjemalloc.so.2')
    print('✅ jemalloc is loaded')
except:
    print('❌ jemalloc not loaded')
\"
" || {
    status_warn "Memory management check failed"
}

echo ""
echo "🎉 DevContainer rebuild completed!"
echo ""
echo "Next steps:"
echo "1. Open VS Code and press F1 → 'Dev Containers: Reopen in Container'"
echo "2. Or connect to Jupyter Lab at: http://localhost:8890 (token: jupyter)"
echo "3. Run diagnostics: bash .devcontainer/troubleshoot.sh"
echo "4. Test GPU from host: ./scripts/test-gpu.ps1 (Windows) or ./scripts/test-gpu.sh (Linux/macOS)"
echo ""
echo "Useful commands:"
echo "  View logs: docker compose -p cancer_bayes_iris_env logs -f"
echo "  Enter container: docker exec -it $CONTAINER_NAME bash"
echo "  Stop container: docker compose -p cancer_bayes_iris_env down"
echo "  Test GPU: ./scripts/test-gpu.ps1 (Windows) or ./scripts/test-gpu.sh (Linux/macOS)"

