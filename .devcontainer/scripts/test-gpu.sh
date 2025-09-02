# PowerShell GPU Testing Script for Windows Docker Development
# ==========================================================
# This script provides Windows-native testing capabilities for GPU frameworks
# in Docker containers, optimized for NVIDIA Blackwell RTX 5080 setups.
#
# Usage:
#   .\scripts\test-gpu.ps1
#   .\scripts\test-gpu.ps1 -Verbose
#   .\scripts\test-gpu.ps1 -ContainerName "custom_container"

param(
  [string]$ComposeProject = "cancer_bayes_iris_env",
  [string]$Service = "datascience",
  [switch]$Verbose
)

function Out-Green($m){ Write-Host "✅ $m" -ForegroundColor Green }
function Out-Red($m){ Write-Host "❌ $m" -ForegroundColor Red }
function Out-Cyan($m){ Write-Host "ℹ️  $m" -ForegroundColor Cyan }

$cid = docker compose -p $ComposeProject ps -q $Service
if (-not $cid) { Out-Red "No container found"; exit 1 }
Out-Green "Container: $cid"

Out-Cyan "nvidia-smi…"
docker exec $cid nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits

Out-Cyan "PyTorch CUDA…"
$pt = @'
import torch, gc
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(1024,1024, device="cuda")
    y = torch.randn(1024,1024, device="cuda")
    _ = (x@y).sum().item()
    del x,y; gc.collect(); torch.cuda.empty_cache()
    print("PyTorch GPU: OK")
else:
    raise SystemExit(1)
'@
docker exec $cid python -c $pt; if ($LASTEXITCODE) { Out-Red "PyTorch failed"; exit 1 } else { Out-Green "PyTorch OK" }

Out-Cyan "JAX CUDA…"
$jx = @'
import jax, importlib.util as u
print("jax:", jax.__version__)
print("devices:", jax.devices())
g = [d for d in jax.devices() if "gpu" in str(d).lower() or "cuda" in str(d).lower()]
if not g: raise SystemExit(1)
import jax.numpy as jnp
x=jnp.ones((1024,1024)); y=(x@x.T).sum(); _=y.block_until_ready()
print("JAX GPU: OK")
'@
docker exec $cid python -c $jx; if ($LASTEXITCODE) { Out-Red "JAX failed"; exit 1 } else { Out-Green "JAX OK" }
