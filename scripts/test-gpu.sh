#!/usr/bin/env bash
set -euo pipefail
PROJECT=${1:-cancer_bayes_iris_env}
SERVICE=${2:-datascience}
CID=$(docker compose -p "$PROJECT" ps -q "$SERVICE")
[ -z "$CID" ] && { echo "no container"; exit 1; }

echo "== nvidia-smi ==" && docker exec "$CID" nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits

echo "== PyTorch ==" && docker exec "$CID" python - <<'PY'
import torch, gc
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
assert torch.cuda.is_available()
print("device:", torch.cuda.get_device_name(0))
x = torch.randn(1024,1024, device="cuda")
y = torch.randn(1024,1024, device="cuda")
_ = (x@y).sum().item()
del x,y; gc.collect(); torch.cuda.empty_cache()
print("PyTorch GPU: OK")
PY

echo "== JAX ==" && docker exec "$CID" python - <<'PY'
import jax, jax.numpy as jnp
print("jax:", jax.__version__)
print("devices:", jax.devices())
assert any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in jax.devices())
x=jnp.ones((1024,1024)); y=(x@x.T).sum(); _=y.block_until_ready()
print("JAX GPU: OK")
PY

