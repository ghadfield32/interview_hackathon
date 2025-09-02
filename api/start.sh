#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# start.sh – Robust FastAPI launcher for Railway / Nixpacks
#
# Adds:
#   • Runtime diagnostics: show PWD + tree slices so we can see what made it
#     into the image.
#   • Defensive path discovery for helper scripts (seed_user.py, ensure_models.py)
#   • Env toggles: SEED_ON_BOOT=0   WARM_MODELS_ON_BOOT=0
#   • Fail‑soft seeding (never crash container if seed script missing)
#   • Inline Python fallback seeder (no external file needed)
# ============================================================================

# ----- helpers ---------------------------------------------------------------
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # directory containing start.sh
_root="$(cd "${_here}/.." && pwd)"                      # repo root (one up from api/)
_api="${_here}"                                         # alias; clarity

log()  { echo -e "$*" >&2; }
die()  { log "❌  $*"; exit 1; }

# ----- sanity env ------------------------------------------------------------
if [[ -z "${PORT:-}" ]]; then
  die "PORT not set – Railway always provides it."
fi
if [[ -z "${SECRET_KEY:-}" ]]; then
  die "SECRET_KEY is not set for the backend service – aborting."
fi

log "🚀  FastAPI boot; PORT=$PORT  PY=$(python -V)"
# redact secret in log dump
env | grep -E 'RAILWAY_|PORT|DATABASE_URL|APP_ENV|ENVIRONMENT|APP_CONFIG_FILE|SECRET_KEY' \
  | sed 's/SECRET_KEY=.*/SECRET_KEY=***/' >&2

# ----- runtime context debug (first deploy triage) ---------------------------
log "📂 Runtime dirs:"
log "  script dir : ${_here}"
log "  repo root  : ${_root}"
log "  api dir    : ${_api}"
log "  pwd        : $(pwd)"
log "  $(ls -alh . | wc -l) entries in pwd"

log "📂 Listing ./api (if present):"
ls -alh ./api || log "  (no ./api directory)"

log "📂 Listing ${_api}/scripts:"
ls -alh "${_api}/scripts" || log "  (no scripts directory)"

# ----- optional local .env ---------------------------------------------------
# (Load after Railway vars so local dev overrides work; harmless if missing)
if [[ -f "${_root}/.env" ]]; then
  # shellcheck disable=SC2046
  export $(grep -Ev '^#' "${_root}/.env" | xargs) || true
fi

# --- after env dump ----------------------------------------------------------
# Set sensible default on Railway if APP_ENV not specified
if [[ -z "${APP_ENV:-}" ]]; then
  export APP_ENV="staging"      # sensible default on Railway
  echo "🔧  APP_ENV not set – defaulting to staging"
fi

# Ensure MLflow datastore exists
MLFLOW_ROOT="/data/mlruns"
mkdir -p "$MLFLOW_ROOT"
export MLFLOW_TRACKING_URI="file:${MLFLOW_ROOT}"
export MLFLOW_REGISTRY_URI="file:${MLFLOW_ROOT}"
echo "📁  MLflow store => $MLFLOW_ROOT"

# ============================================================================
# Database seed
# ============================================================================
seed_db () {
  if [[ "${SEED_ON_BOOT:-1}" != "1" ]]; then
    log "ℹ️  SEED_ON_BOOT=0 – skipping DB seed."
    return 0
  fi

  local cand
  local found=""
  # Try (in order): absolute script dir, repo‑root/api/scripts, pwd-relative
  for cand in \
      "${_api}/scripts/seed_user.py" \
      "${_root}/api/scripts/seed_user.py" \
      "./api/scripts/seed_user.py" \
      "./scripts/seed_user.py"
  do
    if [[ -f "$cand" ]]; then
      found="$cand"
      break
    fi
  done

  if [[ -n "$found" ]]; then
    log "🫘  Seeding DB via $found"
    # Never crash container if seed script blows up
    if ! python "$found"; then
      log "⚠️  Seed script failed (non-fatal)."
    fi
    return 0
  fi

  # --- inline fallback seeder ------------------------------------------------
  log "⚠️  seed_user.py not found in image – using inline DB seeder."
  python - <<'EOF' || log "⚠️  Inline seed failed (continuing)."
import os, asyncio
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeMeta

# Dynamically import app.models (installed site‑package OK)
try:
    from app.models import Base, User  # type: ignore
except Exception as e:
    print(f"INLINE SEEDER: cannot import app.models ({e}) – aborting seed.")
    raise SystemExit(0)

USERNAME = os.getenv("USERNAME_KEY", "alice")
PASSWORD = os.getenv("USER_PASSWORD", "supersecretvalue")
DB_URL   = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./app.db")

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
engine = create_async_engine(DB_URL)
session_factory = async_sessionmaker(engine, expire_on_commit=False)

async def _seed():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as db:
        result = await db.execute(select(User).where(User.username == USERNAME))
        user = result.scalar_one_or_none()
        hashed = pwd.hash(PASSWORD)
        if user:
            user.hashed_password = hashed
            action = "Updated"
        else:
            db.add(User(username=USERNAME, hashed_password=hashed))
            action = "Created"
        await db.commit()
        print(f"INLINE SEEDER: {action} user {USERNAME}")

asyncio.run(_seed())
EOF
}

seed_db   # run it

# ============================================================================
# Optional model warm‑start
# ============================================================================
warm_models () {
  if [[ "${WARM_MODELS_ON_BOOT:-1}" != "1" ]]; then
    log "ℹ️  WARM_MODELS_ON_BOOT=0 – skipping model warm‑start."
    return 0
  fi

  local warm="${_api}/scripts/ensure_models.py"
  if [[ -f "$warm" ]]; then
    log "🏗️  Pre-training / warming models via $warm ..."
    if ! python "$warm"; then
      log "⚠️  pre-train failed – stub will load in API process."
    fi
  else
    log "ℹ️  No ensure_models.py found – skipping pre-train."
  fi
}
warm_models

# ============================================================================
# Launch API
# ============================================================================
log "▶️  Launching uvicorn (app.main:app; app-dir=${_api}) ..."
exec uvicorn app.main:app \
  --host 0.0.0.0 --port "$PORT" \
  --proxy-headers --forwarded-allow-ips="*" --log-level info \
  --app-dir "${_api}"

