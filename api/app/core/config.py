"""
Centralized configuration loader.

Load order (highest precedence last):
    1. Built-in safe defaults (field defaults below)
    2. config.yaml: default block
    3. config.yaml: <APP_ENV> block overlay    # APP_ENV=dev|staging|prod (aliases ok)
    4. Real environment variables               # 12-factor override for secrets

Robust to different container layouts (e.g., running from repo root,
from within `api/`, or in packaged wheels) by *searching* for config.yaml
instead of assuming a fixed number of parent hops.

Usage:
    from app.core.config import settings
"""

from __future__ import annotations

import os
import pathlib
import functools
import yaml
import logging
from typing import Any, Dict, Iterable, List

from pydantic_settings import BaseSettings, SettingsConfigDict
from .env_utils import canonical_env

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Environment tokens                                                         #
# --------------------------------------------------------------------------- #
_APP_ENV_RAW = os.getenv("APP_ENV", "dev")          # raw token (may be 'dev', 'prod', 'production', etc.)
APP_ENV_CANON = canonical_env(_APP_ENV_RAW)         # canonicalized: development/staging/production

# --------------------------------------------------------------------------- #
#  Config path discovery                                                      #
# --------------------------------------------------------------------------- #
def _candidate_paths() -> Iterable[pathlib.Path]:
    """
    Yield candidate config.yaml paths in priority order.

    Order:
      1. $APP_CONFIG_FILE if set
      2. Walk upward from this file looking for `config.yaml`
      3. Walk upward looking for `api/config.yaml`

    We cap the climb at 6 directory levels to avoid runaway loops if something is
    oddly symlinked in containers.
    """
    # 1️⃣ Explicit override
    override = os.getenv("APP_CONFIG_FILE")
    if override:
        p = pathlib.Path(override).expanduser().resolve()
        yield p

    # 2️⃣ Upward search for config.yaml
    here = pathlib.Path(__file__).resolve()
    for parent in list(here.parents)[:6]:
        yield parent / "config.yaml"

    # 3️⃣ Upward search for api/config.yaml
    for parent in list(here.parents)[:6]:
        yield parent / "api" / "config.yaml"


def _discover_config_path() -> pathlib.Path:
    """Return the first existing candidate path or raise with diagnostics."""
    tried: List[str] = []
    for cand in _candidate_paths():
        tried.append(str(cand))
        if cand.is_file():
            return cand

    msg_lines = [
        "config.yaml not found. Searched the following paths (in order):",
        *("  - " + s for s in tried),
        "Set APP_CONFIG_FILE to override."
    ]
    raise FileNotFoundError("\n".join(msg_lines))


# Resolve at import time *once*. We intentionally do **not** resolve lazily in
# _load_yaml() so that import failures are immediate & obvious.
CONFIG_PATH = _discover_config_path()

# --------------------------------------------------------------------------- #
#  Type coercion helpers                                                      #
# --------------------------------------------------------------------------- #
_BOOL_KEYS = {
    "REQUIRE_MODEL_APPROVAL",
    "AUTO_PROMOTE_TO_PRODUCTION",
    "ENABLE_MODEL_COMPARISON",
    "MLFLOW_GC_AFTER_TRAIN",
    "SKIP_BACKGROUND_TRAINING",
    "AUTO_TRAIN_MISSING",
    "UNIT_TESTING",
    "DEBUG_RATELIMIT",
    "CACHE_ENABLED",
    "ENABLE_RATE_LIMIT",
}

def _coerce_types(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize YAML-loaded values prior to BaseSettings construction.

    * Convert ints {0,1} to bool for keys in _BOOL_KEYS.
    * Strip whitespace on string values.
    """
    out = {}
    for k, v in d.items():
        if k in _BOOL_KEYS:
            if isinstance(v, bool):
                out[k] = v
            elif isinstance(v, int):
                out[k] = bool(v)
            elif isinstance(v, str) and v.strip().isdigit():
                out[k] = bool(int(v.strip()))
            else:
                out[k] = v
        elif isinstance(v, str):
            out[k] = v.strip()
        else:
            out[k] = v
    return out


@functools.lru_cache
def _load_yaml(env_token: str) -> dict:
    """
    Read config.yaml (or overridden path) and merge default + env block.

    `env_token` may be 'dev','prod','staging' *or* any alias; we canonicalize.
    """
    raw = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    if "default" not in raw:
        raise KeyError(f"{CONFIG_PATH} must contain a 'default' section")

    canon = canonical_env(env_token)
    # Map canonical name -> possible YAML block keys (accepting short aliases)
    candidate_keys = {
        "development": ("dev", "development"),
        "staging": ("staging", "stage", "preprod"),
        "production": ("prod", "production", "live"),
    }[canon]

    env_block = {}
    for key in candidate_keys:
        if key in raw:
            env_block = raw[key]
            break

    merged = {**raw["default"], **(env_block or {})}
    merged = _coerce_types(merged)

    # ensure ENVIRONMENT reflects canonical env (downstream uses this)
    merged["ENVIRONMENT"] = canon

    if os.getenv("CONFIG_DEBUG") == "1":
        log.info("CONFIG_DEBUG: loaded %s (env=%s canonical=%s)", CONFIG_PATH, env_token, canon)
        log.info("CONFIG_DEBUG: merged keys=%s", sorted(merged.keys()))

    return merged


class _Settings(BaseSettings):
    # --- core fields (defaults are last-resort safe fallbacks) ---------------
    DATABASE_URL: str = "sqlite+aiosqlite:///./app.db"
    SECRET_KEY: str | None = None
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALLOWED_ORIGINS: str = "*"
    REDIS_URL: str = "redis://localhost:6379"

    # Rate Limiting
    RATE_LIMIT_WINDOW: int = 60
    RATE_LIMIT_WINDOW_LIGHT: int = 300
    RATE_LIMIT_LOGIN_WINDOW: int = 20
    RATE_LIMIT_DEFAULT: int = 60
    RATE_LIMIT_CANCER: int = 30
    RATE_LIMIT_LOGIN: int = 3
    RATE_LIMIT_TRAINING: int = 2

    # MLflow
    MLFLOW_EXPERIMENT: str = "ml_fullstack_models"
    MLFLOW_TRACKING_URI: str = "file:./mlruns_local"
    MLFLOW_REGISTRY_URI: str = "file:./mlruns_local"
    RETAIN_RUNS_PER_MODEL: int = 5
    MLFLOW_GC_AFTER_TRAIN: bool = True

    # Model Training
    SKIP_BACKGROUND_TRAINING: bool = False
    AUTO_TRAIN_MISSING: bool = True
    UNIT_TESTING: bool = False

    # Debug
    DEBUG_RATELIMIT: bool = False

    # Rate Limiting
    ENABLE_RATE_LIMIT: bool = True

    # ML backends
    XLA_FLAGS: str = "--xla_force_host_platform_device_count=1"
    PYTENSOR_FLAGS: str = "device=cpu,floatX=float32"

    # MLOps gating
    ENVIRONMENT: str = "development"        # overlaid by YAML
    QUALITY_GATE_ACCURACY_THRESHOLD: float = 0.85
    QUALITY_GATE_F1_THRESHOLD: float = 0.85
    REQUIRE_MODEL_APPROVAL: bool = False
    AUTO_PROMOTE_TO_PRODUCTION: bool = False
    ENABLE_MODEL_COMPARISON: bool = True
    MODEL_AUDIT_ENFORCEMENT: str = "warn"
    MAX_MODEL_VERSIONS_PER_MODEL: int = 5

    # Prediction caching
    CACHE_ENABLED: bool = False
    CACHE_TTL_MINUTES: int = 60

    # Computed field (not from env)
    ENVIRONMENT_CANONICAL: str = APP_ENV_CANON  # injected in build()

    # Optional prod run fallback toggle
    ALLOW_PROD_RUN_FALLBACK: bool = False

    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False, extra="ignore")

    @classmethod
    def build(cls) -> "_Settings":  # type: ignore[override]
        # 1. merge YAML default + env block (based on APP_ENV or default 'dev')
        data = _load_yaml(_APP_ENV_RAW)

        # 2. Honour explicit ENVIRONMENT env‑var if provided (e.g. Railway)
        if "ENVIRONMENT" in os.environ:
            data["ENVIRONMENT"] = os.environ["ENVIRONMENT"].strip()

        # 3. Build settings (env vars overlay)
        inst: "_Settings" = cls(**data)

        # 4. FINAL canonical value after all overlays
        inst.ENVIRONMENT_CANONICAL = canonical_env(inst.ENVIRONMENT)

        # 5. Set the prod run fallback toggle from env var
        inst.ALLOW_PROD_RUN_FALLBACK = bool(int(os.getenv("ALLOW_PROD_RUN_FALLBACK", "0")))

        log.info(
            "📄 Loaded config (ENV=%s ⇒ %s, allow_prod_run_fallback=%s)",
            inst.ENVIRONMENT,
            inst.ENVIRONMENT_CANONICAL,
            inst.ALLOW_PROD_RUN_FALLBACK,
        )
        return inst


# public singleton
settings: _Settings = _Settings.build()



