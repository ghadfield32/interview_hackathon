"""
Environment utilities for canonical environment mapping.

This module provides consistent environment name mapping to eliminate
drift between different environment tokens used across the system.
"""

from __future__ import annotations
import os

_ENV_CANON_MAP = {
    "dev": "development",
    "development": "development",
    "local": "development",
    "staging": "staging",
    "stage": "staging",
    "preprod": "staging",
    "prod": "production",
    "production": "production",
    "live": "production",
}

def canonical_env(name: str | None = None) -> str:
    """
    Map any common environment token to a canonical value:
    development | staging | production.

    Uses APP_ENV first, then ENVIRONMENT, then defaults to development.

    Args:
        name: Environment name to canonicalize. If None, reads from APP_ENV or ENVIRONMENT env vars.

    Returns:
        Canonical environment name: 'development', 'staging', or 'production'
    """
    if name is None:
        name = os.getenv("APP_ENV") or os.getenv("ENVIRONMENT") or "development"
    return _ENV_CANON_MAP.get(name.lower().strip(), "development") 
