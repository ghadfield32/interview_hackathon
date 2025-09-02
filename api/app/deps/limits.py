# api/app/deps/limits.py  ✅ FIXED
"""
Rate‑limiting helpers – now **fail‑safe** when Redis is absent and
compatible with fastapi‑limiter ≥ 0.1.5 which passes *request* and *response*.
"""
from __future__ import annotations

from fastapi_limiter.depends import RateLimiter
from fastapi_limiter import FastAPILimiter
from starlette.requests import Request
from starlette.responses import Response  # NEW
from ..core.config import settings

# ── helpers ──────────────────────────────────────────────────────────────
async def _path_aware_ip(request: Request) -> str:
    fwd = request.headers.get("X-Forwarded-For")
    ip = (fwd.split(",")[0].strip() if fwd else request.client.host)
    return f"{ip}:{request.scope['path']}"

async def _user_or_ip(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return await _path_aware_ip(request)

async def _no_limit(_: Request, __: Response) -> None:  # SIG UPDATED
    """Dummy dependency when rate‑limiting is disabled."""
    return None

# ── factory that returns a dependency compatible with new signature ─────
def _safe_limiter(times: int, seconds: int, identifier):
    base = RateLimiter(times=times, seconds=seconds, identifier=identifier)

    async def wrapper(request: Request, response: Response):  # SIG UPDATED
        # If Redis is unavailable (e.g. tests), skip gracefully
        if FastAPILimiter.redis is None:
            return None
        return await base(request, response)

    return wrapper

# ── public dependencies ---------------------------------------------------
if settings.ENABLE_RATE_LIMIT:
    default_limit  = _safe_limiter(settings.RATE_LIMIT_DEFAULT,
                                   settings.RATE_LIMIT_WINDOW, _user_or_ip)
    light_limit    = _safe_limiter(settings.RATE_LIMIT_DEFAULT * 2,
                                   settings.RATE_LIMIT_WINDOW_LIGHT, _user_or_ip)
    heavy_limit    = _safe_limiter(settings.RATE_LIMIT_CANCER,
                                   settings.RATE_LIMIT_WINDOW, _user_or_ip)
    login_limit    = _safe_limiter(settings.RATE_LIMIT_LOGIN + 1,
                                   settings.RATE_LIMIT_LOGIN_WINDOW, _path_aware_ip)
    training_limit = _safe_limiter(settings.RATE_LIMIT_TRAINING,
                                   settings.RATE_LIMIT_WINDOW * 5, _user_or_ip)
else:
    # All no‑ops when rate‑limiting disabled
    default_limit = light_limit = heavy_limit = login_limit = training_limit = _no_limit

def get_redis():
    """Helper for tests/metrics."""
    return FastAPILimiter.redis
