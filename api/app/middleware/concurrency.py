"""
Semaphore‑based concurrency limiter implemented as **pure ASGI middleware**.
Avoids BaseHTTPMiddleware, so it never triggers the Starlette EndOfStream bug.
"""

import asyncio
from typing import Callable, Set
from starlette.types import ASGIApp, Scope, Receive, Send

class ConcurrencyLimiter:
    def __init__(
        self,
        app: ASGIApp,
        *,
        max_concurrent: int = 4,
        heavy_endpoints: Set[str] | None = None,
    ) -> None:
        self.app = app
        self._sem = asyncio.Semaphore(max_concurrent)
        self.heavy_endpoints = heavy_endpoints or {
            "/api/v1/cancer/predict",
            "/api/v1/iris/train",
            "/api/v1/cancer/train",
        }

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Only apply to HTTP requests that match our list
        if scope["type"] != "http" or scope["path"] not in self.heavy_endpoints:
            await self.app(scope, receive, send)
            return

        async with self._sem:
            await self.app(scope, receive, send) 
