# api/app/db.py
from contextlib import asynccontextmanager
import os, logging, asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from fastapi_limiter import FastAPILimiter
from redis import asyncio as redis
from .models import Base
from .services.ml.model_service import model_service
from .core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database engine & session factory (module-level singletons – cheap & safe)
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./app.db")
engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# Global readiness flag
_app_ready: bool = False

def get_app_ready():
    """Get the current app ready status."""
    return _app_ready

# ---------------------------------------------------------------------------
# FastAPI lifespan – runs ONCE at startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app):
    """
    Application lifespan context-manager.

    * creates DB tables
    * initialises ML models
    * (NEW) wires Redis-backed rate-limiter
    * sets global _app_ready flag
    * disposes resources on shutdown
    """
    global _app_ready

    logger.info("🗄️  Initializing database…  URL=%s", DATABASE_URL)
    try:
        async with engine.begin() as conn:
            # DDL is safe here; it blocks startup until complete
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Database tables created/verified successfully")

        # ── NEW: Initialize FastAPI-Limiter BEFORE serving traffic ──────────
        if settings.ENABLE_RATE_LIMIT:
            try:
                # 1️⃣ Check for an explicit REDIS_URL env var (Railway will supply this)
                env_url = os.getenv("REDIS_URL")

                # 2️⃣ In production, prefer the env var; else use settings.REDIS_URL
                if settings.ENVIRONMENT_CANONICAL == "production" and env_url:
                    redis_url = env_url
                else:
                    redis_url = settings.REDIS_URL

                redis_conn = redis.from_url(
                    redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                await FastAPILimiter.init(redis_conn, prefix="ratelimit")
                logger.info("🚦 Rate-limiter initialised (Redis %s)", redis_url)

                # Optional: clean slate for CI
                if os.getenv("FLUSH_TEST_LIMITS") == "1":
                    try:
                        flushed = await redis_conn.flushdb()
                        logger.info("🧹 Redis FLUSHDB executed for test run, status=%s", flushed)
                    except Exception as e:
                        logger.warning("Could not flush Redis in test mode: %s", e)
            except Exception as e:
                logger.warning("⚠️  Rate-limiter init failed: %s – continuing without limits", e)
        else:
            logger.info("⚠️  Rate limiting disabled by config")

        # Initialize application readiness
        logger.info("🚀 Startup event starting - _app_ready=%s", _app_ready)

        if settings.UNIT_TESTING:
            logger.info("🔒 UNIT_TESTING=1 – startup hooks bypassed")
            _app_ready = True
            logger.info("✅ _app_ready set to True (unit testing)")
        else:
            try:
                # Initialize ModelService first
                logger.info("🔧 Initializing ModelService")
                await model_service.initialize()
                logger.info("✅ ModelService initialized successfully")

                # Start background training tasks
                logger.info("🔄 Starting background training tasks")
                asyncio.create_task(model_service.startup())
                logger.info("✅ Background training tasks started")

                # Set ready to true after initialization (models will load in background)
                _app_ready = True
                logger.info("🚀 FastAPI ready – _app_ready=%s, health probes will pass immediately", _app_ready)

            except Exception as e:
                logger.error("❌ Startup event failed: %s", e)
                import traceback
                logger.error("❌ Startup traceback: %s", traceback.format_exc())
                # Set ready to true anyway so the API can serve requests
                _app_ready = True
                logger.warning("⚠️  Setting _app_ready=True despite startup errors")

        logger.info("🎯 Lifespan startup complete - _app_ready=%s", _app_ready)
        yield
    finally:
        logger.info("🔒 Shutting down…")
        try:
            await FastAPILimiter.close()           # NEW – graceful shutdown
        except Exception:
            pass
        await engine.dispose()

# ---------------------------------------------------------------------------
# Dependency injection helper
# ---------------------------------------------------------------------------
async def get_db() -> AsyncSession:
    """Yield a new DB session per request."""
    async with AsyncSessionLocal() as session:
        yield session


