#!/usr/bin/env python
"""
Idempotent DB seed script.

Creates or updates the default user defined by USERNAME_KEY / USER_PASSWORD
environment variables. Safe to re-run.

* Logs resolved DATABASE_URL & working directory for debugging container builds.
* Tolerates missing .env (warn only).
* Safe imports when run out of tree (adds sibling parent to sys.path if needed).
"""

from __future__ import annotations
import os
import sys
import asyncio
from pathlib import Path

from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import select

# ----- diagnostics -----------------------------------------------------------
HERE = Path(__file__).resolve()
API_DIR = HERE.parents[1]
ROOT = HERE.parents[2]
print(f"[seed_user] __file__={HERE}")
print(f"[seed_user] api_dir={API_DIR}")
print(f"[seed_user] root={ROOT}")
print(f"[seed_user] cwd={Path.cwd()}")

# ----- optional .env load ----------------------------------------------------
ENV_PATH = ROOT / ".env"
if ENV_PATH.is_file():
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_PATH, encoding="utf-8")
        print(f"[seed_user] loaded .env from {ENV_PATH}")
    except UnicodeDecodeError:
        print(f"[seed_user] ⚠️ .env not UTF-8 – skipped")
else:
    print(f"[seed_user] (no .env at {ENV_PATH})")

# ----- import models ---------------------------------------------------------
# Ensure <api_dir> is importable when script run from repo root
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

try:
    from app.models import Base, User  # type: ignore
except Exception as exc:  # pragma: no cover - debug path
    print(f"[seed_user] ❌ cannot import app.models: {exc}")
    raise

# ----- config env ------------------------------------------------------------
USERNAME = os.getenv("USERNAME_KEY", "alice")
PASSWORD = os.getenv("USER_PASSWORD", "supersecretvalue")
DB_URL   = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./app.db")

print(f"[seed_user] DATABASE_URL={DB_URL}")
print(f"[seed_user] USERNAME={USERNAME}")

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
engine = create_async_engine(DB_URL)
Session = async_sessionmaker(engine, expire_on_commit=False)

async def main() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with Session() as db:
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
        print(f"[seed_user] {action} user {USERNAME}")

if __name__ == "__main__":
    asyncio.run(main())
