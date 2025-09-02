#!/usr/bin/env python
"""
Optional warm‑start script for Railway.

(Logging additions for deploy debugging.)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

HERE = Path(__file__).resolve()
API_DIR = HERE.parents[1]
ROOT = HERE.parents[2]
log.info("ensure_models: __file__=%s", HERE)
log.info("ensure_models: api_dir=%s", API_DIR)
log.info("ensure_models: root=%s", ROOT)
log.info("ensure_models: cwd=%s", Path.cwd())

# Ensure we can import `app.*` when executed as a script from repo root
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from app.services.ml.model_service import model_service  # type: ignore


async def _main() -> None:
    # Add lightweight experiment bootstrap
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        # Ensure the experiment exists before any model loading
        client = MlflowClient()
        exp = client.get_experiment_by_name("ml_fullstack_models")
        if exp is None:
            exp_id = client.create_experiment("ml_fullstack_models")
            log.info("ensure_models: created experiment ml_fullstack_models (ID: %s)", exp_id)
        else:
            log.info("ensure_models: found existing experiment ml_fullstack_models (ID: %s)", exp.experiment_id)
    except Exception as e:
        log.warning("ensure_models: experiment bootstrap failed: %s", e)

    log.info("ensure_models: initialize() …")
    await model_service.initialize()
    log.info("ensure_models: startup(auto_train=False) …")
    await model_service.startup(auto_train=False)
    log.info("ensure_models: status=%s", model_service.status)


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except Exception as exc:
        log.exception("ensure_models failed: %s", exc)
        # Non-zero exit? Returning 0 keeps container booting;
        # change to `raise` if you want hard fail.
        sys.exit(0)
