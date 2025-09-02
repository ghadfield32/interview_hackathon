"""
Model service – self-healing startup with background training.
"""

from __future__ import annotations
import asyncio, logging, os, time, socket, shutil, subprocess, hashlib, json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

import mlflow, pandas as pd, numpy as np
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# ------------------------------------------------------------------
# IMPORTANT IMPORT NOTE
# ------------------------------------------------------------------
# This module lives in app/services/ml/.
# To reach the sibling top-level package app/core/ we must step
# *two* levels up (services/ml -> services -> app) before importing.
# Rather than counting dots ('from ...core.config import settings'),
# we choose *absolute imports* for clarity & stability across refactors.
# See Python import system docs + Real Python discussion on why absolute
# imports are preferred for larger projects.  :contentReference[oaicite:25]{index=25}
# ------------------------------------------------------------------

from app.core.config import settings
from app.ml.builtin_trainers import (
    train_iris_random_forest,
    train_iris_logreg,  # NEW
    train_breast_cancer_bayes,
    train_breast_cancer_stub,
)

# NEW imports for registry integration
from importlib import import_module
try:
    from src.registry import registry as dynamic_registry  # noqa
except Exception:
    dynamic_registry = None  # tolerant if path not yet packaged


# from ..core.config import settings
# from ..ml.builtin_trainers import (
#     train_iris_random_forest,
#     train_iris_logreg,  # NEW
#     train_breast_cancer_bayes,
#     train_breast_cancer_stub,
# )


logger = logging.getLogger(__name__)

# --- safe sklearn predict_proba helper ---------------------------------------
def _safe_sklearn_proba(estimator, X, *, log_prefix=""):
    """
    Call estimator.predict_proba(X) but recover from environments where
    joblib/loky -> psutil introspection explodes (e.g., AttributeError: psutil.Process).

    Strategy:
    1. Try fast path.
    2. On AttributeError mentioning psutil (or any RuntimeError from joblib),
       set estimator.n_jobs = 1 if present and retry serially.
    3. As a last resort, call estimator.predict(X) and synthesize 1-hot probs.

    Returns a NumPy array of shape (n_samples, n_classes).
    """
    import numpy as _np
    from joblib import parallel_backend

    # Make sure we have an array / DataFrame scikit can handle
    X_ = X

    # 1st attempt --------------------------------------------------------------
    try:
        return estimator.predict_proba(X_)
    except Exception as e:  # broad → we inspect below
        msg = str(e)
        bad_psutil = "psutil" in msg and "Process" in msg
        if not bad_psutil:
            logger.warning("%s predict_proba failed (%s) – retry single-threaded",
                           log_prefix, e)

        # 2nd attempt: force serial backend -----------------------------------
        try:
            if hasattr(estimator, "n_jobs"):
                try:
                    estimator.n_jobs = 1
                except Exception:  # read-only attr
                    pass
            with parallel_backend("threading", n_jobs=1):
                return estimator.predict_proba(X_)
        except Exception as e2:
            logger.error("%s serial predict_proba failed (%s) – fallback to classes",
                         log_prefix, e2)

    # 3rd attempt: derive 1-hot from predict ----------------------------------
    try:
        preds = estimator.predict(X_)
        preds = _np.asarray(preds, dtype=int)
        n_classes = getattr(estimator, "n_classes_", preds.max() + 1)
        probs = _np.zeros((preds.size, n_classes), dtype=float)
        probs[_np.arange(preds.size), preds] = 1.0
        return probs
    except Exception as e3:
        logger.exception("%s fallback predict also failed (%s)", log_prefix, e3)
        raise  # Let caller handle

# ---------------------------------------------------------------------------
# Cancer column mapping: Pydantic field names ➜ training column names
# ---------------------------------------------------------------------------
_CANCER_COLMAP: dict[str, str] = {
    # Means
    "mean_radius": "mean radius",
    "mean_texture": "mean texture",
    "mean_perimeter": "mean perimeter",
    "mean_area": "mean area",
    "mean_smoothness": "mean smoothness",
    "mean_compactness": "mean compactness",
    "mean_concavity": "mean concavity",
    "mean_concave_points": "mean concave points",
    "mean_symmetry": "mean symmetry",
    "mean_fractal_dimension": "mean fractal dimension",
    # SE
    "se_radius": "radius error",
    "se_texture": "texture error",
    "se_perimeter": "perimeter error",
    "se_area": "area error",
    "se_smoothness": "smoothness error",
    "se_compactness": "compactness error",
    "se_concavity": "concavity error",
    "se_concave_points": "concave points error",
    "se_symmetry": "symmetry error",
    "se_fractal_dimension": "fractal dimension error",
    # Worst
    "worst_radius": "worst radius",
    "worst_texture": "worst texture",
    "worst_perimeter": "worst perimeter",
    "worst_area": "worst area",
    "worst_smoothness": "worst smoothness",
    "worst_compactness": "worst compactness",
    "worst_concavity": "worst concavity",
    "worst_concave_points": "worst concave points",
    "worst_symmetry": "worst symmetry",
    "worst_fractal_dimension": "worst fractal dimension",
}

def _rename_cancer_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame columns match the training schema used by MLflow artefacts.
    Unknown columns are left untouched so legacy models still work.
    """
    return df.rename(columns=_CANCER_COLMAP)

# Trainer mapping for self-healing
TRAINERS = {
    "iris_random_forest": train_iris_random_forest,
    "iris_logreg":        train_iris_logreg,  # NEW
    "breast_cancer_bayes": train_breast_cancer_bayes,
    "breast_cancer_stub":  train_breast_cancer_stub,
}

class ModelService:
    """
    Self-healing model service that loads existing models and schedules
    background training for missing ones.
    """

    _EXECUTOR = ThreadPoolExecutor(max_workers=2)

    def __init__(self) -> None:
        self._unit_test_mode = settings.UNIT_TESTING
        self.initialized = False

        # 🚫 Heavy clients only when NOT unit-testing
        self.client = None if self._unit_test_mode else None  # Will be set in initialize()
        self.mlflow_client = None

        self.models: Dict[str, Any] = {}
        # registry bootstrap flags
        self._registry_loaded = False
        self.status: Dict[str, str] = {
            "iris_random_forest": "missing",
            "iris_logreg":        "missing",  # NEW
            "breast_cancer_bayes": "missing",
            "breast_cancer_stub": "missing",
        }

    # --- Registry integration (increment 1) ---------------------------------
    def _init_registry_once(self):
        if self._registry_loaded:
            return
        if dynamic_registry is None:
            logger.info("Registry package not available yet; skipping dynamic trainer loading.")
            self._registry_loaded = True
            return
        try:
            # Hardcode first migrated trainer; later we will iterate YAML directory.
            from src.registry.registry import load_from_entry_point  # type: ignore
            load_from_entry_point("src.trainers.iris_rf_trainer:IrisRandomForestTrainer")
            self._registry_loaded = True
            logger.info("Dynamic registry initialized with trainers: %s",
                        list(dynamic_registry.all_names()))
        except Exception as e:
            logger.warning("Failed to initialize registry: %s", e)
            self._registry_loaded = True  # prevent retry storm

    def _get_trainer_or_none(self, name: str):
        if not self._registry_loaded:
            self._init_registry_once()
        try:
            from src.registry.registry import get as reg_get  # type: ignore
            return reg_get(name)
        except Exception:
            return None

    async def train_via_registry(self, name: str, overrides: Dict[str, Any] | None = None) -> Optional[str]:
        spec = self._get_trainer_or_none(name)
        if spec is None:
            logger.info("No registry trainer for %s", name)
            return None
        trainer = spec.cls()
        # merge overrides
        params = trainer.merge_hyperparams(overrides or {})
        loop = asyncio.get_running_loop()
        logger.info("Training %s via registry with params=%s", name, params)
        result = await loop.run_in_executor(self._EXECUTOR, lambda: trainer.train(**params))
        # After training, force reload of production candidate (latest run fallback)
        await self._try_load(name)
        return result.run_id

    def _resolve_tracking_uri(self) -> str:
        """
        Resolve MLflow tracking URI with graceful fallback:
          1. Explicit env var MLFLOW_TRACKING_URI
          2. settings.MLFLOW_TRACKING_URI
          3. local file store 'file:./mlruns_local'
        DNS / connection problems downgrade to local file store.
        """
        import socket, urllib.parse, mlflow
        candidates = []
        if os.getenv("MLFLOW_TRACKING_URI"):
            candidates.append(("env", os.getenv("MLFLOW_TRACKING_URI")))
        candidates.append(("settings", settings.MLFLOW_TRACKING_URI))
        candidates.append(("fallback", "file:./mlruns_local"))

        for origin, uri in candidates:
            parsed = urllib.parse.urlparse(uri)
            if parsed.scheme in ("http", "https"):
                host = parsed.hostname
                try:
                    socket.getaddrinfo(host, parsed.port or 80)
                    logger.info("MLflow URI ok (%s): %s", origin, uri)
                    return uri
                except socket.gaierror as e:
                    logger.warning("MLflow URI unresolved (%s=%s) -> %s", origin, uri, e)
            else:
                # file store always acceptable
                logger.info("MLflow file store selected (%s): %s", origin, uri)
                return uri

        return "file:./mlruns_local"

    async def initialize(self) -> None:
        """
        Connect to MLflow – fall back to local file store if the configured
        tracking URI is unreachable *or* the client is missing critical methods
        (e.g. when mlflow-skinny accidentally shadows the full package).
        """
        if self.initialized:
            return

        # Log critical dependency versions for diagnostics
        try:
            import pytensor
            logger.info("📦 PyTensor version: %s", pytensor.__version__)
        except ImportError:
            logger.warning("⚠️  PyTensor not available")
        except Exception as e:
            logger.warning("⚠️  Could not determine PyTensor version: %s", e)

        def _needs_fallback(client) -> bool:
            # any missing attr is a strong signal we are on mlflow-skinny
            return not callable(getattr(client, "list_experiments", None))

        try:
            resolved = self._resolve_tracking_uri()
            mlflow.set_tracking_uri(resolved)
            logger.info("Using tracking URI: %s", resolved)
            self.mlflow_client = MlflowClient(resolved)

            if _needs_fallback(self.mlflow_client):
                raise AttributeError("list_experiments not implemented – skinny build detected")

            # minimal probe (cheap & always present)
            self.mlflow_client.search_experiments(max_results=1)
            logger.info("🟢  Connected to MLflow @ %s", resolved)

        except (MlflowException, socket.gaierror, AttributeError) as exc:
            logger.warning("🔄  Falling back to local MLflow store – %s", exc)
            mlflow.set_tracking_uri("file:./mlruns_local")
            self.mlflow_client = MlflowClient("file:./mlruns_local")
            logger.info("📂  Using local file store ./mlruns_local")

        await self._load_models()
        self.initialized = True

    async def _load_models(self) -> None:
        """Load existing models from MLflow."""
        for name in ["iris_random_forest", "iris_logreg",
                     "breast_cancer_bayes", "breast_cancer_stub"]:
            try:
                await self._try_load(name)
            except Exception as exc:
                logger.error("❌  load %s failed: %s", name, exc)

    async def startup(self, auto_train: bool | None = None) -> None:
        """
        Faster: serve stub immediately; heavy Bayesian job in background.
        """
        if self._unit_test_mode:
            logger.info("🔒 UNIT_TESTING=1 – skipping model loading")
            return                      # 👉 nothing else runs

        # Initialize MLflow connection first
        await self.initialize()

        if settings.SKIP_BACKGROUND_TRAINING and not settings.AUTO_TRAIN_MISSING:
            logger.warning("⏩ Both training flags disabled – models must already exist")
            # We still *try* to load existing artefacts so prod works
            await self._try_load("iris_random_forest")
            await self._try_load("iris_logreg")
            await self._try_load("breast_cancer_bayes")
            return

        auto = auto_train if auto_train is not None else settings.AUTO_TRAIN_MISSING
        logger.info("🔄 Model-service startup (auto_train=%s)", auto)

        # Registry-aware load for migrated models
        self._init_registry_once()

        # Try dynamic load first for iris_random_forest
        loaded_rf = await self._try_load("iris_random_forest")
        if not loaded_rf and auto:
            # prefer registry path
            run_id = await self.train_via_registry("iris_random_forest")
            if run_id:
                await self._try_load("iris_random_forest")

        # Legacy deterministic model (to be migrated later)
        if not await self._try_load("iris_logreg") and auto:
            logger.info("Training iris logistic-regression (legacy path)…")
            await asyncio.get_running_loop().run_in_executor(
                self._EXECUTOR, train_iris_logreg
            )
            await self._try_load("iris_logreg")

        # Bayesian path unchanged (will migrate later)
        if not await self._try_load("breast_cancer_bayes"):
            if not await self._try_load("breast_cancer_stub") and auto:
                logger.info("Training stub cancer model …")
                await asyncio.get_running_loop().run_in_executor(
                    self._EXECUTOR, train_breast_cancer_stub
                )
                await self._try_load("breast_cancer_stub")
            if auto and not settings.SKIP_BACKGROUND_TRAINING:
                logger.info("Scheduling Bayesian retrain in background")
                asyncio.create_task(
                    self._train_and_reload("breast_cancer_bayes", train_breast_cancer_bayes)
                )

    async def _try_load(self, name: str) -> bool:
        """Try to load a model and update status."""
        try:
            model = await self._load_production_model(name)
            if model:
                self.models[name] = model
                self.status[name] = "loaded"
                logger.info("✅ %s loaded", name)
                return True
            self.status.setdefault(name, "missing")
            return False
        except Exception as exc:
            logger.error("❌  load %s failed: %s", name, exc)
            self.status[name] = "failed"
            self.status[f"{name}_last_error"] = str(exc)
            return False

    async def _train_and_reload(self, name: str, trainer) -> None:
        """Train a model in background and reload it, with verbose phase logs."""
        try:
            t0 = time.perf_counter()
            logger.info("🏗️  BEGIN training %s", name)
            self.status[name] = "training"

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._EXECUTOR, trainer)

            logger.info("📦 Training %s complete in %.1fs – re-loading", name,
                        time.perf_counter() - t0)
            model = await self._load_production_model(name)
            if not model:
                raise RuntimeError(f"{name} trained but could not be re-loaded")

            self.models[name] = model
            self.status[name] = "loaded"

            # Trigger retention clean‑up in background
            loop = asyncio.get_running_loop()
            loop.run_in_executor(self._EXECUTOR,
                                 lambda: asyncio.run(self._cleanup_runs(name)))
            logger.info("✅ %s trained & loaded", name)

        except Exception as exc:
            self.status[name] = "failed"
            logger.error("❌ %s failed: %s", name, exc, exc_info=True)  # ← keeps trace
            # NEW: persist last_error for UI / debug endpoint
            self.status[f"{name}_last_error"] = str(exc)

# --- DROP-IN REPLACEMENT ----------------------------------------------------
    async def _load_production_model(self, name: str):
        """
        Load the canonical production model with alias support *and* perform an
        **environment audit** of the recorded vs. runtime dependencies.

        We DO NOT install anything automatically.  Instead we:
            • attempt to load in the usual fallback order (@prod → @staging → Production stage → latest run)
            • after a *successful* load, call `_audit_model_env(uri, name)` to diff
              the model's logged environment spec against the current runtime
              (importlib.metadata) and record mismatches in `self.status`.

        The audit is *diagnostic* unless an optional enforcement policy is enabled
        via env/config (MODEL_ENV_ENFORCEMENT = warn|fail|retrain).

        Returns
        -------
        Loaded MLflow model instance *or* None if nothing could be loaded, or
        load was refused under "fail" policy for env mismatch.
        """
        import sys
        import mlflow
        from mlflow.tracking.artifact_utils import _download_artifact_from_uri
        from packaging.version import Version, InvalidVersion
        import importlib.metadata as im
        import json
        import os

        # Use MLOps configuration for enforcement policy
        policy = settings.MODEL_AUDIT_ENFORCEMENT.lower()

        def _warn_model_env(uri: str) -> None:
            # unchanged best‑effort header check (Python version)
            try:
                local_dir = _download_artifact_from_uri(uri)
                mlmodel_path = Path(local_dir) / "MLmodel"
                if not mlmodel_path.is_file():
                    return
                import yaml
                meta = yaml.safe_load(mlmodel_path.read_text())
                py_model_ver = (
                    meta.get("python_env", {}).get("python")
                    or meta.get("flavors", {})
                        .get("python_function", {})
                        .get("loader_module_python_version")
                )
                runtime_py = f"{sys.version_info.major}.{sys.version_info.minor}"
                if py_model_ver and not py_model_ver.startswith(runtime_py):
                    logger.warning(
                        "⚠️ %s logged under Python %s but runtime is %s; "
                        "deserialization may fail. Consider retraining.",
                        name, py_model_ver, runtime_py
                    )
            except Exception as e:  # best-effort
                logger.debug("env check failed for %s (%s)", uri, e)

        def _audit_model_env(uri: str, model_name: str) -> dict:
            """
            Return a dict: {pkg: {'required': spec, 'current': ver, 'match': bool, 'severity': str}}
            Only logs the pip‑install command if there are mismatches.
            """
            import logging
            from pathlib import Path
            from packaging.version import Version, InvalidVersion
            import importlib.metadata as im
            from mlflow.pyfunc import get_model_dependencies

            audit: dict[str, dict] = {}

            # 1️⃣ Suppress MLflow's own INFO log for pip install
            pyfunc_logger = logging.getLogger("mlflow.pyfunc")
            old_level = pyfunc_logger.level
            pyfunc_logger.setLevel(logging.WARNING)
            try:
                try:
                    deps_path = get_model_dependencies(uri)
                except Exception:
                    deps_path = None
            finally:
                pyfunc_logger.setLevel(old_level)

            # 2️⃣ Read the pip requirements.txt
            req_lines: list[str] = []
            if deps_path and Path(deps_path).is_file():
                for ln in Path(deps_path).read_text().splitlines():
                    ln = ln.strip()
                    if not ln or ln.startswith("#"):
                        continue
                    req_lines.append(ln)

            # 3️⃣ Build the audit by comparing to runtime versions
            for spec in req_lines:
                pkg = spec.split("@", 1)[0].split(";", 1)[0].strip()
                pkg_lc = pkg.lower().replace("_", "-")

                req_ver = None
                if "==" in spec:
                    req_ver = spec.split("==", 1)[1].strip()
                elif ">=" in spec:
                    req_ver = spec.split(">=", 1)[1].strip()

                cur_ver = None
                try:
                    cur_ver = im.version(pkg_lc)
                except Exception:
                    cur_ver = None

                match = True
                sev = "OK"
                if req_ver:
                    try:
                        v_req = Version(req_ver)
                        if cur_ver:
                            v_cur = Version(cur_ver)
                            if v_cur.major != v_req.major:
                                sev, match = "MAJOR_DRT", False
                            elif v_cur != v_req:
                                sev, match = "MINOR_DRT", False
                        else:
                            sev, match = "MISSING", False
                    except InvalidVersion:
                        pass
                elif cur_ver is None:
                    sev, match = "MISSING", False

                audit[pkg_lc] = {
                    "required": req_ver,
                    "current": cur_ver,
                    "match": match,
                    "severity": sev,
                }

            # 4️⃣ Record audit in service status
            self.status[f"{model_name}_dep_audit"] = audit

            # 5️⃣ Only show pip‑install hint if there *are* mismatches
            if deps_path and any(not rec["match"] for rec in audit.values()):
                pyfunc_logger.info(
                    "To install the dependencies that were used to train the model, "
                    "run the following command: 'pip install -r %s'",
                    deps_path,
                )

            # 6️⃣ MLOps policy enforcement based on environment
            if policy in ("fail", "retrain"):
                critical = ("numpy", "scipy", "scikit-learn", "psutil")
                majors = [
                    pkg
                    for pkg, rec in audit.items()
                    if pkg in critical and rec["severity"] == "MAJOR_DRT"
                ]
                if majors:
                    msg = f"Critical env drift for {model_name}: {majors}"
                    logger.error(msg)
                    if policy == "fail":
                        self.status[f"{model_name}_last_error"] = msg
                        return {"_REFUSE_LOAD": True}
                    elif policy == "retrain":
                        logger.warning(
                            "Scheduling background retrain for %s due to env drift",
                            model_name,
                        )
                        asyncio.create_task(
                            self._train_and_reload(model_name, TRAINERS[model_name])
                        )

            return audit

        client = self.mlflow_client

        # MLOps-aware loading order based on environment
        env_canon = settings.ENVIRONMENT_CANONICAL
        if env_canon == "production":
            # Production: strict order - only Production stage or @prod alias
            attempts = [
                ("@prod", f"models:/{name}@prod"),
                ("Production stage", None),  # handle below
            ]
        elif env_canon == "staging":
            # Staging: allow staging versions for testing
            attempts = [
                ("@staging", f"models:/{name}@staging"),
                ("@prod", f"models:/{name}@prod"),
                ("Production stage", None),  # handle below
            ]
        else:
            # Development: most permissive
            attempts = [
                ("@prod", f"models:/{name}@prod"),
                ("@staging", f"models:/{name}@staging"),
                ("Production stage", None),  # handle below
                ("latest run", None),        # handle below
            ]

        # 1️⃣ Try aliases first ------------------------------------------------------
        for alias_name, uri in attempts[:2]:  # Only try aliases
            if uri is None:
                continue
            try:
                _warn_model_env(uri)
                logger.info("↪︎  Loading %s from alias %s", name, alias_name)
                mdl = mlflow.pyfunc.load_model(uri)
                audit = _audit_model_env(uri, name)
                if audit.get("_REFUSE_LOAD"):
                    logger.warning("Refusing %s from %s under policy; continuing fallbacks", name, alias_name)
                else:
                    # --- config hash drift check -------------------------------------------------
                    try:
                        # Candidate may be run-based or version-based; we try to extract run_id param from MLflow model flavor metadata.
                        # Fallback: skip silently.
                        from mlflow import get_tracking_uri
                        tracking_client = self.mlflow_client
                        # try to read run params if we have run context
                        # NOTE: _download_artifact_from_uri gave us `uri`; if it's a runs:/ URI we can parse run_id
                        if uri.startswith("runs:/"):
                            run_id = uri.split("/", 2)[1]
                            run = tracking_client.get_run(run_id)
                            train_hash = run.data.params.get("train_config_hash")
                            if train_hash:
                                from app.core.config import settings as _s
                                cur_hash = hashlib.sha256(json.dumps(_s.model_dump(), sort_keys=True).encode()).hexdigest()
                                if train_hash != cur_hash:
                                    logger.warning(
                                        "⚠️ Config drift for %s: train_hash=%s current=%s",
                                        model_name, train_hash[:8], cur_hash[:8]
                                    )
                    except Exception as _e_hash:  # best-effort
                        logger.debug("Config drift check skipped: %s", _e_hash)
                    return mdl
            except Exception as e:
                logger.debug("Alias %s not available for %s: %s", alias_name, name, e)

        # 2️⃣ Try Production stage ---------------------------------------------------
        try:
            versions = client.search_model_versions(f"name='{name}' AND stage='Production'")
            if versions:
                version = versions[0].version
                uri = f"models:/{name}/{version}"
                _warn_model_env(uri)
                logger.info("↪︎  Loading %s from registry: Production v%s", name, version)
                mdl = mlflow.pyfunc.load_model(uri)
                audit = _audit_model_env(uri, name)
                if audit.get("_REFUSE_LOAD"):
                    logger.warning("Refusing %s Production v%s under policy; continuing fallbacks", name, version)
                else:
                    return mdl
        except Exception as e:
            logger.debug("Production stage not available for %s: %s", name, e)

        # 3️⃣ Try Staging stage (only in dev/staging) ------------------------------
        if settings.ENVIRONMENT != "production":
            try:
                versions = client.search_model_versions(f"name='{name}' AND stage='Staging'")
                if versions:
                    version = versions[0].version
                    uri = f"models:/{name}/{version}"
                    _warn_model_env(uri)
                    logger.info("↪︎  Loading %s from registry: Staging v%s", name, version)
                    mdl = mlflow.pyfunc.load_model(uri)
                    audit = _audit_model_env(uri, name)
                    if audit.get("_REFUSE_LOAD"):
                        logger.warning("Refusing %s Staging v%s under policy; continuing fallbacks", name, version)
                    else:
                        return mdl
            except Exception as e:
                logger.debug("Staging stage not available for %s: %s", name, e)

        # 4️⃣  (Possible) Fallback to latest run – now allowed in prod too
        allow_run_fallback = (
            settings.ENVIRONMENT_CANONICAL != "production"
            or settings.ALLOW_PROD_RUN_FALLBACK
        )
        if allow_run_fallback:
            try:
                runs = []
                for exp in client.search_experiments():
                    runs.extend(client.search_runs(
                        [exp.experiment_id],
                        f"tags.mlflow.runName = '{name}'",
                        order_by=["attributes.start_time DESC"],
                        max_results=1))
                if runs:
                    uri = f"runs:/{runs[0].info.run_id}/model"
                    logger.warning(
                        "⚠️  %s: alias/stage missing – loading *latest run* (%s) "
                        "because ALLOW_PROD_RUN_FALLBACK=%d",
                        name, runs[0].info.run_id, allow_run_fallback,
                    )
                    _warn_model_env(uri)
                    mdl = mlflow.pyfunc.load_model(uri)
                    audit = _audit_model_env(uri, name)
                    if audit.get("_REFUSE_LOAD"):
                        logger.warning("Refusing %s latest run under policy", name)
                    else:
                        return mdl
            except Exception as e:
                logger.debug("Latest‑run fallback failed for %s: %s", name, e)

        logger.error("❌ No suitable model found for %s after all fallbacks", name)
        return None
# --- END DROP-IN REPLACEMENT -------------------------------------------------



    async def evaluate_model_quality(
        self, 
        model_name: str, 
        candidate_run_id: str,
        test_data_path: Optional[str] = None
    ) -> dict:
        """
        Evaluate a candidate model against production baseline.

        This implements quality gates for MLOps:
        1. Load production model (if exists)
        2. Load candidate model from run_id
        3. Evaluate both on test set
        4. Return comparison metrics

        Args:
            model_name: Name of the model to evaluate
            candidate_run_id: MLflow run ID of candidate model
            test_data_path: Optional path to test data (uses built-in if None)

        Returns:
            Dict with evaluation results and promotion decision
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        import pandas as pd

        logger.info("🔍 Evaluating quality gate for %s (candidate: %s)", model_name, candidate_run_id)

        # Load test data
        if model_name.startswith("iris"):
            from sklearn.datasets import load_iris
            from sklearn.model_selection import train_test_split

            iris = load_iris(as_frame=True)
            X, y = iris.data, iris.target
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

        elif model_name.startswith("breast_cancer"):
            from sklearn.datasets import load_breast_cancer
            from sklearn.model_selection import train_test_split

            X, y = load_breast_cancer(return_X_y=True, as_frame=True)
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
            X_test = _rename_cancer_columns(X_test)
        else:
            raise ValueError(f"Unknown model type: {model_name}")

        # Load candidate model
        try:
            candidate_uri = f"runs:/{candidate_run_id}/model"
            candidate_model = mlflow.pyfunc.load_model(candidate_uri)
            logger.info("✅ Loaded candidate model from %s", candidate_run_id)
        except Exception as e:
            logger.error("❌ Failed to load candidate model: %s", e)
            return {
                "promoted": False,
                "error": f"Failed to load candidate model: {e}",
                "candidate_metrics": None,
                "production_metrics": None
            }

        # Evaluate candidate
        try:
            if model_name.startswith("iris"):
                y_pred = candidate_model.predict(X_test)
                if len(y_pred.shape) == 2:  # probabilities
                    y_pred = y_pred.argmax(axis=1)
            else:  # cancer model
                y_pred_proba = candidate_model.predict(X_test)
                y_pred = (y_pred_proba > 0.5).astype(int)

            candidate_metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_macro": f1_score(y_test, y_pred, average="macro"),
                "precision_macro": precision_score(y_test, y_pred, average="macro"),
                "recall_macro": recall_score(y_test, y_pred, average="macro")
            }
            logger.info("📊 Candidate metrics: %s", candidate_metrics)

        except Exception as e:
            logger.error("❌ Failed to evaluate candidate: %s", e)
            return {
                "promoted": False,
                "error": f"Failed to evaluate candidate: {e}",
                "candidate_metrics": None,
                "production_metrics": None
            }

        # Try to load production model for comparison
        production_metrics = None
        try:
            prod_model = await self._load_production_model(model_name)
            if prod_model:
                if model_name.startswith("iris"):
                    y_pred_prod = prod_model.predict(X_test)
                    if len(y_pred_prod.shape) == 2:  # probabilities
                        y_pred_prod = y_pred_prod.argmax(axis=1)
                else:  # cancer model
                    y_pred_proba_prod = prod_model.predict(X_test)
                    y_pred_prod = (y_pred_proba_prod > 0.5).astype(int)

                production_metrics = {
                    "accuracy": accuracy_score(y_test, y_pred_prod),
                    "f1_macro": f1_score(y_test, y_pred_prod, average="macro"),
                    "precision_macro": precision_score(y_test, y_pred_prod, average="macro"),
                    "recall_macro": recall_score(y_test, y_pred_prod, average="macro")
                }
                logger.info("📊 Production metrics: %s", production_metrics)
            else:
                logger.info("📊 No production model found for comparison")

        except Exception as e:
            logger.warning("⚠️  Could not evaluate production model: %s", e)

                # Quality gate decision
        promoted = False
        reason = ""

        if production_metrics:
            # Compare against production baseline
            acc_improvement = candidate_metrics["accuracy"] - production_metrics["accuracy"]
            f1_improvement = candidate_metrics["f1_macro"] - production_metrics["f1_macro"]

            # Quality gate: must maintain or improve performance
            if acc_improvement >= -0.01 and f1_improvement >= -0.01:  # Allow 1% degradation
                promoted = True
                reason = f"Performance maintained (acc: {acc_improvement:+.3f}, f1: {f1_improvement:+.3f})"
            else:
                reason = f"Performance degraded (acc: {acc_improvement:+.3f}, f1: {f1_improvement:+.3f})"
        else:
            # No production baseline - use absolute thresholds from settings
            if candidate_metrics["accuracy"] >= settings.QUALITY_GATE_ACCURACY_THRESHOLD \
               and candidate_metrics["f1_macro"] >= settings.QUALITY_GATE_F1_THRESHOLD:
                promoted = True
                reason = f"Meets minimum thresholds (acc: {candidate_metrics['accuracy']:.3f} >= {settings.QUALITY_GATE_ACCURACY_THRESHOLD}, f1: {candidate_metrics['f1_macro']:.3f} >= {settings.QUALITY_GATE_F1_THRESHOLD})"
            else:
                reason = f"Below minimum thresholds (acc: {candidate_metrics['accuracy']:.3f} < {settings.QUALITY_GATE_ACCURACY_THRESHOLD} or f1: {candidate_metrics['f1_macro']:.3f} < {settings.QUALITY_GATE_F1_THRESHOLD})"

        result = {
            "promoted": promoted,
            "reason": reason,
            "candidate_metrics": candidate_metrics,
            "production_metrics": production_metrics,
            "candidate_run_id": candidate_run_id,
            "model_name": model_name
        }

        # log evaluation metadata back to MLflow
        try:
            client = self.mlflow_client
            client.set_tag(candidate_run_id, f"quality_gate:{settings.ENVIRONMENT_CANONICAL}",
                           "PASSED" if promoted else "FAILED")
            client.set_tag(candidate_run_id, "quality_gate_reason", reason)
        except Exception as e:
            logger.debug("Failed to set MLflow quality gate tags: %s", e)

        logger.info("🎯 Quality gate result: %s - %s", "PASSED" if promoted else "FAILED", reason)
        return result

    async def promote_model_to_staging(
        self, 
        model_name: str, 
        run_id: str
    ) -> dict:
        """
        Promote a model to staging after quality gate passes.

        This is the core MLOps promotion logic:
        1. Evaluate model quality
        2. If passed, register as staging version
        3. Set @staging alias

        Args:
            model_name: Name of the model to promote
            run_id: MLflow run ID of the candidate model

        Returns:
            Dict with promotion result
        """
        logger.info("🚀 Starting promotion process for %s (run: %s)", model_name, run_id)

        # Evaluate quality gate
        eval_result = await self.evaluate_model_quality(model_name, run_id)

        if not eval_result["promoted"]:
            logger.warning("❌ Quality gate failed for %s: %s", model_name, eval_result.get("reason", "Unknown"))
            return {
                "promoted": False,
                "error": eval_result.get("error", eval_result.get("reason", "Quality gate failed")),
                "evaluation": eval_result
            }

        # Promote to staging
        try:
            client = self.mlflow_client
            candidate_uri = f"runs:/{run_id}/model"

            # Create new model version
            mv = client.create_model_version(
                name=model_name,
                source=candidate_uri,
                run_id=run_id
            )

            # Transition to Staging
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Staging"
            )

            # Set @staging alias
            client.set_registered_model_alias(
                name=model_name,
                alias="staging",
                version=mv.version
            )

            logger.info("✅ Successfully promoted %s to staging (version %s)", model_name, mv.version)

            return {
                "promoted": True,
                "version": mv.version,
                "stage": "Staging",
                "alias": "staging",
                "evaluation": eval_result
            }

        except Exception as e:
            error_msg = str(e)
            logger.error("❌ Failed to promote %s to staging: %s", model_name, error_msg)
            return {
                "promoted": False,
                "error": f"Promotion failed: {error_msg}",
                "evaluation": eval_result
            }

    async def promote_model_to_production(
        self, 
        model_name: str,
        version: Optional[int] = None,
        approved_by: Optional[str] = None
    ) -> dict:
        """
        Promote a staging model to production.

        This can be called manually or automatically:
        1. If version specified, promote that specific version
        2. Otherwise, promote the current @staging alias
        3. Set @prod alias for atomic promotion

        Args:
            model_name: Name of the model to promote
            version: Specific version to promote (optional)

        Returns:
            Dict with promotion result
        """
        logger.info("🚀 Promoting %s to production (version: %s)", model_name, version or "staging")

        # Enforce human approval in production if required
        if settings.REQUIRE_MODEL_APPROVAL and settings.ENVIRONMENT_CANONICAL == "production":
            if not approved_by:
                return {
                    "promoted": False,
                    "error": "Approval required: pass approved_by=<user> when promoting to production.",
                }

        try:
            client = self.mlflow_client

            if version is None:
                # Get the current staging version
                staging_versions = client.search_model_versions(
                    f"name='{model_name}' AND stage='Staging'"
                )
                if not staging_versions:
                    return {
                        "promoted": False,
                        "error": f"No staging version found for {model_name}"
                    }
                version = staging_versions[0].version

            # Transition to Production
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production"
            )

            # Set @prod alias for atomic promotion
            client.set_registered_model_alias(
                name=model_name,
                alias="prod",
                version=version
            )

            # record approval metadata as tags
            try:
                client.set_model_version_tag(model_name, version, "approved_by", str(approved_by or "n/a"))
                client.set_model_version_tag(model_name, version, "approved_env", str(settings.ENVIRONMENT_CANONICAL))
            except Exception as e:
                logger.debug("Could not tag model version approval: %s", e)

            logger.info("✅ Successfully promoted %s to production (version %s)", model_name, version)

            return {
                "promoted": True,
                "version": version,
                "stage": "Production",
                "alias": "prod"
            }

        except Exception as e:
            error_msg = str(e)
            logger.error("❌ Failed to promote %s to production: %s", model_name, error_msg)
            return {
                "promoted": False,
                "error": f"Production promotion failed: {error_msg}"
            }

    async def promote_model_to_stage(
        self, 
        model_name: str,
        target_stage: str,
        version: Optional[int] = None,
        approved_by: Optional[str] = None
    ) -> dict:
        """
        Promote a model to a specific stage (staging or production).

        Args:
            model_name: Name of the model to promote
            target_stage: Target stage ('Staging' or 'Production')
            version: Specific version to promote (optional)
            approved_by: User who approved the promotion (optional)

        Returns:
            Dict with promotion result
        """
        logger.info("🚀 Promoting %s to %s (version: %s)", model_name, target_stage, version or "latest")

        # Validate target stage
        if target_stage not in ["Staging", "Production"]:
            return {
                "promoted": False,
                "error": f"Invalid target stage: {target_stage}. Must be 'Staging' or 'Production'"
            }

        # Enforce human approval in production if required
        if target_stage == "Production" and settings.REQUIRE_MODEL_APPROVAL and settings.ENVIRONMENT_CANONICAL == "production":
            if not approved_by:
                return {
                    "promoted": False,
                    "error": "Approval required: pass approved_by=<user> when promoting to production.",
                }

        try:
            client = self.mlflow_client

            if version is None:
                # Get the latest version
                versions = client.search_model_versions(f"name='{model_name}'")
                if not versions:
                    return {
                        "promoted": False,
                        "error": f"No versions found for {model_name}"
                    }
                version = versions[0].version

            # Set appropriate alias (modern approach - skip deprecated stage transitions)
            alias = "prod" if target_stage == "Production" else "staging"
            client.set_registered_model_alias(
                name=model_name,
                alias=alias,
                version=version
            )

            # record approval metadata as tags
            try:
                client.set_model_version_tag(model_name, version, "approved_by", approved_by or "n/a")
                client.set_model_version_tag(model_name, version, "approved_env", settings.ENVIRONMENT_CANONICAL)
            except Exception as e:
                logger.debug("Could not tag model version approval: %s", e)

            logger.info("✅ Successfully promoted %s to %s (version %s)", model_name, target_stage, version)

            return {
                "promoted": True,
                "version": version,
                "stage": target_stage,
                "alias": alias
            }

        except Exception as e:
            error_msg = str(e)
            logger.error("❌ Failed to promote %s to %s: %s", model_name, target_stage, error_msg)
            return {
                "promoted": False,
                "error": f"{target_stage} promotion failed: {error_msg}"
            }

    async def get_model_metrics(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Retrieve metrics for all versions of a registered model.

        This enables MLOps comparison between different model versions
        for quality gate decisions and promotion workflows.

        Args:
            model_name: Name of the registered model

        Returns:
            List of dicts with version info and metrics for each model version
        """
        client = self.mlflow_client
        versions = client.search_model_versions(f"name='{model_name}'")
        results = []

        for v in versions:
            run_id = v.run_id
            try:
                run = client.get_run(run_id)
                # Convert MLflow Metric objects to plain Python values
                metrics = {}
                for key, metric in run.data.metrics.items():
                    metrics[key] = float(metric.value) if hasattr(metric, 'value') else float(metric)
                # Add metadata for better MLOps context
                tags = run.data.tags
                creation_timestamp = v.creation_timestamp
                last_updated_timestamp = v.last_updated_timestamp
            except Exception as e:
                logger.warning(f"Could not fetch metrics for {model_name} v{v.version}: {e}")
                metrics = {}
                tags = {}
                creation_timestamp = None
                last_updated_timestamp = None

            results.append({
                "version": int(v.version),
                "stage": v.current_stage,
                "run_id": run_id,
                "metrics": metrics,
                "tags": tags,
                "creation_timestamp": creation_timestamp,
                "last_updated_timestamp": last_updated_timestamp,
                "description": v.description or ""
            })

        # Sort by version number for consistent ordering
        results.sort(key=lambda x: x["version"])
        return results

    async def compare_model_versions(
        self, 
        model_name: str, 
        version_a: int, 
        version_b: int
    ) -> Dict[str, Any]:
        """
        Compare two specific model versions for MLOps decision making.

        Args:
            model_name: Name of the registered model
            version_a: First version to compare
            version_b: Second version to compare

        Returns:
            Dict with comparison results and recommendation
        """
        client = self.mlflow_client

        # Get both versions
        try:
            version_a_info = client.get_model_version(model_name, version_a)
            version_b_info = client.get_model_version(model_name, version_b)
        except Exception as e:
            return {
                "error": f"Could not fetch model versions: {e}",
                "comparison": None
            }

        # Get metrics for both versions
        metrics_a = await self._get_version_metrics(version_a_info.run_id)
        metrics_b = await self._get_version_metrics(version_b_info.run_id)

        # Compare key metrics
        comparison = {}
        for metric in ["accuracy", "f1_macro", "precision_macro", "recall_macro"]:
            if metric in metrics_a and metric in metrics_b:
                val_a = metrics_a[metric]
                val_b = metrics_b[metric]
                diff = val_b - val_a
                comparison[metric] = {
                    "version_a": val_a,
                    "version_b": val_b,
                    "difference": diff,
                    "improvement": diff > 0
                }

        # Determine recommendation
        improvements = sum(1 for comp in comparison.values() if comp["improvement"])
        total_metrics = len(comparison)

        if total_metrics == 0:
            recommendation = "insufficient_data"
        elif improvements == total_metrics:
            recommendation = "promote_version_b"
        elif improvements == 0:
            recommendation = "keep_version_a"
        else:
            recommendation = "mixed_results"

        return {
            "model_name": model_name,
            "version_a": {
                "version": version_a,
                "stage": version_a_info.current_stage,
                "run_id": version_a_info.run_id,
                "metrics": metrics_a
            },
            "version_b": {
                "version": version_b,
                "stage": version_b_info.current_stage,
                "run_id": version_b_info.run_id,
                "metrics": metrics_b
            },
            "comparison": comparison,
            "recommendation": recommendation,
            "summary": f"Version B improves {improvements}/{total_metrics} metrics"
        }

    async def _get_version_metrics(self, run_id: str) -> Dict[str, float]:
        """Helper to get metrics for a specific run."""
        try:
            run = self.mlflow_client.get_run(run_id)
            # Convert MLflow Metric objects to plain Python values
            metrics = {}
            for key, metric in run.data.metrics.items():
                metrics[key] = float(metric.value) if hasattr(metric, 'value') else float(metric)
            return metrics
        except Exception as e:
            logger.warning(f"Could not fetch metrics for run {run_id}: {e}")
            return {}

    # Manual training endpoints (for UI)
    async def train_iris(self, model_type: str = "rf") -> None:
        """
        Train either the Random Forest or the Logistic Regression
        on the Iris dataset, per the caller's choice.
        """
        if model_type == "rf":
            name, trainer = "iris_random_forest", TRAINERS["iris_random_forest"]
        else:  # "logreg"
            name, trainer = "iris_logreg", TRAINERS["iris_logreg"]

        # reuse your existing helper
        await self._train_and_reload(name, trainer)

    async def train_cancer(self, model_type: str = "bayes") -> None:
        """
        Train either the Bayesian (PyMC) or stub (LogReg)
        on the Breast Cancer dataset, per the caller's choice.
        """
        if model_type == "bayes":
            name, trainer = "breast_cancer_bayes", TRAINERS["breast_cancer_bayes"]
        else:  # "stub"
            name, trainer = "breast_cancer_stub", TRAINERS["breast_cancer_stub"]

        await self._train_and_reload(name, trainer)

    async def train_bayes_cancer_with_params(self, params=None) -> str:
        """
        Train Bayesian cancer model with validated parameters.
        Returns the MLflow run ID.
        """
        from app.ml.builtin_trainers import train_breast_cancer_bayes

        # Run training with parameters
        run_id = train_breast_cancer_bayes(params_obj=params)

        # Reload the model after training
        await self._try_load("breast_cancer_bayes")

        return run_id

    # Predict methods (unchanged from your previous version)
    async def predict_iris(
        self,
        features: List[Dict[str, float]],
        model_type: str = "rf",
    ) -> Tuple[List[str], List[List[float]]]:
        """
        Predict Iris species from measurements.

        Hardens input normalization & always uses a serial, psutil‑safe path
        to compute class probabilities to avoid joblib/loky crashes when
        psutil is broken. Also ensures feature names are preserved to silence
        scikit‑learn's 'X does not have valid feature names' warning. :contentReference[oaicite:23]{index=23}
        """
        if model_type not in ("rf", "logreg"):
            raise ValueError("model_type must be 'rf' or 'logreg'")

        model_name = "iris_random_forest" if model_type == "rf" else "iris_logreg"
        model = self.models.get(model_name)
        if not model:
            raise RuntimeError(f"{model_name} not loaded")

        # construct DF w/ training column names in correct order
        X_df = pd.DataFrame(
            [{
                "sepal length (cm)":  f["sepal_length"],
                "sepal width (cm)":   f["sepal_width"],
                "petal length (cm)":  f["petal_length"],
                "petal width (cm)":   f["petal_width"],
            } for f in features]
        )
        logger.debug("predict_iris(%s) columns=%s", model_name, X_df.columns.tolist())

        # ALWAYS unwrap and call safe helper (skip top-level pyfunc)
        base = model
        try:
            py_model = model.unwrap_python_model()  # mlflow ≥2
            if hasattr(py_model, "model"):
                base = py_model.model
            else:
                base = py_model
        except Exception:
            pass

        probs = _safe_sklearn_proba(base, X_df, log_prefix=model_name)

        # convert to names
        import numpy as _np
        probs = _np.asarray(probs, dtype=float)
        if probs.ndim == 1:  # defensive
            # promote to 3-class; treat as class-0 vs rest
            z = _np.zeros((probs.size, 3), dtype=float)
            z[:, 0] = probs
            z[:, 1:] = (1 - probs) / 2
            probs = z
        preds = probs.argmax(axis=1)
        class_names = ["setosa", "versicolor", "virginica"]
        pred_names = [class_names[int(i)] for i in preds]
        return pred_names, probs.tolist()


    async def predict_cancer(
        self,
        features: List[Dict[str, float]],
        model_type: str = "bayes",
        posterior_samples: Optional[int] = None,
    ) -> Tuple[List[str], List[float], Optional[List[Tuple[float, float]]]]:
        """
        Predict breast cancer diagnosis.

        For stub (sklearn) path we unwrap & call psutil‑safe helper to avoid
        loky/psutil crashes; for bayes path we call model.predict() directly.
        MLflow PythonModel wrappers now expose modern signature. :contentReference[oaicite:24]{index=24}
        """
        if model_type == "bayes":
            model = self.models.get("breast_cancer_bayes") or self.models.get("breast_cancer_stub")
            using_bayes = "breast_cancer_bayes" in self.models and model is self.models["breast_cancer_bayes"]
        elif model_type == "stub":
            model = self.models.get("breast_cancer_stub")
            using_bayes = False
        else:
            raise ValueError("model_type must be 'bayes' or 'stub'")
        if not model:
            raise RuntimeError("No cancer model available")

        X_df_raw = pd.DataFrame(features)
        X_df = _rename_cancer_columns(X_df_raw)

        if using_bayes and hasattr(model, "predict"):
            probs = model.predict(X_df)
        else:
            # unwrap & safe path
            base = model
            try:
                py_model = model.unwrap_python_model()
                base = getattr(py_model, "model", py_model)
            except Exception:
                pass
            probs_full = _safe_sklearn_proba(base, X_df, log_prefix="breast_cancer_stub")
            probs = probs_full[:, 1] if probs_full.ndim == 2 else probs_full

        labels = ["malignant" if p > 0.5 else "benign" for p in probs]

        ci = None
        if posterior_samples and using_bayes:
            try:
                # Access the underlying python model to get the trace
                python_model = model.unwrap_python_model()

                # Access posterior samples for uncertainty quantification
                draws = python_model.trace.posterior
                αg = draws["α"].stack(samples=("chain", "draw"))
                β = draws["β"].stack(samples=("chain", "draw"))

                # Get group indices and standardized features
                g = python_model._quint(X_df)
                Xs = python_model.scaler.transform(X_df)

                # Compute posterior predictive samples
                logits = αg.values[:, g] + np.dot(β.values.T, Xs.T)      # shape (S, N)
                pp = 1 / (1 + np.exp(-logits))

                # Compute 95% credible intervals
                lo, hi = np.percentile(pp, [2.5, 97.5], axis=0)
                ci = list(zip(lo.tolist(), hi.tolist()))

            except Exception as e:
                logger.warning(f"Failed to compute uncertainty intervals: {e}")
                ci = None

        return labels, probs.tolist(), ci

    async def _cleanup_runs(self, model_name: str) -> None:
        """
        Keep the **newest N runs** for `model_name` and drop the rest, then
        optionally invoke `mlflow gc` to purge artifact folders.

        Runs marked *deleted* are still present on disk until GC executes,
        so we always run GC when `settings.MLFLOW_GC_AFTER_TRAIN` is True.
        """
        keep = max(settings.RETAIN_RUNS_PER_MODEL, 0)
        try:
            # 1️⃣ fetch runs newest→oldest
            runs = self.mlflow_client.search_runs(
                experiment_ids=[exp.experiment_id for exp in self.mlflow_client.search_experiments()],
                filter_string=f"tags.mlflow.runName = '{model_name}'",
                order_by=["attributes.start_time DESC"],
            )
            if len(runs) <= keep:
                logger.debug("No pruning needed for %s (runs=%d, keep=%d)",
                             model_name, len(runs), keep)
                return

            to_delete = runs[keep:]
            for r in to_delete:
                self.mlflow_client.delete_run(r.info.run_id)
            logger.info("🗑️  Pruned %d old %s runs; kept %d",
                        len(to_delete), model_name, keep)

            # 2️⃣ garbage‑collect artifacts
            if settings.MLFLOW_GC_AFTER_TRAIN:
                uri = mlflow.get_tracking_uri().removeprefix("file:")
                before = shutil.disk_usage(uri).used
                subprocess.run(
                    ["mlflow", "gc",
                     "--backend-store-uri", uri,
                     "--artifact-store", uri],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                after = shutil.disk_usage(uri).used
                logger.info("🧹 mlflow gc completed (%.2f MB → %.2f MB)",
                            before/1e6, after/1e6)

        except Exception as exc:
            logger.warning("Cleanup for %s failed: %s", model_name, exc)

    async def vacuum_store(self) -> None:
        """Force a *store‑wide* `mlflow gc` (use from cron jobs)."""
        try:
            uri = mlflow.get_tracking_uri().removeprefix("file:")
            before = shutil.disk_usage(uri).used
            subprocess.run(
                ["mlflow", "gc",
                 "--backend-store-uri", uri,
                 "--artifact-store", uri],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            after = shutil.disk_usage(uri).used
            logger.info("🧹 Store-wide vacuum completed (%.2f MB → %.2f MB)",
                        before/1e6, after/1e6)
        except Exception as exc:
            logger.warning("Store vacuum failed: %s", exc)


# Global singleton
model_service = ModelService()
