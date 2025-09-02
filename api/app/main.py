import logging
import os
import asyncio
import json
from fastapi import FastAPI, Request, Depends, BackgroundTasks, status, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
import time
from typing import Optional

from pydantic import BaseModel

# ── NEW: Fix ML backend configuration before any JAX imports ───────────────────────────
from .utils.env_sanitizer import fix_ml_backends
fix_ml_backends()
# ──────────────────────────────────────────────────────────────────────────

# ── NEW: Rate limiting imports ─────────────────────────────────────────────────────────
from fastapi_limiter import FastAPILimiter
from redis import asyncio as redis
# ────────────────────────────────────────────────────────────────────────────────────────

# ── NEW: Concurrency limiting imports ────────────────────────────────────────────────
from .middleware.concurrency import ConcurrencyLimiter
# ────────────────────────────────────────────────────────────────────────────────────────

from .db import lifespan, get_db, get_app_ready
from .security import create_access_token, get_current_user, verify_password
from .crud import get_user_by_username
from .schemas.iris import IrisPredictRequest, IrisPredictResponse, IrisFeatures
from .schemas.cancer import CancerPredictRequest, CancerPredictResponse, CancerFeatures
from .schemas.train import IrisTrainRequest, CancerTrainRequest, BayesTrainRequest, BayesTrainResponse, BayesConfigResponse, BayesRunMetrics
from .services.ml.model_service import model_service
from .core.config import settings
from .deps.limits import default_limit, heavy_limit, login_limit, training_limit, light_limit
from .security import LoginPayload, get_credentials

# ── NEW: guarantee log directory exists ───────────────────────────
os.makedirs("logs", exist_ok=True)
# ──────────────────────────────────────────────────────────────────

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── NEW: Redis cache client for prediction caching ────────────────────────────────────
# Use the same Redis URL logic as in db.py for consistency
if settings.CACHE_ENABLED:
    env_url = os.getenv("REDIS_URL")
    if settings.ENVIRONMENT_CANONICAL == "production" and env_url:
        redis_url = env_url
    else:
        redis_url = settings.REDIS_URL

    cache = redis.from_url(
        redis_url,
        encoding="utf-8",
        decode_responses=True,
    )
    logger.info("📦 Prediction caching enabled (Redis %s)", redis_url)
else:
    cache = None
    logger.info("📦 Prediction caching disabled by config")
# ────────────────────────────────────────────────────────────────────────────────────────

# Pydantic models
class Payload(BaseModel):
    count: int

class PredictionRequest(BaseModel):
    data: Payload

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    input_received: Payload  # Echo back the input for verification

class Token(BaseModel):
    access_token: str
    token_type: str

app = FastAPI(
    title="FastAPI + React ML App",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
    swagger_ui_parameters={"persistAuthorization": True},
    lifespan=lifespan,  # register startup/shutdown events
)

# ── Rate limiting is now initialized in lifespan() ────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────

# Configure CORS with environment-based origins
origins_env = settings.ALLOWED_ORIGINS
origins: list[str] = [o.strip() for o in origins_env.split(",")] if origins_env != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── NEW: Add concurrency limiting middleware ──────────────────────────────────────────
app.add_middleware(ConcurrencyLimiter, max_concurrent=4)
# ────────────────────────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Measure request time and add X-Process-Time header."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.4f}"
    return response

# Health check endpoint
@app.get("/api/v1/health")
async def health_check():
    """Basic health check - always returns 200 if server is running."""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/api/v1/hello")
async def hello(current_user: str = Depends(get_current_user)):
    """Simple endpoint for token validation."""
    return {"message": f"Hello {current_user}!", "status": "authenticated"}

@app.get("/api/v1/ready")
async def ready():
    """Basic readiness check."""
    return {"ready": get_app_ready()}

@app.get("/api/v1/ready/frontend")
async def ready_frontend() -> dict:
    """
    Frontend-safe readiness payload.
    Returns only small, stable fields the React SPA depends on.
    This avoids the large nested dependency audit data that was causing frontend crashes.
    """
    ready_for_login = get_app_ready()
    loaded = set(model_service.models.keys())
    return {
        "ready": ready_for_login,
        "models": {
            "iris": "iris_random_forest" in loaded or "iris_logreg" in loaded,
            "cancer": "breast_cancer_bayes" in loaded or "breast_cancer_stub" in loaded,
        },
        "has_bayes": "breast_cancer_bayes" in loaded,
        "has_stub": "breast_cancer_stub" in loaded,
        "all_models_loaded": all(
            model in loaded 
            for model in ["iris_random_forest", "breast_cancer_bayes"]
        ),
    }

@app.post("/api/v1/token", response_model=Token, dependencies=[Depends(login_limit)])
async def login(
    creds: LoginPayload = Depends(get_credentials),
    db: AsyncSession = Depends(get_db),
):
    """
    Issue a JWT. Accepts **either**
    • JSON {"username": "...", "password": "..."}  *or*
    • classic x‑www‑form‑urlencoded.
    """
    # 1️⃣ readiness gate
    if not get_app_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Backend still loading models. Try again in a moment.",
            headers={"Retry‑After": "10"},
        )

    # 2️⃣ verify credentials
    user = await get_user_by_username(db, creds.username)
    if not user or not verify_password(creds.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid credentials")

    # 3️⃣ issue token
    token = create_access_token(subject=user.username)
    return Token(access_token=token, token_type="bearer")

# --- PATCH: ready_full -------------------------------------------------------
@app.get("/api/v1/ready/full")
async def ready_full(debug: Optional[bool] = False) -> dict:
    """
    Extended readiness probe with environment drift summary.

    Query param:
      debug=1  -> include filtered_status map for troubleshooting.
    """
    ready_for_login = get_app_ready()
    expected = {"iris_random_forest", "breast_cancer_bayes"}  # minimal contract
    loaded = set(model_service.models.keys())

    # ----- helpers -----------------------------------------------------------
    def _is_meta(k: str) -> bool:
        return k.endswith("_dep_audit") or k.endswith("_last_error")

    def _model_status_items():
        for k, v in model_service.status.items():
            if _is_meta(k):
                continue
            yield k, v

    # ----- env drift summary -------------------------------------------------
    drift = {}
    for m in ("iris_random_forest", "iris_logreg", "breast_cancer_bayes", "breast_cancer_stub"):
        audit = model_service.status.get(f"{m}_dep_audit", {})
        critical = any(
            (pkg in ("numpy", "scipy", "scikit-learn", "psutil")) and rec.get("severity") == "MAJOR_DRT"
            for pkg, rec in audit.items()
        )
        drift[m] = {"critical_drift": critical, "details": audit}

    # ----- core fields -------------------------------------------------------
    filtered_status = dict(_model_status_items())
    all_models_loaded = all(v == "loaded" for v in filtered_status.values())
    training = [k for k, v in filtered_status.items() if v == "training"]

    response = {
        "ready": ready_for_login,
        "model_status": model_service.status,  # raw (includes meta)
        "env_drift": drift,
        "all_models_loaded": all_models_loaded,
        "models": {m: (m in loaded) for m in expected},
        "training": training,
    }

    if debug:
        response["status_filtered"] = filtered_status
        response["status_counts"] = {
            "raw": len(model_service.status),
            "filtered": len(filtered_status),
        }

    # Log response size for debugging
    if debug:
        import json
        response_size = len(json.dumps(response))
        logger.info("READY_FULL debug: payload size=%d bytes", response_size)

    logger.debug("READY endpoint – _app_ready=%s", ready_for_login)
    return response
# --- END PATCH ---------------------------------------------------------------



# ── Alias routes (no auth, not shown in OpenAPI) ────────────────────────────
@app.get("/ready/full", include_in_schema=False)
async def ready_full_alias():
    """Alias for front-end calls that miss the /api/v1 prefix."""
    return await ready_full()

@app.get("/health", include_in_schema=False)
async def health_alias():
    """Alias for plain /health (SPA hits it before it knows the prefix)."""
    return await health_check()

@app.post("/token", include_in_schema=False)
async def login_alias(request: Request):
    """
    Alias: accept /token like /api/v1/token.
    Keeps the OAuth2PasswordRequestForm semantics without exposing clutter in docs.
    """
    from fastapi import Form

    # Parse form data manually to match OAuth2PasswordRequestForm behavior
    form_data = await request.form()
    username = form_data.get("username")
    password = form_data.get("password")

    if not username or not password:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="username and password are required"
        )

    # Create a mock OAuth2PasswordRequestForm object
    class MockForm:
        def __init__(self, username, password):
            self.username = username
            self.password = password

    mock_form = MockForm(username, password)

    # Reuse the existing login logic
    db = await get_db().__anext__()
    return await login(mock_form, db)

@app.post("/iris/predict", include_in_schema=False)
async def iris_predict_alias(request: Request):
    """Alias for /api/v1/iris/predict"""
    from .schemas.iris import IrisPredictRequest

    # Parse JSON body
    body = await request.json()
    iris_request = IrisPredictRequest(**body)

    # Reuse the existing prediction logic without authentication for testing
    background_tasks = BackgroundTasks()
    current_user = "test_user"  # Skip authentication for alias endpoints
    return await predict_iris(iris_request, background_tasks, current_user)

@app.post("/cancer/predict", include_in_schema=False)
async def cancer_predict_alias(request: Request):
    """Alias for /api/v1/cancer/predict"""
    from .schemas.cancer import CancerPredictRequest

    # Parse JSON body
    body = await request.json()
    cancer_request = CancerPredictRequest(**body)

    # Reuse the existing prediction logic without authentication for testing
    background_tasks = BackgroundTasks()
    current_user = "test_user"  # Skip authentication for alias endpoints
    return await predict_cancer(cancer_request, background_tasks, current_user)

# ----- on-demand training endpoints ----------------------------------
@app.post("/api/v1/iris/train", status_code=202, dependencies=[Depends(training_limit)])
async def train_iris(
    request: IrisTrainRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user)
):
    """
    Kick off training of the chosen Iris model.
    """
    background_tasks.add_task(
        model_service.train_iris,
        request.model_type
    )
    return {"status": f"started iris training ({request.model_type})"}

@app.post("/api/v1/cancer/train", status_code=202, dependencies=[Depends(training_limit)])
async def train_cancer(
    request: CancerTrainRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user)
):
    """
    Kick off training of the chosen Cancer model.
    """
    background_tasks.add_task(
        model_service.train_cancer,
        request.model_type
    )
    return {"status": f"started cancer training ({request.model_type})"}

@app.get("/api/v1/cancer/bayes/config", response_model=BayesConfigResponse)
async def get_bayes_config(current_user: str = Depends(get_current_user)):
    """
    Get Bayesian training configuration for frontend form generation.
    """
    from .schemas.bayes import BayesCancerParams

    defaults = BayesCancerParams()

    return BayesConfigResponse(
        defaults=defaults,
        bounds={
            "draws": {"min": 200, "max": 20000},
            "tune": {"min": 200, "max": 20000},
            "target_accept": {"min": 0.80, "max": 0.999},
            "max_rhat_warn": {"min": 1.0, "max": 1.1},
            "min_ess_warn": {"min": 50, "max": 5000},
        },
        descriptions={
            "draws": "Number of posterior draws retained. More draws = better MCSE but longer runtime.",
            "tune": "Warmup steps for NUTS adaptation. Should be ≥ 0.2 * draws for good convergence.",
            "target_accept": "Target acceptance rate. Higher values reduce divergences but increase runtime.",
            "compute_waic": "Compute Widely Applicable Information Criterion. Fast but may be less robust than LOO.",
            "compute_loo": "Compute Leave-One-Out cross-validation. More reliable but slower.",
        },
        runtime_estimate={
            "base_seconds_per_sample": 0.001,  # rough estimate
            "chains": 4,
            "overhead_seconds": 5.0,  # model setup, data loading, etc.
        }
    )

@app.post("/api/v1/cancer/bayes/train", response_model=BayesTrainResponse, dependencies=[Depends(training_limit)])
async def train_bayes_cancer(
    request: BayesTrainRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user)
):
    """
    Train Bayesian cancer model with validated hyperparameters.
    """
    if not get_app_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Backend still loading models. Try again in a moment.",
            headers={"Retry‑After": "10"},
        )

    try:
        if request.async_training:
            # Queue for background processing
            job_id = f"bayes_{int(time.time())}"
            background_tasks.add_task(
                model_service.train_bayes_cancer_with_params, 
                request.params
            )
            return BayesTrainResponse(
                run_id="",  # will be set when job completes
                job_id=job_id,
                status="queued",
                message="Training queued for background processing"
            )
        else:
            # Synchronous training
            run_id = await model_service.train_bayes_cancer_with_params(request.params)
            return BayesTrainResponse(
                run_id=run_id,
                status="completed",
                message="Training completed successfully"
            )
    except Exception as e:
        logger.error("Bayesian training failed: %s", e)
        return BayesTrainResponse(
            run_id="",
            status="failed",
            message=str(e)
        )

@app.get("/api/v1/cancer/bayes/runs/{run_id}", response_model=BayesRunMetrics)
async def get_bayes_run_metrics(
    run_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Get metrics for a specific Bayesian training run.
    """
    try:
        import mlflow
        run = mlflow.get_run(run_id)
        metrics = run.data.metrics
        params = run.data.params

        warnings = []
        if metrics.get("rhat_max", 0) > 1.01:
            warnings.append(f"R-hat exceeds threshold: {metrics['rhat_max']:.4f} > 1.01")
        if metrics.get("ess_bulk_min", 0) < 400:
            warnings.append(f"Bulk ESS below threshold: {metrics['ess_bulk_min']:.1f} < 400")

        return BayesRunMetrics(
            run_id=run_id,
            accuracy=metrics.get("accuracy", 0.0),
            rhat_max=metrics.get("rhat_max"),
            ess_bulk_min=metrics.get("ess_bulk_min"),
            ess_tail_min=metrics.get("ess_tail_min"),
            waic=metrics.get("waic"),
            loo=metrics.get("loo"),
            status="completed",
            warnings=warnings
        )
    except Exception as e:
        logger.error("Failed to get run metrics for %s: %s", run_id, e)
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found or metrics unavailable")

# ----- debug endpoints ----------------------------------
@app.get("/api/v1/debug/ready")
async def debug_ready():
    """Debug endpoint to verify configuration loading."""
    return {
        "status": "ready",
        "environment": settings.ENVIRONMENT,
        "rate_limits": {
            "default": settings.RATE_LIMIT_DEFAULT,
            "cancer": settings.RATE_LIMIT_CANCER,
            "login": settings.RATE_LIMIT_LOGIN,
            "training": settings.RATE_LIMIT_TRAINING,
            "window": settings.RATE_LIMIT_WINDOW,
            "window_light": settings.RATE_LIMIT_WINDOW_LIGHT,
        },
        "quality_gates": {
            "accuracy_threshold": settings.QUALITY_GATE_ACCURACY_THRESHOLD,
            "f1_threshold": settings.QUALITY_GATE_F1_THRESHOLD,
        },
        "mlflow": {
            "experiment": settings.MLFLOW_EXPERIMENT,
            "tracking_uri": settings.MLFLOW_TRACKING_URI,
            "registry_uri": settings.MLFLOW_REGISTRY_URI,
        },
        "training": {
            "skip_background": settings.SKIP_BACKGROUND_TRAINING,
            "auto_train_missing": settings.AUTO_TRAIN_MISSING,
        },
        "debug": {
            "debug_ratelimit": settings.DEBUG_RATELIMIT,
        }
    }

# --- effective config debug --------------------------------------------------
@app.get("/api/v1/debug/effective-config")
async def effective_config(current_user: str = Depends(get_current_user)):
    """
    Inspect the *effective* runtime configuration (after YAML + env overrides).

    Sensitive fields are redacted. Use to debug environment drift across
    dev/staging/production deployments.
    """
    from app.core.config import settings

    redacted = {"SECRET_KEY", "DATABASE_URL"}
    cfg = settings.model_dump()
    for k in list(cfg):
        if k.upper() in redacted and cfg[k] is not None:
            cfg[k] = "***redacted***"
    return {
        "environment": settings.ENVIRONMENT_CANONICAL,
        "config": cfg,
    }

# ----- MLOps endpoints (new) ----------------------------------------
@app.post("/api/v1/mlops/evaluate/{model_name}")
async def evaluate_model(
    model_name: str,
    run_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Evaluate a candidate model against production baseline.

    This endpoint is used by CI/CD pipelines to implement quality gates.
    The model is evaluated on a fixed test set and compared to production.
    """
    logger.info(f"User {current_user} evaluating model {model_name} (run: {run_id})")

    result = await model_service.evaluate_model_quality(model_name, run_id)
    return result

@app.post("/api/v1/mlops/promote/{model_name}/staging")
async def promote_to_staging(
    model_name: str,
    run_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Promote a model to staging after quality gate evaluation.

    This endpoint:
    1. Evaluates the model quality
    2. If passed, registers as staging version
    3. Sets @staging alias for atomic promotion
    """
    logger.info(f"User {current_user} promoting {model_name} to staging (run: {run_id})")

    result = await model_service.promote_model_to_staging(model_name, run_id)
    return result

@app.post("/api/v1/mlops/promote/{model_name}/production")
async def promote_to_production(
    model_name: str,
    version: Optional[int] = None,
    approved_by: Optional[str] = None,
    current_user: str = Depends(get_current_user)
):
    """
    Promote a staging model to production.

    This can be called manually or by CI/CD:
    - If version specified, promotes that specific version
    - Otherwise, promotes the current @staging alias
    - Sets @prod alias for atomic promotion
    """
    logger.info(f"User {current_user} promoting {model_name} to production (version: {version})")

    result = await model_service.promote_model_to_production(model_name, version, approved_by)
    return result

@app.post("/api/v1/mlops/reload-model")
async def reload_model(
    model_name: Optional[str] = None,
    current_user: str = Depends(get_current_user)
):
    """
    Hot-reload models from MLflow registry.

    This endpoint allows the container to pick up new models
    without restarting the entire service. Useful for:
    - CI/CD deployments that update models
    - Manual model promotions
    - Testing new model versions

    Args:
        model_name: Specific model to reload (optional, reloads all if None)
    """
    logger.info(f"User {current_user} reloading models (specific: {model_name})")

    try:
        if model_name:
            # Reload specific model
            success = await model_service._try_load(model_name)
            if success:
                return {
                    "reloaded": True,
                    "model": model_name,
                    "status": model_service.status.get(model_name, "unknown")
                }
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Failed to reload model {model_name}"
                )
        else:
            # Reload all models
            reloaded = []
            failed = []

            for name in ["iris_random_forest", "iris_logreg", 
                        "breast_cancer_bayes", "breast_cancer_stub"]:
                try:
                    success = await model_service._try_load(name)
                    if success:
                        reloaded.append(name)
                    else:
                        failed.append(name)
                except Exception as e:
                    logger.error(f"Failed to reload {name}: {e}")
                    failed.append(name)

            return {
                "reloaded": len(failed) == 0,
                "reloaded_models": reloaded,
                "failed_models": failed,
                "status": model_service.status
            }

    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Model reload failed: {e}"
        )

@app.get("/api/v1/mlops/status")
async def mlops_status(current_user: str = Depends(get_current_user)):
    """
    Get MLOps status including model versions and stages.

    Returns comprehensive information about:
    - Model loading status
    - Registry versions and stages
    - Alias assignments
    - Training status
    """
    logger.info(f"User {current_user} requesting MLOps status")

    try:
        client = model_service.mlflow_client

        # Get model registry information
        registry_info = {}
        for model_name in ["iris_random_forest", "iris_logreg", 
                          "breast_cancer_bayes", "breast_cancer_stub"]:
            try:
                versions = client.search_model_versions(f"name='{model_name}'")
                registry_info[model_name] = {
                    "versions": len(versions),
                    "stages": {},
                    "aliases": {}
                }

                # Group by stage
                for v in versions:
                    stage = v.current_stage
                    if stage not in registry_info[model_name]["stages"]:
                        registry_info[model_name]["stages"][stage] = []
                    registry_info[model_name]["stages"][stage].append({
                        "version": v.version,
                        "run_id": v.run_id,
                        "created_at": v.creation_timestamp
                    })

                # Get aliases
                try:
                    aliases = client.get_registered_model_aliases(model_name)
                    registry_info[model_name]["aliases"] = {
                        alias: version for alias, version in aliases.items()
                    }
                except Exception as e:
                    logger.debug(f"Could not get aliases for {model_name}: {e}")

            except Exception as e:
                logger.warning(f"Could not get registry info for {model_name}: {e}")
                registry_info[model_name] = {"error": str(e)}

        return {
            "model_status": model_service.status,
            "loaded_models": list(model_service.models.keys()),
            "registry_info": registry_info,
            "app_ready": get_app_ready(),
            "mlflow_uri": settings.MLFLOW_TRACKING_URI
        }

    except Exception as e:
        logger.error(f"MLOps status failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"MLOps status failed: {e}"
        )

@app.get("/api/v1/mlops/models/{model_name}/metrics")
async def get_model_metrics(
    model_name: str,
    current_user: str = Depends(get_current_user)
):
    """
    Get metrics for all versions of a registered model.

    This endpoint enables MLOps comparison between different model versions
    for quality gate decisions and promotion workflows.
    """
    try:
        metrics = await model_service.get_model_metrics(model_name)
        if not metrics:
            raise HTTPException(
                status_code=404, 
                detail=f"No registered model found with name '{model_name}'"
            )
        return {
            "model_name": model_name,
            "versions": metrics,
            "total_versions": len(metrics)
        }
    except Exception as e:
        logger.error(f"Error fetching metrics for {model_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch model metrics: {str(e)}"
        )

@app.get("/api/v1/mlops/models/{model_name}/compare")
async def compare_model_versions(
    model_name: str,
    version_a: int,
    version_b: int,
    current_user: str = Depends(get_current_user)
):
    """
    Compare two specific model versions for MLOps decision making.

    This endpoint helps determine which model version performs better
    across key metrics like accuracy, F1-score, precision, and recall.
    """
    try:
        comparison = await model_service.compare_model_versions(
            model_name, version_a, version_b
        )

        if "error" in comparison:
            raise HTTPException(
                status_code=400,
                detail=comparison["error"]
            )

        return comparison
    except Exception as e:
        logger.error(f"Error comparing versions for {model_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compare model versions: {str(e)}"
        )

@app.get("/api/v1/mlops/models/{model_name}/quality-gate")
async def check_quality_gate(
    model_name: str,
    version: Optional[int] = None,
    current_user: str = Depends(get_current_user)
):
    """
    Check if a model version passes quality gates.

    This endpoint evaluates a model against production baseline
    or absolute thresholds to determine if it's ready for promotion.
    """
    try:
        # If no version specified, use the latest staging version
        if version is None:
            client = model_service.mlflow_client
            staging_versions = client.search_model_versions(
                f"name='{model_name}' AND stage='Staging'"
            )
            if not staging_versions:
                raise HTTPException(
                    status_code=404,
                    detail=f"No staging version found for {model_name}"
                )
            version = staging_versions[0].version

        # Get the run_id for this version
        version_info = client.get_model_version(model_name, version)
        run_id = version_info.run_id

        # Evaluate quality gate
        eval_result = await model_service.evaluate_model_quality(model_name, run_id)

        return {
            "model_name": model_name,
            "version": version,
            "run_id": run_id,
            "quality_gate_result": eval_result,
            "passes_gate": eval_result["promoted"],
            "reason": eval_result["reason"]
        }
    except Exception as e:
        logger.error(f"Error checking quality gate for {model_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check quality gate: {str(e)}"
        )

@app.get("/api/v1/iris/ready")
async def iris_ready():
    """Check if Iris model is loaded and ready."""
    return {"loaded": "iris_random_forest" in model_service.models}

@app.get("/api/v1/cancer/ready")
async def cancer_ready():
    """Check if Cancer model is loaded and ready."""
    return {"loaded": "breast_cancer_bayes" in model_service.models}

@app.post(
    "/api/v1/iris/predict",
    response_model=IrisPredictResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(light_limit)]
)
async def predict_iris(
    request: IrisPredictRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user),
):
    """
    Predict iris species from measurements, with optional Redis caching.
    """
    logger.info(f"User {current_user} called /iris/predict with {len(request.samples)} samples")

    # Ensure model is loaded
    model_name = "iris_random_forest" if request.model_type == "rf" else "iris_logreg"
    if model_name not in model_service.models:
        raise HTTPException(
            status_code=503,
            detail="Iris model still loading. Try again shortly.",
            headers={"Retry-After": "30"},
        )

    # Build cache key from primitives (avoid Pydantic models)
    serialized_samples = [s.dict() for s in request.samples]
    key = f"iris:{request.model_type}:{json.dumps(serialized_samples, sort_keys=True)}"

    # Try Redis GET if caching enabled
    if settings.CACHE_ENABLED:
        cached = await cache.get(key)
        if cached:
            logger.debug("Cache hit for key %s", key)
            return IrisPredictResponse(**json.loads(cached))

    # Perform prediction
    preds, probs = await model_service.predict_iris(
        features=serialized_samples,
        model_type=request.model_type,
    )

    # Prepare a fully-serializable result dict
    result = {
        "predictions": preds,
        "probabilities": probs,
        "input_received": serialized_samples,
    }

    # Store in cache if enabled
    if settings.CACHE_ENABLED:
        ttl = settings.CACHE_TTL_MINUTES * 60
        await cache.set(key, json.dumps(result), ex=ttl)

    # Audit log in background
    background_tasks.add_task(
        logger.info,
        f"[audit] user={current_user} endpoint=iris input={serialized_samples} output={preds}"
    )

    return IrisPredictResponse(**result)

@app.post(
    "/api/v1/cancer/predict",
    response_model=CancerPredictResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(heavy_limit)]
)
async def predict_cancer(
    request: CancerPredictRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user),
):
    """
    Predict breast-cancer diagnosis, with optional Redis caching.
    """
    logger.info(f"User {current_user} called /cancer/predict with {len(request.samples)} samples")

    # Build cache key from primitives (includes posterior_samples)
    serialized_samples = [s.dict() for s in request.samples]
    key = (
        f"cancer:{request.model_type}:"
        f"{request.posterior_samples or 0}:"
        f"{json.dumps(serialized_samples, sort_keys=True)}"
    )

    # Try Redis GET if caching enabled
    if settings.CACHE_ENABLED:
        cached = await cache.get(key)
        if cached:
            logger.debug("Cache hit for key %s", key)
            return CancerPredictResponse(**json.loads(cached))

    # Perform prediction
    preds, probs, uncertainties = await model_service.predict_cancer(
        features=serialized_samples,
        model_type=request.model_type,
        posterior_samples=request.posterior_samples,
    )

    # Prepare a fully-serializable result dict
    result = {
        "predictions": preds,
        "probabilities": probs,
        "uncertainties": uncertainties,
        "input_received": serialized_samples,
    }

    # Store in cache if enabled
    if settings.CACHE_ENABLED:
        ttl = settings.CACHE_TTL_MINUTES * 60
        await cache.set(key, json.dumps(result), ex=ttl)

    # Audit log in background
    background_tasks.add_task(
        logger.info,
        f"[audit] user={current_user} endpoint=cancer input={serialized_samples} output={preds}"
    )

    return CancerPredictResponse(**result) 

@app.get("/api/v1/debug/compiler")
async def debug_compiler():
    """
    Debug endpoint to check JAX/NumPyro backend configuration.
    Returns information about the JAX backend setup.
    """
    try:
        import jax
        import numpyro
        import pymc as pm

        return {
            "backend": "jax_numpyro",
            "jax_version": jax.__version__,
            "numpyro_version": numpyro.__version__,
            "pymc_version": pm.__version__,
            "jax_devices": str(jax.devices()),
            "jax_platform": jax.default_backend(),
            "status": "jax_backend_configured"
        }
    except ImportError as e:
        return {
            "backend": "unknown",
            "error": f"Import error: {e}",
            "status": "missing_dependencies"
        }
    except Exception as e:
        return {
            "backend": "unknown", 
            "error": f"Configuration error: {e}",
            "status": "configuration_failed"
        }

@app.get("/api/v1/debug/psutil")
async def debug_psutil():
    """
    Debug endpoint to check psutil status and configuration.
    Returns information about psutil module and its Process class.
    """
    import sys, types
    try:
        import psutil
        module_info = {
            "module_path": getattr(psutil, "__file__", "?"),
            "version": getattr(psutil, "__version__", "?"),
            "has_Process": hasattr(psutil, "Process"),
            "sys_path": sys.path
        }

        # Try a safe Process call
        try:
            proc = psutil.Process()
            module_info["process_test"] = {
                "success": True,
                "pid": proc.pid,
                "cpu_count": psutil.cpu_count()
            }
        except Exception as e:
            module_info["process_test"] = {
                "success": False,
                "error": str(e)
            }

        return {
            "status": "loaded",
            "info": module_info
        }
    except ImportError as e:
        return {
            "status": "import_failed",
            "error": str(e)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        } 

@app.get("/api/v1/debug/deps")
async def debug_deps():
    """
    Report recorded vs. runtime dependency versions for each loaded model.

    Uses audit data collected during ModelService._load_production_model().
    Helpful when MLflow logs 'requirements_utils' mismatch warnings.

    NOTE: purely diagnostic – no secrets.
    """
    import importlib.metadata as im
    runtime = {}
    for pkg in ("numpy", "scipy", "scikit-learn", "psutil", "pandas"):
        try:
            runtime[pkg] = im.version(pkg)
        except Exception:
            runtime[pkg] = None

    audits = {k: v for k, v in model_service.status.items() if k.endswith("_dep_audit")}
    return {
        "runtime": runtime,
        "model_audits": audits,
        "enforcement_policy": os.getenv("MODEL_ENV_ENFORCEMENT", "warn"),
    }


@app.get("/api/v1/test/401")
async def test_401():
    """Test endpoint that returns 401 for testing session expiry."""
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Test 401 response"
    )

# ── Debug‑only ratelimit helpers ─────────────────────────────────────────────
from .deps.limits import get_redis, _user_or_ip as user_or_ip

@app.post("/api/v1/debug/ratelimit/reset", include_in_schema=False)
async def rl_reset(request: Request):
    """
    Flush **all** rate‑limit counters bound to the caller (JWT _or_ IP).

    We match every fragment that contains the identifier to survive
    future changes in FastAPI‑Limiter's key schema.
    """
    r = get_redis()
    if not r:
        raise HTTPException(status_code=503, detail="Rate‑limiter not initialised")

    ident = await user_or_ip(request)
    keys = await r.keys(f"ratelimit:*{ident}*")        # <— broader pattern
    if keys:
        await r.delete(*keys)
    return {"reset": len(keys)}

if settings.DEBUG_RATELIMIT:          # OFF by default
    @app.get("/api/v1/debug/ratelimit/{bucket}", include_in_schema=False)
    async def rl_status(bucket: str, request: Request):
        """
        Inspect Redis keys for the current identifier + bucket.
        Handy for CI tests – **never enable in prod**.
        """
        key_prefix = f"ratelimit:{bucket}:{await user_or_ip(request)}"
        r = get_redis()
        keys = await r.keys(f"{key_prefix}*")
        values = await r.mget(keys) if keys else []
        return dict(zip(keys, values)) 
