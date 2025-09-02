from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from .bayes import BayesCancerParams

class IrisTrainRequest(BaseModel):
    """
    Kick off Iris model training.

    • `model_type` – 'rf' (Random‑Forest) | 'logreg'  
    • `hyperparams` – optional scikit‑learn overrides, e.g. {"n_estimators": 500}  
    • `async_training` – true ⇒ returns job_id immediately
    """
    model_type: str = Field(
        default="rf",
        description="Which Iris trainer to run: 'rf' or 'logreg'"
    )
    hyperparams: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional hyper‑parameter overrides"
    )
    async_training: bool = Field(
        default=False,
        description="Run in background and return job ID"
    )

class CancerTrainRequest(BaseModel):
    """
    Train Breast‑Cancer classifiers.

    • `model_type` – 'bayes' (hier‑Bayes) | 'stub' (quick LogisticRegression)  
    • `params` – validated Bayesian hyper‑parameters (only used when model_type='bayes')  
    • `async_training` – background flag
    """
    model_type: str = Field(
        default="bayes",
        description="Which cancer model to train: 'bayes' or 'stub'"
    )
    params: Optional[BayesCancerParams] = Field(
        default=None,
        description="Bayesian hyper‑parameters; ignored for stub model"
    )
    async_training: bool = Field(
        default=False,
        description="Run in background and return job ID"
    )

class BayesTrainRequest(BaseModel):
    """Request model for Bayesian cancer model training"""
    params: Optional[BayesCancerParams] = Field(
        default=None, 
        description="Bayesian hyperparameters. If None, uses defaults."
    )
    async_training: bool = Field(
        default=False,
        description="If True, returns job_id immediately. If False, waits for completion."
    )

class BayesTrainResponse(BaseModel):
    """Response model for Bayesian training"""
    run_id: str = Field(description="MLflow run ID")
    job_id: Optional[str] = Field(default=None, description="Background job ID if async")
    status: str = Field(description="Training status: 'completed', 'queued', 'failed'")
    message: Optional[str] = Field(default=None, description="Status message or error")

class BayesConfigResponse(BaseModel):
    """Response model for Bayesian configuration endpoint"""
    defaults: BayesCancerParams = Field(description="Default hyperparameters")
    bounds: dict = Field(description="Parameter bounds for UI controls")
    descriptions: dict = Field(description="Parameter descriptions for tooltips")
    runtime_estimate: dict = Field(description="Runtime estimation factors")

class BayesRunMetrics(BaseModel):
    """Response model for Bayesian run metrics"""
    run_id: str
    accuracy: float
    rhat_max: Optional[float] = None
    ess_bulk_min: Optional[float] = None
    ess_tail_min: Optional[float] = None
    waic: Optional[float] = None
    loo: Optional[float] = None
    status: str
    warnings: list[str] = Field(default_factory=list) 
