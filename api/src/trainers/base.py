# api/src/trainers/base.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol
import mlflow

@dataclass
class TrainResult:
    run_id: str
    metrics: Dict[str, float]
    artifacts: Dict[str, str] = field(default_factory=dict)

class SupportsPyFunc(Protocol):
    # Minimal protocol if custom loader is needed later
    def predict(self, X): ...

class BaseTrainer:
    """
    Minimal trainer abstraction:
      * implement `train(**hyperparams)` returning TrainResult
      * optionally override default_hyperparams()
    """
    name: str
    model_type: str = "generic"

    def default_hyperparams(self) -> Dict[str, Any]:
        return {}

    def merge_hyperparams(self, overrides: Dict[str, Any] | None) -> Dict[str, Any]:
        params = self.default_hyperparams().copy()
        if overrides:
            params.update({k: v for k, v in overrides.items() if v is not None})
        return params

    def train(self, **hyperparams) -> TrainResult:  # pragma: no cover - interface
        raise NotImplementedError

    # Optional hook – if a trainer needs a special load path
    def load_pyfunc(self, run_uri: str):
        return mlflow.pyfunc.load_model(run_uri) 
