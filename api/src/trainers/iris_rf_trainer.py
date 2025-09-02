# api/src/trainers/iris_rf_trainer.py
from __future__ import annotations
from .base import BaseTrainer, TrainResult
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow

class IrisRandomForestTrainer(BaseTrainer):
    name = "iris_random_forest"
    model_type = "classification"

    def default_hyperparams(self):
        return {
            "n_estimators": 300,
            "max_depth": None,
            "random_state": 42,
        }

    def train(self, **overrides) -> TrainResult:
        hp = self.merge_hyperparams(overrides)
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=hp["random_state"]
        )
        rf = RandomForestClassifier(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            random_state=hp["random_state"],
            n_jobs=-1,
            class_weight="balanced",
        ).fit(Xtr, ytr)

        preds = rf.predict(Xte)
        metrics = {
            "accuracy": accuracy_score(yte, preds),
            "f1_macro": f1_score(yte, preds, average="macro"),
            "precision_macro": precision_score(yte, preds, average="macro"),
            "recall_macro": recall_score(yte, preds, average="macro"),
        }

        class _Wrapper(mlflow.pyfunc.PythonModel):
            def __init__(self, model, cols):
                self.model = model
                self.cols = cols
            def predict(self, context, model_input, params=None):
                import pandas as pd, numpy as np
                df = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input, columns=self.cols)
                return self.model.predict_proba(df)

        with mlflow.start_run(run_name=self.name) as run:
            mlflow.log_params({k: v for k, v in hp.items()})
            mlflow.log_metrics(metrics)
            sig = mlflow.models.signature.infer_signature(X, rf.predict_proba(X))
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=_Wrapper(rf, list(X.columns)),
                registered_model_name=self.name,
                input_example=X.head(),
                signature=sig,
            )
            return TrainResult(run_id=run.info.run_id, metrics=metrics) 
