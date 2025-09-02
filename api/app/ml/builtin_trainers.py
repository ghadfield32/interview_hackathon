# api/ml/builtin_trainers.py
"""
Built-in trainers for Iris RF and Breast-Cancer Bayesian LogReg.
Executed automatically by ModelService when a model is missing.
"""

import logging
logger = logging.getLogger(__name__)

from pathlib import Path
import mlflow, mlflow.sklearn, mlflow.pyfunc
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
import tempfile
import pickle
import warnings
import subprocess
import os
import platform
from app.core.config import settings

# Conditional imports for heavy dependencies
if os.getenv("UNIT_TESTING") != "1" and os.getenv("SKIP_BACKGROUND_TRAINING") != "1":
    import pymc as pm
    import arviz as az
else:
    pm = None
    az = None

# --- ADD THIS NEAR THE TOP (after imports) ----------------------------------
def _ensure_experiment(name: str = "ml_fullstack_models") -> str:
    """
    Guarantee that `name` exists and return its experiment_id.
    Handles the MLflow race where set_experiment returns a dangling ID
    if the experiment folder has not been written yet.
    """
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(name)
    if exp is None:
        exp_id = client.create_experiment(name)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(name)          # marks it the active one
    return exp_id

# ------------------------------------------------------------------
# Honour whatever Settings or the shell already provided; then
# fall back if the host part cannot be resolved quickly.
# ------------------------------------------------------------------
from urllib.parse import urlparse
import socket, time

def _fast_resolve(uri: str) -> bool:
    if uri.startswith("http"):
        host = urlparse(uri).hostname
        try:
            t0 = time.perf_counter()
            socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
            return (time.perf_counter() - t0) < 0.05
        except socket.gaierror:
            return False
    return True



# MLflow tracking URI is now resolved in ModelService.initialize()
# Trainers assume MLflow is already configured and experiments exist
logger.info("📦 Trainers ready - MLflow URI will be resolved at service startup")

MLFLOW_EXPERIMENT = "ml_fullstack_models"

# Remove side-effectful MLflow calls at import time
# Experiments will be created on-demand in each trainer function

# --- psutil health probe ----------------------------------------------------
def _psutil_healthy() -> bool:
    """
    Return True if psutil imports cleanly *and* exposes a working Process() object.
    We cache the result because repeated checks are cheap but noisy in logs.
    """
    global _PSUTIL_HEALTH_CACHE
    try:
        return _PSUTIL_HEALTH_CACHE
    except NameError:
        pass

    ok = False
    try:
        import psutil  # type: ignore
        ok = hasattr(psutil, "Process")
        if ok:
            try:
                _ = psutil.Process().pid  # touch native layer
            except Exception:  # bad native ext
                ok = False
    except Exception:
        ok = False

    _PSUTIL_HEALTH_CACHE = ok
    if not ok:
        logger.warning("🩺 psutil unhealthy – disabling sklearn/joblib parallelism (n_jobs=1).")
    return ok


# -----------------------------------------------------------------------------
#  IRIS – point-estimate Random-Forest (enhanced with better parameters)
# -----------------------------------------------------------------------------
def train_iris_random_forest(
    n_estimators: int = 300,
    max_depth: int | None = None,
    random_state: int = 42,
) -> str:
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import mlflow, mlflow.pyfunc

    # 1️⃣  ALWAYS ensure the experiment exists *inside* the function
    _ensure_experiment(MLFLOW_EXPERIMENT)

    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )

    safe_jobs = -1 if _psutil_healthy() else 1
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=safe_jobs,
        class_weight="balanced",
    ).fit(X_tr, y_tr)

    preds = rf.predict(X_te)
    metrics = {
        "accuracy": accuracy_score(y_te, preds),
        "f1_macro": f1_score(y_te, preds, average="macro"),
        "precision_macro": precision_score(y_te, preds, average="macro"),
        "recall_macro": recall_score(y_te, preds, average="macro"),
    }

    class IrisRFWrapper(mlflow.pyfunc.PythonModel):
        def __init__(self, model):
            if hasattr(model, "n_jobs"):
                model.n_jobs = 1
            self.model = model
            self._cols = list(X.columns)

        def _df(self, arr):
            import pandas as pd, numpy as np
            if isinstance(arr, pd.DataFrame):
                return arr
            return pd.DataFrame(np.asarray(arr), columns=self._cols)

        def predict(self, context, model_input, params=None):
            X_ = self._df(model_input)
            return self.model.predict_proba(X_)

    with mlflow.start_run(run_name="iris_random_forest") as run:
        mlflow.log_metrics(metrics)
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
            "safe_n_jobs": safe_jobs,
        })

        sig = mlflow.models.signature.infer_signature(X, rf.predict_proba(X))
        mlflow.pyfunc.log_model(
            artifact_path="model",            # ✅ correct kw-arg
            python_model=IrisRFWrapper(rf),
            registered_model_name="iris_random_forest",
            input_example=X.head(),
            signature=sig,
        )
        return run.info.run_id


# -----------------------------------------------------------------------------
#  IRIS – logistic-regression trainer (NEW)
# -----------------------------------------------------------------------------

def train_iris_logreg(
    C: float = 1.0,
    max_iter: int = 400,
    random_state: int = 42,
) -> str:
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    import mlflow, mlflow.pyfunc

    # 1️⃣  ALWAYS ensure the experiment exists *inside* the function
    _ensure_experiment(MLFLOW_EXPERIMENT)

    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )

    safe_jobs = -1 if _psutil_healthy() else 1
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=safe_jobs,
        random_state=random_state,
    ).fit(X_tr, y_tr)

    preds = clf.predict(X_te)
    metrics = {
        "accuracy": accuracy_score(y_te, preds),
        "f1_macro": f1_score(y_te, preds, average="macro"),
        "precision_macro": precision_score(y_te, preds, average="macro"),
        "recall_macro": recall_score(y_te, preds, average="macro"),
    }

    class IrisLogRegWrapper(mlflow.pyfunc.PythonModel):
        def __init__(self, model):
            if hasattr(model, "n_jobs"):
                model.n_jobs = 1
            self.model = model
            self._cols = list(X.columns)

        def _df(self, arr):
            import pandas as pd, numpy as np
            if isinstance(arr, pd.DataFrame):
                return arr
            return pd.DataFrame(np.asarray(arr), columns=self._cols)

        def predict(self, context, model_input, params=None):
            X_ = self._df(model_input)
            return self.model.predict_proba(X_)

    with mlflow.start_run(run_name="iris_logreg") as run:
        mlflow.log_metrics(metrics)
        mlflow.log_params({
            "C": C,
            "max_iter": max_iter,
            "random_state": random_state,
            "safe_n_jobs": safe_jobs,
        })

        sig = mlflow.models.signature.infer_signature(X, clf.predict_proba(X))
        mlflow.pyfunc.log_model(
            artifact_path="model",            # ✅ correct kw-arg
            python_model=IrisLogRegWrapper(clf),
            registered_model_name="iris_logreg",
            input_example=X.head(),
            signature=sig,
        )
        return run.info.run_id


# -----------------------------------------------------------------------------
#  BREAST-CANCER STUB – ultra-fast fallback model
# -----------------------------------------------------------------------------
def train_breast_cancer_stub(random_state: int = 42) -> str:
    """
    Ultra-fast fallback binary LogisticRegression on the breast-cancer dataset.

    Serializes safely on Windows by forcing `n_jobs=1` if psutil unhealthy,
    and exports MLflow PythonModel w/ modern signature that returns P(malignant).
    References: joblib parallelism + psutil; MLflow PythonModel signature. :contentReference[oaicite:21]{index=21}
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import mlflow, tempfile, pickle, pandas as pd

    # 1️⃣  ALWAYS ensure the experiment exists *inside* the function
    _ensure_experiment(MLFLOW_EXPERIMENT)

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=random_state
    )

    safe_jobs = -1 if _psutil_healthy() else 1

    clf = LogisticRegression(
        max_iter=200, n_jobs=safe_jobs, random_state=random_state
    ).fit(Xtr, ytr)

    class CancerStubWrapper(mlflow.pyfunc.PythonModel):
        def __init__(self, model):
            if hasattr(model, "n_jobs"):
                try:
                    model.n_jobs = 1
                except Exception:
                    pass
            self.model = model
            self._cols = list(X.columns)

        def _df(self, arr):
            import pandas as pd, numpy as np
            if isinstance(arr, pd.DataFrame):
                return arr
            return pd.DataFrame(np.asarray(arr), columns=self._cols)

        def predict(self, context, model_input, params=None):
            proba = self.model.predict_proba(self._df(model_input))
            return proba[:, 1]  # malignant probability

        def predict_proba(self, X):
            return self.model.predict_proba(self._df(X))

    acc = accuracy_score(yte, clf.predict(Xte))

    with tempfile.TemporaryDirectory() as td, mlflow.start_run(run_name="breast_cancer_stub") as run:
        # Log config hash for reproducibility and drift detection
        from app.core.config import settings as _s
        import hashlib, json
        _cfg_hash = hashlib.sha256(json.dumps(_s.model_dump(), sort_keys=True).encode()).hexdigest()
        mlflow.log_param("train_config_hash", _cfg_hash)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_param("safe_n_jobs", safe_jobs)
        wrapper = CancerStubWrapper(clf)
        sig = mlflow.models.signature.infer_signature(X, wrapper.predict(None, X))
        mlflow.pyfunc.log_model(
            "model",
            python_model=wrapper,
            registered_model_name="breast_cancer_stub",
            input_example=X.head(),
            signature=sig,
        )
        return run.info.run_id


# -----------------------------------------------------------------------------
#  BREAST-CANCER – hierarchical Bayesian logistic regression
# -----------------------------------------------------------------------------

def train_breast_cancer_bayes(
    draws: int = 1000,
    tune: int = 1000,
    target_accept: float = 0.99,
    params_obj=None,   # optional BayesCancerParams
) -> str:
    """
    Hierarchical Bayesian logistic regression with varying intercepts.

    Adds:
      * Validated hyperparams via BayesCancerParams (if provided)
      * Convergence diagnostics: R-hat, bulk/tail ESS
      * Optional WAIC / LOO (guarded; can be disabled)
      * Logged warnings if thresholds exceeded
    """
    import pymc as pm
    import pandas as pd, numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler
    import mlflow, tempfile, pickle
    from pathlib import Path

    # Schema override if provided
    if params_obj is not None:
        draws = params_obj.draws
        tune = params_obj.tune
        target_accept = params_obj.target_accept
        compute_waic = params_obj.compute_waic
        compute_loo = params_obj.compute_loo
        max_rhat_warn = params_obj.max_rhat_warn
        min_ess_warn = params_obj.min_ess_warn
    else:
        compute_waic = True
        compute_loo = False
        max_rhat_warn = 1.01
        min_ess_warn = 400

    logger.info(
        "BayesCancer: draws=%d tune=%d target_accept=%.3f waic=%s loo=%s",
        draws, tune, target_accept, compute_waic, compute_loo
    )

    _ensure_experiment(MLFLOW_EXPERIMENT)

    X_df, y = load_breast_cancer(as_frame=True, return_X_y=True)
    quint, edges = pd.qcut(X_df["mean texture"], 5, labels=False, retbins=True)
    g = np.asarray(quint, dtype="int64")
    scaler = StandardScaler().fit(X_df)
    Xs = scaler.transform(X_df)

    coords = {"group": np.arange(5)}
    with pm.Model(coords=coords) as m:
        α = pm.Normal("α", 0.0, 1.0, dims="group")
        β = pm.Normal("β", 0.0, 1.0, shape=Xs.shape[1])
        logit = α[g] + pm.math.dot(Xs, β)
        pm.Bernoulli("obs", logit_p=logit, observed=y)
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=4,
            nuts_sampler="numpyro",
            target_accept=target_accept,
            progressbar=False,
        )

    # Diagnostics
    import arviz as az
    rhat = az.rhat(idata).to_array().values.max()
    ess_bulk = az.ess(idata, method="bulk").to_array().values.min()
    ess_tail = az.ess(idata, method="tail").to_array().values.min()
    waic_val = None
    loo_val = None
    try:
        if compute_waic:
            waic_val = float(az.waic(idata).waic)
    except Exception as e:
        logger.warning("WAIC computation failed: %s", e)
    try:
        if compute_loo:
            loo_val = float(az.loo(idata).loo)
    except Exception as e:
        logger.warning("LOO computation failed: %s", e)

    # Wrapper
    class _HierBayesWrapper(mlflow.pyfunc.PythonModel):
        def __init__(self, trace, sc, ed, cols):
            self.trace, self.scaler, self.edges, self.cols = trace, sc, ed, cols

        def _quint(self, df):
            col = "mean texture"
            if col not in df.columns and "mean_texture" in df.columns:
                df = df.rename(columns={"mean_texture": col})
            tex = df[col].to_numpy()
            return np.clip(np.digitize(tex, self.edges, right=False), 0, 4)

        def predict(self, context, model_input, params=None):
            df = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input, columns=self.cols)
            xs = self.scaler.transform(df)
            g = self._quint(df)
            αg = self.trace.posterior["α"].median(("chain", "draw")).values
            β = self.trace.posterior["β"].median(("chain", "draw")).values
            log = αg[g] + np.dot(xs, β)
            return 1.0 / (1.0 + np.exp(-log))

    wrapper = _HierBayesWrapper(idata, scaler, edges[1:-1], X_df.columns.tolist())
    preds = wrapper.predict(None, X_df)
    acc = float(((preds > 0.5).astype(int) == y).mean())

    # Threshold warnings
    if rhat > max_rhat_warn:
        logger.warning("R-hat exceeds threshold: %.4f > %.2f", rhat, max_rhat_warn)
    if ess_bulk < min_ess_warn:
        logger.warning("Bulk ESS below threshold: %.1f < %d", ess_bulk, min_ess_warn)

    with tempfile.TemporaryDirectory() as td, mlflow.start_run(run_name="breast_cancer_bayes") as run:
        from app.core.config import settings as _s
        import hashlib, json
        _cfg_hash = hashlib.sha256(json.dumps(_s.model_dump(), sort_keys=True).encode()).hexdigest()

        # Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("rhat_max", rhat)
        mlflow.log_metric("ess_bulk_min", ess_bulk)
        mlflow.log_metric("ess_tail_min", ess_tail)
        if waic_val is not None:
            mlflow.log_metric("waic", waic_val)
        if loo_val is not None:
            mlflow.log_metric("loo", loo_val)

        # Params
        mlflow.log_param("train_config_hash", _cfg_hash)
        mlflow.log_param("draws", draws)
        mlflow.log_param("tune", tune)
        mlflow.log_param("target_accept", target_accept)
        mlflow.log_param("compute_waic", compute_waic)
        mlflow.log_param("compute_loo", compute_loo)

        sc_path = Path(td) / "scaler.pkl"
        pickle.dump(scaler, open(sc_path, "wb"))
        mlflow.pyfunc.log_model(
            "model",
            python_model=wrapper,
            artifacts={"scaler": str(sc_path)},
            registered_model_name="breast_cancer_bayes",
            input_example=X_df.head(),
            signature=mlflow.models.signature.infer_signature(X_df, wrapper.predict(None, X_df)),
        )
        return run.info.run_id
