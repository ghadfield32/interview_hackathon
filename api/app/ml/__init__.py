"""
ML sub-package – exposes built-in trainers so the service can import
`app.ml.builtin_trainers` with an absolute import.
"""

from .builtin_trainers import (
    train_iris_random_forest,
    train_iris_logreg,
    train_breast_cancer_bayes,
    train_breast_cancer_stub,
)

__all__ = [
    "train_iris_random_forest",
    "train_iris_logreg",
    "train_breast_cancer_bayes",
    "train_breast_cancer_stub",
] 
