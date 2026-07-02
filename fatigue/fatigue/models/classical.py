"""Classical ML baselines on engineered features.

These operate on the (n_windows, n_features) matrix from `build_feature_matrix`
and are the strongest *deployable* option: a small Random Forest / gradient
boosting model over ~a dozen features per channel is trivial to run on an MCU
and is far more sample-efficient than a deep net on a 13-subject cohort.

XGBoost / LightGBM are used if installed, otherwise scikit-learn equivalents.
"""
from __future__ import annotations

from typing import Callable, Dict

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def _random_forest(seed: int, n_classes: int):
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def _logreg(seed: int, n_classes: int):
    return LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=seed
    )


def _svm(seed: int, n_classes: int):
    return SVC(
        kernel="rbf", probability=True, class_weight="balanced", random_state=seed
    )


def _grad_boost(seed: int, n_classes: int):
    return GradientBoostingClassifier(random_state=seed)


def _xgboost(seed: int, n_classes: int):
    from xgboost import XGBClassifier  # optional dependency

    return XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic" if n_classes == 2 else "multi:softprob",
        num_class=None if n_classes == 2 else n_classes,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=seed,
    )


CLASSICAL_MODELS: Dict[str, Callable] = {
    "random_forest": _random_forest,
    "logistic_regression": _logreg,
    "svm": _svm,
    "gradient_boosting": _grad_boost,
    "xgboost": _xgboost,
}


def build_classical_model(name: str, seed: int = 42, n_classes: int = 2):
    if name not in CLASSICAL_MODELS:
        raise KeyError(f"Unknown model '{name}'. Options: {list(CLASSICAL_MODELS)}")
    return CLASSICAL_MODELS[name](seed, n_classes)
