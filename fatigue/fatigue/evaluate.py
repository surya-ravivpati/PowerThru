"""Classification metrics for fatigue detection.

Because the task is class-imbalanced (most windows are "not fatigued") and
subject-generalising, plain accuracy is misleading. Balanced accuracy, MCC,
Cohen's kappa and PR-AUC are the headline numbers; per-subject and per-movement
breakdowns expose where the model fails to generalise.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True-negative rate (binary)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute the full metric suite.

    y_score: predicted probability of the positive class (binary) used for
    ROC-AUC / PR-AUC. Optional.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    avg = "binary" if n_classes <= 2 else "macro"

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    if n_classes <= 2:
        metrics["specificity"] = specificity_score(y_true, y_pred)
    if y_score is not None and n_classes == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            pass  # single-class fold
    return metrics


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def per_group_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray
) -> Dict[int, Dict[str, float]]:
    """Metrics computed separately for each group id (subject or movement)."""
    out: Dict[int, Dict[str, float]] = {}
    for g in np.unique(groups):
        mask = groups == g
        out[int(g)] = compute_metrics(y_true[mask], y_pred[mask])
    return out
