"""Classification metrics computed from actual predictions only."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from wifi_tunja_smart_predictor.config import POSITIVE_LABEL, TARGET_LABELS


def evaluate_classifier(estimator: Any, features: pd.DataFrame, y_true: pd.Series) -> dict[str, Any]:
    """Return accuracy, precision, recall, F1, optional ROC-AUC, and confusion matrix.

    Precision, recall, and F1 use ``HIGH`` as the positive class so the minority
    demand class is visible even when overall accuracy looks strong.
    """
    y_true_arr = np.asarray(y_true).astype(str)
    y_pred = np.asarray(estimator.predict(features)).astype(str)
    if not len(y_true_arr) or y_pred.shape != y_true_arr.shape:
        raise ValueError("Classification metrics require matching, non-empty targets and predictions.")
    if not set(y_true_arr).issubset(TARGET_LABELS) or not set(y_pred).issubset(TARGET_LABELS):
        raise ValueError("Classification metrics require only the supported LOW/HIGH labels.")
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(
            precision_score(
                y_true_arr, y_pred, pos_label=POSITIVE_LABEL, zero_division=0
            )
        ),
        "recall": float(
            recall_score(y_true_arr, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true_arr, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "roc_auc": None,
        "confusion_matrix": confusion_matrix(
            y_true_arr, y_pred, labels=list(TARGET_LABELS)
        ).tolist(),
        "labels": list(TARGET_LABELS),
    }
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(features), dtype=float)
        classes = list(estimator.classes_)
        if (
            proba.shape != (len(y_true_arr), len(classes))
            or not np.isfinite(proba).all()
            or (proba < 0).any()
            or (proba > 1).any()
            or not np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        ):
            raise ValueError("Classifier probabilities are invalid for metric evaluation.")
        try:
            if POSITIVE_LABEL in classes:
                positive_index = classes.index(POSITIVE_LABEL)
                y_bin = (y_true_arr == POSITIVE_LABEL).astype(int)
                metrics["roc_auc"] = float(roc_auc_score(y_bin, proba[:, positive_index]))
        except ValueError:
            # ROC-AUC is undefined if only one class is present in y_true.
            metrics["roc_auc"] = None
    return metrics
