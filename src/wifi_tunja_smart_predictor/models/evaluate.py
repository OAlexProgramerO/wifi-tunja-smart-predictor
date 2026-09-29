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
        try:
            proba = estimator.predict_proba(features)
            classes = list(estimator.classes_)
            if POSITIVE_LABEL in classes:
                positive_index = classes.index(POSITIVE_LABEL)
                y_bin = (y_true_arr == POSITIVE_LABEL).astype(int)
                metrics["roc_auc"] = float(roc_auc_score(y_bin, proba[:, positive_index]))
        except ValueError:
            # ROC-AUC is undefined if only one class is present in y_true.
            metrics["roc_auc"] = None
    return metrics
