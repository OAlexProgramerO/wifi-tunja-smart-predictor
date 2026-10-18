from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wifi_tunja_smart_predictor.models.evaluate import evaluate_classifier
from wifi_tunja_smart_predictor.models.regression import evaluate_regressor


class _InvalidProbabilityClassifier:
    classes_ = np.asarray(["LOW", "HIGH"])

    def predict(self, features):
        return np.resize(np.asarray(["LOW", "HIGH"]), len(features))

    def predict_proba(self, features):
        return np.tile([1.1, -0.1], (len(features), 1))


class _InfiniteRegressor:
    def predict(self, features):
        return np.full(len(features), np.inf)


def test_classifier_evaluation_rejects_invalid_probabilities():
    features = pd.DataFrame({"feature": [1, 2]})
    targets = pd.Series(["LOW", "HIGH"])

    with pytest.raises(ValueError, match="probabilities are invalid"):
        evaluate_classifier(_InvalidProbabilityClassifier(), features, targets)


def test_regression_evaluation_rejects_non_finite_predictions():
    features = pd.DataFrame({"feature": [1, 2]})
    targets = pd.Series([1.0, 2.0])

    with pytest.raises(ValueError, match="finite"):
        evaluate_regressor(_InfiniteRegressor(), features, targets)
