"""ColumnTransformer pipelines for numeric and categorical model inputs.

Learned steps (imputation, scaling, encoding) must be fitted on training rows
only. Callers are responsible for temporal splitting before ``fit``.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from wifi_tunja_smart_predictor.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from wifi_tunja_smart_predictor.features.engineering import FeatureEngineer


def build_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    """Build the post-engineering ColumnTransformer.

    Tree models do not require scaling. Linear models, KNN, and GaussianNB
    benefit from standardised numeric columns. One-hot encoding uses
    ``handle_unknown='ignore'`` so unseen category levels at inference time
    become a zero vector instead of a runtime error.
    """
    numeric_steps: list[tuple] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_classifier_pipeline(estimator, *, scale_numeric: bool) -> Pipeline:
    """Feature engineering + preprocessing + classifier as one persistable pipeline."""
    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocess", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )
