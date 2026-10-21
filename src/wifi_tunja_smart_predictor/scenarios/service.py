"""Combined classification, regression, uncertainty, capacity, and proxy explanation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wifi_tunja_smart_predictor.config import (
    CLASSIFICATION_METADATA_PATH,
    CLASSIFICATION_THRESHOLD,
    DATASET_VERSION,
    INTERVAL_CONFIDENCE,
    MODEL_INPUT_COLUMNS,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PREPROCESSING_VERSION,
    REGRESSION_METADATA_PATH,
    REGRESSION_MODEL_PATH,
    SYNTHETIC_DATA_DISCLAIMER,
    TARGET_LABELS,
)
from wifi_tunja_smart_predictor.data.loader import load_analysis_dataset
from wifi_tunja_smart_predictor.exceptions import ModelMetadataError, PredictionError
from wifi_tunja_smart_predictor.models.artifacts import (
    artifact_matches_metadata,
    load_classifier_metadata,
    load_regression_artifact,
    load_regression_metadata,
    load_regression_model,
    model_bundle_metadata_matches,
)
from wifi_tunja_smart_predictor.models.predict import load_model, load_model_artifact
from wifi_tunja_smart_predictor.scenarios.builder import (
    ScenarioBuilder,
    ScenarioContext,
    ScenarioRequest,
)

FEATURE_LABELS = {
    "connections_same_hour_previous_day": "Connections at the same hour yesterday",
    "connections_previous_hour": "Connections in the previous hour",
    "average_connections_last_24_hours": "Average connections over the previous day",
    "average_connections_last_7_days": "Average connections over the previous week",
    "estimated_people_nearby": "Estimated nearby population",
    "nearby_business_activity": "Nearby business activity",
    "nearby_student_population": "Nearby student population",
    "time_period": "Time of day",
    "hour": "Hour of day",
    "traffic_level": "Simulated traffic level",
    "zone_type": "Synthetic zone type",
    "connected_devices": "Devices in the previous network snapshot",
    "channel_utilization_percent": "Channel utilization in the previous snapshot",
}


@dataclass(frozen=True)
class PredictionResult:
    """Structured, reproducible model-based estimate for a synthetic scenario."""

    scenario: dict[str, Any]
    classification: dict[str, Any]
    regression: dict[str, Any]
    capacity: dict[str, Any]
    explanation: dict[str, Any]
    traceability: dict[str, Any] | None = None
    disclaimer: str = SYNTHETIC_DATA_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def _json_metadata(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


class ScenarioPredictionService:
    """Load V3 models once and serve scenario predictions through reusable services."""

    def __init__(
        self,
        frame: pd.DataFrame | None = None,
        classifier: Any | None = None,
        regressor: Any | None = None,
        *,
        classifier_path: Path = MODEL_PATH,
        regressor_path: Path = REGRESSION_MODEL_PATH,
    ) -> None:
        self.frame = frame if frame is not None else load_analysis_dataset()
        dataset_hasher = hashlib.sha256()
        dataset_hasher.update("\0".join(str(column) for column in self.frame.columns).encode())
        dataset_hasher.update(
            pd.util.hash_pandas_object(self.frame, index=True).to_numpy().tobytes()
        )
        self.dataset_sha256 = dataset_hasher.hexdigest()
        self.builder = ScenarioBuilder(self.frame)
        self.classifier_artifact = (
            load_model_artifact(str(classifier_path)) if classifier is None else None
        )
        self.regressor_artifact = (
            load_regression_artifact(regressor_path) if regressor is None else None
        )
        self.classifier = (
            self.classifier_artifact.estimator
            if self.classifier_artifact is not None
            else classifier
        )
        self.regressor = (
            self.regressor_artifact.estimator if self.regressor_artifact is not None else regressor
        )
        if self.classifier is None:
            self.classifier = load_model(str(classifier_path))
        if self.regressor is None:
            self.regressor = load_regression_model(regressor_path)
        if classifier is None and regressor is None:
            self.classification_metadata = load_classifier_metadata()
            self.regression_metadata = load_regression_metadata()
            if self.classifier_artifact is None or self.regressor_artifact is None:
                raise ModelMetadataError("Required model artifact identities are unavailable.")
            if not artifact_matches_metadata(
                self.classification_metadata, self.classifier_artifact
            ) or not artifact_matches_metadata(self.regression_metadata, self.regressor_artifact):
                raise ModelMetadataError("Model artifact integrity metadata does not match.")
            if self.classification_metadata["project_version"] != self.regression_metadata[
                "project_version"
            ] or not model_bundle_metadata_matches(
                self.classification_metadata, self.regression_metadata
            ):
                raise ModelMetadataError("Classifier and regressor metadata are inconsistent.")
            self.model_metadata = self.classification_metadata
        else:
            self.classification_metadata = _json_metadata(CLASSIFICATION_METADATA_PATH)
            self.regression_metadata = _json_metadata(REGRESSION_METADATA_PATH)
            self.model_metadata = _json_metadata(MODEL_METADATA_PATH)

    def predict(self, request: ScenarioRequest) -> PredictionResult:
        """Build the scenario and return both model estimates plus calibrated interval."""
        context = self.builder.build(request)
        features = pd.DataFrame([context.features], columns=MODEL_INPUT_COLUMNS)
        try:
            labels = np.asarray(self.classifier.predict(features)).astype(str)
            probabilities = self.classifier.predict_proba(features)[0]
            classes = [str(label) for label in self.classifier.classes_]
            if not set(classes).issuperset(TARGET_LABELS):
                raise PredictionError(f"Classifier has unsupported classes: {classes}")
            probability_high = float(probabilities[classes.index("HIGH")])
            probability_low = float(probabilities[classes.index("LOW")])
            predicted_connections = max(0.0, float(self.regressor.predict(features)[0]))
        except (AttributeError, ValueError, KeyError) as exc:
            raise PredictionError(f"Model artifacts cannot score the scenario: {exc}") from exc

        interval_confidence = float(
            self.regression_metadata.get("interval_confidence", INTERVAL_CONFIDENCE)
        )
        residual_radius = float(self.regression_metadata.get("calibration_radius", 0.0))
        lower = max(0.0, predicted_connections - residual_radius)
        upper = predicted_connections + residual_radius
        capacity = context.location.access_point_capacity
        utilization = predicted_connections / capacity * 100 if capacity else None
        headroom = max(0, int(round(capacity - predicted_connections)))
        factors = self._explain(context, probability_high, predicted_connections)

        from wifi_tunja_smart_predictor.models.traceability import build_prediction_traceability

        return PredictionResult(
            scenario={
                **context.to_dict(),
                "scenario_location": context.location.zone_name,
                "classification_threshold": CLASSIFICATION_THRESHOLD,
                "model_version": self.classification_metadata.get("project_version"),
                "dataset_version": self.regression_metadata.get("dataset_version", DATASET_VERSION),
                "dataset_sha256": self.dataset_sha256,
                "dataset_fingerprint_algorithm": "pandas_hash_pandas_object_sha256_v1",
                "preprocessing_version": self.regression_metadata.get(
                    "preprocessing_version", PREPROCESSING_VERSION
                ),
                "context_source": "synthetic historical analogs",
            },
            classification={
                "predicted_demand_level": str(labels[0]),
                "probability_low": probability_low,
                "probability_high": probability_high,
                "classification_threshold": CLASSIFICATION_THRESHOLD,
                "model_version": self.classification_metadata.get("project_version"),
                "selected_model": self.classification_metadata.get("selected_model"),
            },
            regression={
                "predicted_connections_next_hour": predicted_connections,
                "prediction_interval_lower": lower,
                "prediction_interval_upper": upper,
                "interval_confidence": interval_confidence,
                "calibration_radius_connections": residual_radius,
                "uncertainty_method": "split conformal absolute-residual quantile on validation",
            },
            capacity={
                "access_point_capacity": capacity,
                "predicted_capacity_utilization_pct": utilization,
                "estimated_capacity_headroom": headroom,
                "estimated_over_capacity_connections": max(0.0, predicted_connections - capacity),
                "interpretation": "Model-derived scenario estimate; connection count is used as a capacity proxy.",
            },
            explanation={
                "top_factors": factors,
                "method": "one-feature-at-a-time replacement against matched analog medians/modes",
                "disclaimer": "Factors describe model sensitivity, not causes or causal effects.",
            },
            traceability=build_prediction_traceability(
                classifier_metadata=self.classification_metadata,
                classifier_artifact=self.classifier_artifact,
                regressor_metadata=self.regression_metadata,
                regressor_artifact=self.regressor_artifact,
            ),
        )

    def _explain(
        self,
        context: ScenarioContext,
        probability_high: float,
        predicted_connections: float,
    ) -> list[dict[str, Any]]:
        """Approximate local model sensitivity with deterministic feature replacement."""
        baseline = context._analog_baseline
        original = pd.DataFrame([context.features], columns=MODEL_INPUT_COLUMNS)
        object_features = original.astype(object)
        ranked: list[dict[str, Any]] = []
        for column in MODEL_INPUT_COLUMNS:
            reference = baseline[column]
            observed = context.features[column]
            if pd.isna(reference) or observed == reference:
                continue
            changed = object_features.copy()
            changed.at[changed.index[0], column] = reference
            alternative = float(
                self.classifier.predict_proba(changed)[0][
                    list(self.classifier.classes_).index("HIGH")
                ]
            )
            delta = probability_high - alternative
            if abs(delta) < 0.005:
                continue
            direction = "higher HIGH score" if delta > 0 else "lower HIGH score"
            ranked.append(
                {
                    "feature": column,
                    "label": FEATURE_LABELS.get(column, column.replace("_", " ").capitalize()),
                    "direction": direction,
                    "probability_high_change": float(delta),
                    "observed_value": _python_value(observed),
                    "analog_reference_value": _python_value(reference),
                }
            )
        ranked.sort(key=lambda item: abs(item["probability_high_change"]), reverse=True)
        if not ranked:
            ranked = [
                {
                    "feature": "historical_analog_context",
                    "label": "Matched historical synthetic context",
                    "direction": "supports model estimate",
                    "probability_high_change": 0.0,
                    "observed_value": round(predicted_connections, 2),
                    "analog_reference_value": round(probability_high, 4),
                }
            ]
        return ranked[:5]


def _python_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
