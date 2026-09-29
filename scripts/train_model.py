#!/usr/bin/env python3
"""Train baseline classifiers, select by validation F1, persist the pipeline."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import joblib

from wifi_tunja_smart_predictor.config import (
    MODEL_COMPARISON_PATH,
    MODEL_INPUT_COLUMNS,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PROCESSED_DATASET_PATH,
    PROJECT_ROOT,
    PROJECT_VERSION,
    SYNTHETIC_DATA_DISCLAIMER,
    TARGET_COLUMN,
    TRAIN_END,
    VALIDATION_END,
)
from wifi_tunja_smart_predictor.data.loader import load_raw_dataset
from wifi_tunja_smart_predictor.models.evaluate import evaluate_classifier
from wifi_tunja_smart_predictor.models.train import temporal_split, train_baseline_models

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    source = PROCESSED_DATASET_PATH if PROCESSED_DATASET_PATH.is_file() else None
    if source is None:
        print("Processed dataset missing; load raw CSV instead.")
        from wifi_tunja_smart_predictor.config import RAW_DATASET_PATH

        frame = load_raw_dataset(RAW_DATASET_PATH).drop_duplicates()
    else:
        print(f"Loading processed dataset: {source}")
        frame = load_raw_dataset(source)

    train, validation, test = temporal_split(frame)
    print(f"Train {train['timestamp'].min()} -> {train['timestamp'].max()} ({len(train):,} rows)")
    print(
        "Validation "
        f"{validation['timestamp'].min()} -> {validation['timestamp'].max()} ({len(validation):,} rows)"
    )
    print(f"Test {test['timestamp'].min()} -> {test['timestamp'].max()} ({len(test):,} rows)")
    print("Class distribution (train):")
    print(train[TARGET_COLUMN].value_counts(normalize=True).to_string())

    comparison, fitted, selected_name = train_baseline_models(train, validation, test)
    MODEL_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    print("\nTest-set comparison (synthetic evaluation only):")
    print(comparison.to_string(index=False))

    selected = fitted[selected_name]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, MODEL_PATH)
    print(f"\nPersisted pipeline: {MODEL_PATH}")

    test_metrics = evaluate_classifier(selected, test[MODEL_INPUT_COLUMNS], test[TARGET_COLUMN])
    validation_metrics = evaluate_classifier(
        selected, validation[MODEL_INPUT_COLUMNS], validation[TARGET_COLUMN]
    )
    comparison_by_model = comparison.set_index("model")
    metadata = {
        "project": "WiFi Tunja Smart Predictor",
        "disclaimer": SYNTHETIC_DATA_DISCLAIMER,
        "synthetic_data_disclaimer": SYNTHETIC_DATA_DISCLAIMER,
        "project_version": PROJECT_VERSION,
        "model_type": type(selected.named_steps["model"]).__name__,
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": "synthetic_seed_42",
        "target_column": TARGET_COLUMN,
        "selected_model": selected_name,
        "selection_rule": "Highest validation F1 for class HIGH; test set was not used for selection.",
        "model_path": MODEL_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "feature_count_model_input": len(MODEL_INPUT_COLUMNS),
        "feature_count": len(MODEL_INPUT_COLUMNS),
        "feature_names": MODEL_INPUT_COLUMNS,
        "model_parameters": selected.named_steps["model"].get_params(),
        "training_duration_seconds": float(
            comparison_by_model.loc[selected_name, "training_time_seconds"]
        ),
        "validation_metrics_selected_model": {
            k: v for k, v in validation_metrics.items() if k != "confusion_matrix"
        },
        "train_period": {
            "start": str(train["timestamp"].min()),
            "end": str(train["timestamp"].max()),
            "rows": int(len(train)),
            "cutoff": TRAIN_END,
        },
        "validation_period": {
            "start": str(validation["timestamp"].min()) if len(validation) else None,
            "end": str(validation["timestamp"].max()) if len(validation) else None,
            "rows": int(len(validation)),
            "cutoff": VALIDATION_END,
        },
        "test_period": {
            "start": str(test["timestamp"].min()),
            "end": str(test["timestamp"].max()),
            "rows": int(len(test)),
        },
        "train_start": str(train["timestamp"].min()),
        "train_end": str(train["timestamp"].max()),
        "validation_start": str(validation["timestamp"].min()),
        "validation_end": str(validation["timestamp"].max()),
        "test_start": str(test["timestamp"].min()),
        "test_end": str(test["timestamp"].max()),
        "train_class_distribution": train[TARGET_COLUMN].value_counts(normalize=True).to_dict(),
        "test_metrics_selected_model": {
            k: v for k, v in test_metrics.items() if k != "confusion_matrix"
        },
        "confusion_matrix": test_metrics["confusion_matrix"],
        "confusion_matrix_labels": test_metrics["labels"],
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(SYNTHETIC_DATA_DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
