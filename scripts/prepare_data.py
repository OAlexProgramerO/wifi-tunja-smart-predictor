#!/usr/bin/env python3
"""Load, validate, deduplicate, engineer features, and write processed CSV.

The raw file ``data/raw/wifi_tunja_public.csv`` is never overwritten.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from wifi_tunja_smart_predictor.config import (
    PROCESSED_DATASET_PATH,
    RAW_DATASET_PATH,
    SAMPLE_DATASET_PATH,
    SYNTHETIC_DATA_DISCLAIMER,
)
from wifi_tunja_smart_predictor.data.loader import load_raw_dataset
from wifi_tunja_smart_predictor.data.validator import validate_dataset
from wifi_tunja_smart_predictor.features.engineering import add_engineered_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("prepare_data")


def main() -> int:
    print("Loading raw synthetic dataset...")
    frame = load_raw_dataset(RAW_DATASET_PATH)
    report = validate_dataset(frame, raise_on_critical=True)
    print(f"Validation: {len(report.expected_issues)} expected quality notes.")
    print(f"Target distribution: {report.target_distribution}")

    before = len(frame)
    cleaned = frame.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(cleaned)} exact duplicate rows.")

    processed = add_engineered_features(cleaned)
    PROCESSED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(PROCESSED_DATASET_PATH, index=False)
    print(f"Wrote processed dataset: {PROCESSED_DATASET_PATH}")

    SAMPLE_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample = processed.sample(n=min(800, len(processed)), random_state=42)
    sample.to_csv(SAMPLE_DATASET_PATH, index=False)
    print(f"Wrote sample dataset: {SAMPLE_DATASET_PATH}")

    transform_log = {
        "disclaimer": SYNTHETIC_DATA_DISCLAIMER,
        "raw_rows": before,
        "raw_columns": int(frame.shape[1]),
        "processed_rows": int(len(processed)),
        "processed_columns": int(processed.shape[1]),
        "date_range": {
            "start": str(pd.to_datetime(cleaned["timestamp"]).min()),
            "end": str(pd.to_datetime(cleaned["timestamp"]).max()),
        },
        "target_distribution": report.target_distribution,
        "expected_quality_issues": report.expected_issues,
        "missing_counts": report.missing_counts,
        "duplicates_removed": int(before - len(cleaned)),
        "transformations": [
            "Parsed timestamps on load.",
            "Validated schema and targets; missing values in sensor-like columns were retained.",
            "Removed exact duplicate rows (generator-injected copies).",
            "Added cyclical and network-ratio features without using target columns.",
        ],
    }
    log_path = PROCESSED_DATASET_PATH.with_suffix(".json")
    log_path.write_text(json.dumps(transform_log, indent=2), encoding="utf-8")
    print(SYNTHETIC_DATA_DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
