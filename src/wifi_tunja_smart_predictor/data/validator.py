"""Dataset validation that separates expected quality issues from contract failures.

The synthetic generator injects a small amount of missingness and exact duplicate
rows. Those are expected quality issues. Missing target labels, absent required
columns, or unparsable timestamps are critical contract violations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from wifi_tunja_smart_predictor.config import (
    CATEGORICAL_ALLOWED_VALUES,
    EXPECTED_MISSING_COLUMNS,
    RANGE_CHECKS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    TARGET_COLUMNS,
    TARGET_LABELS,
)
from wifi_tunja_smart_predictor.exceptions import DatasetValidationError

logger = logging.getLogger(__name__)

# Missingness above this share on an "expected missing" column is still a contract issue.
MAX_EXPECTED_MISSING_RATE = 0.05
MAX_DUPLICATE_RATE = 0.01


@dataclass
class ValidationReport:
    """Structured outcome of a validation pass."""

    n_rows: int
    n_columns: int
    critical_issues: list[str] = field(default_factory=list)
    expected_issues: list[str] = field(default_factory=list)
    missing_counts: dict[str, int] = field(default_factory=dict)
    duplicate_count: int = 0
    target_distribution: dict[str, float] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.critical_issues


def validate_dataset(
    frame: pd.DataFrame,
    *,
    raise_on_critical: bool = True,
) -> ValidationReport:
    """Validate schema, targets, ranges, and expected quality artefacts."""
    critical: list[str] = []
    expected: list[str] = []

    missing_required = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing_required:
        critical.append(f"Missing required columns: {missing_required}")

    report = ValidationReport(n_rows=len(frame), n_columns=frame.shape[1])
    if missing_required:
        report.critical_issues = critical
        if raise_on_critical:
            raise DatasetValidationError("\n".join(critical))
        return report

    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", format="mixed")
    if timestamps.isna().any():
        critical.append("One or more timestamps could not be parsed.")
    elif not pd.api.types.is_datetime64_any_dtype(timestamps.dtype):
        critical.append("Timestamps must use one consistent timezone representation.")
    elif isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        critical.append("Timestamps must be timezone-naive synthetic local times.")

    categorical_columns = set(CATEGORICAL_ALLOWED_VALUES)
    identifier_columns = {"timestamp", "date", "day_name", "wifi_id", "zone_id", "zone_name"}
    numeric_columns = set(REQUIRED_COLUMNS) - categorical_columns - identifier_columns
    numeric_values: dict[str, pd.Series] = {}
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & values.isna()
        if invalid.any():
            critical.append(f"'{column}' contains values that are not numeric.")
        observed = values.dropna().to_numpy(dtype=float)
        if not np.isfinite(observed).all():
            critical.append(f"'{column}' contains a non-finite numeric value.")
        numeric_values[column] = values

    for target in TARGET_COLUMNS:
        if frame[target].isna().any():
            critical.append(f"Target '{target}' contains missing values.")

    invalid_labels = set(frame[TARGET_COLUMN].dropna().astype(str).unique()) - set(TARGET_LABELS)
    if invalid_labels:
        critical.append(f"demand_level has invalid labels: {sorted(invalid_labels)}")

    if "latitude" in frame.columns:
        lat = pd.to_numeric(frame["latitude"], errors="coerce")
        if ((lat < -90) | (lat > 90)).any():
            critical.append("latitude is outside the valid geographic range [-90, 90].")
    if "longitude" in frame.columns:
        lon = pd.to_numeric(frame["longitude"], errors="coerce")
        if ((lon < -180) | (lon > 180)).any():
            critical.append("longitude is outside the valid geographic range [-180, 180].")

    for column, (low, high) in RANGE_CHECKS.items():
        if column not in frame.columns:
            continue
        values = numeric_values.get(column, pd.to_numeric(frame[column], errors="coerce"))
        observed = values.dropna()
        if observed.empty:
            continue
        if observed.min() < low or observed.max() > high:
            expected.append(
                f"'{column}' has values outside the documented synthetic range "
                f"[{low}, {high}] (min={observed.min()}, max={observed.max()})."
            )

    for column, allowed in CATEGORICAL_ALLOWED_VALUES.items():
        if column not in frame.columns:
            continue
        observed = set(frame[column].dropna().astype(str).unique())
        unexpected = observed - allowed
        if unexpected:
            critical.append(f"Unexpected values in '{column}': {sorted(unexpected)}")

    missing_counts = frame.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].to_dict()
    n_rows = max(len(frame), 1)
    for column, count in missing_counts.items():
        rate = count / n_rows
        if column in EXPECTED_MISSING_COLUMNS and rate <= MAX_EXPECTED_MISSING_RATE:
            expected.append(
                f"Column '{column}' has {count} missing values ({rate:.2%}); "
                "this is consistent with the synthetic generator."
            )
        elif column in TARGET_COLUMNS:
            continue
        else:
            if rate > MAX_EXPECTED_MISSING_RATE or column not in EXPECTED_MISSING_COLUMNS:
                # Identifiers and calendar fields must be complete; other unexpected
                # missingness is a contract problem.
                if column not in EXPECTED_MISSING_COLUMNS:
                    critical.append(
                        f"Unexpected missing values in '{column}': {count} ({rate:.2%})."
                    )
                else:
                    critical.append(
                        f"Missing rate for '{column}' is {rate:.2%}, above the 5% allowance."
                    )

    duplicate_count = int(frame.duplicated().sum())
    if duplicate_count:
        rate = duplicate_count / n_rows
        message = f"{duplicate_count} exact duplicate rows ({rate:.2%})."
        if rate <= MAX_DUPLICATE_RATE:
            expected.append(message + " The generator injects a small number of duplicates.")
        else:
            critical.append(message + " Duplicate rate exceeds 1%.")

    unique_observations = frame.drop_duplicates()
    if {"wifi_id", "timestamp"}.issubset(unique_observations.columns):
        conflicting_observations = int(
            unique_observations.duplicated(subset=["wifi_id", "timestamp"]).sum()
        )
        if conflicting_observations:
            critical.append(
                "Multiple non-identical observations share the same access point and timestamp."
            )

    distribution = (
        frame[TARGET_COLUMN].value_counts(normalize=True).astype(float).to_dict()
        if TARGET_COLUMN in frame.columns
        else {}
    )

    report.critical_issues = critical
    report.expected_issues = expected
    report.missing_counts = {str(k): int(v) for k, v in missing_counts.items()}
    report.duplicate_count = duplicate_count
    report.target_distribution = {str(k): float(v) for k, v in distribution.items()}

    if critical:
        logger.error("Critical validation issues: %s", critical)
        if raise_on_critical:
            raise DatasetValidationError("Dataset contract violated:\n- " + "\n- ".join(critical))
    else:
        logger.info(
            "Validation passed with %s expected quality notes and %s duplicates.",
            len(expected),
            duplicate_count,
        )
    return report
