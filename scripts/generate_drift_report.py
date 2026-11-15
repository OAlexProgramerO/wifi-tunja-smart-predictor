#!/usr/bin/env python3
"""Explicitly generate and persist a versioned synthetic feature drift report."""

from __future__ import annotations

import argparse
import json
import os

from wifi_tunja_smart_predictor.config import METRICS_DIR
from wifi_tunja_smart_predictor.models.drift import build_current_drift_report

DRIFT_REPORT_DIR = METRICS_DIR / "drift"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-split",
        choices=("validation", "test"),
        required=True,
        help="Synthetic temporal partition to compare with the training partition.",
    )
    args = parser.parse_args()
    report = build_current_drift_report(args.evaluation_split)
    DRIFT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    destination = (
        DRIFT_REPORT_DIR
        / f"synthetic_drift_v{report['schema_version']}_{args.evaluation_split}.json"
    )
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    print(
        f"Saved {report['status']} drift report to {destination.relative_to(DRIFT_REPORT_DIR.parent)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
