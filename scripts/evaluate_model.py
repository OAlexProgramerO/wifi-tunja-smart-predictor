#!/usr/bin/env python3
"""Evaluate the persisted pipeline on the temporal test holdout and write reports."""

from __future__ import annotations

import json
import logging

from wifi_tunja_smart_predictor.config import (
    CONFUSION_MATRIX_PATH,
    FIGURES_DIR,
    MODEL_INPUT_COLUMNS,
    MODEL_METADATA_PATH,
    PROCESSED_DATASET_PATH,
    SELECTED_MODEL_METRICS_PATH,
    SYNTHETIC_DATA_DISCLAIMER,
    TARGET_COLUMN,
)
from wifi_tunja_smart_predictor.data.loader import load_raw_dataset
from wifi_tunja_smart_predictor.models.evaluate import evaluate_classifier
from wifi_tunja_smart_predictor.models.predict import load_model
from wifi_tunja_smart_predictor.models.train import temporal_split
from wifi_tunja_smart_predictor.visualization.plots import (
    plot_confusion_matrix,
    plot_demand_by_hour,
    plot_demand_by_zone,
    plot_demand_distribution,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    source = PROCESSED_DATASET_PATH
    frame = load_raw_dataset(source)
    _train, _validation, test = temporal_split(frame)
    model = load_model()
    metrics = evaluate_classifier(model, test[MODEL_INPUT_COLUMNS], test[TARGET_COLUMN])

    SELECTED_MODEL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "disclaimer": SYNTHETIC_DATA_DISCLAIMER,
        "note": "Performance measured on the synthetic evaluation dataset.",
        "test_rows": int(len(test)),
        "test_start": str(test["timestamp"].min()),
        "test_end": str(test["timestamp"].max()),
        **{k: v for k, v in metrics.items()},
    }
    SELECTED_MODEL_METRICS_PATH.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: payload[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}, indent=2
        )
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_confusion_matrix(metrics["confusion_matrix"], metrics["labels"])
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=140)
    print(f"Wrote {CONFUSION_MATRIX_PATH}")

    # Static PNG fallbacks for documentation (aggregated Plotly figures exported via write_image
    # would require kaleido; HTML is written instead so no extra dependency is needed).
    plot_demand_distribution(frame).write_html(FIGURES_DIR / "demand_distribution.html")
    plot_demand_by_hour(frame).write_html(FIGURES_DIR / "demand_by_hour.html")
    plot_demand_by_zone(frame).write_html(FIGURES_DIR / "demand_by_zone.html")

    try:
        import matplotlib.pyplot as plt

        ax = (
            frame[TARGET_COLUMN]
            .value_counts()
            .plot(kind="bar", title="Demand distribution (synthetic)")
        )
        ax.figure.tight_layout()
        ax.figure.savefig(FIGURES_DIR / "demand_distribution.png", dpi=140)
        plt.close(ax.figure)

        hour_share = frame.groupby("hour")[TARGET_COLUMN].apply(lambda s: (s == "HIGH").mean())
        ax = hour_share.plot(kind="bar", title="HIGH share by hour (synthetic)")
        ax.figure.tight_layout()
        ax.figure.savefig(FIGURES_DIR / "demand_by_hour.png", dpi=140)
        plt.close(ax.figure)

        zone_share = (
            frame.groupby("zone_type")[TARGET_COLUMN]
            .apply(lambda s: (s == "HIGH").mean())
            .sort_values()
        )
        ax = zone_share.plot(kind="barh", title="HIGH share by zone type (synthetic)")
        ax.figure.tight_layout()
        ax.figure.savefig(FIGURES_DIR / "demand_by_zone.png", dpi=140)
        plt.close(ax.figure)
    except Exception as exc:  # pragma: no cover - reporting convenience only
        print(f"PNG figure export skipped: {exc}")

    if MODEL_METADATA_PATH.is_file():
        print(f"Model metadata: {MODEL_METADATA_PATH}")
    print(SYNTHETIC_DATA_DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
