"""Plotly helpers. Aggregations are preferred over plotting every raw row."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay

from wifi_tunja_smart_predictor.config import TARGET_COLUMN, TARGET_LABELS


def plot_demand_distribution(frame: pd.DataFrame) -> go.Figure:
    counts = frame[TARGET_COLUMN].value_counts().reindex(list(TARGET_LABELS)).fillna(0)
    fig = px.bar(
        x=counts.index.astype(str),
        y=counts.values,
        labels={"x": "Demand level", "y": "Observations"},
        title="Synthetic demand_level distribution",
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_demand_by_hour(frame: pd.DataFrame) -> go.Figure:
    grouped = (
        frame.groupby(["hour", TARGET_COLUMN], observed=True)
        .size()
        .reset_index(name="observations")
    )
    fig = px.bar(
        grouped,
        x="hour",
        y="observations",
        color=TARGET_COLUMN,
        barmode="stack",
        title="Demand level by hour of day (synthetic)",
    )
    fig.update_layout(template="plotly_white", xaxis=dict(dtick=1))
    return fig


def plot_demand_by_weekday(frame: pd.DataFrame) -> go.Figure:
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    work = frame.copy()
    work["weekday"] = work["day_of_week"].map(dict(enumerate(names)))
    grouped = (
        work.groupby(["weekday", TARGET_COLUMN], observed=True)
        .size()
        .reset_index(name="observations")
    )
    fig = px.bar(
        grouped,
        x="weekday",
        y="observations",
        color=TARGET_COLUMN,
        barmode="stack",
        category_orders={"weekday": names},
        title="Demand level by weekday (synthetic)",
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_demand_by_zone(frame: pd.DataFrame) -> go.Figure:
    grouped = (
        frame.groupby(["zone_type", TARGET_COLUMN], observed=True)
        .size()
        .reset_index(name="observations")
    )
    fig = px.bar(
        grouped,
        x="zone_type",
        y="observations",
        color=TARGET_COLUMN,
        barmode="stack",
        title="Demand level by synthetic zone type",
    )
    fig.update_layout(template="plotly_white", xaxis_tickangle=-30)
    return fig


def plot_demand_over_time(frame: pd.DataFrame) -> go.Figure:
    work = frame.copy()
    work["month_start"] = pd.to_datetime(work["timestamp"]).dt.to_period("M").dt.to_timestamp()
    grouped = (
        work.groupby(["month_start", TARGET_COLUMN], observed=True)
        .size()
        .reset_index(name="observations")
    )
    fig = px.line(
        grouped,
        x="month_start",
        y="observations",
        color=TARGET_COLUMN,
        title="Demand observations over time (monthly, synthetic)",
        markers=True,
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_network_metrics(frame: pd.DataFrame) -> go.Figure:
    """Hour-of-day means for selected network snapshot metrics."""
    metrics = [
        "connected_devices",
        "active_sessions",
        "channel_utilization_percent",
        "latency_ms",
        "bandwidth_usage_mbps",
        "packet_loss_percent",
        "signal_strength_dbm",
    ]
    source = frame.copy()
    if {"connected_devices", "access_point_capacity"} <= set(source.columns):
        capacity = pd.to_numeric(source["access_point_capacity"], errors="coerce").replace(
            0, np.nan
        )
        source["capacity_utilization_percent"] = (
            pd.to_numeric(source["connected_devices"], errors="coerce") / capacity * 100
        )
        metrics.append("capacity_utilization_percent")
    available = [col for col in metrics if col in source.columns]
    aggregated = source.groupby("hour", observed=True)[available].mean().reset_index()
    long = aggregated.melt(id_vars="hour", var_name="metric", value_name="mean_value")
    fig = px.line(
        long,
        x="hour",
        y="mean_value",
        color="metric",
        title="Mean network snapshot metrics by hour (synthetic, aggregated)",
        markers=True,
    )
    fig.update_layout(template="plotly_white", xaxis=dict(dtick=1))
    return fig


def plot_confusion_matrix(
    matrix: list[list[int]] | np.ndarray,
    labels: list[str] | None = None,
) -> Figure:
    """Matplotlib confusion-matrix figure for reports/."""
    labels = labels or list(TARGET_LABELS)
    display = ConfusionMatrixDisplay(confusion_matrix=np.asarray(matrix), display_labels=labels)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix (synthetic test set)")
    fig.tight_layout()
    return fig


def plot_confusion_matrix_plotly(
    matrix: list[list[int]] | np.ndarray,
    labels: list[str] | None = None,
) -> go.Figure:
    labels = labels or list(TARGET_LABELS)
    values = np.asarray(matrix)
    fig = px.imshow(
        values,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion matrix (synthetic test set)",
        labels={"x": "Predicted", "y": "Actual"},
    )
    fig.update_layout(template="plotly_white")
    return fig
