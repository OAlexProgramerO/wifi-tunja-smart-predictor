"""Reusable Plotly figures for the dashboard and notebooks."""

from wifi_tunja_smart_predictor.visualization.plots import (
    plot_confusion_matrix,
    plot_demand_by_hour,
    plot_demand_by_zone,
    plot_demand_distribution,
    plot_network_metrics,
)

__all__ = [
    "plot_confusion_matrix",
    "plot_demand_by_hour",
    "plot_demand_by_zone",
    "plot_demand_distribution",
    "plot_network_metrics",
]
