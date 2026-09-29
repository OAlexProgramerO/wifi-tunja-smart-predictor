"""Streamlit analytics dashboard for the synthetic WiFi demand prototype."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from wifi_tunja_smart_predictor.config import (
    CATEGORICAL_ALLOWED_VALUES,
    MODEL_COMPARISON_PATH,
    MODEL_INPUT_COLUMNS,
    MODEL_METADATA_PATH,
    PROCESSED_DATASET_PATH,
    PROJECT_NAME,
    PROJECT_VERSION,
    RAW_DATASET_PATH,
    SELECTED_MODEL_METRICS_PATH,
    SYNTHETIC_DATA_DISCLAIMER,
    TARGET_COLUMN,
)
from wifi_tunja_smart_predictor.data.loader import load_raw_dataset
from wifi_tunja_smart_predictor.exceptions import DatasetNotFoundError, ModelNotFoundError
from wifi_tunja_smart_predictor.models.predict import (
    load_model,
    predict_demand,
    predict_probability,
)
from wifi_tunja_smart_predictor.visualization.plots import (
    plot_confusion_matrix_plotly,
    plot_demand_by_hour,
    plot_demand_by_weekday,
    plot_demand_by_zone,
    plot_demand_distribution,
    plot_demand_over_time,
    plot_network_metrics,
)


@st.cache_data(show_spinner=False)
def _load_frame() -> pd.DataFrame:
    path = PROCESSED_DATASET_PATH if PROCESSED_DATASET_PATH.is_file() else RAW_DATASET_PATH
    return load_raw_dataset(path)


def _json_file(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _section_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def render_overview(frame: pd.DataFrame) -> None:
    _section_header("1. Dataset overview", SYNTHETIC_DATA_DISCLAIMER)
    low = (frame[TARGET_COLUMN] == "LOW").mean()
    high = (frame[TARGET_COLUMN] == "HIGH").mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", f"{len(frame):,}")
    c2.metric("Synthetic access points", int(frame["wifi_id"].nunique()))
    c3.metric("Synthetic zones", int(frame["zone_id"].nunique()))
    c4.metric("Columns", frame.shape[1])
    c5, c6, c7 = st.columns(3)
    c5.metric("Date range start", str(pd.to_datetime(frame["timestamp"]).min().date()))
    c6.metric("Date range end", str(pd.to_datetime(frame["timestamp"]).max().date()))
    c7.metric("LOW / HIGH", f"{low:.1%} / {high:.1%}")


def render_demand(frame: pd.DataFrame) -> None:
    _section_header("2. Demand analysis", "Aggregated views of the synthetic demand_level target.")
    with st.expander("Filter observations", expanded=True):
        dates = pd.to_datetime(frame["timestamp"]).dt.date
        date_range = st.date_input(
            "Prediction date range",
            value=(dates.min(), dates.max()),
            min_value=dates.min(),
            max_value=dates.max(),
        )
        f1, f2, f3 = st.columns(3)
        zone_options = ["All"] + sorted(frame["zone_type"].dropna().unique().tolist())
        wifi_options = ["All"] + sorted(frame["wifi_id"].dropna().unique().tolist())
        weather_options = ["All"] + sorted(frame["weather_condition"].dropna().unique().tolist())
        zone_choice = f1.selectbox("Synthetic zone type", zone_options)
        wifi_choice = f2.selectbox("Synthetic access point", wifi_options)
        weather_choice = f3.selectbox("Simulated weather", weather_options)
        event_choice = st.selectbox("Simulated nearby event", ["All", "Event", "No event"])

    selected = frame.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        selected_dates = pd.to_datetime(selected["timestamp"]).dt.date
        selected = selected.loc[
            (selected_dates >= date_range[0]) & (selected_dates <= date_range[1])
        ]
    if zone_choice != "All":
        selected = selected.loc[selected["zone_type"] == zone_choice]
    if wifi_choice != "All":
        selected = selected.loc[selected["wifi_id"] == wifi_choice]
    if weather_choice != "All":
        selected = selected.loc[selected["weather_condition"] == weather_choice]
    if event_choice == "Event":
        selected = selected.loc[selected["event_nearby"] == 1]
    elif event_choice == "No event":
        selected = selected.loc[selected["event_nearby"] == 0]
    st.caption(f"Showing {len(selected):,} of {len(frame):,} historical synthetic observations.")
    if selected.empty:
        st.warning("No observations match these filters. Adjust the selections to continue.")
        return
    left, right = st.columns(2)
    with left:
        st.plotly_chart(plot_demand_distribution(selected), use_container_width=True)
        st.plotly_chart(plot_demand_by_weekday(selected), use_container_width=True)
    with right:
        st.plotly_chart(plot_demand_by_hour(selected), use_container_width=True)
        st.plotly_chart(plot_demand_by_zone(selected), use_container_width=True)
    st.plotly_chart(plot_demand_over_time(selected), use_container_width=True)


def render_map(frame: pd.DataFrame) -> None:
    _section_header(
        "3. Geographic view",
        "Synthetic coordinates only. These points do not represent real public WiFi infrastructure in Tunja.",
    )
    points = frame.groupby(["wifi_id", "zone_type", "latitude", "longitude"], as_index=False).agg(
        observations=("wifi_id", "size"),
        high_share=(TARGET_COLUMN, lambda s: (s == "HIGH").mean()),
    )
    fig = px.scatter_map(
        points,
        lat="latitude",
        lon="longitude",
        color="zone_type",
        size="observations",
        hover_name="wifi_id",
        hover_data={
            "high_share": ":.1%",
            "observations": True,
            "latitude": False,
            "longitude": False,
        },
        zoom=12,
        map_style="open-street-map",
        title="Synthetic WiFi access points (not real infrastructure)",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=520)
    st.plotly_chart(fig, use_container_width=True)


def render_network(frame: pd.DataFrame) -> None:
    _section_header(
        "4. Network analysis",
        "Means by hour of day from the last completed hour (prediction-time network snapshot).",
    )
    st.plotly_chart(plot_network_metrics(frame), use_container_width=True)


def render_performance() -> None:
    _section_header(
        "5. Model performance",
        "Metrics are computed on the synthetic temporal test holdout. They are not real-world accuracy.",
    )
    if MODEL_COMPARISON_PATH.is_file():
        comparison = pd.read_csv(MODEL_COMPARISON_PATH)
        st.dataframe(comparison, use_container_width=True)
        metric_cols = [
            c
            for c in ["accuracy", "precision", "recall", "f1", "roc_auc"]
            if c in comparison.columns
        ]
        if metric_cols:
            long = comparison.melt(
                id_vars=["model"], value_vars=metric_cols, var_name="metric", value_name="value"
            )
            fig = px.bar(
                long,
                x="model",
                y="value",
                color="metric",
                barmode="group",
                title="Test metrics by model",
            )
            fig.update_layout(template="plotly_white", yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Train models first: python scripts/train_model.py")

    selected = _json_file(SELECTED_MODEL_METRICS_PATH) or _json_file(MODEL_METADATA_PATH)
    metadata = _json_file(MODEL_METADATA_PATH)
    if selected:
        st.markdown(f"**Selected model:** `{metadata.get('selected_model', 'Unavailable')}`")
        st.caption(
            "Selection uses validation F1 for HIGH. The cards below report the later synthetic test period."
        )
        metric_columns = st.columns(5)
        for column, key, label in zip(
            metric_columns,
            ("accuracy", "precision", "recall", "f1", "roc_auc"),
            ("Accuracy", "Precision · HIGH", "Recall · HIGH", "F1 · HIGH", "ROC-AUC"),
        ):
            value = selected.get(key)
            column.metric(label, f"{value:.3f}" if isinstance(value, (float, int)) else "N/A")
        test_period = metadata.get("test_period", {})
        st.caption(
            f"Test period: {test_period.get('start', 'unknown')} to "
            f"{test_period.get('end', 'unknown')} · synthetic observations"
        )
    matrix = selected.get("confusion_matrix") or selected.get("test_metrics_selected_model")
    if isinstance(selected.get("confusion_matrix"), list):
        st.plotly_chart(
            plot_confusion_matrix_plotly(
                selected["confusion_matrix"],
                selected.get("confusion_matrix_labels") or selected.get("labels"),
            ),
            use_container_width=True,
        )
    elif matrix:
        st.write(selected.get("test_metrics_selected_model", selected))


def _default_row(frame: pd.DataFrame) -> pd.Series:
    numeric_medians = frame.select_dtypes(include="number").median(numeric_only=True)
    row = {
        col: numeric_medians.get(col, 0)
        for col in MODEL_INPUT_COLUMNS
        if col in numeric_medians.index
    }
    for col in MODEL_INPUT_COLUMNS:
        if col not in row:
            mode = frame[col].dropna().mode()
            row[col] = (
                mode.iloc[0]
                if len(mode)
                else next(iter(CATEGORICAL_ALLOWED_VALUES.get(col, {"NONE"})))
            )
    return pd.Series(row)


def render_prediction(frame: pd.DataFrame) -> None:
    _section_header(
        "6. Synthetic demand prediction",
        "Inputs are mapped to the same feature contract used at training time. This is not a causal explanation.",
    )
    defaults = _default_row(frame)
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        zone_type = c1.selectbox(
            "Zone type", sorted(CATEGORICAL_ALLOWED_VALUES["zone_type"]), index=0
        )
        time_period = c2.selectbox("Time period", list(CATEGORICAL_ALLOWED_VALUES["time_period"]))
        weather_condition = c3.selectbox(
            "Weather", list(CATEGORICAL_ALLOWED_VALUES["weather_condition"])
        )
        c1, c2, c3, c4 = st.columns(4)
        hour = c1.slider("Hour", 0, 23, int(defaults.get("hour", 12)))
        day_of_week = c2.selectbox(
            "Day of week (Mon=0)",
            options=list(range(7)),
            format_func=lambda i: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i],
            index=int(defaults.get("day_of_week", 2)),
        )
        month = c3.slider("Month", 1, 12, int(defaults.get("month", 6)))
        traffic_level = c4.selectbox(
            "Traffic level", list(CATEGORICAL_ALLOWED_VALUES["traffic_level"])
        )
        c1, c2, c3, c4 = st.columns(4)
        is_weekend = int(c1.checkbox("Weekend", value=bool(defaults.get("is_weekend", 0))))
        is_holiday = int(c2.checkbox("Holiday", value=bool(defaults.get("is_holiday", 0))))
        is_working_day = int(
            c3.checkbox("Working day", value=bool(defaults.get("is_working_day", 1)))
        )
        is_school_day = int(c4.checkbox("School day", value=bool(defaults.get("is_school_day", 1))))
        c1, c2, c3, c4 = st.columns(4)
        temperature_c = c1.number_input(
            "Temperature (°C)", value=float(defaults.get("temperature_c", 14.0))
        )
        humidity_percent = c2.slider(
            "Humidity %", 0.0, 100.0, float(defaults.get("humidity_percent", 70.0))
        )
        precipitation_mm = c3.number_input(
            "Precipitation mm", value=float(defaults.get("precipitation_mm", 0.0))
        )
        wind_speed_kmh = c4.number_input(
            "Wind km/h", value=float(defaults.get("wind_speed_kmh", 8.0))
        )
        c1, c2, c3 = st.columns(3)
        estimated_people_nearby = c1.number_input(
            "People nearby", value=float(defaults.get("estimated_people_nearby", 400))
        )
        public_transport_activity = c2.slider(
            "Public transport activity",
            0.0,
            100.0,
            float(defaults.get("public_transport_activity", 40.0)),
        )
        nearby_business_activity = c3.slider(
            "Business activity", 0.0, 100.0, float(defaults.get("nearby_business_activity", 40.0))
        )
        c1, c2 = st.columns(2)
        nearby_student_population = c1.number_input(
            "Nearby students", value=float(defaults.get("nearby_student_population", 200))
        )
        nearby_worker_population = c2.number_input(
            "Nearby workers", value=float(defaults.get("nearby_worker_population", 200))
        )
        c1, c2, c3 = st.columns(3)
        event_nearby = int(c1.checkbox("Event nearby", value=bool(defaults.get("event_nearby", 0))))
        event_type = c2.selectbox("Event type", list(CATEGORICAL_ALLOWED_VALUES["event_type"]))
        estimated_event_attendance = c3.number_input(
            "Event attendance", value=float(defaults.get("estimated_event_attendance", 0))
        )
        st.markdown("**Network snapshot (hour that just ended)**")
        n1, n2, n3, n4, n5 = st.columns(5)
        access_point_capacity = n1.number_input(
            "AP capacity", value=float(defaults.get("access_point_capacity", 100)), min_value=1.0
        )
        connected_devices = n2.number_input(
            "Connected devices", value=float(defaults.get("connected_devices", 40))
        )
        active_sessions = n3.number_input(
            "Active sessions", value=float(defaults.get("active_sessions", 30))
        )
        average_session_duration_min = n4.number_input(
            "Avg session min", value=float(defaults.get("average_session_duration_min", 20))
        )
        bandwidth_usage_mbps = n5.number_input(
            "Bandwidth Mbps", value=float(defaults.get("bandwidth_usage_mbps", 80))
        )
        n1, n2, n3, n4, n5 = st.columns(5)
        packet_loss_percent = n1.slider(
            "Packet loss %", 0.0, 100.0, float(defaults.get("packet_loss_percent", 1.0))
        )
        latency_ms = n2.number_input("Latency ms", value=float(defaults.get("latency_ms", 25)))
        signal_strength_dbm = n3.number_input(
            "Signal dBm", value=float(defaults.get("signal_strength_dbm", -60))
        )
        channel_utilization_percent = n4.slider(
            "Channel util %", 0.0, 100.0, float(defaults.get("channel_utilization_percent", 40.0))
        )
        network_uptime_percent = n5.slider(
            "Uptime %", 0.0, 100.0, float(defaults.get("network_uptime_percent", 99.0))
        )
        st.markdown("**Historical connections (strictly before prediction time)**")
        h1, h2, h3, h4 = st.columns(4)
        connections_previous_hour = h1.number_input(
            "Prev hour", value=float(defaults.get("connections_previous_hour", 50))
        )
        connections_previous_day = h2.number_input(
            "Prev day mean", value=float(defaults.get("connections_previous_day", 50))
        )
        connections_same_hour_previous_day = h3.number_input(
            "Same hour yesterday",
            value=float(defaults.get("connections_same_hour_previous_day", 50)),
        )
        connections_same_hour_previous_week = h4.number_input(
            "Same hour last week",
            value=float(defaults.get("connections_same_hour_previous_week", 50)),
        )
        h1, h2, h3 = st.columns(3)
        average_connections_last_3_hours = h1.number_input(
            "Avg last 3h", value=float(defaults.get("average_connections_last_3_hours", 50))
        )
        average_connections_last_24_hours = h2.number_input(
            "Avg last 24h", value=float(defaults.get("average_connections_last_24_hours", 50))
        )
        average_connections_last_7_days = h3.number_input(
            "Avg last 7d", value=float(defaults.get("average_connections_last_7_days", 50))
        )
        submitted = st.form_submit_button("Predict Demand")

    payload = {
        "zone_type": zone_type,
        "month": month,
        "day_of_week": day_of_week,
        "hour": hour,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "is_working_day": is_working_day,
        "is_school_day": is_school_day,
        "time_period": time_period,
        "temperature_c": temperature_c,
        "humidity_percent": humidity_percent,
        "precipitation_mm": precipitation_mm,
        "wind_speed_kmh": wind_speed_kmh,
        "weather_condition": weather_condition,
        "estimated_people_nearby": estimated_people_nearby,
        "traffic_level": traffic_level,
        "public_transport_activity": public_transport_activity,
        "nearby_business_activity": nearby_business_activity,
        "nearby_student_population": nearby_student_population,
        "nearby_worker_population": nearby_worker_population,
        "event_nearby": event_nearby,
        "event_type": event_type,
        "estimated_event_attendance": estimated_event_attendance,
        "access_point_capacity": access_point_capacity,
        "connected_devices": connected_devices,
        "active_sessions": active_sessions,
        "average_session_duration_min": average_session_duration_min,
        "bandwidth_usage_mbps": bandwidth_usage_mbps,
        "packet_loss_percent": packet_loss_percent,
        "latency_ms": latency_ms,
        "signal_strength_dbm": signal_strength_dbm,
        "channel_utilization_percent": channel_utilization_percent,
        "network_uptime_percent": network_uptime_percent,
        "connections_previous_hour": connections_previous_hour,
        "connections_previous_day": connections_previous_day,
        "connections_same_hour_previous_day": connections_same_hour_previous_day,
        "connections_same_hour_previous_week": connections_same_hour_previous_week,
        "average_connections_last_3_hours": average_connections_last_3_hours,
        "average_connections_last_24_hours": average_connections_last_24_hours,
        "average_connections_last_7_days": average_connections_last_7_days,
    }

    if submitted:
        try:
            model = load_model()
            label = predict_demand(payload, model=model)[0]
            proba = predict_probability(payload, model=model)
            st.success(f"Predicted demand: **{label}**")
            st.info(
                "HIGH indicates the model assigned the high-demand class for this simulated input; "
                "LOW indicates the low-demand class. It is a model classification, not a real-world forecast."
            )
            if proba:
                st.write(
                    "Class probabilities (from the trained classifier, not a calibrated real-world confidence):"
                )
                st.json(proba[0])
            else:
                st.caption("This estimator does not expose class probabilities.")
        except ModelNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    with st.expander("Input variables used"):
        st.json(payload)
        st.caption(
            "Engineered cyclical and ratio features are computed inside the saved pipeline. "
            "No SHAP/LIME explanations are implemented in VERSION 0.2."
        )


def main() -> None:
    st.set_page_config(page_title=PROJECT_NAME, page_icon="📶", layout="wide")
    st.sidebar.title("📶 WiFi Tunja")
    page = st.sidebar.radio(
        "Navigate",
        [
            "🏠 Overview",
            "📊 Demand Explorer",
            "🗺️ Geographic Analysis",
            "📡 Network Analysis",
            "🤖 Model Performance",
            "🔮 Predict Demand",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )
    st.sidebar.caption(f"Version {PROJECT_VERSION} · Synthetic simulation")
    st.title(page)
    st.caption("Synthetic WiFi demand simulation for Tunja")
    st.info(SYNTHETIC_DATA_DISCLAIMER)

    try:
        frame = _load_frame()
    except DatasetNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    if page == "🏠 Overview":
        st.markdown(
            "Explore a reproducible classification workflow built from simulated observations."
        )
        render_overview(frame)
        st.markdown(
            "**Workflow:** synthetic data → validation → temporal evaluation → shared model → API and dashboard"
        )
    elif page == "📊 Demand Explorer":
        render_demand(frame)
    elif page == "🗺️ Geographic Analysis":
        render_map(frame)
    elif page == "📡 Network Analysis":
        render_network(frame)
    elif page == "🤖 Model Performance":
        render_performance()
    elif page == "🔮 Predict Demand":
        st.markdown(
            "Enter a simulated snapshot and receive a LOW/HIGH classification. This is not a real-time forecast."
        )
        render_prediction(frame)
    else:
        st.markdown("""### Project purpose
This portfolio project demonstrates a maintainable end-to-end machine-learning application.

### Data and target
Every access point, coordinate, demand observation, weather value, event, network metric, and historical value is simulated. The classifier predicts `demand_level` (LOW or HIGH); the future count `connections_next_hour` is not an input.

### Architecture
Reusable logic lives in `src/wifi_tunja_smart_predictor/`. The API and dashboard use the same persisted scikit-learn pipeline. Evaluation uses ordered train, validation, and test periods.

### Limitations
Synthetic results do not establish real-world accuracy or represent municipal infrastructure. The dashboard does not receive live telemetry and predictions are not causal explanations.

### Technology
Python · pandas · scikit-learn · FastAPI · Streamlit · pytest · Ruff

[GitHub repository](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor)""")


if __name__ == "__main__":
    main()
