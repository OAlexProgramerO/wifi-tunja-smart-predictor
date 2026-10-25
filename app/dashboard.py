"""Streamlit analytics dashboard for the synthetic WiFi demand prototype."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from wifi_tunja_smart_predictor.analytics.geospatial import analyze_geospatial_demand
from wifi_tunja_smart_predictor.api_client import (
    ApiClientError,
    assistant_api_request,
    main_api_request,
    main_api_text_request,
)
from wifi_tunja_smart_predictor.assistant.schemas import AssistantContext, ChatRequest
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
from wifi_tunja_smart_predictor.scenarios.builder import ScenarioRequest
from wifi_tunja_smart_predictor.scenarios.service import ScenarioPredictionService
from wifi_tunja_smart_predictor.simulation.metrics import METRIC_REGISTRY
from wifi_tunja_smart_predictor.visualization.plots import (
    plot_confusion_matrix_plotly,
    plot_demand_by_hour,
    plot_demand_by_weekday,
    plot_demand_by_zone,
    plot_demand_distribution,
    plot_demand_over_time,
    plot_network_metrics,
)

logger = logging.getLogger(__name__)


@st.cache_data(show_spinner=False, ttl="15m", max_entries=2)
def _load_frame() -> pd.DataFrame:
    path = PROCESSED_DATASET_PATH if PROCESSED_DATASET_PATH.is_file() else RAW_DATASET_PATH
    return load_raw_dataset(path)


@st.cache_resource(max_entries=1)
def _prediction_service() -> ScenarioPredictionService:
    return ScenarioPredictionService(_load_frame())


@st.cache_data(show_spinner=False, ttl="5m", max_entries=1)
def _load_location_catalog() -> pd.DataFrame:
    return pd.DataFrame(main_api_request("/locations")["locations"])


def _geospatial_zone_summary(frame: pd.DataFrame, start_date=None, end_date=None) -> pd.DataFrame:
    """Compatibility adapter over reusable historical synthetic analytics."""
    result = analyze_geospatial_demand(frame, start_date=start_date, end_date=end_date)
    return result.by_zone.rename(
        columns={
            "mean_observed_next_hour_connections": "mean_next_hour_connections",
            "high_demand_rate": "high_demand_share",
        }
    ).sort_values("mean_next_hour_connections", ascending=False)


def _json_file(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _section_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def _render_prediction_traceability(traceability: dict | None) -> None:
    if not traceability:
        return
    with st.expander("Prediction provenance"):
        st.caption(
            f"Prediction ID: {traceability.get('prediction_id', 'unavailable')} · "
            f"Generated: {traceability.get('predicted_at_utc', 'unavailable')}"
        )
        st.caption(
            f"Application version: {traceability.get('application_version', 'unavailable')} · "
            f"Request ID: {traceability.get('request_id') or 'local prediction'}"
        )
        for label, reference in (
            ("Classifier", traceability.get("classifier")),
            ("Regressor", traceability.get("regressor")),
        ):
            if not reference:
                continue
            st.markdown(
                f"**{label}:** {reference.get('selected_model') or 'model unavailable'} "
                f"(artifact model version {reference.get('model_version') or 'unknown'})"
            )
            digest = reference.get("artifact_sha256")
            if digest:
                st.code(f"SHA-256: {digest}", language=None)


def _render_scenario_result(result: dict) -> None:
    """Present class and numeric estimate as separate model outputs."""
    scenario = result["scenario"]
    location = scenario["location"]
    classification = result["classification"]
    regression = result["regression"]
    with st.container(border=True):
        st.subheader("Demand category · classification")
        st.success(
            f"{classification['predicted_demand_level']} demand · "
            f"{scenario['scenario_mode'].replace('_', ' ').title()}"
        )
        st.caption(
            f"{location['zone_name']} · {location['access_point_id']} · "
            f"{scenario['scenario_time']}"
        )
        st.metric("Probability of HIGH demand", f"{classification['probability_high']:.1%}")
        st.caption("LOW/HIGH is a category; it is not a connection count.")
    with st.container(border=True):
        st.subheader("Next-hour connection estimate · regression")
        c1, c2 = st.columns(2)
        c1.metric("Estimated connections", f"{regression['predicted_connections_next_hour']:.1f}")
        c2.metric(
            "Estimated capacity use",
            f"{result['capacity']['predicted_capacity_utilization_pct']:.1f}%",
        )
        st.caption(
            f"{regression['interval_confidence']:.0%} split-conformal interval: "
            f"{regression['prediction_interval_lower']:.1f}–"
            f"{regression['prediction_interval_upper']:.1f} connections. "
            "This synthetic estimate is not a guarantee; coverage can change under distribution shift."
        )
    st.markdown("**Factors associated with this output** · sensitivity proxy, not causal.")
    st.dataframe(pd.DataFrame(result["explanation"]["top_factors"]), hide_index=True)
    _render_prediction_traceability(result.get("traceability"))


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
        st.plotly_chart(plot_demand_distribution(selected), width="stretch")
        st.plotly_chart(plot_demand_by_weekday(selected), width="stretch")
    with right:
        st.plotly_chart(plot_demand_by_hour(selected), width="stretch")
        st.plotly_chart(plot_demand_by_zone(selected), width="stretch")
    st.plotly_chart(plot_demand_over_time(selected), width="stretch")
    daily = selected.assign(
        observation_date=pd.to_datetime(selected["timestamp"]).dt.date,
        is_high=selected[TARGET_COLUMN].eq("HIGH").astype(float),
    )
    daily_summary = daily.groupby("observation_date", as_index=False).agg(
        mean_connections=("connections_next_hour", "mean"), high_share=("is_high", "mean")
    )
    st.plotly_chart(
        px.line(
            daily_summary,
            x="observation_date",
            y="mean_connections",
            title="Daily mean next-hour connections (synthetic)",
        ),
        width="stretch",
    )
    period_summary = (
        selected.assign(
            month_start=pd.to_datetime(selected["timestamp"]).dt.to_period("M").dt.to_timestamp(),
            is_high=selected[TARGET_COLUMN].eq("HIGH").astype(float),
        )
        .groupby("month_start", as_index=False)
        .agg(mean_connections=("connections_next_hour", "mean"), high_share=("is_high", "mean"))
    )
    st.plotly_chart(
        px.line(
            period_summary,
            x="month_start",
            y="high_share",
            title="Monthly HIGH-demand share (synthetic)",
        ),
        width="stretch",
    )
    weekend_summary = selected.groupby("is_weekend", as_index=False).agg(
        mean_connections=("connections_next_hour", "mean"), rows=("timestamp", "size")
    )
    weekend_summary["period"] = weekend_summary["is_weekend"].map({0: "Weekday", 1: "Weekend"})
    st.plotly_chart(
        px.bar(
            weekend_summary,
            x="period",
            y="mean_connections",
            title="Weekday vs weekend mean connections",
        ),
        width="stretch",
    )


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
    st.plotly_chart(fig, width="stretch")


def render_network(frame: pd.DataFrame) -> None:
    _section_header(
        "4. Network analysis",
        "Means by hour of day from the last completed hour (prediction-time network snapshot).",
    )
    st.plotly_chart(plot_network_metrics(frame), width="stretch")


def render_performance() -> None:
    _section_header(
        "5. Model performance",
        "Metrics are computed on the synthetic temporal test holdout. They are not real-world accuracy.",
    )
    if MODEL_COMPARISON_PATH.is_file():
        comparison = pd.read_csv(MODEL_COMPARISON_PATH)
        st.dataframe(comparison, width="stretch")
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
            st.plotly_chart(fig, width="stretch")
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
            width="stretch",
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
                else sorted(CATEGORICAL_ALLOWED_VALUES.get(col, {"NONE"}))[0]
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
        time_period = c2.selectbox("Time period", sorted(CATEGORICAL_ALLOWED_VALUES["time_period"]))
        weather_condition = c3.selectbox(
            "Weather", sorted(CATEGORICAL_ALLOWED_VALUES["weather_condition"])
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
            "Traffic level", sorted(CATEGORICAL_ALLOWED_VALUES["traffic_level"])
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
        event_type = c2.selectbox("Event type", sorted(CATEGORICAL_ALLOWED_VALUES["event_type"]))
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
            advanced_time = pd.Timestamp(year=2025, month=month, day=1, hour=hour).to_pydatetime()
            result = (
                _prediction_service()
                .predict(
                    ScenarioRequest(
                        datetime=advanced_time, zone=zone_type, context_overrides=payload
                    )
                )
                .to_dict()
            )
            st.session_state["last_scenario_result"] = result
            _render_scenario_result(result)
        except ModelNotFoundError:
            st.error("Model artifacts are unavailable. Initialize the runtime or train the models.")
        except Exception:
            logger.exception("Advanced prediction failed")
            st.error("Prediction could not be completed. Verify model readiness and try again.")

    with st.expander("Input variables used"):
        st.json(payload)
        st.caption(
            "These values use the same feature contract as training. Explanations are local sensitivity proxies, not causal effects."
        )


def render_scenario(frame: pd.DataFrame) -> None:
    _section_header(
        "Scenario prediction",
        "Choose where and when; context comes from deterministic synthetic historical analogs.",
    )
    try:
        catalog = _load_location_catalog()
    except ApiClientError as exc:
        st.error(str(exc))
        return
    zones = catalog[["zone_id", "zone_name"]].drop_duplicates().sort_values("zone_name")
    dates = pd.to_datetime(frame["timestamp"]).dt.date
    today = pd.Timestamp.now(tz="America/Bogota").date()
    selected_date = today if today > dates.max() else max(dates.min(), today)
    with st.form("scenario_form"):
        a, b, c = st.columns(3)
        zone_id = a.selectbox(
            "Synthetic location",
            zones["zone_id"],
            format_func=lambda z: zones.set_index("zone_id").loc[z, "zone_name"],
        )
        date_value = b.date_input("Date", value=selected_date)
        time_value = c.time_input(
            "Time",
            value=pd.Timestamp.now(tz="America/Bogota")
            .replace(minute=0, second=0, microsecond=0)
            .time(),
        )
        with st.expander("Optional context overrides"):
            use_context = st.checkbox("Override simulated traffic and weather")
            traffic = st.selectbox("Traffic", sorted(CATEGORICAL_ALLOWED_VALUES["traffic_level"]))
            weather = st.selectbox(
                "Weather", sorted(CATEGORICAL_ALLOWED_VALUES["weather_condition"])
            )
        submitted = st.form_submit_button("Estimate WiFi demand", type="primary")
    if submitted:
        overrides = {"traffic_level": traffic, "weather_condition": weather} if use_context else {}
        candidate_ap = st.session_state.get("scenario_selected_ap_id")
        if (
            candidate_ap
            and not (catalog["wifi_id"].eq(candidate_ap) & catalog["zone_id"].eq(zone_id)).any()
        ):
            candidate_ap = None
        request = ScenarioRequest(
            datetime=pd.Timestamp.combine(date_value, time_value).to_pydatetime(),
            zone_id=zone_id,
            access_point_id=candidate_ap,
            context_overrides=overrides,
        )
        try:
            st.session_state["last_scenario_result"] = main_api_request(
                "/scenario/predict",
                {
                    "datetime": request.datetime.isoformat(),
                    "location": (
                        {"access_point_id": request.access_point_id}
                        if request.access_point_id
                        else {"zone_id": request.zone_id}
                    ),
                    "advanced_context": overrides,
                },
            )
        except (ApiClientError, ModelNotFoundError, ValueError) as exc:
            st.error(str(exc))
    result = st.session_state.get("last_scenario_result")
    if result:
        _render_scenario_result(result)
        st.markdown("#### Save a reproducible capacity experiment")
        with st.form("save_capacity_experiment"):
            experiment_name = st.text_input("Experiment name", max_chars=120)
            capacity = st.number_input(
                "Simulated capacity (connections)", min_value=0.0, value=100.0
            )
            budget = st.number_input(
                "Additional allocation budget (connections)", min_value=0.0, value=0.0
            )
            save_experiment = st.form_submit_button("Run and save experiment")
        if save_experiment:
            scenario = result["scenario"]
            try:
                saved = main_api_request(
                    "/experiments/scenarios",
                    {
                        "name": experiment_name or "Scenario experiment",
                        "description": "Created from the dashboard scenario workflow.",
                        "datetime": scenario["scenario_time"],
                        "zones": [
                            {
                                "zone_id": scenario["location"]["zone_id"],
                                "capacity_connections": capacity,
                            }
                        ],
                        "resource_budget_connections": budget,
                    },
                )
                st.success(f"Experiment saved: {saved['id']}")
                st.session_state["last_experiment_id"] = saved["id"]
            except (ApiClientError, ValueError) as exc:
                st.error(str(exc))
    st.caption(SYNTHETIC_DATA_DISCLAIMER)


def _render_what_if_record(record: dict, *, spanish: bool = False) -> None:
    """Render persisted what-if outputs using the shared metric definitions."""
    result = record.get("result", {})
    what_if = result.get("what_if", {}) if isinstance(result, dict) else {}
    if not isinstance(what_if, dict):
        st.warning(
            "El registro no contiene una comparación V6."
            if spanish
            else "Saved record does not contain a V6 what-if comparison."
        )
        return
    if what_if.get("assessment_status") != "assessable":
        st.warning(
            ("La comparación no es evaluable: " if spanish else "Comparison is not assessable: ")
            + ", ".join(what_if.get("reason_codes", []))
        )
    baseline = what_if.get("baseline", {})
    baseline_aggregate = baseline.get("aggregate", {})
    language_key = "label_es" if spanish else "label_en"
    metric_label = METRIC_REGISTRY["predicted_demand_connections"][language_key]
    st.subheader(
        f"Referencia · {baseline.get('name', 'No registrado')}"
        if spanish
        else f"Baseline · {baseline.get('name', 'Not recorded')}"
    )
    first, second, third = st.columns(3)
    first.metric(metric_label, f"{baseline_aggregate.get('predicted_demand_connections', 0):.1f}")
    second.metric(
        METRIC_REGISTRY["effective_capacity_connections"][language_key],
        f"{baseline_aggregate.get('effective_capacity_connections', 0):.1f}",
    )
    third.metric(
        METRIC_REGISTRY["unallocated_budget_connections"][language_key],
        f"{baseline_aggregate.get('unallocated_budget_connections', 0):.1f}",
    )
    comparisons = what_if.get("comparisons", [])
    rows = []
    for comparison in comparisons:
        aggregate = comparison.get("aggregate_differences", {})
        rows.append(
            {
                ("Alternativa" if spanish else "Alternative"): comparison.get("alternative_name"),
                ("Evaluación" if spanish else "Assessment"): comparison.get("assessment_status"),
                ("Zonas afectadas" if spanish else "Affected zones"): comparison.get(
                    "affected_zone_count"
                ),
                (
                    "Cambio demanda (conexiones)" if spanish else "Demand Δ (connections)"
                ): aggregate.get("predicted_demand_difference_connections"),
                (
                    "Cambio capacidad (conexiones)" if spanish else "Capacity Δ (connections)"
                ): aggregate.get("effective_capacity_difference_connections"),
                (
                    "Cambio déficit (conexiones)" if spanish else "Deficit Δ (connections)"
                ): aggregate.get("overload_difference_connections"),
                (
                    "Cambio utilización (puntos porcentuales)"
                    if spanish
                    else "Utilization Δ (percentage points)"
                ): aggregate.get("utilization_proxy_difference_pct_points"),
            }
        )
    st.subheader("Escenarios alternativos" if spanish else "Alternative scenarios")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        (
            "La demanda predicha permanece fija; solo cambian los supuestos de capacidad y presupuesto. "
            "La utilización es conexiones predichas sobre capacidad simulada efectiva, no una medición de red."
            if spanish
            else "Predicted demand is held constant across alternatives; only capacity and budget assumptions vary. "
            "Utilization is predicted connections divided by effective simulated capacity, not measured network use."
        )
    )
    for comparison in comparisons:
        with st.expander(
            f"{comparison.get('alternative_name')} · detalle por zona"
            if spanish
            else f"{comparison.get('alternative_name')} · zone details"
        ):
            st.dataframe(
                pd.DataFrame(comparison.get("zones", [])), hide_index=True, width="stretch"
            )
            st.json(comparison.get("alternative", {}).get("aggregate", {}))
    if record.get("id"):
        experiment_id = record["id"]
        try:
            json_report = main_api_request(f"/experiments/{experiment_id}/report?format=json")
            markdown_report = main_api_text_request(
                f"/experiments/{experiment_id}/report?format=markdown"
            )
        except ApiClientError as exc:
            st.error(str(exc))
        else:
            st.download_button(
                "Descargar informe JSON" if spanish else "Download JSON report",
                json.dumps(json_report, indent=2, ensure_ascii=False),
                file_name=f"what-if-{experiment_id}.json",
                mime="application/json",
                key="what_if_json_download",
            )
            st.download_button(
                "Descargar informe Markdown" if spanish else "Download Markdown report",
                markdown_report,
                file_name=f"what-if-{experiment_id}.md",
                mime="text/markdown",
                key="what_if_markdown_download",
            )


def render_scenario_workbench(frame: pd.DataFrame) -> None:
    """Collect explicit capacity assumptions, execute, persist, compare, and export."""
    language = st.radio(
        "Language / Idioma", ["English", "Español"], horizontal=True, key="what_if_language"
    )
    es = language == "Español"
    _section_header(
        "Banco de escenarios" if es else "Scenario Workbench",
        (
            "Compara predicciones fijas del modelo con supuestos sintéticos de capacidad y presupuesto."
            if es
            else "Compare fixed model predictions under alternative synthetic capacity and budget assumptions."
        ),
    )
    st.caption(SYNTHETIC_DATA_DISCLAIMER)
    all_zones = sorted(frame["zone_id"].dropna().astype(str).unique().tolist())
    if not all_zones:
        st.info(
            "No hay zonas sintéticas disponibles."
            if es
            else "No synthetic zones are available for planning."
        )
        return
    selected_zones = st.multiselect(
        "Zonas sintéticas (1–12)" if es else "Synthetic zones (1–12)",
        all_zones,
        default=all_zones[: min(2, len(all_zones))],
        max_selections=12,
        key="what_if_zones",
    )
    alternative_count = st.selectbox(
        "Escenarios alternativos" if es else "Alternative scenarios",
        [1, 2, 3, 4, 5],
        index=0,
        key="what_if_alternative_count",
    )
    dates = pd.to_datetime(frame["timestamp"], errors="coerce").dropna()
    if dates.empty:
        st.info(
            "No hay observaciones históricas con fecha."
            if es
            else "No timestamped historical observations are available."
        )
        return
    start, end = dates.min().date(), dates.max().date()
    zone_capacity = (
        frame.groupby("zone_id")["access_point_capacity"].median().to_dict()
        if "access_point_capacity" in frame
        else {}
    )
    with st.form("what_if_simulation_form"):
        name = st.text_input(
            "Nombre del experimento" if es else "Experiment name",
            value="Comparación hipotética de capacidad" if es else "Capacity what-if comparison",
            max_chars=120,
        )
        description = st.text_input("Descripción" if es else "Description", max_chars=500)
        when_date = st.date_input(
            "Fecha del escenario" if es else "Scenario date",
            value=end,
            min_value=start,
            max_value=end,
        )
        when_time = st.time_input(
            "Hora del escenario" if es else "Scenario hour",
            value=pd.Timestamp("2025-01-01 12:00").time(),
        )
        st.markdown("#### Supuestos de referencia" if es else "#### Baseline assumptions")
        baseline_name = st.text_input(
            "Etiqueta de referencia" if es else "Baseline label",
            value="Referencia" if es else "Baseline",
            max_chars=80,
        )
        baseline_budget = st.number_input(
            (
                "Presupuesto adicional de referencia (conexiones)"
                if es
                else "Baseline additional budget (connections)"
            ),
            min_value=0.0,
            max_value=100_000.0,
            value=0.0,
        )
        baseline_capacities = {}
        if selected_zones:
            columns = st.columns(2)
            for index, zone_id in enumerate(selected_zones):
                baseline_capacities[zone_id] = columns[index % 2].number_input(
                    (
                        f"Capacidad de referencia · {zone_id}"
                        if es
                        else f"Baseline capacity · {zone_id}"
                    ),
                    min_value=0.0,
                    max_value=100_000.0,
                    value=min(100_000.0, max(0.0, float(zone_capacity.get(zone_id, 100.0)))),
                    key=f"what_if_baseline_capacity_{zone_id}",
                )
        alternatives = []
        for alt_index in range(alternative_count):
            with st.expander(
                f"Alternativa {alt_index + 1}" if es else f"Alternative {alt_index + 1}",
                expanded=alt_index == 0,
            ):
                alt_name = st.text_input(
                    (
                        f"Etiqueta de alternativa {alt_index + 1}"
                        if es
                        else f"Alternative {alt_index + 1} label"
                    ),
                    value=f"Alternativa {alt_index + 1}" if es else f"Alternative {alt_index + 1}",
                    max_chars=80,
                    key=f"what_if_alt_name_{alt_index}",
                )
                alt_description = st.text_input(
                    (
                        f"Descripción de alternativa {alt_index + 1}"
                        if es
                        else f"Alternative {alt_index + 1} description"
                    ),
                    max_chars=500,
                    key=f"what_if_alt_description_{alt_index}",
                )
                alt_budget = st.number_input(
                    (
                        f"Presupuesto adicional alternativa {alt_index + 1} (conexiones)"
                        if es
                        else f"Alternative {alt_index + 1} additional budget (connections)"
                    ),
                    min_value=0.0,
                    max_value=100_000.0,
                    value=0.0,
                    key=f"what_if_alt_budget_{alt_index}",
                )
                capacities = {}
                if selected_zones:
                    columns = st.columns(2)
                    for zone_index, zone_id in enumerate(selected_zones):
                        capacities[zone_id] = columns[zone_index % 2].number_input(
                            (
                                f"Capacidad alternativa {alt_index + 1} · {zone_id}"
                                if es
                                else f"Alternative {alt_index + 1} capacity · {zone_id}"
                            ),
                            min_value=0.0,
                            max_value=100_000.0,
                            value=min(
                                100_000.0, max(0.0, float(zone_capacity.get(zone_id, 100.0)))
                            ),
                            key=f"what_if_alt_capacity_{alt_index}_{zone_id}",
                        )
                alternatives.append((alt_name, alt_description, alt_budget, capacities))
        submitted = st.form_submit_button(
            "Ejecutar y guardar comparación" if es else "Run and save comparison",
            type="primary",
            disabled=not selected_zones,
        )
    if submitted:
        payload = {
            "name": name,
            "description": description,
            "datetime": pd.Timestamp.combine(when_date, when_time).isoformat(),
            "baseline": {
                "name": baseline_name,
                "description": "User-defined baseline assumptions.",
                "zones": [
                    {"zone_id": zone_id, "capacity_connections": capacity}
                    for zone_id, capacity in baseline_capacities.items()
                ],
                "resource_budget_connections": baseline_budget,
            },
            "alternatives": [
                {
                    "name": alt_name,
                    "description": alt_description,
                    "zones": [
                        {"zone_id": zone_id, "capacity_connections": capacity}
                        for zone_id, capacity in capacities.items()
                    ],
                    "resource_budget_connections": budget,
                }
                for alt_name, alt_description, budget, capacities in alternatives
            ],
        }
        try:
            saved = main_api_request("/experiments/what-if", payload)
            st.session_state["what_if_experiment_id"] = saved["id"]
            st.session_state["what_if_experiment_record"] = saved
            st.success(
                f"Experimento reproducible guardado: {saved['id']}."
                if es
                else f"Saved reproducible experiment {saved['id']}."
            )
        except ApiClientError as exc:
            st.error(str(exc))
    saved_record = st.session_state.get("what_if_experiment_record")
    experiment_id = st.session_state.get("what_if_experiment_id")
    if experiment_id:
        try:
            saved_record = main_api_request(f"/experiments/{experiment_id}")
            st.session_state["what_if_experiment_record"] = saved_record
        except ApiClientError as exc:
            st.error(str(exc))
    if saved_record:
        _render_what_if_record(saved_record, spanish=es)


def render_geography(frame: pd.DataFrame) -> None:
    _section_header(
        "Geospatial intelligence",
        "Historical synthetic observations by place and time; these are not model predictions.",
    )
    language = st.radio(
        "Language / Idioma", ["English", "Español"], horizontal=True, key="geo_language"
    )
    spanish = language == "Español"
    dates = pd.to_datetime(frame["timestamp"], errors="coerce").dropna()
    if dates.empty:
        st.info(
            "No timestamped synthetic observations are available."
            if not spanish
            else "No hay observaciones sintéticas con fecha."
        )
        return
    start_default, end_default = dates.min().date(), dates.max().date()
    first, second = st.columns(2)
    date_range = first.date_input(
        "Observation dates" if not spanish else "Fechas de observación",
        value=(start_default, end_default),
        min_value=start_default,
        max_value=end_default,
        key="geo_date_range",
    )
    zone_ids = ["All", *sorted(frame["zone_id"].dropna().astype(str).unique())]
    zone_choice = second.selectbox(
        "Synthetic zone" if not spanish else "Zona sintética",
        zone_ids,
        format_func=lambda value: (
            ("All zones" if not spanish else "Todas las zonas") if value == "All" else value
        ),
        key="geo_zone_filter",
    )
    c1, c2, c3 = st.columns(3)
    hour_choice = c1.selectbox(
        "Hour" if not spanish else "Hora", ["All", *range(24)], key="geo_hour_filter"
    )
    weekdays_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekdays_es = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    weekday_labels = weekdays_es if spanish else weekdays_en
    weekday_choice = c2.selectbox(
        "Weekday" if not spanish else "Día de la semana",
        ["All", *weekday_labels],
        key="geo_weekday_filter",
    )
    demand_choice = c3.selectbox(
        "Demand category" if not spanish else "Categoría de demanda",
        ["All", "LOW", "HIGH"],
        key="geo_demand_filter",
    )
    selected_dates = (
        date_range
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2
        else (start_default, end_default)
    )
    try:
        analysis = analyze_geospatial_demand(
            frame,
            start_date=selected_dates[0],
            end_date=selected_dates[1],
            zone_ids=None if zone_choice == "All" else [zone_choice],
            hour=None if hour_choice == "All" else int(hour_choice),
            weekday=None if weekday_choice == "All" else weekday_labels.index(weekday_choice),
            demand_level=None if demand_choice == "All" else demand_choice,
        )
    except ValueError as exc:
        st.error(str(exc))
        return
    st.caption(
        f"{len(analysis.observations):,} {('historical synthetic observations' if not spanish else 'observaciones históricas sintéticas')} · "
        f"{selected_dates[0]}–{selected_dates[1]} · "
        f"{('all hours' if hour_choice == 'All' else f'hour {hour_choice}')} · "
        f"{('all weekdays' if weekday_choice == 'All' else weekday_choice)}"
    )
    if analysis.observations.empty:
        st.info(
            "No observations match these filters. Adjust the selections."
            if not spanish
            else "Ninguna observación coincide con los filtros. Cambia las selecciones."
        )
        return
    mean_connections = analysis.observations["connections_next_hour"].mean()
    high_rate = analysis.observations["demand_level"].dropna().eq("HIGH").mean()
    m1, m2, m3 = st.columns(3)
    m1.metric("Observations" if not spanish else "Observaciones", f"{len(analysis.observations):,}")
    m2.metric(
        (
            "Mean observed next-hour connections"
            if not spanish
            else "Media observada de conexiones próxima hora"
        ),
        f"{mean_connections:.1f}",
    )
    m3.metric(
        "HIGH observation rate" if not spanish else "Proporción de observaciones HIGH",
        f"{high_rate:.1%}",
    )
    st.caption(
        "The connection value is an observed synthetic target, not a prediction."
        if not spanish
        else "Las conexiones son un objetivo sintético observado, no una predicción."
    )

    if analysis.access_points.empty:
        st.warning(
            "No valid synthetic coordinates remain after filtering."
            if not spanish
            else "No quedan coordenadas sintéticas válidas tras filtrar."
        )
    else:
        points = analysis.access_points
        global_max = pd.to_numeric(frame["connections_next_hour"], errors="coerce").max()
        fig = px.scatter_geo(
            points,
            lat="latitude",
            lon="longitude",
            color="mean_observed_next_hour_connections",
            size="observations",
            hover_name="wifi_id",
            hover_data={
                "zone_id": True,
                "observations": True,
                "mean_observed_next_hour_connections": ":.1f",
                "high_demand_rate": ":.1%",
                "latitude": False,
                "longitude": False,
            },
            color_continuous_scale="Viridis",
            range_color=(0, max(1.0, float(global_max) if pd.notna(global_max) else 1.0)),
            labels={
                "mean_observed_next_hour_connections": "Mean observed connections",
                "observations": "Observation count",
                "high_demand_rate": "HIGH observation rate",
                "wifi_id": "Synthetic AP",
            },
            title=(
                "Synthetic access-point observed demand"
                if not spanish
                else "Demanda observada en puntos sintéticos"
            ),
        )
        fig.update_geos(scope="south america", fitbounds="locations", showcountries=True)
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 45, "b": 0}, coloraxis_colorbar_title="Avg connections"
        )
        st.plotly_chart(fig, width="stretch", key="geospatial_demand_map")
        selected_ap = st.selectbox(
            "Access-point detail" if not spanish else "Detalle del punto de acceso",
            points["wifi_id"].tolist(),
            key="geo_selected_ap",
        )
        detail = points.loc[points["wifi_id"] == selected_ap].iloc[0]
        st.write(
            f"**{selected_ap} · {detail.get('zone_id', 'unknown')}** — "
            f"{detail['mean_observed_next_hour_connections']:.1f} mean observed connections, "
            f"{int(detail['observations'])} observations, "
            f"{detail['high_demand_rate']:.1%} HIGH rate."
        )

    left, right = st.columns(2)
    hourly = px.line(
        analysis.by_hour,
        x="hour",
        y="mean_observed_next_hour_connections",
        markers=True,
        title="Mean observed connections by hour" if not spanish else "Media observada por hora",
        labels={"hour": "Hour of day", "mean_observed_next_hour_connections": "Mean connections"},
    )
    hourly.update_layout(xaxis={"dtick": 1}, yaxis_title="Connections / hour")
    left.plotly_chart(hourly, width="stretch")
    weekdays = analysis.by_weekday.copy()
    weekdays["weekday_name"] = weekdays["weekday"].map(dict(enumerate(weekday_labels)))
    weekday_fig = px.bar(
        weekdays,
        x="weekday_name",
        y="mean_observed_next_hour_connections",
        title=(
            "Mean observed connections by weekday"
            if not spanish
            else "Media observada por día de semana"
        ),
        labels={
            "weekday_name": "Weekday" if not spanish else "Día",
            "mean_observed_next_hour_connections": "Mean connections",
        },
    )
    right.plotly_chart(weekday_fig, width="stretch")
    if not analysis.zone_by_hour.empty:
        heatmap_rows = analysis.zone_by_hour.reset_index()
        zone_display = "zone_name" if "zone_name" in heatmap_rows.columns else "zone_id"
        hours = sorted(column for column in heatmap_rows.columns if isinstance(column, int))
        heatmap = px.imshow(
            heatmap_rows.set_index(zone_display)[hours],
            aspect="auto",
            color_continuous_scale="Blues",
            title=(
                "Mean observed connections by zone and hour"
                if not spanish
                else "Media observada por zona y hora"
            ),
            labels={"x": "Hour of day", "y": "Synthetic zone", "color": "Connections"},
        )
        st.plotly_chart(heatmap, width="stretch")
    st.subheader("Zone demand and HIGH rates" if not spanish else "Demanda y tasas HIGH por zona")
    st.dataframe(analysis.by_zone, hide_index=True, width="stretch")
    if analysis.peak_hour is not None:
        peak_day = (
            weekday_labels[analysis.peak_weekday] if analysis.peak_weekday is not None else "—"
        )
        st.info(
            f"Peak mean observed hour: {analysis.peak_hour}:00 · peak weekday: {peak_day}."
            if not spanish
            else f"Hora pico de media observada: {analysis.peak_hour}:00 · día pico: {peak_day}."
        )
    st.caption(SYNTHETIC_DATA_DISCLAIMER)


def render_assistant() -> None:
    _section_header("AI Assistant", "Deterministic, tool-grounded answers. No LLM key is required.")
    st.session_state.setdefault("assistant_session_id", None)
    st.session_state.setdefault("assistant_messages", [])
    suggestions = [
        "What is the demand downtown?",
        "Will demand be high in the north at 6 PM?",
        "How many connections are expected in the south?",
        "¿Cuál es la demanda en el centro?",
        "How many access points are in the dataset?",
        "What are the model limitations?",
        "Explain this dashboard section.",
        "What does this prediction mean?",
        "Why is demand high?",
        "Explain the capacity indicator.",
        "What does this graph show?",
        "Explain model performance.",
        "Explícame esta sección.",
        "¿Qué significa esta predicción?",
        "¿Por qué la demanda es alta?",
        "¿Qué muestra este gráfico?",
        "How is demand historically?",
        "When is demand usually highest?",
        "Compare two zones.",
        "Which zone has higher demand?",
        "What day has the highest demand?",
        "¿Cómo es históricamente la demanda?",
        "¿Cuándo suele ser más alta?",
        "Compara dos zonas.",
        "¿Qué zona tiene mayor demanda?",
        "¿Qué día tiene mayor demanda?",
    ]
    for start in range(0, len(suggestions), 5):
        row = suggestions[start : start + 5]
        columns = st.columns(len(row))
        for column, suggestion in zip(columns, row):
            if column.button(suggestion, key=f"suggest_{suggestion}"):
                st.session_state["assistant_pending"] = suggestion
    for message in st.session_state["assistant_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.caption("Sources: " + ", ".join(message["sources"]))
    prompt = st.chat_input("Ask about demand, a location, the dataset, or model")
    prompt = prompt or st.session_state.pop("assistant_pending", None)
    if prompt:
        st.session_state["assistant_messages"].append({"role": "user", "content": prompt})
        section = st.session_state.get("assistant_dashboard_section")
        scenario = st.session_state.get("last_scenario_result")
        context_values = {"dashboard_section": section} if section else {}
        if scenario:
            loc = scenario["scenario"]["location"]
            context_values.update(
                zone_id=loc["zone_id"],
                access_point_id=loc["access_point_id"],
                datetime=pd.Timestamp(scenario["scenario"]["scenario_time"]).to_pydatetime(),
                scenario_result=scenario,
            )
        context = AssistantContext(**context_values) if context_values else None
        chat_request = ChatRequest(
            message=prompt, session_id=st.session_state["assistant_session_id"], context=context
        )
        try:
            answer = assistant_api_request(
                "/chat", chat_request.model_dump(mode="json", exclude_none=True)
            )
        except ApiClientError as exc:
            st.error(str(exc))
            return
        st.session_state["assistant_session_id"] = answer.session_id
        st.session_state["assistant_messages"].append(
            {"role": "assistant", "content": answer.answer, "sources": answer.sources}
        )
        st.rerun()


def render_performance_v3() -> None:
    render_performance()
    regression_comparison_path = (
        Path(__file__).resolve().parents[1]
        / "reports"
        / "metrics"
        / "regression_model_comparison.csv"
    )
    if regression_comparison_path.is_file():
        _section_header(
            "Regressor candidates",
            "Candidate selection uses validation MAE; test metrics are reported separately.",
        )
        comparison = pd.read_csv(regression_comparison_path)
        st.dataframe(comparison, width="stretch", hide_index=True)
        if "validation_mae" in comparison and "model" in comparison:
            st.plotly_chart(
                px.bar(
                    comparison,
                    x="model",
                    y="validation_mae",
                    title="Validation MAE by regressor (lower is better)",
                ),
                width="stretch",
            )
    regression = _json_file(
        Path(__file__).resolve().parents[1] / "models" / "regression_metadata.json"
    )
    _section_header("Regression and uncertainty")
    metrics = regression.get("test_metrics", {})
    for column, key, label in zip(
        st.columns(4),
        ("mae", "rmse", "r2", "test_interval_coverage"),
        ("MAE", "RMSE", "R²", "Interval coverage"),
    ):
        value = regression.get(key) if key == "test_interval_coverage" else metrics.get(key)
        column.metric(
            label, f"{value:.3f}" if isinstance(value, (float, int)) else "Train models to populate"
        )
    if regression:
        st.caption(
            f"{regression.get('interval_confidence', .9):.0%} nominal interval; calibration radius {regression.get('calibration_radius', 'unavailable')} connections. Test coverage may differ under distribution shift."
        )
    from wifi_tunja_smart_predictor.models.diagnostics import load_valid_diagnostics

    _section_header("Prediction quality diagnostics")
    try:
        report = load_valid_diagnostics()
    except (OSError, ValueError):
        st.warning("Diagnostics are unavailable or stale. Run `python scripts/evaluate_model.py`.")
        return
    st.caption(
        f"Report schema {report['schema_version']} · generated {report['generated_at_utc']} · "
        f"test rows {report['test']['rows']} · all observations are synthetic."
    )
    st.dataframe(
        pd.DataFrame.from_dict(report["test"]["classification"]["class_metrics"], orient="index")
        .rename_axis("class")
        .reset_index(),
        width="stretch",
        hide_index=True,
    )
    classification = report["test"]["classification"]
    st.plotly_chart(
        plot_confusion_matrix_plotly(
            classification["confusion_matrix"], classification["confusion_labels"]
        ),
        width="stretch",
    )
    st.markdown("**Chronological expanding-window evaluation**")
    st.dataframe(pd.DataFrame(report["rolling_windows"]), width="stretch", hide_index=True)
    regression_test = report["test"]["regression"]
    columns = st.columns(4)
    for column, key, label in zip(
        columns,
        ("mae", "rmse", "r2", "mean_prediction_bias"),
        ("MAE", "RMSE", "R²", "Mean prediction bias"),
    ):
        value = regression_test.get(key)
        column.metric(label, f"{value:.3f}" if isinstance(value, (float, int)) else "N/A")
    for warning in report["warnings"]:
        st.caption(f"⚠️ {warning}")


def _render_page(title: str, kind: str) -> None:
    section_names = {
        "Overview": "Overview",
        "Scenario": "Live Scenario",
        "WhatIf": "Scenario Workbench",
        "Demand": "Demand Explorer",
        "Geography": "Geographic Analysis",
        "Network": "Network Analysis",
        "Performance": "Model Performance",
        "Assistant": "AI Assistant",
        "Advanced": "Advanced Prediction",
        "About": "About",
        "Experiments": "Experiment History",
    }
    if kind != "Assistant":
        st.session_state["assistant_dashboard_section"] = section_names.get(kind)
    st.title(title)
    st.caption(
        f"WiFi Tunja Smart Predictor · v{PROJECT_VERSION} · synthetic decision-support prototype"
    )
    if kind == "Overview":
        frame = _load_frame()
        render_overview(frame)
        metadata = _json_file(MODEL_METADATA_PATH)
        regression = _json_file(
            Path(__file__).resolve().parents[1] / "models" / "regression_metadata.json"
        )
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Classifier",
            (
                "Ready"
                if (
                    Path(__file__).resolve().parents[1] / "models" / "wifi_demand_classifier.joblib"
                ).is_file()
                else "Train required"
            ),
        )
        m2.metric(
            "Regressor",
            (
                "Ready"
                if (
                    Path(__file__).resolve().parents[1] / "models" / "wifi_demand_regressor.joblib"
                ).is_file()
                else "Train required"
            ),
        )
        m3.metric(
            "Selected models",
            f"{metadata.get('selected_model', '—')} / {regression.get('selected_model', '—')}",
        )
        st.markdown(
            "Start with a location and time in Scenario Prediction to get a class, connection estimate, interval, capacity proxy, and associated factors."
        )
    elif kind == "Scenario":
        render_scenario(_load_frame())
    elif kind == "WhatIf":
        render_scenario_workbench(_load_frame())
    elif kind == "Demand":
        render_demand(_load_frame())
    elif kind == "Geography":
        render_geography(_load_frame())
    elif kind == "Network":
        render_network(_load_frame())
    elif kind == "Performance":
        render_performance_v3()
    elif kind == "Assistant":
        render_assistant()
    elif kind == "Experiments":
        st.caption(
            "Saved synthetic capacity-planning experiments. Values are estimates in connection counts."
        )
        try:
            page_size = 25
            page_number = st.number_input(
                "History page",
                min_value=1,
                value=1,
                step=1,
                key="experiment_history_page",
            )
            offset = (page_number - 1) * page_size
            page = main_api_request(f"/experiments?limit={page_size}&offset={offset}")
            experiments = page["items"]
            warnings = page.get("integrity_warnings", [])
            for warning in warnings:
                st.warning(
                    f"Record {warning.get('experiment_id', 'unknown')} omitted: {warning.get('reason_code', 'INTEGRITY_WARNING')}"
                )
            if not experiments:
                if offset:
                    st.info(
                        "No experiments on this page. Choose an earlier page to view saved history."
                    )
                else:
                    st.info("No saved experiments yet. Run and save one from Scenario Prediction.")
            else:
                st.caption(
                    f"Showing records {offset + 1}–{offset + len(experiments)} · "
                    "synthetic scenario history"
                )
                integrity_labels = {
                    "verified": "Verified · assessment possible",
                    "incomplete": "Incomplete · assessment not possible",
                    "invalid": "Invalid · assessment not possible",
                }
                for item in experiments:
                    integrity = item.get("integrity", {})
                    status = integrity.get("status", "incomplete")
                    if status != "verified":
                        reasons = ", ".join(integrity.get("reason_codes", []))
                        st.warning(
                            f"{item.get('name', item['id'])}: {integrity_labels.get(status, status)}. "
                            f"Reason codes: {reasons or 'not recorded'}. / "
                            f"{item.get('name', item['id'])}: Incompleto o inválido; evaluación no posible. "
                            f"Códigos: {reasons or 'sin registro'}."
                        )
                selected_id = st.selectbox(
                    "Experiment",
                    [item["id"] for item in experiments],
                    format_func=lambda value: next(
                        f"{item['name']} · {item['created_at_utc']}"
                        for item in experiments
                        if item["id"] == value
                    ),
                )
                record = main_api_request(f"/experiments/{selected_id}")
                st.json(record["result"]["capacity_plan"])
                compare_ids = st.multiselect(
                    "Compare compatible experiments",
                    [
                        item["id"]
                        for item in experiments
                        if item["id"] != selected_id
                        and item.get("integrity", {}).get("status") == "verified"
                        and next(
                            candidate.get("integrity", {}).get("status")
                            for candidate in experiments
                            if candidate["id"] == selected_id
                        )
                        == "verified"
                    ],
                    format_func=lambda value: next(
                        item["name"] for item in experiments if item["id"] == value
                    ),
                )
                if st.button("Compare with selected experiment", disabled=not compare_ids):
                    try:
                        comparison = main_api_request(
                            "/experiments/compare", {"experiment_ids": [selected_id, *compare_ids]}
                        )
                        for item in comparison.get("comparisons", []):
                            if item.get("comparison_status") == "no_shared_zones":
                                st.info(
                                    item.get("interpretation_detail", "No shared zones to compare.")
                                )
                        st.json(comparison)
                    except ApiClientError as exc:
                        st.error(str(exc))
                report = main_api_request(f"/experiments/{selected_id}/report?format=json")
                st.download_button(
                    "Download JSON report",
                    json.dumps(report, indent=2, ensure_ascii=False),
                    file_name=f"experiment-{selected_id}.json",
                    mime="application/json",
                )
        except ApiClientError as exc:
            st.error(str(exc))
    elif kind == "Advanced":
        render_prediction(_load_frame())
    else:
        st.markdown(
            "This prototype uses synthetic access points, coordinates, contexts, and outcomes. It has no live telemetry or real-time provider. Classification predicts `demand_level`; regression estimates `connections_next_hour`. The conformal interval is calibrated on validation residuals. Feature sensitivity does not establish causation. See `docs/` for methodology, API, privacy, and limitations."
        )


def main() -> None:
    st.set_page_config(page_title=PROJECT_NAME, page_icon="📶", layout="wide")
    pages = [
        st.Page(
            lambda: _render_page("Overview", "Overview"),
            title="Overview",
            icon="🏠",
            url_path="overview",
            default=True,
        ),
        st.Page(
            lambda: _render_page("Scenario Prediction", "Scenario"),
            title="Live Scenario",
            icon="📍",
            url_path="scenario",
        ),
        st.Page(
            lambda: _render_page("Demand Explorer", "Demand"),
            title="Demand Explorer",
            icon="📊",
            url_path="demand",
        ),
        st.Page(
            lambda: _render_page("Geospatial Intelligence", "Geography"),
            title="Geospatial Intelligence",
            icon="🗺️",
            url_path="geography",
        ),
        st.Page(
            lambda: _render_page("Scenario Workbench", "WhatIf"),
            title="Scenario Workbench",
            icon="🧪",
            url_path="workbench",
        ),
        st.Page(
            lambda: _render_page("Network Analysis", "Network"),
            title="Network Analysis",
            icon="📡",
            url_path="network",
        ),
        st.Page(
            lambda: _render_page("Model Performance", "Performance"),
            title="Model Performance",
            icon="📈",
            url_path="performance",
        ),
        st.Page(
            lambda: _render_page("AI Assistant", "Assistant"),
            title="AI Assistant",
            icon="💬",
            url_path="assistant",
        ),
        st.Page(
            lambda: _render_page("Advanced Prediction", "Advanced"),
            title="Advanced Prediction",
            icon="🧰",
            url_path="advanced",
        ),
        st.Page(lambda: _render_page("About", "About"), title="About", icon="ℹ️", url_path="about"),
        st.Page(
            lambda: _render_page("Experiment History", "Experiments"),
            title="Experiment History",
            icon="🧪",
            url_path="experiments",
        ),
    ]
    navigation = st.navigation(pages, position="sidebar")
    st.sidebar.caption(f"v{PROJECT_VERSION} · simulated data")
    try:
        navigation.run()
    except DatasetNotFoundError as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
