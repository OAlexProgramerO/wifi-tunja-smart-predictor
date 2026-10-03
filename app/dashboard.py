"""Streamlit analytics dashboard for the synthetic WiFi demand prototype."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from wifi_tunja_smart_predictor.assistant.schemas import AssistantContext, ChatRequest
from wifi_tunja_smart_predictor.assistant.service import AssistantService
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
from wifi_tunja_smart_predictor.geospatial.providers import PlotlyOpenStreetMapProvider
from wifi_tunja_smart_predictor.scenarios.builder import ScenarioRequest
from wifi_tunja_smart_predictor.scenarios.service import ScenarioPredictionService
from wifi_tunja_smart_predictor.visualization.plots import (
    plot_confusion_matrix_plotly,
    plot_demand_by_hour,
    plot_demand_by_weekday,
    plot_demand_by_zone,
    plot_demand_distribution,
    plot_demand_over_time,
    plot_network_metrics,
)


@st.cache_data(show_spinner=False, ttl="15m", max_entries=2)
def _load_frame() -> pd.DataFrame:
    path = PROCESSED_DATASET_PATH if PROCESSED_DATASET_PATH.is_file() else RAW_DATASET_PATH
    return load_raw_dataset(path)


@st.cache_resource(max_entries=1)
def _prediction_service() -> ScenarioPredictionService:
    return ScenarioPredictionService(_load_frame())


@st.cache_resource(max_entries=1)
def _assistant_service() -> AssistantService:
    return AssistantService(_load_frame(), _prediction_service())


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
            st.success(
                f"Predicted demand: **{result['classification']['predicted_demand_level']}**"
            )
            st.metric(
                "Expected next-hour connections",
                f"{result['regression']['predicted_connections_next_hour']:.1f}",
            )
            st.caption(
                f"{result['regression']['interval_confidence']:.0%} calibrated interval: {result['regression']['prediction_interval_lower']:.1f}–{result['regression']['prediction_interval_upper']:.1f}. Synthetic estimate; feature sensitivity is not causal."
            )
            st.dataframe(
                pd.DataFrame(result["explanation"]["top_factors"]), width="stretch", hide_index=True
            )
        except ModelNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

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
    service = _prediction_service()
    catalog = service.builder.resolver.catalog
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
            st.session_state["last_scenario_result"] = service.predict(request).to_dict()
        except (ModelNotFoundError, ValueError) as exc:
            st.error(str(exc))
    result = st.session_state.get("last_scenario_result")
    if result:
        scenario = result["scenario"]
        loc = scenario["location"]
        st.success(
            f"{result['classification']['predicted_demand_level']} demand · {scenario['scenario_mode'].replace('_', ' ').title()}"
        )
        st.caption(f"{loc['zone_name']} · {loc['access_point_id']} · {scenario['scenario_time']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("HIGH probability", f"{result['classification']['probability_high']:.1%}")
        c2.metric(
            "Expected connections next hour",
            f"{result['regression']['predicted_connections_next_hour']:.1f}",
        )
        c3.metric(
            "Estimated capacity use",
            f"{result['capacity']['predicted_capacity_utilization_pct']:.1f}%",
        )
        interval = result["regression"]
        st.caption(
            f"{interval['interval_confidence']:.0%} split-conformal interval: {interval['prediction_interval_lower']:.1f}–{interval['prediction_interval_upper']:.1f} connections."
        )
        st.markdown("**Factors associated with this output** · sensitivity proxy, not causal.")
        st.dataframe(
            pd.DataFrame(result["explanation"]["top_factors"]), width="stretch", hide_index=True
        )
    st.caption(SYNTHETIC_DATA_DISCLAIMER)


def render_geography(frame: pd.DataFrame) -> None:
    _section_header(
        "Geographic analysis", "Select a simulated access point marker for the scenario builder."
    )
    catalog = _prediction_service().builder.resolver.catalog
    fig = PlotlyOpenStreetMapProvider().build_map(catalog)
    selection = st.plotly_chart(
        fig, width="stretch", on_select="rerun", selection_mode="points", key="geography_map"
    )
    points = selection.get("selection", {}).get("points", []) if selection else []
    if points:
        custom_data = points[0].get("customdata") or []
        ap_id = custom_data[0] if custom_data else None
        if ap_id in set(catalog["wifi_id"]):
            st.session_state["scenario_selected_ap_id"] = str(ap_id)
    ap_id = st.session_state.get("scenario_selected_ap_id")
    if ap_id:
        st.write(catalog.loc[catalog["wifi_id"] == ap_id])
        st.info(f"Selected {ap_id}. Use it in Scenario Prediction.")
    st.caption(
        "All coordinates are simulated. The map uses OpenStreetMap tiles without a paid key."
    )


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
    ]
    columns = st.columns(len(suggestions))
    for column, suggestion in zip(columns, suggestions):
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
        answer = _assistant_service().handle(
            ChatRequest(
                message=prompt, session_id=st.session_state["assistant_session_id"], context=context
            )
        )
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


def _render_page(title: str, kind: str) -> None:
    section_names = {
        "Overview": "Overview",
        "Scenario": "Live Scenario",
        "Demand": "Demand Explorer",
        "Geography": "Geographic Analysis",
        "Network": "Network Analysis",
        "Performance": "Model Performance",
        "Assistant": "AI Assistant",
        "Advanced": "Advanced Prediction",
        "About": "About",
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
            lambda: _render_page("Geographic Analysis", "Geography"),
            title="Geographic Analysis",
            icon="🗺️",
            url_path="geography",
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
    ]
    navigation = st.navigation(pages, position="sidebar")
    st.sidebar.caption(f"v{PROJECT_VERSION} · simulated data")
    try:
        navigation.run()
    except DatasetNotFoundError as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
