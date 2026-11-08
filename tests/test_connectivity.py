from __future__ import annotations

import csv
import io
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest
import requests
from fastapi.testclient import TestClient

from api.main import app
from wifi_tunja_smart_predictor.assistant.schemas import ChatRequest
from wifi_tunja_smart_predictor.assistant.service import AssistantService
from wifi_tunja_smart_predictor.connectivity.analysis import (
    analyze_tunja,
    available_tunja_periods,
    compare_tunja_periods,
)
from wifi_tunja_smart_predictor.connectivity.exports import (
    build_connectivity_csv,
    build_connectivity_json,
    build_export_bundle,
)
from wifi_tunja_smart_predictor.connectivity.mintic import (
    API_URL,
    REQUIRED_FIELDS,
    SourceUnavailableError,
    _fetch_records,
    _metadata,
    _request_json,
    get_refresh_history,
    latest_refresh_status,
    load_snapshot,
    refresh_official_data,
)


def record(**updates):
    base = {
        "anno": "2025",
        "trimestre": "1",
        "proveedor": "Proveedor A",
        "codigo_dane_departamento": "15",
        "departamento": "BOYACÁ",
        "codigo_dane_municipio": "15001",
        "municipio": "TUNJA",
        "segmento": "RESIDENCIAL",
        "tecnologia": "FTTH",
        "velocidad_bajada": "100",
        "velocidad_subida": "50",
        "accesos_fijos_internet": "12",
    }
    base.update(updates)
    assert REQUIRED_FIELDS <= base.keys()
    return base


def normalized_record(
    period: str,
    accesses: int,
    *,
    provider: str = "Provider A",
    technology: str = "FTTH",
    downstream: int = 100,
    upstream: int = 50,
):
    year, quarter = period.split("-Q")
    return {
        "period": period,
        "year": int(year),
        "quarter": int(quarter),
        "reported_accesses": accesses,
        "provider": provider,
        "technology": technology,
        "downstream_mbps": downstream,
        "upstream_mbps": upstream,
    }


def test_official_refresh_preserves_raw_provenance_and_separates_tunja_rows(tmp_path):
    raw_rows = [
        record(),
        record(
            proveedor="Proveedor B", tecnologia="HFC", trimestre="2", accesos_fijos_internet="4"
        ),
        record(codigo_dane_municipio="15022", municipio="OTRO MUNICIPIO"),
        record(),
    ]
    snapshot = refresh_official_data(
        tmp_path / "official.sqlite3",
        fetch_records=lambda: raw_rows,
        fetch_metadata=lambda: {"source_updated_at_utc": "2025-07-01T00:00:00+00:00"},
    )
    batch = snapshot["batch"]
    assert batch["retrieved_count"] == 4
    assert batch["accepted_count"] == 2
    assert batch["rejected_count"] == 2
    assert batch["source_updated_at_utc"] == "2025-07-01T00:00:00+00:00"
    assert batch["content_sha256"]
    assert json.loads(batch["error_summary"])["record_outside_tunja"] == 1
    assert json.loads(batch["error_summary"])["duplicate_source_row"] == 1
    assert snapshot["records"][0]["municipality_code"] == "15001"
    assert snapshot["records"][0]["downstream_mbps"] == 100
    assert snapshot["records"][0]["reported_accesses"] == 12
    assert snapshot["records"][0]["raw"] == raw_rows[0]
    with sqlite3.connect(tmp_path / "official.sqlite3") as connection:
        raw_count = connection.execute(
            "SELECT COUNT(*) FROM connectivity_source_raw WHERE batch_id = ?",
            (batch["batch_id"],),
        ).fetchone()[0]
    assert raw_count == len(raw_rows)


def test_connectivity_csv_and_json_exports_include_raw_processed_and_provenance(tmp_path):
    snapshot = refresh_official_data(
        tmp_path / "exports.sqlite3",
        fetch_records=lambda: [record()],
        fetch_metadata=lambda: {"source_updated_at_utc": "2025-07-01T00:00:00+00:00"},
    )

    rows = list(csv.DictReader(io.StringIO(build_connectivity_csv(snapshot))))
    metadata = json.loads(rows[0]["metadata_json"])
    csv_record = rows[1]
    assert rows[0]["export_record_type"] == "metadata"
    assert metadata["source"]["institution"].endswith("(MinTIC)")
    assert metadata["source"]["dataset_id"] == "hwrp-3su3"
    assert metadata["source"]["official_source_url"].startswith("https://www.datos.gov.co/")
    assert (
        metadata["provenance"]["retrieval_timestamp_utc"] == snapshot["batch"]["retrieved_at_utc"]
    )
    assert metadata["provenance"]["validation_status"] == "valid"
    assert metadata["coverage"]["latest_reporting_period"] == "2025-Q1"
    assert metadata["coverage"]["geographic_granularity"] == "municipality"
    assert csv_record["export_record_type"] == "record"
    assert json.loads(csv_record["raw_source_record_json"])["velocidad_bajada"] == "100"
    assert csv_record["processed_downstream_mbps"] == "100"
    assert csv_record["record_validation_status"] == "accepted"

    bundle = json.loads(build_connectivity_json(snapshot))
    assert bundle["data_classification"].startswith("REAL_OFFICIAL_DATA")
    assert bundle["snapshot_status"] == "validated_local_snapshot"
    assert bundle["raw_records"]["items"][0]["velocidad_bajada"] == "100"
    assert bundle["processed_records"]["items"][0]["downstream_mbps"] == 100
    assert bundle["analysis"]["analysis_status"] == "available"
    assert "saturation" in bundle["limitations"]


def test_connectivity_exports_label_stale_snapshot_and_allow_empty_data():
    stale_snapshot = {
        "stale": True,
        "records": [],
        "batch": {
            "source_id": "mintic_fixed_accesses",
            "retrieved_at_utc": "2025-01-02T03:04:05+00:00",
            "validation_status": "valid",
        },
    }
    bundle = build_export_bundle(stale_snapshot)
    csv_rows = list(csv.DictReader(io.StringIO(build_connectivity_csv(stale_snapshot))))
    assert bundle["snapshot_status"] == "stale_local_snapshot"
    assert bundle["record_count"] == 0
    assert bundle["coverage"]["reporting_periods"] == []
    assert bundle["analysis"]["latest_reporting_period"] is None
    assert bundle["analysis"]["analysis_status"] == "available"
    assert len(csv_rows) == 1
    assert csv_rows[0]["export_record_type"] == "metadata"
    assert json.loads(csv_rows[0]["metadata_json"])["snapshot_status"] == "stale_local_snapshot"


def test_connectivity_exports_handle_missing_normalized_fields_without_claiming_validity():
    snapshot = {
        "records": [{"raw": {"proveedor": "Proveedor A"}, "period": "2025-Q1"}],
        "batch": {"validation_status": "valid"},
    }

    bundle = build_export_bundle(snapshot)
    rows = list(csv.DictReader(io.StringIO(build_connectivity_csv(snapshot))))
    exported = json.loads(build_connectivity_json(snapshot))
    assert bundle["analysis"]["analysis_status"] == "unavailable_incomplete_normalized_fields"
    assert bundle["provenance"]["export_validation_status"] == "incomplete_snapshot_records"
    assert rows[1]["record_validation_status"] == "incomplete_normalized_record"
    assert exported["processed_records"]["items"][0]["provider"] is None


def test_normalization_preserves_decimal_speed_values_and_rejects_fractional_counts():
    from wifi_tunja_smart_predictor.connectivity.mintic import _normalize

    normalized = _normalize(record(velocidad_bajada="12.75", velocidad_subida="3.5"))
    assert normalized["downstream_mbps"] == 12.75
    assert normalized["upstream_mbps"] == 3.5
    with pytest.raises(ValueError, match="accesos_fijos_internet"):
        _normalize(record(accesos_fijos_internet="1.5"))


def test_period_comparison_calculates_absolute_and_percentage_differences():
    records = [
        normalized_record("2025-Q1", 10, provider="A", downstream=100),
        normalized_record("2025-Q1", 0, provider="B", downstream=200),
        normalized_record("2025-Q2", 15, provider="A", downstream=100),
        normalized_record("2025-Q2", 7, provider="C", downstream=300),
    ]

    result = compare_tunja_periods(records, "2025-Q1", "2025-Q2")
    total = result["indicators"]["reported_accesses_total"]
    provider_a = next(
        item
        for item in result["indicators"]["provider_reported_accesses"]
        if item["category"] == "A"
    )
    missing_provider = next(
        item
        for item in result["indicators"]["provider_reported_accesses"]
        if item["category"] == "B"
    )
    assert result["status"] == "available"
    assert available_tunja_periods(records) == ["2025-Q1", "2025-Q2"]
    assert total["baseline_value"] == 10
    assert total["comparison_value"] == 22
    assert total["absolute_difference"] == 12
    assert total["percentage_difference"] == 120.0
    assert provider_a["absolute_difference"] == 5
    assert provider_a["percentage_difference"] == 50.0
    assert missing_provider["baseline_value"] == 0
    assert missing_provider["comparison_value"] is None
    assert missing_provider["absolute_difference"] is None
    assert missing_provider["status"] == "missing_in_comparison"
    downstream = result["indicators"]["downstream_band_reported_accesses"]
    assert any(item["status"] == "missing_in_baseline" for item in downstream)
    assert "not measure" in result["interpretation"]


def test_period_comparison_zero_baseline_has_no_percentage_and_missing_period_is_safe():
    zero_baseline = compare_tunja_periods(
        [normalized_record("2025-Q1", 0), normalized_record("2025-Q2", 5)],
        "2025-Q1",
        "2025-Q2",
    )
    total = zero_baseline["indicators"]["reported_accesses_total"]
    assert total["absolute_difference"] == 5
    assert total["percentage_difference"] is None
    assert total["percentage_status"] == "undefined_zero_baseline"

    missing = compare_tunja_periods([normalized_record("2025-Q1", 10)], "2025-Q1", "2025-Q3")
    assert missing["status"] == "period_unavailable"
    assert missing["periods_available"] == {"2025-Q1": True, "2025-Q3": False}
    assert missing["indicators"] == {}
    assert "No values were interpolated" in missing["message"]


def test_period_comparison_ignores_incomplete_invalid_and_negative_records():
    records = [
        normalized_record("2025-Q1", 5),
        normalized_record("2025-Q2", 10),
        {"period": "2025-Q2", "reported_accesses": 100},
        normalized_record("2025-Q2", -3),
        normalized_record("2025-Q9", 20),
    ]
    result = compare_tunja_periods(records, "2025-Q1", "2025-Q2")
    assert result["status"] == "available"
    assert result["invalid_records_ignored"] == 3
    assert result["indicators"]["reported_accesses_total"]["absolute_difference"] == 5


def test_failed_refresh_uses_latest_valid_snapshot_and_marks_failure(tmp_path):
    database = tmp_path / "offline.sqlite3"
    original = refresh_official_data(
        database,
        fetch_records=lambda: [record()],
        fetch_metadata=lambda: {},
    )

    def unavailable():
        raise SourceUnavailableError("offline")

    cached = refresh_official_data(database, fetch_records=unavailable, fetch_metadata=lambda: {})
    assert cached["refresh_error"] == "SourceUnavailableError"
    assert cached["records"]
    assert load_snapshot(database)["batch"]["validation_status"] == "valid"
    assert load_snapshot(database)["batch"]["batch_id"] == original["batch"]["batch_id"]
    history = get_refresh_history(path=database)
    assert history["total"] == 2
    assert history["items"][0]["status"] == "failure"
    assert history["items"][0]["validation_result"] == "not_run"
    assert history["items"][1]["status"] == "success"
    assert cached["refresh_attempt"]["error_summary"] == "SourceUnavailableError"
    status = latest_refresh_status(database)
    assert status["latest_attempt"]["status"] == "failure"
    assert status["snapshot"]["status"] == "cached_local"
    assert status["snapshot"]["is_live"] is False


def test_invalid_refresh_is_recorded_and_preserves_previous_success(tmp_path):
    database = tmp_path / "invalid-attempt.sqlite3"
    original = refresh_official_data(
        database,
        fetch_records=lambda: [record()],
        fetch_metadata=lambda: {},
    )
    failed = refresh_official_data(
        database,
        fetch_records=lambda: [record(trimestre="5")],
        fetch_metadata=lambda: {},
    )

    assert failed["refresh_error"] == "ValueError"
    assert failed["refresh_attempt"]["status"] == "failure"
    assert failed["refresh_attempt"]["records_retrieved"] == 1
    assert failed["refresh_attempt"]["validation_result"] == "invalid"
    assert load_snapshot(database)["batch"]["batch_id"] == original["batch"]["batch_id"]
    latest = latest_refresh_status(database)
    assert latest["latest_attempt"]["status"] == "failure"
    assert latest["latest_successful_attempt"]["snapshot_batch_id"] == original["batch"]["batch_id"]


def test_failed_first_refresh_is_recorded_without_replacing_data(tmp_path):
    database = tmp_path / "first-failure.sqlite3"

    def unavailable():
        raise SourceUnavailableError("offline")

    with pytest.raises(SourceUnavailableError):
        refresh_official_data(database, fetch_records=unavailable, fetch_metadata=lambda: {})

    history = get_refresh_history(path=database)
    assert history["total"] == 1
    assert history["items"][0]["status"] == "failure"
    assert history["items"][0]["records_retrieved"] is None
    assert latest_refresh_status(database)["snapshot"]["status"] == "unavailable"


def test_refresh_history_paginates_newest_first(tmp_path):
    database = tmp_path / "pages.sqlite3"
    for quarter in (1, 2, 3):
        refresh_official_data(
            database,
            fetch_records=lambda quarter=quarter: [record(trimestre=str(quarter))],
            fetch_metadata=lambda: {},
        )

    first = get_refresh_history(limit=2, offset=0, path=database)
    second = get_refresh_history(limit=2, offset=2, path=database)
    assert first["total"] == 3
    assert len(first["items"]) == 2
    assert len(second["items"]) == 1
    assert [item["reporting_period"] for item in first["items"]] == ["2025-Q3", "2025-Q2"]
    assert second["items"][0]["reporting_period"] == "2025-Q1"
    assert all(item["status"] == "success" for item in first["items"] + second["items"])
    assert all(item["records_retrieved"] == 1 for item in first["items"] + second["items"])
    assert all(item["validation_result"] == "valid" for item in first["items"] + second["items"])


def test_unchanged_source_update_skips_full_data_download(tmp_path):
    database = tmp_path / "unchanged.sqlite3"
    metadata = {"source_updated_at_utc": "2025-07-01T00:00:00+00:00"}
    original = refresh_official_data(
        database,
        fetch_records=lambda: [record()],
        fetch_metadata=lambda: metadata,
    )
    refreshed = refresh_official_data(
        database,
        fetch_records=lambda: pytest.fail("unchanged source should not download rows"),
        fetch_metadata=lambda: metadata,
    )
    assert refreshed["unchanged_source"] is True
    assert refreshed["batch"]["batch_id"] == original["batch"]["batch_id"]


def test_snapshot_and_source_registry_mark_old_cache_stale(tmp_path):
    from wifi_tunja_smart_predictor.connectivity.sources import source_registry

    database = tmp_path / "stale.sqlite3"
    snapshot = refresh_official_data(
        database,
        fetch_records=lambda: [record()],
        fetch_metadata=lambda: {"source_updated_at_utc": "2025-07-01T00:00:00+00:00"},
    )
    old_time = (datetime.now(UTC) - timedelta(days=31)).isoformat()
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE connectivity_batches SET retrieved_at_utc = ? WHERE batch_id = ?",
            (old_time, snapshot["batch"]["batch_id"]),
        )
    old_snapshot = load_snapshot(database)
    assert old_snapshot["stale"] is True
    registry = source_registry(
        {
            "mintic_fixed_accesses": {
                "last_successful_retrieval_utc": old_time,
                "last_source_update_utc": "2025-07-01T00:00:00+00:00",
                "status": "stale_local_cache",
            }
        }
    )
    assert registry[0]["status"] == "stale_local_cache"
    assert registry[0]["last_source_update_utc"] == "2025-07-01T00:00:00+00:00"


def test_empty_or_invalid_refresh_never_overwrites_valid_snapshot(tmp_path):
    database = tmp_path / "preserve.sqlite3"
    original = refresh_official_data(
        database,
        fetch_records=lambda: [record()],
        fetch_metadata=lambda: {},
    )
    cached = refresh_official_data(database, fetch_records=lambda: [], fetch_metadata=lambda: {})
    assert cached["refresh_error"] == "ValueError"
    current = load_snapshot(database)
    assert current["batch"]["batch_id"] == original["batch"]["batch_id"]


def test_analysis_does_not_label_accesses_as_saturation():
    analysis = analyze_tunja(
        [
            {
                "reported_accesses": 3,
                "provider": "A",
                "technology": "FTTH",
                "period": "2025-Q1",
                "downstream_mbps": 100,
                "upstream_mbps": 50,
            },
            {
                "reported_accesses": 2,
                "provider": "A",
                "technology": "HFC",
                "period": "2025-Q2",
                "downstream_mbps": 50,
                "upstream_mbps": 10,
            },
        ]
    )
    assert analysis["sum_of_reported_access_counts_across_quarterly_rows"] == 5
    assert analysis["latest_reporting_period"] == "2025-Q2"
    assert analysis["latest_period_provider_accesses"] == [{"name": "A", "reported_accesses": 2}]
    assert "do not measure" in analysis["interpretation"]
    assert "saturation" in analysis["interpretation"]


def test_connectivity_api_returns_503_without_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFICIAL_DATA_DB_PATH", str(tmp_path / "missing.sqlite3"))
    client = TestClient(app)
    assert client.get("/connectivity/tunja").status_code == 503
    assert (
        client.get(
            "/connectivity/tunja/compare?baseline_period=2025-Q1&comparison_period=2025-Q2"
        ).status_code
        == 503
    )
    sources = client.get("/connectivity/sources")
    assert sources.status_code == 200
    registered = {source["source_id"]: source for source in sources.json()["sources"]}
    assert registered["mintic_fixed_accesses"]["dataset_id"] == "hwrp-3su3"
    assert registered["crc_postdata_quality_portal"]["status"] == "reference_only"


def test_connectivity_api_exposes_tunja_analysis_from_snapshot(tmp_path, monkeypatch):
    database = tmp_path / "api.sqlite3"
    monkeypatch.setenv("OFFICIAL_DATA_DB_PATH", str(database))
    refresh_official_data(
        database,
        fetch_records=lambda: [record()],
        fetch_metadata=lambda: {},
    )
    response = TestClient(app).get("/connectivity/tunja")
    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis"]["latest_period_reported_access_total"] == 12
    assert payload["analysis"]["geographic_granularity"] == "municipality"
    assert payload["saturation_ground_truth_available"] is False
    assert payload["provenance"]["transformation_version"] == "mintic-tunja-fixed-access-v1"


def test_connectivity_period_comparison_api_returns_deltas_and_provenance(tmp_path, monkeypatch):
    database = tmp_path / "period-comparison-api.sqlite3"
    monkeypatch.setenv("OFFICIAL_DATA_DB_PATH", str(database))
    refresh_official_data(
        database,
        fetch_records=lambda: [
            record(anno="2025", trimestre="1", accesos_fijos_internet="10"),
            record(anno="2025", trimestre="2", accesos_fijos_internet="15"),
        ],
        fetch_metadata=lambda: {},
    )

    client = TestClient(app)
    response = client.get(
        "/connectivity/tunja/compare?baseline_period=2025-Q1&comparison_period=2025-Q2"
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["comparison"]["status"] == "available"
    assert (
        payload["comparison"]["indicators"]["reported_accesses_total"]["absolute_difference"] == 5
    )
    assert (
        payload["comparison"]["indicators"]["reported_accesses_total"]["percentage_difference"]
        == 50.0
    )
    assert payload["provenance"]["dataset_id"] == "hwrp-3su3"
    assert payload["source"]["geographic_granularity"].startswith("Municipality")

    missing = client.get(
        "/connectivity/tunja/compare?baseline_period=2025-Q1&comparison_period=2024-Q1"
    )
    assert missing.status_code == 200
    assert missing.json()["comparison"]["status"] == "period_unavailable"
    assert (
        client.get(
            "/connectivity/tunja/compare?baseline_period=2025-Q9&comparison_period=2025-Q2"
        ).status_code
        == 422
    )


def test_connectivity_refresh_api_exposes_latest_and_paginated_history(tmp_path, monkeypatch):
    database = tmp_path / "refresh-api.sqlite3"
    monkeypatch.setenv("OFFICIAL_DATA_DB_PATH", str(database))
    refresh_official_data(
        database,
        fetch_records=lambda: [record()],
        fetch_metadata=lambda: {},
    )

    client = TestClient(app)
    latest = client.get("/connectivity/refresh/latest")
    history = client.get("/connectivity/refresh/history?limit=1&offset=0")
    assert latest.status_code == 200
    assert latest.json()["latest_attempt"]["status"] == "success"
    assert latest.json()["snapshot"]["status"] == "cached_local"
    assert latest.json()["snapshot"]["is_live"] is False
    assert history.status_code == 200
    assert history.json()["total"] == 1
    assert history.json()["items"][0]["reporting_period"] == "2025-Q1"
    assert client.get("/connectivity/refresh/history?limit=101").status_code == 422


def test_refresh_status_marks_stale_snapshot_as_cached_not_live(tmp_path):
    database = tmp_path / "stale-refresh.sqlite3"
    snapshot = refresh_official_data(
        database,
        fetch_records=lambda: [record()],
        fetch_metadata=lambda: {},
    )
    stale_time = (datetime.now(UTC) - timedelta(days=31)).isoformat()
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE connectivity_batches SET retrieved_at_utc = ? WHERE batch_id = ?",
            (stale_time, snapshot["batch"]["batch_id"]),
        )

    status = latest_refresh_status(database)
    assert status["latest_attempt"]["status"] == "success"
    assert status["snapshot"]["status"] == "stale_cached"
    assert status["snapshot"]["stale"] is True
    assert status["snapshot"]["is_live"] is False


def test_dashboard_exposes_official_snapshot_without_synthetic_fallback(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("OFFICIAL_DATA_DB_PATH", str(tmp_path / "empty-dashboard.sqlite3"))
    app_test = AppTest.from_string(
        "from app.dashboard import _render_real_connectivity\n_render_real_connectivity()\n",
        default_timeout=30,
    ).run()
    assert not app_test.exception
    assert any("No valid official snapshot" in message.value for message in app_test.info)
    assert any("do not measure" in message.value for message in app_test.warning)


def test_dashboard_labels_stale_empty_snapshot_and_keeps_metadata_downloads_available(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from app import dashboard

    monkeypatch.setattr(
        dashboard,
        "load_snapshot",
        lambda: {
            "batch": {"source_id": "mintic_fixed_accesses", "validation_status": "valid"},
            "records": [],
            "stale": True,
        },
    )
    app_test = AppTest.from_string(
        "from app.dashboard import _render_real_connectivity\n_render_real_connectivity()\n",
        default_timeout=30,
    ).run()
    assert not app_test.exception
    assert any("not live data" in item.value for item in app_test.warning)
    assert any("contains no validated records" in item.value for item in app_test.info)
    assert len(app_test.get("download_button")) == 2


def test_dashboard_shows_refresh_error_separately_from_stale_cached_data(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    database = tmp_path / "dashboard-refresh.sqlite3"
    monkeypatch.setenv("OFFICIAL_DATA_DB_PATH", str(database))
    snapshot = refresh_official_data(
        database,
        fetch_records=lambda: [record()],
        fetch_metadata=lambda: {},
    )

    def offline():
        raise SourceUnavailableError("offline")

    refresh_official_data(database, fetch_records=lambda: [], fetch_metadata=offline)
    stale_time = (datetime.now(UTC) - timedelta(days=31)).isoformat()
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE connectivity_batches SET retrieved_at_utc = ? WHERE batch_id = ?",
            (stale_time, snapshot["batch"]["batch_id"]),
        )
    app_test = AppTest.from_string(
        "from app.dashboard import _render_real_connectivity\n_render_real_connectivity()\n",
        default_timeout=30,
    ).run()
    assert not app_test.exception
    assert any("not live data" in item.value for item in app_test.warning)
    assert any(
        "Latest refresh error: SourceUnavailableError" in item.value for item in app_test.error
    )


def test_dashboard_refresh_runs_in_background_and_reports_completion(monkeypatch):
    from threading import Event

    from streamlit.testing.v1 import AppTest

    from app import dashboard

    refresh_started = Event()
    finish_refresh = Event()

    def slow_refresh():
        refresh_started.set()
        assert finish_refresh.wait(timeout=3)
        return {"batch": {}, "records": [], "stale": False}

    monkeypatch.setattr(dashboard, "refresh_official_data", slow_refresh)
    app_test = AppTest.from_string(
        "from app.dashboard import _render_real_connectivity\n_render_real_connectivity()\n",
        default_timeout=30,
    ).run()
    app_test.button[0].click().run()

    assert refresh_started.wait(timeout=2)
    assert not app_test.exception
    assert any("running in the background" in item.value for item in app_test.info)

    finish_refresh.set()
    future = app_test.session_state[dashboard.OFFICIAL_CONNECTIVITY_REFRESH_FUTURE_KEY]
    future.result(timeout=2)
    app_test.run()
    assert not app_test.exception
    assert any(
        "Official snapshot retrieved and validated" in item.value for item in app_test.success
    )


def test_dashboard_renders_verified_latest_period_not_all_period_sum(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    database = tmp_path / "dashboard.sqlite3"
    monkeypatch.setenv("OFFICIAL_DATA_DB_PATH", str(database))
    refresh_official_data(
        database,
        fetch_records=lambda: [
            record(anno="2025", trimestre="1", accesos_fijos_internet="12"),
            record(anno="2025", trimestre="2", accesos_fijos_internet="5"),
        ],
        fetch_metadata=lambda: {},
    )
    app_test = AppTest.from_string(
        "from app.dashboard import _render_real_connectivity\n_render_real_connectivity()\n",
        default_timeout=30,
    ).run()
    assert not app_test.exception
    assert any(
        metric.label == "Reported accesses · 2025-Q2" and metric.value == "5"
        for metric in app_test.metric
    )
    assert len(app_test.get("plotly_chart")) == 4
    assert [selector.label for selector in app_test.selectbox] == [
        "Baseline period",
        "Comparison period",
    ]
    assert all(selector.options == ["2025-Q1", "2025-Q2"] for selector in app_test.selectbox)
    assert any(
        metric.label == "Reported fixed-access total · comparison period"
        for metric in app_test.metric
    )
    assert [button.label for button in app_test.get("download_button")] == [
        "Download validated records (CSV)",
        "Download analysis and provenance (JSON)",
    ]
    assert any(
        "Sum across repeated quarterly rows (not unique subscribers): 17" in caption.value
        for caption in app_test.caption
    )


def test_assistant_answers_official_source_questions_from_snapshot(
    tmp_path, monkeypatch, mini_frame
):
    database = tmp_path / "assistant.sqlite3"
    monkeypatch.setenv("OFFICIAL_DATA_DB_PATH", str(database))
    refresh_official_data(
        database,
        fetch_records=lambda: [
            record(anno="2025", trimestre="1", proveedor="Proveedor A"),
            record(anno="2025", trimestre="1", proveedor="Proveedor B", tecnologia="HFC"),
        ],
        fetch_metadata=lambda: {"source_updated_at_utc": "2025-07-01T00:00:00+00:00"},
    )
    assistant = AssistantService(mini_frame)

    answer = assistant.handle(
        ChatRequest(message="Which providers are represented in the dataset?")
    )
    assert answer.intent == "OFFICIAL_CONNECTIVITY"
    assert "MinTIC" in answer.answer
    assert "Proveedor A" in answer.structured_result["providers"]
    assert answer.structured_result["reporting_period"] == "2025-Q1"
    assert answer.structured_result["synthetic_fallback_used"] is False

    quality = assistant.handle(ChatRequest(message="What quality indicators are available?"))
    assert quality.intent == "OFFICIAL_CONNECTIVITY"
    assert "not integrated" in quality.answer
    assert "latency" in quality.answer


def test_assistant_states_municipality_limit_and_no_saturation_evidence(
    tmp_path, monkeypatch, mini_frame
):
    database = tmp_path / "assistant-boundaries.sqlite3"
    monkeypatch.setenv("OFFICIAL_DATA_DB_PATH", str(database))
    refresh_official_data(
        database,
        fetch_records=lambda: [record()],
        fetch_metadata=lambda: {},
    )
    assistant = AssistantService(mini_frame)
    neighborhood = assistant.handle(
        ChatRequest(message="Which connectivity data is available at neighborhood level?")
    )
    saturation = assistant.handle(
        ChatRequest(message="¿La evidencia oficial confirma saturación de red?")
    )
    assert "no neighborhood data" in neighborhood.answer
    assert saturation.intent == "OFFICIAL_CONNECTIVITY"
    assert "No." in saturation.answer
    assert "no mide" in saturation.answer


def test_assistant_never_falls_back_to_synthetic_data_for_official_questions(
    tmp_path, monkeypatch, mini_frame
):
    monkeypatch.setenv("OFFICIAL_DATA_DB_PATH", str(tmp_path / "missing-assistant.sqlite3"))
    response = AssistantService(mini_frame).handle(
        ChatRequest(message="What official connectivity data is available for Tunja?")
    )
    assert response.intent == "OFFICIAL_CONNECTIVITY"
    assert response.structured_result["availability"] == "unavailable"
    assert response.structured_result["synthetic_fallback_used"] is False
    assert "No validated official snapshot" in response.answer


def test_official_connector_paginates_and_filters_tunja(monkeypatch):
    import wifi_tunja_smart_predictor.connectivity.mintic as mintic

    monkeypatch.setattr(mintic, "PAGE_SIZE", 2)
    first = [record(), record(proveedor="Proveedor B")]
    second = [record(proveedor="Proveedor C")]
    responses = [Mock(json=Mock(return_value=first)), Mock(json=Mock(return_value=second))]
    for response in responses:
        response.status_code = 200
        response.raise_for_status = Mock()
    calls = []
    request_timeouts = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs["params"]))
        request_timeouts.append(kwargs["timeout"])
        return responses.pop(0)

    monkeypatch.setattr(mintic.requests, "get", fake_get)
    rows = _fetch_records()
    assert len(rows) == 3
    assert [call[1]["$offset"] for call in calls] == [0, 2]
    assert all(call[1]["$where"] == "codigo_dane_municipio = '15001'" for call in calls)
    assert request_timeouts == [(5, 30), (5, 30)]


def test_official_connector_retries_rate_limits_and_uses_timeout(monkeypatch):
    import wifi_tunja_smart_predictor.connectivity.mintic as mintic

    rate_limited = Mock(status_code=429)
    success = Mock(status_code=200, json=Mock(return_value={"id": "hwrp-3su3", "columns": []}))
    success.raise_for_status = Mock()
    responses = [rate_limited, success]
    calls = []
    monkeypatch.setattr(mintic.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        mintic.requests,
        "get",
        lambda url, **kwargs: (calls.append(kwargs) or responses.pop(0)),
    )
    assert _request_json(API_URL) == {"id": "hwrp-3su3", "columns": []}
    assert len(calls) == 2
    assert calls[0]["timeout"] == (5, 30)


def test_official_connector_fails_on_timeout_and_schema_change(monkeypatch):
    import wifi_tunja_smart_predictor.connectivity.mintic as mintic

    monkeypatch.setattr(mintic.time, "sleep", lambda _: None)
    monkeypatch.setattr(mintic.requests, "get", Mock(side_effect=requests.Timeout))
    with pytest.raises(requests.Timeout):
        _request_json(API_URL)

    missing_resource = Mock(status_code=404)
    missing_resource.raise_for_status = Mock(side_effect=requests.HTTPError("not found"))
    monkeypatch.setattr(mintic.requests, "get", Mock(return_value=missing_resource))
    with pytest.raises(requests.HTTPError):
        _request_json(API_URL)

    bad_metadata = {"id": "hwrp-3su3", "columns": [{"fieldName": "anno"}]}
    monkeypatch.setattr(mintic, "_request_json", lambda *_args, **_kwargs: bad_metadata)
    with pytest.raises(ValueError, match="schema"):
        _metadata()


def test_official_connector_rejects_malformed_page_rows(monkeypatch):
    import wifi_tunja_smart_predictor.connectivity.mintic as mintic

    monkeypatch.setattr(mintic, "_request_json", lambda *_args, **_kwargs: [{"anno": "2025"}])
    with pytest.raises(ValueError, match="missing expected fields"):
        _fetch_records()
