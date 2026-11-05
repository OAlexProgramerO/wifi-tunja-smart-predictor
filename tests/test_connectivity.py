from __future__ import annotations

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
from wifi_tunja_smart_predictor.connectivity.analysis import analyze_tunja
from wifi_tunja_smart_predictor.connectivity.mintic import (
    API_URL,
    REQUIRED_FIELDS,
    SourceUnavailableError,
    _fetch_records,
    _metadata,
    _request_json,
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


def test_normalization_preserves_decimal_speed_values_and_rejects_fractional_counts():
    from wifi_tunja_smart_predictor.connectivity.mintic import _normalize

    normalized = _normalize(record(velocidad_bajada="12.75", velocidad_subida="3.5"))
    assert normalized["downstream_mbps"] == 12.75
    assert normalized["upstream_mbps"] == 3.5
    with pytest.raises(ValueError, match="accesos_fijos_internet"):
        _normalize(record(accesos_fijos_internet="1.5"))


def test_failed_refresh_uses_latest_valid_snapshot_and_marks_failure(tmp_path):
    database = tmp_path / "offline.sqlite3"
    refresh_official_data(
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
    assert len(app_test.get("plotly_chart")) == 3
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
