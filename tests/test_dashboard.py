"""Streamlit dashboard smoke coverage using its supported AppTest harness."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_overview_renders_with_v3_navigation(mini_frame, tmp_path, monkeypatch):
    from wifi_tunja_smart_predictor import config

    fake_processed = tmp_path / "processed.csv"
    mini_frame.to_csv(fake_processed, index=False)
    monkeypatch.setattr(config, "PROCESSED_DATASET_PATH", fake_processed)
    dashboard = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"
    app = AppTest.from_file(str(dashboard), default_timeout=30).run()
    assert not app.exception
    assert [item.value for item in app.title] == ["Overview"]
    assert any(metric.label == "Synthetic access points" for metric in app.metric)
