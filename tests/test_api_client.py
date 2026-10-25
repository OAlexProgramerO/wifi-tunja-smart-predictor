from __future__ import annotations

import io
from urllib.error import HTTPError, URLError

import pytest

from wifi_tunja_smart_predictor import api_client


def test_api_base_url_defaults_and_explicit_overrides():
    assert api_client.api_base_url("main", {}) == "http://127.0.0.1:8000"
    assert api_client.api_base_url("assistant", {}) == "http://127.0.0.1:8001"
    assert (
        api_client.api_base_url("main", {"MAIN_API_URL": "http://main-api:8000/"})
        == "http://main-api:8000"
    )


@pytest.mark.parametrize(
    "configured",
    ["", "main-api:8000", "ftp://main-api", "http://user:pass@main-api", "http://host:99999"],
)
def test_api_base_url_rejects_invalid_explicit_values(configured):
    with pytest.raises(ValueError, match=r"MAIN_API_URL must be an absolute HTTP\(S\) origin"):
        api_client.api_base_url("main", {"MAIN_API_URL": configured})


def test_api_request_sends_json_and_decodes_a_response(monkeypatch):
    received = {}

    class FakeResponse(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    def fake_urlopen(request, timeout):
        received["request"] = request
        received["timeout"] = timeout
        return FakeResponse(b'{"status":"ready"}')

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    monkeypatch.setenv("ASSISTANT_API_URL", "http://assistant-api:8001")

    assert api_client.assistant_api_request("/chat", {"message": "hello"}) == {"status": "ready"}
    assert received["request"].full_url == "http://assistant-api:8001/chat"
    assert received["request"].get_method() == "POST"
    assert received["request"].data == b'{"message": "hello"}'
    assert received["timeout"] == api_client.REQUEST_TIMEOUT_SECONDS


def test_api_request_sanitizes_http_and_connection_failures(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise URLError("sensitive host detail")

    monkeypatch.setattr(api_client, "urlopen", unavailable)
    with pytest.raises(api_client.ApiClientError, match="Main API is unavailable") as error:
        api_client.main_api_request("/health")
    assert "sensitive host detail" not in str(error.value)

    def bad_status(*_args, **_kwargs):
        raise HTTPError("http://private-host/", 503, "unavailable", {}, None)

    monkeypatch.setattr(api_client, "urlopen", bad_status)
    with pytest.raises(api_client.ApiClientError, match="HTTP 503") as error:
        api_client.main_api_request("/ready")
    assert "private-host" not in str(error.value)


def test_main_api_text_request_reads_markdown_and_enforces_response_limit(monkeypatch):
    received = {}

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    def fake_urlopen(request, timeout):
        received["request"] = request
        return FakeResponse(b"# Report\n")

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    monkeypatch.setenv("MAIN_API_URL", "http://main-api:8000")
    assert (
        api_client.main_api_text_request("/experiments/id/report?format=markdown") == "# Report\n"
    )
    assert received["request"].get_header("Accept") == "text/markdown"

    monkeypatch.setattr(
        api_client,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(b"x" * (api_client.MAX_TEXT_REPORT_BYTES + 1)),
    )
    with pytest.raises(api_client.ApiClientError, match="exceeds the configured size limit"):
        api_client.main_api_text_request("/report")
