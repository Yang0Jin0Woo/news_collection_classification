from types import SimpleNamespace

import pytest
import requests

from news_classifier.utils.http import HttpClient


@pytest.mark.parametrize("status_code,expected", [
    (401, "BLOCKED_HTTP"), (403, "BLOCKED_HTTP"),
    (429, "BLOCKED_HTTP"), (451, "BLOCKED_HTTP"), (404, "HTTP_ERROR"),
])
def test_http_failures_have_structured_status_without_returning_error_page(monkeypatch, status_code, expected):
    response = requests.Response()
    response.status_code = status_code
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)
    outcome = HttpClient({}).get_with_diagnostics("https://example.com/news")
    assert outcome.status == expected
    assert outcome.http_status == status_code
    assert outcome.response is None
    assert HttpClient({}).get("https://example.com/news") is None


@pytest.mark.parametrize("error,expected", [
    (requests.Timeout("timeout"), "TIMEOUT"),
    (requests.ConnectionError("connection"), "REQUEST_FAILED"),
    (ValueError("unexpected"), "REQUEST_FAILED"),
])
def test_http_transport_failures_remain_best_effort(monkeypatch, error, expected):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(requests, "get", fail)
    outcome = HttpClient({}).get_with_diagnostics("https://example.com/news")
    assert outcome.response is None
    assert outcome.status == expected
    assert outcome.http_status is None


def test_http_success_preserves_request_settings_and_get_contract(monkeypatch):
    calls = []
    response = SimpleNamespace(status_code=200, raise_for_status=lambda: None)
    def get(*args, **kwargs):
        calls.append((args, kwargs))
        return response
    monkeypatch.setattr(requests, "get", get)
    client = HttpClient({"User-Agent": "test"}, timeout_seconds=12)
    assert client.get("https://example.com/news") is response
    assert calls == [(("https://example.com/news",), {
        "headers": {"User-Agent": "test"}, "timeout": 12, "allow_redirects": True,
    })]
