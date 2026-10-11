from types import SimpleNamespace

import pytest
import requests

from news_classifier.utils.http import HttpClient


@pytest.fixture(params=["get", "public_get", "google_post"])
def request_http(request, monkeypatch):
    monkeypatch.setattr(
        "news_classifier.utils.http.is_public_http_url", lambda url, check_dns: True
    )
    client = HttpClient({})
    return {
        "get": lambda: client.get_with_diagnostics("https://example.com/news"),
        "public_get": lambda: client.get_public_with_diagnostics("https://example.com/news"),
        "google_post": lambda: client.post_google_news_lookup({}),
    }[request.param]


@pytest.mark.parametrize("status_code,expected", [
    (401, "BLOCKED_HTTP"), (403, "BLOCKED_HTTP"),
    (429, "BLOCKED_HTTP"), (451, "BLOCKED_HTTP"),
    (400, "HTTP_ERROR"), (404, "HTTP_ERROR"),
    (500, "HTTP_ERROR"), (503, "HTTP_ERROR"),
])
def test_http_failures_have_structured_status_without_returning_error_page(monkeypatch, request_http, status_code, expected):
    response = requests.Response()
    response.status_code = status_code
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: response)
    outcome = request_http()
    assert outcome.status == expected
    assert outcome.http_status == status_code
    assert outcome.response is None
    assert HttpClient({}).get("https://example.com/news") is None


@pytest.mark.parametrize("error,expected", [
    (requests.Timeout("timeout"), "TIMEOUT"),
    (requests.ConnectionError("connection"), "REQUEST_FAILED"),
    (ValueError("unexpected"), "REQUEST_FAILED"),
])
def test_http_transport_failures_remain_best_effort(monkeypatch, request_http, error, expected):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(requests, "get", fail)
    monkeypatch.setattr(requests, "post", fail)
    outcome = request_http()
    assert outcome.response is None
    assert outcome.status == expected
    assert outcome.http_status is None


def test_timeout_after_response_still_has_no_http_status(monkeypatch, request_http):
    def fail():
        raise requests.Timeout("timeout after response")
    response = SimpleNamespace(status_code=503, raise_for_status=fail)
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: response)
    outcome = request_http()
    assert outcome.response is None
    assert outcome.status == "TIMEOUT"
    assert outcome.http_status is None


@pytest.mark.parametrize("method,status_code,expected", [
    ("get_with_diagnostics", 304, "SUCCESS"),
    ("get_public_with_diagnostics", 304, "HTTP_ERROR"),
    ("post_google_news_lookup", 302, "HTTP_ERROR"),
    ("post_google_news_lookup", 304, "HTTP_ERROR"),
])
def test_non_error_response_keeps_each_request_policy(monkeypatch, method, status_code, expected):
    response = requests.Response()
    response.status_code = status_code
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: response)
    monkeypatch.setattr(
        "news_classifier.utils.http.is_public_http_url", lambda url, check_dns: True
    )
    argument = {} if method == "post_google_news_lookup" else "https://example.com/news"
    outcome = getattr(HttpClient({}), method)(argument)
    assert outcome.status == expected
    assert outcome.http_status == status_code
    assert outcome.response is (response if expected == "SUCCESS" else None)


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
