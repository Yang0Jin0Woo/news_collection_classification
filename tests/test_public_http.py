from types import SimpleNamespace

import pytest
import requests

from news_classifier.utils.http import HttpClient
from news_classifier.utils.urls import is_public_http_url


@pytest.mark.parametrize("url", ["http://localhost/news", "http://10.0.0.1", "http://[::1]", "http://127.1",
                                "http://169.254.169.254/", "http://host.local/news", "https://user:pass@example.com",
                                "https://example.com:1234", "file:///news", "https://example.com/\nnews"])
def test_unsafe_article_urls_are_rejected(url, monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("127.0.0.1", 80))])
    assert not is_public_http_url(url, check_dns=True)


def test_public_url_dns_rejects_private_address_in_mixed_answers(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: [
        (2, 1, 6, "", ("93.184.216.34", 443)), (2, 1, 6, "", ("10.0.0.1", 443)),
    ])
    assert not is_public_http_url("https://publisher.example/news", check_dns=True)


def test_private_redirect_is_rejected_before_second_network_request(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))])
    calls = []
    page = SimpleNamespace(status_code=302, headers={"Location": "http://127.0.0.1/admin"}, raise_for_status=lambda: None)
    monkeypatch.setattr(requests, "get", lambda *a, **k: calls.append((a, k)) or page)
    outcome = HttpClient({}).get_public_with_diagnostics("https://publisher.example/news")
    assert outcome.status == "UNSAFE_URL" and len(calls) == 1
    assert calls[0][1]["allow_redirects"] is False


def test_redirect_count_is_bounded(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))])
    calls = []
    page = SimpleNamespace(status_code=302, headers={"Location": "/same"}, raise_for_status=lambda: None)
    monkeypatch.setattr(requests, "get", lambda *a, **k: calls.append(a) or page)
    assert HttpClient({}).get_public_with_diagnostics("https://publisher.example/same", 2).status == "REDIRECT_LIMIT"
    assert len(calls) == 3


def test_google_rpc_uses_fixed_endpoint_no_redirect_and_keeps_timeout(monkeypatch):
    calls = []
    page = SimpleNamespace(status_code=200, raise_for_status=lambda: None)
    monkeypatch.setattr(requests, "post", lambda *a, **k: calls.append((a, k)) or page)
    assert HttpClient({"User-Agent": "test"}, 8).post_google_news_lookup({"f.req": "example"}).response is page
    assert calls[0][0][0].startswith("https://news.google.com/_/DotsSplashUi/data/batchexecute")
    assert calls[0][1]["allow_redirects"] is False and calls[0][1]["timeout"] == 8


@pytest.mark.parametrize("error, expected", [(requests.Timeout(), "TIMEOUT"), (requests.ConnectionError(), "REQUEST_FAILED")])
def test_lookup_transport_errors_are_best_effort(monkeypatch, error, expected):
    def fail(*a, **k):
        raise error
    monkeypatch.setattr(requests, "post", fail)
    assert HttpClient({}).post_google_news_lookup({}).status == expected
