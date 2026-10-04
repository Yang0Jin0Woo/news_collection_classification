import json
from types import SimpleNamespace

import pytest

from news_classifier.collectors.article_scraper import ArticleScraper
from news_classifier.collectors.source_resolver import GoogleNewsSourceResolver, parse_lookup_response
from news_classifier.models import NewsItem
from news_classifier.utils.http import HttpFetchResult


ARTICLE_ID = "CBMi" + "A" * 24
SOURCE = "https://news.google.com/rss/articles/" + ARTICLE_ID
PUBLISHER = "https://publisher.example/news/123"
SIGNATURE = '<div data-n-a-sg="test-signature" data-n-a-ts="123" data-n-a-id="' + ARTICLE_ID + '"></div>'
BODY = "공개 기사의 실제 본문으로 사용할 충분히 긴 문단이며 구체적인 사건에 대한 자세한 설명입니다."


def response(text, url=SOURCE):
    return SimpleNamespace(text=text, url=url, status_code=200)


def rpc_response(url=PUBLISHER):
    return ")]}'\n\n321\n" + json.dumps([["wrb.fr", "Fbv4je", json.dumps(["garturlres", url]), None, "0"]])


def fake_client(*, failure=None, publisher_html=None, initial=SIGNATURE):
    calls = []
    def get(url):
        calls.append(("get", url))
        text = (publisher_html or f"<article><p>{BODY}</p></article>") if url == PUBLISHER else initial
        return HttpFetchResult(response(text, url), "SUCCESS", 200)
    def lookup(data):
        calls.append(("lookup", data))
        return failure or HttpFetchResult(response(rpc_response()), "SUCCESS", 200)
    return SimpleNamespace(get_public_with_diagnostics=get, post_google_news_lookup=lookup), calls


def test_modern_google_url_resolves_and_extracts_body_without_changing_original_link():
    client, calls = fake_client()
    item = NewsItem("any", "제목", SOURCE)
    result = ArticleScraper(client).enrich_with_diagnostics(item)
    assert result.item.content == BODY
    assert result.item.link == SOURCE and not item.content
    assert result.resolved_url == PUBLISHER
    assert result.resolution_status == "GOOGLE_RPC_SOURCE_URL"
    assert result.status == "BODY_EXTRACTED"
    assert [method for method, _ in calls] == ["get", "lookup", "get"]
    envelope = json.loads(calls[1][1]["f.req"])[0][0]
    assert envelope[0] == "Fbv4je"
    assert json.loads(envelope[1])[2:] == [ARTICLE_ID, 123, "test-signature"]


def test_signature_page_fallback_is_bounded_and_does_not_follow_arbitrary_links():
    client, calls = fake_client()
    wrapper = response('<a href="https://advert.example/news">advertisement</a>')
    result = GoogleNewsSourceResolver(client).resolve(SOURCE, wrapper)
    assert result.url == PUBLISHER
    assert calls[0][0] == "get" and calls[0][1].startswith(SOURCE + "?hl=")
    assert len(calls) == 2


def test_no_signature_returns_unresolved_after_one_fallback_without_lookup():
    client, calls = fake_client(initial="<html><p>Google wrapper text</p></html>")
    result = GoogleNewsSourceResolver(client).resolve(SOURCE, response("<html></html>"))
    assert result.status == "SOURCE_URL_UNRESOLVED" and result.url == ""
    assert len(calls) == 1 and calls[0][0] == "get"


@pytest.mark.parametrize("status,code", [("TIMEOUT", None), ("BLOCKED_HTTP", 429), ("HTTP_ERROR", 500)])
def test_lookup_failure_does_not_become_article_body(status, code):
    client, calls = fake_client(failure=HttpFetchResult(None, status, code))
    item = NewsItem("any", "제목", SOURCE)
    result = ArticleScraper(client).enrich_with_diagnostics(item)
    assert result.item is item and result.status == status and result.http_status == code
    assert len(calls) == 2


@pytest.mark.parametrize("url", ["http://127.0.0.1/admin", "http://169.254.169.254/latest/", "file:///tmp/news",
                                "https://news.google.com/read/other", "https://user:secret@publisher.example/news"])
def test_lookup_never_accepts_local_credential_or_google_urls(url):
    assert parse_lookup_response(rpc_response(url)) == ""


@pytest.mark.parametrize("text", ["garbage", "[]", '[["wrb.fr","OtherRpc","[\"garturlres\",\"https://publisher.example/a\"]"]]',
                                 json.dumps([["wrb.fr", "Fbv4je", "bad json"]]),
                                 json.dumps([["wrb.fr", "Fbv4je", json.dumps(["garturlres", PUBLISHER])]] * 2)])
def test_malformed_or_multiple_lookup_results_are_not_guessed(text):
    assert parse_lookup_response(text) == ""


def test_resolved_url_cache_avoids_repeating_lookup():
    client, calls = fake_client()
    resolver = GoogleNewsSourceResolver(client)
    assert resolver.resolve(SOURCE, response(SIGNATURE)).url == PUBLISHER
    assert resolver.resolve(SOURCE, response(SIGNATURE)).url == PUBLISHER
    assert len(calls) == 1


def test_explicit_canonical_can_resolve_without_rpc():
    client, calls = fake_client()
    result = GoogleNewsSourceResolver(client).resolve(SOURCE, response(f'<link rel="canonical" href="{PUBLISHER}">'))
    assert result.url == PUBLISHER and result.status == "HTML_SOURCE_URL" and not calls


def test_json_ld_article_body_is_fallback_and_original_link_is_retained():
    html = '<script type="application/ld+json">' + json.dumps({"@type": "NewsArticle", "articleBody": BODY}) + '</script>'
    client, _ = fake_client(publisher_html=html)
    result = ArticleScraper(client).enrich_with_diagnostics(NewsItem("", "제목", PUBLISHER))
    assert result.item.content == BODY and result.status == "BODY_EXTRACTED"


def test_byte_parsing_uses_publisher_charset_not_bad_requests_text():
    page = SimpleNamespace(text="incorrect decoding", content=f'<meta charset="utf-8"><article><p>{BODY}</p></article>'.encode(),
                           url=PUBLISHER, status_code=200)
    result = ArticleScraper(SimpleNamespace(get=lambda _: page)).enrich(NewsItem("", "제목", PUBLISHER))
    assert result.content == BODY
