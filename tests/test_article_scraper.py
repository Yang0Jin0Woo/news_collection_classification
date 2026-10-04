from types import SimpleNamespace

import pytest

from news_classifier.collectors.article_scraper import ArticleScraper
from news_classifier.models import NewsItem, NewsReference
from news_classifier.utils.http import HttpFetchResult


def test_enrichment_keeps_grouped_original_references():
    reference = NewsReference("related", "B", "2026-10-04", "https://example.com/2")
    item = NewsItem(keyword="AI", title="representative", link="https://example.com/1", related_articles=(reference,))
    client = SimpleNamespace(get=lambda _: SimpleNamespace(text="<p>A sufficiently long paragraph for an article content extraction test.</p>"))
    enriched = ArticleScraper(client).enrich(item)
    assert enriched.content
    assert enriched.related_articles == (reference,)
    assert enriched.group_article_count == 2


def test_enrichment_prefers_article_and_removes_page_noise_and_duplicate_paragraphs():
    paragraph = "식품 신제품 판매를 시작하고 이용 가격과 제공 서비스를 발표했습니다."
    html = f"<nav><p>메뉴와 회사 소개 및 이용 약관을 확인하는 페이지입니다.</p></nav><p>무관한 외부 설명입니다.</p><article><p>{paragraph}</p><p>{paragraph}</p><aside><p>관련 광고와 추천 상품의 소개 내용입니다.</p></aside></article>"
    client = SimpleNamespace(get=lambda _: SimpleNamespace(text=html))
    item = NewsItem(keyword="식품", title="신제품", link="https://example.com")
    assert ArticleScraper(client).enrich(item).content == paragraph


def test_empty_page_keeps_preexisting_content():
    client = SimpleNamespace(get=lambda _: SimpleNamespace(text="<p>짧음</p>"))
    item = NewsItem(keyword="", title="제목", link="https://example.com", content="이미 확보한 본문")
    assert ArticleScraper(client).enrich(item) == item


@pytest.mark.parametrize("status,http_status", [
    ("TIMEOUT", None), ("BLOCKED_HTTP", 403), ("HTTP_ERROR", 404),
    ("REQUEST_FAILED", None),
])
def test_scraper_records_http_outcomes_without_removing_original_content(status, http_status):
    client = SimpleNamespace(get_with_diagnostics=lambda _: HttpFetchResult(None, status, http_status))
    item = NewsItem("any", "title", "https://example.com/news", content="기존 본문")
    result = ArticleScraper(client).enrich_with_diagnostics(item)
    assert result.item is item
    assert result.status == status
    assert result.http_status == http_status


@pytest.mark.parametrize("url,html,status", [
    ("https://news.google.com/rss/articles/encoded", "<p>A sufficiently long wrapper paragraph should not become article content.</p>", "GOOGLE_NEWS_WRAPPER"),
    ("https://example.com/rss", "<rss><channel><item><description>News feed summary</description></item></channel></rss>", "RSS_WRAPPER"),
    ("https://example.com/atom", "<feed><entry><title>news</title></entry></feed>", "RSS_WRAPPER"),
    ("https://consent.google.com/news", "<p>A sufficiently long consent notice must not become article content.</p>", "CONSENT_OR_BLOCK_PAGE"),
    ("https://example.com/news", "<title>Just a moment...</title><p>A sufficiently long challenge notice should not become article content.</p>", "CONSENT_OR_BLOCK_PAGE"),
    ("https://example.com/news", "<title>Access Denied</title><p>A sufficiently long blocked notice should not become article content.</p>", "CONSENT_OR_BLOCK_PAGE"),
    ("https://example.com/news", "<p>짧음</p>", "NO_USABLE_PARAGRAPHS"),
])
def test_wrappers_block_pages_and_empty_paragraphs_are_not_article_body(url, html, status):
    calls = []
    def get(link):
        calls.append(link)
        return SimpleNamespace(text=html, url=url, status_code=200)
    item = NewsItem("any", "title", url)
    outcome = ArticleScraper(SimpleNamespace(get=get)).enrich_with_diagnostics(item)
    assert outcome.item is item
    assert outcome.status == status
    assert calls == [url]  # No canonical links or extra external requests.


def test_existing_http_redirect_to_publisher_is_eligible_for_body_extraction():
    paragraph = "Publisher article includes a sufficiently long and usable paragraph."
    response = SimpleNamespace(text=f"<article><p>{paragraph}</p></article>", url="https://publisher.example/article", status_code=200)
    item = NewsItem("any", "title", "https://news.google.com/rss/articles/encoded")
    outcome = ArticleScraper(SimpleNamespace(get=lambda _: response)).enrich_with_diagnostics(item)
    assert outcome.status == "BODY_EXTRACTED"
    assert outcome.item.content == paragraph
    assert outcome.item.link == item.link


def test_missing_link_does_not_make_http_request():
    client = SimpleNamespace(get=lambda _: pytest.fail("unexpected request"))
    item = NewsItem("any", "title", "")
    outcome = ArticleScraper(client).enrich_with_diagnostics(item)
    assert outcome.status == "NO_LINK"
    assert outcome.item is item
