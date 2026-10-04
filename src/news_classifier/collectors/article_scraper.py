from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from urllib.parse import urlsplit
from news_classifier.collectors.article_text import extract_article_text

from news_classifier.collectors.source_resolver import GoogleNewsSourceResolver, fetch_article_page

from news_classifier.models import NewsItem
from news_classifier.utils.http import HttpClient
from news_classifier.utils.text import safe_truncate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArticleEnrichmentResult:
    """본문 보강 결과. 본문을 확보하지 못한 사유도 기록."""
    item: NewsItem
    status: str
    http_status: int | None = None
    resolved_url: str = ""
    resolution_status: str = ""
    extraction_diagnostics: dict = field(default_factory=dict)


class ArticleScraper:
    """Best-effort article text extractor.

    The scraper is intentionally conservative. Many news sites block scraping
    or require special parsing. This class is used as an optional enrichment
    step and the pipeline still works when it cannot fetch content.
    """

    def __init__(self, http_client: HttpClient, max_chars: int = 1500):
        self.http_client = http_client
        self.max_chars = max_chars
        self.source_resolver = GoogleNewsSourceResolver(http_client)

    def enrich(self, item: NewsItem) -> NewsItem:
        return self.enrich_with_diagnostics(item).item

    def enrich_with_diagnostics(self, item: NewsItem) -> ArticleEnrichmentResult:
        if not item.link:
            return ArticleEnrichmentResult(item, "NO_LINK")
        outcome = fetch_article_page(self.http_client, item.link)
        response, http_status = outcome.response, outcome.http_status
        if response is None:
            return ArticleEnrichmentResult(item, outcome.status, http_status)
        response_url = getattr(response, "url", None) or item.link
        hostname = (urlsplit(response_url).hostname or "").casefold()
        resolved_url, resolution_status = "", "DIRECT_ARTICLE"
        if hostname == "news.google.com":
            resolution = self.source_resolver.resolve(item.link, response)
            if not resolution.url:
                return ArticleEnrichmentResult(item, resolution.status, resolution.http_status,
                                               resolution_status=resolution.status)
            resolved_url, resolution_status = resolution.url, resolution.status
            publisher = fetch_article_page(self.http_client, resolved_url)
            response, http_status = publisher.response, publisher.http_status
            if response is None:
                return ArticleEnrichmentResult(item, publisher.status, http_status, resolved_url, resolution_status)
            response_url = getattr(response, "url", "") or resolved_url
            hostname = (urlsplit(response_url).hostname or "").casefold()
            if hostname == "news.google.com":
                return ArticleEnrichmentResult(item, "SOURCE_URL_UNRESOLVED", http_status,
                                               resolved_url, resolution_status)
        elif response_url != item.link:
            resolved_url, resolution_status = response_url, "HTTP_SOURCE_REDIRECT"
        if hostname in {"consent.google.com", "www.google.com", "google.com"}:
            return ArticleEnrichmentResult(item, "CONSENT_OR_BLOCK_PAGE", http_status, resolved_url, resolution_status)
        # Parse bytes when available so publisher charset/meta declarations work.
        extraction = extract_article_text(getattr(response, "content", None) or response.text,
                                          title=item.title, page_url=response_url)
        diagnostics = extraction.diagnostics()
        if extraction.reason == "rss_document":
            return ArticleEnrichmentResult(item, "RSS_WRAPPER", http_status, resolved_url, resolution_status, diagnostics)
        if extraction.reason == "consent_or_block_page":
            return ArticleEnrichmentResult(item, "CONSENT_OR_BLOCK_PAGE", http_status, resolved_url, resolution_status, diagnostics)
        content = safe_truncate(extraction.content, self.max_chars)
        if not content:
            # 기존에 확보한 본문은 비어 있는 페이지 때문에 삭제하지 않음.
            return ArticleEnrichmentResult(item, "NO_USABLE_PARAGRAPHS", http_status, resolved_url, resolution_status, diagnostics)
        return ArticleEnrichmentResult(replace(item, content=content), "BODY_EXTRACTED", http_status,
                                       resolved_url, resolution_status, diagnostics)

    def enrich_many(self, items: list[NewsItem]) -> list[NewsItem]:
        return [self.enrich(item) for item in items]

    def enrich_many_with_diagnostics(self, items: list[NewsItem]) -> list[ArticleEnrichmentResult]:
        return [self.enrich_with_diagnostics(item) for item in items]
