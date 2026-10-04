from __future__ import annotations

import logging
import json
from dataclasses import dataclass, replace
from urllib.parse import urlsplit
from bs4 import BeautifulSoup

from news_classifier.collectors.source_resolver import GoogleNewsSourceResolver, fetch_article_page

from news_classifier.models import NewsItem
from news_classifier.utils.http import HttpClient
from news_classifier.utils.text import clean_text, normalize_key, safe_truncate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArticleEnrichmentResult:
    """본문 보강 결과. 본문을 확보하지 못한 사유도 기록."""
    item: NewsItem
    status: str
    http_status: int | None = None
    resolved_url: str = ""
    resolution_status: str = ""


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
        soup = BeautifulSoup(getattr(response, "content", None) or response.text, "html.parser")
        if soup.find("rss") is not None or soup.find("feed") is not None:
            return ArticleEnrichmentResult(item, "RSS_WRAPPER", http_status, resolved_url, resolution_status)
        page_title = clean_text(soup.title.get_text(" ") if soup.title else "").casefold()
        if page_title in {"access denied", "just a moment...", "attention required! | cloudflare"} or page_title.startswith("before you continue to google"):
            return ArticleEnrichmentResult(item, "CONSENT_OR_BLOCK_PAGE", http_status, resolved_url, resolution_status)
        structured_bodies = []
        def article_bodies(value):
            if isinstance(value, list):
                for child in value:
                    article_bodies(child)
            elif isinstance(value, dict):
                types = value.get("@type", [])
                types = [types] if isinstance(types, str) else types
                if (isinstance(types, list) and any(t in {"NewsArticle", "Article", "ReportageNewsArticle"} for t in types)
                        and isinstance(value.get("articleBody"), str)):
                    body = clean_text(BeautifulSoup(value["articleBody"], "html.parser").get_text(" "))
                    if len(body) >= 40:
                        structured_bodies.append(body)
                if "@graph" in value:
                    article_bodies(value["@graph"])
        for tag in soup.find_all("script", type="application/ld+json"):
            try:
                article_bodies(json.loads(tag.string or tag.get_text()))
            except (ValueError, TypeError, RecursionError):
                continue
        for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form"]):
            tag.decompose()
        article = soup.find("article")
        main = soup.find("main") or soup.find(attrs={"role": "main"})
        root = article if article is not None else (main if main is not None else soup)
        paragraphs = []
        seen = {normalize_key(item.title)}
        for paragraph in root.find_all("p"):
            text = clean_text(paragraph.get_text(" "))
            key = normalize_key(text)
            if len(text) < 20 or key in seen:
                continue
            seen.add(key)
            paragraphs.append(text)
        content = " ".join(paragraphs)
        if not content and len(set(structured_bodies)) == 1:
            content = structured_bodies[0]
        content = safe_truncate(content, self.max_chars)
        if not content:
            # 기존에 확보한 본문은 비어 있는 페이지 때문에 삭제하지 않음.
            return ArticleEnrichmentResult(item, "NO_USABLE_PARAGRAPHS", http_status, resolved_url, resolution_status)
        return ArticleEnrichmentResult(replace(item, content=content), "BODY_EXTRACTED", http_status,
                                       resolved_url, resolution_status)

    def enrich_many(self, items: list[NewsItem]) -> list[NewsItem]:
        return [self.enrich(item) for item in items]

    def enrich_many_with_diagnostics(self, items: list[NewsItem]) -> list[ArticleEnrichmentResult]:
        return [self.enrich_with_diagnostics(item) for item in items]
