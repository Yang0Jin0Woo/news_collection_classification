from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from urllib.parse import urlsplit
from bs4 import BeautifulSoup

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


class ArticleScraper:
    """Best-effort article text extractor.

    The scraper is intentionally conservative. Many news sites block scraping
    or require special parsing. This class is used as an optional enrichment
    step and the pipeline still works when it cannot fetch content.
    """

    def __init__(self, http_client: HttpClient, max_chars: int = 1500):
        self.http_client = http_client
        self.max_chars = max_chars

    def enrich(self, item: NewsItem) -> NewsItem:
        return self.enrich_with_diagnostics(item).item

    def enrich_with_diagnostics(self, item: NewsItem) -> ArticleEnrichmentResult:
        if not item.link:
            return ArticleEnrichmentResult(item, "NO_LINK")
        fetch = getattr(self.http_client, "get_with_diagnostics", None)
        if callable(fetch):
            outcome = fetch(item.link)
            response, http_status = outcome.response, outcome.http_status
            if response is None:
                return ArticleEnrichmentResult(item, outcome.status, http_status)
        else:
            # 기존 get만 제공하는 사용자 클라이언트와 테스트 대역 유지.
            response = self.http_client.get(item.link)
            http_status = getattr(response, "status_code", None)
        if response is None:
            return ArticleEnrichmentResult(item, "REQUEST_FAILED", http_status)
        response_url = getattr(response, "url", None) or item.link
        hostname = (urlsplit(response_url).hostname or "").casefold()
        if hostname == "news.google.com":
            # RSS 링크의 Google 중계 페이지는 기사 본문으로 사용하지 않음.
            # 검증되지 않은 canonical/외부 링크를 추가로 따라가지 않음.
            return ArticleEnrichmentResult(item, "GOOGLE_NEWS_WRAPPER", http_status)
        if hostname == "consent.google.com":
            return ArticleEnrichmentResult(item, "CONSENT_OR_BLOCK_PAGE", http_status)
        soup = BeautifulSoup(response.text, "html.parser")
        if soup.find("rss") is not None or soup.find("feed") is not None:
            return ArticleEnrichmentResult(item, "RSS_WRAPPER", http_status)
        page_title = clean_text(soup.title.get_text(" ") if soup.title else "").casefold()
        if page_title in {"access denied", "just a moment...", "attention required! | cloudflare"} or page_title.startswith("before you continue to google"):
            return ArticleEnrichmentResult(item, "CONSENT_OR_BLOCK_PAGE", http_status)
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
        content = safe_truncate(content, self.max_chars)
        if not content:
            # 기존에 확보한 본문은 비어 있는 페이지 때문에 삭제하지 않음.
            return ArticleEnrichmentResult(item, "NO_USABLE_PARAGRAPHS", http_status)
        return ArticleEnrichmentResult(replace(item, content=content), "BODY_EXTRACTED", http_status)

    def enrich_many(self, items: list[NewsItem]) -> list[NewsItem]:
        return [self.enrich(item) for item in items]

    def enrich_many_with_diagnostics(self, items: list[NewsItem]) -> list[ArticleEnrichmentResult]:
        return [self.enrich_with_diagnostics(item) for item in items]
