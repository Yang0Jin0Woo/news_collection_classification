from __future__ import annotations

import logging
from dataclasses import replace
from bs4 import BeautifulSoup

from news_classifier.models import NewsItem
from news_classifier.utils.http import HttpClient
from news_classifier.utils.text import clean_text, safe_truncate

logger = logging.getLogger(__name__)


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
        if not item.link:
            return item
        response = self.http_client.get(item.link)
        if response is None:
            return item
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        paragraphs = [clean_text(p.get_text(" ")) for p in soup.find_all("p")]
        content = " ".join(p for p in paragraphs if len(p) >= 20)
        content = safe_truncate(content, self.max_chars)
        return replace(item, content=content)

    def enrich_many(self, items: list[NewsItem]) -> list[NewsItem]:
        return [self.enrich(item) for item in items]
