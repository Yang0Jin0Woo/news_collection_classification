from __future__ import annotations

import logging
from dataclasses import replace
from bs4 import BeautifulSoup

from news_classifier.models import NewsItem
from news_classifier.utils.http import HttpClient
from news_classifier.utils.text import clean_text, normalize_key, safe_truncate

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
            return item  # 기존에 확보한 본문은 비어 있는 페이지 때문에 삭제하지 않음.
        return replace(item, content=content)

    def enrich_many(self, items: list[NewsItem]) -> list[NewsItem]:
        return [self.enrich(item) for item in items]
