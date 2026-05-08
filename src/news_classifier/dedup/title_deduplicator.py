from __future__ import annotations

from news_classifier.models import NewsItem
from news_classifier.utils.text import normalize_key
from news_classifier.utils.hash import stable_hash


class TitleSourceDeduplicator:
    def deduplicate(self, items: list[NewsItem]) -> list[NewsItem]:
        seen: set[tuple[str, str]] = set()
        deduped: list[NewsItem] = []
        for item in items:
            key = (normalize_key(item.title), normalize_key(item.source))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped


class LinkHashDeduplicator:
    def deduplicate(self, items: list[NewsItem]) -> list[NewsItem]:
        seen: set[str] = set()
        result: list[NewsItem] = []
        for item in items:
            key = stable_hash(item.link or item.title)
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result
