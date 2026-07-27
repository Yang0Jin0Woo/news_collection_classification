from __future__ import annotations

from news_classifier.models import NewsItem
from news_classifier.utils.text import normalize_key


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
