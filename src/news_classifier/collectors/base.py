from __future__ import annotations

from abc import ABC, abstractmethod
from news_classifier.models import NewsItem


class NewsCollector(ABC):
    @abstractmethod
    def fetch(self, keyword: str, limit: int) -> list[NewsItem]:
        raise NotImplementedError
