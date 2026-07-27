from __future__ import annotations

from abc import ABC, abstractmethod
from news_classifier.models import NewsItem


class CollectionError(RuntimeError):
    """뉴스 수집 단계에서 발생한 오류."""


class NetworkCollectionError(CollectionError):
    """RSS 서버 요청 자체가 실패한 경우."""


class FeedParseError(CollectionError):
    """응답은 받았지만 RSS 구조를 해석할 수 없는 경우."""


class NewsCollector(ABC):
    @abstractmethod
    def fetch(self, keyword: str, limit: int) -> list[NewsItem]:
        raise NotImplementedError
