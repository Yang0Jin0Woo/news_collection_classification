from __future__ import annotations

import logging
import urllib.parse
import xml.etree.ElementTree as ET

from news_classifier.collectors.base import (
    FeedParseError,
    NetworkCollectionError,
    NewsCollector,
)
from news_classifier.models import NewsItem
from news_classifier.utils.http import HttpClient
from news_classifier.utils.text import clean_text, normalize_description, strip_html
from news_classifier.utils.time import parse_rss_datetime

logger = logging.getLogger(__name__)

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"


class GoogleNewsRssCollector(NewsCollector):
    def __init__(self, http_client: HttpClient):
        self.http_client = http_client

    # 검색 키워드와 기간 조건을 Google News RSS URL로 변환
    def build_url(self, keyword: str, days: int = 7) -> str:
        query = f"{keyword} when:{days}d"
        encoded = urllib.parse.quote(query)
        return GOOGLE_NEWS_RSS.format(query=encoded)

    # Google News RSS에서 뉴스 목록 수집
    def fetch(self, keyword: str, limit: int = 10) -> list[NewsItem]:
        url = self.build_url(keyword)

        # 요청 실패와 실제 검색 결과 없음은 서로 다른 상태로 전달
        response = self.http_client.get(url)
        if response is None:
            raise NetworkCollectionError("Google News RSS 요청에 실패했습니다.")

        # RSS XML 응답 파싱
        try:
            root = ET.fromstring(response.text)
        except Exception as exc:
            logger.warning("RSS parse failed: %s", exc)
            raise FeedParseError("Google News RSS 응답을 해석할 수 없습니다.") from exc

        # channel이 없는 응답은 정상적인 빈 검색 결과가 아닌 잘못된 피드
        channel = root.find("channel")
        if channel is None:
            raise FeedParseError("Google News RSS 응답에 channel 요소가 없습니다.")

        items: list[NewsItem] = []
        feed_items = channel.findall("item")

        # RSS item 태그를 NewsItem 데이터 구조로 변환(원본 XML 데이터 -> 공통된 객체로 변환하여 일관되게 사용)
        for item in feed_items[:limit]:
            source_tag = item.find("source")
            news = NewsItem(
                keyword=keyword,
                title=clean_text(item.findtext("title", default="")),
                link=clean_text(item.findtext("link", default="")),
                published_at=parse_rss_datetime(clean_text(item.findtext("pubDate", default=""))),
                source=clean_text(source_tag.text if source_tag is not None else ""),
                description=normalize_description(strip_html(item.findtext("description", default=""))),
            )

            # 제목과 링크가 있는 기사만 최종 수집 목록에 추가
            if news.title and news.link:
                items.append(news)

        if feed_items and not items:
            raise FeedParseError("RSS 기사에서 필수 제목과 링크를 찾을 수 없습니다.")

        return items
