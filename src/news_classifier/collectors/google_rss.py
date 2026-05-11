from __future__ import annotations

import logging
import urllib.parse
import xml.etree.ElementTree as ET

from news_classifier.collectors.base import NewsCollector
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

        # RSS 요청 실패 시 빈 목록 반환
        response = self.http_client.get(url)
        if response is None:
            return []

        # RSS XML 응답 파싱
        try:
            root = ET.fromstring(response.text)
        except Exception as exc:
            logger.warning("RSS parse failed: %s", exc)
            return []

        # channel 태그가 없으면 수집 결과 없음 처리
        channel = root.find("channel")
        if channel is None:
            return []

        items: list[NewsItem] = []

        # RSS item 태그를 NewsItem 데이터 구조로 변환(원본 XML 데이터 -> 공통된 객체로 변환하여 일관되게 사용)
        for item in channel.findall("item")[:limit]:
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

        return items