from __future__ import annotations

import logging
from dataclasses import dataclass

from news_classifier.classifiers.base import NewsClassifier
from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.collectors.base import NewsCollector
from news_classifier.collectors.article_scraper import ArticleScraper
from news_classifier.dedup.title_deduplicator import TitleSourceDeduplicator
from news_classifier.models import ClassifiedNews

logger = logging.getLogger(__name__)


@dataclass
class NewsPipeline:
    collector: NewsCollector
    classifier: NewsClassifier
    postprocessor: ClassificationPostProcessor
    deduplicator: TitleSourceDeduplicator
    article_scraper: ArticleScraper | None = None

    # 뉴스 수집부터 분류 결과 생성까지의 전체 파이프라인
    def run(self, keyword: str, limit: int = 10, enrich_content: bool = False) -> list[ClassifiedNews]:
        logger.info("news pipeline started keyword=%s limit=%s", keyword, limit)

        # 1. 키워드 기반 뉴스 수집
        news_list = self.collector.fetch(keyword=keyword, limit=limit)
        logger.info("fetched=%s", len(news_list))

        # 2. 중복 정보 제거
        news_list = self.deduplicator.deduplicate(news_list)
        logger.info("deduplicated=%s", len(news_list))

        # 3. 기사 페이지 본문을 추가 수집해 분류 입력 데이터 보강
        if enrich_content and self.article_scraper is not None:
            news_list = self.article_scraper.enrich_many(news_list)
            logger.info("content enrichment finished")

        # 4. 모델 분류 및 규칙 기반 후처리
        results: list[ClassifiedNews] = []
        for idx, item in enumerate(news_list, start=1):
            logger.info("classifying %s/%s | %s", idx, len(news_list), item.title)

            # 모델 기반 1차 분류
            prediction = self.classifier.classify(item.classification_text())

            # 모델 결과와 규칙 보정 결과를 반영함으로써 최종적인 결과 생성
            classified = self.postprocessor.process(item, prediction)
            results.append(classified)

        return results