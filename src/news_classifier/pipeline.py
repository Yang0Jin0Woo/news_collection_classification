from __future__ import annotations

import logging
from dataclasses import dataclass

from news_classifier.classifiers.base import NewsClassifier
from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.collectors.base import NewsCollector
from news_classifier.collectors.article_scraper import ArticleScraper
from news_classifier.dedup.title_deduplicator import TitleSourceDeduplicator
from news_classifier.models import ClassifiedNews, NewsItem

logger = logging.getLogger(__name__)


def _chunks(items: list[NewsItem], size: int) -> list[list[NewsItem]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


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
        print("뉴스 수집 중...")

        # 1. 키워드 기반 뉴스 수집
        news_list = self.collector.fetch(keyword=keyword, limit=limit)
        logger.info("fetched=%s", len(news_list))
        print(f"수집 완료: {len(news_list)}건")

        # 2. 중복 정보 제거
        news_list = self.deduplicator.deduplicate(news_list)
        logger.info("deduplicated=%s", len(news_list))
        print(f"중복 제거 완료: {len(news_list)}건")

        # 3. 기사 페이지 본문을 추가 수집해 분류 입력 데이터 보강
        if enrich_content and self.article_scraper is not None:
            print("기사 본문 보강 중...")
            news_list = self.article_scraper.enrich_many(news_list)
            logger.info("content enrichment finished")
            print("기사 본문 보강 완료")

        # 4. 모델 분류 및 규칙 기반 후처리
        results: list[ClassifiedNews] = []
        if not news_list:
            print("분류할 뉴스 없음")
            return results

        batch_size = max(1, getattr(self.classifier, "batch_size", 1))
        batches = _chunks(news_list, batch_size)
        print("분류 모델 로딩 및 뉴스 분류 시작...")

        for batch_idx, batch in enumerate(batches, start=1):
            start = (batch_idx - 1) * batch_size + 1
            end = min(start + len(batch) - 1, len(news_list))
            print(f"뉴스 분류 중: {start}-{end}/{len(news_list)}")
            logger.info("classifying batch %s/%s size=%s", batch_idx, len(batches), len(batch))

            # 모델 기반 1차 분류
            predictions = self.classifier.classify_many([item.classification_text() for item in batch])

            for item, prediction in zip(batch, predictions):
                # 모델 결과와 규칙 보정 결과를 반영함으로써 최종적인 결과 생성
                classified = self.postprocessor.process(item, prediction)
                results.append(classified)

        print(f"분류 완료: {len(results)}건")
        print("\n최종 분류 기사")
        for idx, classified in enumerate(results, start=1):
            item = classified.item
            source = f" - {item.source}" if item.source else ""
            print(
                f"{idx}. [{classified.rule_decision.final_label}] "
                f"{item.title}{source}"
            )
        print()

        return results
