from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from news_classifier.classifiers.base import NewsClassifier
from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.collectors.base import (
    CollectionError,
    NetworkCollectionError,
    NewsCollector,
)
from news_classifier.collectors.article_scraper import ArticleScraper
from news_classifier.dedup.title_deduplicator import TitleSourceDeduplicator
from news_classifier.models import (
    ClassifiedNews,
    NewsItem,
    PipelineError,
    PipelineResult,
    PipelineStatistics,
    PipelineStatus,
)
from news_classifier.utils.text import strip_source_suffix

logger = logging.getLogger(__name__)


def _chunks(items: list[NewsItem], size: int) -> list[list[NewsItem]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def _statistics(
    requested_limit: int,
    collected_count: int,
    deduplicated_count: int,
    results: list[ClassifiedNews],
) -> PipelineStatistics:
    return PipelineStatistics(
        requested_limit=requested_limit,
        collected_count=collected_count,
        deduplicated_count=deduplicated_count,
        classified_count=len(results),
        rule_applied_count=sum(
            1 for row in results if row.decision_source == "RULE"
        ),
        review_required_count=sum(1 for row in results if row.review_required),
    )


@dataclass
class NewsPipeline:
    collector: NewsCollector
    classifier: NewsClassifier
    postprocessor: ClassificationPostProcessor
    deduplicator: TitleSourceDeduplicator
    article_scraper: ArticleScraper | None = None

    # 뉴스 수집부터 분류 결과 생성까지의 전체 파이프라인
    def run(
        self,
        keyword: str,
        limit: int = 10,
        enrich_content: bool = False,
    ) -> PipelineResult:
        logger.info("news pipeline started keyword=%s limit=%s", keyword, limit)
        print("뉴스 수집 중...")

        # 1. 키워드 기반 뉴스 수집
        try:
            collected_news = self.collector.fetch(keyword=keyword, limit=limit)
        except NetworkCollectionError as exc:
            logger.error("network collection failed: %s", exc)
            return PipelineResult(
                status=PipelineStatus.NETWORK_ERROR,
                errors=[
                    PipelineError(
                        stage="COLLECTION",
                        code="NETWORK_FAILURE",
                        message=str(exc),
                    )
                ],
                statistics=PipelineStatistics(requested_limit=limit),
            )
        except CollectionError as exc:
            logger.error("news collection failed: %s", exc)
            return PipelineResult(
                status=PipelineStatus.COLLECTION_ERROR,
                errors=[
                    PipelineError(
                        stage="COLLECTION",
                        code="INVALID_FEED",
                        message=str(exc),
                    )
                ],
                statistics=PipelineStatistics(requested_limit=limit),
            )
        except Exception as exc:
            logger.exception("unexpected news collection failure")
            return PipelineResult(
                status=PipelineStatus.COLLECTION_ERROR,
                errors=[
                    PipelineError(
                        stage="COLLECTION",
                        code="UNEXPECTED_COLLECTION_FAILURE",
                        message=str(exc),
                    )
                ],
                statistics=PipelineStatistics(requested_limit=limit),
            )

        news_list = [
            replace(item, title=strip_source_suffix(item.title, item.source))
            for item in collected_news
        ]
        collected_count = len(news_list)
        logger.info("fetched=%s", collected_count)
        print(f"수집 완료: {collected_count}건")

        # 2. 완전 중복과 확실한 동일 사건 묶기, 대표 기사만 후속 분류
        news_list = self.deduplicator.deduplicate(news_list)
        deduplicated_count = len(news_list)
        logger.info("deduplicated=%s", deduplicated_count)
        print(f"중복 묶기 완료: 수집 {collected_count}건 → 대표 기사 {deduplicated_count}건")

        if not news_list:
            print("검색 결과 없음")
            return PipelineResult(
                status=PipelineStatus.NO_RESULTS,
                statistics=_statistics(
                    limit,
                    collected_count,
                    deduplicated_count,
                    [],
                ),
            )

        # 3. 기사 페이지 본문을 추가 수집해 분류 입력 데이터 보강
        if enrich_content and self.article_scraper is not None:
            print("기사 본문 보강 중...")
            news_list = self.article_scraper.enrich_many(news_list)
            logger.info("content enrichment finished")
            print("기사 본문 보강 완료")

        # 4. 모델 분류 및 규칙 기반 후처리
        results: list[ClassifiedNews] = []
        batch_size = max(1, getattr(self.classifier, "batch_size", 1))
        batches = _chunks(news_list, batch_size)
        print("분류 모델 로딩 및 뉴스 분류 시작...")

        for batch_idx, batch in enumerate(batches, start=1):
            start = (batch_idx - 1) * batch_size + 1
            end = min(start + len(batch) - 1, len(news_list))
            print(f"뉴스 분류 중: {start}-{end}/{len(news_list)}")
            logger.info("classifying batch %s/%s size=%s", batch_idx, len(batches), len(batch))

            # 모델 기반 1차 분류
            try:
                predictions = self.classifier.classify_many(
                    [item.classification_text() for item in batch]
                )
            except Exception as exc:
                logger.exception("model classification failed")
                return PipelineResult(
                    status=PipelineStatus.MODEL_ERROR,
                    results=results.copy(),
                    errors=[
                        PipelineError(
                            stage="CLASSIFICATION",
                            code="MODEL_INFERENCE_FAILURE",
                            message=str(exc),
                        )
                    ],
                    statistics=_statistics(
                        limit,
                        collected_count,
                        deduplicated_count,
                        results,
                    ),
                )

            if len(predictions) != len(batch):
                message = (
                    "분류 입력과 모델 출력 개수가 다릅니다: "
                    f"input={len(batch)}, output={len(predictions)}"
                )
                logger.error(message)
                return PipelineResult(
                    status=PipelineStatus.MODEL_ERROR,
                    results=results.copy(),
                    errors=[
                        PipelineError(
                            stage="CLASSIFICATION",
                            code="PREDICTION_COUNT_MISMATCH",
                            message=message,
                        )
                    ],
                    statistics=_statistics(
                        limit,
                        collected_count,
                        deduplicated_count,
                        results,
                    ),
                )

            failed_count = sum(
                1 for prediction in predictions if prediction.label == "분류실패"
            )
            if failed_count:
                message = f"모델이 {failed_count}건의 분류 결과를 생성하지 못했습니다."
                logger.error(message)
                return PipelineResult(
                    status=PipelineStatus.MODEL_ERROR,
                    results=results.copy(),
                    errors=[
                        PipelineError(
                            stage="CLASSIFICATION",
                            code="MODEL_PREDICTION_FAILURE",
                            message=message,
                        )
                    ],
                    statistics=_statistics(
                        limit,
                        collected_count,
                        deduplicated_count,
                        results,
                    ),
                )

            for item, prediction in zip(batch, predictions, strict=True):
                # 모델 결과와 규칙 보정 결과를 반영함으로써 최종적인 결과 생성
                classified = self.postprocessor.process(item, prediction)
                results.append(classified)

        print(f"분류 완료: {len(results)}건")
        print("\n최종 분류 기사")
        for idx, classified in enumerate(results, start=1):
            item = classified.item
            source = f" - {item.source}" if item.source else ""
            group = f" [동일 사건 {item.group_article_count}건]" if item.related_articles else ""
            print(
                f"{idx}. [{classified.rule_decision.final_label}] "
                f"{item.title}{source}{group}"
            )
        print()

        return PipelineResult(
            status=PipelineStatus.SUCCESS,
            results=results,
            statistics=_statistics(
                limit,
                collected_count,
                deduplicated_count,
                results,
            ),
        )
