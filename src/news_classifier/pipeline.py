from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Literal, TypeVar

from news_classifier.classifiers.base import NewsClassifier
from news_classifier.classifiers.input_text import has_changed_article_input
from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.collectors.base import (
    CollectionError,
    NetworkCollectionError,
    NewsCollector,
)
from news_classifier.collectors.article_scraper import ArticleEnrichmentResult, ArticleScraper
from news_classifier.dedup.title_deduplicator import TitleSourceDeduplicator
from news_classifier.models import (
    ClassifiedNews,
    ModelPrediction,
    NewsItem,
    PipelineError,
    PipelineResult,
    PipelineStatistics,
    PipelineStatus,
    ReviewReclassification,
)
from news_classifier.utils.text import clean_text, strip_source_suffix

logger = logging.getLogger(__name__)


_ChunkItem = TypeVar("_ChunkItem")


def _chunks(items: list[_ChunkItem], size: int) -> list[list[_ChunkItem]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


@dataclass(frozen=True)
class _EnrichmentDetails:
    status: str = ""
    http_status: int | None = None
    resolved_url: str = ""
    resolution_status: str = ""
    extraction_diagnostics: dict = field(default_factory=dict)

    @classmethod
    def from_outcome(cls, outcome: ArticleEnrichmentResult) -> _EnrichmentDetails:
        return cls(
            status=outcome.status, http_status=outcome.http_status,
            resolved_url=getattr(outcome, "resolved_url", ""),
            resolution_status=getattr(outcome, "resolution_status", ""),
            extraction_diagnostics=getattr(outcome, "extraction_diagnostics", {}),
        )


@dataclass(frozen=True)
class _ReviewRetry:
    index: int
    item: NewsItem
    initial: ClassifiedNews
    enrichment: _EnrichmentDetails


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


def _failure_result(
    status: PipelineStatus,
    error: PipelineError,
    limit: int,
    collected_count: int = 0,
    deduplicated_count: int = 0,
    results: list[ClassifiedNews] | None = None,
) -> PipelineResult:
    partial_results = results.copy() if results is not None else []
    return PipelineResult(
        status=status,
        results=partial_results,
        errors=[error],
        statistics=_statistics(limit, collected_count, deduplicated_count, partial_results),
    )


@dataclass
class NewsPipeline:
    collector: NewsCollector
    classifier: NewsClassifier
    postprocessor: ClassificationPostProcessor
    deduplicator: TitleSourceDeduplicator
    article_scraper: ArticleScraper | None = None
    review_enrichment_enabled: bool = True

    def _classify_batch(
        self, items: list[NewsItem],
        stage: Literal["CLASSIFICATION", "RECLASSIFICATION"],
    ) -> list[ModelPrediction] | PipelineError:
        is_retry = stage == "RECLASSIFICATION"
        try:
            predictions = self.classifier.classify_many([
                item.classification_text() for item in items
            ])
        except Exception as exc:
            logger.exception(
                "review reclassification failed" if is_retry else "model classification failed"
            )
            return PipelineError(stage, "MODEL_INFERENCE_FAILURE", str(exc))

        if len(predictions) != len(items):
            prefix = "재판단 입력" if is_retry else "분류 입력"
            message = (
                f"{prefix}과 모델 출력 개수가 다릅니다: "
                f"input={len(items)}, output={len(predictions)}"
            )
            if not is_retry:
                logger.error(message)
            return PipelineError(stage, "PREDICTION_COUNT_MISMATCH", message)

        failed_count = sum(prediction.label == "분류실패" for prediction in predictions)
        if failed_count:
            message = (
                "모델이 본문 보강 후 재판단 결과를 생성하지 못했습니다."
                if is_retry else f"모델이 {failed_count}건의 분류 결과를 생성하지 못했습니다."
            )
            if not is_retry:
                logger.error(message)
            return PipelineError(stage, "MODEL_PREDICTION_FAILURE", message)
        return predictions

    def _reclassify_reviews(
        self, results: list[ClassifiedNews], *, already_enriched: bool,
        enrichment_outcomes: dict[int, _EnrichmentDetails] | None = None,
    ) -> PipelineError | None:
        """검색어 분기 없이 검토 기사만 본문 보강. 입력이 달라질 때 최대 1회 재판단."""
        retry_items: list[_ReviewRetry] = []

        def metadata(
            row: ClassifiedNews, status: str, enrichment: _EnrichmentDetails,
        ) -> ReviewReclassification:
            return ReviewReclassification(
                status=status,
                initial_model_category=row.model_prediction.label,
                initial_model_score=row.model_prediction.score,
                initial_score_margin=row.model_prediction.margin,
                initial_final_category=row.rule_decision.final_label,
                initial_rule_reason=row.rule_decision.rule_reason,
                enrichment_status=enrichment.status or status,
                http_status=enrichment.http_status,
                resolved_url=enrichment.resolved_url,
                resolution_status=enrichment.resolution_status,
                extraction_diagnostics=enrichment.extraction_diagnostics or {},
            )

        for index, row in enumerate(results):
            if row.rule_decision.final_label != "검토필요":
                continue
            enrichment = _EnrichmentDetails()
            if not self.review_enrichment_enabled:
                status = "DISABLED"
            elif already_enriched:
                status = "ALREADY_ENRICHED"
                enrichment = (enrichment_outcomes or {}).get(index, _EnrichmentDetails(status=status))
            elif clean_text(row.item.content):
                status = "CONTENT_PRESENT"
            elif self.article_scraper is None:
                status = "NO_SCRAPER"
            elif not row.item.link:
                status = "NO_LINK"
            else:
                try:
                    # 보강 대상의 식별 정보와 묶음은 유지하고 새 본문만 사용.
                    enrich = getattr(self.article_scraper, "enrich_with_diagnostics", None)
                    if callable(enrich):
                        outcome = enrich(row.item)
                        enriched = outcome.item
                        enrichment = _EnrichmentDetails.from_outcome(outcome)
                    else:
                        enriched = self.article_scraper.enrich(row.item)
                        enrichment = _EnrichmentDetails(
                            status="BODY_EXTRACTED" if clean_text(enriched.content) else "NO_CONTENT",
                        )
                    item = replace(row.item, content=enriched.content)
                    max_length = getattr(self.classifier, "max_sequence_length", 1200)
                    if not clean_text(item.content):
                        status = enrichment.status or "NO_CONTENT"
                    elif not has_changed_article_input(row.item, item, max_length):
                        status = "UNCHANGED_INPUT"
                    else:
                        retry_items.append(_ReviewRetry(index, item, row, enrichment))
                        status = "PENDING"
                except Exception as exc:
                    logger.warning("review content enrichment failed: %s", exc)
                    status = "FETCH_FAILED"
            results[index] = replace(row, review_reclassification=metadata(
                row, status, enrichment,
            ))

        outcomes = Counter(
            row.review_reclassification.enrichment_status
            for row in results if row.review_reclassification is not None
        )
        if outcomes:
            logger.info("review enrichment outcomes: %s", dict(sorted(outcomes.items())))
            descriptions = {
                "BODY_EXTRACTED": "본문 확보", "TIMEOUT": "요청 시간 초과",
                "BLOCKED_HTTP": "접근 차단", "HTTP_ERROR": "HTTP 오류",
                "REQUEST_FAILED": "요청 실패", "FETCH_FAILED": "보강 처리 실패",
                "GOOGLE_NEWS_WRAPPER": "Google 뉴스 중계 페이지",
                "RSS_WRAPPER": "RSS 문서", "CONSENT_OR_BLOCK_PAGE": "동의 또는 차단 페이지",
                "NO_USABLE_PARAGRAPHS": "본문 문단 없음", "NO_CONTENT": "본문 없음",
                "NO_LINK": "기사 링크 없음", "NO_SCRAPER": "본문 수집기 없음",
                "CONTENT_PRESENT": "기존 본문 유지", "DISABLED": "본문 보강 비활성",
                "ALREADY_ENRICHED": "이미 본문 보강 시도",
                "SOURCE_URL_UNRESOLVED": "원문 URL 조회 실패", "UNSAFE_URL": "안전하지 않은 URL 제외",
                "REDIRECT_LIMIT": "리디렉션 한도 초과",
            }
            print("검토 기사 본문 보강 결과: " + ", ".join(
                f"{descriptions.get(status, status)} {count}건"
                for status, count in sorted(outcomes.items())
            ))

        if not retry_items:
            return None
        print(f"검토 기사 본문 보강 후 재판단: {len(retry_items)}건")
        batch_size = max(1, getattr(self.classifier, "batch_size", 1))
        for batch in _chunks(retry_items, batch_size):
            predictions = self._classify_batch(
                [retry.item for retry in batch], "RECLASSIFICATION",
            )
            if isinstance(predictions, PipelineError):
                return predictions
            for retry, prediction in zip(batch, predictions, strict=True):
                revised = self.postprocessor.process(retry.item, prediction)
                results[retry.index] = replace(
                    revised, created_at=retry.initial.created_at,
                    review_reclassification=metadata(retry.initial, "RECLASSIFIED", retry.enrichment),
                )
        resolved_count = sum(
            not results[retry.index].review_required for retry in retry_items
        )
        print(f"재판단 완료: {len(retry_items)}건, 검토 해소: {resolved_count}건")
        return None

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
            return _failure_result(
                PipelineStatus.NETWORK_ERROR,
                PipelineError("COLLECTION", "NETWORK_FAILURE", str(exc)),
                limit,
            )
        except CollectionError as exc:
            logger.error("news collection failed: %s", exc)
            return _failure_result(
                PipelineStatus.COLLECTION_ERROR,
                PipelineError("COLLECTION", "INVALID_FEED", str(exc)),
                limit,
            )
        except Exception as exc:
            logger.exception("unexpected news collection failure")
            return _failure_result(
                PipelineStatus.COLLECTION_ERROR,
                PipelineError("COLLECTION", "UNEXPECTED_COLLECTION_FAILURE", str(exc)),
                limit,
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
        enrichment_outcomes: dict[int, _EnrichmentDetails] = {}
        if enrich_content and self.article_scraper is not None:
            print("기사 본문 보강 중...")
            enrich_many = getattr(self.article_scraper, "enrich_many_with_diagnostics", None)
            if callable(enrich_many):
                outcomes = enrich_many(news_list)
                news_list = [
                    replace(original, content=outcome.item.content)
                    for original, outcome in zip(news_list, outcomes, strict=True)
                ]
                enrichment_outcomes = {
                    index: _EnrichmentDetails.from_outcome(outcome)
                    for index, outcome in enumerate(outcomes)
                }
            else:
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
            predictions = self._classify_batch(batch, "CLASSIFICATION")
            if isinstance(predictions, PipelineError):
                return _failure_result(
                    PipelineStatus.MODEL_ERROR, predictions,
                    limit, collected_count, deduplicated_count, results,
                )

            for item, prediction in zip(batch, predictions, strict=True):
                # 모델 결과와 규칙 보정 결과를 반영함으로써 최종적인 결과 생성
                classified = self.postprocessor.process(item, prediction)
                results.append(classified)

        retry_error = self._reclassify_reviews(
            results, already_enriched=enrich_content and self.article_scraper is not None,
            enrichment_outcomes=enrichment_outcomes,
        )
        if retry_error is not None:
            return _failure_result(
                PipelineStatus.MODEL_ERROR, retry_error,
                limit, collected_count, deduplicated_count, results,
            )

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
