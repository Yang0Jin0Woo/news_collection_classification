import pytest

from news_classifier.classifiers.base import NewsClassifier
from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.collectors.base import CollectionError, NetworkCollectionError, NewsCollector
from news_classifier.dedup.title_deduplicator import TitleSourceDeduplicator
from news_classifier.models import ModelPrediction, NewsItem, PipelineStatus
from news_classifier.pipeline import NewsPipeline
from news_classifier.rules.default_rules import DEFAULT_RULE_SET


class EmptyCollector(NewsCollector):
    def fetch(self, keyword: str, limit: int):
        return []


class NetworkFailureCollector(NewsCollector):
    def fetch(self, keyword: str, limit: int):
        raise NetworkCollectionError("네트워크 연결 실패")


class OneItemCollector(NewsCollector):
    def fetch(self, keyword: str, limit: int):
        return [NewsItem(keyword=keyword, title="AI 뉴스", link="1")]


class TwoItemCollector(NewsCollector):
    def fetch(self, keyword: str, limit: int):
        return [
            NewsItem(keyword=keyword, title="첫 번째 AI 뉴스", link="1"),
            NewsItem(keyword=keyword, title="두 번째 AI 뉴스", link="2"),
        ]


class SuccessfulClassifier(NewsClassifier):
    def classify(self, text: str):
        return ModelPrediction("기술개발", 0.8, 0.2)


class MissingPredictionClassifier(SuccessfulClassifier):
    def classify_many(self, texts: list[str]):
        return []


class LateMissingPredictionClassifier(SuccessfulClassifier):
    batch_size = 1

    def __init__(self):
        self.call_count = 0

    def classify_many(self, texts: list[str]):
        self.call_count += 1
        if self.call_count == 2:
            return []
        return [self.classify(text) for text in texts]


class FailedPredictionClassifier(SuccessfulClassifier):
    def classify_many(self, texts: list[str]):
        return [ModelPrediction.failed() for _ in texts]


class ExceptionClassifier(SuccessfulClassifier):
    def classify_many(self, texts: list[str]):
        raise RuntimeError("모델 로드 실패")


def pipeline(collector: NewsCollector, classifier: NewsClassifier) -> NewsPipeline:
    return NewsPipeline(
        collector=collector,
        classifier=classifier,
        postprocessor=ClassificationPostProcessor(RuleEngine(DEFAULT_RULE_SET)),
        deduplicator=TitleSourceDeduplicator(),
    )


def test_pipeline_returns_no_results_as_non_failure():
    result = pipeline(EmptyCollector(), SuccessfulClassifier()).run("AI")

    assert result.status == PipelineStatus.NO_RESULTS
    assert result.succeeded is True
    assert result.results == []
    assert result.errors == []


def test_pipeline_returns_network_error():
    result = pipeline(NetworkFailureCollector(), SuccessfulClassifier()).run("AI")

    assert result.status == PipelineStatus.NETWORK_ERROR
    assert result.succeeded is False
    assert result.errors[0].code == "NETWORK_FAILURE"


def test_pipeline_fails_when_prediction_count_does_not_match_input():
    result = pipeline(OneItemCollector(), MissingPredictionClassifier()).run("AI")

    assert result.status == PipelineStatus.MODEL_ERROR
    assert result.results == []
    assert result.errors[0].code == "PREDICTION_COUNT_MISMATCH"


def test_pipeline_reports_partial_progress_instead_of_silently_dropping_items():
    result = pipeline(
        TwoItemCollector(),
        LateMissingPredictionClassifier(),
    ).run("AI")

    assert result.status == PipelineStatus.MODEL_ERROR
    assert len(result.results) == 1
    assert result.statistics.classified_count == 1
    assert result.statistics.deduplicated_count == 2
    assert result.errors[0].code == "PREDICTION_COUNT_MISMATCH"


def test_pipeline_fails_when_model_returns_failed_prediction():
    result = pipeline(OneItemCollector(), FailedPredictionClassifier()).run("AI")

    assert result.status == PipelineStatus.MODEL_ERROR
    assert result.results == []
    assert result.errors[0].code == "MODEL_PREDICTION_FAILURE"


def test_pipeline_fails_when_classifier_raises():
    result = pipeline(OneItemCollector(), ExceptionClassifier()).run("AI")

    assert result.status == PipelineStatus.MODEL_ERROR
    assert result.results == []
    assert result.errors[0].code == "MODEL_INFERENCE_FAILURE"

@pytest.mark.parametrize("stage", ["CLASSIFICATION", "RECLASSIFICATION"])
@pytest.mark.parametrize("classifier,code", [
    (MissingPredictionClassifier, "PREDICTION_COUNT_MISMATCH"),
    (FailedPredictionClassifier, "MODEL_PREDICTION_FAILURE"),
    (ExceptionClassifier, "MODEL_INFERENCE_FAILURE"),
])
def test_common_batch_validation_preserves_stage_and_message(stage, classifier, code):
    from news_classifier.models import PipelineError

    result = pipeline(OneItemCollector(), classifier())._classify_batch(
        [NewsItem("검색어", "지역 소식", "1")], stage,
    )
    assert isinstance(result, PipelineError)
    assert result.stage == stage
    assert result.code == code
    if code == "PREDICTION_COUNT_MISMATCH":
        prefix = "재판단 입력" if stage == "RECLASSIFICATION" else "분류 입력"
        assert result.message == f"{prefix}과 모델 출력 개수가 다릅니다: input=1, output=0"
    elif code == "MODEL_PREDICTION_FAILURE":
        assert result.message == (
            "모델이 본문 보강 후 재판단 결과를 생성하지 못했습니다."
            if stage == "RECLASSIFICATION" else "모델이 1건의 분류 결과를 생성하지 못했습니다."
        )
    else:
        assert result.message == "모델 로드 실패"


@pytest.mark.parametrize("exception,status,code", [
    (NetworkCollectionError("실패"), PipelineStatus.NETWORK_ERROR, "NETWORK_FAILURE"),
    (CollectionError("실패"), PipelineStatus.COLLECTION_ERROR, "INVALID_FEED"),
    (RuntimeError("실패"), PipelineStatus.COLLECTION_ERROR, "UNEXPECTED_COLLECTION_FAILURE"),
])
def test_common_failure_result_preserves_collection_errors_and_statistics(exception, status, code):
    class Collector(NewsCollector):
        def fetch(self, keyword, limit):
            raise exception

    result = pipeline(Collector(), SuccessfulClassifier()).run("검색어", limit=7)
    assert result.status == status
    assert result.errors[0].stage == "COLLECTION"
    assert result.errors[0].code == code
    assert result.errors[0].message == "실패"
    assert result.results == []
    assert result.statistics.requested_limit == 7
    assert result.statistics.collected_count == 0
    assert result.statistics.deduplicated_count == 0
    assert result.statistics.classified_count == 0
    assert result.statistics.rule_applied_count == 0
    assert result.statistics.review_required_count == 0


def test_failure_result_snapshots_partial_progress():
    from news_classifier.models import PipelineError
    from news_classifier.pipeline import _failure_result

    processor = ClassificationPostProcessor(RuleEngine(DEFAULT_RULE_SET))
    partial = [processor.process(NewsItem("검색어", "지역 소식", "1"), ModelPrediction("사회", .8, .2))]
    error = PipelineError("CLASSIFICATION", "PREDICTION_COUNT_MISMATCH", "실패")
    result = _failure_result(PipelineStatus.MODEL_ERROR, error, 7, 2, 2, partial)
    partial.clear()

    assert len(result.results) == 1
    assert result.errors == [error]
    assert result.statistics.requested_limit == 7
    assert result.statistics.collected_count == 2
    assert result.statistics.deduplicated_count == 2
    assert result.statistics.classified_count == 1
