from news_classifier.classifiers.base import NewsClassifier
from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.collectors.base import NetworkCollectionError, NewsCollector
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
