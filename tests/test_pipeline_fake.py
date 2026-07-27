from news_classifier.classifiers.base import NewsClassifier
from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.collectors.base import NewsCollector
from news_classifier.dedup.title_deduplicator import TitleSourceDeduplicator
from news_classifier.models import ModelPrediction, NewsItem
from news_classifier.models import PipelineStatus
from news_classifier.pipeline import NewsPipeline
from news_classifier.rules.default_rules import RULES


class FakeCollector(NewsCollector):
    def fetch(self, keyword: str, limit: int):
        return [
            NewsItem(keyword=keyword, title="AI 신제품 출시", source="A", link="1", description="솔루션 출시"),
            NewsItem(keyword=keyword, title="AI 신제품 출시", source="A", link="2", description="솔루션 출시"),
        ]


class FakeClassifier(NewsClassifier):
    def classify(self, text: str):
        return ModelPrediction("제품/서비스", 0.80, 0.20, ["제품/서비스"], [0.80])


class FakeBatchClassifier(NewsClassifier):
    batch_size = 2

    def __init__(self):
        self.batch_lengths = []

    def classify(self, text: str):
        return ModelPrediction("제품/서비스", 0.80, 0.20, ["제품/서비스"], [0.80])

    def classify_many(self, texts: list[str]):
        self.batch_lengths.append(len(texts))
        return [self.classify(text) for text in texts]


class ThreeItemCollector(NewsCollector):
    def fetch(self, keyword: str, limit: int):
        return [
            NewsItem(keyword=keyword, title="AI 신제품 출시", source="A", link="1", description="솔루션 출시"),
            NewsItem(keyword=keyword, title="반도체 투자 확대", source="B", link="2", description="시장 성장"),
            NewsItem(keyword=keyword, title="정부 정책 발표", source="C", link="3", description="지원 정책"),
        ]


def test_pipeline_runs_with_fake_dependencies():
    pipeline = NewsPipeline(
        collector=FakeCollector(),
        classifier=FakeClassifier(),
        postprocessor=ClassificationPostProcessor(RuleEngine(RULES)),
        deduplicator=TitleSourceDeduplicator(),
    )
    result = pipeline.run("AI", limit=10)
    assert result.status == PipelineStatus.SUCCESS
    assert len(result.results) == 1
    assert result.results[0].rule_decision.final_label == "제품/서비스"
    assert result.errors == []
    assert result.statistics.collected_count == 2
    assert result.statistics.deduplicated_count == 1
    assert result.statistics.classified_count == 1


def test_pipeline_uses_batch_classifier_and_prints_progress(capsys):
    classifier = FakeBatchClassifier()
    pipeline = NewsPipeline(
        collector=ThreeItemCollector(),
        classifier=classifier,
        postprocessor=ClassificationPostProcessor(RuleEngine(RULES)),
        deduplicator=TitleSourceDeduplicator(),
    )

    result = pipeline.run("AI", limit=10)

    assert result.status == PipelineStatus.SUCCESS
    assert len(result.results) == 3
    assert classifier.batch_lengths == [2, 1]
    output = capsys.readouterr().out
    assert "뉴스 수집 중..." in output
    assert "뉴스 분류 중: 1-2/3" in output
    assert "뉴스 분류 중: 3-3/3" in output
    assert "분류 완료: 3건" in output
    assert "최종 분류 기사" in output
    assert "1. [제품/서비스] AI 신제품 출시 - A" in output
