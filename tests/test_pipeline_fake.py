from news_classifier.classifiers.base import NewsClassifier
from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.collectors.base import NewsCollector
from news_classifier.dedup.title_deduplicator import TitleSourceDeduplicator
from news_classifier.models import ModelPrediction, NewsItem
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


def test_pipeline_runs_with_fake_dependencies():
    pipeline = NewsPipeline(
        collector=FakeCollector(),
        classifier=FakeClassifier(),
        postprocessor=ClassificationPostProcessor(RuleEngine(RULES)),
        deduplicator=TitleSourceDeduplicator(),
    )
    result = pipeline.run("AI", limit=10)
    assert len(result) == 1
    assert result[0].rule_decision.final_label == "제품/서비스"
