from news_classifier.classifiers.base import NewsClassifier
from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.collectors.base import NewsCollector
from news_classifier.dedup.title_deduplicator import TitleSourceDeduplicator
from news_classifier.models import ClassifiedNews, ModelPrediction, NewsItem, RuleDecision
from news_classifier.pipeline import NewsPipeline
from news_classifier.rules.default_rules import RULES
from news_classifier.utils.text import strip_source_suffix


class SourceSuffixCollector(NewsCollector):
    def fetch(self, keyword: str, limit: int):
        return [
            NewsItem(
                keyword=keyword,
                title="디스플레이 업계, 반도체 신사업 추진 - 디일렉",
                source="디일렉",
                link="https://example.com/news",
            )
        ]


class OneShotClassifier(NewsClassifier):
    batch_size = 4

    def classify(self, text: str):
        return ModelPrediction("기술개발", 0.8, 0.2, ["기술개발"], [0.8])

    def classify_many(self, texts: list[str]):
        return [self.classify(text) for text in texts]


def test_strip_source_suffix_removes_trailing_source():
    assert strip_source_suffix("디스플레이 업계, 반도체 신사업 추진 - 디일렉", "디일렉") == (
        "디스플레이 업계, 반도체 신사업 추진"
    )


def test_strip_source_suffix_removes_repeated_trailing_source():
    assert strip_source_suffix("반도체 이익 국민 배당 - 조선일보 - 조선일보", "조선일보") == (
        "반도체 이익 국민 배당"
    )


def test_strip_source_suffix_keeps_title_without_matching_source():
    assert strip_source_suffix("반도체 수출통제는 주요의제 아니었다", "연합뉴스") == (
        "반도체 수출통제는 주요의제 아니었다"
    )


def test_to_row_uses_cleaned_title():
    item = NewsItem(
        keyword="반도체",
        title=strip_source_suffix("반도체 수출통제는 주요의제 아니었다 - 연합뉴스", "연합뉴스"),
        source="연합뉴스",
        link="https://example.com/news",
    )
    row = ClassifiedNews(
        item=item,
        model_prediction=ModelPrediction("정책/규제", 0.8, 0.2),
        rule_decision=RuleDecision("정책/규제", False, "", "정책/규제", 0, {}),
        model_confidence_level="높음",
    ).to_row()

    assert row["title"] == "반도체 수출통제는 주요의제 아니었다"
    assert row["source"] == "연합뉴스"


def test_pipeline_cleans_title_before_output_and_storage(capsys):
    pipeline = NewsPipeline(
        collector=SourceSuffixCollector(),
        classifier=OneShotClassifier(),
        postprocessor=ClassificationPostProcessor(RuleEngine(RULES)),
        deduplicator=TitleSourceDeduplicator(),
    )

    result = pipeline.run("반도체", limit=10)
    output = capsys.readouterr().out

    assert result[0].item.title == "디스플레이 업계, 반도체 신사업 추진"
    assert result[0].to_row()["title"] == "디스플레이 업계, 반도체 신사업 추진"
    assert "디스플레이 업계, 반도체 신사업 추진 - 디일렉" in output
    assert "디일렉 - 디일렉" not in output
