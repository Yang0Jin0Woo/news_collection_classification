import json

from news_classifier.models import CLASSIFIED_NEWS_COLUMNS, ModelPrediction, NewsItem
from news_classifier.dedup.title_deduplicator import TitleSourceDeduplicator
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.rules.default_rules import DEFAULT_RULE_SET
from news_classifier.storage.csv_store import CsvNewsStore


def test_empty_csv_contains_fixed_headers(tmp_path):
    path = tmp_path / "empty.csv"
    store = CsvNewsStore(str(path))

    store.save([])
    loaded = store.load()

    assert path.exists()
    assert loaded.empty
    assert list(loaded.columns) == CLASSIFIED_NEWS_COLUMNS


def test_csv_preserves_grouped_sources_and_links(tmp_path):
    item = TitleSourceDeduplicator().deduplicate([
        NewsItem(keyword="AI", title="같은 뉴스 제목", source="A", link="https://example.com/1"),
        NewsItem(keyword="AI", title="같은 뉴스 제목", source="A", link="https://example.com/2"),
    ])[0]
    classified = ClassificationPostProcessor(RuleEngine(DEFAULT_RULE_SET)).process(
        item, ModelPrediction("기술개발", 0.8, 0.2)
    )
    store = CsvNewsStore(str(tmp_path / "group.csv"))
    store.save([classified])
    loaded = store.load()
    assert len(loaded) == 1
    assert loaded.iloc[0]["group_article_count"] == 2
    references = json.loads(loaded.iloc[0]["related_articles"])
    assert references == [{"title": "같은 뉴스 제목", "source": "A", "published_at": "", "link": "https://example.com/2"}]
