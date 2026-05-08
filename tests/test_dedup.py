from news_classifier.dedup.title_deduplicator import TitleSourceDeduplicator
from news_classifier.models import NewsItem


def test_title_source_deduplicator():
    items = [
        NewsItem(keyword="AI", title="같은 제목", source="A", link="1"),
        NewsItem(keyword="AI", title="같은 제목", source="A", link="2"),
        NewsItem(keyword="AI", title="같은 제목", source="B", link="3"),
    ]
    result = TitleSourceDeduplicator().deduplicate(items)
    assert len(result) == 2
