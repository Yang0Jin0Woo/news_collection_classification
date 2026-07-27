import pytest

from news_classifier.collectors.base import FeedParseError, NetworkCollectionError
from news_classifier.collectors.google_rss import GoogleNewsRssCollector


class FakeResponse:
    def __init__(self, text: str):
        self.text = text


class FakeHttpClient:
    def __init__(self, response):
        self.response = response

    def get(self, url: str):
        return self.response


def test_collector_distinguishes_network_failure_from_no_results():
    collector = GoogleNewsRssCollector(FakeHttpClient(None))

    with pytest.raises(NetworkCollectionError):
        collector.fetch("AI")


def test_collector_returns_empty_list_for_valid_feed_without_items():
    collector = GoogleNewsRssCollector(
        FakeHttpClient(FakeResponse("<rss><channel></channel></rss>"))
    )

    assert collector.fetch("검색되지않는키워드") == []


def test_collector_rejects_malformed_feed():
    collector = GoogleNewsRssCollector(FakeHttpClient(FakeResponse("<rss>")))

    with pytest.raises(FeedParseError):
        collector.fetch("AI")


def test_collector_rejects_feed_without_channel():
    collector = GoogleNewsRssCollector(FakeHttpClient(FakeResponse("<rss></rss>")))

    with pytest.raises(FeedParseError):
        collector.fetch("AI")
