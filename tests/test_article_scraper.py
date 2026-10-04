from types import SimpleNamespace

from news_classifier.collectors.article_scraper import ArticleScraper
from news_classifier.models import NewsItem, NewsReference


def test_enrichment_keeps_grouped_original_references():
    reference = NewsReference("related", "B", "2026-10-04", "https://example.com/2")
    item = NewsItem(keyword="AI", title="representative", link="https://example.com/1", related_articles=(reference,))
    client = SimpleNamespace(get=lambda _: SimpleNamespace(text="<p>A sufficiently long paragraph for an article content extraction test.</p>"))
    enriched = ArticleScraper(client).enrich(item)
    assert enriched.content
    assert enriched.related_articles == (reference,)
    assert enriched.group_article_count == 2
