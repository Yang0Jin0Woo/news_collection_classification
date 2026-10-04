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


def test_enrichment_prefers_article_and_removes_page_noise_and_duplicate_paragraphs():
    paragraph = "식품 신제품 판매를 시작하고 이용 가격과 제공 서비스를 발표했습니다."
    html = f"<nav><p>메뉴와 회사 소개 및 이용 약관을 확인하는 페이지입니다.</p></nav><p>무관한 외부 설명입니다.</p><article><p>{paragraph}</p><p>{paragraph}</p><aside><p>관련 광고와 추천 상품의 소개 내용입니다.</p></aside></article>"
    client = SimpleNamespace(get=lambda _: SimpleNamespace(text=html))
    item = NewsItem(keyword="식품", title="신제품", link="https://example.com")
    assert ArticleScraper(client).enrich(item).content == paragraph


def test_empty_page_keeps_preexisting_content():
    client = SimpleNamespace(get=lambda _: SimpleNamespace(text="<p>짧음</p>"))
    item = NewsItem(keyword="", title="제목", link="https://example.com", content="이미 확보한 본문")
    assert ArticleScraper(client).enrich(item) == item
