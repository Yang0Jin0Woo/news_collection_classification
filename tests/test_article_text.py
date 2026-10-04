"""Synthetic extraction regressions, not real classification accuracy evidence."""
import json

import pytest

from news_classifier.collectors.article_text import extract_article_text


BODY = "해당 기관은 지역의 교육과 연구 사업에 대한 구체적인 실행 계획과 지원 내용을 발표했습니다."
SECOND = "사업에 참여하는 기관과 관계자는 추진 일정과 협력 방안을 함께 설명하며 세부 내용을 공개했습니다."
URL = "https://publisher.example/news/123"
TITLE = "지역 사업 계획 발표"


@pytest.mark.parametrize("noise", ["author_info", "gl-logo-container", "fontsize-container", "news_card"])
def test_first_article_can_be_noise_and_real_body_is_selected(noise):
    html = f'<article class="{noise}"><p>관련 기사 목록과 메뉴에 나오는 무관한 설명으로 판단에 사용하면 안 되는 내용.</p></article><main><article id="article-view-content-div"><p>{BODY}</p><p>{SECOND}</p></article></main>'
    result = extract_article_text(html, title=TITLE, page_url=URL)
    assert result.content == BODY + " " + SECOND
    assert result.method == "BODY_CONTAINER_BLOCKS"
    assert result.paragraph_count == 2


def test_br_only_body_is_preferred_over_wrapper_recommendations():
    html = f'<div class="view_body"><article class="author_info">기자 소개</article><section class="news_view">{BODY}<br>{SECOND}<br>{BODY}</section><p>다른 기사에 등장하는 무관한 추천 내용으로 본문에 들어가면 안 되는 설명입니다.</p></div>'
    result = extract_article_text(html, title=TITLE, page_url=URL)
    assert result.content == BODY + " " + SECOND
    assert result.paragraph_count == 2


def test_inline_spans_are_not_lost_and_noise_is_removed():
    html = '<div class="article-body"><div>연구진은 <span>새로운 분석 방법을</span> 활용하여 사업 계획의 근거와 상세한 내용을 발표했습니다.</div><div class="related"><p>관련 기사의 무관한 내용으로 실제 본문에 들어가면 안 되는 충분히 긴 설명입니다.</p></div><div hidden><p>숨겨진 무관한 본문과 광고에 대한 충분히 긴 안내 설명입니다.</p></div></div>'
    result = extract_article_text(html)
    assert "새로운 분석 방법을" in result.content
    assert "무관한" not in result.content


def test_page_wide_div_text_is_not_an_article():
    result = extract_article_text(f'<div>홈페이지 이용 약관과 일반 안내입니다. {BODY}</div>')
    assert not result.content and result.reason == "no_usable_body"


def test_multiple_independent_bodies_are_not_concatenated_or_guessed():
    result = extract_article_text(f'<div class="article-body">{BODY}</div><div class="article-body">{SECOND}</div>')
    assert not result.content and result.reason == "multiple_body_candidates"


def test_duplicate_mobile_body_is_used_only_once():
    result = extract_article_text(f'<div class="article-body">{BODY}</div><div class="article-body">{BODY}</div>')
    assert result.content == BODY


def test_link_list_is_not_accepted_as_a_body():
    result = extract_article_text(f'<div class="article-body"><a href="/other">{BODY}</a></div>')
    assert not result.content


def arc_html(**changes):
    story = {"type": "story", "website_url": "/news/123", "headlines": {"basic": TITLE},
             "content_restrictions": {"content_code": "0"},
             "content_elements": [{"type": "text", "content": BODY}, {"type": "text", "content": SECOND},
                                  {"type": "image", "caption": "사진 설명"}],
             "related_content": {"basic": [{"type": "text", "content": "읽지 말아야 할 관련 기사"}]}}
    story.update(changes)
    return '<div id="article"></div><script>window.Fusion=window.Fusion||{};Fusion.globalContent=' + json.dumps(story) + ';throw new Error("never execute");</script>'


def test_current_public_arc_story_is_decoded_as_json_not_executed():
    result = extract_article_text(arc_html(), title=TITLE, page_url=URL)
    assert result.content == BODY + " " + SECOND
    assert result.method == "ARC_STORY_JSON" and result.paragraph_count == 2
    assert "관련 기사" not in result.content


@pytest.mark.parametrize("changes", [
    {"website_url": "/news/other"}, {"type": "collection"}, {"headlines": {"basic": "다른 기사 제목"}},
    {"content_elements": [{"type": "image", "caption": BODY}]},
])
def test_unrelated_or_non_story_arc_data_is_not_used(changes):
    assert not extract_article_text(arc_html(**changes), title=TITLE, page_url=URL).content


@pytest.mark.parametrize("restriction", [{"content_code": "premium"}, {"content_code": []}, {"restricted": True}, []])
def test_restricted_arc_data_is_not_extracted(restriction):
    result = extract_article_text(arc_html(content_restrictions=restriction), title=TITLE, page_url=URL)
    assert not result.content and result.reason == "restricted_story_data"


def test_json_ld_body_with_wrong_identity_is_not_used():
    data = {"@type": "NewsArticle", "url": URL + "/other", "articleBody": BODY}
    result = extract_article_text('<script type="application/ld+json">' + json.dumps(data) + '</script>', page_url=URL)
    assert not result.content


def test_query_parameters_distinguish_articles():
    data = {"@type": "NewsArticle", "url": "https://publisher.example/view?idxno=2", "articleBody": BODY}
    result = extract_article_text('<script type="application/ld+json">' + json.dumps(data) + '</script>',
                                  page_url="https://publisher.example/view?idxno=1")
    assert not result.content


@pytest.mark.parametrize("script", ['Fusion.globalContent={not json}',
                                      'Fusion.globalContent=JSON.parse("not executed")',
                                      'Fusion.globalContent=null',
                                      'Fusion.globalContent={"type":"story","headlines":null}'])
def test_malformed_or_executable_story_data_fails_closed(script):
    assert not extract_article_text('<script>' + script + '</script>', title=TITLE, page_url=URL).content


@pytest.mark.parametrize("html", ['<article>' + BODY + '</article>',
                                  '<article><div>' + BODY + '</div></article>',
                                  '<main><p>' + BODY + '</p></main>'])
def test_non_keyword_specific_semantic_body_formats(html):
    assert extract_article_text(html, title="서로 다른 분야의 기사").content == BODY
