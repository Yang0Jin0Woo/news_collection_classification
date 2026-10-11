from dataclasses import replace

import pytest

from news_classifier.classifiers.input_text import (
    has_changed_article_input,
    prepare_model_input,
)
from news_classifier.models import NewsItem


@pytest.mark.parametrize("text,limit,expected", [
    (" 첫째\n 둘째 &amp; 셋째 ", 1200, "첫째 둘째 & 셋째"),
    (" A   B &nbsp; C ", 3, "A B"),
    ("&nbsp;\n", 1200, ""),
    ("기사 내용", 0, ""),
])
def test_prepare_model_input_cleans_before_truncating(text, limit, expected):
    assert prepare_model_input(text, limit) == expected


@pytest.mark.parametrize("limit,content,changed", [
    (10, "새로운 기사 본문", False),
    (1200, "새로운 기사 본문", True),
    (1200, "&nbsp;", False),
])
def test_article_input_change_uses_effective_model_text(limit, content, changed):
    before = NewsItem("검색어", "지역 소식", "https://example.com/news")
    after = replace(before, content=content)
    assert has_changed_article_input(before, after, limit) is changed


def test_article_input_comparison_ignores_keyword_and_spacing_only_changes():
    before = NewsItem("첫 검색어", "지역 소식", "", content="A   B")
    after = replace(before, keyword="다른 검색어", content="A B")
    assert not has_changed_article_input(before, after)


def test_zero_shot_inference_uses_shared_input_preparation_without_loading_model():
    from news_classifier.classifiers.zero_shot_classifier import ZeroShotNewsClassifier

    received = []

    def infer(**kwargs):
        received.extend(kwargs["sequences"])
        return [{"labels": ["사회"], "scores": [.8]} for _ in kwargs["sequences"]]

    classifier = ZeroShotNewsClassifier(
        "synthetic-model", "fixed-revision", ["사회"],
        candidate_descriptions={"사회": "사회"}, max_sequence_length=3,
    )
    classifier._classifier = infer
    predictions = classifier.classify_many(["&nbsp;", " A\n B &amp; C "])

    assert received == [prepare_model_input(" A\n B &amp; C ", 3)] == ["A B"]
    assert predictions[0].label == "분류실패"
    assert predictions[1].label == "사회"
