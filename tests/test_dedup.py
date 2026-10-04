from dataclasses import replace

import pytest

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
    assert result[0].group_article_count == 2
    assert result[0].related_articles[0].link == "2"


TITLE = "충남도립대 컴퓨터공학과, IoT·그린바이오 융합인재 키운다"


def article(title=TITLE, source="A", published_at="2026-10-04T10:00:00+09:00", **kwargs):
    return NewsItem(keyword="컴퓨터공학", title=title, source=source,
                    published_at=published_at, link=f"https://example.com/{source}", **kwargs)


def test_same_event_groups_cross_source_aliases_and_preserves_originals():
    items = [article(), article(TITLE.replace("키운다", "양성"), "B"), article(source="C")]
    result = TitleSourceDeduplicator().deduplicate(items)
    assert len(result) == 1
    assert result[0].title == TITLE
    assert result[0].source == "A"
    assert result[0].group_article_count == 3
    assert [ref.source for ref in result[0].related_articles] == ["B", "C"]
    assert [ref.link for ref in result[0].related_articles] == [item.link for item in items[1:]]
    assert "example.com/B" not in result[0].classification_text()


def test_shared_computer_science_phrase_does_not_merge_different_events():
    items = [
        article(),
        article("광운대 정보과학교육원 컴퓨터공학 신입생 선발", "B"),
        article("충남도립대 컴퓨터공학과, 전공 벽 낮춘다", "C"),
        article("충남도립대 컴퓨터공학과, 신산업 융합인재 키운다", "D"),
    ]
    assert len(TitleSourceDeduplicator().deduplicate(items)) == 4


@pytest.mark.parametrize("date", ["", "invalid", "2026-10-07T10:00:00+09:00"])
def test_cross_source_articles_without_nearby_known_dates_are_kept(date):
    assert len(TitleSourceDeduplicator().deduplicate([article(), article(source="B", published_at=date)])) == 2


def test_same_title_and_source_far_apart_are_not_assumed_same_event():
    assert len(TitleSourceDeduplicator().deduplicate([
        article(), article(published_at="2026-10-07T10:00:00+09:00")
    ])) == 2


def test_date_comparison_handles_timezone_and_window_boundary():
    result = TitleSourceDeduplicator().deduplicate([
        article(), article(source="B", published_at="2026-10-06T01:00:00+00:00")
    ])
    assert len(result) == 1
    outside = article(source="C", published_at="2026-10-06T01:00:01+00:00")
    assert len(TitleSourceDeduplicator().deduplicate([article(), outside])) == 2


@pytest.mark.parametrize("different", [
    TITLE.replace("충남도립대", "충북도립대"),
    TITLE.replace("융합인재", "신입생"),
    TITLE + " 20명",
    TITLE.replace("키운다", "양성 취소"),
])
def test_entity_number_and_event_changes_are_not_merged(different):
    assert len(TitleSourceDeduplicator().deduplicate([article(), article(different, "B")])) == 2


def test_independent_descriptions_can_confirm_small_title_variations():
    description = "학생들이 센서와 생명공학 실습 수업을 함께 수강하는 융합 교육과정을 마련했다"
    result = TitleSourceDeduplicator().deduplicate([
        article(description=description),
        article(TITLE.replace("융합인재", "융합인재 본격 정식"), "B", description=description),
    ])
    assert len(result) == 1


@pytest.mark.parametrize("description", [
    "졸업생 취업 지원 프로그램을 운영하고 해외 기업과 협약을 체결했다",
    "학생 20명을 대상으로 센서와 생명공학 실습 수업을 제공한다",
    "학생들이 센서와 생명공학 실습 수업을 함께 수강하는 교육과정은 취소됐다",
])
def test_conflicting_descriptions_keep_articles_separate(description):
    result = TitleSourceDeduplicator().deduplicate([
        article(description="학생들이 센서와 생명공학 실습 수업을 함께 수강하는 융합 교육과정을 마련했다"),
        article(source="B", description=description),
    ])
    assert len(result) == 2


def test_description_with_different_organization_keeps_articles_separate():
    description = "충남도립대 학생들이 센서와 생명공학 실습 수업을 함께 수강하는 융합 교육과정을 마련했다"
    assert len(TitleSourceDeduplicator().deduplicate([
        article(description=description),
        article(source="B", description=description.replace("충남도립대", "충북도립대")),
    ])) == 2


def test_title_only_rss_description_does_not_relax_title_threshold():
    left = article()
    right = article(TITLE.replace("융합인재", "융합인재 본격 정식"), "B")
    left = replace(left, description=f"{left.title} - {left.source}")
    right = replace(right, description=f"{right.title} - {right.source}")
    assert len(TitleSourceDeduplicator().deduplicate([left, right])) == 2


def test_complete_link_grouping_avoids_date_chaining():
    result = TitleSourceDeduplicator().deduplicate([
        article(published_at="2026-10-04T12:00:00+00:00"),
        article(source="B", published_at="2026-10-03T00:00:00+00:00"),
        article(source="C", published_at="2026-10-06T00:00:00+00:00"),
    ])
    assert len(result) == 2
    assert result[0].group_article_count == 2


def test_grouping_is_idempotent_and_preserves_reference_metadata():
    dedup = TitleSourceDeduplicator()
    once = dedup.deduplicate([article(), article(source="B")])
    assert dedup.deduplicate(once) == once


def test_short_generic_titles_are_not_grouped_across_sources():
    result = TitleSourceDeduplicator().deduplicate([
        article("컴퓨터공학 취업 소식", "A"), article("컴퓨터공학 취업 소식", "B")
    ])
    assert len(result) == 2


def test_pre_grouped_input_does_not_allow_date_chaining():
    dedup = TitleSourceDeduplicator()
    first = dedup.deduplicate([
        article(published_at="2026-10-04T12:00:00+00:00"),
        article(source="B", published_at="2026-10-03T00:00:00+00:00"),
    ])
    late = article(source="C", published_at="2026-10-06T00:00:00+00:00")
    result = dedup.deduplicate(first + [late])
    assert len(result) == 2
    assert sum(item.group_article_count for item in result) == 3


def test_same_title_and_source_with_conflicting_descriptions_are_kept():
    result = TitleSourceDeduplicator().deduplicate([
        article(description="학생들이 센서와 생명공학 실습 수업을 함께 수강하는 교육과정은 진행된다"),
        article(description="학생들이 센서와 생명공학 실습 수업을 함께 수강하는 교육과정은 취소됐다"),
    ])
    assert len(result) == 2
