import csv
from dataclasses import asdict
import hashlib

import pytest

from news_classifier.evaluation_dataset import (
    dataset_sha256,
    evaluation_input,
    load_reviewed_cases,
    read_csv_dataset,
    row_to_item,
    rows_to_items,
)
from news_classifier.models import NewsItem


FIELDNAMES = [
    "id",
    "event_id",
    "split",
    "keyword",
    "title",
    "description",
    "gold_label",
    "review_status",
    "reviewed_by",
]


def write_cases(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def confirmed_case(case_id, event_id, split, label, title):
    return {
        "id": case_id,
        "event_id": event_id,
        "split": split,
        "keyword": "정답을 암시하는 고유 검색어",
        "title": title,
        "description": "기사 설명",
        "gold_label": label,
        "review_status": "confirmed",
        "reviewed_by": "reviewer",
    }


def test_evaluation_input_excludes_search_keyword():
    item = NewsItem(
        keyword="정답을 암시하는 고유 검색어",
        title="기사 제목",
        description="기사 설명",
        link="https://example.com",
    )

    text = evaluation_input(item)

    assert "정답을 암시하는 고유 검색어" not in text
    assert "기사 제목" in text
    assert "기사 설명" in text


def test_reviewed_cases_reject_event_split_leakage(tmp_path):
    path = tmp_path / "cases.csv"
    write_cases(
        path,
        [
            confirmed_case("1", "same-event", "development", "기술개발", "제목 1"),
            confirmed_case("2", " SAME-EVENT ", "evaluation", "기타/무관", "제목 2"),
        ],
    )

    with pytest.raises(ValueError, match="both development and evaluation"):
        load_reviewed_cases(
            path,
            "development",
            ["기술개발", "기타/무관"],
        )


def test_reviewed_cases_load_confirmed_split_with_all_labels(tmp_path):
    path = tmp_path / "cases.csv"
    write_cases(
        path,
        [
            confirmed_case("1", "event-1", "development", "기술개발", "제목 1"),
            confirmed_case("2", "event-2", "development", "기타/무관", "제목 2"),
        ],
    )

    rows = load_reviewed_cases(
        path,
        "development",
        ["기술개발", "기타/무관"],
    )

    assert [row["id"] for row in rows] == ["1", "2"]


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_common_csv_reader_preserves_columns_values_and_source(tmp_path, encoding, newline):
    path = tmp_path / "articles.csv"
    path.write_text(
        f'title,description,content,extra{newline}"기사, 제목","첫 줄{newline}둘째 줄",,  값  {newline}',
        encoding=encoding, newline="",
    )
    original = path.read_bytes()
    columns, rows = read_csv_dataset(path)
    assert columns == ["title", "description", "content", "extra"]
    assert rows == [{"title": "기사, 제목", "description": f"첫 줄{newline}둘째 줄", "content": "", "extra": "  값  "}]
    assert path.read_bytes() == original
    assert dataset_sha256(path) == hashlib.sha256(original).hexdigest()


@pytest.mark.parametrize("raw,reason", [
    ("title,title\nfirst,second\n", "duplicate"),
    ("title,content\nfirst\n", "malformed"),
    ("title\nfirst,extra\n", "malformed"),
])
def test_common_csv_reader_rejects_ambiguous_or_incomplete_rows(tmp_path, raw, reason):
    path = tmp_path / "bad.csv"
    path.write_text(raw, encoding="utf-8-sig")
    original = path.read_bytes()
    with pytest.raises(ValueError, match=reason):
        read_csv_dataset(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("raw,columns", [("", []), ("title,content\n", ["title", "content"])])
def test_common_csv_reader_leaves_empty_dataset_policy_to_the_tool(tmp_path, raw, columns):
    path = tmp_path / "empty.csv"
    path.write_text(raw, encoding="utf-8")
    assert read_csv_dataset(path) == (columns, [])


def test_common_article_conversion_preserves_text_without_review_hints():
    row = {
        "keyword": "검색어 힌트", "title": "  기사 제목  ", "link": "https://example.com/news",
        "source": "매체", "published_at": "2026-10-11", "description": "기사\n설명",
        "content": "원문 본문", "gold_label": "정답 힌트", "final_category": "저장 판정 힌트",
        "reviewed_by": "검토자 힌트", "review_status": "confirmed", "id": "식별자 힌트",
    }
    original = dict(row)
    item = row_to_item(row)
    assert asdict(item) == {
        **{field: row[field] for field in ("keyword", "title", "link", "source", "published_at", "description", "content")},
        "related_articles": (),
    }
    assert rows_to_items([row]) == [item]
    for field in ("keyword", "gold_label", "final_category", "reviewed_by", "id"):
        assert row[field] not in evaluation_input(item)
    assert row == original


def test_common_article_conversion_defaults_missing_or_null_fields_to_empty_text():
    row = {"title": "기사", "link": None, "content": None, "source": ""}
    assert row_to_item(row) == NewsItem(keyword="", title="기사", link="")
    assert row == {"title": "기사", "link": None, "content": None, "source": ""}


def test_reviewed_case_reader_also_rejects_duplicate_columns_and_extra_values(tmp_path):
    path = tmp_path / "cases.csv"
    for raw, reason in (
        ("id,title,title\n1,first,second\n", "duplicate"),
        ("id,title\n1,first,extra\n", "malformed"),
    ):
        path.write_text(raw, encoding="utf-8")
        with pytest.raises(ValueError, match=reason):
            load_reviewed_cases(path, "development", ["기술개발"])
