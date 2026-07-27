import csv

import pytest

from news_classifier.evaluation_dataset import (
    evaluation_input,
    load_reviewed_cases,
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
