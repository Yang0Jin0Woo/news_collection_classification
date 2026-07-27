from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from news_classifier.models import NewsItem
from news_classifier.utils.text import normalize_key


REQUIRED_COLUMNS = {
    "id",
    "split",
    "keyword",
    "title",
    "description",
    "gold_label",
    "review_status",
    "reviewed_by",
}


def event_id_from_title(title: str) -> str:
    normalized = normalize_key(title)
    if not normalized:
        raise ValueError("event title is empty")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reviewed_cases(
    path: Path,
    split: str,
    labels: list[str],
) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"missing dataset columns: {sorted(missing_columns)}")
        all_rows = list(reader)

    ids: set[str] = set()
    event_splits: dict[str, set[str]] = {}
    for row in all_rows:
        case_id = row["id"].strip()
        if not case_id or case_id in ids:
            raise ValueError(f"duplicate or empty case id: {case_id}")
        ids.add(case_id)

        event_id = (
            row.get("event_id", "").strip().casefold()
            or event_id_from_title(row["title"])
        )
        row["event_id"] = event_id
        row_split = row["split"].strip()
        if row_split in {"development", "evaluation"}:
            event_splits.setdefault(event_id, set()).add(row_split)

    leaked_events = sorted(
        event_id
        for event_id, splits in event_splits.items()
        if len(splits) > 1
    )
    if leaked_events:
        raise ValueError(
            "events appear in both development and evaluation splits: "
            f"{leaked_events}"
        )

    rows = [row for row in all_rows if row["split"].strip() == split]
    if not rows:
        raise ValueError(f"no rows found for split={split}")

    for row in rows:
        if row["review_status"].strip() != "confirmed":
            raise ValueError(f"case {row['id']} is not manually confirmed")
        if not row["reviewed_by"].strip():
            raise ValueError(f"case {row['id']} has no reviewer")
        if row.get("suggested_label", "").strip():
            raise ValueError(
                f"case {row['id']} contains a suggested label that can bias review"
            )
        if row["gold_label"] not in labels:
            raise ValueError(f"case {row['id']} has invalid gold label")

    covered_labels = {row["gold_label"] for row in rows}
    if covered_labels != set(labels):
        missing = sorted(set(labels) - covered_labels)
        raise ValueError(f"evaluation split does not cover all labels: {missing}")
    return rows


def rows_to_items(rows: list[dict[str, str]]) -> list[NewsItem]:
    return [
        NewsItem(
            keyword=row["keyword"],
            title=row["title"],
            link=row.get("link", ""),
            source=row.get("source", ""),
            published_at=row.get("published_at", ""),
            description=row["description"],
        )
        for row in rows
    ]


def evaluation_input(item: NewsItem) -> str:
    """검색어가 정답 힌트가 되지 않도록 기사 내용만 모델에 전달한다."""
    return item.classification_text(include_keyword=False)
