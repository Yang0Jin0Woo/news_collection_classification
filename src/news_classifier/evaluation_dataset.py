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
    *,
    require_explicit_event_ids: bool = False,
) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"missing dataset columns: {sorted(missing_columns)}")
        if require_explicit_event_ids and "event_id" not in (reader.fieldnames or []):
            raise ValueError("explicit event_id column is required for decision calibration")
        all_rows = list(reader)

    ids: set[str] = set()
    event_splits: dict[str, set[str]] = {}
    for row in all_rows:
        if any(row.get(field) is None for field in REQUIRED_COLUMNS):
            raise ValueError("dataset contains a malformed row with missing values")
        case_id = row["id"].strip()
        if not case_id or case_id in ids:
            raise ValueError(f"duplicate or empty case id: {case_id}")
        ids.add(case_id)

        raw_event_id = (row.get("event_id") or "").strip().casefold()
        if (require_explicit_event_ids
                and row["split"].strip() in {"development", "evaluation"}
                and not raw_event_id):
            raise ValueError(f"case {case_id} requires an explicitly reviewed event_id")
        event_id = (
            raw_event_id
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
            content=row.get("content", ""),
        )
        for row in rows
    ]


def rows_with_article_context(
    rows: list[dict[str, str]], items: list[NewsItem],
) -> list[dict[str, str]]:
    """검증 근거도 실제 판정 입력과 일치하도록 원본을 보존한 행 복사."""
    if len(rows) != len(items):
        raise ValueError("reviewed rows and actual article inputs must align")
    return [
        {**row, "title": item.title, "description": item.description,
         "source": item.source, "content": item.content}
        for row, item in zip(rows, items, strict=True)
    ]


def evaluation_input(item: NewsItem) -> str:
    """검색어가 정답 힌트가 되지 않도록 기사 내용만 모델에 전달한다."""
    return item.classification_text(include_keyword=False)


def legacy_evaluation_input(item: NewsItem) -> str:
    """동일한 원문으로 입력 정제 변경 전후를 비교하기 위한 이전 입력 형식."""
    parts = [f"기사제목: {item.title}"]
    if item.description:
        parts.append(f"기사설명: {item.description}")
    if item.content:
        parts.append(f"기사본문요약: {item.content[:500]}")
    return "\n".join(parts)


def validate_unseen_keywords(
    development_rows: list[dict[str, str]], evaluation_rows: list[dict[str, str]],
) -> None:
    """새 검색어 검증을 요청한 경우 개발용 검색어의 평가 재사용 차단."""
    development = {normalize_key(row["keyword"]) for row in development_rows}
    evaluation = {normalize_key(row["keyword"]) for row in evaluation_rows}
    if "" in development or "" in evaluation:
        raise ValueError("unseen-keyword evaluation requires non-empty keywords")
    overlap = sorted(development & evaluation)
    if overlap:
        raise ValueError(f"keywords appear in both development and evaluation: {overlap}")
