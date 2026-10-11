"""Blind article review and readiness checks, not automatic gold annotation."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from news_classifier.rules.event_candidates import article_review_id
from news_classifier.utils.text import clean_text, normalize_key


ARTICLE_FIELDS = ("keyword", "title", "description", "content", "source", "published_at", "link")
REVIEW_COLUMNS = (
    "id", "event_id", "independent_event_check", "split", *ARTICLE_FIELDS,
    "gold_label", "review_status", "reviewed_by", "main_event", "main_event_evidence", "notes",
)


def _value(row: Mapping, field: str) -> str:
    return str(row.get(field) or "")


def prepare_accuracy_review(
    candidates: Sequence[Mapping], confirmed_cases: Sequence[Mapping], labels: Sequence[str],
) -> list[dict[str, str]]:
    """Hide stored predictions; only explicitly supplied confirmed cases retain gold."""
    prepared: dict[str, dict[str, str]] = {}
    for rows, preserve_confirmation in ((candidates, False), (confirmed_cases, True)):
        for source in rows:
            if not isinstance(source, Mapping):
                raise ValueError("review article must be a field mapping")
            if not clean_text(_value(source, "title")):
                raise ValueError("review article requires a non-empty title")
            article_id, _ = article_review_id(source)
            reference = _value(source, "link").strip() or article_id
            row = {
                field: _value(source, field) for field in ARTICLE_FIELDS
            }
            row.update(
                id=article_id, event_id="", independent_event_check="pending", split="",
                gold_label="", review_status="pending", reviewed_by="",
                main_event="", main_event_evidence="", notes="",
            )
            if preserve_confirmation:
                if (_value(source, "review_status").strip() != "confirmed"
                        or _value(source, "gold_label") not in labels
                        or not _value(source, "reviewed_by").strip()
                        or _value(source, "split").strip() != "development"):
                    raise ValueError("imported gold requires an explicitly confirmed development case")
                for field in (
                    "event_id", "independent_event_check", "split", "gold_label",
                    "review_status", "reviewed_by", "main_event", "main_event_evidence", "notes",
                ):
                    row[field] = _value(source, field)
                row["independent_event_check"] = row["independent_event_check"] or "pending"
                # Existing cases are development only: never promote them to holdout.
                row["split"] = "development"
                prepared[reference] = row
            else:
                prepared.setdefault(reference, row)
    result = list(prepared.values())
    ids = [row["id"] for row in result]
    if len(ids) != len(set(ids)):
        raise ValueError("different review articles have the same article id")
    return result


def accuracy_review_readiness(
    rows: Sequence[Mapping], labels: Sequence[str], *, minimum_events_per_label: int = 2,
) -> dict:
    """Validate review attestations and separation before any model execution.

    Two events per label is only a pilot coverage gate, not evidence of stable
    accuracy. No semantic gold labels or event identities are inferred here.
    """
    if type(minimum_events_per_label) is not int or minimum_events_per_label < 1:
        raise ValueError("minimum_events_per_label must be a positive integer")
    labels = list(labels)
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("review labels must be non-empty and unique")
    ids = Counter(_value(row, "id").strip() for row in rows)
    duplicate_ids = sorted(case_id for case_id, count in ids.items() if count > 1)
    event_splits: dict[str, set[str]] = {}
    article_splits: dict[str, set[str]] = {}
    split_keywords = {split: set() for split in ("development", "evaluation")}
    per_label = {split: {label: set() for label in labels} for split in split_keywords}
    missing = Counter()
    ready_rows = 0
    confirmed_gold = 0
    unknown_gold = set()
    invalid_splits = set()
    pending_ids = []
    for row in rows:
        case_id = _value(row, "id").strip()
        split = _value(row, "split").strip()
        event_id = _value(row, "event_id").strip().casefold()
        keyword = normalize_key(_value(row, "keyword"))
        gold = _value(row, "gold_label")
        reviewed = (_value(row, "review_status").strip() == "confirmed"
                    and bool(_value(row, "reviewed_by").strip()))
        if reviewed and gold in labels:
            confirmed_gold += 1
        if gold and gold not in labels:
            unknown_gold.add(gold)
        if split and split not in split_keywords:
            invalid_splits.add(split)
        if split in split_keywords:
            if keyword:
                split_keywords[split].add(keyword)
            if event_id:
                event_splits.setdefault(event_id, set()).add(split)
            link = _value(row, "link").strip()
            if link:
                article_splits.setdefault(link, set()).add(split)
        checks = {
            "article_id": bool(case_id) and ids[case_id] == 1,
            "article_title": bool(clean_text(_value(row, "title"))),
            "article_link": bool(_value(row, "link").strip()),
            "confirmed_gold": reviewed and gold in labels,
            "explicit_event_id": bool(event_id),
            "event_confirmation": _value(row, "independent_event_check").strip() == "confirmed",
            "split": split in split_keywords,
            "keyword": bool(keyword),
            "main_event": bool(clean_text(_value(row, "main_event"))),
            "main_event_evidence": bool(clean_text(_value(row, "main_event_evidence"))),
            "no_suggested_label": not _value(row, "suggested_label").strip(),
        }
        for field, valid in checks.items():
            if not valid:
                missing[field] += 1
        if not all(checks.values()):
            pending_ids.append(case_id)
            continue
        ready_rows += 1
        per_label[split][gold].add(event_id)
    leaked_events = sorted(event for event, splits in event_splits.items() if len(splits) > 1)
    leaked_keywords = sorted(split_keywords["development"] & split_keywords["evaluation"])
    leaked_articles = sorted(link for link, splits in article_splits.items() if len(splits) > 1)
    counts = {split: {label: len(events) for label, events in groups.items()}
              for split, groups in per_label.items()}
    insufficient = {split: [label for label in labels if counts[split][label] < minimum_events_per_label]
                    for split in counts}
    ready = bool(rows) and not (
        pending_ids or leaked_events or leaked_keywords or leaked_articles or any(insufficient.values())
    )
    return {
        "schema_version": 1,
        "status": "ready_for_evaluation" if ready else "awaiting_review",
        "ready": ready,
        "independent_evaluation_completed": False,
        "article_count": len(rows),
        "confirmed_gold_count": confirmed_gold,
        "ready_article_count": ready_rows,
        "articles_requiring_completion": len(pending_ids),
        "incomplete_fields": dict(sorted(missing.items())),
        "pending_article_ids_preview": pending_ids[:10],
        "duplicate_article_ids": duplicate_ids,
        "unknown_gold_labels": sorted(unknown_gold),
        "invalid_splits": sorted(invalid_splits),
        "events_crossing_splits": leaked_events,
        "article_links_crossing_splits": leaked_articles,
        "keywords_crossing_splits": leaked_keywords,
        "minimum_independent_events_per_label": minimum_events_per_label,
        "independent_events_per_label": counts,
        "labels_needing_more_events": insufficient,
        "labels": labels,
        "note": (
            "Only review metadata and dataset separation checked; no model accuracy measured. "
            "Read original articles to confirm gold and event identities. "
            "Minimum coverage is a pilot gate, not statistically sufficient accuracy evidence."
        ),
    }


def require_accuracy_review_ready(
    rows: Sequence[Mapping], labels: Sequence[str], *, minimum_events_per_label: int = 2,
) -> dict:
    report = accuracy_review_readiness(rows, labels, minimum_events_per_label=minimum_events_per_label)
    if not report["ready"]:
        raise ValueError(
            "독립 평가 자료 미준비: 정답·사건·주요 사건 근거 확인, 개발/평가 분리, "
            "별도 검색어와 주제별 사건 수 확인 필요. prepare_accuracy_review.py check로 상세 확인"
        )
    return report
