"""Audit stored CSVs without collection, model execution, or changing their labels.

Usage: python scripts/audit_review_reasons.py --input results.csv candidates.csv
Missing scores/margins are reported as unsupported, never reconstructed from top3.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

from news_classifier.rules.default_rules import DEFAULT_RULE_SET


UNKNOWN = "(unknown)"
REVIEW_LABEL = "검토필요"
ERROR_LABEL = "분류실패"


def _value(row: dict[str, str], field: str) -> str:
    return (row.get(field) or "").strip()


def _number(row: dict[str, str], field: str | None) -> float | None:
    if field is None:
        return None
    try:
        value = float(_value(row, field))
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) and 0 <= value <= 1 else None


def _review_state(row: dict[str, str]) -> bool | None:
    category = _value(row, "final_category")
    status = _value(row, "final_decision_status").upper()
    source = _value(row, "decision_source").upper()
    flag = _value(row, "review_required").casefold()
    if (category in {REVIEW_LABEL, ERROR_LABEL}
            or status in {"REVIEW_REQUIRED", "ERROR"}
            or source in {"REVIEW", "ERROR"}
            or flag in {"true", "1", "y", "yes"}):
        return True
    if (category in DEFAULT_RULE_SET.candidate_labels
            or status == "DECIDED" or source in {"MODEL", "RULE"}
            or flag in {"false", "0", "n", "no"}):
        return False
    return None


def _review_reason(row: dict[str, str]) -> str:
    if (_value(row, "final_category") == ERROR_LABEL
            or _value(row, "final_decision_status").upper() == "ERROR"
            or _value(row, "decision_source").upper() == "ERROR"):
        return "model_error"
    reason = "".join(_value(row, "rule_reason").split())
    for prefix, bucket in (
        ("근거없음", "no_rule_evidence"),
        ("규칙점수부족", "insufficient_rule_score"),
        ("주제간점수차이부족", "insufficient_rule_margin"),
        ("주제근거충돌", "conflicting_rule_evidence"),
    ):
        if reason.startswith(prefix):
            return bucket
    return "other_recorded_reason" if reason else "reason_not_recorded"


def _metric(rows, field, threshold):
    values = [_number(row, field) for row in rows]
    valid = [value for value in values if value is not None]
    return {
        "status": "supported" if field else "unsupported",
        "column": field,
        "reference_threshold": threshold,
        "valid_count": len(valid),
        "missing_or_invalid_count": len(values) - len(valid),
        "below_threshold_count": sum(value < threshold for value in valid),
        "minimum": min(valid) if valid else None,
        "maximum": max(valid) if valid else None,
        "mean": sum(valid) / len(valid) if valid else None,
    }


def _reclassification(rows, fields):
    supported = "review_reclassification" in fields
    payload = {
        "status": "supported" if supported else "unsupported",
        "status_counts": {},
        "recorded_count": None,
        "not_applicable_count": None,
        "unknown_count": None,
        "initial_review_count": None,
        "reclassified_count": None,
        "resolved_count": None,
        "remaining_review_count": None,
        "resolution_unknown_count": None,
        "initial_rule_reasons": {},
        "initial_rule_reason_unknown_count": None,
        "enrichment_status_counts": {},
        "enrichment_status_missing_count": None,
        "http_status_counts": {},
    }
    if not supported:
        return payload
    supported_statuses = {
        "DISABLED", "ALREADY_ENRICHED", "CONTENT_PRESENT", "NO_SCRAPER",
        "NO_LINK", "NO_CONTENT", "UNCHANGED_INPUT", "PENDING", "FETCH_FAILED",
        "RECLASSIFIED",
        "TIMEOUT", "BLOCKED_HTTP", "HTTP_ERROR", "REQUEST_FAILED",
        "GOOGLE_NEWS_WRAPPER", "RSS_WRAPPER", "CONSENT_OR_BLOCK_PAGE",
        "NO_USABLE_PARAGRAPHS",
        "SOURCE_URL_UNRESOLVED", "UNSAFE_URL", "REDIRECT_LIMIT",
    }
    records = []
    unknown = not_applicable = 0
    for row in rows:
        try:
            record = json.loads(_value(row, "review_reclassification"))
        except (ValueError, TypeError):
            unknown += 1
            continue
        if record == {}:
            not_applicable += 1
            continue
        if (not isinstance(record, dict)
                or not isinstance(record.get("status"), str)
                or record["status"] not in supported_statuses
                or record.get("initial_final_category") != REVIEW_LABEL):
            unknown += 1
            continue
        records.append((row, record))
    retried = [row for row, record in records if record["status"] == "RECLASSIFIED"]
    reasons = Counter()
    reason_unknown = 0
    for _, record in records:
        reason = record.get("initial_rule_reason")
        if not isinstance(reason, str) or not reason.strip():
            reason_unknown += 1
            continue
        reasons[_review_reason({"rule_reason": reason})] += 1
    payload.update({
        "status_counts": dict(sorted(Counter(record["status"] for _, record in records).items())),
        "recorded_count": len(records),
        "not_applicable_count": not_applicable,
        "unknown_count": unknown,
        "initial_review_count": len(records),
        "reclassified_count": len(retried),
        "resolved_count": sum(_review_state(row) is False for row in retried),
        "remaining_review_count": sum(_review_state(row) is True for row in retried),
        "resolution_unknown_count": sum(_review_state(row) is None for row in retried),
        "initial_rule_reasons": dict(sorted(reasons.items())),
        "initial_rule_reason_unknown_count": reason_unknown,
        "enrichment_status_counts": dict(sorted(Counter(
            record["enrichment_status"] for _, record in records
            if isinstance(record.get("enrichment_status"), str) and record["enrichment_status"].strip()
        ).items())),
        "enrichment_status_missing_count": sum(
            not isinstance(record.get("enrichment_status"), str) or not record["enrichment_status"].strip()
            for _, record in records
        ),
        "http_status_counts": dict(sorted(Counter(
            str(record["http_status"]) for _, record in records
            if type(record.get("http_status")) is int and 100 <= record["http_status"] <= 599
        ).items())),
    })
    return payload


def summarize_rows(rows, fields, score_threshold, margin_threshold):
    """Count rows as representatives; do not multiply by grouped-source counts."""
    states = [_review_state(row) for row in rows]
    reviewed = [row for row, state in zip(rows, states) if state is True]
    known = sum(state is not None for state in states)
    score_field = next(
        (field for field in ("model_category_score", "model_confidence") if field in fields),
        None,
    )
    margin_field = "score_margin" if "score_margin" in fields else None
    content_supported = "content" in fields
    return {
        "representative_article_count": len(rows),
        "known_decision_count": known,
        "review_required_count": len(reviewed),
        "review_state_unknown_count": len(rows) - known,
        "review_rate_among_known_decisions": len(reviewed) / known if known else None,
        "review_category_count": sum(_value(row, "final_category") == REVIEW_LABEL for row in rows),
        "error_category_count": sum(_value(row, "final_category") == ERROR_LABEL for row in rows),
        "review_reasons": dict(sorted(Counter(_review_reason(row) for row in reviewed).items())),
        "model_score": _metric(rows, score_field, score_threshold),
        "model_margin": _metric(rows, margin_field, margin_threshold),
        "review_model_score": _metric(reviewed, score_field, score_threshold),
        "review_model_margin": _metric(reviewed, margin_field, margin_threshold),
        "review_low_score_and_margin_count": sum(
            _number(row, score_field) is not None
            and _number(row, margin_field) is not None
            and _number(row, score_field) < score_threshold
            and _number(row, margin_field) < margin_threshold
            for row in reviewed
        ) if score_field and margin_field else None,
        "review_reclassification": _reclassification(rows, fields),
        "body": {
            "status": "supported" if content_supported else "unsupported",
            "present_count": sum(bool(_value(row, "content")) for row in rows) if content_supported else None,
            "empty_count": sum(not _value(row, "content") for row in rows) if content_supported else None,
            "review_present_count": sum(bool(_value(row, "content")) for row in reviewed) if content_supported else None,
            "review_empty_count": sum(not _value(row, "content") for row in reviewed) if content_supported else None,
        },
    }


def gold_coverage(rows):
    """Only manually confirmed labels count; suggested/model labels never do."""
    labels = DEFAULT_RULE_SET.candidate_labels
    confirmed = []
    excluded = Counter()
    for row in rows:
        gold = _value(row, "gold_label")
        if not gold:
            excluded["missing_gold_label"] += 1
        elif gold not in labels:
            excluded["invalid_gold_label"] += 1
        elif _value(row, "review_status") != "confirmed":
            excluded["not_confirmed"] += 1
        elif not _value(row, "reviewed_by"):
            excluded["missing_reviewer"] += 1
        elif _value(row, "suggested_label"):
            excluded["suggested_label_present"] += 1
        else:
            confirmed.append(row)
    counts = Counter(_value(row, "gold_label") for row in confirmed)
    support = {}
    for label in labels:
        label_rows = [row for row in confirmed if _value(row, "gold_label") == label]
        events = {_value(row, "event_id").casefold() for row in label_rows if _value(row, "event_id")}
        support[label] = {
            "confirmed_article_count": counts[label],
            "explicit_event_count": len(events),
            "event_id_missing_count": sum(not _value(row, "event_id") for row in label_rows),
            "keywords": sorted({_value(row, "keyword") for row in label_rows if _value(row, "keyword")}),
            "by_split": dict(sorted(Counter(_value(row, "split") or UNKNOWN for row in label_rows).items())),
        }
    event_splits = defaultdict(set)
    split_keywords = defaultdict(set)
    for row in confirmed:
        split = _value(row, "split")
        if split in {"development", "evaluation"}:
            if _value(row, "event_id"):
                event_splits[_value(row, "event_id").casefold()].add(split)
            if _value(row, "keyword"):
                split_keywords[split].add(_value(row, "keyword").casefold())
    return {
        "confirmed_gold_count": len(confirmed),
        "excluded_count": len(rows) - len(confirmed),
        "excluded_reasons": dict(sorted(excluded.items())),
        "candidate_label_count": len(labels),
        "covered_label_count": len(counts),
        "missing_labels": [label for label in labels if not counts[label]],
        "label_support": support,
        "events_shared_between_development_and_evaluation": sorted(
            event for event, splits in event_splits.items() if len(splits) > 1
        ),
        "keywords_shared_between_development_and_evaluation": sorted(
            split_keywords["development"] & split_keywords["evaluation"]
        ),
        "human_confirmation_required": len(confirmed) < len(rows),
        "note": "Support is observed coverage, not proof of accuracy or optimal thresholds. Missing event IDs are not inferred from titles.",
    }


def audit_csv(path: Path, score_threshold: float, margin_threshold: float):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        if not fields or "title" not in fields:
            raise ValueError(f"CSV must contain a title column: {path}")
        if len(fields) != len(set(fields)):
            raise ValueError(f"duplicate CSV column names: {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"malformed CSV row: {path}")
    groups = {}
    for field, output_key in (("keyword", "by_keyword"), ("model_category", "by_predicted_category")):
        grouped = defaultdict(list)
        for row in rows:
            grouped[_value(row, field) or UNKNOWN].append(row)
        groups[output_key] = {
            key: summarize_rows(group, fields, score_threshold, margin_threshold)
            for key, group in sorted(grouped.items())
        }
    return {
        "input": str(path.resolve()),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "columns": fields,
        "summary": summarize_rows(rows, fields, score_threshold, margin_threshold),
        **groups,
        "gold_coverage": gold_coverage(rows),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only stored news CSV review audit; no model execution")
    parser.add_argument("--input", "--dataset", nargs="+", required=True, type=Path, help="Stored collector or candidate CSV files")
    parser.add_argument("--output", type=Path, help="Optional new JSON path; existing files are never overwritten")
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_RULE_SET.decision.ambiguity_score)
    parser.add_argument("--margin-threshold", type=float, default=DEFAULT_RULE_SET.decision.ambiguity_margin)
    args = parser.parse_args(argv)
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (args.score_threshold, args.margin_threshold)):
        parser.error("reference thresholds must be finite numbers between 0 and 1")
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a new --output path")
    resolved = [path.resolve() for path in args.input]
    if len(resolved) != len(set(resolved)):
        parser.error("duplicate input paths are not allowed")
    try:
        payload = {
            "schema_version": 1,
            "mode": "stored_csv_read_only",
            "counting_unit": "CSV row (one representative); grouped source articles are not added",
            "notes": [
                "Reference score/margin cutoffs are descriptive only; original runtime settings cannot be reconstructed from CSV.",
                "Default reference cutoffs use the current base rule policy, not a stored or applied decision profile.",
                "Missing columns are unsupported, blank/invalid metric values remain unknown, and top3 scores are never used to infer margins.",
                "Files are reported separately; overlapping articles across inputs are not deduplicated or combined.",
                "No model, network collection, classification, threshold adjustment, or source CSV write occurs.",
                "Retry outcomes are counted only from explicit saved metadata; missing or malformed metadata never implies success.",
            ],
            "files": [audit_csv(path, args.score_threshold, args.margin_threshold) for path in args.input],
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as file:
                file.write(serialized + "\n")
        print(serialized)
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
