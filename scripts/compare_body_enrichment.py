"""Replay stored articles with improved bodies; never overwrite source CSV.

This is an input/behavior comparison, not human-confirmed accuracy evaluation.
New rule candidates are reported for review and never activated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from news_classifier.config import AppSettings
from news_classifier.models import NewsItem
from news_classifier.rules.event_candidates import CANDIDATE_PHRASES, audit_candidate
from news_classifier.rules.policy import rule_set_fingerprint
from news_classifier.service import build_pipeline


def compare_rows(rows, pipeline):
    items = [NewsItem(**{key: row.get(key, "") or "" for key in
                        ("keyword", "title", "link", "source", "published_at", "description", "content")})
             for row in rows]
    predictions = pipeline.classifier.classify_many([item.classification_text() for item in items])
    if len(predictions) != len(items) or any(p.label == "분류실패" for p in predictions):
        raise ValueError("baseline replay model failed; no comparison report generated")
    before = [pipeline.postprocessor.process(item, prediction)
              for item, prediction in zip(items, predictions, strict=True)]
    after = list(before)
    attempts, retries = {}, []
    for index, result in enumerate(before):
        if not result.review_required or result.item.content.strip():
            continue
        outcome = pipeline.article_scraper.enrich_with_diagnostics(result.item)
        attempts[index] = outcome
        item = replace(result.item, content=outcome.item.content)
        if item.content.strip() and item.classification_text() != result.item.classification_text():
            retries.append((index, item))
    if retries:
        predictions = pipeline.classifier.classify_many([item.classification_text() for _, item in retries])
        if len(predictions) != len(retries) or any(p.label == "분류실패" for p in predictions):
            raise ValueError("body replay model failed; no comparison report generated")
        for (index, item), prediction in zip(retries, predictions, strict=True):
            after[index] = pipeline.postprocessor.process(item, prediction)
    retry_indices = {index for index, _ in retries}
    records = []
    for index, (row, previous, selected) in enumerate(zip(rows, before, after, strict=True)):
        outcome = attempts.get(index)
        records.append({
            "title": row["title"], "keyword": row.get("keyword", ""), "link": row.get("link", ""),
            "stored_category_not_gold": row.get("final_category", ""),
            "before_category": previous.rule_decision.final_label,
            "after_category": selected.rule_decision.final_label,
            "after_decision_source": selected.decision_source,
            "after_rule_reason": selected.rule_decision.rule_reason,
            "reclassified": index in retry_indices,
            "enrichment_status": outcome.status if outcome else "NOT_ATTEMPTED",
            "resolved_url": outcome.resolved_url if outcome else "",
            "extraction_diagnostics": outcome.extraction_diagnostics if outcome else {},
            "body_char_count": len(selected.item.content),
            "gold_label": "", "event_id": "", "reviewed_by": "", "review_status": "pending",
        })
    # Only candidate presence is reported, never asserted to be the main subject.
    candidate_rows = [{"title": result.item.title, "description": result.item.description,
                       "content": result.item.content, "source": result.item.source,
                       "keyword": result.item.keyword, "link": result.item.link} for result in after]
    candidates = []
    for category, phrases in CANDIDATE_PHRASES.items():
        for phrase in phrases:
            audit = audit_candidate(phrase, category, candidate_rows, pipeline.postprocessor.rule_engine.rule_set)
            if audit["matching_article_count"]:
                candidates.append({"proposed_category_not_gold": category, "phrase": phrase,
                                   "matching_article_count": audit["matching_article_count"],
                                   "existing_aliases": audit["existing_aliases"],
                                   "scoring_conflicts": audit["scoring_conflicts"],
                                   "activation_eligible": False})
    return {
        "representative_article_count": len(rows),
        "stored_review_count": sum(row.get("final_category") == "검토필요" for row in rows),
        "before_review_count": sum(result.review_required for result in before),
        "after_review_count": sum(result.review_required for result in after),
        "reclassified_count": len(retries),
        "resolved_count": sum(not after[index].review_required for index in retry_indices),
        "previous_automatic_label_changes": sum(
            not old.review_required and old.rule_decision.final_label != new.rule_decision.final_label
            for old, new in zip(before, after, strict=True)),
        "before_category_counts": dict(Counter(r.rule_decision.final_label for r in before)),
        "after_category_counts": dict(Counter(r.rule_decision.final_label for r in after)),
        "cases": records, "pending_rule_candidates": candidates,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="저장 기사 본문 보강 전후 동작 비교, 원본 CSV 보존 및 정답 확인 필요")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="기존 파일이 없는 새 JSON 경로")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.resolve() == args.input.resolve():
        parser.error("기존 파일 보존: 새 출력 경로 필요")
    source_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    with args.input.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"title", "link", "final_category"} <= set(reader.fieldnames):
            parser.error("수집 결과 CSV에 title, link, final_category 필요")
        rows = list(reader)
    if not 1 <= len(rows) <= 100 or any(None in row or any(v is None for v in row.values()) for row in rows):
        parser.error("정상 형식의 대표 기사 1~100건 필요")
    settings = AppSettings()
    pipeline = build_pipeline(settings)
    comparison = compare_rows(rows, pipeline)
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != source_hash:
        parser.error("비교 도중 원본 CSV 변경 감지: 보고서 생성 중단")
    payload = {"schema_version": 1, "mode": "stored_article_body_replay_not_accuracy_evaluation",
               "source_csv": str(args.input.resolve()), "source_sha256": source_hash,
               "source_modified": False, "new_rss_collection": False, "human_gold_confirmed": False,
               "automatic_rule_activation": False,
               "model": settings.classification_model, "model_revision": settings.classification_model_revision,
               "rule_fingerprint": rule_set_fingerprint(pipeline.postprocessor.rule_engine.rule_set),
               "note": "동일 저장 기사에 현재 모델과 규칙 적용. 기존 자동 판정 입력 유지, 검토 기사 본문만 보강. 정답과 독립 사건 및 미사용 검색어 평가는 별도 필요.",
               **comparison}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"본문 보강 비교: 검토 {comparison['before_review_count']}건 → {comparison['after_review_count']}건")
    print(f"비교 보고서: {args.output} (정답 정확도 평가 아님)")


if __name__ == "__main__":
    main()
