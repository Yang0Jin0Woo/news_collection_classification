"""Human-confirmed development cases only; no activation or independent accuracy claim."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.classifiers.rule_engine import EVENT_ANCHOR_POLICY, RuleEngine
from news_classifier.config import AppSettings
from news_classifier.evaluation import calculate_metrics
from news_classifier.models import NewsItem
from news_classifier.rules.default_rules import CANDIDATE_LABELS
from news_classifier.rules.event_candidates import rule_evidence_records
from news_classifier.rules.policy import rule_set_fingerprint
from news_classifier.service import build_pipeline
from news_classifier.utils.text import safe_truncate


def load_cases(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(payload, dict)
            or payload.get("profile_type") != "human_confirmed_rule_context_cases"
            or payload.get("dataset_role") != "development"):
        raise ValueError("사람이 확인한 development JSON만 비교 가능")
    rows = payload.get("cases")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 100:
        raise ValueError("확인된 기사 1~100건 필요")
    ids = set()
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip()
                                          for key in ("id", "title", "content", "gold_label", "reviewed_by")):
            raise ValueError("기사 ID, 제목, 본문, 사람 정답과 검토자 필요")
        if (row.get("review_status") != "confirmed" or row.get("split") != "development"
                or row["gold_label"] not in CANDIDATE_LABELS or row.get("suggested_label")
                or row["id"] in ids):
            raise ValueError("미확인 정답, 추천 라벨, 중복 ID 또는 평가 기사 사용 금지")
        if any(not isinstance(row.get(key, ""), str)
               for key in ("keyword", "link", "source", "description")):
            raise ValueError("기사 입력 필드의 문자열 형식 필요")
        ids.add(row["id"])
    return rows


def compare_cases(rows, pipeline):
    items = [NewsItem(
        keyword=row.get("keyword", ""), title=row["title"], link=row.get("link", ""),
        source=row.get("source", ""), description=row.get("description", ""),
        content=safe_truncate(row["content"], 1500),
    ) for row in rows]
    predictions = pipeline.classifier.classify_many([item.classification_text() for item in items])
    if len(predictions) != len(items) or any(pred.label == "분류실패" for pred in predictions):
        raise ValueError("모델 실행 실패: 정확도 또는 전후 비교 생성 중단")
    rules = pipeline.postprocessor.rule_engine.rule_set
    before_processor = ClassificationPostProcessor(RuleEngine(rules))
    candidate_engine = RuleEngine(rules, experimental_event_anchors=True)
    after_processor = ClassificationPostProcessor(candidate_engine)
    before = [before_processor.process(item, pred) for item, pred in zip(items, predictions, strict=True)]
    after = [after_processor.process(item, pred) for item, pred in zip(items, predictions, strict=True)]
    gold = [row["gold_label"] for row in rows]
    metrics = {}
    for name, results in (("before", before), ("experimental_candidate", after)):
        report = calculate_metrics(gold, [r.rule_decision.final_label for r in results], CANDIDATE_LABELS).to_dict()
        metrics[name] = {key: report[key] for key in (
            "total", "accuracy", "coverage", "decided_accuracy", "review_count", "review_rate",
            "wrong_decided_count", "error_count",
        )}
    records = [{
        "id": row["id"], "title": item.title, "link": item.link, "gold_label": row["gold_label"],
        "reviewed_by": row["reviewed_by"], "model_category": pred.label,
        "model_score": pred.score, "model_margin": pred.margin,
        "before_category": old.rule_decision.final_label,
        "after_category": new.rule_decision.final_label,
        "before_source": old.decision_source, "after_source": new.decision_source,
        "before_reason": old.rule_decision.rule_reason, "after_reason": new.rule_decision.rule_reason,
        "previous_correct_harmed": old.rule_decision.final_label == row["gold_label"] and new.rule_decision.final_label != row["gold_label"],
        "previous_wrong_corrected": old.rule_decision.final_label != row["gold_label"] and new.rule_decision.final_label == row["gold_label"],
        "wrong_automatic_to_review": old.rule_decision.final_label in CANDIDATE_LABELS and old.rule_decision.final_label != row["gold_label"] and new.rule_decision.final_label == "검토필요",
        "rule_evidence_not_semantic_judgment": rule_evidence_records(
            {**row, "content": item.content}, before_processor.rule_engine,
        ),
    } for row, item, pred, old, new in zip(rows, items, predictions, before, after, strict=True)]
    return {
        "gold_topic_counts": dict(Counter(gold)), "metrics": metrics, "cases": records,
        "previous_correct_harmed_count": sum(r["previous_correct_harmed"] for r in records),
        "wrong_automatic_to_review_count": sum(r["wrong_automatic_to_review"] for r in records),
        "all_topics_covered": set(gold) == set(CANDIDATE_LABELS),
        "rule_fingerprint_sha256": rule_set_fingerprint(rules),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="사용자 확인 개발 기사로 주요 사건 근거 보류안 비교, 기본 실행에 적용하지 않음")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.resolve() == args.dataset.resolve():
        parser.error("원본과 기존 결과 보존: 새 JSON 출력 경로 필요")
    source_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    rows = load_cases(args.dataset)
    settings = replace(AppSettings(), event_rule_profile_path="", decision_calibration_path="")
    pipeline = build_pipeline(settings)
    result = compare_cases(rows, pipeline)
    if hashlib.sha256(args.dataset.read_bytes()).hexdigest() != source_hash:
        parser.error("비교 중 원본 변경: 보고서 생성 중단")
    payload = {
        "schema_version": 1, "profile_type": "rule_context_comparison_report_not_runtime_profile",
        "dataset": str(args.dataset.resolve()), "dataset_sha256": source_hash,
        "evaluation_scope": "human_confirmed_development_cases_only",
        "candidate_policy": EVENT_ANCHOR_POLICY,
        "source_modified": False, "automatic_activation": False, "independent_evaluation_completed": False,
        "model": settings.classification_model, "model_revision": settings.classification_model_revision,
        "input_body_limit": 1500, "model_body_limit": 500,
        "note": "검토가 늘어도 잘못된 자동 보정을 보류하는 비교안. 앞 두 문장이 주요 사건을 보장하지 않으며 새로운 독립 사건과 검색어의 정답 평가 전 기본 적용 불가. 모든 주제의 성능 수치 아님.",
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
    before, after = result["metrics"]["before"], result["metrics"]["experimental_candidate"]
    print(f"개발 기사 비교 {len(rows)}건: 자동 분류 오답 {before['wrong_decided_count']} → {after['wrong_decided_count']}, 검토 {before['review_count']} → {after['review_count']}")
    print(f"기본 적용 없음, 독립 평가 미완료: {args.output}")


if __name__ == "__main__":
    main()
