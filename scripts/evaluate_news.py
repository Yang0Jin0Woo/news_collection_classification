from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from news_classifier.config import AppSettings
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.evaluation import (
    calculate_metrics,
    calculate_rule_correction_metrics,
    render_comparison_markdown,
)
from news_classifier.evaluation_dataset import (
    dataset_sha256,
    evaluation_input,
    load_reviewed_cases,
    rows_to_items,
)
from news_classifier.models import CLASSIFICATION_INPUT_POLICY, NewsItem
from news_classifier.rules.default_rules import CANDIDATE_LABELS
from news_classifier.rules.policy import (
    rule_set_fingerprint,
    validate_development_rule_errors,
    validate_development_rule_evidence,
)
from news_classifier.service import build_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="수동 라벨 뉴스 분류 성능을 비교합니다.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--split",
        choices=["development", "evaluation"],
        default="evaluation",
    )
    parser.add_argument("--output-dir", default="evaluation_results")
    return parser


def rule_only_label(rule_engine, item: NewsItem) -> str:
    return rule_engine.rule_only_decision(
        item.title,
        item.description,
        item.content,
    )


def baseline_rule_set(rule_set):
    """같은 모델 출력으로 비교하기 위해 추가 development 규칙만 제외."""
    return replace(rule_set, labels=tuple(
        replace(label, terms=tuple(term for term in label.terms if term.origin != "development"))
        for label in rule_set.labels
    ))


def main() -> None:
    args = build_parser().parse_args()
    dataset_path = Path(args.dataset)
    rows = load_reviewed_cases(dataset_path, args.split, CANDIDATE_LABELS)
    items = rows_to_items(rows)

    pipeline = build_pipeline(AppSettings())
    rule_engine = pipeline.postprocessor.rule_engine
    runtime_rule_set = rule_engine.rule_set
    has_development_rules = any(
        term.origin == "development"
        for label_policy in runtime_rule_set.labels
        for term in label_policy.terms
    )
    if has_development_rules:
        development_rows = (
            rows
            if args.split == "development"
            else load_reviewed_cases(
                dataset_path,
                "development",
                CANDIDATE_LABELS,
            )
        )
        validate_development_rule_evidence(
            runtime_rule_set,
            development_rows,
        )

    predictions = pipeline.classifier.classify_many(
        [evaluation_input(item) for item in items]
    )
    if len(predictions) != len(items):
        raise RuntimeError("model prediction count does not match evaluation cases")
    if has_development_rules and args.split == "development":
        validate_development_rule_errors(
            runtime_rule_set,
            rows,
            [prediction.label for prediction in predictions],
        )

    expected = [row["gold_label"] for row in rows]
    rule_rankings = [
        rule_engine.rank_rules(item.title, item.description, item.content)
        for item in items
    ]
    rule_only_labels = [rule_only_label(rule_engine, item) for item in items]
    hybrid_results = [
        pipeline.postprocessor.process(item, prediction)
        for item, prediction in zip(items, predictions, strict=True)
    ]
    baseline_engine = RuleEngine(baseline_rule_set(runtime_rule_set))
    baseline_processor = ClassificationPostProcessor(baseline_engine)
    baseline_decisions = [
        baseline_processor.process(item, prediction).rule_decision
        for item, prediction in zip(items, predictions, strict=True)
    ]
    predicted_by_method = {
        "모델 단독": [prediction.label for prediction in predictions],
        "규칙 단독": rule_only_labels,
        "추가 규칙 전 하이브리드": [decision.final_label for decision in baseline_decisions],
        "하이브리드": [
            result.rule_decision.final_label for result in hybrid_results
        ],
    }
    reports = {
        name: calculate_metrics(expected, predicted, CANDIDATE_LABELS)
        for name, predicted in predicted_by_method.items()
    }
    correction_metrics = calculate_rule_correction_metrics(
        expected,
        predicted_by_method["모델 단독"],
        predicted_by_method["하이브리드"],
        [result.rule_decision.rule_applied for result in hybrid_results],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_payload = {
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha256(dataset_path),
        "split": args.split,
        "model": AppSettings().classification_model,
        "model_revision": AppSettings().classification_model_revision,
        "labels": CANDIDATE_LABELS,
        "input_policy": {
            "name": CLASSIFICATION_INPUT_POLICY,
            "include_keyword": False,
        },
        "rule_policy": {
            "version": runtime_rule_set.version,
            "matcher_version": runtime_rule_set.matcher_version,
            "fingerprint_sha256": rule_set_fingerprint(runtime_rule_set),
            "change_policy": runtime_rule_set.change_policy,
            "decision": asdict(runtime_rule_set.decision),
        },
        "reports": {name: report.to_dict() for name, report in reports.items()},
        "baseline_rule_fingerprint_sha256": rule_set_fingerprint(baseline_engine.rule_set),
        "baseline_note": "추가 development 규칙만 제외, 모델 출력과 판정 기준값은 동일",
        "rule_corrections": correction_metrics,
        "cases": [
            {
                "id": row["id"],
                "event_id": row["event_id"],
                "gold_label": row["gold_label"],
                "model_label": prediction.label,
                "model_score": prediction.score,
                "model_margin": prediction.margin,
                "rule_label": predicted_by_method["규칙 단독"][idx],
                "rule_best_label": rule_rankings[idx].first.label,
                "rule_best_score": rule_rankings[idx].first.weighted_score,
                "rule_strong_match_count": (
                    rule_rankings[idx].first.strong_match_count
                ),
                "rule_second_label": rule_rankings[idx].second.label,
                "rule_second_score": rule_rankings[idx].second.weighted_score,
                "rule_score_margin": rule_rankings[idx].score_margin,
                "matched_rule_terms": list(
                    rule_rankings[idx].first.matched_terms
                ),
                "hybrid_label": predicted_by_method["하이브리드"][idx],
                "baseline_hybrid_label": baseline_decisions[idx].final_label,
                "hybrid_rule_applied": hybrid_results[idx].rule_decision.rule_applied,
                "hybrid_reason": hybrid_results[idx].rule_decision.rule_reason,
            }
            for idx, (row, prediction) in enumerate(
                zip(rows, predictions, strict=True)
            )
        ],
    }
    (output_dir / "evaluation_results.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "evaluation_report.md").write_text(
        render_comparison_markdown(reports, CANDIDATE_LABELS)
        + "\n## 규칙 적용 결과\n\n"
        + "추가 규칙 전 비교는 development 규칙만 제외한 결과이며, 모델 출력과 판정 기준값은 동일.\n\n"
        + f"- 규칙 적용: {correction_metrics['rule_applied_count']}건\n"
        + f"- 실제 주제 변경: {correction_metrics['rule_changed_count']}건\n"
        + f"- 규칙 적용 결과 오답: {correction_metrics['wrong_rule_count']}건\n"
        + f"- 모델 오답을 정답으로 변경: {correction_metrics['corrected_count']}건\n"
        + f"- 모델 정답을 오답으로 변경: {correction_metrics['harmful_change_count']}건\n",
        encoding="utf-8",
    )
    print(f"평가 보고서 저장: {output_dir / 'evaluation_report.md'}")


if __name__ == "__main__":
    main()
