from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from news_classifier.classifiers.zero_shot_classifier import HYPOTHESIS_TEMPLATE
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.config import AppSettings
from news_classifier.decision_calibration import (
    calibrate_decision_policy, create_decision_profile, evaluate_decision_policy,
)
from news_classifier.evaluation_dataset import (
    evaluation_input, load_reviewed_cases, rows_to_items, rows_with_article_context,
    validate_unseen_keywords,
)
from news_classifier.models import CLASSIFICATION_INPUT_POLICY
from news_classifier.rules.default_rules import CANDIDATE_LABELS
from news_classifier.rules.policy import (
    validate_development_rule_errors, validate_development_rule_evidence,
    without_development_rules,
)
from news_classifier.utils.text import article_description
from news_classifier.service import build_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="확정된 development로 최종 판정 기준을 보정하고, 미사용 검색어 evaluation 통과 후 저장합니다.",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="evaluation_results/decision_calibration.json")
    parser.add_argument("--minimum-support", type=int, default=30)
    parser.add_argument("--accuracy-lower-bound", type=float, default=0.85)
    parser.add_argument("--input-mode", choices=["without_body", "with_body"], default="without_body",
                        help="본문 없는 첫 판정 또는 본문 보강 후 판정 중 하나만 보정")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.minimum_support < 2 or not 0.5 < args.accuracy_lower_bound < 1:
        parser.error("최소 독립사건 수는 2 이상, 정확도 하한 목표는 0.5 초과 1 미만이어야 합니다")
    output_path = Path(args.output)
    dataset_path = Path(args.dataset)
    if output_path.resolve() == dataset_path.resolve():
        parser.error("정답 CSV를 보정 프로필로 덮어쓸 수 없습니다")
    if output_path.exists() and not args.force:
        parser.error(f"기존 프로필이 있습니다. 덮어쓰려면 --force를 사용하세요: {output_path}")
    try:
        development_rows = load_reviewed_cases(
            dataset_path, "development", CANDIDATE_LABELS, require_explicit_event_ids=True,
        )
        evaluation_rows = load_reviewed_cases(
            dataset_path, "evaluation", CANDIDATE_LABELS, require_explicit_event_ids=True,
        )
        validate_unseen_keywords(development_rows, evaluation_rows)
        if args.input_mode == "with_body" and any(
            not row.get("content", "").strip() for row in development_rows + evaluation_rows
        ):
            raise ValueError("with_body mode requires non-empty content for every confirmed case")
        settings = AppSettings(confidence_calibration_path="")
        # Existing final-decision profiles must never become the tuning baseline.
        if hasattr(settings, "decision_calibration_path"):
            settings = replace(settings, decision_calibration_path="")
        pipeline = build_pipeline(settings)
        rule_set = pipeline.postprocessor.rule_engine.rule_set
        development_items = rows_to_items(development_rows)
        evaluation_items = rows_to_items(evaluation_rows)
        if args.input_mode == "without_body":
            development_items = [replace(item, content="") for item in development_items]
            evaluation_items = [replace(item, content="") for item in evaluation_items]
        development_evidence_rows = rows_with_article_context(development_rows, development_items)
        validate_development_rule_evidence(rule_set, development_evidence_rows)
        development_predictions = pipeline.classifier.classify_many([
            evaluation_input(item) for item in development_items
        ])
        if len(development_predictions) != len(development_items):
            raise ValueError("model prediction count does not match development cases")
        baseline_engine = RuleEngine(without_development_rules(rule_set))
        validate_development_rule_errors(rule_set, development_evidence_rows, [
            baseline_engine.decide(
                item.title, article_description(item.title, item.description, item.source),
                item.content, prediction,
            ).final_label
            for item, prediction in zip(development_items, development_predictions, strict=True)
        ])
        result = calibrate_decision_policy(
            development_items, development_predictions,
            [row["gold_label"] for row in development_rows], rule_set,
            event_ids=[row["event_id"] for row in development_rows],
            minimum_support=args.minimum_support,
            accuracy_lower_bound=args.accuracy_lower_bound,
            input_mode=args.input_mode,
        )
        evaluation_predictions = pipeline.classifier.classify_many([
            evaluation_input(item) for item in evaluation_items
        ])
        evaluation = evaluate_decision_policy(
            evaluation_items, evaluation_predictions,
            [row["gold_label"] for row in evaluation_rows], rule_set, result.policy,
            event_ids=[row["event_id"] for row in evaluation_rows],
            minimum_support=args.minimum_support,
            accuracy_lower_bound=args.accuracy_lower_bound,
            input_mode=args.input_mode,
        )
        if not evaluation.passed:
            raise ValueError("independent evaluation failed: " + "; ".join(evaluation.failures))
        payload = create_decision_profile(
            result, evaluation, dataset_path=dataset_path,
            model_name=settings.classification_model,
            model_revision=settings.classification_model_revision,
            candidate_hypotheses=pipeline.classifier.candidate_hypotheses,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            input_policy=CLASSIFICATION_INPUT_POLICY,
            max_sequence_length=settings.max_sequence_length, base_rule_set=rule_set,
            development_rows=development_rows, evaluation_rows=evaluation_rows,
            input_mode=args.input_mode,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(f"최종 판정 보정 중단, 기본 기준 유지: {exc}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        # Exclusive temp creation avoids replacing an unrelated unfinished file.
        with temporary_path.open("x", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
        temporary_path.replace(output_path)
    except OSError as exc:
        parser.error(f"보정 프로필 저장 실패: {exc}")
    print(f"최종 판정 보정 파일 저장: {output_path}")
    print(
        "공통 최종 판정 기준: "
        f"모델 점수 {result.policy.ambiguity_score:.4f}, "
        f"점수 차이 {result.policy.ambiguity_margin:.4f}, "
        f"점수 하한 {result.policy.review_model_score:.4f}",
    )
    print(
        f"독립 평가 검토필요: {evaluation.baseline.total - evaluation.baseline.support} → "
        f"{evaluation.selected.total - evaluation.selected.support}, "
        f"새 자동 확정의 정확도 하한: {evaluation.newly_decided.accuracy_lower_bound:.3f}",
    )
    for name, evidence in (("수정 전", evaluation.baseline), ("수정 후", evaluation.selected)):
        accuracy = f"{evidence.accuracy:.1%}" if evidence.support else "평가 불가(자동 분류 0건)"
        print(
            f"{name}: 검토필요 {(1 - evidence.coverage):.1%}, "
            f"자동 분류 {evidence.support}건, 오분류 {evidence.support - evidence.correct}건, "
            f"자동 분류 정확도 {accuracy}",
        )
    print(
        f"새 자동 분류: {evaluation.newly_decided.support}건, "
        f"오분류 {evaluation.newly_decided.support - evaluation.newly_decided.correct}건, "
        f"관측 정확도 {evaluation.newly_decided.accuracy:.1%}",
    )
    print("신뢰도 높음/보통/낮음 표시 기준은 변경하지 않았습니다.")


if __name__ == "__main__":
    main()
