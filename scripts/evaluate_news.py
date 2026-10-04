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
    calculate_technology_bias_metrics,
    render_comparison_markdown,
)
from news_classifier.evaluation_dataset import (
    dataset_sha256,
    evaluation_input,
    legacy_evaluation_input,
    load_reviewed_cases,
    rows_to_items,
    rows_with_article_context,
    validate_unseen_keywords,
)
from news_classifier.classifiers.topic_descriptions import (
    LEGACY_CANDIDATE_LABELS,
    TOPIC_DESCRIPTION_POLICY,
)
from news_classifier.classifiers.zero_shot_classifier import HYPOTHESIS_TEMPLATE
from news_classifier.models import CLASSIFICATION_INPUT_POLICY, NewsItem
from news_classifier.rules.default_rules import (
    CANDIDATE_LABELS,
    LEGACY_UNRELATED_SIGNAL_GROUPS,
)
from news_classifier.rules.event_candidates import rule_evidence_records
from news_classifier.rules.policy import (
    rule_set_fingerprint,
    validate_development_rule_errors,
    validate_development_rule_evidence,
    without_development_rules,
)
from news_classifier.rules.review_validation import audit_rule_holdout, evaluate_review_reduction
from news_classifier.rules.event_profile import create_event_rule_profile, read_rule_proposals, runtime_bindings
from news_classifier.service import build_pipeline
from news_classifier.utils.text import article_description, clean_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="수동 라벨 뉴스 분류 성능을 비교합니다.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--split",
        choices=["development", "evaluation"],
        default="evaluation",
    )
    parser.add_argument("--output-dir", default="evaluation_results")
    parser.add_argument("--compare-legacy", action="store_true", help="같은 모델과 원문으로 이번 변경 전후 비교")
    parser.add_argument("--compare-short-labels", action="store_true", help="같은 17개 주제와 입력으로 짧은 주제명과 사건 설명 비교")
    parser.add_argument("--require-unseen-keywords", action="store_true", help="개발용과 겹치지 않는 검색어만 평가")
    parser.add_argument("--decision-profile", default=None, help="독립 평가를 통과한 최종 판정 기준 파일")
    parser.add_argument("--rules-file", default=None, help="development 사건 ID가 명시된 규칙 제안 JSON")
    parser.add_argument("--export-rule-profile", default=None, help="독립 평가 통과 시에만 새 규칙 적용용 JSON 생성")
    return parser


def rule_only_label(rule_engine, item: NewsItem) -> str:
    return rule_engine.rule_only_decision(
        item.title,
        article_description(item.title, item.description, item.source),
        item.content,
    )


def baseline_rule_set(rule_set):
    """같은 모델 출력으로 비교하기 위해 추가 development 규칙만 제외."""
    return without_development_rules(rule_set)


def legacy_rule_set(rule_set):
    """이번 사건 근거 구분 전의 규칙. 추가 development 규칙은 제외."""
    baseline = baseline_rule_set(rule_set)
    return replace(baseline, version="rules-v2-weighted-longest", unrelated_signal_groups=LEGACY_UNRELATED_SIGNAL_GROUPS, labels=tuple(
        replace(label, terms=tuple(replace(term, context_only=False) for term in label.terms))
        for label in baseline.labels
        if label.label in LEGACY_CANDIDATE_LABELS
    ))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.export_rule_profile:
        if not args.rules_file or args.split != "evaluation":
            parser.error("규칙 프로필 생성은 --rules-file 및 --split evaluation 필요")
        export_path = Path(args.export_rule_profile)
        if export_path.exists() or export_path.resolve() in {Path(args.dataset).resolve(), Path(args.rules_file).resolve()}:
            parser.error("원본 및 기존 프로필 보존: 새 --export-rule-profile 경로 필요")
    if args.rules_file and args.decision_profile:
        parser.error("새 규칙은 판정 기준 변경과 분리하여 먼저 평가 필요")
    if args.require_unseen_keywords and args.split != "evaluation":
        parser.error("새 검색어 검증은 --split evaluation에서만 가능")
    dataset_path = Path(args.dataset)
    rows = load_reviewed_cases(dataset_path, args.split, CANDIDATE_LABELS)
    if args.require_unseen_keywords:
        validate_unseen_keywords(
            load_reviewed_cases(dataset_path, "development", CANDIDATE_LABELS), rows,
        )
    items = rows_to_items(rows)

    settings = AppSettings()
    if args.rules_file:
        settings = replace(settings, event_rule_profile_path="", decision_calibration_path="")
    if args.decision_profile is not None:
        settings = replace(settings, decision_calibration_path=args.decision_profile)
    pipeline = build_pipeline(settings)
    profile_input_mode = getattr(pipeline.postprocessor, "decision_input_mode", None)
    if profile_input_mode == "without_body":
        items = [replace(item, content="") for item in items]
    elif profile_input_mode == "with_body" and any(not clean_text(item.content) for item in items):
        parser.error("본문 보강 판정 프로필의 평가에는 모든 기사의 본문 필요")
    rule_engine = pipeline.postprocessor.rule_engine
    if args.rules_file:
        proposed = read_rule_proposals(args.rules_file, rule_engine.rule_set)
        rule_engine = RuleEngine(proposed)
        pipeline.postprocessor = ClassificationPostProcessor(rule_engine)
    runtime_rule_set = rule_engine.rule_set
    has_development_rules = any(
        term.origin == "development"
        for label_policy in runtime_rule_set.labels
        for term in label_policy.terms
    )
    if has_development_rules:
        # Proposed event rules require real event identities; title hashes do
        # not establish independence and must not count as reviewed evidence.
        rows = load_reviewed_cases(
            dataset_path, args.split, CANDIDATE_LABELS, require_explicit_event_ids=True,
        )
        development_rows = (
            rows
            if args.split == "development"
            else load_reviewed_cases(
                dataset_path,
                "development",
                CANDIDATE_LABELS,
                require_explicit_event_ids=True,
            )
        )
        if args.split == "evaluation":
            validate_unseen_keywords(development_rows, rows)
        development_items = rows_to_items(development_rows)
        if profile_input_mode == "without_body":
            development_items = [replace(item, content="") for item in development_items]
        elif profile_input_mode == "with_body" and any(not clean_text(item.content) for item in development_items):
            parser.error("본문 보강 판정 프로필의 개발 근거에도 모든 기사의 본문 필요")
        development_evidence_rows = rows_with_article_context(development_rows, development_items)
        validate_development_rule_evidence(
            runtime_rule_set,
            development_evidence_rows,
        )

    predictions = pipeline.classifier.classify_many(
        [evaluation_input(item) for item in items]
    )
    if len(predictions) != len(items):
        raise RuntimeError("model prediction count does not match evaluation cases")
    if has_development_rules and args.split == "evaluation":
        development_predictions = pipeline.classifier.classify_many(
            [evaluation_input(item) for item in development_items]
        )
        if len(development_predictions) != len(development_items):
            raise RuntimeError("model prediction count does not match development rule evidence")
        development_baseline = ClassificationPostProcessor(RuleEngine(baseline_rule_set(runtime_rule_set)))
        validate_development_rule_errors(runtime_rule_set, development_evidence_rows, [
            development_baseline.process(item, prediction).rule_decision.final_label
            for item, prediction in zip(development_items, development_predictions, strict=True)
        ])
    expected = [row["gold_label"] for row in rows]
    rule_rankings = [
        rule_engine.rank_rules(
            item.title, article_description(item.title, item.description, item.source), item.content,
        )
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
    if has_development_rules and args.split == "development":
        # A model-correct article can still be a hybrid REVIEW; new event rules
        # must address a confirmed error of the actual previous final path.
        validate_development_rule_errors(
            runtime_rule_set, rows_with_article_context(rows, items),
            [decision.final_label for decision in baseline_decisions],
        )
    predicted_by_method = {
        "모델 단독": [prediction.label for prediction in predictions],
        "규칙 단독": rule_only_labels,
        "추가 규칙 전 하이브리드": [decision.final_label for decision in baseline_decisions],
        "하이브리드": [
            result.rule_decision.final_label for result in hybrid_results
        ],
    }
    legacy_predictions = []
    legacy_results = []
    if args.compare_legacy:
        legacy_predictions = pipeline.classifier.classify_many_legacy(
            [legacy_evaluation_input(item) for item in items],
            candidate_labels=LEGACY_CANDIDATE_LABELS,
        )
        if len(legacy_predictions) != len(items):
            raise RuntimeError("legacy model prediction count does not match evaluation cases")
        legacy_processor = ClassificationPostProcessor(
            RuleEngine(legacy_rule_set(runtime_rule_set)), clean_context=False,
        )
        legacy_results = [
            legacy_processor.process(item, prediction)
            for item, prediction in zip(items, legacy_predictions, strict=True)
        ]
        predicted_by_method["변경 전 모델 단독"] = [prediction.label for prediction in legacy_predictions]
        predicted_by_method["변경 전 하이브리드"] = [result.rule_decision.final_label for result in legacy_results]
        # 모델 설명/입력 변경과 규칙 변경의 영향을 따로 확인.
        predicted_by_method["새 모델 입력과 이전 규칙"] = [
            legacy_processor.process(item, prediction).rule_decision.final_label
            for item, prediction in zip(items, predictions, strict=True)
        ]
    if args.compare_short_labels:
        short_predictions = pipeline.classifier.classify_many_legacy(
            [evaluation_input(item) for item in items], candidate_labels=CANDIDATE_LABELS,
        )
        if len(short_predictions) != len(items):
            raise RuntimeError("short-label prediction count does not match evaluation cases")
        predicted_by_method["짧은 주제명 모델"] = [prediction.label for prediction in short_predictions]
        predicted_by_method["짧은 주제명 하이브리드"] = [
            pipeline.postprocessor.process(item, prediction).rule_decision.final_label
            for item, prediction in zip(items, short_predictions, strict=True)
        ]
    reports = {
        name: calculate_metrics(expected, predicted, CANDIDATE_LABELS)
        for name, predicted in predicted_by_method.items()
    }
    bias_metrics = {
        name: calculate_technology_bias_metrics(expected, predicted)
        for name, predicted in predicted_by_method.items()
    }
    keywords = sorted({row["keyword"] for row in rows})
    per_keyword = {}
    for keyword in keywords:
        indices = [idx for idx, row in enumerate(rows) if row["keyword"] == keyword]
        gold = [expected[idx] for idx in indices]
        per_keyword[keyword] = {
            name: {
                **calculate_metrics(gold, [predicted[idx] for idx in indices], CANDIDATE_LABELS).to_dict(),
                **calculate_technology_bias_metrics(gold, [predicted[idx] for idx in indices]),
            }
            for name, predicted in predicted_by_method.items()
        }
    correction_metrics = calculate_rule_correction_metrics(
        expected,
        predicted_by_method["모델 단독"],
        predicted_by_method["하이브리드"],
        [result.rule_decision.rule_applied for result in hybrid_results],
    )
    rule_review_validation = evaluate_review_reduction(
        expected, predicted_by_method["추가 규칙 전 하이브리드"],
        predicted_by_method["하이브리드"], labels=CANDIDATE_LABELS,
        keywords=[row["keyword"] for row in rows], event_ids=[row["event_id"] for row in rows],
        has_development_rules=has_development_rules, split=args.split,
        unseen_keywords_validated=args.require_unseen_keywords or (has_development_rules and args.split == "evaluation"),
        minimum_independent_events=runtime_rule_set.minimum_development_evidence_events,
    )
    if has_development_rules and args.split == "evaluation":
        rule_review_validation["per_rule_holdout"] = audit_rule_holdout(
            runtime_rule_set, rows, hybrid_results,
        )
        if not rule_review_validation["per_rule_holdout"]["passed"]:
            rule_review_validation["failures"].append("each new expression requires correct held-out independent rule cases")
            rule_review_validation.update(status="not_validated", passed=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_payload = {
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha256(dataset_path),
        "split": args.split,
        "model": AppSettings().classification_model,
        "model_revision": AppSettings().classification_model_revision,
        "labels": CANDIDATE_LABELS,
        "topic_description_policy": TOPIC_DESCRIPTION_POLICY,
        "candidate_hypotheses": pipeline.classifier.candidate_hypotheses,
        "hypothesis_template": HYPOTHESIS_TEMPLATE,
        "unseen_keywords_validated": args.require_unseen_keywords or (has_development_rules and args.split == "evaluation"),
        "compare_legacy": args.compare_legacy,
        "compare_short_labels": args.compare_short_labels,
        "legacy_labels": list(LEGACY_CANDIDATE_LABELS) if args.compare_legacy else None,
        "input_policy": {
            "name": CLASSIFICATION_INPUT_POLICY,
            "include_keyword": False,
            "decision_profile_input_mode": profile_input_mode,
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
        "rule_review_validation": rule_review_validation,
        "technology_bias": bias_metrics,
        "per_keyword": per_keyword,
        "cases": [
            {
                "id": row["id"],
                "event_id": row["event_id"],
                "gold_label": row["gold_label"],
                "model_label": prediction.label,
                "model_score": prediction.score,
                "model_margin": prediction.margin,
                "keyword": row["keyword"],
                "has_body": bool(item.content.strip()),
                "legacy_model_label": legacy_predictions[idx].label if legacy_predictions else None,
                "legacy_hybrid_label": legacy_results[idx].rule_decision.final_label if legacy_results else None,
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
                "previous_correct_harmed": (
                    baseline_decisions[idx].final_label == row["gold_label"]
                    and predicted_by_method["하이브리드"][idx] != row["gold_label"]
                ),
                "previous_wrong_corrected": (
                    baseline_decisions[idx].final_label != row["gold_label"]
                    and predicted_by_method["하이브리드"][idx] == row["gold_label"]
                ),
                "rule_evidence_not_semantic_judgment": rule_evidence_records(
                    {**row, "title": item.title, "description": item.description,
                     "source": item.source, "content": item.content}, rule_engine,
                ),
                "hybrid_rule_applied": hybrid_results[idx].rule_decision.rule_applied,
                "hybrid_reason": hybrid_results[idx].rule_decision.rule_reason,
            }
            for idx, (row, item, prediction) in enumerate(
                zip(rows, items, predictions, strict=True)
            )
        ],
    }
    (output_dir / "evaluation_results.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    bias_lines = [
        "\n## 기술개발 오분류 확인\n",
        "| 방식 | 다른 주제 건수 | 기술개발 오분류 | 다른 주제의 기술개발 오분류율 |",
        "|---|---:|---:|---:|",
    ]
    for name, metrics in bias_metrics.items():
        bias_lines.append(
            f"| {name} | {metrics['non_technology_support']} | "
            f"{metrics['technology_false_positive_count']} | {metrics['technology_false_positive_rate']:.4f} |"
        )
    bias_lines.extend([
        "", "## 검색어별 하이브리드 결과", "",
        "| 검색어 | 기사 수 | 정확도 | 검토 비율 | 다른 주제 건수 | 기술개발 오분류 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for keyword, methods in per_keyword.items():
        metrics = methods["하이브리드"]
        safe_keyword = keyword.replace("|", "\\|").replace("\n", " ")
        bias_lines.append(
            f"| {safe_keyword} | {metrics['total']} | {metrics['accuracy']:.4f} | "
            f"{metrics['review_rate']:.4f} | {metrics['non_technology_support']} | "
            f"{metrics['technology_false_positive_count']} |"
        )
    (output_dir / "evaluation_report.md").write_text(
        render_comparison_markdown(reports, CANDIDATE_LABELS)
        + "\n## 규칙 적용 결과\n\n"
        + "추가 규칙 전 비교는 development 규칙만 제외한 결과이며, 모델 출력과 판정 기준값은 동일.\n\n"
        + f"- 규칙 적용: {correction_metrics['rule_applied_count']}건\n"
        + f"- 실제 주제 변경: {correction_metrics['rule_changed_count']}건\n"
        + f"- 규칙 적용 결과 오답: {correction_metrics['wrong_rule_count']}건\n"
        + f"- 모델 오답을 정답으로 변경: {correction_metrics['corrected_count']}건\n"
        + f"- 모델 정답을 오답으로 변경: {correction_metrics['harmful_change_count']}건\n"
        + "\n## 검토 기사에 대한 새 규칙 검증\n\n"
        + f"- 검증 상태: {rule_review_validation['status']}\n"
        + f"- 새 자동 분류: {rule_review_validation['newly_automatic_count']}건\n"
        + f"- 새 자동 분류 오답: {rule_review_validation['newly_automatic_wrong_count']}건\n"
        + f"- 기존 정답 훼손: {rule_review_validation['previous_correct_harmed_count']}건\n"
        + ("- 미완료 조건: " + "; ".join(rule_review_validation["failures"]) + "\n" if rule_review_validation["failures"] else "")
        + "- 보고서의 관측 검증이며 규칙 자동 활성화 없음. 적은 표본으로 미래 정확도 보장 불가.\n\n"
        + "\n".join(bias_lines) + "\n"
        + ("\n변경 전 비교는 같은 모델 가중치와 원문에 원래 10개 주제명과 입력 형식, 산업 전용 기타 규칙을 적용한 결과. 새 주제를 예측할 수 없는 체계이며 실제 과거 실행 파일과는 별개.\n" if args.compare_legacy else ""),
        encoding="utf-8",
    )
    print(f"평가 보고서 저장: {output_dir / 'evaluation_report.md'}")
    if has_development_rules:
        print(f"새 규칙의 검토 감소 검증: {rule_review_validation['status']}")
    if args.export_rule_profile:
        try:
            payload = create_event_rule_profile(
                runtime_rule_set, rule_review_validation,
                runtime_bindings(settings, pipeline.classifier, CLASSIFICATION_INPUT_POLICY,
                                 HYPOTHESIS_TEMPLATE, baseline_engine.rule_set), dataset_sha256(dataset_path),
            )
        except ValueError as exc:
            parser.error(f"규칙 적용용 파일 생성 중단: {exc}")
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with export_path.open("x", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"검증된 규칙 적용용 파일 생성: {export_path}")


if __name__ == "__main__":
    main()
