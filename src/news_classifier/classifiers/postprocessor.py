from __future__ import annotations

from dataclasses import replace

from news_classifier.classifiers.confidence import (
    ConfidenceThresholds,
    confidence_level,
)
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.models import ClassifiedNews, ModelPrediction, NewsItem, RuleDecision
from news_classifier.utils.text import article_description, clean_text


class ClassificationPostProcessor:
    def __init__(
        self,
        rule_engine: RuleEngine,
        confidence_thresholds: ConfidenceThresholds | None = None,
        clean_context: bool = True,
        baseline_rule_engine: RuleEngine | None = None,
        decision_input_mode: str | None = None,
    ):
        self.rule_engine = rule_engine
        self.confidence_thresholds = confidence_thresholds or ConfidenceThresholds()
        self.clean_context = clean_context
        if decision_input_mode not in {None, "without_body", "with_body"}:
            raise ValueError("unsupported final decision input mode")
        if decision_input_mode is not None and baseline_rule_engine is None:
            raise ValueError("input-bound decision profiles require a baseline rule engine")
        self.baseline_rule_engine = baseline_rule_engine
        self.decision_input_mode = decision_input_mode

    def process(self, item: NewsItem, prediction: ModelPrediction) -> ClassifiedNews:
        if prediction.label == "분류실패":
            decision = RuleDecision(
                final_label="분류실패",
                rule_applied=False,
                rule_reason="모델 분류 실패",
                rule_best_label="",
                rule_match_count=0,
            )
        else:
            engine = self.rule_engine
            # 본문으로 검증한 완화 기준을 제목만 있는 기사에 적용하지 않음(역방향도 동일).
            if self.decision_input_mode is not None:
                actual_mode = "with_body" if clean_text(item.content) else "without_body"
                if actual_mode != self.decision_input_mode:
                    engine = self.baseline_rule_engine
            decision = engine.decide(
                title=item.title,
                description=(
                    article_description(item.title, item.description, item.source)
                    if self.clean_context else item.description
                ),
                content=item.content,
                prediction=prediction,
            )
            if (
                self.decision_input_mode is not None
                and engine is self.rule_engine
                and not decision.rule_applied
                and decision.final_label == prediction.label
            ):
                baseline = self.baseline_rule_engine.decide(
                    title=item.title,
                    description=(
                        article_description(item.title, item.description, item.source)
                        if self.clean_context else item.description
                    ),
                    content=item.content,
                    prediction=prediction,
                )
                if baseline.final_label == "검토필요":
                    policy = engine.rule_set.decision
                    # MODEL 경로와 원점수 유지. 규칙 적용으로 오인되지 않도록
                    # 검증된 공통 기준에 의해 새로 자동 분류된 이유만 기록.
                    decision = replace(decision, rule_reason=(
                        "검증된 공통 판정 기준으로 모델 유지: "
                        f"모델 점수 {prediction.score:.4f} >= {policy.ambiguity_score:.4f}, "
                        f"1순위와 2순위 점수 차이 {prediction.margin:.4f} "
                        f">= {policy.ambiguity_margin:.4f}; "
                        f"기본 기준 검토 사유: {baseline.rule_reason}"
                    ))
        return ClassifiedNews(
            item=item,
            model_prediction=prediction,
            rule_decision=decision,
            model_confidence_level=confidence_level(
                prediction.score,
                prediction.margin,
                self.confidence_thresholds,
            ),
        )
