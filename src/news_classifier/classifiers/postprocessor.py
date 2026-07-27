from __future__ import annotations

from news_classifier.classifiers.confidence import (
    ConfidenceThresholds,
    confidence_level,
)
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.models import ClassifiedNews, ModelPrediction, NewsItem, RuleDecision


class ClassificationPostProcessor:
    def __init__(
        self,
        rule_engine: RuleEngine,
        confidence_thresholds: ConfidenceThresholds | None = None,
    ):
        self.rule_engine = rule_engine
        self.confidence_thresholds = confidence_thresholds or ConfidenceThresholds()

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
            decision = self.rule_engine.decide(
                title=item.title,
                description=item.description,
                content=item.content,
                prediction=prediction,
            )
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
