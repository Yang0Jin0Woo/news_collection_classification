from __future__ import annotations

from news_classifier.classifiers.confidence import confidence_level
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.models import ClassifiedNews, ModelPrediction, NewsItem


class ClassificationPostProcessor:
    def __init__(self, rule_engine: RuleEngine):
        self.rule_engine = rule_engine

    def process(self, item: NewsItem, prediction: ModelPrediction) -> ClassifiedNews:
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
            confidence_level=confidence_level(prediction.score, prediction.margin),
        )
