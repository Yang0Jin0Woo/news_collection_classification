from __future__ import annotations

from abc import ABC, abstractmethod
from news_classifier.models import ModelPrediction


class NewsClassifier(ABC):
    @abstractmethod
    def classify(self, text: str) -> ModelPrediction:
        raise NotImplementedError

    def classify_many(self, texts: list[str]) -> list[ModelPrediction]:
        return [self.classify(text) for text in texts]
