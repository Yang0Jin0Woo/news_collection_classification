from __future__ import annotations

import logging
from typing import Sequence

import torch
from transformers import pipeline

from news_classifier.classifiers.base import NewsClassifier
from news_classifier.models import ModelPrediction
from news_classifier.utils.text import clean_text

logger = logging.getLogger(__name__)


class ZeroShotNewsClassifier(NewsClassifier):
    def __init__(
        self,
        model_name: str,
        candidate_labels: Sequence[str],
        max_sequence_length: int = 1200,
    ) -> None:
        self.model_name = model_name
        self.candidate_labels = list(candidate_labels)
        self.max_sequence_length = max_sequence_length
        self._classifier = None

    def _load(self):
        if self._classifier is not None:
            return self._classifier
        device = 0 if torch.cuda.is_available() else -1
        logger.info("loading classifier model=%s device=%s", self.model_name, "cuda" if device == 0 else "cpu")
        self._classifier = pipeline(
            task="zero-shot-classification",
            model=self.model_name,
            framework="pt",
            device=device,
        )
        return self._classifier

    def classify(self, text: str) -> ModelPrediction:
        text = clean_text(text)
        if not text:
            return ModelPrediction.failed()
        try:
            classifier = self._load()
            with torch.no_grad():
                result = classifier(
                    sequences=text[: self.max_sequence_length],
                    candidate_labels=self.candidate_labels,
                    hypothesis_template="이 뉴스의 핵심 주제는 {}입니다.",
                    multi_label=False,
                )
            labels = list(result["labels"])
            scores = [float(score) for score in result["scores"]]
            top1 = scores[0] if scores else 0.0
            top2 = scores[1] if len(scores) > 1 else 0.0
            return ModelPrediction(
                label=labels[0] if labels else "분류실패",
                score=top1,
                margin=top1 - top2,
                top3_labels=labels[:3],
                top3_scores=scores[:3],
            )
        except Exception as exc:
            logger.warning("classification failed: %s", exc)
            return ModelPrediction.failed()
