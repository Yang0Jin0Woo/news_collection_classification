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
        batch_size: int = 4,
    ) -> None:
        self.model_name = model_name
        self.candidate_labels = list(candidate_labels)
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self._classifier = None

    # zero-shot 분류 모델을 최초 1회만 로드
    def _load(self):
        if self._classifier is not None:
            return self._classifier

        # GPU 사용 가능 시 cuda, 아니면 CPU 사용
        device = 0 if torch.cuda.is_available() else -1
        logger.info("loading classifier model=%s device=%s", self.model_name, "cuda" if device == 0 else "cpu")

        self._classifier = pipeline(
            task="zero-shot-classification",
            model=self.model_name,
            framework="pt",
            device=device,
        )
        return self._classifier

    def _prediction_from_result(self, result: dict) -> ModelPrediction:
        labels = list(result["labels"])
        scores = [float(score) for score in result["scores"]]

        # top1 점수와 top2 점수 차이로 분류 확신도 계산(top1-top2 차이: 1순위 라벨을 얼마나 확신하는지 보는 기준. 차이가 적으면 애매하다고 판단.)
        top1 = scores[0] if scores else 0.0
        top2 = scores[1] if len(scores) > 1 else 0.0

        return ModelPrediction(
            label=labels[0] if labels else "분류실패",
            score=top1,
            margin=top1 - top2,
            top3_labels=labels[:3],
            top3_scores=scores[:3],
        )

    # 입력 텍스트를 후보 라벨 중 하나로 분류
    def classify(self, text: str) -> ModelPrediction:
        return self.classify_many([text])[0]

    # 여러 입력 텍스트를 batch로 묶어 분류
    def classify_many(self, texts: list[str]) -> list[ModelPrediction]:
        cleaned_texts = [clean_text(text)[: self.max_sequence_length] for text in texts]
        predictions = [ModelPrediction.failed() for _ in cleaned_texts]
        valid_items = [(idx, text) for idx, text in enumerate(cleaned_texts) if text]

        # 분류할 텍스트가 없으면 실패 결과 반환
        if not valid_items:
            return predictions

        try:
            classifier = self._load()

            # 모델 추론 시 gradient 계산 비활성화(경사하강법)
            with torch.no_grad():
                results = classifier(
                    sequences=[text for _, text in valid_items],
                    candidate_labels=self.candidate_labels,
                    hypothesis_template="이 뉴스의 핵심 주제는 {}입니다.",
                    multi_label=False,
                    batch_size=self.batch_size,
                )

            if isinstance(results, dict):
                results = [results]

            for (idx, _), result in zip(valid_items, results):
                predictions[idx] = self._prediction_from_result(result)

        # 모델 로드 또는 분류 실패 시 실패 결과 반환
        except Exception as exc:
            logger.warning("classification failed: %s", exc)
            return [ModelPrediction.failed() for _ in cleaned_texts]

        return predictions
