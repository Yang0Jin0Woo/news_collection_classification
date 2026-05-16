from __future__ import annotations

from dataclasses import dataclass

from news_classifier.classifiers.confidence import is_ambiguous
from news_classifier.models import ModelPrediction, RuleDecision
from news_classifier.utils.text import clean_text


@dataclass(frozen=True)
class RuleEngineConfig:
    # 모델 결과를 그대로 신뢰할 최소 점수
    base_rule_override_threshold: float = 0.55

    # top1-top2 점수 차이를 판단하는 최소 차이
    min_margin_threshold: float = 0.08

    # 규칙 보정을 적용하기 위한 최소 키워드 매칭 수
    min_rule_match_count: int = 2

    # 검토필요로 분류할 낮은 모델 점수 기준
    review_needed_score_threshold: float = 0.40

    # 낮은 신뢰도일 때 단일 키워드만으로도 보정할 수 있는 선명한 카테고리
    single_match_override_labels: tuple[str, ...] = (
        "금융/투자",
        "노동/노사",
        "국제/통상",
        "정책/규제",
    )


class RuleEngine:
    def __init__(self, rules: dict[str, list[str]], config: RuleEngineConfig | None = None):
        self.rules = rules
        self.config = config or RuleEngineConfig()

    # 제목, 설명, 본문에서 카테고리별 키워드 매칭 점수 계산
    def calculate_scores(self, title: str, description: str = "", content: str = "") -> dict[str, int]:
        text = f"{clean_text(title)} {clean_text(description)} {clean_text(content)}".lower()

        scores: dict[str, int] = {}
        for label, keywords in self.rules.items():
            scores[label] = sum(1 for keyword in keywords if keyword.lower() in text)

        return scores

    # 모델 예측 결과와 규칙 점수를 함께 보고 최종 카테고리 결정
    def decide(
        self,
        title: str,
        description: str,
        content: str,
        prediction: ModelPrediction,
    ) -> RuleDecision:
        label_scores = self.calculate_scores(title, description, content)

        # 가장 많이 매칭된 규칙 카테고리 선택. 동점이면 금융/노사/통상/정책처럼 신호가 선명한 라벨을 우선한다.
        best_rule_label = (
            max(
                label_scores,
                key=lambda label: (
                    label_scores[label],
                    label in self.config.single_match_override_labels,
                ),
            )
            if label_scores
            else prediction.label
        )
        best_rule_score = label_scores.get(best_rule_label, 0)
        ambiguous = is_ambiguous(prediction.score, prediction.margin)

        # 모델 점수와 차이가 충분하면 모델 결과 유지
        if (
            prediction.score >= self.config.base_rule_override_threshold
            and prediction.margin >= self.config.min_margin_threshold
        ):
            return RuleDecision(
                final_label=prediction.label,
                rule_applied=False,
                rule_reason="",
                rule_best_label=best_rule_label,
                rule_match_count=best_rule_score,
                label_scores=label_scores,
            )

        # 규칙 키워드 근거가 충분하면 규칙 기반 카테고리로 보정
        if best_rule_score >= self.config.min_rule_match_count:
            return RuleDecision(
                final_label=best_rule_label,
                rule_applied=True,
                rule_reason=f"{best_rule_label} 키워드 {best_rule_score}개 매칭",
                rule_best_label=best_rule_label,
                rule_match_count=best_rule_score,
                label_scores=label_scores,
            )

        # 낮은 신뢰도 기사에서 금융/노사/통상/정책처럼 신호가 분명한 키워드는 더 적극 반영
        if (
            ambiguous
            and best_rule_label in self.config.single_match_override_labels
            and best_rule_score >= 1
        ):
            return RuleDecision(
                final_label=best_rule_label,
                rule_applied=True,
                rule_reason=f"낮은 신뢰도에서 {best_rule_label} 키워드 {best_rule_score}개 매칭",
                rule_best_label=best_rule_label,
                rule_match_count=best_rule_score,
                label_scores=label_scores,
            )

        # 모델 점수도 낮고 규칙 근거도 없으면 검토필요 처리
        if (
            best_rule_score == 0
            and (
                prediction.score < self.config.review_needed_score_threshold
                or ambiguous
            )
        ):
            return RuleDecision(
                final_label="검토필요",
                rule_applied=True,
                rule_reason="모델 점수 낮고 규칙 근거 없음",
                rule_best_label=best_rule_label,
                rule_match_count=best_rule_score,
                label_scores=label_scores,
            )

        # 그 외에는 모델 예측 결과 유지
        return RuleDecision(
            final_label=prediction.label,
            rule_applied=False,
            rule_reason="",
            rule_best_label=best_rule_label,
            rule_match_count=best_rule_score,
            label_scores=label_scores,
        )
