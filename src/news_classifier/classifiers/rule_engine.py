from __future__ import annotations

from dataclasses import dataclass

from news_classifier.models import ModelPrediction, RuleDecision
from news_classifier.utils.text import clean_text


@dataclass(frozen=True)
class RuleEngineConfig:
    base_rule_override_threshold: float = 0.55
    min_margin_threshold: float = 0.08
    min_rule_match_count: int = 2
    review_needed_score_threshold: float = 0.40


class RuleEngine:
    def __init__(self, rules: dict[str, list[str]], config: RuleEngineConfig | None = None):
        self.rules = rules
        self.config = config or RuleEngineConfig()

    def calculate_scores(self, title: str, description: str = "", content: str = "") -> dict[str, int]:
        text = f"{clean_text(title)} {clean_text(description)} {clean_text(content)}".lower()
        scores: dict[str, int] = {}
        for label, keywords in self.rules.items():
            scores[label] = sum(1 for keyword in keywords if keyword.lower() in text)
        return scores

    def decide(
        self,
        title: str,
        description: str,
        content: str,
        prediction: ModelPrediction,
    ) -> RuleDecision:
        label_scores = self.calculate_scores(title, description, content)
        best_rule_label = max(label_scores, key=label_scores.get) if label_scores else prediction.label
        best_rule_score = label_scores.get(best_rule_label, 0)

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

        if best_rule_score >= self.config.min_rule_match_count:
            return RuleDecision(
                final_label=best_rule_label,
                rule_applied=True,
                rule_reason=f"{best_rule_label} 키워드 {best_rule_score}개 매칭",
                rule_best_label=best_rule_label,
                rule_match_count=best_rule_score,
                label_scores=label_scores,
            )

        if prediction.score < self.config.review_needed_score_threshold and best_rule_score == 0:
            return RuleDecision(
                final_label="검토필요",
                rule_applied=True,
                rule_reason="모델 점수 낮고 규칙 근거 없음",
                rule_best_label=best_rule_label,
                rule_match_count=best_rule_score,
                label_scores=label_scores,
            )

        return RuleDecision(
            final_label=prediction.label,
            rule_applied=False,
            rule_reason="",
            rule_best_label=best_rule_label,
            rule_match_count=best_rule_score,
            label_scores=label_scores,
        )
