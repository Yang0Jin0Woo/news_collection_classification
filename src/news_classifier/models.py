from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class NewsItem:
    keyword: str
    title: str
    link: str
    source: str = ""
    published_at: str = ""
    description: str = ""
    content: str = ""

    def classification_text(self) -> str:
        parts = [f"검색주제: {self.keyword}", f"기사제목: {self.title}"]
        if self.description:
            parts.append(f"기사설명: {self.description}")
        if self.content:
            parts.append(f"기사본문요약: {self.content[:500]}")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ModelPrediction:
    label: str
    score: float
    margin: float
    top3_labels: list[str] = field(default_factory=list)
    top3_scores: list[float] = field(default_factory=list)

    @staticmethod
    def failed() -> "ModelPrediction":
        return ModelPrediction("분류실패", 0.0, 0.0, [], [])


@dataclass(frozen=True)
class RuleDecision:
    final_label: str
    rule_applied: bool
    rule_reason: str
    rule_best_label: str
    rule_match_count: int
    label_scores: dict[str, int]


@dataclass(frozen=True)
class ClassifiedNews:
    item: NewsItem
    model_prediction: ModelPrediction
    rule_decision: RuleDecision
    confidence_level: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_row(self) -> dict:
        return {
            "keyword": self.item.keyword,
            "title": self.item.title,
            "source": self.item.source,
            "published_at": self.item.published_at,
            "link": self.item.link,
            "description": self.item.description,
            "content": self.item.content,
            "classification_text": self.item.classification_text(),
            "model_category": self.model_prediction.label,
            "model_category_score": round(self.model_prediction.score, 4),
            "score_margin": round(self.model_prediction.margin, 4),
            "final_category": self.rule_decision.final_label,
            "confidence_level": self.confidence_level,
            "rule_applied": "Y" if self.rule_decision.rule_applied else "N",
            "rule_reason": self.rule_decision.rule_reason,
            "rule_best_label": self.rule_decision.rule_best_label,
            "rule_match_count": self.rule_decision.rule_match_count,
            "top3_labels": ", ".join(self.model_prediction.top3_labels),
            "top3_scores": ", ".join(str(round(x, 4)) for x in self.model_prediction.top3_scores),
            "created_at": self.created_at,
        }
