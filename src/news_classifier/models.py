from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum


DECISION_SOURCE_MODEL = "MODEL"
DECISION_SOURCE_RULE = "RULE"
DECISION_SOURCE_REVIEW = "REVIEW"
DECISION_SOURCE_ERROR = "ERROR"

FINAL_STATUS_DECIDED = "DECIDED"
FINAL_STATUS_REVIEW_REQUIRED = "REVIEW_REQUIRED"
FINAL_STATUS_ERROR = "ERROR"


class PipelineStatus(str, Enum):
    SUCCESS = "SUCCESS"
    NO_RESULTS = "NO_RESULTS"
    NETWORK_ERROR = "NETWORK_ERROR"
    COLLECTION_ERROR = "COLLECTION_ERROR"
    MODEL_ERROR = "MODEL_ERROR"


@dataclass(frozen=True)
class PipelineError:
    stage: str
    code: str
    message: str


@dataclass(frozen=True)
class PipelineStatistics:
    requested_limit: int = 0
    collected_count: int = 0
    deduplicated_count: int = 0
    classified_count: int = 0
    rule_applied_count: int = 0
    review_required_count: int = 0


@dataclass(frozen=True)
class PipelineResult:
    status: PipelineStatus
    results: list["ClassifiedNews"] = field(default_factory=list)
    errors: list[PipelineError] = field(default_factory=list)
    statistics: PipelineStatistics = field(default_factory=PipelineStatistics)

    @property
    def succeeded(self) -> bool:
        return self.status in {
            PipelineStatus.SUCCESS,
            PipelineStatus.NO_RESULTS,
        }


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
    model_confidence_level: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def model_confidence(self) -> float:
        """최종 라벨이 아닌 모델 원예측에 대한 점수다."""
        return self.model_prediction.score

    @property
    def decision_source(self) -> str:
        if self.rule_decision.final_label == "분류실패":
            return DECISION_SOURCE_ERROR
        if self.rule_decision.final_label == "검토필요":
            return DECISION_SOURCE_REVIEW
        if self.rule_decision.rule_applied:
            return DECISION_SOURCE_RULE
        return DECISION_SOURCE_MODEL

    @property
    def review_required(self) -> bool:
        return self.decision_source in {
            DECISION_SOURCE_REVIEW,
            DECISION_SOURCE_ERROR,
        }

    @property
    def final_decision_status(self) -> str:
        if self.decision_source == DECISION_SOURCE_ERROR:
            return FINAL_STATUS_ERROR
        if self.review_required:
            return FINAL_STATUS_REVIEW_REQUIRED
        return FINAL_STATUS_DECIDED

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
            "model_confidence": round(self.model_confidence, 4),
            "model_confidence_level": self.model_confidence_level,
            "score_margin": round(self.model_prediction.margin, 4),
            "final_category": self.rule_decision.final_label,
            "decision_source": self.decision_source,
            "review_required": self.review_required,
            "final_decision_status": self.final_decision_status,
            "rule_applied": "Y" if self.rule_decision.rule_applied else "N",
            "rule_reason": self.rule_decision.rule_reason,
            "rule_best_label": self.rule_decision.rule_best_label,
            "rule_match_count": self.rule_decision.rule_match_count,
            "top3_labels": ", ".join(self.model_prediction.top3_labels),
            "top3_scores": ", ".join(str(round(x, 4)) for x in self.model_prediction.top3_scores),
            "created_at": self.created_at,
        }


CLASSIFIED_NEWS_COLUMNS = [
    "keyword",
    "title",
    "source",
    "published_at",
    "link",
    "description",
    "content",
    "classification_text",
    "model_category",
    "model_category_score",
    "model_confidence",
    "model_confidence_level",
    "score_margin",
    "final_category",
    "decision_source",
    "review_required",
    "final_decision_status",
    "rule_applied",
    "rule_reason",
    "rule_best_label",
    "rule_match_count",
    "top3_labels",
    "top3_scores",
    "created_at",
]
