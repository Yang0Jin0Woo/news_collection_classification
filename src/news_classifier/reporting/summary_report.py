from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from news_classifier.models import ClassifiedNews


@dataclass(frozen=True)
class SummaryReport:
    total_count: int
    category_counts: dict[str, int]
    low_confidence_count: int
    rule_applied_count: int
    review_count: int
    error_count: int

    def to_markdown(self) -> str:
        lines = ["# 뉴스 분류 요약", ""]
        lines.append(f"- 전체 기사 수: {self.total_count}")
        lines.append(f"- 낮은 모델 신뢰도 기사 수: {self.low_confidence_count}")
        lines.append(f"- 규칙 보정 적용 기사 수: {self.rule_applied_count}")
        lines.append(f"- 검토필요 기사 수: {self.review_count}")
        lines.append(f"- 분류 오류 기사 수: {self.error_count}")
        lines.append("")
        lines.append("## 카테고리별 기사 수")
        for category, count in self.category_counts.items():
            lines.append(f"- {category}: {count}")
        return "\n".join(lines)


def build_summary(rows: list[ClassifiedNews]) -> SummaryReport:
    counter = Counter(row.rule_decision.final_label for row in rows)
    return SummaryReport(
        total_count=len(rows),
        category_counts=dict(counter),
        low_confidence_count=sum(
            1 for row in rows if row.model_confidence_level == "낮음"
        ),
        rule_applied_count=sum(1 for row in rows if row.decision_source == "RULE"),
        review_count=sum(1 for row in rows if row.decision_source == "REVIEW"),
        error_count=sum(1 for row in rows if row.decision_source == "ERROR"),
    )
