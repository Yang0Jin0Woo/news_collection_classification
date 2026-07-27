from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CategoryMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class EvaluationReport:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    total: int
    per_category: dict[str, CategoryMetrics]
    confusion_matrix: dict[str, dict[str, int]]

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def calculate_metrics(
    expected: list[str],
    predicted: list[str],
    labels: list[str],
) -> EvaluationReport:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted lengths must match")
    if not expected:
        raise ValueError("evaluation data is empty")

    unknown_expected = sorted(set(expected) - set(labels))
    if unknown_expected:
        raise ValueError(f"unknown expected labels: {unknown_expected}")

    extra_columns = ["검토필요", "분류실패", "기타"]
    columns = labels + extra_columns
    confusion = {
        label: {column: 0 for column in columns}
        for label in labels
    }

    for gold, prediction in zip(expected, predicted):
        normalized_prediction = prediction if prediction in columns else "기타"
        confusion[gold][normalized_prediction] += 1

    per_category: dict[str, CategoryMetrics] = {}
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(
            confusion[other_label][label]
            for other_label in labels
            if other_label != label
        )
        false_negative = sum(
            count
            for predicted_label, count in confusion[label].items()
            if predicted_label != label
        )
        support = sum(confusion[label].values())
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_category[label] = CategoryMetrics(precision, recall, f1, support)

    category_metrics = list(per_category.values())
    correct = sum(confusion[label][label] for label in labels)
    return EvaluationReport(
        accuracy=correct / len(expected),
        macro_precision=sum(metric.precision for metric in category_metrics) / len(labels),
        macro_recall=sum(metric.recall for metric in category_metrics) / len(labels),
        macro_f1=sum(metric.f1 for metric in category_metrics) / len(labels),
        total=len(expected),
        per_category=per_category,
        confusion_matrix=confusion,
    )


def render_comparison_markdown(
    reports: dict[str, EvaluationReport],
    labels: list[str],
) -> str:
    lines = [
        "# 뉴스 분류 성능 평가",
        "",
        "## 방식별 비교",
        "",
        "| 방식 | 정확도 | 거시 평균 정밀도 | 거시 평균 재현율 | 거시 평균 F1 | 평가 건수 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, report in reports.items():
        lines.append(
            f"| {name} | {report.accuracy:.4f} | {report.macro_precision:.4f} | "
            f"{report.macro_recall:.4f} | {report.macro_f1:.4f} | {report.total} |"
        )

    for name, report in reports.items():
        lines.extend([
            "",
            f"## {name} 카테고리별 성능",
            "",
            "| 카테고리 | 정밀도 | 재현율 | F1 | 건수 |",
            "|---|---:|---:|---:|---:|",
        ])
        for label in labels:
            metric = report.per_category[label]
            lines.append(
                f"| {label} | {metric.precision:.4f} | {metric.recall:.4f} | "
                f"{metric.f1:.4f} | {metric.support} |"
            )

        columns = labels + ["검토필요", "분류실패", "기타"]
        lines.extend([
            "",
            f"### {name} 혼동행렬",
            "",
            "| 실제＼예측 | " + " | ".join(columns) + " |",
            "|---|" + "|".join("---:" for _ in columns) + "|",
        ])
        for label in labels:
            row = report.confusion_matrix[label]
            lines.append(
                f"| {label} | " + " | ".join(str(row[column]) for column in columns) + " |"
            )

    return "\n".join(lines) + "\n"
