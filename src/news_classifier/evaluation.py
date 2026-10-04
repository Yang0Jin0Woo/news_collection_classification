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
    coverage: float
    decided_accuracy: float
    review_count: int
    error_count: int
    per_category: dict[str, CategoryMetrics]
    confusion_matrix: dict[str, dict[str, int]]

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "review_rate": self.review_count / self.total,
            "wrong_decided_count": sum(
                count
                for gold, row in self.confusion_matrix.items()
                for prediction, count in row.items()
                if prediction in self.per_category and prediction != gold
            ),
        }


def calculate_rule_correction_metrics(
    expected: list[str],
    model_labels: list[str],
    final_labels: list[str],
    rule_applied: list[bool],
) -> dict[str, int | float]:
    """규칙 적용과 실제 라벨 변경을 구분하고, 정답을 오답으로 바꾼 경우 별도 집계."""
    if not (len(expected) == len(model_labels) == len(final_labels) == len(rule_applied)):
        raise ValueError("rule correction inputs must have matching lengths")
    applied = [
        (gold, model, final)
        for gold, model, final, used in zip(
            expected, model_labels, final_labels, rule_applied, strict=True
        )
        if used
    ]
    changed = [(gold, model, final) for gold, model, final in applied if model != final]
    wrong = sum(gold != final for gold, _, final in applied)
    return {
        "rule_applied_count": len(applied),
        "rule_changed_count": len(changed),
        "wrong_rule_count": wrong,
        "wrong_rule_rate": _safe_divide(wrong, len(applied)),
        "corrected_count": sum(model != gold and final == gold for gold, model, final in changed),
        "harmful_change_count": sum(model == gold and final != gold for gold, model, final in changed),
    }


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
    decided_pairs = [
        (gold, prediction)
        for gold, prediction in zip(expected, predicted)
        if prediction in labels
    ]
    decided_correct = sum(
        gold == prediction for gold, prediction in decided_pairs
    )
    return EvaluationReport(
        accuracy=correct / len(expected),
        macro_precision=sum(metric.precision for metric in category_metrics) / len(labels),
        macro_recall=sum(metric.recall for metric in category_metrics) / len(labels),
        macro_f1=sum(metric.f1 for metric in category_metrics) / len(labels),
        total=len(expected),
        coverage=len(decided_pairs) / len(expected),
        decided_accuracy=_safe_divide(decided_correct, len(decided_pairs)),
        review_count=sum(prediction == "검토필요" for prediction in predicted),
        error_count=sum(prediction == "분류실패" for prediction in predicted),
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
        "| 방식 | 정확도 | 거시 평균 F1 | 결정 커버리지 | 결정 건 정확도 | 검토 비율 | 오분류 | 검토 | 오류 | 평가 건수 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, report in reports.items():
        lines.append(
            f"| {name} | {report.accuracy:.4f} | {report.macro_f1:.4f} | "
            f"{report.coverage:.4f} | {report.decided_accuracy:.4f} | "
            f"{report.review_count / report.total:.4f} | "
            f"{report.to_dict()['wrong_decided_count']} | "
            f"{report.review_count} | {report.error_count} | {report.total} |"
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
