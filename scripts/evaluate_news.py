from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from news_classifier.config import AppSettings
from news_classifier.evaluation import calculate_metrics, render_comparison_markdown
from news_classifier.models import NewsItem
from news_classifier.rules.default_rules import CANDIDATE_LABELS
from news_classifier.service import build_pipeline


REQUIRED_COLUMNS = {
    "id",
    "split",
    "keyword",
    "title",
    "description",
    "gold_label",
    "review_status",
    "reviewed_by",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="수동 라벨 뉴스 분류 성능을 비교합니다.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=["development", "evaluation"], default="evaluation")
    parser.add_argument("--output-dir", default="evaluation_results")
    return parser


def load_reviewed_cases(path: Path, split: str) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"missing dataset columns: {sorted(missing_columns)}")
        rows = [row for row in reader if row["split"] == split]

    if not rows:
        raise ValueError(f"no rows found for split={split}")

    for row in rows:
        if row["review_status"] != "confirmed" or not row["reviewed_by"].strip():
            raise ValueError(f"case {row['id']} is not manually confirmed")
        if row["gold_label"] not in CANDIDATE_LABELS:
            raise ValueError(f"case {row['id']} has invalid gold label")

    covered_labels = {row["gold_label"] for row in rows}
    if covered_labels != set(CANDIDATE_LABELS):
        missing = sorted(set(CANDIDATE_LABELS) - covered_labels)
        raise ValueError(f"evaluation split does not cover all labels: {missing}")
    return rows


def rule_only_label(rule_engine, item: NewsItem) -> str:
    scores = rule_engine.calculate_scores(item.title, item.description, item.content)
    best_score = max(scores.values(), default=0)
    if best_score == 0:
        return "검토필요"
    best_labels = [label for label, score in scores.items() if score == best_score]
    return best_labels[0] if len(best_labels) == 1 else "검토필요"


def main() -> None:
    args = build_parser().parse_args()
    dataset_path = Path(args.dataset)
    rows = load_reviewed_cases(dataset_path, args.split)
    items = [
        NewsItem(
            keyword=row["keyword"],
            title=row["title"],
            link=row.get("link", ""),
            source=row.get("source", ""),
            published_at=row.get("published_at", ""),
            description=row["description"],
        )
        for row in rows
    ]

    pipeline = build_pipeline(AppSettings())
    predictions = pipeline.classifier.classify_many(
        [item.classification_text() for item in items]
    )
    if len(predictions) != len(items):
        raise RuntimeError("model prediction count does not match evaluation cases")

    rule_engine = pipeline.postprocessor.rule_engine
    expected = [row["gold_label"] for row in rows]
    predicted_by_method = {
        "모델 단독": [prediction.label for prediction in predictions],
        "규칙 단독": [rule_only_label(rule_engine, item) for item in items],
        "하이브리드": [
            pipeline.postprocessor.process(item, prediction).rule_decision.final_label
            for item, prediction in zip(items, predictions)
        ],
    }
    reports = {
        name: calculate_metrics(expected, predicted, CANDIDATE_LABELS)
        for name, predicted in predicted_by_method.items()
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_payload = {
        "dataset": str(dataset_path),
        "split": args.split,
        "model": AppSettings().classification_model,
        "labels": CANDIDATE_LABELS,
        "reports": {name: report.to_dict() for name, report in reports.items()},
    }
    (output_dir / "evaluation_results.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "evaluation_report.md").write_text(
        render_comparison_markdown(reports, CANDIDATE_LABELS),
        encoding="utf-8",
    )
    print(f"평가 보고서 저장: {output_dir / 'evaluation_report.md'}")


if __name__ == "__main__":
    main()
