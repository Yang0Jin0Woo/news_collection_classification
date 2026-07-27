from __future__ import annotations

import argparse
import json
from pathlib import Path

from news_classifier.classifiers.zero_shot_classifier import HYPOTHESIS_TEMPLATE
from news_classifier.confidence_calibration import (
    CalibrationSample,
    calibrate_confidence,
)
from news_classifier.config import AppSettings
from news_classifier.evaluation_dataset import (
    dataset_sha256,
    evaluation_input,
    load_reviewed_cases,
    rows_to_items,
)
from news_classifier.models import CLASSIFICATION_INPUT_POLICY
from news_classifier.rules.default_rules import CANDIDATE_LABELS
from news_classifier.service import build_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="확정된 development 뉴스로 모델 신뢰도 기준을 보정합니다."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--output",
        default="evaluation_results/confidence_calibration.json",
    )
    parser.add_argument("--minimum-support", type=int, default=15)
    parser.add_argument("--high-lower-bound", type=float, default=0.80)
    parser.add_argument("--medium-lower-bound", type=float, default=0.65)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_path = Path(args.dataset)
    rows = load_reviewed_cases(
        dataset_path,
        "development",
        CANDIDATE_LABELS,
    )
    items = rows_to_items(rows)
    settings = AppSettings(confidence_calibration_path="")
    pipeline = build_pipeline(settings)
    predictions = pipeline.classifier.classify_many(
        [evaluation_input(item) for item in items]
    )
    if len(predictions) != len(items):
        raise RuntimeError("model prediction count does not match calibration cases")
    if any(prediction.label == "분류실패" for prediction in predictions):
        raise RuntimeError("model failed while generating calibration predictions")

    samples = [
        CalibrationSample(
            score=prediction.score,
            margin=prediction.margin,
            correct=prediction.label == row["gold_label"],
        )
        for row, prediction in zip(rows, predictions, strict=True)
    ]
    result = calibrate_confidence(
        samples,
        high_accuracy_lower_bound=args.high_lower_bound,
        medium_accuracy_lower_bound=args.medium_lower_bound,
        minimum_support=args.minimum_support,
    )

    payload = {
        "schema_version": 1,
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha256(dataset_path),
        "split": "development",
        "model": {
            "name": settings.classification_model,
            "revision": settings.classification_model_revision,
        },
        "candidate_labels": CANDIDATE_LABELS,
        "hypothesis_template": HYPOTHESIS_TEMPLATE,
        "input_policy": {
            "name": CLASSIFICATION_INPUT_POLICY,
            "include_keyword": False,
        },
        "targets": {
            "high_accuracy_lower_bound": args.high_lower_bound,
            "medium_accuracy_lower_bound": args.medium_lower_bound,
            "minimum_support": args.minimum_support,
        },
        "thresholds": result.thresholds.to_dict(),
        "evidence": {
            "total": result.total,
            "high": result.high.to_dict(),
            "medium": result.medium.to_dict(),
        },
    }
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"calibration output already exists; use --force: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    print(f"신뢰도 보정 파일 저장: {output_path}")


if __name__ == "__main__":
    main()
