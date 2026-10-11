"""Shared output checks; callers retain their error and evaluation policies."""
from __future__ import annotations

from collections.abc import Sequence

from news_classifier.models import ModelPrediction


def validate_model_predictions(
    predictions: Sequence[ModelPrediction],
    expected_count: int,
    *,
    count_error: str,
    failure_error: str | None = None,
    exception_type: type[Exception] = ValueError,
) -> None:
    """Check alignment; omit failure_error to retain failed predictions for evaluation."""
    if len(predictions) != expected_count:
        raise exception_type(count_error)
    if failure_error is not None and any(prediction.label == "분류실패" for prediction in predictions):
        raise exception_type(failure_error)
