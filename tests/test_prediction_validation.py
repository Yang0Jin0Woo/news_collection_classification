from types import SimpleNamespace

import pytest

from news_classifier.classifiers.prediction_validation import validate_model_predictions
from news_classifier.models import ModelPrediction
from news_classifier.rules.default_rules import DEFAULT_RULE_SET
from scripts import calibrate_confidence


@pytest.mark.parametrize("exception_type", [ValueError, RuntimeError])
@pytest.mark.parametrize("predictions,expected_count", [([], 1), ([ModelPrediction.failed()], 2),
                                                        ([ModelPrediction("사회", .8, .2)] * 2, 1)])
def test_prediction_count_error_precedes_failure_and_preserves_message(exception_type, predictions, expected_count):
    original = list(predictions)
    with pytest.raises(exception_type) as exc:
        validate_model_predictions(predictions, expected_count, count_error="count error",
                                   failure_error="failed prediction", exception_type=exception_type)
    assert type(exc.value) is exception_type
    assert str(exc.value) == "count error"
    assert predictions == original


@pytest.mark.parametrize("exception_type", [ValueError, RuntimeError])
def test_strict_prediction_validation_preserves_failure_message(exception_type):
    with pytest.raises(exception_type) as exc:
        validate_model_predictions([ModelPrediction.failed()], 1, count_error="count error",
                                   failure_error="failed prediction", exception_type=exception_type)
    assert type(exc.value) is exception_type
    assert str(exc.value) == "failed prediction"


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("allow_failure", [False, True])
def test_prediction_validation_preserves_order_values_and_objects(container, allow_failure):
    good = ModelPrediction("사회", .8, .2, ["사회", "정치"], [.8, .6])
    predictions = container([good, ModelPrediction.failed() if allow_failure else good])
    original = container(predictions)
    validate_model_predictions(predictions, 2, count_error="count error",
                               failure_error=None if allow_failure else "failed prediction")
    assert predictions == original
    assert all(before is after for before, after in zip(original, predictions, strict=True))


def test_prediction_validation_accepts_empty_aligned_batch():
    validate_model_predictions([], 0, count_error="count error", failure_error="failed prediction")


@pytest.mark.parametrize("predictions,message", [
    ([], "model prediction count does not match calibration cases"),
    ([ModelPrediction.failed()], "model failed while generating calibration predictions"),
])
def test_confidence_calibration_cli_keeps_error_policy_without_writing(tmp_path, monkeypatch, predictions, message):
    dataset, output = tmp_path / "source.csv", tmp_path / "profile.json"
    dataset.write_text("source to preserve", encoding="utf-8")
    rows = [{"title": "기사", "keyword": "검색어", "gold_label": "사회"}]
    pipeline = SimpleNamespace(
        classifier=SimpleNamespace(classify_many=lambda _: predictions),
        postprocessor=SimpleNamespace(rule_engine=SimpleNamespace(rule_set=DEFAULT_RULE_SET)),
    )
    monkeypatch.setattr(calibrate_confidence, "load_reviewed_cases", lambda *args, **kwargs: rows)
    monkeypatch.setattr(calibrate_confidence, "build_pipeline", lambda _: pipeline)
    monkeypatch.setattr(calibrate_confidence, "calibrate_confidence",
                        lambda *args, **kwargs: pytest.fail("invalid predictions must not calibrate"))
    monkeypatch.setattr("sys.argv", ["calibrate", "--dataset", str(dataset), "--output", str(output)])
    with pytest.raises(RuntimeError) as exc:
        calibrate_confidence.main()
    assert str(exc.value) == message
    assert dataset.read_text(encoding="utf-8") == "source to preserve"
    assert not output.exists()
    assert not output.with_suffix(".json.tmp").exists()
