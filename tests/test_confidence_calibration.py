import json

import pytest

from news_classifier.classifiers.confidence import (
    ConfidenceThresholds,
    confidence_level,
    load_confidence_thresholds,
)
from news_classifier.confidence_calibration import (
    CalibrationSample,
    calibrate_confidence,
)


def calibration_samples():
    return (
        [CalibrationSample(0.90, 0.30, True) for _ in range(20)]
        + [CalibrationSample(0.60, 0.10, idx < 18) for idx in range(20)]
        + [CalibrationSample(0.30, 0.01, idx < 5) for idx in range(20)]
    )


def test_calibration_finds_evidence_backed_high_and_medium_tiers():
    result = calibrate_confidence(
        calibration_samples(),
        high_accuracy_lower_bound=0.80,
        medium_accuracy_lower_bound=0.60,
        minimum_support=15,
    )

    assert result.high.support == 20
    assert result.medium.support == 20
    assert confidence_level(0.90, 0.30, result.thresholds) == "높음"
    assert confidence_level(0.60, 0.10, result.thresholds) == "보통"
    assert confidence_level(0.30, 0.01, result.thresholds) == "낮음"


def test_calibration_rejects_insufficient_data():
    with pytest.raises(ValueError, match="not enough"):
        calibrate_confidence(
            [CalibrationSample(0.90, 0.30, True)],
            minimum_support=2,
        )


def test_calibration_profile_validates_runtime_metadata(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": {"name": "model", "revision": "revision"},
                "candidate_labels": ["기술개발", "기타/무관"],
                "hypothesis_template": "template {}",
                "input_policy": {
                    "name": "article_only_v1",
                    "include_keyword": False,
                },
                "thresholds": ConfidenceThresholds().to_dict(),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    thresholds = load_confidence_thresholds(
        path,
        expected_model_name="model",
        expected_model_revision="revision",
        expected_candidate_labels=["기술개발", "기타/무관"],
        expected_input_policy="article_only_v1",
        expected_hypothesis_template="template {}",
    )

    assert thresholds == ConfidenceThresholds()

    with pytest.raises(ValueError, match="revision"):
        load_confidence_thresholds(
            path,
            expected_model_revision="different-revision",
        )


def test_calibration_rejects_missing_or_changed_topic_descriptions(tmp_path):
    path = tmp_path / "calibration.json"
    payload = {"schema_version": 1, "thresholds": ConfidenceThresholds().to_dict()}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="topic descriptions"):
        load_confidence_thresholds(path, expected_candidate_hypotheses=["event description"])
    payload["candidate_hypotheses"] = ["event description"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_confidence_thresholds(path, expected_candidate_hypotheses=["event description"]) == ConfidenceThresholds()
    with pytest.raises(ValueError, match="topic descriptions"):
        load_confidence_thresholds(path, expected_candidate_hypotheses=["changed description"])
