"""Synthetic script coverage; not human confirmation of real-world accuracy."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from scripts import compare_rule_context
from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.models import ModelPrediction
from news_classifier.rules.default_rules import DEFAULT_RULE_SET


def rows():
    return [{"id": "fixture-1", "title": "학교 인재 양성 소식", "content": "학생을 양성한다. 교육 체계를 구축한다. 관계자는 과거 회사 합병과 사업 협력을 맡았다.",
             "split": "development", "gold_label": "교육/취업", "review_status": "confirmed", "reviewed_by": "fixture reviewer"}]


def fake_pipeline(prediction=None):
    return SimpleNamespace(
        classifier=SimpleNamespace(classify_many=lambda texts: [prediction or ModelPrediction("기업동향", .25, .01) for _ in texts]),
        postprocessor=ClassificationPostProcessor(RuleEngine(DEFAULT_RULE_SET)),
    )


def test_comparison_preserves_sources_and_separates_wrong_automatic_from_review():
    source = rows()
    original = deepcopy(source)
    result = compare_rule_context.compare_cases(source, fake_pipeline())
    assert result["metrics"]["before"]["wrong_decided_count"] == 1
    assert result["metrics"]["experimental_candidate"]["wrong_decided_count"] == 0
    assert result["metrics"]["experimental_candidate"]["review_count"] == 1
    assert result["cases"][0]["previous_wrong_corrected"] is False
    assert result["wrong_automatic_to_review_count"] == 1
    assert result["all_topics_covered"] is False
    assert source == original


def test_confident_correct_model_is_retained():
    result = compare_rule_context.compare_cases(rows(), fake_pipeline(ModelPrediction("교육/취업", .9, .2)))
    assert result["metrics"]["before"]["accuracy"] == 1
    assert result["metrics"]["experimental_candidate"]["accuracy"] == 1
    assert result["previous_correct_harmed_count"] == 0


def write_dataset(path, source):
    path.write_text(json.dumps({"profile_type": "human_confirmed_rule_context_cases", "dataset_role": "development", "cases": source}, ensure_ascii=False), encoding="utf-8")


@pytest.mark.parametrize("field,value", [
    ("gold_label", ""), ("review_status", "pending"), ("reviewed_by", ""),
    ("split", "evaluation"), ("suggested_label", "교육/취업"),
])
def test_unconfirmed_or_heldout_data_rejected_before_inference(tmp_path, field, value):
    source = rows()
    source[0][field] = value
    p = tmp_path / "fixture.json"
    write_dataset(p, source)
    with pytest.raises(ValueError):
        compare_rule_context.load_cases(p)


def test_count_mismatch_and_failed_prediction_abort():
    pipeline = fake_pipeline()
    pipeline.classifier.classify_many = lambda texts: []
    with pytest.raises(ValueError, match="모델 실행 실패"):
        compare_rule_context.compare_cases(rows(), pipeline)
    with pytest.raises(ValueError, match="모델 실행 실패"):
        compare_rule_context.compare_cases(rows(), fake_pipeline(ModelPrediction.failed()))


def test_cli_source_and_existing_output_are_preserved(tmp_path, monkeypatch):
    dataset, output = tmp_path / "fixture.json", tmp_path / "result.json"
    write_dataset(dataset, rows())
    original = dataset.read_bytes()
    monkeypatch.setattr(compare_rule_context, "build_pipeline", lambda settings: fake_pipeline())
    compare_rule_context.main(["--dataset", str(dataset), "--output", str(output)])
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["profile_type"] == "rule_context_comparison_report_not_runtime_profile"
    assert not payload["automatic_activation"] and not payload["independent_evaluation_completed"]
    assert dataset.read_bytes() == original
    report = output.read_bytes()
    with pytest.raises(SystemExit):
        compare_rule_context.main(["--dataset", str(dataset), "--output", str(output)])
    assert output.read_bytes() == report


def test_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "duplicate.json"
    write_dataset(p, rows() * 2)
    with pytest.raises(ValueError):
        compare_rule_context.load_cases(p)
