"""Synthetic rule-profile tests, not evidence about real article accuracy."""
from copy import deepcopy
import json

import pytest

from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.rules.default_rules import DEFAULT_RULE_SET
from news_classifier.rules.event_profile import (
    create_event_rule_profile, load_event_rule_profile, read_rule_proposals,
    rules_from_records,
)


RECORD = {"category": "금융/투자", "phrase": "합성 금융 사건", "strength": "STRONG",
          "evidence_event_ids": ["development-a", "development-b"]}


def evaluated_profile():
    rules = rules_from_records(DEFAULT_RULE_SET, [deepcopy(RECORD)])
    bindings = {"baseline": "synthetic-test", "input": "test"}
    validation = {
        "status": "passed", "passed": True, "failures": [],
        "newly_automatic_count": 2, "newly_automatic_independent_events": 2,
        "newly_automatic_wrong_count": 0, "new_wrong_decision_count": 0,
        "previous_correct_harmed_count": 0,
        "baseline": {"total": 2, "review_count": 2, "automatic_count": 0, "wrong_automatic_count": 0},
        "selected": {"total": 2, "review_count": 0, "automatic_count": 2, "wrong_automatic_count": 0},
        "per_rule_holdout": {"passed": True, "terms": [{
            "category": RECORD["category"], "phrase": RECORD["phrase"], "passed": True,
            "correct_rule_independent_events": 2, "wrong_rule_count": 0,
        }]},
    }
    return create_event_rule_profile(rules, validation, bindings, "a" * 64), bindings


def test_empty_profile_keeps_defaults():
    assert load_event_rule_profile("", DEFAULT_RULE_SET, {}) is DEFAULT_RULE_SET


def test_profile_preserves_thresholds_and_can_be_used_for_any_keyword(tmp_path):
    payload, bindings = evaluated_profile()
    path = tmp_path / "approved.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    rules = load_event_rule_profile(path, DEFAULT_RULE_SET, bindings)
    assert rules.decision == DEFAULT_RULE_SET.decision
    assert RuleEngine(rules).rule_only_decision(RECORD["phrase"], "", "") == "금융/투자"


@pytest.mark.parametrize("tamper", [
    lambda p: p.update(profile_type="event_rule_candidates"),
    lambda p: p.update(schema_version=True),
    lambda p: p.update(bindings={"wrong": True}),
    lambda p: p.update(dataset_sha256="not-a-hash"),
    lambda p: p["review_requirements"].update(human_confirmed=False),
    lambda p: p["validation"].update(passed=False),
    lambda p: p["validation"].update(newly_automatic_wrong_count=1),
    lambda p: p["validation"].update(previous_correct_harmed_count=1),
    lambda p: p["validation"]["per_rule_holdout"]["terms"][0].update(correct_rule_independent_events=1),
    lambda p: p["validation"]["per_rule_holdout"].update(terms=[]),
])
def test_unvalidated_or_incompatible_profiles_cannot_activate(tmp_path, tamper):
    payload, bindings = evaluated_profile()
    expected = deepcopy(bindings)
    tamper(payload)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_event_rule_profile(path, DEFAULT_RULE_SET, expected)


@pytest.mark.parametrize("updates", [
    {"category": "기타/무관"}, {"strength": []}, {"evidence_event_ids": []},
    {"evidence_event_ids": ["same", "same"]}, {"phrase": "제품"},
])
def test_invalid_proposals_are_rejected(updates):
    with pytest.raises(ValueError):
        rules_from_records(DEFAULT_RULE_SET, [{**RECORD, **updates}])


def test_candidate_file_can_be_evaluated_but_not_used_as_runtime_profile(tmp_path):
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps({"schema_version": 1, "profile_type": "event_rule_candidates",
                                "rules": [RECORD]}), encoding="utf-8")
    assert read_rule_proposals(path, DEFAULT_RULE_SET)
    with pytest.raises(ValueError, match="candidate/audit"):
        load_event_rule_profile(path, DEFAULT_RULE_SET, {})


def test_service_uses_confirmed_profile_for_all_keywords_without_loading_weights(tmp_path):
    from news_classifier.config import AppSettings
    from news_classifier.classifiers.zero_shot_classifier import ZeroShotNewsClassifier, HYPOTHESIS_TEMPLATE
    from news_classifier.models import CLASSIFICATION_INPUT_POLICY, NewsItem, ModelPrediction
    from news_classifier.rules.event_profile import runtime_bindings
    from news_classifier.service import build_pipeline

    path = tmp_path / "approved.json"
    settings = AppSettings(event_rule_profile_path=str(path),
                           decision_calibration_path="", confidence_calibration_path="")
    classifier = ZeroShotNewsClassifier(model_name=settings.classification_model,
        candidate_labels=DEFAULT_RULE_SET.candidate_labels, model_revision=settings.classification_model_revision,
        max_sequence_length=settings.max_sequence_length, batch_size=settings.classification_batch_size)
    payload, _ = evaluated_profile()
    payload["bindings"] = runtime_bindings(settings, classifier, CLASSIFICATION_INPUT_POLICY,
        HYPOTHESIS_TEMPLATE, DEFAULT_RULE_SET)
    path.write_text(json.dumps(payload), encoding="utf-8")
    pipeline = build_pipeline(settings)
    for keyword in ("AI", "자동차", "스포츠", "앞으로 입력할 임의의 검색어"):
        result = pipeline.postprocessor.process(
            NewsItem(keyword, RECORD["phrase"], ""), ModelPrediction("교육/취업", .2, .01))
        assert result.rule_decision.final_label == "금융/투자"
        assert result.rule_decision.rule_applied
    assert pipeline.postprocessor.rule_engine.rule_set.decision == DEFAULT_RULE_SET.decision
