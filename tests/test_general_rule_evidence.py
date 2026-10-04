"""합성 근거로 일반 주제 규칙의 안전한 연결을 검증. 실제 정확도 평가 아님."""

import argparse
import copy
import csv
import json
from dataclasses import replace

import pytest

from scripts import evaluate_news, review_rule_candidates
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.classifiers.topic_descriptions import (
    GENERAL_NEWS_LABELS,
    LEGACY_CANDIDATE_LABELS,
)
from news_classifier.models import ModelPrediction
from news_classifier.rules import default_rules
from news_classifier.rules.policy import (
    RuleStrength,
    RuleTerm,
    validate_development_rule_errors,
    validate_development_rule_evidence,
    validate_rule_set,
)


def synthetic_term(phrase="합성 검증 구문", *, events=("synthetic-1", "synthetic-2")):
    return RuleTerm(
        phrase,
        strength=RuleStrength.STRONG,
        origin="development",
        evidence_event_ids=events,
    )


def activate_general_rule(monkeypatch, label, term=None):
    monkeypatch.setattr(
        default_rules, "_DEVELOPMENT_RULE_TERMS",
        ((label, term or synthetic_term()),),
    )
    return default_rules._build_default_rule_set()


def confirmed_support(event, **changes):
    return {
        "event_id": event,
        "split": "development",
        "review_status": "confirmed",
        "reviewed_by": "synthetic-human-reviewer",
        "suggested_label": "",
        "gold_label": "교육/취업",
        "title": "합성 검증 구문",
        "description": "",
        "content": "",
        **changes,
    }


@pytest.mark.parametrize("label", GENERAL_NEWS_LABELS)
def test_default_general_topics_still_have_no_unverified_direct_rules(label):
    policy = default_rules.DEFAULT_RULE_SET.label_policy(label)
    assert policy.no_direct_rules
    assert policy.terms == ()
    assert label not in default_rules.RULE_KEYWORDS
    assert default_rules._DEVELOPMENT_RULE_TERMS == ()


@pytest.mark.parametrize("label", GENERAL_NEWS_LABELS)
def test_evidenced_general_topic_can_correct_only_uncertain_model(monkeypatch, label):
    rules = activate_general_rule(monkeypatch, label)
    policy = rules.label_policy(label)
    assert not policy.no_direct_rules
    assert policy.terms == (synthetic_term(),)
    assert rules.decision == default_rules.DEFAULT_RULE_SET.decision
    engine = RuleEngine(rules)
    assert engine.rule_only_decision("합성 검증 구문") == label
    result = engine.decide(
        "합성 검증 구문", "", "", ModelPrediction("기타/무관", 0.3, 0.01),
    )
    assert result.final_label == label
    assert result.rule_applied
    confident = engine.decide(
        "합성 검증 구문", "", "", ModelPrediction("기타/무관", 0.8, 0.2),
    )
    assert confident.final_label == "기타/무관"
    assert not confident.rule_applied


@pytest.mark.parametrize("term, error", [
    (RuleTerm("합성 검증 구문"), "development origin"),
    (synthetic_term(events=()), "event evidence"),
    (synthetic_term(events=("synthetic-1",)), "insufficient"),
    (synthetic_term(events=("synthetic-1", "synthetic-1")), "duplicates"),
    (synthetic_term(events=("EV-1", " ev-1 ")), "duplicates"),
])
def test_general_rule_rejects_unconfirmed_or_insufficient_evidence(monkeypatch, term, error):
    with pytest.raises(ValueError, match=error):
        activate_general_rule(monkeypatch, "교육/취업", term)


@pytest.mark.parametrize("label", ["기타/무관", "없는 주제"])
def test_other_and_unknown_labels_cannot_be_promoted(monkeypatch, label):
    with pytest.raises(ValueError, match="unknown or no-direct"):
        activate_general_rule(monkeypatch, label)


def test_general_rule_evidence_still_requires_confirmed_development_errors(monkeypatch):
    rules = activate_general_rule(monkeypatch, "교육/취업")
    rows = [
        confirmed_support(event)
        for event in ("synthetic-1", "synthetic-2")
    ]
    validate_development_rule_evidence(rules, rows)
    validate_development_rule_errors(rules, rows, ["검토필요", "노동/노사"])
    pending = copy.deepcopy(rows)
    pending[0]["review_status"] = "pending"
    with pytest.raises(ValueError, match="confirmed development"):
        validate_development_rule_evidence(rules, pending)
    with pytest.raises(ValueError, match="baseline error"):
        validate_development_rule_errors(rules, rows, ["교육/취업", "노동/노사"])


@pytest.mark.parametrize("changes, error", [
    ({"reviewed_by": " "}, "human reviewer"),
    ({"suggested_label": "교육/취업"}, "suggested labels"),
    ({"event_id": ""}, "does not exist"),
    ({"title": "다른 사건", "keyword": "합성 검증 구문"}, "does not score"),
    ({"title": "다른 사건", "description": "합성 검증 구문",
      "source": "합성 검증 구문"}, "does not score"),
])
def test_rule_evidence_requires_blind_human_confirmation_and_article_support(monkeypatch, changes, error):
    rules = activate_general_rule(monkeypatch, "교육/취업")
    rows = [confirmed_support("synthetic-1", **changes), confirmed_support("synthetic-2")]
    with pytest.raises(ValueError, match=error):
        validate_development_rule_evidence(rules, rows)


@pytest.mark.parametrize("field", ["title", "description", "content"])
def test_confirmed_rule_support_can_be_in_each_usable_article_field(monkeypatch, field):
    rules = activate_general_rule(monkeypatch, "교육/취업")
    rows = [
        confirmed_support(event, **{"title": "다른 제목", field: "합성 검증 구문"})
        for event in ("synthetic-1", "synthetic-2")
    ]
    validate_development_rule_evidence(rules, rows)


def test_supporting_phrase_must_actually_score_after_longest_overlap(monkeypatch):
    rules = activate_general_rule(monkeypatch, "교육/취업")
    labels = list(rules.labels)
    index = next(index for index, policy in enumerate(labels) if policy.label == "금융/투자")
    labels[index] = replace(labels[index], terms=labels[index].terms + (
        RuleTerm("합성 검증 구문 상세 사건", RuleStrength.STRONG),
    ))
    rules = replace(rules, labels=tuple(labels))
    rows = [
        confirmed_support(event, title="합성 검증 구문 상세 사건")
        for event in ("synthetic-1", "synthetic-2")
    ]
    with pytest.raises(ValueError, match="does not score"):
        validate_development_rule_evidence(rules, rows)


def test_baseline_error_must_be_on_same_phrase_supporting_article(monkeypatch):
    rules = activate_general_rule(monkeypatch, "교육/취업")
    rows = [
        confirmed_support("synthetic-1"),
        confirmed_support("synthetic-1", title="구문이 없는 같은 사건의 다른 기사"),
        confirmed_support("synthetic-2"),
    ]
    validate_development_rule_evidence(rules, rows)
    with pytest.raises(ValueError, match="same phrase-supporting article"):
        validate_development_rule_errors(rules, rows, ["교육/취업", "노동/노사", "노동/노사"])


def test_pre_rule_hybrid_review_is_a_valid_baseline_problem(monkeypatch):
    rules = activate_general_rule(monkeypatch, "교육/취업")
    rows = [confirmed_support(event) for event in ("synthetic-1", "synthetic-2")]
    validate_development_rule_errors(rules, rows, ["검토필요", "검토필요"])


@pytest.mark.parametrize("baseline_label", ["", "분류실패", "알 수 없는 출력"])
def test_failed_or_unknown_baseline_is_not_classification_error_evidence(monkeypatch, baseline_label):
    rules = activate_general_rule(monkeypatch, "교육/취업")
    rows = [confirmed_support(event) for event in ("synthetic-1", "synthetic-2")]
    with pytest.raises(ValueError, match="baseline error"):
        validate_development_rule_errors(rules, rows, [baseline_label, "노동/노사"])


def test_all_general_rule_priorities_and_empty_baselines_remain_valid(monkeypatch):
    monkeypatch.setattr(
        default_rules, "_DEVELOPMENT_RULE_TERMS",
        tuple(
            (label, synthetic_term(f"합성 검증 구문 {index}"))
            for index, label in enumerate(GENERAL_NEWS_LABELS)
        ),
    )
    rules = default_rules._build_default_rule_set()
    validate_rule_set(rules)
    priorities = [policy.tie_priority for policy in rules.direct_rule_labels]
    assert len(set(priorities)) == len(priorities)
    assert not rules.label_policy("기타/무관").terms
    baseline = evaluate_news.baseline_rule_set(rules)
    validate_rule_set(baseline)
    assert baseline == default_rules.DEFAULT_RULE_SET
    assert all(baseline.label_policy(label).no_direct_rules for label in GENERAL_NEWS_LABELS)
    # 비교용 이전 열 개 주제를 유지하며 현재 활성 규칙은 변경하지 않는다.
    legacy = evaluate_news.legacy_rule_set(rules)
    validate_rule_set(legacy)
    assert legacy.candidate_labels == LEGACY_CANDIDATE_LABELS
    assert all(not term.context_only for policy in legacy.labels for term in policy.terms)
    assert all(not rules.label_policy(label).no_direct_rules for label in GENERAL_NEWS_LABELS)


@pytest.mark.parametrize("label", GENERAL_NEWS_LABELS)
def test_candidate_audit_accepts_model_only_general_topics_without_confirming(label):
    rows = [{"id": "synthetic", "title": "합성 검증 구문", "review_status": "pending"}]
    before = copy.deepcopy(rows)
    result = review_rule_candidates.audit_candidate("합성 검증 구문", label, rows)
    assert result["status"] == "pending_human_review"
    assert result["proposed_category"] == label
    assert result["human_confirmation_required"]
    assert result["matching_article_count"] == 1
    assert default_rules.DEFAULT_RULE_SET.label_policy(label).no_direct_rules
    assert rows == before


@pytest.mark.parametrize("label", ["기타/무관", "없는 주제"])
def test_candidate_audit_rejects_other_and_unknown_categories(label):
    with pytest.raises(ValueError, match="existing non-other"):
        review_rule_candidates.audit_candidate("합성 검증 구문", label, [])


@pytest.mark.parametrize("value", [
    "교육/취업", "교육/취업:", "기타/무관:구문", "없는 주제:구문",
])
def test_explicit_candidate_rejects_invalid_or_other_category(value):
    with pytest.raises(argparse.ArgumentTypeError):
        review_rule_candidates.parse_candidate(value)


def test_explicit_candidates_replace_defaults_and_leave_dataset_pending(tmp_path, monkeypatch):
    dataset = tmp_path / "synthetic.csv"
    with dataset.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "title", "split", "review_status"])
        writer.writeheader()
        writer.writerow({
            "id": "synthetic", "title": "합성 교육 표현과 합성 스포츠 표현",
            "split": "development", "review_status": "pending",
        })
    original = dataset.read_bytes()
    output = tmp_path / "audit.json"
    monkeypatch.setattr("sys.argv", [
        "review", "--dataset", str(dataset), "--output", str(output),
        "--candidate", "교육/취업:합성 교육 표현",
        "--candidate", "스포츠:합성 스포츠 표현",
    ])
    review_rule_candidates.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert [candidate["proposed_category"] for candidate in report["candidates"]] == [
        "교육/취업", "스포츠",
    ]
    assert all(candidate["human_confirmation_required"] for candidate in report["candidates"])
    assert dataset.read_bytes() == original


def test_existing_employment_candidate_suggestions_use_education_taxonomy():
    education = review_rule_candidates.CANDIDATE_PHRASES["교육/취업"]
    labor = review_rule_candidates.CANDIDATE_PHRASES["노동/노사"]
    assert "신입생 모집" in education
    assert "채용 공고" in education
    assert "채용 공고" not in labor
    assert "직원 해고" in labor
