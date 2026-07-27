from dataclasses import replace

import pytest

from news_classifier.rules.default_rules import (
    DEFAULT_RULE_SET,
    validate_rule_configuration,
)
from news_classifier.rules.policy import (
    LabelRulePolicy,
    RuleDecisionPolicy,
    RuleSet,
    RuleStrength,
    RuleTerm,
    rule_set_fingerprint,
    validate_development_rule_errors,
    validate_development_rule_evidence,
)


def simple_rule_set(
    labels: tuple[LabelRulePolicy, ...] | None = None,
) -> RuleSet:
    return RuleSet(
        version="test-rules",
        matcher_version="test-matcher",
        labels=labels or (
            LabelRulePolicy("기술개발", (RuleTerm("연구"),), 20),
            LabelRulePolicy("제품/서비스", (RuleTerm("출시"),), 10),
            LabelRulePolicy(
                "기타/무관",
                (),
                30,
                no_direct_rules=True,
            ),
        ),
        domain_terms=("ai",),
        unrelated_signal_groups=(("야구",),),
        decision=RuleDecisionPolicy(),
        other_label="기타/무관",
        technology_label="기술개발",
    )


def test_default_rule_configuration_is_valid():
    validate_rule_configuration()


def test_rule_configuration_rejects_keyword_shared_by_labels():
    rule_set = simple_rule_set(labels=(
        LabelRulePolicy("기술개발", (RuleTerm("AI"),), 20),
        LabelRulePolicy("제품/서비스", (RuleTerm("ai"),), 10),
        LabelRulePolicy("기타/무관", (), 30, no_direct_rules=True),
    ))

    with pytest.raises(ValueError, match="multiple labels"):
        validate_rule_configuration(rule_set)


def test_label_without_direct_rules_is_supported():
    validate_rule_configuration(simple_rule_set())


def test_label_without_direct_rules_cannot_contain_terms():
    rule_set = simple_rule_set(labels=(
        LabelRulePolicy("기술개발", (RuleTerm("연구"),), 20),
        LabelRulePolicy("제품/서비스", (RuleTerm("출시"),), 10),
        LabelRulePolicy(
            "기타/무관",
            (RuleTerm("스포츠"),),
            30,
            no_direct_rules=True,
        ),
    ))

    with pytest.raises(ValueError, match="no direct rules"):
        validate_rule_configuration(rule_set)


def test_rule_configuration_rejects_duplicate_tie_priority():
    rule_set = simple_rule_set(labels=(
        LabelRulePolicy("기술개발", (RuleTerm("연구"),), 10),
        LabelRulePolicy("제품/서비스", (RuleTerm("출시"),), 10),
        LabelRulePolicy("기타/무관", (), 30, no_direct_rules=True),
    ))

    with pytest.raises(ValueError, match="priorities"):
        validate_rule_configuration(rule_set)


def test_development_rule_requires_confirmed_event_evidence():
    labels = list(simple_rule_set().labels)
    labels[0] = replace(
        labels[0],
        terms=(RuleTerm("새 규칙", origin="development"),),
    )
    rule_set = replace(simple_rule_set(), labels=tuple(labels))

    with pytest.raises(ValueError, match="event evidence"):
        validate_rule_configuration(rule_set)


def test_spacing_aliases_must_use_the_same_strength():
    rule_set = simple_rule_set(labels=(
        LabelRulePolicy(
            "기술개발",
            (
                RuleTerm("AI반도체", RuleStrength.STRONG),
                RuleTerm("AI 반도체", RuleStrength.WEAK),
            ),
            20,
        ),
        LabelRulePolicy("제품/서비스", (RuleTerm("출시"),), 10),
        LabelRulePolicy("기타/무관", (), 30, no_direct_rules=True),
    ))

    with pytest.raises(ValueError, match="aliases"):
        validate_rule_configuration(rule_set)


def test_rule_fingerprint_is_stable_for_term_order_but_tracks_policy_changes():
    labels = list(simple_rule_set().labels)
    labels[0] = replace(
        labels[0],
        terms=(RuleTerm("연구"), RuleTerm("개발")),
    )
    original = replace(simple_rule_set(), labels=tuple(labels))
    labels[0] = replace(
        labels[0],
        terms=tuple(reversed(labels[0].terms)),
    )
    reordered = replace(simple_rule_set(), labels=tuple(labels))
    changed = replace(
        original,
        decision=replace(original.decision, min_rule_score=3),
    )

    assert rule_set_fingerprint(original) == rule_set_fingerprint(reordered)
    assert rule_set_fingerprint(original) != rule_set_fingerprint(changed)


def test_development_evidence_must_exist_as_confirmed_labeled_rows():
    labels = list(simple_rule_set().labels)
    labels[0] = replace(
        labels[0],
        terms=(
            RuleTerm(
                "새 규칙",
                origin="development",
                evidence_event_ids=("event-1", "event-2"),
            ),
        ),
    )
    rule_set = replace(simple_rule_set(), labels=tuple(labels))
    rows = [
        {
            "event_id": event_id,
            "split": "development",
            "review_status": "confirmed",
            "gold_label": "기술개발",
        }
        for event_id in ("event-1", "event-2")
    ]

    validate_rule_configuration(rule_set)
    validate_development_rule_evidence(rule_set, rows)
    validate_development_rule_errors(
        rule_set,
        rows,
        ["제품/서비스", "제품/서비스"],
    )

    with pytest.raises(ValueError, match="does not exist"):
        validate_development_rule_evidence(rule_set, rows[:1])

    wrong_label_rows = [dict(row) for row in rows]
    wrong_label_rows[0]["gold_label"] = "제품/서비스"
    with pytest.raises(ValueError, match="gold label"):
        validate_development_rule_evidence(rule_set, wrong_label_rows)

    with pytest.raises(ValueError, match="baseline error"):
        validate_development_rule_errors(
            rule_set,
            rows,
            ["기술개발", "제품/서비스"],
        )
