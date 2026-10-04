"""Synthetic event-rule conflicts; not evidence of real-news classification accuracy."""
from dataclasses import replace

import pytest

from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.models import ModelPrediction
from news_classifier.rules.default_rules import DEFAULT_RULE_SET
from news_classifier.rules.policy import RuleStrength, RuleTerm


def event_term(phrase, strength=RuleStrength.WEAK, *, development=True):
    return RuleTerm(
        phrase,
        strength,
        origin="development" if development else "legacy",
        evidence_event_ids=("synthetic-event-a", "synthetic-event-b") if development else (),
    )


def conflict_engine(*, development=True, allow_override=False, extra_terms=()):
    labels = list(DEFAULT_RULE_SET.labels)
    index = next(i for i, policy in enumerate(labels) if policy.label == "교육/취업")
    labels[index] = replace(
        labels[index],
        terms=(
            event_term("synthetic education event", RuleStrength.STRONG, development=development),
            event_term("synthetic admission", development=development),
        ) + extra_terms,
        no_direct_rules=False,
        allow_technology_bias_override=allow_override,
    )
    return RuleEngine(replace(DEFAULT_RULE_SET, labels=tuple(labels)))


CONFLICT_TITLE = "synthetic education event synthetic admission 공모가"


@pytest.mark.parametrize("score, margin", [(0.3, 0.01), (0.51, 0.1), (0.7, 0.02)])
def test_uncertain_model_does_not_turn_competing_strong_event_evidence_into_rule(score, margin):
    engine = conflict_engine()
    ranking = engine.rank_rules(CONFLICT_TITLE)
    assert ranking.first.weighted_score == 3
    assert ranking.second.weighted_score == 2
    assert ranking.score_margin == 1
    result = engine.decide(CONFLICT_TITLE, "", "", ModelPrediction("사회", score, margin))
    assert result.final_label == "검토필요"
    assert not result.rule_applied
    assert result.rule_reason.startswith("주제 근거 충돌")
    assert "교육/취업" in result.rule_reason
    assert "금융/투자" in result.rule_reason
    assert engine.rule_only_decision(CONFLICT_TITLE) == "검토필요"


def test_confident_model_path_remains_unchanged_despite_new_rule_conflict():
    result = conflict_engine().decide(CONFLICT_TITLE, "", "", ModelPrediction("사회", 0.8, 0.2))
    assert result.final_label == "사회"
    assert not result.rule_applied
    assert result.rule_reason == ""


def test_new_rule_conflict_disables_confident_technology_bias_override():
    engine = conflict_engine(
        allow_override=True,
        extra_terms=(event_term("synthetic school notice", RuleStrength.STRONG),),
    )
    title = CONFLICT_TITLE + " synthetic school notice"
    ranking = engine.rank_rules(title)
    assert ranking.first.weighted_score == 5
    assert ranking.score_margin == 3
    result = engine.decide(title, "", "", ModelPrediction("기술개발", 0.8, 0.2))
    assert result.final_label == "기술개발"
    assert not result.rule_applied


def test_legacy_only_competing_strong_rules_keep_existing_score_margin_behavior():
    engine = conflict_engine(development=False)
    result = engine.decide(CONFLICT_TITLE, "", "", ModelPrediction("사회", 0.3, 0.01))
    assert result.final_label == "교육/취업"
    assert result.rule_applied
    assert engine.rule_only_decision(CONFLICT_TITLE) == "교육/취업"


def test_legacy_only_technology_override_keeps_existing_behavior():
    engine = conflict_engine(
        development=False,
        allow_override=True,
        extra_terms=(event_term("synthetic school notice", RuleStrength.STRONG, development=False),),
    )
    result = engine.decide(CONFLICT_TITLE + " synthetic school notice", "", "", ModelPrediction("기술개발", 0.8, 0.2))
    assert result.final_label == "교육/취업"
    assert result.rule_applied
    assert result.rule_reason.startswith("기술개발 편향 보정")


def test_a_weak_rival_is_not_a_strong_event_conflict():
    engine = conflict_engine()
    title = "synthetic education event synthetic admission IPO"
    result = engine.decide(title, "", "", ModelPrediction("사회", 0.3, 0.01))
    assert result.final_label == "교육/취업"
    assert result.rule_applied
    assert engine.rule_only_decision(title) == "교육/취업"


def test_unmatched_development_term_does_not_change_legacy_only_article():
    engine = conflict_engine(
        development=False,
        extra_terms=(event_term("unmatched new event", RuleStrength.STRONG),),
    )
    result = engine.decide(CONFLICT_TITLE, "", "", ModelPrediction("사회", 0.3, 0.01))
    assert result.final_label == "교육/취업"
    assert result.rule_applied


def test_a_scoring_weak_development_term_also_triggers_conflict_safety():
    labels = list(DEFAULT_RULE_SET.labels)
    index = next(i for i, policy in enumerate(labels) if policy.label == "교육/취업")
    labels[index] = replace(labels[index], terms=(
        event_term("synthetic education event", RuleStrength.STRONG, development=False),
        event_term("synthetic admission"),
    ), no_direct_rules=False)
    engine = RuleEngine(replace(DEFAULT_RULE_SET, labels=tuple(labels)))
    result = engine.decide(CONFLICT_TITLE, "", "", ModelPrediction("사회", 0.3, 0.01))
    assert result.final_label == "검토필요"
    assert not result.rule_applied


def test_new_term_shadowed_by_longer_phrase_does_not_count_as_development_contribution():
    labels = list(DEFAULT_RULE_SET.labels)
    education = next(i for i, policy in enumerate(labels) if policy.label == "교육/취업")
    finance = next(i for i, policy in enumerate(labels) if policy.label == "금융/투자")
    labels[education] = replace(labels[education], terms=(
        event_term("synthetic education", RuleStrength.STRONG),
        event_term("synthetic school report", RuleStrength.STRONG, development=False),
        event_term("synthetic admission", development=False),
    ), no_direct_rules=False)
    labels[finance] = replace(labels[finance], terms=labels[finance].terms + (
        event_term("synthetic education event", RuleStrength.STRONG, development=False),
    ))
    engine = RuleEngine(replace(DEFAULT_RULE_SET, labels=tuple(labels)))
    title = "synthetic education event synthetic school report synthetic admission"
    ranking = engine.rank_rules(title)
    assert ranking.first.weighted_score == 3
    assert ranking.second.weighted_score == 2
    assert "synthetic education" not in ranking.first.matched_terms
    result = engine.decide(title, "", "", ModelPrediction("사회", 0.3, 0.01))
    assert result.final_label == "교육/취업"
    assert result.rule_applied


def test_new_rule_conflict_guard_is_shared_by_description_and_body():
    engine = conflict_engine()
    for description, content in ((CONFLICT_TITLE, ""), ("", CONFLICT_TITLE)):
        result = engine.decide("unrelated headline", description, content, ModelPrediction("사회", 0.3, 0.01))
        assert result.final_label == "검토필요"
        assert engine.rule_only_decision("unrelated headline", description, content) == "검토필요"
