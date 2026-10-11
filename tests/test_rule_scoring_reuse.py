"""Scoped rule-score reuse checks; synthetic cases, not news accuracy evidence."""
from dataclasses import replace

import pytest

from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.classifiers.rule_engine import RuleEngine, RuleScoringSession
from news_classifier.models import ModelPrediction, NewsItem
from news_classifier.rules.default_rules import DEFAULT_RULE_SET, LEGACY_UNRELATED_SIGNAL_GROUPS


def count_scoring(monkeypatch, engine):
    calls = []
    score = engine._score_labels

    def record(text):
        calls.append(text)
        return score(text)

    monkeypatch.setattr(engine, "_score_labels", record)
    return calls


@pytest.mark.parametrize("guarded", [False, True])
@pytest.mark.parametrize("title,description,content", [
    ("신제품 출시와 서비스 제공", "고객용 서비스 업데이트", "신제품을 출시한다."),
    ("교육 소식", "", "교육을 진행한다. 학생을 양성한다. 과거 합병과 계약을 소개한다."),
    ("단신", "", ""),
    ("API 발표", "capital expenditure", ""),
])
def test_session_preserves_rule_only_decision_and_every_prediction_reason(guarded, title, description, content):
    engine = RuleEngine(DEFAULT_RULE_SET, experimental_event_anchors=guarded)
    scoring = RuleScoringSession(engine)
    fields = (title, description, content)
    assert scoring.rank_rules(*fields) == engine.rank_rules(*fields)
    assert engine.rule_only_decision(*fields, scoring=scoring) == engine.rule_only_decision(*fields)
    for prediction in (ModelPrediction("기술개발", .2, .01), ModelPrediction("교육/취업", .85, .3)):
        assert engine.decide(*fields, prediction, scoring=scoring) == engine.decide(*fields, prediction)


def test_session_reuses_one_ranking_but_keeps_exact_fields_and_body_changes_separate(monkeypatch):
    engine = RuleEngine(DEFAULT_RULE_SET)
    scoring = RuleScoringSession(engine)
    calls = count_scoring(monkeypatch, engine)
    ranking = scoring.rank_rules("IPO 발표")
    assert engine.rule_only_decision("IPO 발표", scoring=scoring) == "검토필요"
    engine.decide("IPO 발표", "", "", ModelPrediction("사회", .2, .01), scoring=scoring)
    assert scoring.rank_rules("IPO 발표") is ranking
    assert len(calls) == 1
    # Even normalized-equivalent strings and field redistribution stay distinct.
    scoring.rank_rules("IPO  발표")
    scoring.rank_rules("IPO", "발표")
    scoring.rank_rules("IPO 발표", "", "정부 정책과 규제 발표")
    assert len(calls) == 4


def test_session_is_engine_bound_and_has_no_global_article_cache(monkeypatch):
    first = RuleEngine(DEFAULT_RULE_SET)
    second = RuleEngine(DEFAULT_RULE_SET)
    calls = count_scoring(monkeypatch, first)
    scoring = RuleScoringSession(first)
    scoring.rank_rules("단신")
    RuleScoringSession(first).rank_rules("단신")
    assert len(calls) == 2
    with pytest.raises(ValueError, match="different engine"):
        second.decide("단신", "", "", ModelPrediction("사회", .2, .01), scoring=scoring)
    first.rule_set = replace(DEFAULT_RULE_SET, version="changed-scope")
    with pytest.raises(ValueError, match="rule set changed"):
        scoring.rank_rules("단신")


def test_unrelated_and_failed_paths_do_not_eagerly_calculate_rules(monkeypatch):
    engine = RuleEngine(replace(DEFAULT_RULE_SET, unrelated_signal_groups=LEGACY_UNRELATED_SIGNAL_GROUPS))
    scoring = RuleScoringSession(engine)
    calls = count_scoring(monkeypatch, engine)
    assert engine.rule_only_decision("야구 경기 결과", scoring=scoring) == "기타/무관"
    assert engine.decide("야구 경기 결과", "", "", ModelPrediction("스포츠", .8, .3),
                         scoring=scoring).final_label == "기타/무관"
    result = ClassificationPostProcessor(engine).process(
        NewsItem("q", "분류 실패", ""), ModelPrediction.failed(), scoring=scoring,
    )
    assert result.rule_decision.rule_reason == "모델 분류 실패"
    assert calls == []


def test_profile_fallback_and_baseline_reason_do_not_reuse_selected_scores(monkeypatch):
    selected = RuleEngine(replace(DEFAULT_RULE_SET, decision=replace(
        DEFAULT_RULE_SET.decision, review_model_score=.22, ambiguity_score=.22,
    )))
    baseline = RuleEngine(DEFAULT_RULE_SET)
    selected_calls = count_scoring(monkeypatch, selected)
    baseline_calls = count_scoring(monkeypatch, baseline)
    scoring = RuleScoringSession(selected)
    processor = ClassificationPostProcessor(selected, baseline_rule_engine=baseline,
                                            decision_input_mode="with_body")
    prediction = ModelPrediction("정치", .22, .07)
    title_only = NewsItem("q", "일반 기사", "")
    result = processor.process(title_only, prediction, scoring=scoring)
    assert result.rule_decision.final_label == "검토필요"
    assert len(selected_calls) == 0 and len(baseline_calls) == 1
    with_body = replace(title_only, content="일반 본문")
    scoring.rank_rules(with_body.title, "", with_body.content)
    result = processor.process(with_body, prediction, scoring=scoring)
    assert result.rule_decision.final_label == "정치"
    assert "기본 기준 검토 사유" in result.rule_decision.rule_reason
    assert len(selected_calls) == 1 and len(baseline_calls) == 2
