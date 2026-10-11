"""Synthetic rule checks only, not measured real-news accuracy."""
from dataclasses import replace

import pytest

from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.models import ModelPrediction, NewsItem
from news_classifier.evaluation_dataset import rows_with_article_context
from news_classifier.rules.default_rules import DEFAULT_RULE_SET
from news_classifier.rules.policy import RuleStrength, RuleTerm
from news_classifier.rules.review_validation import audit_rule_holdout, evaluate_review_reduction
from news_classifier.rules import review_validation


def comparison(**changes):
    values = dict(expected=["금융/투자", "제품/서비스"], baseline=["검토필요"] * 2,
                  selected=["금융/투자", "제품/서비스"], labels=list(DEFAULT_RULE_SET.candidate_labels),
                  keywords=["unused-query-a", "unused-query-b"], event_ids=["event-a", "event-b"],
                  has_development_rules=True, split="evaluation", unseen_keywords_validated=True)
    values.update(changes)
    return evaluate_review_reduction(**values)


def test_heldout_comparison_counts_new_decisions_without_relabelling_inputs():
    report = comparison()
    assert report["passed"]
    assert report["baseline"]["review_rate"] == 1
    assert report["selected"]["review_rate"] == 0
    assert report["baseline"]["automatic_accuracy"] is None
    assert report["newly_automatic_accuracy"] == 1
    assert report["newly_automatic_independent_events"] == 2
    assert report["per_keyword"]["unused-query-a"]["newly_automatic_count"] == 1


@pytest.mark.parametrize("changes", [
    {"split": "development"}, {"unseen_keywords_validated": False},
    {"selected": ["금융/투자", "검토필요"]},
    {"selected": ["금융/투자", "금융/투자"]},
    {"event_ids": ["same-event", " SAME-EVENT "]},
    {"baseline": ["금융/투자", "제품/서비스"], "selected": ["검토필요", "금융/투자"]},
])
def test_review_drop_alone_cannot_pass_validation(changes):
    report = comparison(**changes)
    assert not report["passed"]
    assert report["failures"]


def test_default_rules_are_not_reported_as_new_validated_rules():
    assert comparison(has_development_rules=False)["status"] == "no_new_rules"
    assert not comparison(has_development_rules=False)["passed"]
    with pytest.raises(ValueError, match="event ids"):
        comparison(event_ids=["", "event-b"])


def proposed_rules():
    labels = list(DEFAULT_RULE_SET.labels)
    index = next(i for i, policy in enumerate(labels) if policy.label == "금융/투자")
    labels[index] = replace(labels[index], terms=labels[index].terms + (RuleTerm(
        "synthetic equity event", strength=RuleStrength.STRONG, origin="development",
        evidence_event_ids=("development-a", "development-b"),
    ),))
    return replace(DEFAULT_RULE_SET, labels=tuple(labels))


def holdout_rows(labels=("금융/투자", "금융/투자")):
    return [{"id": str(i), "event_id": f"evaluation-{i}", "gold_label": label}
            for i, label in enumerate(labels)]


def holdout_results(rules, **changes):
    processor = ClassificationPostProcessor(RuleEngine(rules))
    return [processor.process(NewsItem(keyword=query, title="synthetic equity event", link=""),
                              ModelPrediction("사회", 0.2, 0.01))
            for query in ("query-a", "query-b")]


def test_each_added_expression_needs_its_own_independent_correct_rule_cases():
    rules = proposed_rules()
    assert audit_rule_holdout(rules, holdout_rows(), holdout_results(rules))["passed"]
    duplicate = [{**row, "event_id": "same"} for row in holdout_rows()]
    assert not audit_rule_holdout(rules, duplicate, holdout_results(rules))["passed"]
    report = audit_rule_holdout(rules, holdout_rows(("사회", "정치")), holdout_results(rules))
    assert report["terms"][0]["wrong_rule_count"] == 2
    assert not report["passed"]


def test_expression_in_raw_body_cannot_count_when_actual_input_has_no_body():
    rules = proposed_rules()
    rows = [{**row, "content": "synthetic equity event"} for row in holdout_rows()]
    processor = ClassificationPostProcessor(RuleEngine(rules))
    results = [processor.process(NewsItem(keyword="q", title="unrelated", link=""),
                                  ModelPrediction("사회", 0.2, 0.01)) for _ in rows]
    assert not audit_rule_holdout(rules, rows, results)["passed"]


def test_evidence_context_rows_preserve_source_and_only_use_actual_inputs():
    source = [{"id": "1", "title": "source", "content": "body not used", "gold_label": "사회"}]
    item = NewsItem(keyword="q", title="actual", description="desc", content="", link="")
    context = rows_with_article_context(source, [item])
    assert context[0]["title"] == "actual"
    assert context[0]["content"] == ""
    assert source[0]["title"] == "source"
    assert source[0]["content"] == "body not used"
    assert context[0]["gold_label"] == source[0]["gold_label"]
    with pytest.raises(ValueError, match="align"):
        rows_with_article_context(source, [])


def test_holdout_prepares_actual_article_context_once_for_all_terms(monkeypatch):
    rules = proposed_rules()
    rules = replace(rules, labels=tuple(
        replace(policy, terms=policy.terms + (RuleTerm(
            "synthetic bond event", strength=RuleStrength.STRONG, origin="development",
            evidence_event_ids=("development-a", "development-b"),
        ),)) if policy.label == "금융/투자" else policy for policy in rules.labels
    ))
    rows = [{**row, "title": "stored title", "content": "stored unused body"} for row in holdout_rows()]
    original = [dict(row) for row in rows]
    results = holdout_results(rules)
    calls = []

    def prepare(source, items):
        calls.append((source, items))
        return rows_with_article_context(source, items)

    monkeypatch.setattr(review_validation, "rows_with_article_context", prepare)
    report = audit_rule_holdout(rules, rows, results)
    assert len(calls) == 1
    assert calls[0][1] == [result.item for result in results]
    assert len(report["terms"]) == 2
    assert report["terms"][0]["correct_rule_independent_events"] == 2
    assert report["terms"][1]["correct_rule_independent_events"] == 0
    assert rows == original


def test_holdout_keeps_alignment_error_before_preparing_context(monkeypatch):
    monkeypatch.setattr(review_validation, "rows_with_article_context",
                        lambda *args: pytest.fail("unaligned context must not be prepared"))
    with pytest.raises(ValueError) as exc:
        audit_rule_holdout(proposed_rules(), holdout_rows(), [])
    assert str(exc.value) == "rule holdout rows and results must align"


def test_holdout_reuses_scores_across_terms_without_merging_event_metadata(monkeypatch):
    rules = proposed_rules()
    rules = replace(rules, labels=tuple(
        replace(policy, terms=policy.terms + (RuleTerm(
            "synthetic bond event", strength=RuleStrength.STRONG, origin="development",
            evidence_event_ids=("development-a", "development-b"),
        ),)) if policy.label == "금융/투자" else policy for policy in rules.labels
    ))
    results = holdout_results(rules)
    calls = []
    score = RuleEngine._score_labels

    def record(engine, text):
        calls.append(text)
        return score(engine, text)

    monkeypatch.setattr(RuleEngine, "_score_labels", record)
    report = audit_rule_holdout(rules, holdout_rows(), results)
    assert len(calls) == 1
    assert report["terms"][0]["correct_rule_independent_events"] == 2
    assert report["terms"][1]["correct_rule_independent_events"] == 0
