"""실제 평가 CLI 흐름의 합성 통합 테스트. 실제 기사나 정확도 근거 아님."""
from dataclasses import replace
import csv
import json
from types import SimpleNamespace

import pytest

from scripts import evaluate_news
from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.models import ModelPrediction
from news_classifier.rules.default_rules import CANDIDATE_LABELS, DEFAULT_RULE_SET
from news_classifier.rules.policy import RuleStrength, RuleTerm
from news_classifier.rules.event_profile import load_event_rule_profile, runtime_bindings
from news_classifier.models import CLASSIFICATION_INPUT_POLICY
from news_classifier.classifiers.zero_shot_classifier import HYPOTHESIS_TEMPLATE


ALPHA = "합성알파근거"
BETA = "합성베타근거"
FINANCE = "금융/투자"
DEV_EVENTS = ("development-finance-alpha-1", "development-finance-alpha-2")


def synthetic_rows(*, second_term=False):
    rows = []
    for split in ("development", "evaluation"):
        for index, label in enumerate(CANDIDATE_LABELS):
            rows.append({
                "id": f"{split}-coverage-{index}",
                "event_id": f"{split}-coverage-event-{index}",
                "split": split, "keyword": f"{split}-query",
                "title": f"합성자료 {split} cover {index}", "description": "",
                "gold_label": label, "review_status": "confirmed",
                "reviewed_by": "synthetic-test-fixture",
            })
        for index in (1, 2):
            # Only development supports BETA: its own held-out cases are absent.
            extra_phrase = f" {BETA}" if second_term and split == "development" else ""
            rows.append({
                "id": f"{split}-finance-alpha-{index}",
                "event_id": f"{split}-finance-alpha-{index}",
                "split": split, "keyword": f"{split}-query",
                "title": f"합성자료 {split} {ALPHA}{extra_phrase} 사례 {index}",
                "description": "", "gold_label": FINANCE,
                "review_status": "confirmed", "reviewed_by": "synthetic-test-fixture",
            })
    return rows


def write_fixture(path, rows, *, include_event_column=True):
    fields = ["id", "event_id", "split", "keyword", "title", "description", "content",
              "gold_label", "review_status", "reviewed_by"]
    if not include_event_column:
        fields.remove("event_id")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fixture_rule_set(*, second_term=False):
    terms = [RuleTerm(ALPHA, RuleStrength.STRONG, "development", DEV_EVENTS)]
    if second_term:
        terms.append(RuleTerm(BETA, RuleStrength.STRONG, "development", DEV_EVENTS))
    return replace(DEFAULT_RULE_SET, labels=tuple(
        replace(policy, terms=policy.terms + tuple(terms)) if policy.label == FINANCE else policy
        for policy in DEFAULT_RULE_SET.labels
    ))


def install_fake_pipeline(monkeypatch, rows, *, second_term=False, input_mode=None):
    gold_by_title = {row["title"]: row["gold_label"] for row in rows}
    inference_calls, build_calls = [], []

    def classify_many(texts):
        inference_calls.append(texts)
        # Correct model top1 is intentionally uncertain: the previous hybrid
        # must REVIEW it, making confirmed final-path errors valid evidence.
        return [ModelPrediction(
            gold_by_title[text.splitlines()[0].removeprefix("기사제목: ")], .2, .01,
        ) for text in texts]

    pipeline = SimpleNamespace(
        classifier=SimpleNamespace(
            classify_many=classify_many, candidate_hypotheses=list(CANDIDATE_LABELS),
        ),
        postprocessor=ClassificationPostProcessor(
            RuleEngine(fixture_rule_set(second_term=second_term)),
            baseline_rule_engine=RuleEngine(DEFAULT_RULE_SET) if input_mode else None,
            decision_input_mode=input_mode,
        ),
    )

    def build(settings):
        build_calls.append(settings)
        return pipeline

    monkeypatch.setattr(evaluate_news, "build_pipeline", build)
    return build_calls, inference_calls


def run_evaluation(monkeypatch, dataset, output):
    # No --require-unseen-keywords: event rules must enforce this by default.
    monkeypatch.setattr("sys.argv", [
        "evaluate_news", "--dataset", str(dataset), "--output-dir", str(output),
    ])
    evaluate_news.main()


def test_event_rule_evaluation_accepts_model_correct_but_previous_hybrid_review(tmp_path, monkeypatch):
    rows = synthetic_rows()
    dataset, output = tmp_path / "fixture.csv", tmp_path / "report"
    write_fixture(dataset, rows)
    original = dataset.read_bytes()
    _, inference_calls = install_fake_pipeline(monkeypatch, rows)

    run_evaluation(monkeypatch, dataset, output)

    payload = json.loads((output / "evaluation_results.json").read_text(encoding="utf-8"))
    validation = payload["rule_review_validation"]
    assert len(inference_calls) == 2  # Held-out and development, one batch each.
    assert all(len(call) == len(CANDIDATE_LABELS) + 2 for call in inference_calls)
    assert validation["status"] == "passed"
    assert validation["passed"] is True
    assert validation["newly_automatic_count"] == 2
    assert validation["newly_automatic_independent_events"] == 2
    assert validation["newly_automatic_wrong_count"] == 0
    assert validation["previous_correct_harmed_count"] == 0
    assert validation["per_rule_holdout"]["passed"] is True
    assert validation["per_rule_holdout"]["terms"][0]["correct_rule_independent_events"] == 2
    assert payload["unseen_keywords_validated"] is True
    assert all(case["model_label"] == case["gold_label"] for case in payload["cases"])
    assert all(case["baseline_hybrid_label"] == "검토필요" for case in payload["cases"])
    corrected = [case for case in payload["cases"] if "finance-alpha" in case["id"]]
    assert all(case["hybrid_label"] == FINANCE and case["hybrid_rule_applied"] for case in corrected)
    assert dataset.read_bytes() == original


def test_event_rules_reject_shared_development_and_evaluation_keyword_by_default(tmp_path, monkeypatch):
    rows = synthetic_rows()
    for row in rows:
        row["keyword"] = "같은 검색어"
    dataset, output = tmp_path / "fixture.csv", tmp_path / "report"
    write_fixture(dataset, rows)
    _, inference_calls = install_fake_pipeline(monkeypatch, rows)

    with pytest.raises(ValueError, match="keywords appear in both"):
        run_evaluation(monkeypatch, dataset, output)

    assert inference_calls == []
    assert not output.exists()


@pytest.mark.parametrize("missing", ["column", "development_value", "evaluation_value"])
def test_event_rule_evaluation_requires_explicit_event_ids(tmp_path, monkeypatch, missing):
    rows = synthetic_rows()
    if missing != "column":
        split = missing.removesuffix("_value")
        next(row for row in rows if row["split"] == split)["event_id"] = ""
    dataset, output = tmp_path / "fixture.csv", tmp_path / "report"
    write_fixture(dataset, rows, include_event_column=missing != "column")
    _, inference_calls = install_fake_pipeline(monkeypatch, rows)

    with pytest.raises(ValueError, match="explicit"):
        run_evaluation(monkeypatch, dataset, output)

    assert inference_calls == []
    assert not output.exists()


def test_each_added_expression_needs_its_own_heldout_support(tmp_path, monkeypatch):
    rows = synthetic_rows(second_term=True)
    dataset, output = tmp_path / "fixture.csv", tmp_path / "report"
    write_fixture(dataset, rows)
    _, inference_calls = install_fake_pipeline(monkeypatch, rows, second_term=True)

    run_evaluation(monkeypatch, dataset, output)

    payload = json.loads((output / "evaluation_results.json").read_text(encoding="utf-8"))
    validation = payload["rule_review_validation"]
    assert len(inference_calls) == 2
    assert validation["newly_automatic_count"] == 2
    assert validation["newly_automatic_wrong_count"] == 0
    assert validation["status"] == "not_validated"
    assert validation["passed"] is False
    holdout = validation["per_rule_holdout"]
    assert holdout["passed"] is False
    terms = {term["phrase"]: term for term in holdout["terms"]}
    assert terms[ALPHA]["passed"] is True
    assert terms[BETA]["correct_rule_independent_events"] == 0
    assert terms[BETA]["passed"] is False
    assert any("each new expression" in reason for reason in validation["failures"])


@pytest.mark.parametrize("split", ["development", "evaluation"])
def test_without_body_profile_rejects_development_evidence_present_only_in_raw_body(tmp_path, monkeypatch, split):
    rows = synthetic_rows()
    for row in rows:
        if row["event_id"] in DEV_EVENTS:
            row["title"] = row["title"].replace(ALPHA, "표현없음")
            row["content"] = f"본문에만 존재하는 {ALPHA}"
    dataset, output = tmp_path / "fixture.csv", tmp_path / "report"
    write_fixture(dataset, rows)
    original = dataset.read_bytes()
    _, inference_calls = install_fake_pipeline(monkeypatch, rows, input_mode="without_body")
    monkeypatch.setattr("sys.argv", [
        "evaluate_news", "--dataset", str(dataset), "--split", split,
        "--output-dir", str(output),
    ])

    with pytest.raises(ValueError, match="phrase does not score in article context"):
        evaluate_news.main()

    assert inference_calls == []
    assert not output.exists()
    assert dataset.read_bytes() == original


@pytest.mark.parametrize("second_term", [False, True])
def test_proposal_evaluation_exports_only_independently_passed_profiles(tmp_path, monkeypatch, second_term):
    rows = synthetic_rows(second_term=second_term)
    dataset, output = tmp_path / "fixture.csv", tmp_path / "report"
    write_fixture(dataset, rows)
    original = dataset.read_bytes()
    manifest, approved = tmp_path / "proposal.json", tmp_path / "approved.json"
    terms = [ALPHA, BETA] if second_term else [ALPHA]
    manifest.write_text(json.dumps({"schema_version": 1, "profile_type": "event_rule_candidates",
        "rules": [{"category": FINANCE, "phrase": term, "strength": "STRONG",
                   "evidence_event_ids": list(DEV_EVENTS)} for term in terms]}), encoding="utf-8")
    build_calls, _ = install_fake_pipeline(monkeypatch, rows)
    original_builder = evaluate_news.build_pipeline
    pipelines = []

    def build(settings):
        pipeline = original_builder(settings)
        pipeline.postprocessor = ClassificationPostProcessor(RuleEngine(DEFAULT_RULE_SET))
        pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(evaluate_news, "build_pipeline", build)
    monkeypatch.setattr("sys.argv", ["evaluate_news", "--dataset", str(dataset),
        "--output-dir", str(output), "--rules-file", str(manifest),
        "--export-rule-profile", str(approved)])
    if second_term:
        with pytest.raises(SystemExit) as error:
            evaluate_news.main()
        assert error.value.code == 2
        assert not approved.exists()
    else:
        evaluate_news.main()
        bindings = runtime_bindings(build_calls[0], pipelines[0].classifier,
            CLASSIFICATION_INPUT_POLICY, HYPOTHESIS_TEMPLATE, DEFAULT_RULE_SET)
        rules = load_event_rule_profile(approved, DEFAULT_RULE_SET, bindings)
        assert RuleEngine(rules).rule_only_decision(ALPHA, "", "") == FINANCE
        assert rules.decision == DEFAULT_RULE_SET.decision
        assert json.loads(approved.read_text(encoding="utf-8"))["validation"]["passed"] is True
    assert dataset.read_bytes() == original


def test_rule_profile_export_never_overwrites_existing_file(tmp_path, monkeypatch):
    approved = tmp_path / "approved.json"
    approved.write_text("keep", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["evaluate_news", "--dataset", "absent.csv",
        "--rules-file", "absent.json", "--export-rule-profile", str(approved)])
    with pytest.raises(SystemExit) as error:
        evaluate_news.main()
    assert error.value.code == 2
    assert approved.read_text(encoding="utf-8") == "keep"


def test_short_label_comparison_keeps_same_topics_inputs_and_default_rules(tmp_path, monkeypatch):
    rows = synthetic_rows()
    dataset, output = tmp_path / "fixture.csv", tmp_path / "report"
    write_fixture(dataset, rows)
    install_fake_pipeline(monkeypatch, rows)
    original_builder = evaluate_news.build_pipeline
    comparison_calls = []

    def build(settings):
        pipeline = original_builder(settings)

        def short(texts, *, candidate_labels):
            comparison_calls.append((texts, candidate_labels))
            return [ModelPrediction("기술개발", .2, .01) for _ in texts]

        pipeline.classifier.classify_many_legacy = short
        return pipeline

    monkeypatch.setattr(evaluate_news, "build_pipeline", build)
    monkeypatch.setattr("sys.argv", ["evaluate_news", "--dataset", str(dataset),
        "--output-dir", str(output), "--compare-short-labels"])
    evaluate_news.main()
    payload = json.loads((output / "evaluation_results.json").read_text(encoding="utf-8"))
    assert payload["compare_short_labels"] is True
    assert comparison_calls[0][1] == CANDIDATE_LABELS
    assert all("기사제목:" in text for text in comparison_calls[0][0])
    assert "짧은 주제명 모델" in payload["reports"]
    assert "짧은 주제명 하이브리드" in payload["reports"]
    assert payload["rule_policy"]["decision"] == evaluate_news.asdict(DEFAULT_RULE_SET.decision)
