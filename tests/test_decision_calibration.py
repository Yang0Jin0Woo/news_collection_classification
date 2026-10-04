import csv
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.decision_calibration import (
    calibrate_decision_policy, create_decision_profile, evaluate_decision_policy,
    load_decision_policy, load_decision_profile,
)
from news_classifier.evaluation_dataset import rows_to_items
from news_classifier.models import ModelPrediction
from news_classifier.rules.default_rules import DEFAULT_RULE_SET
from scripts import calibrate_decision_policy as script


def samples(split="development", review_score=0.45, review_margin=0.02):
    labels = list(DEFAULT_RULE_SET.candidate_labels) * 4
    rows = [{
        "id": f"{split}-{index}", "event_id": f"{split}-event-{index}",
        "split": split, "keyword": f"{split}-{index % 2}",
        "title": f"case {index}", "description": "", "gold_label": label,
        "review_status": "confirmed", "reviewed_by": "synthetic-test",
    } for index, label in enumerate(labels)]
    predictions = [ModelPrediction(label, review_score, review_margin) if index % 2 == 0
                   else ModelPrediction(label, 0.8, 0.2)
                   for index, label in enumerate(labels)]
    return rows, rows_to_items(rows), predictions, labels


def calibrate(rows, items, predictions, labels, **kwargs):
    return calibrate_decision_policy(items, predictions, labels, DEFAULT_RULE_SET,
                                    event_ids=[row["event_id"] for row in rows], **kwargs)


def profile(tmp_path, input_mode="without_body", review_score=0.45, review_margin=0.02):
    development_rows, items, predictions, labels = samples(
        review_score=review_score, review_margin=review_margin,
    )
    if input_mode == "with_body":
        development_rows = [{**row, "content": "article context"} for row in development_rows]
        items = rows_to_items(development_rows)
    result = calibrate(development_rows, items, predictions, labels, input_mode=input_mode)
    evaluation_rows, items, predictions, labels = samples("evaluation", review_score, review_margin)
    if input_mode == "with_body":
        evaluation_rows = [{**row, "content": "article context"} for row in evaluation_rows]
        items = rows_to_items(evaluation_rows)
    evaluation = evaluate_decision_policy(
        items, predictions, labels, DEFAULT_RULE_SET, result.policy,
        event_ids=[row["event_id"] for row in evaluation_rows],
        input_mode=input_mode,
    )
    dataset = tmp_path / "synthetic.csv"
    with dataset.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(development_rows[0]))
        writer.writeheader()
        writer.writerows(development_rows + evaluation_rows)
    payload = create_decision_profile(
        result, evaluation, dataset_path=dataset, model_name="model", model_revision="revision",
        candidate_hypotheses=list(DEFAULT_RULE_SET.candidate_labels),
        hypothesis_template="template {}", input_policy="article", max_sequence_length=1200,
        base_rule_set=DEFAULT_RULE_SET, development_rows=development_rows,
        evaluation_rows=evaluation_rows,
        input_mode=input_mode,
    )
    return dataset, payload


def load(path, loader=load_decision_policy, **changes):
    kwargs = dict(
        base_rule_set=DEFAULT_RULE_SET, expected_model_name="model", expected_model_revision="revision",
        expected_candidate_labels=list(DEFAULT_RULE_SET.candidate_labels),
        expected_candidate_hypotheses=list(DEFAULT_RULE_SET.candidate_labels),
        expected_hypothesis_template="template {}", expected_input_policy="article",
        expected_max_sequence_length=1200,
    )
    kwargs.update(changes)
    return loader(path, **kwargs)


def test_joint_calibration_reduces_reviews_without_changing_other_branches():
    rows, items, predictions, labels = samples()
    result = calibrate(rows, items, predictions, labels)
    assert result.evidence.passed
    assert result.evidence.baseline.support == 34
    assert result.evidence.selected.support == 68
    assert result.evidence.newly_decided.independent_support == 34
    assert result.policy.ambiguity_score == 0.45
    assert result.policy.ambiguity_margin == 0.02
    assert replace(result.policy, ambiguity_score=0.50, ambiguity_margin=0.05) == DEFAULT_RULE_SET.decision
    before = RuleEngine(DEFAULT_RULE_SET)
    after = RuleEngine(replace(DEFAULT_RULE_SET, decision=result.policy))
    for title, prediction in [
        ("주가 투자", ModelPrediction("정치", 0.8, 0.2)),
        ("주가 투자", ModelPrediction("정치", 0.45, 0.02)),
        ("case", ModelPrediction("정치", 0.39, 0.02)),
    ]:
        assert before.decide(title, "", "", prediction) == after.decide(title, "", "", prediction)


def test_calibration_refuses_wrong_new_decisions_or_inflated_duplicate_support():
    rows, items, predictions, labels = samples()
    wrong = [replace(p, label=labels[(index + 1) % len(labels)]) if index % 2 == 0 else p
             for index, p in enumerate(predictions)]
    with pytest.raises(ValueError, match="keep the default"):
        calibrate(rows, items, wrong, labels)
    repeated_events = [row["gold_label"] for row in rows]
    with pytest.raises(ValueError, match="keep the default"):
        calibrate_decision_policy(items, predictions, labels, DEFAULT_RULE_SET,
                                  event_ids=repeated_events)


def test_lower_scores_are_evaluated_without_the_hidden_old_review_floor():
    rows, items, predictions, labels = samples(review_score=0.22, review_margin=0.07)
    result = calibrate(rows, items, predictions, labels)
    assert result.policy.ambiguity_score == result.policy.review_model_score == 0.22
    assert result.policy.ambiguity_margin == 0.05
    assert result.evidence.baseline.support == 34
    assert result.evidence.selected.support == 68
    assert result.evidence.newly_decided.correct == 34
    assert replace(result.policy, review_model_score=0.40,
                   ambiguity_score=0.50, ambiguity_margin=0.05) == DEFAULT_RULE_SET.decision
    before = RuleEngine(DEFAULT_RULE_SET)
    after = RuleEngine(replace(DEFAULT_RULE_SET, decision=result.policy))
    for title, prediction in [
        ("주가 투자", ModelPrediction("정치", 0.8, 0.2)),
        ("주가 투자", ModelPrediction("정치", 0.22, 0.07)),
        ("주가 투자", ModelPrediction("정치", 0.0, 0.0)),
        ("case", ModelPrediction("정치", 0.21, 0.07)),
        ("case", ModelPrediction("정치", 0.22, 0.049)),
    ]:
        assert before.decide(title, "", "", prediction) == after.decide(title, "", "", prediction)


@pytest.mark.parametrize("score,margin", [(0.0, 0.0), (0.22, 0.0)])
def test_zero_scores_and_exact_ties_never_become_automatic_through_calibration(score, margin):
    with pytest.raises(ValueError, match="keep the default"):
        calibrate(*samples(review_score=score, review_margin=margin))


def test_relaxing_low_scores_still_requires_correct_new_decisions():
    rows, items, predictions, labels = samples(review_score=0.22, review_margin=0.07)
    wrong = [replace(p, label=labels[(index + 1) % len(labels)]) if index % 2 == 0 else p
             for index, p in enumerate(predictions)]
    with pytest.raises(ValueError, match="keep the default"):
        calibrate(rows, items, wrong, labels)


@pytest.mark.parametrize("changes", [
    {"review_model_score": 0.20},
    {"review_model_score": 0.20, "ambiguity_score": 0.22},
    {"review_model_score": 0.0, "ambiguity_score": 0.0},
    {"ambiguity_margin": 0.0},
])
def test_calibration_rejects_independent_floor_changes_and_zero_gates(changes):
    rows, items, predictions, labels = samples()
    with pytest.raises(ValueError):
        evaluate_decision_policy(
            items, predictions, labels, DEFAULT_RULE_SET,
            replace(DEFAULT_RULE_SET.decision, **changes),
            event_ids=[row["event_id"] for row in rows],
        )


def test_low_score_profile_roundtrip_records_real_model_values_and_new_decision_reason(tmp_path):
    _, payload = profile(tmp_path, review_score=0.22, review_margin=0.07)
    assert payload["schema_version"] == 2
    assert payload["thresholds"]["review_model_score"] == 0.22
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load(path, loader=load_decision_profile)
    processor = ClassificationPostProcessor(
        RuleEngine(replace(DEFAULT_RULE_SET, decision=loaded.policy)),
        baseline_rule_engine=RuleEngine(DEFAULT_RULE_SET), decision_input_mode=loaded.input_mode,
    )
    _, items, _, _ = samples()
    prediction = ModelPrediction("정치", 0.22, 0.07, ["정치", "사회"], [0.22, 0.15])
    for keyword in ("AI 투자 계획", "스포츠", "새로운 검색어"):
        row = processor.process(replace(items[0], keyword=keyword), prediction)
        assert row.rule_decision.final_label == "정치"
        assert row.model_prediction is prediction
        assert row.model_confidence == 0.22
        assert row.model_confidence_level == "낮음"
        assert row.decision_source == "MODEL"
        assert not row.rule_decision.rule_applied
        assert "검증된 공통 판정 기준" in row.rule_decision.rule_reason
        assert "0.2200" in row.rule_decision.rule_reason
        assert "기본 기준 검토 사유" in row.rule_decision.rule_reason
    # A body-stage mismatch cannot borrow the title-only relaxed policy.
    assert processor.process(replace(items[0], content="body"), prediction).review_required
    assert processor.process(items[0], replace(prediction, margin=0.0)).review_required
    assert processor.process(items[0], replace(prediction, score=0.0, margin=0.0)).review_required
    baseline = ClassificationPostProcessor(RuleEngine(DEFAULT_RULE_SET))
    assert baseline.process(items[0], prediction).review_required


def test_v1_profiles_remain_compatible_but_cannot_bypass_their_original_floor(tmp_path):
    _, payload = profile(tmp_path)
    payload["schema_version"] = 1
    payload["thresholds"].pop("review_model_score")
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load(path).review_model_score == 0.40
    assert load(path).ambiguity_score == 0.45
    payload["thresholds"]["ambiguity_score"] = 0.22
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load(path)


def test_independent_evaluation_rejects_valid_but_inaccurate_predictions():
    result = calibrate(*samples())
    rows, items, predictions, labels = samples("evaluation")
    predictions = [replace(p, label=labels[(index + 1) % len(labels)]) if index % 2 == 0 else p
                   for index, p in enumerate(predictions)]
    evaluation = evaluate_decision_policy(
        items, predictions, labels, DEFAULT_RULE_SET, result.policy,
        event_ids=[row["event_id"] for row in rows],
    )
    assert not evaluation.passed
    assert evaluation.newly_decided.correct == 0
    assert evaluation.previous_decisions_changed == evaluation.harmful_changes == 0
    assert any("accuracy lower bound" in failure for failure in evaluation.failures)


def test_pooled_accuracy_cannot_hide_a_wrong_topic_or_keyword_group():
    labels = list(DEFAULT_RULE_SET.candidate_labels) * 20
    _, original_items, _, _ = samples()
    wrong_label = "스포츠"
    items = [replace(original_items[index % len(original_items)],
                     keyword="wrong-group" if gold == wrong_label else "good-group")
             for index, gold in enumerate(labels)]
    predictions = [ModelPrediction("정치" if label == wrong_label else label, 0.45, 0.02)
                   for label in labels]
    event_ids = [str(index) for index in range(len(items))]
    evaluation = evaluate_decision_policy(
        items, predictions, labels, DEFAULT_RULE_SET,
        replace(DEFAULT_RULE_SET.decision, ambiguity_score=0.45, ambiguity_margin=0.02),
        event_ids=event_ids,
    )
    assert evaluation.newly_decided.accuracy_lower_bound > 0.85
    assert not evaluation.passed
    assert any("for label" in reason for reason in evaluation.failures)
    assert any("for keyword" in reason for reason in evaluation.failures)
    with pytest.raises(ValueError, match="keep the default"):
        calibrate_decision_policy(items, predictions, labels, DEFAULT_RULE_SET, event_ids=event_ids)


@pytest.mark.parametrize("mode", ["without_body", "with_body"])
def test_profile_roundtrip_preserves_body_stage_and_rejects_mismatched_cases(tmp_path, mode):
    _, payload = profile(tmp_path, mode)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load(path, loader=load_decision_profile).input_mode == mode
    assert load(None, loader=load_decision_profile).input_mode is None
    rows, items, predictions, labels = samples()
    if mode == "without_body":
        items = [replace(item, content="body") for item in items]
    with pytest.raises(ValueError, match="input mode"):
        calibrate(rows, items, predictions, labels, input_mode=mode)


@pytest.mark.parametrize("binding,value", [
    ("expected_model_revision", "changed"), ("expected_candidate_hypotheses", ["changed"]),
    ("expected_max_sequence_length", 500), ("expected_input_policy", "old-input"),
    ("base_rule_set", replace(DEFAULT_RULE_SET, version="new-rule-version")),
])
def test_profile_fails_closed_when_runtime_metadata_changes(tmp_path, binding, value):
    _, payload = profile(tmp_path)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load(path).ambiguity_margin == 0.02
    with pytest.raises(ValueError, match="does not match runtime"):
        load(path, **{binding: value})


@pytest.mark.parametrize("mutation", [
    lambda p: p["validation"].update(evaluation_required=True),
    lambda p: p.pop("evaluation"),
    lambda p: p["evaluation"]["evidence"]["newly_decided"].update(accuracy_lower_bound=1.0),
    lambda p: p["evaluation"]["evidence"].update(harmful_changes=1),
    lambda p: p["thresholds"].update(model_keep_score=0.1),
    lambda p: p["search"].update(selection_split="evaluation"),
    lambda p: p["input_policy"].pop("input_mode"),
])
def test_profile_rejects_missing_unvalidated_or_inconsistent_evidence(tmp_path, mutation):
    _, payload = profile(tmp_path)
    mutation(payload)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load(path)
    assert load(None) == DEFAULT_RULE_SET.decision


@pytest.mark.parametrize("dataset_mode,input_mode", [
    ("without_body", "without_body"), ("with_body", "with_body"), ("with_body", "without_body"),
])
def test_script_uses_confirmed_splits_and_saves_only_after_independent_success(tmp_path, monkeypatch, capsys, dataset_mode, input_mode):
    dataset, _ = profile(tmp_path, dataset_mode)
    original_dataset = dataset.read_bytes()
    calls = []
    def classify_many(texts):
        calls.append(texts)
        return samples("development" if len(calls) == 1 else "evaluation")[2]
    pipeline = SimpleNamespace(
        classifier=SimpleNamespace(classify_many=classify_many,
                                   candidate_hypotheses=list(DEFAULT_RULE_SET.candidate_labels)),
        postprocessor=ClassificationPostProcessor(RuleEngine(DEFAULT_RULE_SET)),
    )
    monkeypatch.setattr(script, "build_pipeline", lambda settings: pipeline)
    output = tmp_path / "result.json"
    monkeypatch.setattr("sys.argv", ["calibrate", "--dataset", str(dataset), "--output", str(output), "--input-mode", input_mode])
    script.main()
    assert len(calls) == 2
    printed = capsys.readouterr().out
    assert "공통 최종 판정 기준" in printed
    assert "수정 전: 검토필요 50.0%" in printed
    assert "수정 후: 검토필요 0.0%" in printed
    assert "새 자동 분류: 34건, 오분류 0건" in printed
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["validation"]["evaluation_required"] is False
    assert payload["search"]["selection_split"] == "development"
    assert payload["input_policy"]["input_mode"] == input_mode
    assert dataset.read_bytes() == original_dataset
    assert all("검색주제" not in text for batch in calls for text in batch)
    assert all(("기사본문" in text) == (input_mode == "with_body") for batch in calls for text in batch)
    output.unlink()
    calls.clear()
    def bad_evaluation(texts):
        good = classify_many(texts)
        return good if len(calls) == 1 else [ModelPrediction.failed() for _ in good]
    pipeline.classifier.classify_many = bad_evaluation
    with pytest.raises(SystemExit):
        script.main()
    assert not output.exists()


@pytest.mark.parametrize("invalid", ["pending", "blank_event", "missing_event_column", "overlap_keywords", "leaked_event", "missing_body"])
def test_script_rejects_unconfirmed_or_unidentified_events_before_model_loading(tmp_path, monkeypatch, invalid):
    dataset, _ = profile(tmp_path)
    with dataset.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if invalid == "pending":
        rows[0]["review_status"] = "pending"
    elif invalid == "blank_event":
        rows[0]["event_id"] = ""
    elif invalid == "missing_event_column":
        for row in rows:
            row.pop("event_id")
    elif invalid == "overlap_keywords":
        rows[-1]["keyword"] = rows[0]["keyword"]
    elif invalid == "leaked_event":
        rows[-1]["event_id"] = rows[0]["event_id"]
    with dataset.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(script, "build_pipeline", lambda _: pytest.fail("model must not load"))
    monkeypatch.setattr("sys.argv", ["calibrate", "--dataset", str(dataset), "--output", str(tmp_path / "no.json")]
                        + (["--input-mode", "with_body"] if invalid == "missing_body" else []))
    with pytest.raises(SystemExit):
        script.main()
