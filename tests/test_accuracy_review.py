"""Synthetic workflow tests; these fixtures are not real accuracy evidence."""
from copy import deepcopy
import csv
import json
from types import SimpleNamespace

import pytest

from news_classifier.accuracy_review import (
    REVIEW_COLUMNS, accuracy_review_readiness, prepare_accuracy_review, require_accuracy_review_ready,
)
from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.models import ModelPrediction
from news_classifier.rules.default_rules import CANDIDATE_LABELS, DEFAULT_RULE_SET
from scripts import evaluate_news, prepare_accuracy_review as review_cli


LABELS = ["기술개발", "교육/취업"]


def reviewed_rows(labels=LABELS):
    return [
        {
            "id": f"{split}-{index}-{number}", "event_id": f"{split}-event-{index}-{number}",
            "independent_event_check": "confirmed", "split": split, "keyword": f"{split} query",
            "title": f"synthetic {split} {index} {number}", "description": "",
            "content": "synthetic article body", "source": "fixture",
            "link": f"https://example.test/{split}/{index}/{number}",
            "gold_label": label, "review_status": "confirmed", "reviewed_by": "synthetic-fixture",
            "main_event": "synthetic event", "main_event_evidence": "synthetic evidence",
        }
        for split in ("development", "evaluation")
        for index, label in enumerate(labels) for number in (1, 2)
    ]


def write_rows(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def test_preparation_hides_predictions_and_does_not_invent_gold_events_or_split():
    candidate = {**reviewed_rows()[0], "model_category": "기술개발", "final_category": "검토필요",
                 "suggested_label": "교육/취업", "rule_reason": "prediction", "notes": "prediction hint"}
    original = deepcopy(candidate)
    row = prepare_accuracy_review([candidate, candidate], [], LABELS)[0]
    assert row["gold_label"] == row["event_id"] == row["split"] == row["reviewed_by"] == ""
    assert row["main_event"] == row["main_event_evidence"] == row["notes"] == ""
    assert row["review_status"] == row["independent_event_check"] == "pending"
    assert not ({"model_category", "final_category", "suggested_label", "rule_reason"} & row.keys())
    assert row["content"] == candidate["content"] and row["link"] == candidate["link"]
    assert candidate == original


def test_existing_approved_gold_is_preserved_but_event_confirmation_is_not_inferred():
    approved = {**reviewed_rows()[0], "event_id": "", "independent_event_check": "pending",
                "main_event": "", "main_event_evidence": ""}
    rows = prepare_accuracy_review([approved], [approved], LABELS)
    assert len(rows) == 1
    assert rows[0]["gold_label"] == approved["gold_label"]
    assert rows[0]["review_status"] == "confirmed" and rows[0]["split"] == "development"
    report = accuracy_review_readiness(rows, LABELS)
    assert report["confirmed_gold_count"] == 1 and not report["ready"]
    assert report["ready_article_count"] == 0
    assert approved["event_id"] == ""


@pytest.mark.parametrize("changes", [
    {"review_status": "pending"}, {"reviewed_by": ""}, {"gold_label": "invalid"},
    {"split": "evaluation"},
])
def test_confirmation_import_rejects_unapproved_or_holdout_data(changes):
    with pytest.raises(ValueError, match="explicitly confirmed development"):
        prepare_accuracy_review([], [{**reviewed_rows()[0], **changes}], LABELS)


def test_preparation_rejects_duplicate_ids_for_different_articles():
    left, right = reviewed_rows()[:2]
    right["id"] = left["id"]
    with pytest.raises(ValueError, match="same article id"):
        prepare_accuracy_review([left, right], [], LABELS)


@pytest.mark.parametrize("candidate", [{"title": " &nbsp; "}, 5])
def test_preparation_rejects_invalid_articles(candidate):
    with pytest.raises(ValueError):
        prepare_accuracy_review([candidate], [], LABELS)


def test_readiness_is_not_an_accuracy_or_event_semantics_claim():
    rows = reviewed_rows()
    original = deepcopy(rows)
    report = require_accuracy_review_ready(rows, LABELS)
    assert report["ready"] and report["ready_article_count"] == len(rows)
    assert report["confirmed_gold_count"] == len(rows)
    assert report["independent_evaluation_completed"] is False
    assert report["independent_events_per_label"]["evaluation"] == dict.fromkeys(LABELS, 2)
    assert rows == original


@pytest.mark.parametrize("field,value,missing", [
    ("id", "", "article_id"), ("title", "&nbsp;", "article_title"),
    ("link", "", "article_link"), ("gold_label", "invalid", "confirmed_gold"),
    ("review_status", "pending", "confirmed_gold"), ("reviewed_by", "", "confirmed_gold"),
    ("event_id", "", "explicit_event_id"),
    ("independent_event_check", "pending", "event_confirmation"),
    ("split", "", "split"), ("keyword", "", "keyword"),
    ("main_event", "&nbsp;", "main_event"), ("main_event_evidence", "", "main_event_evidence"),
    ("suggested_label", "prediction hint", "no_suggested_label"),
])
def test_readiness_requires_each_annotation_and_excludes_prediction_hints(field, value, missing):
    rows = reviewed_rows()
    rows[0][field] = value
    report = accuracy_review_readiness(rows, LABELS)
    assert not report["ready"] and report["incomplete_fields"][missing] == 1
    with pytest.raises(ValueError, match="독립 평가 자료 미준비"):
        require_accuracy_review_ready(rows, LABELS)


@pytest.mark.parametrize("field", ["event_id", "link", "keyword"])
def test_readiness_blocks_event_article_and_query_leakage(field):
    rows = reviewed_rows()
    rows[4][field] = rows[0][field]
    report = accuracy_review_readiness(rows, LABELS)
    assert not report["ready"]
    key = {"event_id": "events_crossing_splits", "link": "article_links_crossing_splits",
           "keyword": "keywords_crossing_splits"}[field]
    assert report[key]


def test_readiness_normalizes_event_and_keyword_identity():
    rows = reviewed_rows()
    rows[4]["event_id"] = " " + rows[0]["event_id"].upper() + " "
    rows[4]["keyword"] = rows[0]["keyword"].upper().replace(" ", "  ")
    report = accuracy_review_readiness(rows, LABELS)
    assert report["events_crossing_splits"] and report["keywords_crossing_splits"]


def test_multiple_articles_from_one_event_do_not_meet_minimum_event_count():
    rows = reviewed_rows()
    rows[1]["event_id"] = rows[0]["event_id"]
    report = accuracy_review_readiness(rows, LABELS)
    assert not report["ready"]
    assert report["independent_events_per_label"]["development"][LABELS[0]] == 1
    assert LABELS[0] in report["labels_needing_more_events"]["development"]


@pytest.mark.parametrize("minimum", [0, -1, True, 1.5])
def test_invalid_minimum_is_rejected(minimum):
    with pytest.raises(ValueError, match="positive integer"):
        accuracy_review_readiness(reviewed_rows(), LABELS, minimum_events_per_label=minimum)


def test_empty_data_is_not_ready():
    assert not accuracy_review_readiness([], LABELS)["ready"]


def test_prepare_cli_preserves_source_and_existing_output(tmp_path, monkeypatch):
    source, output = tmp_path / "source.csv", tmp_path / "review.csv"
    write_rows(source, reviewed_rows()[:1])
    original = source.read_bytes()
    monkeypatch.setattr("sys.argv", ["review", "prepare", "--dataset", str(source), "--output", str(output)])
    review_cli.main()
    report = json.loads(output.with_suffix(".readiness.json").read_text(encoding="utf-8"))
    assert report["confirmed_gold_count"] == 0 and not report["ready"]
    saved = output.read_bytes()
    with pytest.raises(SystemExit) as exc:
        review_cli.main()
    assert exc.value.code == 2
    assert source.read_bytes() == original and output.read_bytes() == saved


def test_check_cli_pending_data_has_nonzero_exit_and_no_model(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    write_rows(source, prepare_accuracy_review(reviewed_rows()[:1], [], LABELS))
    monkeypatch.setattr("sys.argv", ["review", "check", "--dataset", str(source)])
    with pytest.raises(SystemExit) as exc:
        review_cli.main()
    assert exc.value.code == 1


def test_strict_evaluation_stops_before_model_when_annotations_are_incomplete(tmp_path, monkeypatch):
    source = tmp_path / "pending.csv"
    write_rows(source, prepare_accuracy_review(reviewed_rows()[:1], [], LABELS))
    monkeypatch.setattr(evaluate_news, "build_pipeline", lambda _: pytest.fail("model must not be built"))
    monkeypatch.setattr("sys.argv", ["evaluate", "--dataset", str(source), "--require-independent-events"])
    with pytest.raises(SystemExit) as exc:
        evaluate_news.main()
    assert exc.value.code == 2


def test_strict_evaluation_writes_review_scope_without_changing_source(tmp_path, monkeypatch):
    rows = reviewed_rows(CANDIDATE_LABELS)
    source, output = tmp_path / "confirmed.csv", tmp_path / "results"
    write_rows(source, rows)
    original = source.read_bytes()
    gold_by_title = {row["title"]: row["gold_label"] for row in rows}
    inference_calls = []
    def classify_many(texts):
        inference_calls.append(texts)
        return [ModelPrediction(gold_by_title[text.splitlines()[0].removeprefix("기사제목: ")], .8, .3)
                for text in texts]
    pipeline = SimpleNamespace(
        classifier=SimpleNamespace(classify_many=classify_many, candidate_hypotheses=list(CANDIDATE_LABELS)),
        postprocessor=ClassificationPostProcessor(RuleEngine(DEFAULT_RULE_SET)),
    )
    monkeypatch.setattr(evaluate_news, "build_pipeline", lambda _: pipeline)
    monkeypatch.setattr("sys.argv", ["evaluate", "--dataset", str(source), "--require-independent-events",
                                   "--output-dir", str(output)])
    evaluate_news.main()
    report = json.loads((output / "evaluation_results.json").read_text(encoding="utf-8"))
    assert report["independent_review_readiness"]["ready"]
    assert report["unseen_keywords_validated"]
    assert report["evaluation_scope"].startswith("stored_article_classification")
    assert inference_calls
    assert source.read_bytes() == original
