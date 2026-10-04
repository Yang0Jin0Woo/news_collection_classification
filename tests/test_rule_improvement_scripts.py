from dataclasses import replace
import csv
import json
from types import SimpleNamespace

import pytest

from scripts import collect_evaluation_news, evaluate_news, review_rule_candidates
from news_classifier.collectors.base import NetworkCollectionError
from news_classifier.models import NewsItem, ModelPrediction
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.rules.default_rules import DEFAULT_RULE_SET
from news_classifier.rules.policy import RuleTerm


@pytest.fixture(autouse=True)
def no_collection_sleep(monkeypatch):
    monkeypatch.setattr(collect_evaluation_news.time, "sleep", lambda _: None)


class FakeCollector:
    def fetch(self, query, limit):
        if query == "failure":
            raise NetworkCollectionError("offline")
        return [NewsItem(keyword=query, title=f"{query}-{index}", link=f"https://example.com/{query}/{index}") for index in range(limit)]


def test_collection_round_robin_includes_later_searches():
    rows = collect_evaluation_news.collect_candidates(FakeCollector(), ["A", "B", "C"], 6, 5, set())
    assert [row["keyword"] for row in rows] == ["A", "B", "C", "A", "B", "C"]
    assert all(row["gold_label"] == "" and row["review_status"] == "pending" for row in rows)
    assert all(row["split"] == "" and row["reviewed_by"] == "" for row in rows)


def test_collection_continues_after_one_query_failure():
    rows = collect_evaluation_news.collect_candidates(FakeCollector(), ["failure", "B"], 3, 4, set())
    assert len(rows) == 3
    assert all(row["keyword"] == "B" for row in rows)


def test_collection_preserves_seen_events_across_groups():
    seen = set()
    first = collect_evaluation_news.collect_candidates(FakeCollector(), ["A"], 2, 3, seen)
    second = collect_evaluation_news.collect_candidates(FakeCollector(), ["A"], 2, 3, seen)
    assert len(first) == 2
    assert len(second) == 1
    assert first[0]["event_id"] != second[0]["event_id"]


def test_candidate_script_detects_existing_spacing_alias():
    report = review_rule_candidates.audit_candidate("시장규모", "시장/산업", [])
    assert report["status"] == "existing_or_conflicting_rule"
    assert report["existing_aliases"][0]["category"] == "시장/산업"


def test_candidate_script_detects_cross_category_overlap_without_confirming():
    rows = [{"id": "1", "event_id": "temporary", "title": "신입 공개 채용", "description": ""}]
    report = review_rule_candidates.audit_candidate("공개 채용", "노동/노사", rows)
    assert report["matching_article_count"] == 1
    assert {"category": "기업동향", "phrase": "채용"} in report["overlapping_rules"]
    assert report["human_confirmation_required"] is True
    assert rows[0].get("gold_label") is None


def test_candidate_script_does_not_match_ascii_inside_word():
    report = review_rule_candidates.audit_candidate("api", "제품/서비스", [{"id": "1", "title": "capital expenditure"}])
    assert report["matching_article_count"] == 0


def test_baseline_excludes_only_development_rules_and_keeps_thresholds():
    labels = list(DEFAULT_RULE_SET.labels)
    labels[0] = replace(labels[0], terms=labels[0].terms + (RuleTerm("새 표현", origin="development", evidence_event_ids=("1", "2")),))
    modified = replace(DEFAULT_RULE_SET, labels=tuple(labels))
    baseline = evaluate_news.baseline_rule_set(modified)
    assert baseline == DEFAULT_RULE_SET
    assert baseline.decision == modified.decision


def test_collection_refuses_to_overwrite_existing_review_file(tmp_path, monkeypatch):
    output = tmp_path / "existing.csv"
    output.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["collect", "--output", str(output)])
    with pytest.raises(SystemExit) as exc:
        collect_evaluation_news.main()
    assert exc.value.code == 2
    assert output.read_text(encoding="utf-8") == "preserve"


def test_candidate_selection_rejects_evaluation_split(tmp_path, monkeypatch):
    dataset = tmp_path / "cases.csv"
    dataset.write_text("id,title,split\n1,title,evaluation\n", encoding="utf-8")
    output = tmp_path / "candidates.json"
    monkeypatch.setattr("sys.argv", ["review", "--dataset", str(dataset), "--output", str(output)])
    with pytest.raises(SystemExit) as exc:
        review_rule_candidates.main()
    assert exc.value.code == 2
    assert not output.exists()


def test_evaluation_compares_same_predictions_and_preserves_model_errors(tmp_path, monkeypatch):
    dataset = tmp_path / "synthetic.csv"
    labels = list(DEFAULT_RULE_SET.candidate_labels)
    fields = ["id", "event_id", "split", "keyword", "title", "description", "gold_label", "review_status", "reviewed_by"]
    with dataset.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({
            "id": str(index), "event_id": str(index), "split": "evaluation",
            "keyword": "unused", "title": f"case {index}", "description": "",
            "gold_label": label, "review_status": "confirmed", "reviewed_by": "synthetic-test",
        } for index, label in enumerate(labels))

    calls = []
    def classify_many(texts):
        calls.append(texts)
        return [ModelPrediction(label, 0.8, 0.2) for label in labels[:-1]] + [ModelPrediction.failed()]

    pipeline = SimpleNamespace(
        classifier=SimpleNamespace(classify_many=classify_many),
        postprocessor=ClassificationPostProcessor(RuleEngine(DEFAULT_RULE_SET)),
    )
    monkeypatch.setattr(evaluate_news, "build_pipeline", lambda _: pipeline)
    output = tmp_path / "report"
    monkeypatch.setattr("sys.argv", ["evaluate", "--dataset", str(dataset), "--output-dir", str(output)])
    evaluate_news.main()
    payload = json.loads((output / "evaluation_results.json").read_text(encoding="utf-8"))
    assert len(calls) == 1
    assert payload["reports"]["하이브리드"] == payload["reports"]["추가 규칙 전 하이브리드"]
    assert payload["cases"][-1]["baseline_hybrid_label"] == "분류실패"
    assert payload["reports"]["하이브리드"]["error_count"] == 1
    assert "검토 비율" in (output / "evaluation_report.md").read_text(encoding="utf-8")


def test_no_results_does_not_create_empty_candidate_file(tmp_path, monkeypatch):
    monkeypatch.setattr(FakeCollector, "fetch", lambda *args, **kwargs: [])
    monkeypatch.setattr(collect_evaluation_news, "GoogleNewsRssCollector", lambda _: FakeCollector())
    output = tmp_path / "empty.csv"
    monkeypatch.setattr("sys.argv", ["collect", "--output", str(output)])
    with pytest.raises(SystemExit) as exc:
        collect_evaluation_news.main()
    assert exc.value.code == 1
    assert not output.exists()
