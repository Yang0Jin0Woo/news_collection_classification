import csv
import json

import pytest

from scripts import audit_review_reasons as audit


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_audit_reports_saved_review_reasons_and_both_model_checks(tmp_path):
    path = write_csv(tmp_path / "results.csv", [
        {"keyword": "경제", "title": "first", "model_category": "금융/투자", "model_category_score": "0.3", "score_margin": "0.02", "final_category": "검토필요", "rule_reason": "근거 없음: 모델 불확실", "content": "", "group_article_count": "8"},
        {"keyword": "과학", "title": "second", "model_category": "기술개발", "model_category_score": "0.45", "score_margin": "0.1", "final_category": "검토필요", "rule_reason": "규칙 점수 부족: 최고 1점", "content": "body"},
        {"keyword": "경제", "title": "third", "model_category": "금융/투자", "model_category_score": "0.7", "score_margin": "0.01", "final_category": "검토필요", "rule_reason": "주제 간 점수 차이 부족: 2대2", "content": ""},
        {"keyword": "스포츠", "title": "fourth", "model_category": "스포츠", "model_category_score": "0.8", "score_margin": "0.2", "final_category": "스포츠", "rule_reason": "", "content": "body"},
    ])
    before = path.read_bytes()
    report = audit.audit_csv(path, 0.5, 0.05)
    summary = report["summary"]
    assert path.read_bytes() == before
    assert summary["representative_article_count"] == 4
    assert summary["review_required_count"] == 3
    assert summary["review_rate_among_known_decisions"] == 0.75
    assert summary["review_reasons"] == {"no_rule_evidence": 1, "insufficient_rule_score": 1, "insufficient_rule_margin": 1}
    assert summary["review_model_score"]["below_threshold_count"] == 2
    assert summary["review_model_margin"]["below_threshold_count"] == 2
    assert summary["review_low_score_and_margin_count"] == 1
    assert summary["body"]["review_empty_count"] == 2
    assert report["by_keyword"]["경제"]["representative_article_count"] == 2
    assert report["by_predicted_category"]["금융/투자"]["review_required_count"] == 2


def test_missing_margin_is_unsupported_not_inferred_from_top3(tmp_path):
    path = write_csv(tmp_path / "old.csv", [{"title": "old", "final_category": "검토필요", "model_confidence": "0.6", "top3_scores": "0.6, 0.59, 0.1"}])
    summary = audit.audit_csv(path, 0.5, 0.05)["summary"]
    assert summary["model_score"]["column"] == "model_confidence"
    assert summary["model_margin"]["status"] == "unsupported"
    assert summary["model_margin"]["valid_count"] == 0
    assert summary["review_low_score_and_margin_count"] is None
    assert summary["body"]["status"] == "unsupported"
    assert summary["body"]["empty_count"] is None


def test_candidates_do_not_become_classified_or_gold_from_suggestions(tmp_path):
    path = write_csv(tmp_path / "candidates.csv", [{"title": "candidate", "keyword": "경제", "suggested_label": "금융/투자", "gold_label": "", "review_status": "pending", "reviewed_by": ""}])
    report = audit.audit_csv(path, 0.5, 0.05)
    assert report["summary"]["review_state_unknown_count"] == 1
    assert report["summary"]["review_rate_among_known_decisions"] is None
    assert report["gold_coverage"]["confirmed_gold_count"] == 0
    assert report["gold_coverage"]["candidate_label_count"] == 17
    assert report["gold_coverage"]["covered_label_count"] == 0
    assert report["by_predicted_category"][audit.UNKNOWN]["representative_article_count"] == 1


def test_confirmed_gold_requires_reviewer_and_no_suggestion():
    valid = {"gold_label": "스포츠", "review_status": "confirmed", "reviewed_by": "human", "suggested_label": "", "keyword": "A", "event_id": "event1", "split": "development"}
    rows = [valid, {**valid, "reviewed_by": ""}, {**valid, "suggested_label": "스포츠"}, {**valid, "review_status": "pending"}, {**valid, "gold_label": "unknown"}, {**valid, "split": "evaluation", "keyword": "B"}, {**valid, "event_id": "", "keyword": "C"}]
    result = audit.gold_coverage(rows)
    assert result["confirmed_gold_count"] == 3
    assert result["excluded_count"] == 4
    assert result["label_support"]["스포츠"]["explicit_event_count"] == 1
    assert result["label_support"]["스포츠"]["event_id_missing_count"] == 1
    assert result["events_shared_between_development_and_evaluation"] == ["event1"]
    assert result["keywords_shared_between_development_and_evaluation"] == []
    assert len(result["missing_labels"]) == 16


@pytest.mark.parametrize("value", ["", "not-number", "nan", "inf", "-0.1", "1.1"])
def test_invalid_scores_are_unknown(value, tmp_path):
    path = write_csv(tmp_path / "invalid.csv", [{"title": "invalid", "model_category_score": value, "score_margin": value, "final_category": "검토필요"}])
    summary = audit.audit_csv(path, 0.5, 0.05)["summary"]
    assert summary["model_score"]["status"] == "supported"
    assert summary["model_score"]["valid_count"] == 0
    assert summary["model_score"]["missing_or_invalid_count"] == 1
    assert summary["model_margin"]["mean"] is None


def test_model_errors_are_distinct_from_no_rule_evidence(tmp_path):
    path = write_csv(tmp_path / "errors.csv", [{"title": "error", "final_category": "분류실패", "rule_reason": "모델 분류 실패"}])
    summary = audit.audit_csv(path, 0.5, 0.05)["summary"]
    assert summary["error_category_count"] == 1
    assert summary["review_required_count"] == 1
    assert summary["review_reasons"] == {"model_error": 1}


@pytest.mark.parametrize("status", [
    "TIMEOUT", "BLOCKED_HTTP", "HTTP_ERROR", "REQUEST_FAILED",
    "GOOGLE_NEWS_WRAPPER", "RSS_WRAPPER", "CONSENT_OR_BLOCK_PAGE",
    "NO_USABLE_PARAGRAPHS",
])
def test_body_failure_diagnostics_are_known_and_never_count_as_reclassified(tmp_path, status):
    record = json.dumps({
        "status": status, "enrichment_status": status, "http_status": 403,
        "initial_final_category": "검토필요", "initial_rule_reason": "주제 근거 충돌: 두 주제",
    }, ensure_ascii=False)
    source = write_csv(tmp_path / "diagnostics.csv", [{
        "title": "synthetic", "final_category": "검토필요",
        "rule_reason": "주제 근거 충돌: 두 주제", "review_reclassification": record,
    }])
    original = source.read_bytes()
    summary = audit.audit_csv(source, 0.5, 0.05)["summary"]
    metadata = summary["review_reclassification"]
    assert summary["review_reasons"] == {"conflicting_rule_evidence": 1}
    assert metadata["unknown_count"] == 0
    assert metadata["status_counts"] == {status: 1}
    assert metadata["enrichment_status_counts"] == {status: 1}
    assert metadata["http_status_counts"] == {"403": 1}
    assert metadata["reclassified_count"] == metadata["resolved_count"] == 0
    assert source.read_bytes() == original


def test_output_is_optional_and_json_is_printed(tmp_path, capsys):
    path = write_csv(tmp_path / "in.csv", [{"title": "news"}])
    audit.main(["--input", str(path)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "stored_csv_read_only"
    assert len(payload["files"]) == 1
    assert list(tmp_path.iterdir()) == [path]


def test_output_is_new_and_source_bytes_stay_unchanged(tmp_path, capsys):
    source = write_csv(tmp_path / "in.csv", [{"title": "news"}])
    before = source.read_bytes()
    output = tmp_path / "audit.json"
    audit.main(["--dataset", str(source), "--output", str(output)])
    assert source.read_bytes() == before
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 1
    capsys.readouterr()


def test_existing_output_and_source_cannot_be_overwritten(tmp_path):
    source = write_csv(tmp_path / "in.csv", [{"title": "news"}])
    before = source.read_bytes()
    with pytest.raises(SystemExit) as exc:
        audit.main(["--input", str(source), "--output", str(source)])
    assert exc.value.code == 2
    assert source.read_bytes() == before


def test_duplicate_input_paths_are_rejected(tmp_path):
    source = write_csv(tmp_path / "in.csv", [{"title": "news"}])
    with pytest.raises(SystemExit) as exc:
        audit.main(["--input", str(source), str(source)])
    assert exc.value.code == 2


@pytest.mark.parametrize("threshold", ["nan", "inf", "-1", "2"])
def test_invalid_threshold_is_rejected(tmp_path, threshold):
    source = write_csv(tmp_path / "in.csv", [{"title": "news"}])
    with pytest.raises(SystemExit) as exc:
        audit.main(["--input", str(source), "--score-threshold", threshold])
    assert exc.value.code == 2


def test_multiple_inputs_remain_separate(tmp_path, capsys):
    first = write_csv(tmp_path / "first.csv", [{"title": "same", "final_category": "스포츠"}])
    second = write_csv(tmp_path / "second.csv", [{"title": "same"}])
    audit.main(["--input", str(first), str(second)])
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["files"]) == 2
    assert payload["files"][0]["summary"]["known_decision_count"] == 1
    assert payload["files"][1]["summary"]["known_decision_count"] == 0


def test_schema_rejects_duplicate_or_malformed_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("title,title\nfirst,second\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        audit.audit_csv(path, 0.5, 0.05)
    path.write_text("title,keyword\nfirst\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        audit.audit_csv(path, 0.5, 0.05)


def test_old_csv_retry_outcomes_are_unsupported_not_zero_success(tmp_path):
    source = write_csv(tmp_path / "old.csv", [{"title": "news", "final_category": "검토필요"}])
    metadata = audit.audit_csv(source, 0.5, 0.05)["summary"]["review_reclassification"]
    assert metadata["status"] == "unsupported"
    assert metadata["initial_review_count"] is None
    assert metadata["reclassified_count"] is None
    assert metadata["resolved_count"] is None


def test_retry_metadata_reports_initial_causes_and_saved_resolution(tmp_path):
    def record(status, reason):
        return json.dumps({"status": status, "initial_final_category": "검토필요", "initial_rule_reason": reason}, ensure_ascii=False)

    source = write_csv(tmp_path / "new.csv", [
        {"title": "resolved", "final_category": "사회", "review_reclassification": record("RECLASSIFIED", "근거 없음: 모델 판단 불확실")},
        {"title": "uncertain", "final_category": "검토필요", "review_reclassification": record("RECLASSIFIED", "규칙 점수 부족: 1점")},
        {"title": "no-body", "final_category": "검토필요", "review_reclassification": record("NO_CONTENT", "주제 간 점수 차이 부족: 2대2")},
        {"title": "confident", "final_category": "스포츠", "review_reclassification": "{}"},
    ])
    result = audit.audit_csv(source, 0.5, 0.05)
    metadata = result["summary"]["review_reclassification"]
    assert metadata["status"] == "supported"
    assert metadata["initial_review_count"] == 3
    assert metadata["reclassified_count"] == 2
    assert metadata["resolved_count"] == 1
    assert metadata["remaining_review_count"] == 1
    assert metadata["not_applicable_count"] == 1
    assert metadata["unknown_count"] == 0
    assert metadata["status_counts"] == {"NO_CONTENT": 1, "RECLASSIFIED": 2}
    assert metadata["initial_rule_reasons"] == {"no_rule_evidence": 1, "insufficient_rule_score": 1, "insufficient_rule_margin": 1}
    assert metadata["initial_rule_reason_unknown_count"] == 0


@pytest.mark.parametrize("raw", ["invalid-json", "", "null", "[]", "{\"status\":\"UNKNOWN\"}", "{\"status\":\"RECLASSIFIED\",\"initial_final_category\":\"스포츠\"}"])
def test_invalid_retry_metadata_is_unknown_not_success(tmp_path, raw):
    source = write_csv(tmp_path / "bad-meta.csv", [{"title": "news", "final_category": "사회", "review_reclassification": raw}])
    metadata = audit.audit_csv(source, 0.5, 0.05)["summary"]["review_reclassification"]
    assert metadata["status"] == "supported"
    assert metadata["unknown_count"] == 1
    assert metadata["recorded_count"] == 0
    assert metadata["resolved_count"] == 0


def test_retry_metadata_does_not_reconstruct_missing_initial_reason_or_final_state(tmp_path):
    raw = json.dumps({"status": "RECLASSIFIED", "initial_final_category": "검토필요", "initial_model_score": 0.3, "initial_score_margin": 0.01})
    source = write_csv(tmp_path / "incomplete.csv", [{"title": "news", "review_reclassification": raw}])
    metadata = audit.audit_csv(source, 0.5, 0.05)["summary"]["review_reclassification"]
    assert metadata["reclassified_count"] == 1
    assert metadata["resolved_count"] == 0
    assert metadata["resolution_unknown_count"] == 1
    assert metadata["initial_rule_reasons"] == {}
    assert metadata["initial_rule_reason_unknown_count"] == 1


def test_extraction_reasons_and_methods_are_counted_only_when_explicitly_recorded(tmp_path):
    records = [{"status": "RECLASSIFIED", "initial_final_category": "검토필요",
                "extraction_diagnostics": {"method": "BODY_CONTAINER_BLOCKS", "reason": "body_found"}},
               {"status": "NO_USABLE_PARAGRAPHS", "initial_final_category": "검토필요",
                "extraction_diagnostics": {"method": "", "reason": "multiple_body_candidates"}},
               {"status": "NO_USABLE_PARAGRAPHS", "initial_final_category": "검토필요"}]
    source = write_csv(tmp_path / "diagnostics.csv", [
        {"title": str(i), "review_reclassification": json.dumps(record)} for i, record in enumerate(records)])
    metadata = audit.audit_csv(source, .5, .05)["summary"]["review_reclassification"]
    assert metadata["extraction_method_counts"] == {"BODY_CONTAINER_BLOCKS": 1}
    assert metadata["extraction_reason_counts"] == {"body_found": 1, "multiple_body_candidates": 1}
