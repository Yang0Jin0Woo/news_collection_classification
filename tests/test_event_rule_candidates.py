from copy import deepcopy
from dataclasses import replace
import json
import re

import pytest

from scripts import review_rule_candidates
from news_classifier.rules.default_rules import DEFAULT_RULE_SET
from news_classifier.rules.event_candidates import (
    CANDIDATE_PHRASES,
    article_review_records,
    audit_candidate,
)
from news_classifier.rules.policy import RuleTerm


def audit(phrase, label, rows, **kwargs):
    return audit_candidate(phrase, label, rows, DEFAULT_RULE_SET, **kwargs)


def test_candidate_catalog_covers_every_non_other_topic_in_both_languages():
    assert set(CANDIDATE_PHRASES) == set(DEFAULT_RULE_SET.candidate_labels) - {DEFAULT_RULE_SET.other_label}
    assert len(CANDIDATE_PHRASES) == 16
    for phrases in CANDIDATE_PHRASES.values():
        assert any(re.search("[가-힣]", phrase) for phrase in phrases)
        assert any(re.search("[A-Za-z]", phrase) and not re.search("[가-힣]", phrase) for phrase in phrases)


@pytest.mark.parametrize("keyword", ["openai", "삼성", "공공 기관", "스포츠", "arbitrary never-seen query"])
def test_candidate_match_ignores_search_keyword(keyword):
    rows = [{"keyword": keyword, "title": "A corporation announced an IPO", "content": ""}]
    report = audit("IPO", "금융/투자", rows)
    assert report["matching_article_count"] == 1
    assert report["activation_eligible"] is False
    assert report["matches"][0]["keyword"] == keyword


@pytest.mark.parametrize("phrase,label", [("인재 양성", "교육/취업"), ("생산 능력 확대", "생산/공급망")])
def test_new_event_expressions_are_pending_candidates_not_default_rules(phrase, label):
    assert phrase in CANDIDATE_PHRASES[label]
    report = audit(phrase, label, [{"keyword": "어떤 검색어든", "title": phrase + " 계획 발표"}])
    assert report["matching_article_count"] == 1 and not report["activation_eligible"]
    assert not any(term.origin == "development" for policy in DEFAULT_RULE_SET.labels for term in policy.terms)


def test_keyword_and_source_only_occurrences_are_not_candidate_evidence():
    rows = [{"keyword": "IPO", "title": "A company report - IPO", "description": "IPO", "source": "IPO"}]
    report = audit("IPO", "금융/투자", rows)
    assert report["matching_article_count"] == 0


def test_body_matching_has_explicit_normalized_field_coordinates_without_source_mutation():
    rows = [{"id": "1", "title": "회사 소식", "description": "", "content": "  발표: ＩＰＯ\n plans.", "source": "News"}]
    before = deepcopy(rows)
    report = audit("IPO", "금융/투자", rows)
    evidence = report["matches"][0]["match_evidence"]
    assert len(evidence) == 1
    assert evidence[0]["field"] == "content"
    assert evidence[0]["matched_text"] == "ipo"
    assert evidence[0]["start"] == 4
    assert evidence[0]["end"] == 7
    assert evidence[0]["offset_basis"] == "normalized_nfkc_clean_casefold_field"
    assert report["matches"][0]["content_available"] is True
    assert rows == before


def test_title_and_duplicate_rss_description_count_once_not_two_articles():
    rows = [{"id": "1", "title": "IPO 계획 발표", "description": "IPO 계획 발표", "content": ""}]
    report = audit("IPO", "금융/투자", rows)
    assert report["matching_article_count"] == 1
    assert [e["field"] for e in report["matches"][0]["match_evidence"]] == ["title"]


def test_ascii_word_boundary_is_reused_for_body_evidence():
    rows = [{"title": "Report", "content": "capital expenditure and API2 updates"}]
    assert audit("api", "제품/서비스", rows)["matching_article_count"] == 0


def test_existing_context_term_is_distinguished_from_scoring_conflict():
    labels = list(DEFAULT_RULE_SET.labels)
    labels[0] = replace(labels[0], terms=labels[0].terms + (RuleTerm("sample context", context_only=True),))
    rule_set = replace(DEFAULT_RULE_SET, labels=tuple(labels))
    report = audit_candidate("sample context", "제품/서비스", [], rule_set)
    assert report["existing_aliases"] == [{"category": "기술개발", "phrase": "sample context"}]
    assert report["context_only_overlaps"][0]["context_only"] is True
    assert report["context_only_overlaps"][0]["scoring"] is False
    assert report["scoring_conflicts"] == []


def test_intertopic_scoring_and_candidate_alias_conflicts_are_reported_not_activated():
    catalog = {"금융/투자": ["IPO"], "기업동향": ["ipo"]}
    report = audit("IPO", "기업동향", [], candidate_phrases=catalog)
    assert report["candidate_conflicts"] == [{"category": "금융/투자", "phrase": "IPO"}]
    assert any(term["category"] == "금융/투자" for term in report["scoring_conflicts"])
    assert report["activation_eligible"] is False


@pytest.mark.parametrize("text,kind", [
    ("회사가 IPO를 취소했다", "cancellation"),
    ("The corporation scraps an IPO", "cancellation"),
    ("The corporation denies IPO plans", "denial"),
    ("IPO와 다른 투자 방식을 비교", "comparison"),
    ("IPO versus direct listing", "comparison"),
])
def test_context_is_a_human_review_warning_not_inferred_gold(text, kind):
    report = audit("IPO", "금융/투자", [{"title": text}])
    match = report["matches"][0]
    assert kind in {warning["kind"] for warning in match["context_warnings"]}
    assert all(warning["status"] == "human_context_check_required" for warning in match["context_warnings"])
    assert match["topic_mentions_are_gold_labels"] is False
    assert "gold_label" not in match
    assert report["activation_eligible"] is False


def test_article_mentions_in_two_topics_are_visible_for_primary_event_review():
    rows = [{"title": "IPO 추진 기업의 공장 증설 계획"}]
    report = audit("IPO", "금융/투자", rows)
    assert {"category": "생산/공급망", "phrases": ["공장 증설"]} in report["matches"][0]["other_candidate_topic_mentions"]


def test_supplied_event_ids_do_not_prove_independent_events_or_activate_rule():
    rows = [{
        "id": str(index), "title": "IPO 발표", "event_id": f"event-{index}",
        "split": "development", "gold_label": "금융/투자", "review_status": "confirmed", "reviewed_by": "human",
    } for index in range(2)]
    report = audit("IPO", "금융/투자", rows)
    evidence = report["development_evidence"]
    assert evidence["confirmed_row_count"] == 2
    assert evidence["provided_distinct_event_id_count"] == 2
    assert evidence["independent_event_count"] is None
    assert evidence["independent_events_verified"] is False
    assert evidence["baseline_errors_verified"] is False
    assert report["activation_eligible"] is False


def test_pending_or_machine_suggestion_is_not_confirmed_development_evidence():
    rows = [{"title": "IPO 발표", "event_id": "one", "split": "development", "gold_label": "금융/투자", "review_status": "confirmed", "reviewed_by": "human", "suggested_label": "금융/투자"}]
    assert audit("IPO", "금융/투자", rows)["development_evidence"]["confirmed_row_count"] == 0


def test_stored_csv_without_ids_uses_only_article_hash_and_blank_human_confirmation():
    rows = [{"keyword": "openai", "title": "IPO 발표", "link": "https://example.com/a", "model_category": "금융/투자", "final_category": "검토필요"}]
    before = deepcopy(rows)
    report = audit("IPO", "금융/투자", rows)
    match = report["matches"][0]
    assert match["id"].startswith("article-")
    assert match["article_id_basis"] == "stable_article_reference_not_event"
    assert match["event_id"] == ""
    assert report["development_evidence"]["independent_event_count"] is None
    records = article_review_records(rows)
    assert records[0]["article_id"] == match["id"]
    assert all(records[0]["human_confirmation"][key] == "" for key in ("event_id", "gold_label", "reviewed_by", "split"))
    assert records[0]["human_confirmation"]["review_status"] == "pending"
    assert records[0]["stored_prediction_not_gold"]["model_category"] == "금융/투자"
    assert rows == before


@pytest.mark.parametrize("split", ["evaluation", " Evaluation ", "TEST", "validation"])
def test_evaluation_leakage_rejected_by_public_audit_api(split):
    with pytest.raises(ValueError, match="평가용"):
        audit("IPO", "금융/투자", [{"title": "IPO", "split": split}])


def test_custom_candidates_cli_preserves_source_and_exports_confirmation_checklist(tmp_path, monkeypatch):
    dataset = tmp_path / "stored.csv"
    dataset.write_text("title,keyword,content\n일반 회사 소식,unseen,IPO plan\n", encoding="utf-8")
    before = dataset.read_bytes()
    output = tmp_path / "new-report.json"
    monkeypatch.setattr("sys.argv", ["review", "--dataset", str(dataset), "--output", str(output), "--candidate", "금융/투자:IPO"])
    review_rule_candidates.main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["query_independent"] is True
    assert payload["source_modified"] is False
    assert payload["automatic_activation"] is False
    assert payload["model_only_categories"] == ["기타/무관"]
    assert len(payload["covered_categories"]) == 16
    assert len(payload["candidates"]) == 1
    assert payload["candidates"][0]["matches"][0]["match_evidence"][0]["field"] == "content"
    assert payload["articles_for_human_confirmation"][0]["human_confirmation"]["gold_label"] == ""
    assert payload["source_keywords"] == ["unseen"]
    assert payload["human_review_checklist"]
    assert dataset.read_bytes() == before


def test_cli_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["review", "--dataset", str(tmp_path / "irrelevant.csv"), "--output", str(output)])
    with pytest.raises(SystemExit) as exc:
        review_rule_candidates.main()
    assert exc.value.code == 2
    assert output.read_text(encoding="utf-8") == "preserve"


def test_other_topic_cannot_be_proposed_as_direct_rule():
    with pytest.raises(ValueError, match="non-other"):
        audit("uncategorized", "기타/무관", [])


def test_missing_id_does_not_mix_unmatched_row_into_development_evidence():
    rows = [
        {"title": "IPO announcement"},
        {"title": "Unrelated", "event_id": "other", "split": "development", "gold_label": "금융/투자", "review_status": "confirmed", "reviewed_by": "human"},
    ]
    assert audit("IPO", "금융/투자", rows)["development_evidence"]["confirmed_row_count"] == 0
