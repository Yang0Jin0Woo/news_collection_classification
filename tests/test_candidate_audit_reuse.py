from copy import deepcopy

import pytest

from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.rules import event_candidates
from news_classifier.rules.default_rules import DEFAULT_RULE_SET


CATALOG = {
    "금융/투자": ("IPO", "initial public offering"),
    "생산/공급망": ("공장 증설",),
    "기업동향": ("합병",),
}


def test_batch_candidate_audits_match_individual_reports_and_preserve_rows():
    rows = [
        {"id": "1", "title": "IPO 취소와 공장 증설 계획 - 뉴스", "source": "뉴스", "keyword": "임의 검색어"},
        {"title": "회사 소식", "content": "The former executive described an initial public offering.", "link": "https://example.com/a"},
        {"title": "합병 계약", "description": "합병 계약", "content": "", "split": "development"},
    ]
    selected = [(label, phrase) for label, phrases in CATALOG.items() for phrase in phrases]
    original = deepcopy(rows)
    expected = [
        event_candidates.audit_candidate(phrase, label, rows, DEFAULT_RULE_SET, candidate_phrases=CATALOG)
        for label, phrase in selected
    ]
    assert event_candidates.audit_candidates(selected, rows, DEFAULT_RULE_SET, candidate_phrases=CATALOG) == expected
    assert rows == original


def test_batch_prepares_article_fields_and_ids_once(monkeypatch):
    rows = [{"title": "IPO와 공장 증설"}, {"title": "일반 소식", "content": "합병 발표"}]
    calls = {"fields": 0, "id": 0}
    original_fields = event_candidates.article_fields
    original_id = event_candidates.article_review_id

    def fields(row):
        calls["fields"] += 1
        return original_fields(row)

    def article_id(row):
        calls["id"] += 1
        return original_id(row)

    monkeypatch.setattr(event_candidates, "article_fields", fields)
    monkeypatch.setattr(event_candidates, "article_review_id", article_id)
    event_candidates.audit_candidates(
        [("금융/투자", "IPO"), ("생산/공급망", "공장 증설"), ("기업동향", "합병")],
        rows, DEFAULT_RULE_SET, candidate_phrases=CATALOG,
    )
    assert calls == {"fields": len(rows), "id": len(rows)}


def test_batch_matches_each_text_phrase_pair_once(monkeypatch):
    calls = []
    original = event_candidates._phrase_spans

    def spans(text, phrase):
        calls.append((text, phrase))
        return original(text, phrase)

    monkeypatch.setattr(event_candidates, "_phrase_spans", spans)
    event_candidates.audit_candidates(
        [("금융/투자", "IPO"), ("금융/투자", "IPO")],
        [{"title": "IPO 취소와 공장 증설"}], DEFAULT_RULE_SET, candidate_phrases=CATALOG,
    )
    assert len(calls) == len(set(calls))


def test_candidate_cache_does_not_outlive_audit_or_reuse_changed_rows():
    rows = [{"title": "IPO announcement"}]
    selected = [("금융/투자", "IPO")]
    first = event_candidates.audit_candidates(selected, rows, DEFAULT_RULE_SET)
    rows[0]["title"] = "Unrelated announcement"
    second = event_candidates.audit_candidates(selected, rows, DEFAULT_RULE_SET)
    assert first[0]["matching_article_count"] == 1
    assert second[0]["matching_article_count"] == 0


def test_span_reuse_preserves_single_entity_decoding():
    phrase = "&amp;amp;"
    row = {"title": "정보", "content": "a &amp;amp;amp; b"}
    expected = event_candidates.phrase_evidence(phrase, event_candidates.article_fields(row))
    report = event_candidates.audit_candidates(
        [("기업동향", phrase)], [row], DEFAULT_RULE_SET,
        candidate_phrases={"기업동향": (phrase,)},
    )[0]
    assert expected
    assert report["matches"][0]["match_evidence"] == expected


def test_candidate_preparation_does_not_replace_runtime_title_evidence():
    row = {"title": "일반 소식 - IPO", "source": "IPO"}
    report = event_candidates.audit_candidates([("금융/투자", "IPO")], [row], DEFAULT_RULE_SET)[0]
    assert report["matching_article_count"] == 0
    evidence = event_candidates.rule_evidence_records(row, RuleEngine(DEFAULT_RULE_SET))
    assert any(term["phrase"].casefold() == "ipo" for term in evidence["terms"])


@pytest.mark.parametrize("split", ["evaluation", " Evaluation ", "validation"])
def test_batch_does_not_accept_evaluation_candidates(split):
    with pytest.raises(ValueError, match="평가용"):
        event_candidates.audit_candidates(
            [("금융/투자", "IPO")], [{"title": "IPO", "split": split}], DEFAULT_RULE_SET,
        )
