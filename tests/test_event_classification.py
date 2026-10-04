"""합성 사례로 정책/연결 동작 검증. 실제 뉴스 정확도 평가 자료는 아님."""
from dataclasses import replace
import pytest

from news_classifier.classifiers.zero_shot_classifier import ZeroShotNewsClassifier
from news_classifier.classifiers.topic_descriptions import TOPIC_DESCRIPTIONS
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.evaluation import calculate_technology_bias_metrics
from news_classifier.evaluation_dataset import rows_to_items, validate_unseen_keywords
from news_classifier.models import NewsItem, ModelPrediction
from news_classifier.rules.default_rules import CANDIDATE_LABELS, DEFAULT_RULE_SET
from news_classifier.rules.policy import rule_set_fingerprint
from news_classifier.utils.text import article_description


def test_topic_descriptions_cover_all_configured_topics():
    assert set(TOPIC_DESCRIPTIONS) == set(CANDIDATE_LABELS)
    assert len(set(TOPIC_DESCRIPTIONS.values())) == len(CANDIDATE_LABELS)


def test_model_uses_descriptions_but_returns_public_labels_and_reuses_weights(monkeypatch):
    classifier = ZeroShotNewsClassifier("fake", "revision", CANDIDATE_LABELS)
    calls = []

    def fake_model(**kwargs):
        calls.append(kwargs)
        labels = kwargs["candidate_labels"]
        return [{"labels": [labels[1], labels[0]], "scores": [0.8, 0.2]} for _ in kwargs["sequences"]]

    monkeypatch.setattr(classifier, "_load", lambda: fake_model)
    new = classifier.classify_many(["자동차 신제품 출시", "제약 서비스 출시"])
    old = classifier.classify_many_legacy(["자동차 신제품 출시"])
    assert calls[0]["candidate_labels"] == list(TOPIC_DESCRIPTIONS.values())
    assert calls[1]["candidate_labels"] == CANDIDATE_LABELS
    assert [prediction.label for prediction in new + old] == ["제품/서비스"] * 3
    assert new[0].top3_labels == ["제품/서비스", "기술개발"]
    assert new[0].margin == pytest.approx(0.6)


def test_unknown_model_label_is_a_failure_not_a_description_in_csv(monkeypatch):
    classifier = ZeroShotNewsClassifier("fake", "revision", CANDIDATE_LABELS)
    monkeypatch.setattr(classifier, "_load", lambda: lambda **_: {"labels": ["unexpected"], "scores": [1.0]})
    assert classifier.classify("기사").label == "분류실패"


@pytest.mark.parametrize("descriptions", [{"기술개발": ""}, {"기술개발": "same", "제품/서비스": "same"}])
def test_invalid_descriptions_are_rejected(descriptions):
    with pytest.raises(ValueError, match="descriptions"):
        ZeroShotNewsClassifier("fake", "revision", ["기술개발", "제품/서비스"], candidate_descriptions=descriptions)


@pytest.mark.parametrize("description", [
    "신제품 출시 - 언론사", "신제품 출시 언론사", "'신제품' 출시 언론사", "언론사",
])
def test_title_echo_and_source_only_are_not_additional_context(description):
    assert article_description("신제품 출시", description, "언론사") == ""


def test_useful_description_and_keyword_inside_title_are_kept():
    item = NewsItem(keyword="정답힌트", title="자동차 신제품 출시", link="", source="언론사",
                    description="판매 시작일과 이용 가격 발표 언론사", content="본문 내용")
    text = item.classification_text()
    assert "자동차" in text and "판매 시작일" in text and "본문 내용" in text
    assert "정답힌트" not in text and "언론사" not in text
    assert item.description.endswith("언론사")  # 원본 보존


def test_input_budget_reserves_body_context():
    text = NewsItem(keyword="", title="제목" * 1000, description="설명" * 1000,
                    content="핵심본문" * 200, link="").classification_text()
    assert len(text) < 1200
    assert "기사본문: 핵심본문" in text


@pytest.mark.parametrize("title", [
    "GPU NPU 생성형 AI LLM", "전고체 배터리 LFP NCM", "OLED QD-OLED 마이크로LED",
    "휴머노이드 협동로봇 로봇팔", "원전 SMR ESS 태양광 풍력",
])
def test_technology_entities_alone_do_not_create_rule_evidence(title):
    engine = RuleEngine(DEFAULT_RULE_SET)
    assert engine.calculate_scores(title)["기술개발"] == 0
    decision = engine.decide(title, "", "", ModelPrediction("기술개발", 0.3, 0.01))
    assert decision.final_label == "검토필요"
    assert not decision.rule_applied


@pytest.mark.parametrize("title,label", [
    ("자동차 연구 개발 성능 고도화", "기술개발"),
    ("식품 신제품 출시 서비스 공개", "제품/서비스"),
    ("게임 기업 인수 합병 계약", "기업동향"),
    ("의료 공장 양산 증설 공급망", "생산/공급망"),
    ("농업 정책 규제 법안 제도", "정책/규제"),
    ("유통 주가 목표가 영업이익", "금융/투자"),
    ("건설 시장 규모 수요 회복", "시장/산업"),
    ("관광 노동조합 파업 단체교섭", "노동/노사"),
    ("식품 수출 통제 무역장벽", "국제/통상"),
])
def test_event_evidence_works_outside_ai(title, label):
    decision = RuleEngine(DEFAULT_RULE_SET).decide(title, "", "", ModelPrediction("기술개발", 0.3, 0.01))
    assert decision.final_label == label
    assert decision.rule_applied


def test_company_and_country_names_alone_are_not_event_evidence():
    scores = RuleEngine(DEFAULT_RULE_SET).calculate_scores("삼성 구글 현대차 미국 중국 일본")
    assert scores["기업동향"] == scores["국제/통상"] == 0


def test_context_flag_changes_rule_fingerprint():
    labels = tuple(replace(label, terms=tuple(replace(term, context_only=False) for term in label.terms))
                   for label in DEFAULT_RULE_SET.labels)
    assert rule_set_fingerprint(DEFAULT_RULE_SET) != rule_set_fingerprint(replace(DEFAULT_RULE_SET, labels=labels))


def test_evaluation_restores_optional_body_without_breaking_old_csv():
    base = {"keyword": "", "title": "제목", "description": "설명"}
    assert rows_to_items([base])[0].content == ""
    assert rows_to_items([{**base, "content": "실제 본문"}])[0].content == "실제 본문"


def test_new_keyword_validation_rejects_normalized_overlap():
    with pytest.raises(ValueError, match="both development and evaluation"):
        validate_unseen_keywords([{"keyword": " AI 반도체 "}], [{"keyword": "ai반도체"}])
    validate_unseen_keywords([{"keyword": "자동차"}], [{"keyword": "식품"}])


def test_technology_bias_counts_mistakes_not_category_balance():
    metrics = calculate_technology_bias_metrics(
        ["기술개발", "제품/서비스", "정책/규제", "금융/투자"],
        ["기술개발", "기술개발", "검토필요", "금융/투자"],
    )
    assert metrics["technology_predicted_count"] == 2
    assert metrics["technology_false_positive_count"] == 1
    assert metrics["technology_false_positive_rate"] == pytest.approx(1 / 3)


def test_source_name_in_rss_description_is_not_rule_evidence():
    item = NewsItem(keyword="임의검색어", title="내용 부족", description="내용 부족 기술개발신문",
                    source="기술개발신문", link="")
    processor = ClassificationPostProcessor(RuleEngine(DEFAULT_RULE_SET))
    result = processor.process(item, ModelPrediction("기술개발", 0.3, 0.01))
    assert result.rule_decision.final_label == "검토필요"
    assert result.rule_decision.rule_match_count == 0
    assert result.item.description == item.description
