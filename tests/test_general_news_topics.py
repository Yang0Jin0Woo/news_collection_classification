"""합성 예측으로 일반 뉴스 주제의 연결 검증. 실제 뉴스 정확도 평가 아님."""

import json

import pytest

from news_classifier.classifiers.base import NewsClassifier
from news_classifier.classifiers.confidence import (
    ConfidenceThresholds,
    load_confidence_thresholds,
)
from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.classifiers.topic_descriptions import (
    GENERAL_NEWS_LABELS,
    TOPIC_DESCRIPTIONS,
)
from news_classifier.collectors.base import NewsCollector
from news_classifier.dedup.title_deduplicator import TitleSourceDeduplicator
from news_classifier.evaluation import calculate_metrics
from news_classifier.models import ModelPrediction, NewsItem, PipelineStatus
from news_classifier.pipeline import NewsPipeline
from news_classifier.reporting.summary_report import build_summary
from news_classifier.rules.default_rules import (
    CANDIDATE_LABELS,
    DEFAULT_RULE_SET,
    validate_rule_configuration,
)


def test_general_news_taxonomy_has_unique_descriptions_for_all_seventeen_topics():
    assert GENERAL_NEWS_LABELS == (
        "교육/취업", "사회", "정치", "문화/연예", "스포츠", "건강/의료", "생활/환경",
    )
    assert len(CANDIDATE_LABELS) == 17
    assert len(set(CANDIDATE_LABELS)) == len(CANDIDATE_LABELS)
    assert set(TOPIC_DESCRIPTIONS) == set(CANDIDATE_LABELS)
    assert len(set(TOPIC_DESCRIPTIONS.values())) == len(CANDIDATE_LABELS)
    assert all(description.strip() for description in TOPIC_DESCRIPTIONS.values())
    validate_rule_configuration(DEFAULT_RULE_SET)


@pytest.mark.parametrize("label", GENERAL_NEWS_LABELS)
def test_new_topics_have_no_unverified_direct_rules(label):
    policy = DEFAULT_RULE_SET.label_policy(label)
    assert policy.no_direct_rules
    assert policy.terms == ()
    assert label not in {item.label for item in DEFAULT_RULE_SET.direct_rule_labels}


@pytest.mark.parametrize("label", GENERAL_NEWS_LABELS)
@pytest.mark.parametrize("title", [
    "프로야구 경기 결과와 연장전 승리 소식",
    "정부 지원 정책과 법안 제도 규제 소식",
])
def test_confident_new_topic_is_not_overridden_by_legacy_context(label, title):
    # 모델 결과 보존 정책을 검증하는 합성 입력. 제목의 실제 정답을 주장하지 않음.
    item = NewsItem(keyword="공통검색어", title=title, link="")
    result = ClassificationPostProcessor(RuleEngine(DEFAULT_RULE_SET)).process(
        item, ModelPrediction(label, 0.8, 0.2),
    )
    assert result.rule_decision.final_label == label
    assert not result.rule_decision.rule_applied
    assert result.decision_source == "MODEL"


def test_default_general_news_policy_does_not_force_sports_to_other():
    engine = RuleEngine(DEFAULT_RULE_SET)
    assert DEFAULT_RULE_SET.unrelated_signal_groups == ()
    decision = engine.decide(
        "프로야구 경기 결과", "연장전 끝에 승리했다", "",
        ModelPrediction("스포츠", 0.8, 0.2),
    )
    assert decision.final_label == "스포츠"
    assert not decision.rule_applied
    # 검증된 스포츠 규칙이 없는 상태에서 규칙 단독은 주제를 강제하지 않음.
    assert engine.rule_only_decision("프로야구 경기 결과", "연장전 끝에 승리했다") == "검토필요"


@pytest.mark.parametrize("label", GENERAL_NEWS_LABELS)
def test_uncertain_new_topic_without_event_evidence_still_requires_review(label):
    item = NewsItem(keyword="공통검색어", title="짧은 단신", description="내용 부족", link="")
    result = ClassificationPostProcessor(RuleEngine(DEFAULT_RULE_SET)).process(
        item, ModelPrediction(label, 0.3, 0.01),
    )
    assert result.rule_decision.final_label == "검토필요"
    assert not result.rule_decision.rule_applied
    assert result.rule_decision.rule_match_count == 0
    assert result.rule_decision.rule_reason.startswith("근거 없음:")
    assert result.decision_source == "REVIEW"


def test_one_keyword_can_preserve_multiple_general_final_topics_in_pipeline():
    # 판정 연결만 검증하며 검색어별 카테고리 균등 분포를 강제하지 않음.
    cases = [
        ("지역 대학교 입학 상담 행사", "교육/취업"),
        ("주민 안전과 범죄 예방 소식", "사회"),
        ("국회의원 선거와 정당 활동 소식", "정치"),
        ("영화와 공연 예술 소식", "문화/연예"),
        ("축구 경기 결과와 선수 기록", "스포츠"),
        ("건강검진과 질병 예방 소식", "건강/의료"),
        ("분리배출과 날씨 생활 소식", "생활/환경"),
    ]

    class GeneralNewsCollector(NewsCollector):
        def fetch(self, keyword, limit):
            return [
                NewsItem(keyword=keyword, title=title, source="합성 자료",
                         link=f"https://example.com/synthetic/{index}")
                for index, (title, _) in enumerate(cases[:limit])
            ]

    class GeneralNewsClassifier(NewsClassifier):
        def __init__(self):
            self.inputs = []

        def classify(self, text):
            self.inputs.append(text)
            label = next(label for title, label in cases if title in text)
            return ModelPrediction(label, 0.8, 0.2)

    classifier = GeneralNewsClassifier()
    result = NewsPipeline(
        collector=GeneralNewsCollector(),
        classifier=classifier,
        postprocessor=ClassificationPostProcessor(RuleEngine(DEFAULT_RULE_SET)),
        deduplicator=TitleSourceDeduplicator(),
    ).run("자동차", limit=len(cases))

    assert result.status == PipelineStatus.SUCCESS
    assert {item.item.keyword for item in result.results} == {"자동차"}
    assert all("자동차" not in text for text in classifier.inputs)
    assert [item.rule_decision.final_label for item in result.results] == list(GENERAL_NEWS_LABELS)
    assert all(item.decision_source == "MODEL" for item in result.results)
    summary = build_summary(result.results)
    assert summary.category_counts == {label: 1 for label in GENERAL_NEWS_LABELS}
    assert summary.review_count == 0
    assert summary.total_count == len(cases)


def test_evaluation_metrics_include_general_topics_without_treating_review_as_topic():
    expected = list(GENERAL_NEWS_LABELS)
    predicted = ["검토필요"] + expected[1:]
    report = calculate_metrics(expected, predicted, CANDIDATE_LABELS)
    assert set(report.per_category) == set(CANDIDATE_LABELS)
    assert "검토필요" not in report.per_category
    assert report.review_count == 1
    assert report.confusion_matrix[expected[0]]["검토필요"] == 1
    for label in GENERAL_NEWS_LABELS:
        assert report.per_category[label].support == 1
    assert report.to_dict()["wrong_decided_count"] == 0


def test_old_ten_topic_confidence_profile_is_not_reused_for_expanded_taxonomy(tmp_path):
    path = tmp_path / "old-confidence.json"
    old_labels = [label for label in CANDIDATE_LABELS if label not in GENERAL_NEWS_LABELS]
    assert len(old_labels) == 10
    path.write_text(json.dumps({
        "schema_version": 1,
        "candidate_labels": old_labels,
        "thresholds": ConfidenceThresholds().to_dict(),
    }, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="candidate labels"):
        load_confidence_thresholds(path, expected_candidate_labels=CANDIDATE_LABELS)
