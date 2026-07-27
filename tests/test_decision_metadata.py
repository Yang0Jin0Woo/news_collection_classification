from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.models import ModelPrediction, NewsItem
from news_classifier.reporting.summary_report import build_summary
from news_classifier.rules.default_rules import DEFAULT_RULE_SET


def classify(
    prediction: ModelPrediction,
    title: str = "짧은 단신",
    description: str = "내용 부족",
):
    item = NewsItem(
        keyword="AI",
        title=title,
        description=description,
        link="https://example.com/news",
    )
    processor = ClassificationPostProcessor(RuleEngine(DEFAULT_RULE_SET))
    return processor.process(item, prediction)


def test_model_confidence_is_bound_to_original_model_prediction():
    result = classify(
        ModelPrediction("기술개발", 0.85, 0.80),
        title="상장 첫날 따따블 코스모로보틱스",
        description="단숨에 로봇 대장주 우뚝",
    )
    row = result.to_row()

    assert row["model_category"] == "기술개발"
    assert row["model_confidence"] == 0.85
    assert row["model_confidence_level"] == "높음"
    assert row["final_category"] == "금융/투자"
    assert row["decision_source"] == "RULE"
    assert row["review_required"] is False
    assert row["final_decision_status"] == "DECIDED"
    assert "confidence_level" not in row


def test_review_decision_is_not_counted_as_rule_correction():
    result = classify(ModelPrediction("기술개발", 0.20, 0.01))

    assert result.rule_decision.final_label == "검토필요"
    assert result.rule_decision.rule_applied is False
    assert result.decision_source == "REVIEW"
    assert result.review_required is True
    assert result.final_decision_status == "REVIEW_REQUIRED"

    report = build_summary([result])
    assert report.rule_applied_count == 0
    assert report.review_count == 1
    assert report.error_count == 0


def test_failed_model_is_recorded_as_error_and_requires_review():
    result = classify(ModelPrediction.failed())

    assert result.rule_decision.final_label == "분류실패"
    assert result.decision_source == "ERROR"
    assert result.review_required is True
    assert result.final_decision_status == "ERROR"

    report = build_summary([result])
    assert report.rule_applied_count == 0
    assert report.review_count == 0
    assert report.error_count == 1
