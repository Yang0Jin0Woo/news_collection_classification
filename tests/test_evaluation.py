import pytest

from news_classifier.evaluation import calculate_metrics


LABELS = ["기술개발", "제품/서비스", "기업동향"]


def test_calculate_metrics_for_perfect_predictions():
    report = calculate_metrics(LABELS, LABELS, LABELS)

    assert report.accuracy == 1.0
    assert report.macro_f1 == 1.0
    assert all(metric.f1 == 1.0 for metric in report.per_category.values())


def test_calculate_metrics_counts_review_needed_as_false_negative():
    report = calculate_metrics(
        expected=["기술개발", "제품/서비스", "기업동향"],
        predicted=["기술개발", "검토필요", "제품/서비스"],
        labels=LABELS,
    )

    assert report.accuracy == pytest.approx(1 / 3)
    assert report.confusion_matrix["제품/서비스"]["검토필요"] == 1
    assert report.confusion_matrix["기업동향"]["제품/서비스"] == 1


def test_calculate_metrics_rejects_unknown_expected_label():
    with pytest.raises(ValueError, match="unknown expected labels"):
        calculate_metrics(["알 수 없음"], ["기술개발"], LABELS)
