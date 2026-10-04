import pytest

from news_classifier.evaluation import calculate_metrics, calculate_rule_correction_metrics


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


def test_review_rate_and_wrong_decisions_are_reported_separately():
    report = calculate_metrics(
        ["기술개발", "제품/서비스", "기업동향"],
        ["기술개발", "검토필요", "제품/서비스"], LABELS,
    ).to_dict()
    assert report["review_rate"] == pytest.approx(1 / 3)
    assert report["wrong_decided_count"] == 1


def test_rule_correction_metrics_separate_agreement_correction_and_harm():
    metrics = calculate_rule_correction_metrics(
        ["A", "A", "A", "A", "A"],
        ["A", "B", "A", "B", "B"],
        ["A", "A", "B", "B", "검토필요"],
        [True, True, True, True, False],
    )
    assert metrics == {
        "rule_applied_count": 4, "rule_changed_count": 2,
        "wrong_rule_count": 2, "wrong_rule_rate": 0.5,
        "corrected_count": 1, "harmful_change_count": 1,
    }


def test_rule_correction_metrics_without_rules_have_zero_counts():
    metrics = calculate_rule_correction_metrics(["A"], ["A"], ["A"], [False])
    assert all(value == 0 for value in metrics.values())


def test_rule_correction_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError, match="matching lengths"):
        calculate_rule_correction_metrics(["A"], ["A"], [], [True])
