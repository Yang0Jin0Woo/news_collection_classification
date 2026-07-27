import pytest

from data.sample_news_cases import SAMPLE_NEWS_CASES
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.models import ModelPrediction
from news_classifier.rules.default_rules import CANDIDATE_LABELS, RULES


@pytest.mark.parametrize(
    "case",
    SAMPLE_NEWS_CASES,
    ids=[case["id"] for case in SAMPLE_NEWS_CASES],
)
def test_rule_engine_matches_expected_label(case):
    decision = RuleEngine(RULES).decide(
        title=case["title"],
        description=case["description"],
        content="",
        prediction=ModelPrediction("검토필요", 0.30, 0.01, [], []),
    )

    assert decision.final_label == case["expected"]
    assert decision.rule_applied is True
    assert decision.rule_match_count >= 2


def test_sample_cases_cover_every_candidate_label():
    assert {case["expected"] for case in SAMPLE_NEWS_CASES} == set(CANDIDATE_LABELS)
    assert len({case["id"] for case in SAMPLE_NEWS_CASES}) == len(SAMPLE_NEWS_CASES)
