from dataclasses import replace

import pytest

from news_classifier.classifiers import postprocessor
from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.models import ModelPrediction, NewsItem
from news_classifier.rules.default_rules import DEFAULT_RULE_SET


@pytest.mark.parametrize("clean_context", [False, True])
@pytest.mark.parametrize("input_mode", ["without_body", "with_body"])
def test_chosen_and_baseline_engines_reuse_one_prepared_description(monkeypatch, clean_context, input_mode):
    selected = RuleEngine(replace(DEFAULT_RULE_SET, decision=replace(
        DEFAULT_RULE_SET.decision, review_model_score=.22, ambiguity_score=.22, ambiguity_margin=.05,
    )))
    baseline = RuleEngine(DEFAULT_RULE_SET)
    calls = {"selected": [], "baseline": [], "description": []}
    for name, engine in (("selected", selected), ("baseline", baseline)):
        decide = engine.decide

        def record(*, _name=name, _decide=decide, **kwargs):
            calls[_name].append(kwargs)
            return _decide(**kwargs)

        monkeypatch.setattr(engine, "decide", record)
    describe = postprocessor.article_description

    def prepare(*args):
        calls["description"].append(args)
        return describe(*args)

    monkeypatch.setattr(postprocessor, "article_description", prepare)
    item = NewsItem("query", "기사 제목", "", source="매체", description="기사 제목 - 매체",
                    content="상세 본문" if input_mode == "with_body" else "")
    prediction = ModelPrediction("정치", .22, .07)
    processor = ClassificationPostProcessor(selected, clean_context=clean_context,
                                            baseline_rule_engine=baseline, decision_input_mode=input_mode)
    result = processor.process(item, prediction)
    assert len(calls["description"]) == int(clean_context)
    assert len(calls["selected"]) == len(calls["baseline"]) == 1
    assert calls["selected"] == calls["baseline"]
    assert calls["selected"][0]["description"] == ("" if clean_context else item.description)
    assert result.item is item
    assert result.model_prediction is prediction
    assert result.decision_source == "MODEL"
    assert "기본 기준 검토 사유" in result.rule_decision.rule_reason


def test_failed_prediction_does_not_prepare_description(monkeypatch):
    monkeypatch.setattr(postprocessor, "article_description",
                        lambda *args: pytest.fail("failed prediction must not prepare context"))
    result = ClassificationPostProcessor(RuleEngine(DEFAULT_RULE_SET)).process(
        NewsItem("query", "기사", ""), ModelPrediction.failed(),
    )
    assert result.rule_decision.final_label == "분류실패"
    assert result.rule_decision.rule_reason == "모델 분류 실패"
