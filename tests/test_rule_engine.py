from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.models import ModelPrediction
from news_classifier.rules.default_rules import RULES


def test_rule_engine_keeps_strong_model_result():
    engine = RuleEngine(RULES)
    decision = engine.decide(
        title="정부가 AI 반도체 지원 정책 발표",
        description="정책과 지원 내용 공개",
        content="",
        prediction=ModelPrediction("정책/규제", 0.80, 0.20, [], []),
    )
    assert decision.final_label == "정책/규제"
    assert decision.rule_applied is False


def test_rule_engine_overrides_weak_model_when_rule_is_strong():
    engine = RuleEngine(RULES)
    decision = engine.decide(
        title="신제품 출시와 고객사 적용 확대",
        description="서비스 업데이트와 솔루션 적용",
        content="",
        prediction=ModelPrediction("시장/투자", 0.42, 0.01, [], []),
    )
    assert decision.final_label == "제품/서비스"
    assert decision.rule_applied is True


def test_rule_engine_marks_review_needed_when_no_evidence():
    engine = RuleEngine(RULES)
    decision = engine.decide(
        title="짧은 단신",
        description="내용 부족",
        content="",
        prediction=ModelPrediction("기술개발", 0.20, 0.01, [], []),
    )
    assert decision.final_label == "검토필요"
