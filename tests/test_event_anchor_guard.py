"""Synthetic regression tests; not independent human-reviewed accuracy evidence."""
import pytest

from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.models import ModelPrediction
from news_classifier.rules.default_rules import DEFAULT_RULE_SET


def decide(title, body, *, guarded):
    return RuleEngine(DEFAULT_RULE_SET, experimental_event_anchors=guarded).decide(
        title, "", body, ModelPrediction("기술개발", .25, .01),
    )


def test_background_merger_history_cannot_support_experimental_rule_override():
    title = "대학 인력 양성 기관 개소"
    body = "대학은 학생을 양성한다. 교육 체계를 구축한다. 담당자는 과거 회사 합병 업무의 경력자이며 사업과 협력을 담당했다."
    assert decide(title, body, guarded=False).final_label == "기업동향"
    changed = decide(title, body, guarded=True)
    assert changed.final_label == "검토필요" and not changed.rule_applied
    assert changed.rule_reason.startswith("주요 사건 근거 부족")


def test_project_description_without_lead_evidence_cannot_force_technology():
    title = "기관의 예산 사용 현황 확인"
    body = "예산을 아직 쓰지 않았다. 집행 지연 상황이다. 해당 프로젝트에는 연구 개발과 성능 실증 기술이 쓰인다."
    assert decide(title, body, guarded=False).final_label == "기술개발"
    assert decide(title, body, guarded=True).final_label == "검토필요"


@pytest.mark.parametrize("title,body,label", [
    ("두 회사의 합병 계약 발표", "합병 계약을 발표했다. 경영 변화가 이어진다.", "기업동향"),
    ("신제품 출시와 서비스 제공", "신제품을 출시한다. 고객에게 서비스를 제공한다.", "제품/서비스"),
    ("정부 정책과 규제 발표", "정부가 정책과 규제를 발표했다. 적용 내용을 설명했다.", "정책/규제"),
    ("생산 공장의 양산 증설", "공장이 양산에 들어간다. 생산 증설 계획이다.", "생산/공급망"),
])
def test_experimental_guard_preserves_anchored_rule_cases(title, body, label):
    assert decide(title, body, guarded=False).final_label == label
    assert decide(title, body, guarded=True).final_label == label


def test_confident_model_is_preserved_without_rule_anchor():
    engine = RuleEngine(DEFAULT_RULE_SET, experimental_event_anchors=True)
    result = engine.decide("학생 교육 소식", "", "과거 회사 합병과 계약 이력을 소개했다.",
                           ModelPrediction("교육/취업", .85, .3))
    assert result.final_label == "교육/취업" and not result.rule_applied


def test_default_engine_never_enables_experiment_implicitly():
    assert RuleEngine(DEFAULT_RULE_SET).experimental_event_anchors is False
    with pytest.raises(ValueError, match="boolean"):
        RuleEngine(DEFAULT_RULE_SET, experimental_event_anchors="true")


def test_rule_only_prototype_uses_same_guard():
    title = "교육 소식"
    body = "교육을 진행한다. 학생을 양성한다. 과거 합병과 계약을 소개한다."
    plain = RuleEngine(DEFAULT_RULE_SET)
    guarded = RuleEngine(DEFAULT_RULE_SET, experimental_event_anchors=True)
    assert plain.rule_only_decision(title, "", body) == "기업동향"
    assert guarded.rule_only_decision(title, "", body) == "검토필요"
