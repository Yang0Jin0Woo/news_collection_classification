from dataclasses import replace

import pytest

from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.models import ModelPrediction
from news_classifier.rules.default_rules import DEFAULT_RULE_SET as RULES
from news_classifier.rules.policy import (
    LabelRulePolicy,
    RuleDecisionPolicy,
    RuleSet,
    RuleStrength,
    RuleTerm,
)


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
        prediction=ModelPrediction("금융/투자", 0.42, 0.01, [], []),
    )
    assert decision.final_label == "제품/서비스"
    assert decision.rule_applied is True


def test_rule_engine_overrides_low_confidence_finance_signal():
    engine = RuleEngine(RULES)
    decision = engine.decide(
        title="필라델피아 반도체 지수 SOX의 함정",
        description="뉴욕증시와 코스피 흐름 분석",
        content="",
        prediction=ModelPrediction("기술개발", 0.45, 0.01, [], []),
    )
    assert decision.final_label == "금융/투자"
    assert decision.rule_applied is True


def test_rule_engine_overrides_low_confidence_labor_signal():
    engine = RuleEngine(RULES)
    decision = engine.decide(
        title="삼성 반도체 사장단, 직접 노조 만나러 평택행",
        description="조건 없이 대화하겠다는 노사 관련 소식",
        content="",
        prediction=ModelPrediction("기술개발", 0.48, 0.01, [], []),
    )
    assert decision.final_label == "노동/노사"
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


def test_rule_engine_classifies_battery_supply_chain_signal():
    engine = RuleEngine(RULES)
    decision = engine.decide(
        title="배터리 양극재와 리튬 공급망 확보 경쟁",
        description="이차전지 소재 조달과 생산 라인 증설",
        content="",
        prediction=ModelPrediction("시장/산업", 0.39, 0.01, [], []),
    )
    assert decision.final_label == "생산/공급망"
    assert decision.rule_applied is True


def test_rule_engine_classifies_ai_technology_signal():
    engine = RuleEngine(RULES)
    decision = engine.decide(
        title="생성형 AI LLM 추론 성능 고도화",
        description="멀티모달 파운데이션 모델 개발 경쟁",
        content="",
        prediction=ModelPrediction("시장/산업", 0.38, 0.01, [], []),
    )
    assert decision.final_label == "기술개발"
    assert decision.rule_applied is True


def test_rule_engine_classifies_display_production_signal():
    engine = RuleEngine(RULES)
    decision = engine.decide(
        title="OLED 패널 증착 라인 수율 개선",
        description="디스플레이 생산 설비와 유리기판 공급 확대",
        content="",
        prediction=ModelPrediction("금융/투자", 0.37, 0.01, [], []),
    )
    assert decision.final_label == "생산/공급망"
    assert decision.rule_applied is True


def test_rule_engine_classifies_robot_product_signal():
    engine = RuleEngine(RULES)
    decision = engine.decide(
        title="물류 로봇 신제품 출시",
        description="서비스 로봇 솔루션을 고객사에 적용",
        content="",
        prediction=ModelPrediction("기술개발", 0.36, 0.01, [], []),
    )
    assert decision.final_label == "제품/서비스"
    assert decision.rule_applied is True


def test_rule_engine_classifies_energy_policy_signal():
    engine = RuleEngine(RULES)
    decision = engine.decide(
        title="정부 전력정책 개편과 전기요금 제도 논의",
        description="원전 정책과 재생에너지 정책 지원 방안",
        content="",
        prediction=ModelPrediction("시장/산업", 0.35, 0.01, [], []),
    )
    assert decision.final_label == "정책/규제"
    assert decision.rule_applied is True


def test_rule_engine_overrides_strong_technology_bias_with_finance_signal():
    engine = RuleEngine(RULES)
    decision = engine.decide(
        title="LG전자, 로봇주로 다시 보나",
        description="홈로봇 확장성에 목표가 48% 상향 등장",
        content="",
        prediction=ModelPrediction("기술개발", 0.57, 0.27, [], []),
    )
    assert decision.final_label == "금융/투자"
    assert decision.rule_applied is True


def test_rule_engine_overrides_strong_technology_bias_with_listing_signal():
    engine = RuleEngine(RULES)
    decision = engine.decide(
        title="상장 첫날 따따블 코스모로보틱스",
        description="단숨에 로봇 대장주 우뚝",
        content="",
        prediction=ModelPrediction("기술개발", 0.85, 0.80, [], []),
    )
    assert decision.final_label == "금융/투자"
    assert decision.rule_applied is True


def test_rule_engine_overrides_strong_technology_bias_with_robot_product_signal():
    engine = RuleEngine(RULES)
    decision = engine.decide(
        title="사람이 타고 변신까지 가능한 거대 메카 로봇 출시",
        description="탑승형 로봇 신제품 공개",
        content="",
        prediction=ModelPrediction("기술개발", 0.82, 0.78, [], []),
    )
    assert decision.final_label == "제품/서비스"
    assert decision.rule_applied is True


def test_rule_engine_classifies_robot_stock_surge_as_finance():
    engine = RuleEngine(RULES)
    decision = engine.decide(
        title="삼성전자만 오르는 줄 알았는데 LG전자 로봇 날개 달고 한 달새 88% 폭등",
        description="로봇주 강세와 주가 상승 분석",
        content="",
        prediction=ModelPrediction("기술개발", 0.32, 0.10, [], []),
    )
    assert decision.final_label == "금융/투자"
    assert decision.rule_applied is True


def test_rule_engine_classifies_robot_service_launch_as_product():
    engine = RuleEngine(RULES)
    decision = engine.decide(
        title="현대차·기아, 양재사옥에 관수 로봇 등 3종 로봇 서비스 개시",
        description="사옥 내 로봇 서비스 운영과 도입 확대",
        content="",
        prediction=ModelPrediction("기술개발", 0.76, 0.60, [], []),
    )
    assert decision.final_label == "제품/서비스"
    assert decision.rule_applied is True


def test_rule_engine_classifies_battery_policy_despite_technology_bias():
    engine = RuleEngine(RULES)
    decision = engine.decide(
        title="정부, 배터리 여권 인증 제도 시행",
        description="IRA 대응과 이차전지 보조금 지원 정책 발표",
        content="",
        prediction=ModelPrediction("기술개발", 0.74, 0.50, [], []),
    )
    assert decision.final_label == "정책/규제"
    assert decision.rule_applied is True


def test_rule_engine_classifies_ai_product_despite_technology_bias():
    engine = RuleEngine(RULES)
    decision = engine.decide(
        title="네이버, 생성형 AI 검색 앱 정식 출시",
        description="AI 서비스 구독 기능과 API 공개",
        content="",
        prediction=ModelPrediction("기술개발", 0.78, 0.62, [], []),
    )
    assert decision.final_label == "제품/서비스"
    assert decision.rule_applied is True


def test_rule_engine_classifies_display_market_despite_technology_bias():
    engine = RuleEngine(RULES)
    decision = engine.decide(
        title="OLED 패널 가격 반등에 디스플레이 시장 회복 전망",
        description="TV 패널 수요 회복과 시장 점유율 경쟁",
        content="",
        prediction=ModelPrediction("기술개발", 0.73, 0.52, [], []),
    )
    assert decision.final_label == "시장/산업"
    assert decision.rule_applied is True


def test_rule_engine_classifies_energy_supply_chain_despite_technology_bias():
    engine = RuleEngine(RULES)
    decision = engine.decide(
        title="전력망 확충에 변압기 공급망 증설 경쟁",
        description="송전망 설비와 케이블 생산 확대",
        content="",
        prediction=ModelPrediction("기술개발", 0.75, 0.55, [], []),
    )
    assert decision.final_label == "생산/공급망"
    assert decision.rule_applied is True


def test_rule_engine_classifies_lifestyle_robot_as_product():
    engine = RuleEngine(RULES)
    decision = engine.decide(
        title="로봇이 빨래 개고 마사지하고 살림꾼으로 등장",
        description="가사 로봇과 생활 로봇 서비스 공개",
        content="",
        prediction=ModelPrediction("기술개발", 0.72, 0.51, [], []),
    )
    assert decision.final_label == "제품/서비스"
    assert decision.rule_applied is True


def test_rule_engine_keeps_sports_in_general_news_taxonomy():
    decision = RuleEngine(RULES).decide(
        title="프로야구 경기 결과",
        description="주말 경기에서 연장전 끝에 승리했다.",
        content="",
        prediction=ModelPrediction("스포츠", 0.85, 0.30, [], []),
    )

    assert decision.final_label == "스포츠"
    assert decision.rule_applied is False


def small_rule_set(
    *,
    product_strength: RuleStrength = RuleStrength.WEAK,
    reverse_label_order: bool = False,
) -> RuleSet:
    technology = LabelRulePolicy(
        "기술개발",
        (RuleTerm("alpha"),),
        tie_priority=20,
    )
    product = LabelRulePolicy(
        "제품/서비스",
        (RuleTerm("beta", strength=product_strength),),
        tie_priority=10,
        allow_technology_bias_override=True,
    )
    direct_labels = (product, technology) if reverse_label_order else (technology, product)
    return RuleSet(
        version="test-weighted-rules",
        matcher_version="test-longest-matcher",
        labels=direct_labels + (
            LabelRulePolicy(
                "기타/무관",
                (),
                tie_priority=30,
                no_direct_rules=True,
            ),
        ),
        domain_terms=("ai",),
        unrelated_signal_groups=(("야구",), ("경기 결과",)),
        decision=RuleDecisionPolicy(
            min_rule_score=2,
            min_rule_score_margin=1,
            strong_override_score=4,
            strong_override_margin=2,
        ),
        other_label="기타/무관",
        technology_label="기술개발",
    )


def test_ascii_rule_does_not_match_inside_another_word():
    engine = RuleEngine(RULES)

    scores = engine.calculate_scores(
        "반도체 capital expenditure 확대",
        "business cost analysis",
    )

    assert scores["제품/서비스"] == 0  # api, os 부분 문자열 오탐 방지
    assert scores["기술개발"] == 0  # ess 부분 문자열 오탐 방지


def test_ascii_acronym_matches_when_joined_to_korean_text():
    engine = RuleEngine(RULES)

    scores = engine.calculate_scores("API서비스를 공개했다")
    decision = engine.decide(
        "AI반도체 프로야구 경기 결과 분석",
        "",
        "",
        ModelPrediction("기술개발", 0.80, 0.20, [], []),
    )

    assert scores["제품/서비스"] >= 1
    assert decision.final_label == "기술개발"


def test_korean_compound_uses_longer_rule_without_substring_double_count():
    engine = RuleEngine(RULES)

    ranking = engine.rank_rules("AI 스타트업 투자유치 성공")

    assert ranking.score_for("기업동향").matched_terms == ("투자유치",)
    assert ranking.score_for("기업동향").weighted_score == 2
    assert ranking.score_for("금융/투자").weighted_score == 0


def test_korean_compounds_and_particles_keep_rule_recall():
    engine = RuleEngine(RULES)

    technology = engine.calculate_scores("AI반도체산업 투자 확대")
    policy = engine.calculate_scores("정부에서도 AI 지원책을 발표")
    company = engine.calculate_scores("삼성전자 조직 개편")

    assert technology["기술개발"] == 0  # 분야명은 사건 근거가 아님
    assert policy["정책/규제"] >= 2
    assert company["기업동향"] == 1  # 기업명 제외, 조직 변화 근거만 반영


def test_spacing_aliases_have_the_same_score():
    engine = RuleEngine(RULES)

    spaced = engine.calculate_scores("탄소 규제 시행")
    compact = engine.calculate_scores("탄소규제 시행")
    export_spaced = engine.calculate_scores("수출 규제 강화")
    export_compact = engine.calculate_scores("수출규제 강화")

    assert spaced["정책/규제"] == compact["정책/규제"] == 2
    assert (
        export_spaced["국제/통상"]
        == export_compact["국제/통상"]
        == 2
    )


def test_longer_cross_label_phrase_suppresses_shorter_rule():
    engine = RuleEngine(RULES)

    ranking = engine.rank_rules("글로벌 공급망 재편")

    assert ranking.score_for("국제/통상").weighted_score == 2
    assert ranking.score_for("생산/공급망").weighted_score == 0
    assert engine.rule_only_decision("글로벌 공급망 재편") == "국제/통상"


def test_long_phrase_suppresses_overlapping_shorter_terms():
    engine = RuleEngine(RULES)

    market_score = engine.rank_rules(
        "AI 시장 점유율 확대"
    ).score_for("시장/산업")

    assert market_score.weighted_score == 2
    assert market_score.match_count == 1
    assert market_score.matched_terms == ("시장 점유율",)


def test_repeated_phrase_counts_as_one_piece_of_evidence():
    engine = RuleEngine(RULES)

    once = engine.calculate_scores("AI 시장 점유율 확대")
    repeated = engine.calculate_scores(
        "AI 시장 점유율 확대, AI 시장 점유율 재확인"
    )

    assert repeated["시장/산업"] == once["시장/산업"]


def test_general_weak_words_do_not_force_a_rule_override():
    decision = RuleEngine(RULES).decide(
        title="AI 기술 시장 사업 지원 논의",
        description="일반적인 현황을 소개했다.",
        content="",
        prediction=ModelPrediction("기술개발", 0.30, 0.01, [], []),
    )

    assert decision.final_label == "검토필요"
    assert decision.rule_applied is False
    assert "규칙 점수 부족" in decision.rule_reason


def test_strong_phrase_has_more_weight_than_weak_word():
    engine = RuleEngine(RULES)

    strong_score = engine.calculate_scores("AI 시장 점유율 확대")["시장/산업"]
    weak_score = engine.calculate_scores("AI 시장 분석")["시장/산업"]

    assert strong_score == 2
    assert weak_score == 1


def test_rule_tie_uses_explicit_priority_but_requires_review():
    engine = RuleEngine(small_rule_set())

    ranking = engine.rank_rules("alpha beta")
    decision = engine.decide(
        "alpha beta",
        "",
        "",
        ModelPrediction("기술개발", 0.30, 0.01, [], []),
    )

    assert ranking.first.label == "제품/서비스"
    assert ranking.score_margin == 0
    assert decision.final_label == "검토필요"
    assert decision.rule_applied is False
    assert engine.rule_only_decision("alpha beta") == "검토필요"


def test_rule_ranking_does_not_depend_on_label_insertion_order():
    normal = RuleEngine(small_rule_set()).rank_rules("alpha beta")
    reversed_order = RuleEngine(
        small_rule_set(reverse_label_order=True)
    ).rank_rules("alpha beta")

    assert normal.first.label == reversed_order.first.label == "제품/서비스"
    assert normal.tied_labels == reversed_order.tied_labels


def test_rule_override_requires_top1_top2_score_margin():
    engine = RuleEngine(
        small_rule_set(product_strength=RuleStrength.STRONG)
    )

    decision = engine.decide(
        "alpha beta",
        "",
        "",
        ModelPrediction("기술개발", 0.30, 0.01, [], []),
    )

    assert decision.final_label == "제품/서비스"
    assert decision.rule_applied is True
    assert "2위 대비 +1" in decision.rule_reason


def test_weak_rules_alone_do_not_override_a_confident_model():
    decision = RuleEngine(RULES).decide(
        "AI 시장 산업 전망 수요",
        "",
        "",
        ModelPrediction("기술개발", 0.90, 0.80, [], []),
    )

    assert decision.final_label == "기술개발"
    assert decision.rule_applied is False


def test_nfkc_normalization_and_korean_particle_matching():
    engine = RuleEngine(RULES)

    scores = engine.calculate_scores(
        "정부가 ＡＩ 정책을 발표했다"
    )

    assert scores["정책/규제"] == 2


@pytest.mark.parametrize(
    ("title", "strength", "reason"),
    [
        ("unmatched", RuleStrength.WEAK, "근거 없음"),
        ("beta", RuleStrength.WEAK, "규칙 점수 부족"),
        ("alpha beta", RuleStrength.WEAK, "규칙 점수 부족"),
        ("alpha beta", RuleStrength.STRONG, "주제 간 점수 차이 부족"),
    ],
)
def test_review_reason_distinguishes_missing_score_and_margin(title, strength, reason):
    rules = small_rule_set(product_strength=strength)
    rules = replace(rules, decision=replace(rules.decision, min_rule_score_margin=2))
    decision = RuleEngine(rules).decide(
        title, "", "", ModelPrediction("기술개발", 0.30, 0.01, [], [])
    )

    assert decision.final_label == "검토필요"
    assert decision.rule_applied is False
    assert decision.rule_reason.startswith(reason + ":")


def test_review_reason_uses_custom_minimum_score():
    rules = small_rule_set(product_strength=RuleStrength.STRONG)
    rules = replace(rules, decision=replace(rules.decision, min_rule_score=3))
    decision = RuleEngine(rules).decide(
        "beta", "", "", ModelPrediction("기술개발", 0.30, 0.01, [], [])
    )

    assert "최고 2점, 최소 3점 필요" in decision.rule_reason


def test_rule_at_minimum_score_and_margin_still_applies():
    decision = RuleEngine(small_rule_set(product_strength=RuleStrength.STRONG)).decide(
        "alpha beta", "", "", ModelPrediction("기술개발", 0.30, 0.01, [], [])
    )

    assert decision.final_label == "제품/서비스"
    assert decision.rule_applied is True
    assert decision.rule_reason.startswith("규칙 보정:")
