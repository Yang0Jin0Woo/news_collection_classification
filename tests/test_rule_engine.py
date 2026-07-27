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


def test_rule_engine_marks_clearly_unrelated_news_as_other():
    decision = RuleEngine(RULES).decide(
        title="프로야구 경기 결과",
        description="주말 경기에서 연장전 끝에 승리했다.",
        content="",
        prediction=ModelPrediction("노동/노사", 0.85, 0.30, [], []),
    )

    assert decision.final_label == "기타/무관"
    assert decision.rule_applied is True
