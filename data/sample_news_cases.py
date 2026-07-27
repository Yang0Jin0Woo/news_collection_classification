"""수동으로 선별한 규칙 엔진 회귀 테스트용 뉴스 사례.

이 데이터는 실제 서비스 정확도를 주장하기 위한 벤치마크가 아니다.
각 카테고리의 대표 문맥이 의도한 규칙으로 분류되는지 검증한다.
"""

SAMPLE_NEWS_CASES = [
    {
        "id": "technology-ai-chip",
        "expected": "기술개발",
        "title": "국내 연구진, 차세대 AI 반도체 알고리즘 개발",
        "description": "추론 성능과 모델 최적화 기술을 공개했다.",
    },
    {
        "id": "technology-solid-state-battery",
        "expected": "기술개발",
        "title": "전고체 배터리 에너지밀도 향상 연구",
        "description": "차세대 소재 프로토타입의 성능 실증을 진행했다.",
    },
    {
        "id": "product-ai-search",
        "expected": "제품/서비스",
        "title": "네이버, 생성형 AI 검색 앱 정식 출시",
        "description": "사용자 대상 구독 서비스와 API를 공개했다.",
    },
    {
        "id": "product-logistics-robot",
        "expected": "제품/서비스",
        "title": "물류 로봇 신제품 고객사 도입",
        "description": "서비스 로봇 솔루션 운영을 개시했다.",
    },
    {
        "id": "company-semiconductor-partnership",
        "expected": "기업동향",
        "title": "삼성전자와 인텔, 반도체 사업 제휴",
        "description": "양사가 글로벌 파트너십과 공동 사업 계약을 체결했다.",
    },
    {
        "id": "company-battery-acquisition",
        "expected": "기업동향",
        "title": "배터리 기업, 해외 법인 인수 추진",
        "description": "신사업 진출과 조직 확장 전략을 발표했다.",
    },
    {
        "id": "supply-battery-material",
        "expected": "생산/공급망",
        "title": "배터리 양극재 공장 생산라인 증설",
        "description": "리튬 조달과 소재 공급망을 강화한다.",
    },
    {
        "id": "supply-semiconductor-foundry",
        "expected": "생산/공급망",
        "title": "반도체 파운드리 수율 개선 설비 가동",
        "description": "부품 조달과 생산 라인을 확대했다.",
    },
    {
        "id": "policy-ai-law",
        "expected": "정책/규제",
        "title": "정부, AI 기본법 시행령 발표",
        "description": "개인정보 규제와 가이드라인 제도를 마련했다.",
    },
    {
        "id": "policy-battery-subsidy",
        "expected": "정책/규제",
        "title": "정부, 전기차 배터리 보조금 인증제도 개편",
        "description": "산업부가 지원 정책과 표준을 발표했다.",
    },
    {
        "id": "finance-robot-listing",
        "expected": "금융/투자",
        "title": "로봇주 상장 첫날 주가 급등",
        "description": "공모가 대비 수익률 상승으로 증권가 목표가가 상향됐다.",
    },
    {
        "id": "finance-semiconductor-earnings",
        "expected": "금융/투자",
        "title": "반도체 기업 영업이익 흑자 전환",
        "description": "증권사 실적 리포트가 투자의견 매수를 유지했다.",
    },
    {
        "id": "market-oled",
        "expected": "시장/산업",
        "title": "OLED 패널 시장 수요 회복 전망",
        "description": "디스플레이 시장 점유율과 가격 경쟁을 분석했다.",
    },
    {
        "id": "market-ai-datacenter",
        "expected": "시장/산업",
        "title": "AI 데이터센터 전력 수요 성장",
        "description": "에너지 산업 시장 규모와 업황 전망이 개선됐다.",
    },
    {
        "id": "labor-semiconductor-union",
        "expected": "노동/노사",
        "title": "반도체 노조, 임금교섭 결렬로 파업 예고",
        "description": "노사 단체교섭과 근로자 고용이 쟁점으로 떠올랐다.",
    },
    {
        "id": "labor-battery-factory",
        "expected": "노동/노사",
        "title": "배터리 공장 직원 노조와 임금 협상",
        "description": "노동조합이 고용 조건과 교섭안을 발표했다.",
    },
    {
        "id": "trade-semiconductor-export-control",
        "expected": "국제/통상",
        "title": "미국과 중국, 반도체 수출통제 협상",
        "description": "미중 무역 제재와 글로벌 공급망을 논의했다.",
    },
    {
        "id": "trade-eu-battery-tariff",
        "expected": "국제/통상",
        "title": "EU, 중국산 배터리에 관세 부과",
        "description": "수입 규제와 핵심광물 통상 협상을 이어간다.",
    },
]
