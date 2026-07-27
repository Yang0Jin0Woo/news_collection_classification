from news_classifier.rules.policy import (
    LabelRulePolicy,
    RuleDecisionPolicy,
    RuleSet,
    RuleStrength,
    RuleTerm,
    normalize_rule_text,
    validate_rule_set,
)


_OTHER_LABEL = "기타/무관"
_CANDIDATE_LABELS = (
    "기술개발",
    "제품/서비스",
    "기업동향",
    "생산/공급망",
    "정책/규제",
    "금융/투자",
    "시장/산업",
    "노동/노사",
    "국제/통상",
    _OTHER_LABEL,
)

_LEGACY_KEYWORDS_BY_LABEL = {
    "금융/투자": [
        "etf", "주가", "주식", "증권", "투자", "투자처", "펀드",
        "수익률", "목표가", "매수", "매도", "시총", "상장", "공모",
        "배당", "국민 배당", "초과이윤", "영업이익", "손익", "흑자",
        "적자", "컨센서스", "월가", "뉴욕증시", "코스피", "나스닥",
        "필라델피아", "sox", "지수", "슈퍼사이클", "밸류에이션",
        "ipo", "상장예비심사", "투자의견", "리포트", "실적", "어닝",
        "배터리주", "로봇주", "전력기기주", "ai주", "ai 관련주",
        "폭등", "급등", "상향", "하향", "대장주", "따따블",
        "상장 첫날", "공모가", "증시 입성", "오르는", "한 달새",
        "상승", "강세", "랠리",
    ],
    "시장/산업": [
        "시장", "산업", "업황", "수요", "가격", "메모리",
        "d램", "dram", "낸드", "nand", "점유율", "매출", "출하량",
        "성장률", "전망", "생태계", "기금", "경쟁력", "수급",
        "호황", "불황", "사이클", "반도체 생태계", "배터리 시장",
        "이차전지 시장", "전기차 수요", "ess 시장", "ai 시장",
        "생성형 ai 시장", "데이터센터 수요", "디스플레이 시장",
        "oled 시장", "패널 가격", "로봇 시장", "자동화 시장",
        "전력 시장", "에너지 시장", "전력 수요", "전력난",
        "시장 규모", "시장 점유율", "수요 둔화", "수요 회복",
    ],
    "정책/규제": [
        "정부", "정책", "규제", "법안", "지원", "산업부", "과기정통부",
        "환경부", "금감원", "공정위", "국회", "세제", "인허가",
        "가이드라인", "표준", "인증", "보조금", "제도", "시행령",
        "관세",
        "ira", "인플레이션 감축법", "배터리 여권", "탄소규제",
        "탄소 규제", "ai 기본법", "ai 규제", "저작권", "개인정보",
        "데이터 규제", "로봇 규제", "실증특례", "전기요금", "전력정책",
        "전력 정책", "에너지 정책", "재생에너지 정책", "원전 정책",
    ],
    "생산/공급망": [
        "공급망", "생산", "양산", "공장", "수율", "라인", "파운드리",
        "소부장", "재고", "출하", "납품", "조달", "설비", "제조",
        "증설", "가동", "원자재", "물류", "부품",
        "배터리셀", "배터리 셀", "셀 공장", "양극재", "음극재",
        "전해액", "분리막", "리튬", "니켈", "코발트", "흑연",
        "전구체", "광물", "소재", "패널", "oled 라인", "증착",
        "유리기판", "감속기", "센서", "액추에이터", "로봇 부품",
        "변압기", "전력망", "송전망", "배전망", "송전선", "케이블",
    ],
    "기술개발": [
        "ai반도체", "npu", "gpu", "기술", "연구", "개발", "고도화",
        "성능", "아키텍처", "알고리즘", "모델", "특허", "차세대", "플랫폼",
        "실증", "데이터", "학습", "추론", "최적화", "프로토타입",
        "전고체", "전고체 배터리", "lfp", "ncm", "나트륨이온",
        "나트륨 이온", "리튬황", "배터리 수명", "에너지밀도",
        "생성형 ai", "llm", "대규모언어모델", "파운데이션 모델",
        "멀티모달", "ai 에이전트", "온디바이스 ai", "딥러닝",
        "oled", "qd-oled", "마이크로led", "microled", "유기발광",
        "봉지", "픽셀", "휴머노이드", "협동로봇", "자율주행 로봇",
        "로봇팔", "제어기", "slam", "원전", "smr", "ess", "재생에너지",
        "태양광", "풍력", "수소", "전력저장", "스마트그리드",
    ],
    "제품/서비스": [
        "출시", "서비스", "제품", "솔루션", "탑재", "적용", "고객사",
        "신제품", "프로", "버전", "업데이트", "구독", "사용자",
        "기능", "앱", "API", "베타", "상용화", "패키지",
        "배터리팩", "배터리 팩", "배터리 교체", "충전 서비스",
        "챗봇", "ai 서비스", "ai 검색", "코파일럿", "saaS",
        "tv 패널", "모니터", "웨어러블", "스마트폰 패널",
        "서비스 로봇", "물류 로봇", "배송 로봇", "청소 로봇",
        "홈로봇", "메카 로봇", "정찰 로봇", "정찰", "투입",
        "관수 로봇", "가사 로봇", "생활 로봇", "빨래", "마사지", "살림꾼",
        "os", "운영체제", "개시", "도입", "운영", "공개",
        "충전기", "전기차 충전", "에너지 관리", "전력 솔루션",
    ],
    "기업동향": [
        "협력", "제휴", "인수", "합병", "확장", "진출", "수주",
        "계약", "전략", "조직", "채용", "사업", "파트너십",
        "투자유치", "MOU", "컨소시엄", "법인", "대표", "임원",
        "사장단", "삼성", "sk하이닉스", "tsmc", "인텔",
        "lg에너지솔루션", "삼성sdi", "sk온", "catl", "파나소닉",
        "openai", "오픈ai", "구글", "마이크로소프트", "네이버",
        "lg디스플레이", "삼성디스플레이", "boe", "현대차", "두산로보틱스",
        "레인보우로보틱스", "lg전자", "코스모로보틱스", "한국전력", "한전",
        "두산에너빌리티", "현대차·기아", "현대차그룹", "기아",
        "os 경쟁", "사업 확장", "경쟁", "각축",
    ],
    "노동/노사": [
        "노조", "노동조합", "무노조", "노사", "임금", "파업", "교섭",
        "단체교섭", "쟁의", "고용", "근로자", "직원", "평택행",
        "조건 없이 대화",
    ],
    "국제/통상": [
        "미중", "미·중", "미국", "중국", "일본", "대만", "eu",
        "무역", "통상", "수출", "수입", "협상", "공급망 동맹",
        "칩4", "수출통제", "수출 통제", "ustr", "정상회담", "제재",
        "eu 배터리", "중국산 배터리", "희토류", "핵심광물",
        "수출 규제", "수입 규제", "무역장벽", "해외 진출",
        "글로벌 공급망", "국제에너지기구", "iea", "opec",
    ],
}

_DOMAIN_TERMS = (
    "반도체",
    "배터리",
    "이차전지",
    "2차전지",
    "ai",
    "인공지능",
    "디스플레이",
    "oled",
    "로봇",
    "전력",
    "에너지",
    "원전",
    "smr",
    "ess",
)

_UNRELATED_SIGNAL_GROUPS = (
    ("프로야구", "야구"),
    ("축구", "농구", "배구"),
    ("경기 결과", "연장전"),
    ("선수", "감독"),
    ("연예", "배우", "가수", "아이돌"),
    ("드라마", "영화", "예능"),
    ("날씨", "기상"),
    ("여행", "축제", "맛집"),
    ("요리", "레시피"),
)

# 기존 규칙을 임의로 늘리지 않고, 의미가 구체적인 기존 구문만 강한 근거로 분리한다.
# 앞으로 추가하는 규칙은 development 데이터의 confirmed event_id를 근거로 남겨야 한다.
_STRONG_KEYWORDS = frozenset({
    "목표가",
    "상장예비심사",
    "투자의견",
    "상장 첫날",
    "공모가",
    "증시 입성",
    "영업이익",
    "ai 관련주",
    "시장 규모",
    "시장 점유율",
    "수요 둔화",
    "수요 회복",
    "데이터센터 수요",
    "패널 가격",
    "전력난",
    "인플레이션 감축법",
    "배터리 여권",
    "ai 기본법",
    "실증특례",
    "전기요금",
    "탄소규제",
    "탄소 규제",
    "공급망",
    "양산",
    "파운드리",
    "유리기판",
    "oled 라인",
    "송전망",
    "배전망",
    "ai반도체",
    "전고체 배터리",
    "대규모언어모델",
    "파운데이션 모델",
    "온디바이스 ai",
    "에너지밀도",
    "스마트그리드",
    "상용화",
    "신제품",
    "ai 서비스",
    "ai 검색",
    "충전 서비스",
    "서비스 로봇",
    "물류 로봇",
    "배송 로봇",
    "청소 로봇",
    "관수 로봇",
    "가사 로봇",
    "생활 로봇",
    "전기차 충전",
    "투자유치",
    "파트너십",
    "컨소시엄",
    "인수",
    "합병",
    "수주",
    "계약",
    "사업 확장",
    "노동조합",
    "단체교섭",
    "파업",
    "쟁의",
    "조건 없이 대화",
    "수출통제",
    "수출 통제",
    "공급망 동맹",
    "정상회담",
    "수출 규제",
    "수입 규제",
    "무역장벽",
    "글로벌 공급망",
})

_TIE_PRIORITY = {
    "금융/투자": 10,
    "노동/노사": 20,
    "국제/통상": 30,
    "정책/규제": 40,
    "생산/공급망": 50,
    "제품/서비스": 60,
    "기업동향": 70,
    "시장/산업": 80,
    "기술개발": 90,
}

_TECHNOLOGY_BIAS_OVERRIDE_LABELS = frozenset(
    set(_LEGACY_KEYWORDS_BY_LABEL) - {"기술개발"}
)

# 새 규칙은 이 목록에 development origin과 confirmed event_id를 함께 기록한다.
_DEVELOPMENT_RULE_TERMS: tuple[tuple[str, RuleTerm], ...] = ()


def _build_default_rule_set() -> RuleSet:
    unknown_development_labels = sorted({
        label
        for label, _ in _DEVELOPMENT_RULE_TERMS
        if label not in _CANDIDATE_LABELS or label == _OTHER_LABEL
    })
    if unknown_development_labels:
        raise ValueError(
            "development rules contain unknown or no-direct labels: "
            f"{unknown_development_labels}"
        )
    if any(
        term.origin != "development" or not term.evidence_event_ids
        for _, term in _DEVELOPMENT_RULE_TERMS
    ):
        raise ValueError(
            "new default rules require development origin and event evidence"
        )

    normalized_strong = {
        normalize_rule_text(keyword) for keyword in _STRONG_KEYWORDS
    }
    known_keywords = {
        normalize_rule_text(keyword)
        for keywords in _LEGACY_KEYWORDS_BY_LABEL.values()
        for keyword in keywords
    }
    known_keywords.update(
        normalize_rule_text(term.phrase)
        for _, term in _DEVELOPMENT_RULE_TERMS
    )
    unknown_strong = sorted(normalized_strong - known_keywords)
    if unknown_strong:
        raise ValueError(f"unknown strong rule keywords: {unknown_strong}")

    labels: list[LabelRulePolicy] = []
    for label in _CANDIDATE_LABELS:
        if label == _OTHER_LABEL:
            labels.append(LabelRulePolicy(
                label=label,
                terms=(),
                tie_priority=100,
                no_direct_rules=True,
            ))
            continue

        legacy_terms = tuple(
            RuleTerm(
                phrase=keyword,
                strength=(
                    RuleStrength.STRONG
                    if normalize_rule_text(keyword) in normalized_strong
                    else RuleStrength.WEAK
                ),
            )
            for keyword in _LEGACY_KEYWORDS_BY_LABEL[label]
        )
        development_terms = tuple(
            term
            for development_label, term in _DEVELOPMENT_RULE_TERMS
            if development_label == label
        )
        terms = legacy_terms + development_terms
        labels.append(LabelRulePolicy(
            label=label,
            terms=terms,
            tie_priority=_TIE_PRIORITY[label],
            allow_technology_bias_override=(
                label in _TECHNOLOGY_BIAS_OVERRIDE_LABELS
            ),
        ))

    return RuleSet(
        version="rules-v2-weighted-longest",
        matcher_version="nfkc-token-boundary-longest-v1",
        labels=tuple(labels),
        domain_terms=_DOMAIN_TERMS,
        unrelated_signal_groups=_UNRELATED_SIGNAL_GROUPS,
        decision=RuleDecisionPolicy(),
        other_label=_OTHER_LABEL,
        technology_label="기술개발",
    )


DEFAULT_RULE_SET = _build_default_rule_set()

# 기존 공개 이름도 통합 정책을 가리키며, 실제 기준은 DEFAULT_RULE_SET 하나이다.
CANDIDATE_LABELS = list(DEFAULT_RULE_SET.candidate_labels)
OTHER_LABEL = DEFAULT_RULE_SET.other_label
NO_DIRECT_RULE_LABELS = frozenset(
    item.label for item in DEFAULT_RULE_SET.labels if item.no_direct_rules
)
MODEL_ONLY_LABELS = NO_DIRECT_RULE_LABELS
RULE_KEYWORDS = {
    item.label: [term.phrase for term in item.terms]
    for item in DEFAULT_RULE_SET.direct_rule_labels
}
RULES = DEFAULT_RULE_SET


def validate_rule_configuration(
    rule_set: RuleSet = DEFAULT_RULE_SET,
) -> None:
    """통합된 라벨·키워드·가중치·보정 정책을 검증한다."""
    validate_rule_set(rule_set)
