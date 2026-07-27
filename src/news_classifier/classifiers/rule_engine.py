from __future__ import annotations

from dataclasses import dataclass

from news_classifier.classifiers.confidence import is_ambiguous
from news_classifier.models import ModelPrediction, RuleDecision
from news_classifier.rules.default_rules import OTHER_LABEL
from news_classifier.utils.text import clean_text


DOMAIN_TERMS = (
    "반도체", "배터리", "이차전지", "2차전지", "ai", "인공지능",
    "디스플레이", "oled", "로봇", "전력", "에너지", "원전", "smr", "ess",
)

CONTEXT_SIGNALS: dict[str, tuple[str, ...]] = {
    "금융/투자": (
        "주가", "증권", "투자", "목표가", "상향", "하향", "상장", "공모",
        "대장주", "따따블", "폭등", "급등", "강세", "실적", "영업이익",
    ),
    "시장/산업": (
        "시장", "산업", "수요", "전망", "점유율", "가격", "경쟁", "성장",
        "업황", "수급", "생태계", "전력난",
    ),
    "정책/규제": (
        "정부", "정책", "규제", "법안", "지원", "보조금", "인증", "표준",
        "요금", "제도", "관세", "수출통제", "가이드라인",
    ),
    "생산/공급망": (
        "공급망", "생산", "양산", "공장", "라인", "수율", "소재", "광물",
        "부품", "조달", "납품", "증설", "설비", "제조", "송전망", "변압기",
    ),
    "제품/서비스": (
        "출시", "공개", "서비스", "개시", "도입", "운영", "탑재", "적용",
        "앱", "api", "솔루션", "제품", "신제품", "패키지",
    ),
    "기업동향": (
        "협력", "제휴", "인수", "합병", "수주", "계약", "사업", "진출",
        "확장", "파트너십", "대표", "조직", "채용", "경쟁",
    ),
    "노동/노사": (
        "노조", "노사", "임금", "파업", "교섭", "고용", "근로자", "직원",
    ),
    "국제/통상": (
        "미국", "중국", "일본", "eu", "수출", "수입", "통상", "무역",
        "제재", "관세", "협상", "해외", "글로벌",
    ),
}

UNRELATED_SIGNAL_GROUPS = (
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


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


@dataclass(frozen=True)
class RuleEngineConfig:
    # 모델 결과를 그대로 신뢰할 최소 점수
    base_rule_override_threshold: float = 0.55

    # top1-top2 점수 차이를 판단하는 최소 차이
    min_margin_threshold: float = 0.08

    # 규칙 보정을 적용하기 위한 최소 키워드 매칭 수
    min_rule_match_count: int = 2

    # 모델이 강하게 예측해도 규칙 근거가 이 수 이상이면 규칙 보정을 허용
    strong_rule_override_match_count: int = 2

    # 검토필요로 분류할 낮은 모델 점수 기준
    review_needed_score_threshold: float = 0.40

    # 실제 개발 데이터로 보정 가능한 애매한 모델 예측 기준
    ambiguity_score_threshold: float = 0.50
    ambiguity_margin_threshold: float = 0.05

    # 낮은 신뢰도일 때 단일 키워드만으로도 보정할 수 있는 선명한 카테고리
    single_match_override_labels: tuple[str, ...] = (
        "금융/투자",
        "노동/노사",
        "국제/통상",
        "정책/규제",
    )

    # 기술개발로 쏠리기 쉬운 도메인에서, 명확한 비기술 신호가 있으면 보정할 라벨
    strong_rule_override_labels: tuple[str, ...] = (
        "금융/투자",
        "제품/서비스",
        "기업동향",
        "생산/공급망",
        "시장/산업",
        "국제/통상",
        "정책/규제",
        "노동/노사",
    )


class RuleEngine:
    def __init__(self, rules: dict[str, list[str]], config: RuleEngineConfig | None = None):
        self.rules = rules
        self.config = config or RuleEngineConfig()

    # 제목, 설명, 본문에서 카테고리별 키워드 매칭 점수 계산
    def calculate_scores(self, title: str, description: str = "", content: str = "") -> dict[str, int]:
        text = f"{clean_text(title)} {clean_text(description)} {clean_text(content)}".lower()
        has_domain_context = _contains_any(text, DOMAIN_TERMS)

        scores: dict[str, int] = {}
        for label, keywords in self.rules.items():
            scores[label] = sum(1 for keyword in keywords if keyword.lower() in text)
            if has_domain_context and _contains_any(text, CONTEXT_SIGNALS.get(label, ())):
                scores[label] += 1

        return scores

    # 모델 예측 결과와 규칙 점수를 함께 보고 최종 카테고리 결정
    def decide(
        self,
        title: str,
        description: str,
        content: str,
        prediction: ModelPrediction,
    ) -> RuleDecision:
        normalized_text = (
            f"{clean_text(title)} {clean_text(description)} "
            f"{clean_text(content)}"
        ).lower()
        domain_matches = sum(
            term in normalized_text for term in DOMAIN_TERMS
        )
        unrelated_matches = sum(
            _contains_any(normalized_text, group)
            for group in UNRELATED_SIGNAL_GROUPS
        )
        if domain_matches == 0 and unrelated_matches >= 2:
            return RuleDecision(
                final_label=OTHER_LABEL,
                rule_applied=True,
                rule_reason=(
                    f"비관련 문맥 {unrelated_matches}개 매칭, "
                    "기술 도메인 근거 없음"
                ),
                rule_best_label=OTHER_LABEL,
                rule_match_count=unrelated_matches,
            )

        label_scores = self.calculate_scores(title, description, content)

        # 가장 많이 매칭된 규칙 카테고리 선택. 동점이면 금융/노사/통상/정책처럼 신호가 선명한 라벨을 우선한다.
        best_rule_label = (
            max(
                label_scores,
                key=lambda label: (
                    label_scores[label],
                    label in self.config.single_match_override_labels,
                ),
            )
            if label_scores
            else prediction.label
        )
        best_rule_score = label_scores.get(best_rule_label, 0)
        ambiguous = is_ambiguous(
            prediction.score,
            prediction.margin,
            min_score=self.config.ambiguity_score_threshold,
            min_margin=self.config.ambiguity_margin_threshold,
        )

        # 로봇/AI처럼 모델이 기술개발로 과하게 쏠릴 때, 명확한 비기술 규칙 근거가 있으면 보정
        if (
            prediction.label == "기술개발"
            and best_rule_label != prediction.label
            and best_rule_label in self.config.strong_rule_override_labels
            and best_rule_score >= self.config.strong_rule_override_match_count
        ):
            return RuleDecision(
                final_label=best_rule_label,
                rule_applied=True,
                rule_reason=f"기술개발 편향 보정: {best_rule_label} 키워드 {best_rule_score}개 매칭",
                rule_best_label=best_rule_label,
                rule_match_count=best_rule_score,
            )

        # 모델 점수와 차이가 충분하면 모델 결과 유지
        if (
            prediction.score >= self.config.base_rule_override_threshold
            and prediction.margin >= self.config.min_margin_threshold
        ):
            return RuleDecision(
                final_label=prediction.label,
                rule_applied=False,
                rule_reason="",
                rule_best_label=best_rule_label,
                rule_match_count=best_rule_score,
            )

        # 규칙 키워드 근거가 충분하면 규칙 기반 카테고리로 보정
        if best_rule_score >= self.config.min_rule_match_count:
            return RuleDecision(
                final_label=best_rule_label,
                rule_applied=True,
                rule_reason=f"{best_rule_label} 키워드 {best_rule_score}개 매칭",
                rule_best_label=best_rule_label,
                rule_match_count=best_rule_score,
            )

        # 낮은 신뢰도 기사에서 금융/노사/통상/정책처럼 신호가 분명한 키워드는 더 적극 반영
        if (
            ambiguous
            and best_rule_label in self.config.single_match_override_labels
            and best_rule_score >= 1
        ):
            return RuleDecision(
                final_label=best_rule_label,
                rule_applied=True,
                rule_reason=f"낮은 신뢰도에서 {best_rule_label} 키워드 {best_rule_score}개 매칭",
                rule_best_label=best_rule_label,
                rule_match_count=best_rule_score,
            )

        # 모델 점수도 낮고 규칙 근거도 없으면 검토필요 처리
        if (
            best_rule_score == 0
            and (
                prediction.score < self.config.review_needed_score_threshold
                or ambiguous
            )
        ):
            return RuleDecision(
                final_label="검토필요",
                rule_applied=False,
                rule_reason="모델 점수 낮고 규칙 근거 없음",
                rule_best_label=best_rule_label,
                rule_match_count=best_rule_score,
            )

        # 그 외에는 모델 예측 결과 유지
        return RuleDecision(
            final_label=prediction.label,
            rule_applied=False,
            rule_reason="",
            rule_best_label=best_rule_label,
            rule_match_count=best_rule_score,
        )
