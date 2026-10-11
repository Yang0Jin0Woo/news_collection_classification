from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re

from news_classifier.classifiers.confidence import is_ambiguous
from news_classifier.models import ModelPrediction, RuleDecision
from news_classifier.utils.text import clean_text
from news_classifier.rules.policy import (
    RuleSet,
    RuleTerm,
    normalize_rule_text,
    validate_rule_set,
)


_ASCII_WORD_CHARACTER = re.compile(r"[A-Za-z0-9]")
_KOREAN_CHARACTER = re.compile(r"[가-힣]")
EVENT_ANCHOR_POLICY = "title-description-first-two-sentences-v1-experimental"


@dataclass(frozen=True)
class LabelRuleScore:
    label: str
    weighted_score: int
    match_count: int
    strong_match_count: int
    matched_terms: tuple[str, ...]


@dataclass(frozen=True)
class RuleRanking:
    ranked: tuple[LabelRuleScore, ...]
    score_margin: int
    tied_labels: tuple[str, ...]

    @property
    def first(self) -> LabelRuleScore:
        return self.ranked[0]

    @property
    def second(self) -> LabelRuleScore:
        return self.ranked[1]

    def score_for(self, label: str) -> LabelRuleScore:
        for item in self.ranked:
            if item.label == label:
                return item
        raise KeyError(label)


class RuleScoringSession:
    """One-run ranking reuse for one engine and the exact article input only."""

    def __init__(self, engine: RuleEngine):
        self._engine = engine
        self._rule_set = engine.rule_set
        self._rankings: dict[tuple[str, str, str], RuleRanking] = {}

    @property
    def engine(self) -> RuleEngine:
        return self._engine

    def require_engine(self, engine: RuleEngine) -> None:
        if engine is not self._engine:
            raise ValueError("rule scoring session belongs to a different engine")
        if engine.rule_set is not self._rule_set:
            raise ValueError("rule scoring session rule set changed")

    def rank_rules(
        self, title: str, description: str = "", content: str = "",
    ) -> RuleRanking:
        self.require_engine(self._engine)
        key = (title, description, content)
        if key not in self._rankings:
            self._rankings[key] = self._engine.rank_rules(*key)
        return self._rankings[key]

    def _rank_normalized_input(
        self, engine: RuleEngine, title: str, description: str, content: str, text: str,
    ) -> RuleRanking:
        self.require_engine(engine)
        key = (title, description, content)
        if key not in self._rankings:
            self._rankings[key] = engine._rank_normalized_text(text)
        return self._rankings[key]


def _uses_ascii_boundaries(normalized_phrase: str) -> bool:
    return not bool(_KOREAN_CHARACTER.search(normalized_phrase))


def _start_is_allowed(
    text: str,
    match_start: int,
    *,
    ascii_boundaries: bool,
) -> bool:
    if match_start == 0:
        return True
    if not ascii_boundaries:
        return True
    return not bool(_ASCII_WORD_CHARACTER.fullmatch(text[match_start - 1]))


def _suffix_is_allowed(
    text: str,
    match_end: int,
    *,
    ascii_boundaries: bool,
) -> bool:
    if not ascii_boundaries:
        return True
    return (
        match_end >= len(text)
        or not bool(_ASCII_WORD_CHARACTER.fullmatch(text[match_end]))
    )


@lru_cache(maxsize=1024)
def _phrase_pattern(normalized_phrase: str) -> re.Pattern[str]:
    parts: list[str] = []
    previous_character = ""
    for character in normalized_phrase:
        if character.isspace():
            if not parts or parts[-1] != r"\s*":
                parts.append(r"\s*")
            previous_character = ""
            continue
        script_transition = (
            previous_character
            and (
                bool(_KOREAN_CHARACTER.fullmatch(previous_character))
                != bool(_KOREAN_CHARACTER.fullmatch(character))
            )
            and (
                previous_character.isalnum()
                and character.isalnum()
            )
        )
        if script_transition and (not parts or parts[-1] != r"\s*"):
            parts.append(r"\s*")
        parts.append(re.escape(character))
        previous_character = character
    return re.compile("".join(parts))


@dataclass(frozen=True)
class _PreparedPhrase:
    normalized: str
    pattern: re.Pattern[str]
    ascii_boundaries: bool

    def spans(self, text: str) -> tuple[tuple[int, int], ...]:
        if not self.normalized:
            return ()
        spans: list[tuple[int, int]] = []
        for match in self.pattern.finditer(text):
            start, end = match.span()
            if not _start_is_allowed(text, start, ascii_boundaries=self.ascii_boundaries):
                continue
            if not _suffix_is_allowed(text, end, ascii_boundaries=self.ascii_boundaries):
                continue
            spans.append((start, end))
        return tuple(spans)


def _prepare_phrase(phrase: str) -> _PreparedPhrase:
    normalized = normalize_rule_text(phrase)
    return _PreparedPhrase(
        normalized, _phrase_pattern(normalized), _uses_ascii_boundaries(normalized),
    )


def _phrase_spans(text: str, phrase: str) -> tuple[tuple[int, int], ...]:
    return _prepare_phrase(phrase).spans(text)


def _has_phrase(text: str, phrase: str) -> bool:
    return bool(_phrase_spans(text, phrase))


def _overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < occupied_end and end > occupied_start for occupied_start, occupied_end in occupied)


@dataclass(frozen=True)
class _PreparedTerm:
    label: str
    term: RuleTerm
    phrase: _PreparedPhrase


class RuleEngine:
    def __init__(self, rule_set: RuleSet, *, experimental_event_anchors: bool = False):
        validate_rule_set(rule_set)
        if type(experimental_event_anchors) is not bool:
            raise ValueError("experimental_event_anchors must be boolean")
        self.rule_set = rule_set
        # Evaluation-only prototype. Default CLI/service and saved profiles remain unchanged.
        self.experimental_event_anchors = experimental_event_anchors
        self._direct_labels = rule_set.direct_rule_labels
        self._tie_priority = {item.label: item.tie_priority for item in self._direct_labels}
        self._candidate_order = {
            label: index for index, label in enumerate(rule_set.candidate_labels)
        }
        self._label_policies = {item.label: item for item in rule_set.labels}
        # RuleSet is immutable: prepare matching and longest-first order once per engine.
        self._prepared_terms = tuple(sorted(
            (
                _PreparedTerm(policy.label, term, _prepare_phrase(term.phrase))
                for policy in self._direct_labels
                for term in policy.terms
                if not term.context_only
            ),
            key=lambda item: (
                -sum(character.isalnum() for character in item.phrase.normalized),
                -int(item.term.strength),
                self._tie_priority[item.label],
                item.phrase.normalized,
                item.label,
            ),
        ))
        self._domain_phrases = tuple(_prepare_phrase(term) for term in rule_set.domain_terms)
        self._unrelated_phrases = tuple(
            tuple(_prepare_phrase(term) for term in group)
            for group in rule_set.unrelated_signal_groups
        )
        self._development_phrases = {
            policy.label: {
                term.phrase
                for term in policy.terms
                if term.origin == "development"
            }
            for policy in rule_set.labels
        }

    def _has_event_anchor(self, title: str, description: str, content: str, label: str) -> bool:
        """제목/설명/본문 첫 두 문장에 같은 주제의 충분한 근거가 있는지 비교용 확인.

        위치만으로 주요 사건을 확정하는 기능은 아님. 독립 정답 평가 전 기본 적용 금지.
        """
        if not self.experimental_event_anchors:
            return True
        body = clean_text(content)
        sentences = re.split(r"(?<=[.!?。！？])\s+", body)
        lead = " ".join(sentences[:2])[:500]
        anchor = self.rank_rules(clean_text(title)[:250], clean_text(description)[:300], lead)
        policy = self.rule_set.decision
        return (
            anchor.first.label == label
            and anchor.first.weighted_score >= policy.min_rule_score
            and anchor.score_margin >= policy.min_rule_score_margin
        )

    def _development_conflict_labels(self, ranking: RuleRanking) -> tuple[str, ...]:
        """새 규칙이 실제 기여한 경우 서로 다른 강한 사건 근거의 충돌 확인."""
        strong_labels = tuple(
            score.label for score in ranking.ranked if score.strong_match_count
        )
        if len(strong_labels) < 2:
            return ()
        development_contributed = any(
            phrase in self._development_phrases[score.label]
            for score in ranking.ranked
            for phrase in score.matched_terms
        )
        return strong_labels if development_contributed else ()

    @staticmethod
    def _conflict_reason(labels: tuple[str, ...]) -> str:
        return (
            f"주제 근거 충돌: {', '.join(labels)}에 강한 근거 동시 확인, "
            "신규 규칙 자동 보정 보류"
        )

    def _score_labels(self, text: str) -> tuple[LabelRuleScore, ...]:
        occupied: list[tuple[int, int]] = []
        matched_terms: dict[str, list[str]] = {
            item.label: [] for item in self._direct_labels
        }
        weighted_scores = {
            item.label: 0 for item in self._direct_labels
        }
        strong_match_counts = {
            item.label: 0 for item in self._direct_labels
        }

        for prepared in self._prepared_terms:
            available_spans = [
                span
                for span in prepared.phrase.spans(text)
                if not _overlaps(span, occupied)
            ]
            if not available_spans:
                continue
            occupied.extend(available_spans)
            matched_terms[prepared.label].append(prepared.term.phrase)
            weighted_scores[prepared.label] += int(prepared.term.strength)
            if int(prepared.term.strength) > 1:
                strong_match_counts[prepared.label] += 1

        return tuple(
            LabelRuleScore(
                label=item.label,
                weighted_score=weighted_scores[item.label],
                match_count=len(matched_terms[item.label]),
                strong_match_count=strong_match_counts[item.label],
                matched_terms=tuple(matched_terms[item.label]),
            )
            for item in self._direct_labels
        )

    def rank_rules(
        self,
        title: str,
        description: str = "",
        content: str = "",
    ) -> RuleRanking:
        text = normalize_rule_text(f"{title} {description} {content}")
        return self._rank_normalized_text(text)

    def _rank_normalized_text(self, text: str) -> RuleRanking:
        scores = self._score_labels(text)
        ranked = tuple(sorted(
            scores,
            key=lambda item: (
                -item.weighted_score,
                self._tie_priority[item.label],
                item.label,
            ),
        ))
        if len(ranked) < 2:
            raise ValueError("at least two direct rule labels are required")

        first_score = ranked[0].weighted_score
        tied_labels = tuple(
            item.label for item in ranked if item.weighted_score == first_score
        )
        return RuleRanking(
            ranked=ranked,
            score_margin=first_score - ranked[1].weighted_score,
            tied_labels=tied_labels,
        )

    def calculate_scores(
        self,
        title: str,
        description: str = "",
        content: str = "",
    ) -> dict[str, int]:
        ranking = self.rank_rules(title, description, content)
        return {
            item.label: item.weighted_score
            for item in sorted(
                ranking.ranked,
                key=lambda score: self._candidate_order[score.label],
            )
        }

    def _unrelated_signal_count(self, text: str) -> int:
        has_domain_context = any(
            phrase.spans(text) for phrase in self._domain_phrases
        )
        if has_domain_context:
            return 0
        return sum(
            any(phrase.spans(text) for phrase in group)
            for group in self._unrelated_phrases
        )

    def rule_only_decision(
        self,
        title: str,
        description: str = "",
        content: str = "",
        *,
        scoring: RuleScoringSession | None = None,
    ) -> str:
        text = normalize_rule_text(f"{title} {description} {content}")
        unrelated_count = self._unrelated_signal_count(text)
        if unrelated_count >= 2:
            return self.rule_set.other_label

        ranking = self._ranking_for_input(title, description, content, text, scoring)
        if self._development_conflict_labels(ranking):
            return "검토필요"
        decision = self.rule_set.decision
        if (
            ranking.first.weighted_score >= decision.min_rule_score
            and ranking.score_margin >= decision.min_rule_score_margin
            and self._has_event_anchor(title, description, content, ranking.first.label)
        ):
            return ranking.first.label
        return "검토필요"

    @staticmethod
    def _rule_reason(prefix: str, ranking: RuleRanking) -> str:
        first = ranking.first
        terms = ", ".join(first.matched_terms[:3])
        return (
            f"{prefix}: {first.label} 규칙 점수 {first.weighted_score}, "
            f"근거 {first.match_count}개, 2위 대비 +{ranking.score_margin}"
            + (f" ({terms})" if terms else "")
        )

    def _review_reason(self, ranking: RuleRanking) -> str:
        """근거 없음, 총점 부족, 주제 간 차이 부족 순서로 검토 원인 구분."""
        first_score = ranking.first.weighted_score
        policy = self.rule_set.decision
        if first_score == 0:
            return "근거 없음: 일치하는 규칙 키워드 없음, 모델 판단 불확실"
        if first_score < policy.min_rule_score:
            return (
                f"규칙 점수 부족: 최고 {first_score}점, "
                f"최소 {policy.min_rule_score}점 필요, 모델 판단 불확실"
            )
        return (
            f"주제 간 점수 차이 부족: {first_score} 대 "
            f"{ranking.second.weighted_score}, 차이 {ranking.score_margin}점, "
            f"최소 {policy.min_rule_score_margin}점 필요, 모델 판단 불확실"
        )

    @staticmethod
    def _decision(
        *,
        final_label: str,
        rule_applied: bool,
        rule_reason: str,
        ranking: RuleRanking,
    ) -> RuleDecision:
        return RuleDecision(
            final_label=final_label,
            rule_applied=rule_applied,
            rule_reason=rule_reason,
            rule_best_label=ranking.first.label,
            rule_match_count=ranking.first.match_count,
        )

    def decide(
        self,
        title: str,
        description: str,
        content: str,
        prediction: ModelPrediction,
        *,
        scoring: RuleScoringSession | None = None,
    ) -> RuleDecision:
        text = normalize_rule_text(f"{title} {description} {content}")
        unrelated_count = self._unrelated_signal_count(text)
        if unrelated_count >= 2:
            return RuleDecision(
                final_label=self.rule_set.other_label,
                rule_applied=True,
                rule_reason=(
                    f"비관련 문맥 {unrelated_count}개 매칭, "
                    "기술 도메인 근거 없음"
                ),
                rule_best_label=self.rule_set.other_label,
                rule_match_count=unrelated_count,
            )

        ranking = self._ranking_for_input(title, description, content, text, scoring)
        conflict_labels = self._development_conflict_labels(ranking)
        decision = self.rule_set.decision
        first_policy = self._label_policies[ranking.first.label]
        model_confident = (
            prediction.score >= decision.model_keep_score
            and prediction.margin >= decision.model_keep_margin
        )
        ambiguous = is_ambiguous(
            prediction.score,
            prediction.margin,
            min_score=decision.ambiguity_score,
            min_margin=decision.ambiguity_margin,
        )
        clear_rule = (
            ranking.first.weighted_score >= decision.min_rule_score
            and ranking.score_margin >= decision.min_rule_score_margin
        )
        strong_override = (
            ranking.first.weighted_score >= decision.strong_override_score
            and ranking.score_margin >= decision.strong_override_margin
            and ranking.first.strong_match_count >= 1
            and not conflict_labels
            and self._has_event_anchor(title, description, content, ranking.first.label)
        )

        if model_confident:
            if (
                prediction.label == self.rule_set.technology_label
                and ranking.first.label != prediction.label
                and first_policy.allow_technology_bias_override
                and strong_override
            ):
                return self._decision(
                    final_label=ranking.first.label,
                    rule_applied=True,
                    rule_reason=self._rule_reason(
                        "기술개발 편향 보정",
                        ranking,
                    ),
                    ranking=ranking,
                )
            return self._decision(
                final_label=prediction.label,
                rule_applied=False,
                rule_reason="",
                ranking=ranking,
            )

        if conflict_labels:
            return self._decision(
                final_label="검토필요",
                rule_applied=False,
                rule_reason=self._conflict_reason(conflict_labels),
                ranking=ranking,
            )

        if clear_rule:
            if not self._has_event_anchor(title, description, content, ranking.first.label):
                return self._decision(
                    final_label="검토필요", rule_applied=False, ranking=ranking,
                    rule_reason=(f"주요 사건 근거 부족: {ranking.first.label}의 규칙 근거가 "
                                 "제목, 설명, 본문 첫 두 문장에서 충분하지 않아 보정 보류"),
                )
            return self._decision(
                final_label=ranking.first.label,
                rule_applied=True,
                rule_reason=self._rule_reason("규칙 보정", ranking),
                ranking=ranking,
            )

        if (
            prediction.score < decision.review_model_score
            or ambiguous
        ):
            return self._decision(
                final_label="검토필요",
                rule_applied=False,
                rule_reason=self._review_reason(ranking),
                ranking=ranking,
            )

        return self._decision(
            final_label=prediction.label,
            rule_applied=False,
            rule_reason="",
            ranking=ranking,
        )

    def _ranking_for_input(
        self, title: str, description: str, content: str, text: str,
        scoring: RuleScoringSession | None,
    ) -> RuleRanking:
        if scoring is None:
            return self._rank_normalized_text(text)
        return scoring._rank_normalized_input(self, title, description, content, text)
