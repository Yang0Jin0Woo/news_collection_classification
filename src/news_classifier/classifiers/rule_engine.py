from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re

from news_classifier.classifiers.confidence import is_ambiguous
from news_classifier.models import ModelPrediction, RuleDecision
from news_classifier.rules.policy import (
    RuleSet,
    RuleTerm,
    normalize_rule_text,
    validate_rule_set,
)


_ASCII_WORD_CHARACTER = re.compile(r"[A-Za-z0-9]")
_KOREAN_CHARACTER = re.compile(r"[가-힣]")


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


def _phrase_spans(text: str, phrase: str) -> tuple[tuple[int, int], ...]:
    normalized_phrase = normalize_rule_text(phrase)
    if not normalized_phrase:
        return ()

    spans: list[tuple[int, int]] = []
    ascii_boundaries = _uses_ascii_boundaries(normalized_phrase)
    for match in _phrase_pattern(normalized_phrase).finditer(text):
        start, end = match.span()
        if not _start_is_allowed(
            text,
            start,
            ascii_boundaries=ascii_boundaries,
        ):
            continue
        if not _suffix_is_allowed(
            text,
            end,
            ascii_boundaries=ascii_boundaries,
        ):
            continue
        spans.append((start, end))
    return tuple(spans)


def _has_phrase(text: str, phrase: str) -> bool:
    return bool(_phrase_spans(text, phrase))


def _overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < occupied_end and end > occupied_start for occupied_start, occupied_end in occupied)


def _term_sort_key(term: RuleTerm) -> tuple[int, int, str]:
    normalized = normalize_rule_text(term.phrase)
    semantic_length = sum(character.isalnum() for character in normalized)
    return (-semantic_length, -int(term.strength), normalized)


class RuleEngine:
    def __init__(self, rule_set: RuleSet):
        validate_rule_set(rule_set)
        self.rule_set = rule_set

    def _score_labels(self, text: str) -> tuple[LabelRuleScore, ...]:
        priority = {
            item.label: item.tie_priority
            for item in self.rule_set.direct_rule_labels
        }
        occupied: list[tuple[int, int]] = []
        matched_terms: dict[str, list[str]] = {
            item.label: [] for item in self.rule_set.direct_rule_labels
        }
        weighted_scores = {
            item.label: 0 for item in self.rule_set.direct_rule_labels
        }
        strong_match_counts = {
            item.label: 0 for item in self.rule_set.direct_rule_labels
        }
        terms = [
            (label_policy, term)
            for label_policy in self.rule_set.direct_rule_labels
            for term in label_policy.terms
            if not term.context_only
        ]

        def labeled_term_sort_key(item):
            label_policy, term = item
            length, strength, normalized = _term_sort_key(term)
            return (
                length,
                strength,
                priority[label_policy.label],
                normalized,
                label_policy.label,
            )

        terms.sort(key=labeled_term_sort_key)

        for label_policy, term in terms:
            available_spans = [
                span
                for span in _phrase_spans(text, term.phrase)
                if not _overlaps(span, occupied)
            ]
            if not available_spans:
                continue
            occupied.extend(available_spans)
            matched_terms[label_policy.label].append(term.phrase)
            weighted_scores[label_policy.label] += int(term.strength)
            if int(term.strength) > 1:
                strong_match_counts[label_policy.label] += 1

        return tuple(
            LabelRuleScore(
                label=item.label,
                weighted_score=weighted_scores[item.label],
                match_count=len(matched_terms[item.label]),
                strong_match_count=strong_match_counts[item.label],
                matched_terms=tuple(matched_terms[item.label]),
            )
            for item in self.rule_set.direct_rule_labels
        )

    def rank_rules(
        self,
        title: str,
        description: str = "",
        content: str = "",
    ) -> RuleRanking:
        text = normalize_rule_text(f"{title} {description} {content}")
        priority = {
            item.label: item.tie_priority
            for item in self.rule_set.direct_rule_labels
        }
        scores = self._score_labels(text)
        ranked = tuple(sorted(
            scores,
            key=lambda item: (
                -item.weighted_score,
                priority[item.label],
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
                key=lambda score: self.rule_set.candidate_labels.index(score.label),
            )
        }

    def _unrelated_signal_count(
        self,
        title: str,
        description: str,
        content: str,
    ) -> int:
        text = normalize_rule_text(f"{title} {description} {content}")
        has_domain_context = any(
            _has_phrase(text, term) for term in self.rule_set.domain_terms
        )
        if has_domain_context:
            return 0
        return sum(
            any(_has_phrase(text, term) for term in group)
            for group in self.rule_set.unrelated_signal_groups
        )

    def rule_only_decision(
        self,
        title: str,
        description: str = "",
        content: str = "",
    ) -> str:
        unrelated_count = self._unrelated_signal_count(
            title,
            description,
            content,
        )
        if unrelated_count >= 2:
            return self.rule_set.other_label

        ranking = self.rank_rules(title, description, content)
        decision = self.rule_set.decision
        if (
            ranking.first.weighted_score >= decision.min_rule_score
            and ranking.score_margin >= decision.min_rule_score_margin
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
    ) -> RuleDecision:
        unrelated_count = self._unrelated_signal_count(
            title,
            description,
            content,
        )
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

        ranking = self.rank_rules(title, description, content)
        decision = self.rule_set.decision
        first_policy = self.rule_set.label_policy(ranking.first.label)
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

        if clear_rule:
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
