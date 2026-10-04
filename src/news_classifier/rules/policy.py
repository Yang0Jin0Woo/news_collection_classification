from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import IntEnum
import hashlib
import json
from collections.abc import Iterable, Mapping
import unicodedata

from news_classifier.utils.text import clean_text


class RuleStrength(IntEnum):
    WEAK = 1
    STRONG = 2


@dataclass(frozen=True)
class RuleTerm:
    phrase: str
    strength: RuleStrength = RuleStrength.WEAK
    origin: str = "legacy"
    evidence_event_ids: tuple[str, ...] = ()
    context_only: bool = False


@dataclass(frozen=True)
class LabelRulePolicy:
    label: str
    terms: tuple[RuleTerm, ...]
    tie_priority: int
    allow_technology_bias_override: bool = False
    no_direct_rules: bool = False


@dataclass(frozen=True)
class RuleDecisionPolicy:
    model_keep_score: float = 0.55
    model_keep_margin: float = 0.08
    min_rule_score: int = 2
    min_rule_score_margin: int = 1
    strong_override_score: int = 4
    strong_override_margin: int = 2
    review_model_score: float = 0.40
    ambiguity_score: float = 0.50
    ambiguity_margin: float = 0.05


@dataclass(frozen=True)
class RuleSet:
    version: str
    matcher_version: str
    labels: tuple[LabelRulePolicy, ...]
    domain_terms: tuple[str, ...]
    unrelated_signal_groups: tuple[tuple[str, ...], ...]
    decision: RuleDecisionPolicy
    other_label: str
    technology_label: str
    change_policy: str = "confirmed_development_errors_only"
    minimum_development_evidence_events: int = 2

    @property
    def candidate_labels(self) -> tuple[str, ...]:
        return tuple(item.label for item in self.labels)

    @property
    def direct_rule_labels(self) -> tuple[LabelRulePolicy, ...]:
        return tuple(item for item in self.labels if not item.no_direct_rules)

    def label_policy(self, label: str) -> LabelRulePolicy:
        for item in self.labels:
            if item.label == label:
                return item
        raise KeyError(label)


def normalize_rule_text(value: str) -> str:
    return unicodedata.normalize("NFKC", clean_text(value)).casefold()


def rule_set_fingerprint(rule_set: RuleSet) -> str:
    labels = []
    for label_policy in rule_set.labels:
        terms = sorted(
            (
                {
                    "phrase": normalize_rule_text(term.phrase),
                    "strength": int(term.strength),
                    "origin": term.origin,
                    "context_only": term.context_only,
                    "evidence_event_ids": sorted(
                        event_id.strip().casefold()
                        for event_id in term.evidence_event_ids
                    ),
                }
                for term in label_policy.terms
            ),
            key=lambda item: (
                item["phrase"],
                item["strength"],
                item["origin"],
                item["evidence_event_ids"],
            ),
        )
        labels.append({
            "label": label_policy.label,
            "tie_priority": label_policy.tie_priority,
            "allow_technology_bias_override": (
                label_policy.allow_technology_bias_override
            ),
            "no_direct_rules": label_policy.no_direct_rules,
            "terms": terms,
        })

    unrelated_groups = sorted(
        tuple(sorted(normalize_rule_text(term) for term in group))
        for group in rule_set.unrelated_signal_groups
    )
    payload = {
        "version": rule_set.version,
        "matcher_version": rule_set.matcher_version,
        "labels": labels,
        "domain_terms": sorted(
            normalize_rule_text(term) for term in rule_set.domain_terms
        ),
        "unrelated_signal_groups": unrelated_groups,
        "decision": asdict(rule_set.decision),
        "other_label": rule_set.other_label,
        "technology_label": rule_set.technology_label,
        "change_policy": rule_set.change_policy,
        "minimum_development_evidence_events": (
            rule_set.minimum_development_evidence_events
        ),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_rule_set(rule_set: RuleSet) -> None:
    labels = rule_set.candidate_labels
    label_set = set(labels)
    if len(labels) != len(label_set):
        raise ValueError("candidate labels contain duplicates")
    if rule_set.other_label not in label_set:
        raise ValueError("other label is not a candidate label")
    if rule_set.technology_label not in label_set:
        raise ValueError("technology label is not a candidate label")
    if not rule_set.version.strip() or not rule_set.matcher_version.strip():
        raise ValueError("rule and matcher versions must not be empty")
    if rule_set.change_policy != "confirmed_development_errors_only":
        raise ValueError("new rules must be based on confirmed development errors")
    if (
        not isinstance(rule_set.minimum_development_evidence_events, int)
        or rule_set.minimum_development_evidence_events < 1
    ):
        raise ValueError("minimum development evidence must be a positive integer")

    priorities = [item.tie_priority for item in rule_set.direct_rule_labels]
    if len(priorities) < 2:
        raise ValueError("at least two direct rule labels are required")
    if any(not isinstance(priority, int) or priority < 0 for priority in priorities):
        raise ValueError("rule tie priorities must be non-negative integers")
    if len(priorities) != len(set(priorities)):
        raise ValueError("rule tie priorities contain duplicates")

    keyword_owners: dict[str, str] = {}
    alias_owners: dict[str, str] = {}
    for label_policy in rule_set.labels:
        if label_policy.no_direct_rules and label_policy.terms:
            raise ValueError(
                f"label with no direct rules contains terms: {label_policy.label}"
            )
        if not label_policy.no_direct_rules and not label_policy.terms:
            raise ValueError(f"direct rule label has no terms: {label_policy.label}")

        seen_for_label: set[str] = set()
        alias_strengths: dict[str, RuleStrength] = {}
        for term in label_policy.terms:
            normalized = normalize_rule_text(term.phrase)
            if not normalized:
                raise ValueError(f"empty rule keyword: label={label_policy.label}")
            if normalized in seen_for_label:
                raise ValueError(
                    "duplicate rule keyword: "
                    f"label={label_policy.label}, keyword={term.phrase}"
                )
            seen_for_label.add(normalized)

            alias_key = normalized.replace(" ", "")
            previous_strength = alias_strengths.get(alias_key)
            if (
                previous_strength is not None
                and previous_strength != term.strength
            ):
                raise ValueError(
                    "rule aliases must use the same strength: "
                    f"label={label_policy.label}, keyword={term.phrase}"
                )
            alias_strengths[alias_key] = term.strength
            previous_alias_owner = alias_owners.get(alias_key)
            if (
                previous_alias_owner is not None
                and previous_alias_owner != label_policy.label
            ):
                raise ValueError(
                    "rule aliases belong to multiple labels: "
                    f"keyword={term.phrase}, "
                    f"labels={previous_alias_owner},{label_policy.label}"
                )
            alias_owners[alias_key] = label_policy.label

            previous_owner = keyword_owners.get(normalized)
            if previous_owner is not None:
                raise ValueError(
                    "rule keyword belongs to multiple labels: "
                    f"keyword={term.phrase}, "
                    f"labels={previous_owner},{label_policy.label}"
                )
            keyword_owners[normalized] = label_policy.label

            if not isinstance(term.strength, RuleStrength):
                raise ValueError("rule strength must be WEAK or STRONG")
            if not isinstance(term.context_only, bool):
                raise ValueError("context_only must be boolean")
            if term.context_only and term.origin == "development":
                raise ValueError("development evidence rules must contribute to scoring")
            if term.origin not in {"legacy", "development"}:
                raise ValueError(f"unsupported rule origin: {term.origin}")
            if term.origin == "development" and not term.evidence_event_ids:
                raise ValueError(
                    "development rule requires confirmed event evidence: "
                    f"label={label_policy.label}, keyword={term.phrase}"
                )
            if any(not event_id.strip() for event_id in term.evidence_event_ids):
                raise ValueError("rule evidence event ids must not be empty")
            normalized_event_ids = {event_id.strip().casefold() for event_id in term.evidence_event_ids}
            if len(term.evidence_event_ids) != len(normalized_event_ids):
                raise ValueError("rule evidence event ids contain duplicates")
            if (
                term.origin == "development"
                and len(term.evidence_event_ids)
                < rule_set.minimum_development_evidence_events
            ):
                raise ValueError(
                    "development rule has insufficient independent event evidence"
                )

    decision = rule_set.decision
    probability_values = (
        decision.model_keep_score,
        decision.model_keep_margin,
        decision.review_model_score,
        decision.ambiguity_score,
        decision.ambiguity_margin,
    )
    if any(not 0.0 <= value <= 1.0 for value in probability_values):
        raise ValueError("model and margin thresholds must be between 0 and 1")
    integer_values = (
        decision.min_rule_score,
        decision.min_rule_score_margin,
        decision.strong_override_score,
        decision.strong_override_margin,
    )
    if any(not isinstance(value, int) or value < 0 for value in integer_values):
        raise ValueError("rule score thresholds must be non-negative integers")
    if decision.min_rule_score <= 0:
        raise ValueError("minimum rule score must be positive")
    if decision.strong_override_score < decision.min_rule_score:
        raise ValueError("strong override score must not be lower than rule score")
    if decision.strong_override_margin < decision.min_rule_score_margin:
        raise ValueError("strong override margin must not be lower than rule margin")
    if decision.review_model_score > decision.ambiguity_score:
        raise ValueError("review score must not exceed ambiguity score")
    if decision.ambiguity_score > decision.model_keep_score:
        raise ValueError("ambiguity score must not exceed model keep score")
    if decision.ambiguity_margin > decision.model_keep_margin:
        raise ValueError("ambiguity margin must not exceed model keep margin")

    if not rule_set.domain_terms:
        raise ValueError("domain terms must not be empty")
    if any(not normalize_rule_text(term) for term in rule_set.domain_terms):
        raise ValueError("domain terms must not contain empty values")
    if any(not group for group in rule_set.unrelated_signal_groups):
        raise ValueError("unrelated signal groups must not be empty")
    if any(
        not normalize_rule_text(term)
        for group in rule_set.unrelated_signal_groups
        for term in group
    ):
        raise ValueError("unrelated signal terms must not be empty")


def without_development_rules(rule_set: RuleSet) -> RuleSet:
    """Keep the same decision gates while comparing against pre-addition rules."""
    labels = []
    for label in rule_set.labels:
        terms = tuple(term for term in label.terms if term.origin != "development")
        labels.append(replace(label, terms=terms, no_direct_rules=not terms))
    return replace(rule_set, labels=tuple(labels))


def validate_development_rule_evidence(
    rule_set: RuleSet,
    rows: Iterable[Mapping[str, str]],
) -> None:
    # Import at validation time to keep the matcher dependent on the policy,
    # rather than creating a module-level circular import.
    from news_classifier.classifiers.rule_engine import RuleEngine

    rule_engine = RuleEngine(rule_set)
    evidence_rows: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        event_id = (row.get("event_id") or "").strip().casefold()
        if event_id:
            evidence_rows.setdefault(event_id, []).append(row)

    for label_policy in rule_set.labels:
        for term in label_policy.terms:
            if term.origin != "development":
                continue
            for raw_event_id in term.evidence_event_ids:
                event_id = raw_event_id.strip().casefold()
                matches = evidence_rows.get(event_id, [])
                if not matches:
                    raise ValueError(
                        "rule evidence event does not exist in the dataset: "
                        f"{raw_event_id}"
                    )
                if any(
                    (row.get("split") or "").strip() != "development"
                    or (row.get("review_status") or "").strip() != "confirmed"
                    or (row.get("gold_label") or "").strip() != label_policy.label
                    for row in matches
                ):
                    raise ValueError(
                        "rule evidence must be confirmed development data "
                        f"with gold label {label_policy.label}: {raw_event_id}"
                    )
                if any(not (row.get("reviewed_by") or "").strip() for row in matches):
                    raise ValueError(
                        "rule evidence requires a named human reviewer: "
                        f"{raw_event_id}"
                    )
                if any((row.get("suggested_label") or "").strip() for row in matches):
                    raise ValueError(
                        "rule evidence must not contain suggested labels: "
                        f"{raw_event_id}"
                    )
                if not any(
                    rule_term_matches_row(rule_engine, label_policy.label, term, row)
                    for row in matches
                ):
                    raise ValueError(
                        "rule evidence phrase does not score in article context: "
                        f"keyword={term.phrase}, event={raw_event_id}"
                    )


def rule_term_matches_row(
    rule_engine,
    label: str,
    term: RuleTerm,
    row: Mapping[str, str],
) -> bool:
    """기사의 실제 점수 근거 확인. 검색어 및 출처만의 일치는 제외한다."""
    from news_classifier.utils.text import article_description

    title = row.get("title") or ""
    description = article_description(
        title,
        row.get("description") or "",
        row.get("source") or "",
    )
    ranking = rule_engine.rank_rules(title, description, row.get("content") or "")
    normalized_phrase = normalize_rule_text(term.phrase)
    return any(
        normalize_rule_text(phrase) == normalized_phrase
        for phrase in ranking.score_for(label).matched_terms
    )


def validate_development_rule_errors(
    rule_set: RuleSet,
    rows: list[Mapping[str, str]],
    baseline_labels: list[str],
) -> None:
    if len(rows) != len(baseline_labels):
        raise ValueError("rule evidence rows and baseline labels must align")
    validate_development_rule_evidence(rule_set, rows)
    from news_classifier.classifiers.rule_engine import RuleEngine

    rule_engine = RuleEngine(rule_set)
    valid_baseline_labels = set(rule_set.candidate_labels) | {"검토필요"}

    for label_policy in rule_set.labels:
        for term in label_policy.terms:
            if term.origin != "development":
                continue
            for raw_event_id in term.evidence_event_ids:
                event_id = raw_event_id.strip().casefold()
                has_supporting_error = any(
                    (row.get("event_id") or "").strip().casefold() == event_id
                    and predicted_label in valid_baseline_labels
                    and predicted_label != (row.get("gold_label") or "").strip()
                    and rule_term_matches_row(rule_engine, label_policy.label, term, row)
                    for row, predicted_label in zip(rows, baseline_labels, strict=True)
                )
                if not has_supporting_error:
                    raise ValueError(
                        "rule evidence is not a confirmed baseline error "
                        "on the same phrase-supporting article: "
                        f"{raw_event_id}"
                    )
