"""Evidence-backed final decisions; confidence display calibration is separate.

Only the review score/margin gates may change. The old review-score floor
tracks the selected score gate so it cannot silently prevent a validated
relaxation. Profiles require event-disjoint, unseen-keyword evaluation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path

from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.confidence_calibration import wilson_lower_bound
from news_classifier.evaluation_dataset import dataset_sha256, validate_unseen_keywords
from news_classifier.models import ModelPrediction, NewsItem
from news_classifier.rules.policy import (
    RuleDecisionPolicy, RuleSet, rule_set_fingerprint, validate_rule_set,
)
from news_classifier.utils.text import article_description, normalize_key


INPUT_MODES = frozenset({"without_body", "with_body"})


@dataclass(frozen=True)
class LoadedDecisionProfile:
    policy: RuleDecisionPolicy
    input_mode: str | None


@dataclass(frozen=True)
class DecisionEvidence:
    total: int
    support: int
    correct: int
    coverage: float
    accuracy: float
    independent_support: int
    independent_correct: int
    accuracy_lower_bound: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DecisionPolicyEvaluation:
    baseline: DecisionEvidence
    selected: DecisionEvidence
    newly_decided: DecisionEvidence
    previous_decisions_changed: int
    harmful_changes: int
    passed: bool
    failures: tuple[str, ...]
    per_label: dict[str, dict] = field(default_factory=dict)
    per_keyword: dict[str, dict] = field(default_factory=dict)
    input_mode: str = "without_body"

    def to_dict(self) -> dict:
        values = asdict(self)
        values["failures"] = list(self.failures)
        return values


@dataclass(frozen=True)
class DecisionCalibrationResult:
    policy: RuleDecisionPolicy
    evidence: DecisionPolicyEvaluation
    minimum_support: int
    accuracy_lower_bound: float
    candidates_tested: int


def _validate_targets(minimum_support: int, accuracy_lower_bound: float) -> None:
    if type(minimum_support) is not int or minimum_support < 2:
        raise ValueError("minimum support must be an integer of at least 2")
    if type(accuracy_lower_bound) not in (int, float) or not 0.5 < accuracy_lower_bound < 1:
        raise ValueError("accuracy lower bound must be between 0.5 and 1")


def _validate_samples(items, predictions, gold_labels, rule_set, event_ids, input_mode) -> None:
    validate_rule_set(rule_set)
    if not items or not (len(items) == len(predictions) == len(gold_labels) == len(event_ids)):
        raise ValueError("decision calibration cases and predictions must align")
    if set(gold_labels) != set(rule_set.candidate_labels):
        raise ValueError("decision calibration must cover all candidate labels")
    if not isinstance(input_mode, str) or input_mode not in INPUT_MODES:
        raise ValueError("invalid decision calibration input mode")
    if (input_mode == "without_body" and any(item.content.strip() for item in items)
            or input_mode == "with_body" and any(not item.content.strip() for item in items)):
        raise ValueError("decision calibration items do not match the declared body input mode")
    keywords = {normalize_key(item.keyword) for item in items}
    if "" in keywords or len(keywords) < 2:
        raise ValueError("decision calibration requires multiple non-empty keywords")
    gold_by_event = {}
    for prediction, gold, event_id in zip(predictions, gold_labels, event_ids, strict=True):
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("decision calibration requires independent event ids")
        key = event_id.strip().casefold()
        if key in gold_by_event and gold_by_event[key] != gold:
            raise ValueError("the same event has conflicting gold labels")
        gold_by_event[key] = gold
        if (
            prediction.label not in rule_set.candidate_labels
            or type(prediction.score) not in (int, float)
            or type(prediction.margin) not in (int, float)
            or not math.isfinite(prediction.score)
            or not math.isfinite(prediction.margin)
            or not 0 <= prediction.margin <= prediction.score <= 1
        ):
            raise ValueError("decision calibration contains failed or invalid model predictions")


def _validate_policy(base_rule_set: RuleSet, policy: RuleDecisionPolicy) -> None:
    base = asdict(base_rule_set.decision)
    values = asdict(policy)
    for name in base.keys() - {"review_model_score", "ambiguity_score", "ambiguity_margin"}:
        if values[name] != base[name]:
            raise ValueError(f"decision calibration may not change {name}")
    if (
        not 0 < policy.ambiguity_score <= base_rule_set.decision.ambiguity_score
        or not 0 < policy.ambiguity_margin <= base_rule_set.decision.ambiguity_margin
    ):
        raise ValueError("decision calibration may only relax positive score/margin gates")
    expected_floor = min(base_rule_set.decision.review_model_score, policy.ambiguity_score)
    if policy.review_model_score != expected_floor:
        raise ValueError("review score floor must track the selected score gate")
    validate_rule_set(replace(base_rule_set, decision=policy))


def _decide(items, predictions, rule_set):
    engine = RuleEngine(rule_set)
    return [engine.decide(
        item.title,
        article_description(item.title, item.description, item.source),
        item.content,
        prediction,
    ).final_label for item, prediction in zip(items, predictions, strict=True)]


def _evidence(labels, gold_labels, event_ids, candidate_labels) -> DecisionEvidence:
    selected = [index for index, label in enumerate(labels) if label in candidate_labels]
    events = defaultdict(list)
    for index in selected:
        events[event_ids[index].strip().casefold()].append(labels[index] == gold_labels[index])
    correct = sum(labels[index] == gold_labels[index] for index in selected)
    independent_correct = sum(all(results) for results in events.values())
    support = len(selected)
    return DecisionEvidence(
        total=len(labels), support=support, correct=correct,
        coverage=support / len(labels), accuracy=correct / support if support else 0.0,
        independent_support=len(events), independent_correct=independent_correct,
        accuracy_lower_bound=wilson_lower_bound(independent_correct, len(events)),
    )


def _compare(baseline, selected, gold_labels, event_ids, candidate_labels,
             minimum_support, accuracy_lower_bound, *, keywords, input_mode,
             include_breakdown=False) -> DecisionPolicyEvaluation:
    previous_changed = sum(
        before in candidate_labels and before != after
        for before, after in zip(baseline, selected, strict=True)
    )
    harmful = sum(
        before == gold and after != gold
        for before, after, gold in zip(baseline, selected, gold_labels, strict=True)
    )
    newly = [after if before == "검토필요" else "검토필요"
             for before, after in zip(baseline, selected, strict=True)]
    baseline_evidence = _evidence(baseline, gold_labels, event_ids, candidate_labels)
    selected_evidence = _evidence(selected, gold_labels, event_ids, candidate_labels)
    new_evidence = _evidence(newly, gold_labels, event_ids, candidate_labels)
    failures = []
    for name, evidence in (("automatic decisions", selected_evidence), ("new decisions", new_evidence)):
        if evidence.independent_support < minimum_support:
            failures.append(f"{name}: insufficient independent event support")
        if evidence.accuracy_lower_bound < accuracy_lower_bound:
            failures.append(f"{name}: accuracy lower bound below target")
    if selected_evidence.support <= baseline_evidence.support:
        failures.append("review coverage did not improve")
    if previous_changed:
        failures.append("previous automatic decisions changed")
    if harmful:
        failures.append("previous correct decisions were harmed")
    per_label, per_keyword = {}, {}
    keyword_keys = [normalize_key(keyword) for keyword in keywords]
    # A large easy group must not conceal a newly accepted wrong topic/query.
    # Small groups do not receive a precision guarantee: their CI is reported.
    for group_name, group_keys in (("label", gold_labels), ("keyword", keyword_keys)):
        counts, correct_counts = Counter(), Counter()
        for key, after, gold in zip(group_keys, newly, gold_labels, strict=True):
            if after in candidate_labels:
                counts[key] += 1
                correct_counts[key] += int(after == gold)
        for key, support in counts.items():
            if correct_counts[key] / support < accuracy_lower_bound:
                failures.append(f"new decisions: observed accuracy below target for {group_name} {key!r}")
    if include_breakdown:
        for group_keys, output in ((gold_labels, per_label), (keyword_keys, per_keyword)):
            groups = defaultdict(list)
            for index, key in enumerate(group_keys):
                groups[key].append(index)
            for key, indices in groups.items():
                output[key] = {
                    name: _evidence([values[i] for i in indices], [gold_labels[i] for i in indices],
                                    [event_ids[i] for i in indices], candidate_labels).to_dict()
                    for name, values in (("baseline", baseline), ("selected", selected), ("newly_decided", newly))
                }
    return DecisionPolicyEvaluation(
        baseline_evidence, selected_evidence, new_evidence,
        previous_changed, harmful, not failures, tuple(failures), per_label, per_keyword, input_mode,
    )


def evaluate_decision_policy(
    items: list[NewsItem], predictions: list[ModelPrediction], gold_labels: list[str],
    base_rule_set: RuleSet, policy: RuleDecisionPolicy, *, event_ids: list[str],
    minimum_support: int = 30, accuracy_lower_bound: float = 0.85, input_mode: str = "without_body",
) -> DecisionPolicyEvaluation:
    _validate_targets(minimum_support, accuracy_lower_bound)
    _validate_samples(items, predictions, gold_labels, base_rule_set, event_ids, input_mode)
    _validate_policy(base_rule_set, policy)
    return _compare(
        _decide(items, predictions, base_rule_set),
        _decide(items, predictions, replace(base_rule_set, decision=policy)),
        gold_labels, event_ids, base_rule_set.candidate_labels,
        minimum_support, accuracy_lower_bound, include_breakdown=True,
        keywords=[item.keyword for item in items], input_mode=input_mode,
    )


def calibrate_decision_policy(
    items: list[NewsItem], predictions: list[ModelPrediction], gold_labels: list[str],
    rule_set: RuleSet, *, event_ids: list[str], minimum_support: int = 30,
    accuracy_lower_bound: float = 0.85, input_mode: str = "without_body",
) -> DecisionCalibrationResult:
    """Search development-only observations; never tune on the evaluation set."""
    _validate_targets(minimum_support, accuracy_lower_bound)
    _validate_samples(items, predictions, gold_labels, rule_set, event_ids, input_mode)
    base = rule_set.decision
    baseline = _decide(items, predictions, rule_set)
    # Ranking/rule precedence is expensive and invariant. Obtain possible final
    # labels through the real engine, then test the effective score/margin gates.
    # The old 0.40 floor must not hide lower-score development observations.
    relaxed = _decide(items, predictions, replace(rule_set, decision=replace(
        base, review_model_score=0.0, ambiguity_score=0.0, ambiguity_margin=0.0,
    )))
    eligible = [prediction for before, after, prediction in zip(
        baseline, relaxed, predictions, strict=True,
    ) if before == "검토필요" and after in rule_set.candidate_labels
        and prediction.score > 0 and prediction.margin > 0]
    # Use observed boundaries, not an arbitrary lower default. Zero scores and
    # exact top-two ties never become automatic decisions through calibration.
    scores = sorted({base.ambiguity_score} | {
        p.score for p in eligible if p.score <= base.ambiguity_score
    })
    margins = sorted({base.ambiguity_margin} | {
        p.margin for p in eligible if p.margin <= base.ambiguity_margin
    })
    best = None
    candidates_tested = 0
    for score in scores:
        for margin in margins:
            if (score, margin) == (base.ambiguity_score, base.ambiguity_margin):
                continue
            candidates_tested += 1
            selected = [after if before == "검토필요" and p.score >= score and p.margin >= margin
                        else before for before, after, p in zip(baseline, relaxed, predictions, strict=True)]
            evidence = _compare(
                baseline, selected, gold_labels, event_ids, rule_set.candidate_labels,
                minimum_support, accuracy_lower_bound,
                keywords=[item.keyword for item in items], input_mode=input_mode,
            )
            if not evidence.passed:
                continue
            # For equal coverage prefer stronger evidence, then less relaxation.
            key = (evidence.selected.support, evidence.newly_decided.accuracy_lower_bound,
                   evidence.selected.accuracy_lower_bound, score, margin)
            if best is None or key > best[0]:
                best = (key, replace(base, review_model_score=min(base.review_model_score, score),
                                     ambiguity_score=score, ambiguity_margin=margin), evidence)
    if best is None:
        raise ValueError("no evidence-backed decision policy reduces review; keep the default policy")
    verified = evaluate_decision_policy(
        items, predictions, gold_labels, rule_set, best[1], event_ids=event_ids,
        minimum_support=minimum_support, accuracy_lower_bound=accuracy_lower_bound,
        input_mode=input_mode,
    )
    if replace(verified, per_label={}, per_keyword={}) != best[2] or not verified.passed:
        raise RuntimeError("decision policy simulation disagrees with rule-engine decisions")
    return DecisionCalibrationResult(
        best[1], verified, minimum_support, accuracy_lower_bound, candidates_tested,
    )


def create_decision_profile(
    result: DecisionCalibrationResult, evaluation: DecisionPolicyEvaluation, *,
    dataset_path: Path, model_name: str, model_revision: str,
    candidate_hypotheses: list[str], hypothesis_template: str, input_policy: str,
    max_sequence_length: int, base_rule_set: RuleSet,
    development_rows: list[dict[str, str]], evaluation_rows: list[dict[str, str]],
    input_mode: str = "without_body",
) -> dict:
    """Create a deployable profile only after independent validation passes."""
    if not result.evidence.passed or not evaluation.passed:
        raise ValueError("independent evaluation must pass before creating a decision profile")
    if not isinstance(input_mode, str) or input_mode not in INPUT_MODES or result.evidence.input_mode != input_mode or evaluation.input_mode != input_mode:
        raise ValueError("decision profile evidence has incompatible body input modes")
    validate_unseen_keywords(development_rows, evaluation_rows)
    development_events = {row["event_id"].strip().casefold() for row in development_rows}
    evaluation_events = {row["event_id"].strip().casefold() for row in evaluation_rows}
    if development_events & evaluation_events:
        raise ValueError("decision profile development and evaluation events overlap")
    _validate_policy(base_rule_set, result.policy)
    digest = dataset_sha256(dataset_path)
    def split_metadata(rows, split, evidence):
        if (
            len(rows) != evidence.selected.total
            or any(row.get("split", "").strip() != split
                   or row.get("review_status", "").strip() != "confirmed"
                   or not row.get("reviewed_by", "").strip()
                   or not row.get("event_id", "").strip()
                   or input_mode == "with_body" and not row.get("content", "").strip()
                   or row.get("suggested_label", "").strip() for row in rows)
        ):
            raise ValueError("decision profile requires manually confirmed split data")
        return {
            "dataset": str(dataset_path), "dataset_sha256": digest, "split": split,
            "confirmed": True, "label_support": dict(Counter(row["gold_label"] for row in rows)),
            "keyword_count": len({normalize_key(row["keyword"]) for row in rows}),
            "event_count": len({row["event_id"].strip().casefold() for row in rows}),
            "evidence": evidence.to_dict(),
        }
    payload = {
        "schema_version": 2, "profile_type": "final_decision_policy",
        "model": {"name": model_name, "revision": model_revision},
        "candidate_labels": list(base_rule_set.candidate_labels),
        "candidate_hypotheses": candidate_hypotheses,
        "hypothesis_template": hypothesis_template,
        "input_policy": {"name": input_policy, "include_keyword": False,
                         "max_sequence_length": max_sequence_length, "input_mode": input_mode},
        "rule_policy": {"fingerprint_sha256": rule_set_fingerprint(base_rule_set),
                        "baseline_decision": asdict(base_rule_set.decision)},
        "thresholds": {"review_model_score": result.policy.review_model_score,
                       "ambiguity_score": result.policy.ambiguity_score,
                       "ambiguity_margin": result.policy.ambiguity_margin},
        "targets": {"minimum_support": result.minimum_support,
                    "accuracy_lower_bound": result.accuracy_lower_bound},
        "development": split_metadata(development_rows, "development", result.evidence),
        "evaluation": split_metadata(evaluation_rows, "evaluation", evaluation),
        "search": {"selection_split": "development", "candidates_tested": result.candidates_tested},
        "validation": {"status": "passed", "evaluation_required": False,
                       "events_disjoint": True, "unseen_keywords_validated": True},
    }
    for split in ("development", "evaluation"):
        _validate_split_metadata(payload[split], split, base_rule_set.candidate_labels,
                                 result.minimum_support, result.accuracy_lower_bound, input_mode)
    return payload


def _validated_evidence(values: dict) -> DecisionEvidence:
    if not isinstance(values, dict) or set(values) != set(DecisionEvidence.__dataclass_fields__):
        raise ValueError("invalid decision calibration support evidence")
    integers = ("total", "support", "correct", "independent_support", "independent_correct")
    if any(type(values[key]) is not int or values[key] < 0 for key in integers):
        raise ValueError("invalid decision calibration support counts")
    evidence = DecisionEvidence(**values)
    if not (
        0 <= evidence.correct <= evidence.support <= evidence.total
        and 0 <= evidence.independent_correct <= evidence.independent_support <= evidence.support
        and evidence.independent_correct <= evidence.correct
        and evidence.independent_support - evidence.independent_correct <= evidence.support - evidence.correct
        and evidence.total > 0
    ):
        raise ValueError("inconsistent decision calibration support counts")
    expected = {
        "coverage": evidence.support / evidence.total,
        "accuracy": evidence.correct / evidence.support if evidence.support else 0.0,
        "accuracy_lower_bound": wilson_lower_bound(evidence.independent_correct, evidence.independent_support),
    }
    for key, value in expected.items():
        actual = values[key]
        if type(actual) not in (float, int) or not math.isfinite(actual) or not math.isclose(actual, value, abs_tol=1e-10):
            raise ValueError("inconsistent decision calibration accuracy evidence")
    return evidence


def _validate_split_metadata(metadata, split, candidate_labels, minimum_support, target, input_mode):
    if not isinstance(metadata, dict) or metadata.get("split") != split or metadata.get("confirmed") is not True:
        raise ValueError("decision calibration requires confirmed development and evaluation evidence")
    digest = metadata.get("dataset_sha256", "")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("decision calibration dataset fingerprint is missing")
    label_support = metadata.get("label_support")
    if (
        not isinstance(label_support, dict) or set(label_support) != set(candidate_labels)
        or any(type(count) is not int or count < 1 for count in label_support.values())
        or type(metadata.get("keyword_count")) is not int or metadata["keyword_count"] < 2
    ):
        raise ValueError("decision calibration requires all labels and multiple keywords")
    evidence = metadata.get("evidence")
    if (
        not isinstance(evidence, dict) or evidence.get("passed") is not True
        or evidence.get("input_mode") != input_mode
        or evidence.get("failures") != [] or type(evidence.get("previous_decisions_changed")) is not int
        or evidence["previous_decisions_changed"] != 0
        or type(evidence.get("harmful_changes")) is not int or evidence["harmful_changes"] != 0
    ):
        raise ValueError("decision calibration validation did not pass")
    baseline, selected, newly = (_validated_evidence(evidence.get(key))
                                 for key in ("baseline", "selected", "newly_decided"))
    if (
        baseline.total != selected.total or newly.total != selected.total
        or sum(label_support.values()) != selected.total
        or type(metadata.get("event_count")) is not int
        or not selected.independent_support <= metadata["event_count"] <= selected.total
        or selected.support <= baseline.support
        or newly.support != selected.support - baseline.support
        or newly.correct != selected.correct - baseline.correct
        or baseline.independent_support > selected.independent_support
        or newly.independent_support > selected.independent_support
        or metadata["keyword_count"] > selected.total
    ):
        raise ValueError("inconsistent decision calibration baseline comparison")
    for item in (selected, newly):
        if item.independent_support < minimum_support or item.accuracy_lower_bound < target:
            raise ValueError("decision calibration evidence is below the requested safety target")
    per_label = evidence.get("per_label")
    if not isinstance(per_label, dict) or set(per_label) != set(candidate_labels):
        raise ValueError("decision calibration per-label evidence is missing")
    for name, overall in (("baseline", baseline), ("selected", selected), ("newly_decided", newly)):
        parts = []
        for label in candidate_labels:
            if not isinstance(per_label[label], dict):
                raise ValueError("invalid decision calibration per-label evidence")
            part = _validated_evidence(per_label[label].get(name))
            if part.total != label_support[label]:
                raise ValueError("inconsistent decision calibration per-label totals")
            if name == "newly_decided" and part.support and part.accuracy < target:
                raise ValueError("decision calibration new per-label accuracy is below target")
            parts.append(part)
        for key in ("total", "support", "correct", "independent_support", "independent_correct"):
            if sum(getattr(part, key) for part in parts) != getattr(overall, key):
                raise ValueError("inconsistent decision calibration per-label evidence")
    per_keyword = evidence.get("per_keyword")
    if (not isinstance(per_keyword, dict) or len(per_keyword) != metadata["keyword_count"]
            or any(not isinstance(key, str) or not key.strip() for key in per_keyword)):
        raise ValueError("decision calibration per-keyword evidence is missing")
    for name, overall in (("baseline", baseline), ("selected", selected), ("newly_decided", newly)):
        parts = []
        for group in per_keyword.values():
            if not isinstance(group, dict):
                raise ValueError("invalid decision calibration per-keyword evidence")
            part = _validated_evidence(group.get(name))
            if name == "newly_decided" and part.support and part.accuracy < target:
                raise ValueError("decision calibration new per-keyword accuracy is below target")
            parts.append(part)
        # One manually identified event can legitimately appear under two queries.
        for key in ("total", "support", "correct"):
            if sum(getattr(part, key) for part in parts) != getattr(overall, key):
                raise ValueError("inconsistent decision calibration per-keyword evidence")


def load_decision_profile(
    calibration_path: str | Path | None = None, *, base_rule_set: RuleSet,
    expected_model_name: str, expected_model_revision: str,
    expected_candidate_labels: list[str], expected_candidate_hypotheses: list[str],
    expected_hypothesis_template: str, expected_input_policy: str,
    expected_max_sequence_length: int,
) -> LoadedDecisionProfile:
    if not calibration_path:
        return LoadedDecisionProfile(base_rule_set.decision, None)
    try:
        payload = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("decision calibration file could not be loaded") from exc
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] not in {1, 2} or payload.get("profile_type") != "final_decision_policy":
        raise ValueError("unsupported decision calibration schema")
    input_metadata = payload.get("input_policy")
    input_mode = input_metadata.get("input_mode") if isinstance(input_metadata, dict) else None
    if not isinstance(input_mode, str) or input_mode not in INPUT_MODES:
        raise ValueError("decision calibration has an invalid body input mode")
    bindings = {
        "model": {"name": expected_model_name, "revision": expected_model_revision},
        "candidate_labels": expected_candidate_labels,
        "candidate_hypotheses": expected_candidate_hypotheses,
        "hypothesis_template": expected_hypothesis_template,
        "input_policy": {"name": expected_input_policy, "include_keyword": False,
                         "max_sequence_length": expected_max_sequence_length, "input_mode": input_mode},
        "rule_policy": {"fingerprint_sha256": rule_set_fingerprint(base_rule_set),
                        "baseline_decision": asdict(base_rule_set.decision)},
    }
    for key, expected in bindings.items():
        if payload.get(key) != expected:
            raise ValueError(f"decision calibration {key} does not match runtime; recalibrate")
    if input_metadata.get("include_keyword") is not False or type(input_metadata.get("max_sequence_length")) is not int:
        raise ValueError("invalid decision calibration input policy")
    validation = payload.get("validation")
    if (
        not isinstance(validation, dict)
        or set(validation) != {"status", "evaluation_required", "events_disjoint", "unseen_keywords_validated"}
        or validation.get("status") != "passed"
        or validation.get("evaluation_required") is not False
        or validation.get("events_disjoint") is not True
        or validation.get("unseen_keywords_validated") is not True
    ):
        raise ValueError("decision calibration requires independent unseen-keyword evaluation")
    search = payload.get("search")
    if not isinstance(search, dict) or search.get("selection_split") != "development" or type(search.get("candidates_tested")) is not int or search["candidates_tested"] < 1:
        raise ValueError("decision calibration must be selected on development data only")
    targets = payload.get("targets")
    if not isinstance(targets, dict) or set(targets) != {"minimum_support", "accuracy_lower_bound"}:
        raise ValueError("decision calibration safety targets are missing")
    _validate_targets(targets["minimum_support"], targets["accuracy_lower_bound"])
    for split in ("development", "evaluation"):
        _validate_split_metadata(payload.get(split), split, expected_candidate_labels,
                                 targets["minimum_support"], targets["accuracy_lower_bound"], input_mode)
    thresholds = payload.get("thresholds")
    threshold_keys = {"ambiguity_score", "ambiguity_margin"}
    if payload["schema_version"] == 2:
        threshold_keys.add("review_model_score")
    if (
        not isinstance(thresholds, dict) or set(thresholds) != threshold_keys
        or any(type(value) not in (float, int) or not math.isfinite(value) for value in thresholds.values())
    ):
        raise ValueError("invalid decision calibration ambiguity thresholds")
    policy = replace(base_rule_set.decision, **thresholds)
    _validate_policy(base_rule_set, policy)
    return LoadedDecisionProfile(policy, input_mode)


def load_decision_policy(calibration_path: str | Path | None = None, **runtime_metadata) -> RuleDecisionPolicy:
    """Compatibility helper; runtime processors should use the mode-aware loader."""
    return load_decision_profile(calibration_path, **runtime_metadata).policy
