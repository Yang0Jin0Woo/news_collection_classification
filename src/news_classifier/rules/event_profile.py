"""Explicit human-reviewed rule proposals and independently evaluated profiles.

Profiles are trusted local configuration, not signed attestations. Candidate
files alone never activate rules in collect; use evaluate_news to export them.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from news_classifier.rules.policy import (
    RuleStrength, RuleTerm, rule_set_fingerprint, validate_rule_set,
)


def read_rule_proposals(path, base):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] != 1 or payload.get("profile_type") != "event_rule_candidates":
        raise ValueError("expected event_rule_candidates manifest, not an audit or approved profile")
    return rules_from_records(base, payload.get("rules"))


def rules_from_records(base, records):
    if not isinstance(records, list) or not 1 <= len(records) <= 128:
        raise ValueError("event rules must be a non-empty list of at most 128 expressions")
    additions = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"category", "phrase", "strength", "evidence_event_ids"}:
            raise ValueError("event rule record has unsupported fields")
        label, phrase, strength, events = (record[key] for key in ("category", "phrase", "strength", "evidence_event_ids"))
        if (not isinstance(label, str) or label not in base.candidate_labels or label == base.other_label
                or not isinstance(phrase, str) or not phrase.strip()
                or not isinstance(strength, str) or strength not in {"WEAK", "STRONG"}
                or not isinstance(events, list) or any(not isinstance(event, str) for event in events)):
            raise ValueError("invalid event rule category, expression, strength or event evidence")
        additions.setdefault(label, []).append(RuleTerm(
            phrase.strip(), RuleStrength[strength], "development", tuple(events),
        ))
    result = replace(base, labels=tuple(
        replace(policy, terms=policy.terms + tuple(additions.get(policy.label, [])), no_direct_rules=False)
        if policy.label in additions else policy for policy in base.labels
    ))
    validate_rule_set(result)
    return result


def rule_records(rule_set):
    return [{"category": policy.label, "phrase": term.phrase, "strength": term.strength.name,
             "evidence_event_ids": list(term.evidence_event_ids)}
            for policy in rule_set.labels for term in policy.terms if term.origin == "development"]


def runtime_bindings(settings, classifier, input_policy, hypothesis_template, base):
    return {
        "model": {"name": settings.classification_model, "revision": settings.classification_model_revision},
        "candidate_labels": list(base.candidate_labels),
        "candidate_hypotheses": list(classifier.candidate_hypotheses),
        "input_policy": input_policy, "hypothesis_template": hypothesis_template,
        "max_sequence_length": settings.max_sequence_length,
        "baseline_rule_fingerprint": rule_set_fingerprint(base),
    }


def validate_observed_rule_results(validation, rule_set):
    if not isinstance(validation, dict) or validation.get("status") != "passed" or validation.get("passed") is not True or validation.get("failures") != []:
        raise ValueError("event rules require a passed independent review-reduction evaluation")
    minimum = rule_set.minimum_development_evidence_events
    for key in ("newly_automatic_count", "newly_automatic_independent_events"):
        if type(validation.get(key)) is not int or validation[key] < minimum:
            raise ValueError("insufficient independently evaluated new rule decisions")
    if validation["newly_automatic_independent_events"] > validation["newly_automatic_count"]:
        raise ValueError("event count cannot exceed article count")
    for key in ("newly_automatic_wrong_count", "new_wrong_decision_count", "previous_correct_harmed_count"):
        if type(validation.get(key)) is not int or validation[key] != 0:
            raise ValueError("new event rules introduced wrong or harmful decisions")
    before, after = validation.get("baseline"), validation.get("selected")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("missing before/after event rule evaluation")
    for stage in (before, after):
        for key in ("total", "review_count", "automatic_count", "wrong_automatic_count"):
            if type(stage.get(key)) is not int or stage[key] < 0:
                raise ValueError("invalid event rule evaluation counts")
        if stage["total"] < 1 or stage["review_count"] + stage["automatic_count"] > stage["total"] or stage["wrong_automatic_count"] > stage["automatic_count"]:
            raise ValueError("inconsistent event rule evaluation counts")
    if before["total"] != after["total"] or before["review_count"] <= after["review_count"]:
        raise ValueError("event rule evaluation must reduce review count on the same cases")
    holdout = validation.get("per_rule_holdout")
    if not isinstance(holdout, dict) or holdout.get("passed") is not True or not isinstance(holdout.get("terms"), list):
        raise ValueError("each event expression requires independent held-out rule evidence")
    expected = {(record["category"], record["phrase"]) for record in rule_records(rule_set)}
    actual = set()
    for term in holdout["terms"]:
        if (not isinstance(term, dict) or not isinstance(term.get("category"), str) or not isinstance(term.get("phrase"), str)
                or term.get("passed") is not True
                or type(term.get("correct_rule_independent_events")) is not int
                or term["correct_rule_independent_events"] < minimum
                or type(term.get("wrong_rule_count")) is not int or term["wrong_rule_count"] != 0):
            raise ValueError("invalid per-expression held-out event evidence")
        key = (term["category"], term["phrase"])
        if key in actual:
            raise ValueError("duplicate held-out event expression")
        actual.add(key)
    if actual != expected:
        raise ValueError("held-out expressions do not match the proposed event rules")


def create_event_rule_profile(rule_set, validation, bindings, dataset_hash):
    validate_observed_rule_results(validation, rule_set)
    return {"schema_version": 1, "profile_type": "confirmed_event_rules", "bindings": bindings,
            "rules": rule_records(rule_set), "dataset_sha256": dataset_hash,
            "review_requirements": {"human_confirmed": True, "events_disjoint": True, "unseen_keywords": True},
            "validation": validation,
            "note": "Trusted local configuration. Observed small-sample validation is not a guarantee for future keywords."}


def load_event_rule_profile(path, base, expected_bindings):
    if not path:
        return base
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(payload, dict) or type(payload.get("schema_version")) is not int
            or payload["schema_version"] != 1 or payload.get("profile_type") != "confirmed_event_rules"):
        raise ValueError("candidate/audit files cannot activate event rules; export a validated rule profile")
    if payload.get("bindings") != expected_bindings:
        raise ValueError("event rule profile does not match the current model/input/base rules; re-evaluate")
    requirements = payload.get("review_requirements")
    if (not isinstance(requirements, dict) or set(requirements) != {"human_confirmed", "events_disjoint", "unseen_keywords"}
            or any(value is not True for value in requirements.values())):
        raise ValueError("event rule profile requires human confirmation and independent events/keywords")
    digest = payload.get("dataset_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
        raise ValueError("event rule profile is missing its evaluated dataset hash")
    result = rules_from_records(base, payload.get("rules"))
    validate_observed_rule_results(payload.get("validation"), result)
    return result
