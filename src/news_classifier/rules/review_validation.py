"""Compare new event rules against the unchanged hybrid, not the model alone."""
from __future__ import annotations

from collections import Counter

from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.rules.policy import rule_term_matches_row


def audit_rule_holdout(rule_set, rows, classified_results) -> dict:
    """Each proposed expression needs its own held-out successful rule cases."""
    if len(rows) != len(classified_results):
        raise ValueError("rule holdout rows and results must align")
    engine = RuleEngine(rule_set)
    terms = []
    for policy in rule_set.labels:
        for term in policy.terms:
            if term.origin != "development":
                continue
            correct_events, wrong_cases, other_topic_matches = set(), [], 0
            for row, result in zip(rows, classified_results, strict=True):
                actual = {**row, "title": result.item.title,
                          "description": result.item.description, "source": result.item.source,
                          "content": result.item.content}
                if not rule_term_matches_row(engine, policy.label, term, actual):
                    continue
                gold = row["gold_label"]
                if gold != policy.label:
                    other_topic_matches += 1
                decision = result.rule_decision
                if not decision.rule_applied or decision.final_label != policy.label:
                    continue
                if gold == policy.label:
                    event = row.get("event_id", "").strip().casefold()
                    if event:
                        correct_events.add(event)
                else:
                    wrong_cases.append(row.get("id", ""))
            terms.append({
                "category": policy.label, "phrase": term.phrase,
                "correct_rule_independent_events": len(correct_events),
                "wrong_rule_count": len(wrong_cases), "wrong_case_ids": wrong_cases,
                "other_topic_matching_articles": other_topic_matches,
                "passed": len(correct_events) >= rule_set.minimum_development_evidence_events
                          and not wrong_cases,
            })
    return {"terms": terms, "passed": bool(terms) and all(term["passed"] for term in terms)}


def evaluate_review_reduction(
    expected: list[str], baseline: list[str], selected: list[str], *,
    labels: list[str], keywords: list[str], event_ids: list[str],
    has_development_rules: bool, split: str, unseen_keywords_validated: bool,
    minimum_independent_events: int = 2,
) -> dict:
    if not expected or not (len(expected) == len(baseline) == len(selected)
                            == len(keywords) == len(event_ids)):
        raise ValueError("rule review comparison inputs must be non-empty and aligned")
    if any(label not in labels for label in expected):
        raise ValueError("unknown gold label in rule review comparison")
    if has_development_rules and any(not event.strip() for event in event_ids):
        raise ValueError("new rule evaluation requires explicitly reviewed event ids")

    newly = [i for i, (before, after) in enumerate(zip(baseline, selected, strict=True))
             if before == "검토필요" and after in labels]
    harmful = sum(before == gold and after != gold
                  for gold, before, after in zip(expected, baseline, selected, strict=True))
    new_wrong = sum(after in labels and after != gold and before != after
                    for gold, before, after in zip(expected, baseline, selected, strict=True))
    wrong_newly = sum(selected[i] != expected[i] for i in newly)
    independent_newly = {event_ids[i].strip().casefold() for i in newly if event_ids[i].strip()}

    def stage(values):
        automatic = [i for i, value in enumerate(values) if value in labels]
        wrong = sum(values[i] != expected[i] for i in automatic)
        counts = Counter(values)
        return {
            "total": len(values), "review_count": counts["검토필요"],
            "review_rate": counts["검토필요"] / len(values),
            "automatic_count": len(automatic), "wrong_automatic_count": wrong,
            "automatic_accuracy": 1 - wrong / len(automatic) if automatic else None,
            "category_counts": {label: counts[label] for label in labels},
        }

    failures = []
    if has_development_rules:
        if split != "evaluation":
            failures.append("independent evaluation is still required")
        if not unseen_keywords_validated:
            failures.append("unseen-keyword validation is still required")
        if len(independent_newly) < minimum_independent_events:
            failures.append("insufficient newly automatic independent events")
        if baseline.count("검토필요") <= selected.count("검토필요"):
            failures.append("review count did not decrease")
        if wrong_newly or new_wrong:
            failures.append("new rules introduced wrong automatic decisions")
        if harmful:
            failures.append("previous correct decisions were harmed")

    def group_summary(keys):
        return {key: {
            "total": sum(value == key for value in keys),
            "newly_automatic_count": sum(keys[i] == key for i in newly),
            "newly_automatic_wrong_count": sum(keys[i] == key and selected[i] != expected[i]
                                               for i in newly),
        } for key in sorted(set(keys))}

    passed = has_development_rules and not failures
    return {
        "status": "no_new_rules" if not has_development_rules else ("passed" if passed else "not_validated"),
        "passed": passed, "failures": failures,
        "baseline": stage(baseline), "selected": stage(selected),
        "newly_automatic_count": len(newly),
        "newly_automatic_wrong_count": wrong_newly,
        "newly_automatic_accuracy": 1 - wrong_newly / len(newly) if newly else None,
        "newly_automatic_independent_events": len(independent_newly) if all(event_ids[i].strip() for i in newly) else None,
        "new_wrong_decision_count": new_wrong,
        "previous_correct_harmed_count": harmful,
        "per_gold_category": group_summary(expected),
        "per_keyword": group_summary([keyword.strip().casefold() for keyword in keywords]),
        "note": "Observed comparison only; small samples do not establish future accuracy. No rule activation performed.",
    }
