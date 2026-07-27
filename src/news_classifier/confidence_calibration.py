from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from news_classifier.classifiers.confidence import ConfidenceThresholds


@dataclass(frozen=True)
class CalibrationSample:
    score: float
    margin: float
    correct: bool


@dataclass(frozen=True)
class ThresholdEvidence:
    score: float
    margin: float
    support: int
    coverage: float
    accuracy: float
    accuracy_lower_bound: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationResult:
    thresholds: ConfidenceThresholds
    high: ThresholdEvidence
    medium: ThresholdEvidence
    total: int


def wilson_lower_bound(correct: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    proportion = correct / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    adjustment = z * math.sqrt(
        (proportion * (1 - proportion) + z * z / (4 * total)) / total
    )
    return (centre - adjustment) / denominator


def _threshold_candidates(
    samples: list[CalibrationSample],
    target_lower_bound: float,
    minimum_support: int,
    maximum_score: float = 1.0,
    maximum_margin: float = 1.0,
    exclude_score: float | None = None,
    exclude_margin: float | None = None,
) -> list[ThresholdEvidence]:
    score_candidates = sorted({sample.score for sample in samples})
    margin_candidates = sorted({sample.margin for sample in samples})
    candidates: list[ThresholdEvidence] = []

    for score in score_candidates:
        if score > maximum_score:
            continue
        for margin in margin_candidates:
            if margin > maximum_margin:
                continue
            selected = [
                sample
                for sample in samples
                if sample.score >= score and sample.margin >= margin
                and not (
                    exclude_score is not None
                    and exclude_margin is not None
                    and sample.score >= exclude_score
                    and sample.margin >= exclude_margin
                )
            ]
            support = len(selected)
            if support < minimum_support:
                continue
            correct = sum(sample.correct for sample in selected)
            lower_bound = wilson_lower_bound(correct, support)
            if lower_bound < target_lower_bound:
                continue
            candidates.append(
                ThresholdEvidence(
                    score=score,
                    margin=margin,
                    support=support,
                    coverage=support / len(samples),
                    accuracy=correct / support,
                    accuracy_lower_bound=lower_bound,
                )
            )

    return candidates


def _best_evidence(candidates: list[ThresholdEvidence]) -> ThresholdEvidence:
    if not candidates:
        raise ValueError("no confidence threshold satisfies the requested evidence")
    return max(
        candidates,
        key=lambda item: (
            item.support,
            item.accuracy_lower_bound,
            item.accuracy,
            -item.score,
            -item.margin,
        ),
    )


def calibrate_confidence(
    samples: list[CalibrationSample],
    high_accuracy_lower_bound: float = 0.80,
    medium_accuracy_lower_bound: float = 0.65,
    minimum_support: int = 15,
) -> CalibrationResult:
    if len(samples) < minimum_support:
        raise ValueError("not enough development samples for calibration")
    if not 0 < medium_accuracy_lower_bound <= high_accuracy_lower_bound < 1:
        raise ValueError("invalid calibration accuracy targets")
    for sample in samples:
        if (
            not math.isfinite(sample.score)
            or not math.isfinite(sample.margin)
            or not 0.0 <= sample.margin <= sample.score <= 1.0
        ):
            raise ValueError("calibration samples contain invalid score or margin")

    high_candidates = _threshold_candidates(
        samples,
        high_accuracy_lower_bound,
        minimum_support,
    )
    pairs: list[tuple[ThresholdEvidence, ThresholdEvidence]] = []
    for high_candidate in high_candidates:
        medium_candidates = _threshold_candidates(
            samples,
            medium_accuracy_lower_bound,
            minimum_support,
            maximum_score=high_candidate.score,
            maximum_margin=high_candidate.margin,
            exclude_score=high_candidate.score,
            exclude_margin=high_candidate.margin,
        )
        if medium_candidates:
            pairs.append((high_candidate, _best_evidence(medium_candidates)))
    if not pairs:
        raise ValueError(
            "no compatible high and medium confidence thresholds satisfy "
            "the requested evidence"
        )
    high, medium = max(
        pairs,
        key=lambda pair: (
            pair[0].support + pair[1].support,
            pair[0].support,
            pair[0].accuracy_lower_bound,
            pair[1].accuracy_lower_bound,
        ),
    )
    thresholds = ConfidenceThresholds(
        high_score=high.score,
        high_margin=high.margin,
        medium_score=medium.score,
        medium_margin=medium.margin,
    ).validate()
    return CalibrationResult(
        thresholds=thresholds,
        high=high,
        medium=medium,
        total=len(samples),
    )
