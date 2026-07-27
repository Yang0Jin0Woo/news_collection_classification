from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ConfidenceThresholds:
    high_score: float = 0.70
    high_margin: float = 0.10
    medium_score: float = 0.50
    medium_margin: float = 0.05

    def validate(self) -> "ConfidenceThresholds":
        values = asdict(self)
        if any(not 0.0 <= value <= 1.0 for value in values.values()):
            raise ValueError("confidence thresholds must be between 0 and 1")
        if self.high_score < self.medium_score:
            raise ValueError("high_score must be greater than or equal to medium_score")
        if self.high_margin < self.medium_margin:
            raise ValueError("high_margin must be greater than or equal to medium_margin")
        return self

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "ConfidenceThresholds":
        try:
            thresholds = cls(
                high_score=float(values["high_score"]),
                high_margin=float(values["high_margin"]),
                medium_score=float(values["medium_score"]),
                medium_margin=float(values["medium_margin"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid confidence calibration thresholds") from exc
        return thresholds.validate()


def load_confidence_thresholds(
    calibration_path: str | Path | None = None,
    *,
    expected_model_name: str | None = None,
    expected_model_revision: str | None = None,
    expected_candidate_labels: list[str] | None = None,
    expected_input_policy: str | None = None,
    expected_hypothesis_template: str | None = None,
) -> ConfidenceThresholds:
    if not calibration_path:
        return ConfidenceThresholds()

    path = Path(calibration_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"confidence calibration file could not be loaded: {path}") from exc

    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported confidence calibration schema")

    model = payload.get("model")
    if expected_model_name is not None:
        if not isinstance(model, dict) or model.get("name") != expected_model_name:
            raise ValueError("confidence calibration model name does not match")
    if expected_model_revision is not None:
        if not isinstance(model, dict) or model.get("revision") != expected_model_revision:
            raise ValueError("confidence calibration model revision does not match")
    if (
        expected_candidate_labels is not None
        and payload.get("candidate_labels") != expected_candidate_labels
    ):
        raise ValueError("confidence calibration candidate labels do not match")
    input_policy = payload.get("input_policy")
    if expected_input_policy is not None:
        if (
            not isinstance(input_policy, dict)
            or input_policy.get("name") != expected_input_policy
            or input_policy.get("include_keyword") is not False
        ):
            raise ValueError("confidence calibration input policy does not match")
    if (
        expected_hypothesis_template is not None
        and payload.get("hypothesis_template") != expected_hypothesis_template
    ):
        raise ValueError("confidence calibration hypothesis template does not match")

    values = payload.get("thresholds")
    if not isinstance(values, dict):
        raise ValueError("confidence calibration file has no thresholds object")
    return ConfidenceThresholds.from_dict(values)


def confidence_level(
    score: float,
    margin: float,
    thresholds: ConfidenceThresholds | None = None,
) -> str:
    thresholds = thresholds or ConfidenceThresholds()
    if score >= thresholds.high_score and margin >= thresholds.high_margin:
        return "높음"
    if score >= thresholds.medium_score and margin >= thresholds.medium_margin:
        return "보통"
    return "낮음"


def is_ambiguous(
    score: float,
    margin: float,
    min_score: float = 0.50,
    min_margin: float = 0.05,
) -> bool:
    return score < min_score or margin < min_margin
