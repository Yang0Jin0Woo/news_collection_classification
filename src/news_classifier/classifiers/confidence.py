def confidence_level(score: float, margin: float) -> str:
    if score >= 0.70 and margin >= 0.10:
        return "높음"
    if score >= 0.50 and margin >= 0.05:
        return "보통"
    return "낮음"


def is_ambiguous(score: float, margin: float, min_score: float = 0.50, min_margin: float = 0.05) -> bool:
    return score < min_score or margin < min_margin
