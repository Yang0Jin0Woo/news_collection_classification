from dataclasses import dataclass, field
import os
from pathlib import Path

from dotenv import load_dotenv


def load_environment(dotenv_path: str | Path | None = None) -> bool:
    """기존 환경변수를 우선하면서 선택한 `.env` 파일을 불러온다."""
    return load_dotenv(dotenv_path=dotenv_path, override=False)


load_environment()


@dataclass(frozen=True)
class AppSettings:
    """`.env`와 운영체제 환경변수에서 애플리케이션 설정을 읽는다."""

    user_agent: str = field(default_factory=lambda: os.getenv(
        "NEWS_USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PortfolioNewsBot/1.0",
    ))
    output_csv: str = field(default_factory=lambda: os.getenv(
        "NEWS_OUTPUT_CSV", "news_analysis_results.csv"
    ))
    output_db: str = field(default_factory=lambda: os.getenv(
        "NEWS_OUTPUT_DB", "news_analysis.db"
    ))
    classification_model: str = field(default_factory=lambda: os.getenv(
        "NEWS_MODEL", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    ))
    classification_model_revision: str = field(default_factory=lambda: os.getenv(
        "NEWS_MODEL_REVISION",
        "b5113eb38ab63efdd7f280f8c144ea8b13f978ce",
    ))
    confidence_calibration_path: str = field(default_factory=lambda: os.getenv(
        "NEWS_CONFIDENCE_CALIBRATION", ""
    ))
    decision_calibration_path: str = field(default_factory=lambda: os.getenv(
        "NEWS_DECISION_CALIBRATION", ""
    ))
    event_rule_profile_path: str = field(default_factory=lambda: os.getenv(
        "NEWS_EVENT_RULE_PROFILE", ""
    ))
    review_enrichment_enabled: bool = True
    request_timeout_seconds: int = 15
    max_sequence_length: int = 1200
    classification_batch_size: int = 4

    @property
    def headers(self) -> dict[str, str]:
        return {"User-Agent": self.user_agent}
