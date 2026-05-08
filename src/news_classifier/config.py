from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class AppSettings:
    """Application settings loaded from environment variables.

    The class intentionally avoids a hard dependency on external config
    libraries so the project can run in small interview/demo environments.
    """

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
    request_timeout_seconds: int = 15
    default_limit: int = 10
    max_sequence_length: int = 1200
    base_rule_override_threshold: float = 0.55
    min_margin_threshold: float = 0.08
    min_rule_match_count: int = 2
    review_needed_score_threshold: float = 0.40

    @property
    def headers(self) -> dict[str, str]:
        return {"User-Agent": self.user_agent}
