from __future__ import annotations

from news_classifier.classifiers.confidence import load_confidence_thresholds
from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.classifiers.rule_engine import RuleEngine, RuleEngineConfig
from news_classifier.classifiers.zero_shot_classifier import (
    HYPOTHESIS_TEMPLATE,
    ZeroShotNewsClassifier,
)
from news_classifier.collectors.article_scraper import ArticleScraper
from news_classifier.collectors.google_rss import GoogleNewsRssCollector
from news_classifier.config import AppSettings
from news_classifier.dedup.title_deduplicator import TitleSourceDeduplicator
from news_classifier.models import CLASSIFICATION_INPUT_POLICY
from news_classifier.pipeline import NewsPipeline
from news_classifier.rules.default_rules import (
    CANDIDATE_LABELS,
    RULES,
    validate_rule_configuration,
)
from news_classifier.utils.http import HttpClient


def build_pipeline(settings: AppSettings | None = None) -> NewsPipeline:
    settings = settings or AppSettings()
    validate_rule_configuration()
    confidence_thresholds = load_confidence_thresholds(
        settings.confidence_calibration_path,
        expected_model_name=settings.classification_model,
        expected_model_revision=settings.classification_model_revision,
        expected_candidate_labels=CANDIDATE_LABELS,
        expected_input_policy=CLASSIFICATION_INPUT_POLICY,
        expected_hypothesis_template=HYPOTHESIS_TEMPLATE,
    )
    http_client = HttpClient(settings.headers, settings.request_timeout_seconds)
    collector = GoogleNewsRssCollector(http_client)
    scraper = ArticleScraper(http_client)
    classifier = ZeroShotNewsClassifier(
        model_name=settings.classification_model,
        model_revision=settings.classification_model_revision,
        candidate_labels=CANDIDATE_LABELS,
        max_sequence_length=settings.max_sequence_length,
        batch_size=settings.classification_batch_size,
    )
    rule_engine = RuleEngine(
        RULES,
        RuleEngineConfig(
            base_rule_override_threshold=settings.base_rule_override_threshold,
            min_margin_threshold=settings.min_margin_threshold,
            min_rule_match_count=settings.min_rule_match_count,
            review_needed_score_threshold=settings.review_needed_score_threshold,
        ),
    )
    postprocessor = ClassificationPostProcessor(
        rule_engine,
        confidence_thresholds=confidence_thresholds,
    )
    return NewsPipeline(
        collector=collector,
        classifier=classifier,
        postprocessor=postprocessor,
        deduplicator=TitleSourceDeduplicator(),
        article_scraper=scraper,
    )
