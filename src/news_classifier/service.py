from __future__ import annotations

from dataclasses import replace

from news_classifier.classifiers.confidence import load_confidence_thresholds
from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.classifiers.zero_shot_classifier import (
    HYPOTHESIS_TEMPLATE,
    ZeroShotNewsClassifier,
)
from news_classifier.collectors.article_scraper import ArticleScraper
from news_classifier.collectors.google_rss import GoogleNewsRssCollector
from news_classifier.config import AppSettings
from news_classifier.dedup.title_deduplicator import TitleSourceDeduplicator
from news_classifier.decision_calibration import load_decision_profile
from news_classifier.models import CLASSIFICATION_INPUT_POLICY
from news_classifier.pipeline import NewsPipeline
from news_classifier.rules.default_rules import (
    CANDIDATE_LABELS,
    DEFAULT_RULE_SET,
    validate_rule_configuration,
)
from news_classifier.rules.event_profile import load_event_rule_profile, runtime_bindings
from news_classifier.utils.http import HttpClient


def build_pipeline(settings: AppSettings | None = None) -> NewsPipeline:
    settings = settings or AppSettings()
    validate_rule_configuration(DEFAULT_RULE_SET)
    classifier = ZeroShotNewsClassifier(
        model_name=settings.classification_model,
        model_revision=settings.classification_model_revision,
        candidate_labels=CANDIDATE_LABELS,
        max_sequence_length=settings.max_sequence_length,
        batch_size=settings.classification_batch_size,
    )
    runtime_rules = load_event_rule_profile(
        settings.event_rule_profile_path, DEFAULT_RULE_SET,
        runtime_bindings(settings, classifier, CLASSIFICATION_INPUT_POLICY, HYPOTHESIS_TEMPLATE, DEFAULT_RULE_SET),
    )
    confidence_thresholds = load_confidence_thresholds(
        settings.confidence_calibration_path,
        expected_model_name=settings.classification_model,
        expected_model_revision=settings.classification_model_revision,
        expected_candidate_labels=CANDIDATE_LABELS,
        expected_input_policy=CLASSIFICATION_INPUT_POLICY,
        expected_hypothesis_template=HYPOTHESIS_TEMPLATE,
        expected_candidate_hypotheses=classifier.candidate_hypotheses,
    )
    http_client = HttpClient(settings.headers, settings.request_timeout_seconds)
    collector = GoogleNewsRssCollector(http_client)
    scraper = ArticleScraper(http_client)
    decision_profile = load_decision_profile(
        settings.decision_calibration_path,
        base_rule_set=runtime_rules,
        expected_model_name=settings.classification_model,
        expected_model_revision=settings.classification_model_revision,
        expected_candidate_labels=CANDIDATE_LABELS,
        expected_candidate_hypotheses=classifier.candidate_hypotheses,
        expected_hypothesis_template=HYPOTHESIS_TEMPLATE,
        expected_input_policy=CLASSIFICATION_INPUT_POLICY,
        expected_max_sequence_length=settings.max_sequence_length,
    )
    rule_engine = RuleEngine(replace(runtime_rules, decision=decision_profile.policy))
    postprocessor = ClassificationPostProcessor(
        rule_engine,
        confidence_thresholds=confidence_thresholds,
        baseline_rule_engine=RuleEngine(runtime_rules),
        decision_input_mode=decision_profile.input_mode,
    )
    return NewsPipeline(
        collector=collector,
        classifier=classifier,
        postprocessor=postprocessor,
        deduplicator=TitleSourceDeduplicator(),
        article_scraper=scraper,
        review_enrichment_enabled=settings.review_enrichment_enabled,
    )
