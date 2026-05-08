from __future__ import annotations

import logging
from dataclasses import dataclass

from news_classifier.classifiers.base import NewsClassifier
from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.collectors.base import NewsCollector
from news_classifier.collectors.article_scraper import ArticleScraper
from news_classifier.dedup.title_deduplicator import TitleSourceDeduplicator
from news_classifier.models import ClassifiedNews

logger = logging.getLogger(__name__)


@dataclass
class NewsPipeline:
    collector: NewsCollector
    classifier: NewsClassifier
    postprocessor: ClassificationPostProcessor
    deduplicator: TitleSourceDeduplicator
    article_scraper: ArticleScraper | None = None

    def run(self, keyword: str, limit: int = 10, enrich_content: bool = False) -> list[ClassifiedNews]:
        logger.info("news pipeline started keyword=%s limit=%s", keyword, limit)
        news_list = self.collector.fetch(keyword=keyword, limit=limit)
        logger.info("fetched=%s", len(news_list))

        news_list = self.deduplicator.deduplicate(news_list)
        logger.info("deduplicated=%s", len(news_list))

        if enrich_content and self.article_scraper is not None:
            news_list = self.article_scraper.enrich_many(news_list)
            logger.info("content enrichment finished")

        results: list[ClassifiedNews] = []
        for idx, item in enumerate(news_list, start=1):
            logger.info("classifying %s/%s | %s", idx, len(news_list), item.title)
            prediction = self.classifier.classify(item.classification_text())
            classified = self.postprocessor.process(item, prediction)
            results.append(classified)
        return results
