"""Model input preparation shared by inference and review replay."""
from __future__ import annotations

from news_classifier.models import NewsItem
from news_classifier.utils.text import clean_text


def prepare_model_input(text: str, max_sequence_length: int) -> str:
    return clean_text(text)[:max_sequence_length]


def has_changed_article_input(
    before: NewsItem, after: NewsItem, max_sequence_length: int = 1200,
) -> bool:
    """Compare only the cleaned, length-limited text passed to the model."""
    return prepare_model_input(
        before.classification_text(), max_sequence_length,
    ) != prepare_model_input(after.classification_text(), max_sequence_length)
