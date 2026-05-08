from __future__ import annotations

import re
from dataclasses import dataclass
from news_classifier.utils.text import clean_text


@dataclass(frozen=True)
class TextFeatures:
    length: int
    digit_count: int
    uppercase_count: int
    has_percent: bool
    has_money: bool
    keyword_hits: dict[str, int]


def count_keyword_hits(text: str, keywords_by_label: dict[str, list[str]]) -> dict[str, int]:
    normalized = clean_text(text).lower()
    return {
        label: sum(1 for keyword in keywords if keyword.lower() in normalized)
        for label, keywords in keywords_by_label.items()
    }


def extract_text_features(text: str, keywords_by_label: dict[str, list[str]]) -> TextFeatures:
    text = clean_text(text)
    return TextFeatures(
        length=len(text),
        digit_count=sum(1 for ch in text if ch.isdigit()),
        uppercase_count=sum(1 for ch in text if ch.isupper()),
        has_percent="%" in text or "퍼센트" in text,
        has_money=bool(re.search(r"(원|달러|억원|조원|매출|영업이익)", text)),
        keyword_hits=count_keyword_hits(text, keywords_by_label),
    )
