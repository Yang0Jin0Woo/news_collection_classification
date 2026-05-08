from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from news_classifier.models import ClassifiedNews

SCHEMA = """
CREATE TABLE IF NOT EXISTS classified_news (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword TEXT NOT NULL,
    title TEXT NOT NULL,
    source TEXT,
    published_at TEXT,
    link TEXT,
    description TEXT,
    content TEXT,
    classification_text TEXT,
    model_category TEXT,
    model_category_score REAL,
    score_margin REAL,
    final_category TEXT,
    confidence_level TEXT,
    rule_applied TEXT,
    rule_reason TEXT,
    rule_best_label TEXT,
    rule_match_count INTEGER,
    top3_labels TEXT,
    top3_scores TEXT,
    created_at TEXT,
    unique_key TEXT UNIQUE
);
"""


class SqliteNewsStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.init_schema()

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def save(self, rows: Iterable[ClassifiedNews]) -> None:
        with self.connect() as conn:
            for row in rows:
                data = row.to_row()
                unique_key = f"{data['keyword']}|{data['title']}|{data['source']}"
                data["unique_key"] = unique_key
                conn.execute(
                    """
                    INSERT OR REPLACE INTO classified_news (
                        keyword, title, source, published_at, link, description, content,
                        classification_text, model_category, model_category_score, score_margin,
                        final_category, confidence_level, rule_applied, rule_reason,
                        rule_best_label, rule_match_count, top3_labels, top3_scores,
                        created_at, unique_key
                    ) VALUES (
                        :keyword, :title, :source, :published_at, :link, :description, :content,
                        :classification_text, :model_category, :model_category_score, :score_margin,
                        :final_category, :confidence_level, :rule_applied, :rule_reason,
                        :rule_best_label, :rule_match_count, :top3_labels, :top3_scores,
                        :created_at, :unique_key
                    )
                    """,
                    data,
                )

    def list_recent(self, limit: int = 100) -> list[dict]:
        with self.connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM classified_news ORDER BY published_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(row) for row in rows]
