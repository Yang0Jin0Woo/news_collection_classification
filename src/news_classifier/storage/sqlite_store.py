from __future__ import annotations

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
    model_confidence REAL,
    model_confidence_level TEXT,
    score_margin REAL,
    final_category TEXT,
    decision_source TEXT,
    review_required INTEGER,
    final_decision_status TEXT,
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

REQUIRED_RESULT_COLUMNS = {
    "model_confidence": "REAL",
    "model_confidence_level": "TEXT",
    "decision_source": "TEXT",
    "review_required": "INTEGER",
    "final_decision_status": "TEXT",
}


class SqliteNewsStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.init_schema()

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            existing_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(classified_news)")
            }
            for column, column_type in REQUIRED_RESULT_COLUMNS.items():
                if column not in existing_columns:
                    conn.execute(
                        f"ALTER TABLE classified_news ADD COLUMN {column} {column_type}"
                    )

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
                        classification_text, model_category, model_category_score,
                        model_confidence, model_confidence_level, score_margin,
                        final_category, decision_source, review_required,
                        final_decision_status, rule_applied, rule_reason,
                        rule_best_label, rule_match_count, top3_labels, top3_scores,
                        created_at, unique_key
                    ) VALUES (
                        :keyword, :title, :source, :published_at, :link, :description, :content,
                        :classification_text, :model_category, :model_category_score,
                        :model_confidence, :model_confidence_level, :score_margin,
                        :final_category, :decision_source, :review_required,
                        :final_decision_status, :rule_applied, :rule_reason,
                        :rule_best_label, :rule_match_count, :top3_labels, :top3_scores,
                        :created_at, :unique_key
                    )
                    """,
                    data,
                )
