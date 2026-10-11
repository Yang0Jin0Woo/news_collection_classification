from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from news_classifier.models import CLASSIFIED_NEWS_COLUMNS, ClassifiedNews

_INSERT_COLUMNS = (*CLASSIFIED_NEWS_COLUMNS, "unique_key")
_INSERT_SQL = (
    f"INSERT OR REPLACE INTO classified_news ({', '.join(_INSERT_COLUMNS)}) "
    f"VALUES ({', '.join(f':{column}' for column in _INSERT_COLUMNS)})"
)

_COLUMN_DEFINITIONS = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "keyword": "TEXT NOT NULL",
    "title": "TEXT NOT NULL",
    "source": "TEXT",
    "published_at": "TEXT",
    "link": "TEXT",
    "description": "TEXT",
    "content": "TEXT",
    "classification_text": "TEXT",
    "model_category": "TEXT",
    "model_category_score": "REAL",
    "model_confidence": "REAL",
    "model_confidence_level": "TEXT",
    "score_margin": "REAL",
    "final_category": "TEXT",
    "decision_source": "TEXT",
    "review_required": "INTEGER",
    "final_decision_status": "TEXT",
    "rule_applied": "TEXT",
    "rule_reason": "TEXT",
    "rule_best_label": "TEXT",
    "rule_match_count": "INTEGER",
    "top3_labels": "TEXT",
    "top3_scores": "TEXT",
    "created_at": "TEXT",
    "group_article_count": "INTEGER NOT NULL DEFAULT 1",
    "related_articles": "TEXT NOT NULL DEFAULT '[]'",
    "review_reclassification": "TEXT NOT NULL DEFAULT '{}'",
    "unique_key": "TEXT UNIQUE",
}

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS classified_news (\n    "
    + ",\n    ".join(f"{column} {definition}" for column, definition in _COLUMN_DEFINITIONS.items())
    + "\n);\n"
)

# Keep the existing migration scope and ordering; other legacy columns are untouched.
REQUIRED_RESULT_COLUMNS = {
    column: _COLUMN_DEFINITIONS[column]
    for column in (
        "model_confidence", "model_confidence_level", "decision_source",
        "review_required", "final_decision_status", "group_article_count",
        "related_articles", "review_reclassification",
    )
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
                conn.execute(_INSERT_SQL, data)
