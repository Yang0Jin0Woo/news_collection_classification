import sqlite3

from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.models import ModelPrediction, NewsItem
from news_classifier.rules.default_rules import RULES
from news_classifier.storage.sqlite_store import SqliteNewsStore


LEGACY_SCHEMA = """
CREATE TABLE classified_news (
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


def rule_result():
    return ClassificationPostProcessor(RuleEngine(RULES)).process(
        NewsItem(
            keyword="AI",
            title="상장 첫날 따따블 코스모로보틱스",
            description="단숨에 로봇 대장주 우뚝",
            link="https://example.com/news",
        ),
        ModelPrediction("기술개발", 0.85, 0.80),
    )


def test_sqlite_store_saves_decision_metadata(tmp_path):
    path = tmp_path / "news.db"
    store = SqliteNewsStore(str(path))

    store.save([rule_result()])
    row = store.list_recent(1)[0]

    assert row["model_confidence"] == 0.85
    assert row["model_confidence_level"] == "높음"
    assert row["final_category"] == "금융/투자"
    assert row["decision_source"] == "RULE"
    assert row["review_required"] == 0
    assert row["final_decision_status"] == "DECIDED"


def test_sqlite_store_migrates_existing_table(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(LEGACY_SCHEMA)

    store = SqliteNewsStore(str(path))
    store.save([rule_result()])

    with sqlite3.connect(path) as conn:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(classified_news)")
        }
        row = conn.execute(
            """
            SELECT model_confidence, decision_source, review_required,
                   final_decision_status
            FROM classified_news
            """
        ).fetchone()

    assert {
        "model_confidence",
        "model_confidence_level",
        "decision_source",
        "review_required",
        "final_decision_status",
    } <= columns
    assert row == (0.85, "RULE", 0, "DECIDED")
