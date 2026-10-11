import sqlite3
import json
from dataclasses import replace

import pytest

from news_classifier.classifiers.postprocessor import ClassificationPostProcessor
from news_classifier.classifiers.rule_engine import RuleEngine
from news_classifier.models import (
    CLASSIFIED_NEWS_COLUMNS, ModelPrediction, NewsItem, NewsReference, ReviewReclassification,
)
from news_classifier.rules.default_rules import DEFAULT_RULE_SET
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
    return ClassificationPostProcessor(RuleEngine(DEFAULT_RULE_SET)).process(
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
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        row = dict(
            conn.execute(
                "SELECT * FROM classified_news ORDER BY id DESC LIMIT 1"
            ).fetchone()
        )

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


def test_sqlite_preserves_original_articles_in_group(tmp_path):
    result = rule_result()
    result = replace(result, item=replace(result.item, related_articles=(
        NewsReference("관련 원문 제목", "B", "2026-10-04T10:00:00+09:00", "https://example.com/other"),
    )))
    store = SqliteNewsStore(str(tmp_path / "group.db"))
    store.save([result])
    with store.connect() as conn:
        count, raw = conn.execute("SELECT group_article_count, related_articles FROM classified_news").fetchone()
    assert count == 2
    assert json.loads(raw)[0]["link"] == "https://example.com/other"


def test_group_columns_migrate_without_altering_existing_row(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(LEGACY_SCHEMA)
        conn.execute("INSERT INTO classified_news(keyword, title, source, link) VALUES ('AI', 'original', 'A', 'original-link')")
    store = SqliteNewsStore(str(path))
    with store.connect() as conn:
        row = conn.execute("SELECT title, link, group_article_count, related_articles FROM classified_news").fetchone()
    assert row == ("original", "original-link", 1, "[]")


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("final_label", ["금융/투자", "검토필요"])
def test_sqlite_saves_all_common_columns_in_new_and_legacy_tables(tmp_path, legacy, final_label):
    path = tmp_path / "all_columns.db"
    if legacy:
        with sqlite3.connect(path) as conn:
            conn.executescript(LEGACY_SCHEMA)
    result = rule_result()
    result = replace(
        result,
        item=replace(
            result.item, source="원문 매체", published_at="2026-10-04T10:00:00+09:00",
            content="기사 본문", related_articles=(
                NewsReference("관련 제목", "B", "2026-10-04", "https://example.com/other"),
            ),
        ),
        model_prediction=ModelPrediction(
            "기술개발", 0.851234, 0.801234,
            ["기술개발", "금융/투자", "기업동향"], [0.851234, 0.05, 0.02],
        ),
        rule_decision=replace(result.rule_decision, final_label=final_label),
        review_reclassification=ReviewReclassification(
            status="RECLASSIFIED", initial_model_category="기술개발",
            initial_model_score=0.3, initial_score_margin=0.01,
            initial_final_category="검토필요", initial_rule_reason="근거 없음",
            enrichment_status="BODY_EXTRACTED", http_status=200,
            resolved_url="https://example.com/news",
        ),
    )
    expected = result.to_row()
    assert list(expected) == CLASSIFIED_NEWS_COLUMNS
    store = SqliteNewsStore(str(path))
    store.save(iter([result]))
    with store.connect() as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(classified_news)")]
        stored = conn.execute(
            f"SELECT {', '.join(CLASSIFIED_NEWS_COLUMNS)} FROM classified_news"
        ).fetchone()
    if legacy:
        assert set(CLASSIFIED_NEWS_COLUMNS) <= set(columns)
    else:
        assert columns == ["id", *CLASSIFIED_NEWS_COLUMNS, "unique_key"]
    assert stored == tuple(expected[column] for column in CLASSIFIED_NEWS_COLUMNS)


def test_sqlite_keeps_replace_identity_and_generator_input(tmp_path):
    result = rule_result()
    store = SqliteNewsStore(str(tmp_path / "replace.db"))
    store.save(iter([result]))
    updated = replace(result, item=replace(result.item, link="https://example.com/updated"))
    store.save(iter([updated]))
    with store.connect() as conn:
        rows = conn.execute("SELECT title, link, unique_key FROM classified_news").fetchall()
    expected = updated.to_row()
    assert rows == [(
        expected["title"], expected["link"],
        f"{expected['keyword']}|{expected['title']}|{expected['source']}",
    )]
