import json

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from news_classifier.dashboard_streamlit import grouped_article_details, original_article_count
from news_classifier.storage.csv_store import CsvNewsStore


def dashboard_app():
    from news_classifier.dashboard_streamlit import render_dashboard
    render_dashboard()


def dashboard_data(grouped):
    df = pd.DataFrame({
        "title": ["대표 기사", "별도 기사"],
        "source": ["A", "B"],
        "final_category": ["기술개발", "검토필요"],
        "decision_source": ["MODEL", "REVIEW"],
        "review_required": [False, True],
        "link": ["https://example.com/1", "https://example.com/2"],
    })
    if grouped:
        df["group_article_count"] = [3, 1]
        df["related_articles"] = [json.dumps([
            {"title": "관련 1", "source": "C", "link": "https://example.com/3"},
            {"title": "관련 2", "source": "D", "link": "https://example.com/4"},
        ]), "[]"]
    return df


@pytest.mark.parametrize("grouped", [False, True])
def test_dashboard_displays_representatives_and_reads_legacy_csv(monkeypatch, grouped):
    df = dashboard_data(grouped)
    monkeypatch.setattr(CsvNewsStore, "load", lambda _: df.copy())
    app = AppTest.from_function(dashboard_app).run()
    assert not app.exception
    assert [metric.value for metric in app.metric] == ["4" if grouped else "2", "2", "0", "1"]
    assert len(app.dataframe[0].value) == 2
    if grouped:
        assert len(app.dataframe[1].value) == 3
        assert app.dataframe[1].value["원문"].tolist() == [
            "https://example.com/1", "https://example.com/3", "https://example.com/4"
        ]
    app.multiselect[0].set_value(["기술개발"]).run()
    assert not app.exception
    assert app.metric[0].value == ("3" if grouped else "1")
    assert app.metric[1].value == "1"


def test_details_handles_invalid_or_missing_group_metadata():
    df = pd.DataFrame({"title": ["A", "B", "C"], "related_articles": ["bad-json", "{}", None]})
    assert grouped_article_details(df).empty
    assert grouped_article_details(pd.DataFrame({"title": ["old"]})).empty
    assert original_article_count(pd.DataFrame({"group_article_count": [3, None, "bad", 0]})) == 6
