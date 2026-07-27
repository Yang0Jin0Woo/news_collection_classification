from __future__ import annotations

import pandas as pd
import streamlit as st

from news_classifier.config import AppSettings
from news_classifier.storage.csv_store import CsvNewsStore


def render_dashboard(default_path: str = "news_analysis_results.csv") -> None:
    st.set_page_config(page_title="News Classifier Dashboard", layout="wide")
    st.title("뉴스 수집 및 분류 대시보드")
    path = st.text_input("CSV 경로", default_path)
    df = CsvNewsStore(path).load()
    if df.empty:
        st.info("데이터가 없습니다. CLI로 먼저 수집을 실행하세요.")
        return

    categories = sorted(df["final_category"].dropna().unique()) if "final_category" in df.columns else []
    selected = st.multiselect("카테고리", categories, default=categories)
    filtered = df[df["final_category"].isin(selected)] if selected else df

    metric_columns = st.columns(3)
    metric_columns[0].metric("전체 기사", len(filtered))
    rule_count = (
        int((filtered["decision_source"] == "RULE").sum())
        if "decision_source" in filtered.columns
        else 0
    )
    metric_columns[1].metric("규칙 보정", rule_count)
    review_count = (
        int(
            filtered["review_required"]
            .astype(str)
            .str.lower()
            .isin({"true", "1", "y"})
            .sum()
        )
        if "review_required" in filtered.columns
        else 0
    )
    metric_columns[2].metric("검토 필요", review_count)

    if "final_category" in filtered.columns:
        st.bar_chart(filtered.groupby("final_category").size())

    display_columns = [
        "title",
        "source",
        "model_category",
        "model_confidence",
        "model_confidence_level",
        "final_category",
        "decision_source",
        "final_decision_status",
        "review_required",
        "rule_reason",
        "published_at",
        "link",
    ]
    available_columns = [
        column for column in display_columns if column in filtered.columns
    ]
    st.dataframe(
        filtered[available_columns] if available_columns else filtered,
        use_container_width=True,
    )


if __name__ == "__main__":
    render_dashboard(AppSettings().output_csv)
