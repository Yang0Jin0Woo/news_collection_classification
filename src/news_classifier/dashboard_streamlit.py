from __future__ import annotations

import pandas as pd
import streamlit as st

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

    st.metric("전체 기사", len(filtered))
    if "final_category" in filtered.columns:
        st.bar_chart(filtered.groupby("final_category").size())
    st.dataframe(filtered, use_container_width=True)


if __name__ == "__main__":
    render_dashboard()
