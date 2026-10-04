from __future__ import annotations

import pandas as pd
import streamlit as st
import json

from news_classifier.config import AppSettings
from news_classifier.storage.csv_store import CsvNewsStore


def original_article_count(df: pd.DataFrame) -> int:
    if "group_article_count" not in df.columns:
        return len(df)  # 기존 CSV도 그대로 조회 가능
    return int(pd.to_numeric(df["group_article_count"], errors="coerce").fillna(1).clip(lower=1).sum())


def grouped_article_details(df: pd.DataFrame) -> pd.DataFrame:
    """대표 및 묶인 원문 목록. 오래된 CSV와 잘못된 JSON에도 조회 중단 방지."""
    details = []
    for _, row in df.iterrows():
        raw = row.get("related_articles", "[]")
        try:
            references = json.loads(raw) if isinstance(raw, str) else []
        except (ValueError, TypeError):
            references = []
        if not isinstance(references, list) or not references:
            continue
        articles = [row.to_dict()] + [ref for ref in references if isinstance(ref, dict)]
        for article in articles:
            details.append({
                "대표 제목": row.get("title", ""),
                "기사 제목": article.get("title", ""),
                "언론사": article.get("source", ""),
                "발행일": article.get("published_at", ""),
                "원문": article.get("link", ""),
            })
    return pd.DataFrame(details)


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

    metric_columns = st.columns(4)
    metric_columns[0].metric("수집 기사", original_article_count(filtered))
    metric_columns[1].metric("대표 기사", len(filtered))
    rule_count = (
        int((filtered["decision_source"] == "RULE").sum())
        if "decision_source" in filtered.columns
        else 0
    )
    metric_columns[2].metric("규칙 보정", rule_count)
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
    metric_columns[3].metric("검토 필요", review_count)
    st.caption("선택한 카테고리 기준 수치이며, 카테고리 통계와 분류 판단은 대표 기사 기준입니다.")

    if "final_category" in filtered.columns:
        category_counts = (
            filtered.groupby("final_category")
            .size()
            .rename_axis("카테고리")
            .reset_index(name="대표 기사 수")
        )
        st.vega_lite_chart(
            category_counts,
            {
                "mark": {"type": "bar", "size": 24},
                "encoding": {
                    "x": {
                        "field": "카테고리",
                        "type": "nominal",
                        "sort": "-y",
                        "axis": {
                            "labelAngle": -30,
                            "labelOverlap": False,
                            "labelLimit": 140,
                        },
                    },
                    "y": {
                        "field": "대표 기사 수",
                        "type": "quantitative",
                        "axis": {"tickMinStep": 1},
                    },
                    "tooltip": [
                        {"field": "카테고리", "type": "nominal"},
                        {"field": "대표 기사 수", "type": "quantitative"},
                    ],
                },
            },
            width="stretch",
            height=320,
        )

    display_columns = [
        "title",
        "source",
        "group_article_count",
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
        width="stretch",
        column_config={"group_article_count": "묶음 기사 수"},
    )
    details = grouped_article_details(filtered)
    if not details.empty:
        with st.expander("묶인 기사와 원문 링크 보기"):
            st.caption("각 묶음의 대표 기사와 함께 수집된 다른 기사 목록입니다.")
            st.dataframe(details, width="stretch", hide_index=True,
                         column_config={"원문": st.column_config.LinkColumn("원문")})


if __name__ == "__main__":
    render_dashboard(AppSettings().output_csv)
