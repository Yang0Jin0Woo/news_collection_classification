from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path

from news_classifier.collectors.google_rss import GoogleNewsRssCollector
from news_classifier.config import AppSettings
from news_classifier.utils.http import HttpClient
from news_classifier.utils.text import normalize_key, strip_source_suffix


QUERIES_BY_LABEL = {
    "기술개발": ["AI 기술 개발 연구", "반도체 기술 연구 개발", "배터리 기술 개발"],
    "제품/서비스": ["신제품 출시 서비스", "AI 서비스 출시", "로봇 제품 출시"],
    "기업동향": ["기업 인수 합병 제휴", "기업 수주 계약", "기업 조직 채용"],
    "생산/공급망": ["공장 생산 공급망", "반도체 생산 증설", "배터리 소재 공급망"],
    "정책/규제": ["정부 정책 규제", "AI 법안 규제", "배터리 보조금 정책"],
    "금융/투자": ["주가 투자 실적", "기업 상장 공모", "증권 목표가 실적"],
    "시장/산업": ["시장 전망 수요", "산업 성장 점유율", "시장 가격 경쟁"],
    "노동/노사": ["노조 파업 임금", "노사 교섭 고용", "근로자 노동조합"],
    "국제/통상": ["수출 관세 통상", "미국 중국 무역 제재", "수출통제 협상"],
}

FIELDNAMES = [
    "id",
    "split",
    "keyword",
    "title",
    "description",
    "source",
    "published_at",
    "link",
    "suggested_label",
    "gold_label",
    "review_status",
    "reviewed_by",
    "notes",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="실제 뉴스 수동 평가 후보를 수집합니다.")
    parser.add_argument("--output", default="data/evaluation/real_news_candidates.csv")
    parser.add_argument("--target-per-label", type=int, default=20)
    parser.add_argument("--per-query", type=int, default=20)
    return parser


def _case_id(link: str, title: str, source: str) -> str:
    raw = link or f"{title}|{source}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    args = build_parser().parse_args()
    settings = AppSettings()
    collector = GoogleNewsRssCollector(
        HttpClient(settings.headers, settings.request_timeout_seconds)
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for suggested_label, queries in QUERIES_BY_LABEL.items():
        label_rows: list[dict[str, str]] = []
        for query in queries:
            for item in collector.fetch(query, limit=args.per_query):
                title = strip_source_suffix(item.title, item.source)
                key = (normalize_key(title), normalize_key(item.source))
                if key in seen:
                    continue
                seen.add(key)
                label_rows.append({
                    "id": _case_id(item.link, title, item.source),
                    "split": "",
                    "keyword": query,
                    "title": title,
                    "description": item.description,
                    "source": item.source,
                    "published_at": item.published_at,
                    "link": item.link,
                    "suggested_label": suggested_label,
                    "gold_label": "",
                    "review_status": "pending",
                    "reviewed_by": "",
                    "notes": "",
                })
                if len(label_rows) >= args.target_per_label:
                    break
            if len(label_rows) >= args.target_per_label:
                break
            time.sleep(0.2)
        rows.extend(label_rows)
        print(f"{suggested_label}: {len(label_rows)}건")

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"수동 검토 후보 저장: {output_path} ({len(rows)}건)")


if __name__ == "__main__":
    main()
