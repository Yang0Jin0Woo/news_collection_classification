from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path

from news_classifier.collectors.google_rss import GoogleNewsRssCollector
from news_classifier.config import AppSettings
from news_classifier.evaluation_dataset import event_id_from_title
from news_classifier.utils.http import HttpClient
from news_classifier.utils.text import strip_source_suffix


EVALUATION_QUERIES = [
    "AI 반도체",
    "인공지능",
    "배터리",
    "로봇",
    "OLED 디스플레이",
    "전력 에너지",
]
NEGATIVE_CONTROL_QUERIES = [
    "프로야구 경기",
    "연예 영화",
    "날씨 여행 축제",
]

FIELDNAMES = [
    "id",
    "event_id",
    "split",
    "keyword",
    "title",
    "description",
    "source",
    "published_at",
    "link",
    "gold_label",
    "review_status",
    "reviewed_by",
    "notes",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="실제 뉴스 수동 평가 후보를 수집합니다.")
    parser.add_argument("--output", default="data/evaluation/real_news_candidates.csv")
    parser.add_argument("--target-total", type=int, default=180)
    parser.add_argument("--negative-total", type=int, default=20)
    parser.add_argument("--per-query", type=int, default=40)
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
    seen_events: set[str] = set()

    def collect_queries(queries: list[str], target_count: int) -> None:
        if target_count <= 0:
            return
        start_count = len(rows)
        for query in queries:
            query_count = 0
            for item in collector.fetch(query, limit=args.per_query):
                title = strip_source_suffix(item.title, item.source)
                event_id = event_id_from_title(title)
                if event_id in seen_events:
                    continue
                seen_events.add(event_id)
                rows.append({
                    "id": _case_id(item.link, title, item.source),
                    "event_id": event_id,
                    "split": "",
                    "keyword": query,
                    "title": title,
                    "description": item.description,
                    "source": item.source,
                    "published_at": item.published_at,
                    "link": item.link,
                    "gold_label": "",
                    "review_status": "pending",
                    "reviewed_by": "",
                    "notes": "",
                })
                query_count += 1
                if len(rows) - start_count >= target_count:
                    break
            print(f"{query}: {query_count}건")
            if len(rows) - start_count >= target_count:
                break
            time.sleep(0.2)

    negative_target = min(args.negative_total, args.target_total)
    collect_queries(NEGATIVE_CONTROL_QUERIES, negative_target)
    collect_queries(
        EVALUATION_QUERIES,
        args.target_total - len(rows),
    )

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"수동 검토 후보 저장: {output_path} ({len(rows)}건)")


if __name__ == "__main__":
    main()
