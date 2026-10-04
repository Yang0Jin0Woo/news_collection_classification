from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path
from collections import deque

from news_classifier.collectors.base import CollectionError

from news_classifier.collectors.google_rss import GoogleNewsRssCollector
from news_classifier.collectors.article_scraper import ArticleScraper
from news_classifier.models import NewsItem
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
    "AI 연구 특허",
    "신제품 서비스 출시",
    "기업 인수 합병 협약",
    "공장 증설 양산",
    "법안 규제 시행",
    "기업 실적 공모",
    "시장 규모 수요",
    "노사 임금 교섭",
    "IT 취업 채용",
    "수출 통상 협상",
    "자동차",
    "의료 제약",
    "식품",
    "게임",
    "유통 물류",
    "건설",
    "농업",
    "관광",
    "교육 대학 입시",
    "취업 채용 직업 훈련",
    "사회 사건 사고 재난",
    "정치 정당 선거",
    "문화 공연 전시",
    "프로야구 경기",
    "축구 농구 대회",
    "연예 영화",
    "건강 질병 치료",
    "생활 여행 요리",
    "날씨 기후 환경",
]

FIELDNAMES = [
    "id",
    "event_id",
    "split",
    "keyword",
    "title",
    "description",
    "content",
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
    parser.add_argument("--per-query", type=int, default=40)
    parser.add_argument("--query", action="append", help="추가 검색어, 반복 지정 가능")
    parser.add_argument("--queries-only", action="store_true", help="지정한 --query만 수집, 새 검색어 평가 자료용")
    parser.add_argument("--enrich-content", action="store_true", help="선택적 기사 본문 보강 시도")
    parser.add_argument("--overwrite", action="store_true", help="기존 후보 파일 덮어쓰기 허용")
    return parser


def _case_id(link: str, title: str, source: str) -> str:
    raw = link or f"{title}|{source}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def collect_candidates(collector, queries, target_count, per_query, seen_events):
    """검색어별 결과를 순환 선택하여 앞 검색어만으로 표본이 채워지는 현상 방지."""
    if target_count <= 0:
        return []
    pools = []
    for query in queries:
        try:
            items = collector.fetch(query, limit=per_query)
        except CollectionError as exc:
            print(f"{query}: 수집 실패 ({exc})", flush=True)
            continue
        pools.append((query, deque(items)))
        print(f"{query}: 후보 {len(items)}건", flush=True)
        time.sleep(0.2)

    rows = []
    while len(rows) < target_count and any(pool for _, pool in pools):
        for query, pool in pools:
            while pool:
                item = pool.popleft()
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
                    "content": item.content,
                    "source": item.source,
                    "published_at": item.published_at,
                    "link": item.link,
                    "gold_label": "",
                    "review_status": "pending",
                    "reviewed_by": "",
                    "notes": "",
                })
                break
            if len(rows) >= target_count:
                break
    return rows


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.target_total < 1 or args.per_query < 1:
        parser.error("target-total 및 per-query는 양수 필요")
    if args.queries_only and not args.query:
        parser.error("--queries-only에는 --query 지정 필요")
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        parser.error("기존 후보 파일 보존: 새 --output 경로 또는 --overwrite 지정 필요")
    settings = AppSettings()
    collector = GoogleNewsRssCollector(
        HttpClient(settings.headers, settings.request_timeout_seconds)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    seen_events: set[str] = set()

    queries = list(dict.fromkeys(
        (args.query or []) if args.queries_only else EVALUATION_QUERIES + (args.query or [])
    ))
    rows.extend(collect_candidates(
        collector, queries, args.target_total, args.per_query, seen_events
    ))

    if not rows:
        parser.exit(1, "수집된 후보 없음: 파일 저장 생략\n")

    if args.enrich_content:
        scraper = ArticleScraper(HttpClient(settings.headers, settings.request_timeout_seconds))
        for row in rows:
            row["content"] = scraper.enrich(NewsItem(
                keyword=row["keyword"], title=row["title"], link=row["link"],
                source=row["source"], description=row["description"],
            )).content

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"수동 검토 후보 저장: {output_path} ({len(rows)}건)")


if __name__ == "__main__":
    main()
