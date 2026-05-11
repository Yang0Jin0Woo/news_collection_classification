from __future__ import annotations

import argparse
import logging

from news_classifier.config import AppSettings
from news_classifier.logging_config import configure_logging
from news_classifier.service import build_pipeline
from news_classifier.storage.csv_store import CsvNewsStore
from news_classifier.storage.sqlite_store import SqliteNewsStore
from news_classifier.reporting.summary_report import build_summary


def build_parser() -> argparse.ArgumentParser:
    
    parser = argparse.ArgumentParser(description="News collection and classification pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # 뉴스 수집 및 분류 실행 명령어 정의
    collect = sub.add_parser("collect", help="Collect and classify news")
    collect.add_argument("--keyword", required=True)          # 검색 키워드
    collect.add_argument("--limit", type=int, default=10)     # 수집할 뉴스 개수
    collect.add_argument("--csv", default=None)               # CSV 저장 경로
    collect.add_argument("--sqlite", default=None)            # SQLite 저장 경로
    collect.add_argument("--enrich-content", action="store_true")  # 기사 본문 보강 여부

    return parser


def main() -> None:
    
    # 로그 설정 및 기본 실행 설정 로드
    configure_logging(logging.INFO)
    settings = AppSettings()

    # 터미널에서 입력한 명령어와 옵션 파싱
    parser = build_parser()
    args = parser.parse_args()

    # collect 명령어 실행 시 뉴스 수집 및 분류 파이프라인 실행
    if args.command == "collect":
        pipeline = build_pipeline(settings)
        results = pipeline.run(
            keyword=args.keyword,
            limit=args.limit,
            enrich_content=args.enrich_content,
        )

        # 분류 결과 CSV 저장
        csv_path = args.csv or settings.output_csv
        CsvNewsStore(csv_path).save(results)
        print(f"CSV saved: {csv_path}")

        # SQLite 옵션이 있으면 DB에도 누적 저장
        if args.sqlite:
            SqliteNewsStore(args.sqlite).save(results)
            print(f"SQLite saved: {args.sqlite}")

        # 실행 결과 리포트 출력
        report = build_summary(results)
        print(report.to_markdown())


if __name__ == "__main__":
    main()