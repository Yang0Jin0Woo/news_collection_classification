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

    collect = sub.add_parser("collect", help="Collect and classify news")
    collect.add_argument("--keyword", required=True)
    collect.add_argument("--limit", type=int, default=10)
    collect.add_argument("--csv", default=None)
    collect.add_argument("--sqlite", default=None)
    collect.add_argument("--enrich-content", action="store_true")

    return parser


def main() -> None:
    configure_logging(logging.INFO)
    settings = AppSettings()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "collect":
        pipeline = build_pipeline(settings)
        results = pipeline.run(
            keyword=args.keyword,
            limit=args.limit,
            enrich_content=args.enrich_content,
        )

        csv_path = args.csv or settings.output_csv
        CsvNewsStore(csv_path).save(results)
        print(f"CSV saved: {csv_path}")

        if args.sqlite:
            SqliteNewsStore(args.sqlite).save(results)
            print(f"SQLite saved: {args.sqlite}")

        report = build_summary(results)
        print(report.to_markdown())


if __name__ == "__main__":
    main()
