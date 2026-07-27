from __future__ import annotations

import argparse
import logging
from pathlib import Path
import subprocess
import sys

from news_classifier.config import AppSettings
from news_classifier.logging_config import configure_logging
from news_classifier.models import PipelineStatus
from news_classifier.storage.csv_store import CsvNewsStore
from news_classifier.storage.sqlite_store import SqliteNewsStore
from news_classifier.reporting.summary_report import build_summary
from news_classifier.utils.validation import (
    ValidationError,
    require_non_blank,
    validate_limit,
)


def parse_limit(value: str) -> int:
    try:
        limit = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("limit must be an integer") from exc

    try:
        return validate_limit(limit)
    except ValidationError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    
    parser = argparse.ArgumentParser(description="News collection and classification pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # 뉴스 수집 및 분류 실행 명령어 정의
    collect = sub.add_parser("collect", help="Collect and classify news")
    collect.add_argument("--keyword", default=None)           # 검색 키워드
    collect.add_argument("--limit", type=parse_limit, default=10)  # 수집할 뉴스 개수
    collect.add_argument("--csv", default=None)               # CSV 저장 경로
    collect.add_argument(
        "--sqlite",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="Save to SQLite; omit PATH to use NEWS_OUTPUT_DB",
    )
    collect.add_argument("--enrich-content", action="store_true")  # 기사 본문 보강 여부

    sub.add_parser("dashboard", help="Open the Streamlit dashboard")

    return parser


def resolve_keyword(raw_keyword: str | None) -> str:
    if raw_keyword is None:
        raw_keyword = input("검색 키워드를 입력하세요: ")
    return require_non_blank(raw_keyword, "keyword")


def run_dashboard() -> int:
    dashboard_path = Path(__file__).with_name("dashboard_streamlit.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
    ]
    try:
        return subprocess.run(command, check=False).returncode
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        print(f"대시보드 실행 실패: {exc}", file=sys.stderr)
        return 1


def run_cli(argv: list[str] | None = None) -> int:
    # 로그 설정 및 기본 실행 설정 로드
    configure_logging(logging.INFO)
    settings = AppSettings()

    # 터미널에서 입력한 명령어와 옵션 파싱
    parser = build_parser()
    args = parser.parse_args(argv)

    # collect 명령어 실행 시 뉴스 수집 및 분류 파이프라인 실행
    if args.command == "collect":
        try:
            keyword = resolve_keyword(args.keyword)
        except ValidationError as exc:
            parser.error(str(exc))

        from news_classifier.service import build_pipeline

        pipeline = build_pipeline(settings)
        pipeline_result = pipeline.run(
            keyword=keyword,
            limit=args.limit,
            enrich_content=args.enrich_content,
        )

        print(f"Pipeline status: {pipeline_result.status.value}")
        if not pipeline_result.succeeded:
            for error in pipeline_result.errors:
                print(
                    f"[{error.stage}/{error.code}] {error.message}",
                    file=sys.stderr,
                )
            print("실패한 실행 결과는 저장하지 않았습니다.", file=sys.stderr)
            return 1

        # 분류 결과 CSV 저장
        csv_path = args.csv or settings.output_csv
        try:
            CsvNewsStore(csv_path).save(pipeline_result.results)
        except Exception as exc:
            print(f"CSV 저장 실패: {exc}", file=sys.stderr)
            return 1

        if pipeline_result.status == PipelineStatus.NO_RESULTS:
            print(f"CSV saved with headers only: {csv_path}")
        else:
            print(f"CSV saved: {csv_path}")

        # SQLite 옵션이 있으면 DB에도 누적 저장
        if args.sqlite is not None and pipeline_result.results:
            sqlite_path = args.sqlite or settings.output_db
            try:
                SqliteNewsStore(sqlite_path).save(pipeline_result.results)
            except Exception as exc:
                print(f"SQLite 저장 실패: {exc}", file=sys.stderr)
                return 1
            print(f"SQLite saved: {sqlite_path}")
        elif args.sqlite is not None:
            print("SQLite 저장 생략: 저장할 뉴스가 없습니다.")

        # 실행 결과 리포트 출력
        report = build_summary(pipeline_result.results)
        print(report.to_markdown())
        return 0

    if args.command == "dashboard":
        return run_dashboard()

    return 1


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
