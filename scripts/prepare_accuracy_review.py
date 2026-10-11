"""Prepare blind review CSV and check readiness without news/model requests."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from news_classifier.accuracy_review import REVIEW_COLUMNS, accuracy_review_readiness, prepare_accuracy_review
from news_classifier.evaluation_dataset import dataset_sha256, read_csv_dataset
from news_classifier.rules.default_rules import CANDIDATE_LABELS
from news_classifier.utils.json_reports import write_new_json


def build_parser():
    parser = argparse.ArgumentParser(description="실제 기사 정답 확인 자료 준비와 독립 평가 준비 검사")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="예측값을 제외한 새 정답 확인 CSV 생성")
    prepare.add_argument("--dataset", action="append", required=True)
    prepare.add_argument("--confirmed-json", action="append", default=[])
    prepare.add_argument("--output", required=True)
    check = commands.add_parser("check", help="정답·사건·주제·검색어 분리 검사, 모델 실행 없음")
    check.add_argument("--dataset", required=True)
    check.add_argument("--report", help="기존 파일을 덮어쓰지 않는 새 준비 검사 JSON")
    for command in (prepare, check):
        command.add_argument("--min-events-per-label", type=int, default=2)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.min_events_per_label < 1:
            raise ValueError("min-events-per-label은 양수 필요")
        if args.command == "prepare":
            output = Path(args.output)
            report_path = output.with_suffix(".readiness.json")
            inputs = [Path(path) for path in args.dataset + args.confirmed_json]
            if output.exists() or report_path.exists() or output.resolve() in {path.resolve() for path in inputs}:
                raise ValueError("원본과 기존 결과 보존: 새 --output 경로 지정 필요")
            candidates = [row for path in args.dataset for row in read_csv_dataset(Path(path))[1]]
            confirmed = []
            for path in args.confirmed_json:
                payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
                if (not isinstance(payload, dict) or payload.get("dataset_role") != "development"
                        or not isinstance(payload.get("cases"), list)):
                    raise ValueError("확정 JSON은 development 역할과 cases 목록 필요")
                confirmed.extend(payload["cases"])
            rows = prepare_accuracy_review(candidates, confirmed, CANDIDATE_LABELS)
            if not rows:
                raise ValueError("정답 확인 후보 없음")
            report = accuracy_review_readiness(rows, CANDIDATE_LABELS,
                                              minimum_events_per_label=args.min_events_per_label)
            report["source_datasets"] = [{"path": str(path), "sha256": dataset_sha256(path)} for path in inputs]
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", newline="", encoding="utf-8-sig") as file:
                writer = csv.DictWriter(file, fieldnames=REVIEW_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            report["dataset_sha256"] = dataset_sha256(output)
            write_new_json(report_path, report)
            print(f"정답 확인 CSV: {output} ({len(rows)}건)")
            print(f"평가 준비 검사: {report_path}")
        else:
            path = Path(args.dataset)
            _, rows = read_csv_dataset(path)
            report = accuracy_review_readiness(rows, CANDIDATE_LABELS,
                                              minimum_events_per_label=args.min_events_per_label)
            report.update(dataset=str(path), dataset_sha256=dataset_sha256(path))
            if args.report:
                write_new_json(args.report, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        print("독립 평가 준비 완료" if report["ready"] else "정답·사건 확인 대기: 실제 정확도 평가 미완료")
        if args.command == "check" and not report["ready"]:
            parser.exit(1)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
