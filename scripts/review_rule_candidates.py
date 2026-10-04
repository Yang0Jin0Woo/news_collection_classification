"""실제 기사에서 규칙 후보의 출현과 기존 규칙 중복을 점검. 자동 적용 금지."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from news_classifier.evaluation_dataset import dataset_sha256
from news_classifier.rules.default_rules import DEFAULT_RULE_SET
from news_classifier.rules.event_candidates import (
    CANDIDATE_PHRASES,
    EVENT_CANDIDATE_POLICY,
    HUMAN_REVIEW_CHECKLIST,
    article_review_records,
    audit_candidate as _audit_candidate,
)
from news_classifier.rules.policy import normalize_rule_text


def audit_candidate(phrase, label, rows, rule_set=DEFAULT_RULE_SET, *, candidate_phrases=None):
    return _audit_candidate(
        phrase, label, rows, rule_set, candidate_phrases=candidate_phrases,
    )


def parse_candidate(value):
    label, separator, phrase = value.partition(":")
    label, phrase = label.strip(), phrase.strip()
    if (
        not separator
        or label not in DEFAULT_RULE_SET.candidate_labels
        or label == DEFAULT_RULE_SET.other_label
        or not normalize_rule_text(phrase)
    ):
        raise argparse.ArgumentTypeError(
            "--candidate는 '카테고리:표현' 형식이며 기타/무관은 사용할 수 없습니다"
        )
    return label, phrase


def main():
    parser = argparse.ArgumentParser(
        description="16개 일반 뉴스 주제의 국문/영문 사건 표현 후보 감사. 정답 확정 및 규칙 자동 적용 없음",
        epilog="후보 감사 → 원문과 독립 사건 확인 → development 정답/기존 오류 확인 → 별도 evaluation 검증 후 활성화",
    )
    parser.add_argument("--dataset", required=True, help="development 또는 미지정 기사 후보 CSV. evaluation 사용 금지")
    parser.add_argument("--output", required=True, help="새 감사 JSON 경로. 원본 CSV 및 기존 JSON 덮어쓰기 금지")
    parser.add_argument(
        "--candidate", action="append", type=parse_candidate, default=[],
        metavar="CATEGORY:PHRASE",
        help="검토할 표현을 명시(반복 가능). 지정하면 기본 후보 대신 이 표현만 감사",
    )
    args = parser.parse_args()
    dataset = Path(args.dataset)
    output = Path(args.output)
    if output.exists():
        parser.error("기존 결과 보존: 새 --output 경로 지정 필요")
    with dataset.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        parser.error("기사 후보가 없는 데이터")
    selected = args.candidate or [
        (label, phrase)
        for label, phrases in CANDIDATE_PHRASES.items()
        for phrase in phrases
    ]
    catalog = {label: list(phrases) for label, phrases in CANDIDATE_PHRASES.items()}
    for label, phrase in selected:
        if phrase not in catalog.setdefault(label, []):
            catalog[label].append(phrase)
    try:
        candidates = [
            audit_candidate(phrase, label, rows, candidate_phrases=catalog)
            for label, phrase in selected
        ]
    except ValueError as exc:
        parser.error(str(exc))
    payload = {
        "schema_version": 2,
        "candidate_policy": EVENT_CANDIDATE_POLICY,
        "dataset": str(dataset),
        "dataset_sha256": dataset_sha256(dataset),
        "query_independent": True,
        "source_modified": False,
        "automatic_activation": False,
        "covered_categories": list(CANDIDATE_PHRASES),
        "model_only_categories": [DEFAULT_RULE_SET.other_label],
        "minimum_independent_development_events": DEFAULT_RULE_SET.minimum_development_evidence_events,
        "human_review_checklist": list(HUMAN_REVIEW_CHECKLIST),
        "source_keywords": sorted({row.get("keyword", "").strip() for row in rows if row.get("keyword", "").strip()}),
        "source_scope_note": "이 보고서는 입력 CSV에 저장된 검색어/기사 범위만 감사하며 모든 검색어에서의 성능을 검증한 결과가 아님",
        "articles_for_human_confirmation": article_review_records(rows),
        "note": (
            "주제와 표현은 제안이며 정답 아님. 같은 사건의 다른 제목을 사람이 묶은 뒤 "
            "development 정답과 기존 최종 판단의 오류/검토 확인 필요. evaluation 기사는 후보 선정에 사용 금지. "
            "제목/설명/본문의 표현 출현은 주제 정답의 증거가 아니며 자동 기사 ID는 독립 사건 ID가 아님."
        ),
        "candidates": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # exists() 확인 후에도 다른 작업이 만들었으면 보존하는 배타적 생성.
    try:
        with output.open("x", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
    except FileExistsError:
        parser.error("기존 결과 보존: 새 --output 경로 지정 필요")
    print(f"규칙 후보 점검 저장: {output}")


if __name__ == "__main__":
    main()
