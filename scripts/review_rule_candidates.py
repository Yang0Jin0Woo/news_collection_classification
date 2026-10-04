"""실제 기사에서 규칙 후보의 출현과 기존 규칙 중복을 점검. 자동 적용 금지."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from news_classifier.classifiers.rule_engine import _has_phrase
from news_classifier.evaluation_dataset import dataset_sha256
from news_classifier.rules.default_rules import DEFAULT_RULE_SET
from news_classifier.rules.policy import normalize_rule_text


# 수집 기사에서 확인할 구체적인 표현 후보. 정답 라벨이나 확정된 새 규칙이 아님.
# 기타/무관은 직접 키워드 규칙을 추가하지 않는 기존 구조 유지.
CANDIDATE_PHRASES = {
    "기술개발": ("특허 출원", "예측 모델"),
    "제품/서비스": ("렌탈 서비스", "서비스 시작"),
    "기업동향": ("지원 협약", "인수 추진"),
    "생산/공급망": ("양산 돌입", "생산시설 증설"),
    "정책/규제": ("국회 통과", "윤리기준"),
    "금융/투자": ("공모주 청약", "공모가"),
    "시장/산업": ("시장 전망", "시장 규모"),
    "노동/노사": ("취업박람회", "공개 채용", "신입채용"),
    "국제/통상": ("통상 협상", "무역협정"),
}


def audit_candidate(phrase, label, rows, rule_set=DEFAULT_RULE_SET):
    if label not in rule_set.candidate_labels or rule_set.label_policy(label).no_direct_rules:
        raise ValueError("candidate must use an existing direct-rule category")
    normalized = normalize_rule_text(phrase)
    if not normalized:
        raise ValueError("candidate phrase must not be empty")
    alias = normalized.replace(" ", "")
    existing = []
    overlaps = []
    for policy in rule_set.direct_rule_labels:
        for term in policy.terms:
            term_text = normalize_rule_text(term.phrase)
            if term_text.replace(" ", "") == alias:
                existing.append({"category": policy.label, "phrase": term.phrase})
            elif _has_phrase(normalized, term_text) or _has_phrase(term_text, normalized):
                overlaps.append({"category": policy.label, "phrase": term.phrase})
    matches = [
        {"id": row["id"], "event_id": row.get("event_id", ""),
         "title": row["title"], "link": row.get("link", "")}
        for row in rows
        if _has_phrase(normalize_rule_text(f"{row['title']} {row.get('description', '')}"), phrase)
    ]
    return {
        "phrase": phrase,
        "proposed_category": label,
        "status": "existing_or_conflicting_rule" if existing else "pending_human_review",
        "existing_aliases": existing,
        "overlapping_rules": overlaps,
        "matching_article_count": len(matches),
        "matches": matches,
        "human_confirmation_required": True,
    }


def main():
    parser = argparse.ArgumentParser(description="규칙 후보의 중복과 출현 기사 점검")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dataset = Path(args.dataset)
    output = Path(args.output)
    if output.exists():
        parser.error("기존 결과 보존: 새 --output 경로 지정 필요")
    with dataset.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        parser.error("기사 후보가 없는 데이터")
    if any(row.get("split", "").strip() == "evaluation" for row in rows):
        parser.error("평가용 기사로 규칙 후보를 선정할 수 없음")
    payload = {
        "dataset": str(dataset),
        "dataset_sha256": dataset_sha256(dataset),
        "note": (
            "주제와 표현은 제안이며 정답 아님. 같은 사건의 다른 제목을 사람이 묶은 뒤 "
            "development 정답과 모델 오류 확인 필요. evaluation 기사는 후보 선정에 사용 금지."
        ),
        "candidates": [
            audit_candidate(phrase, label, rows)
            for label, phrases in CANDIDATE_PHRASES.items()
            for phrase in phrases
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"규칙 후보 점검 저장: {output}")


if __name__ == "__main__":
    main()
