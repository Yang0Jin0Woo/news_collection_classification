"""검색어와 무관한 사건 표현 후보. 정답이나 활성화된 규칙이 아님."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib

from news_classifier.classifiers.rule_engine import _has_phrase, _phrase_spans
from news_classifier.rules.policy import RuleSet, normalize_rule_text
from news_classifier.utils.text import article_description, clean_text, strip_source_suffix


EVENT_CANDIDATE_POLICY = "general-event-candidates-v1-human-review"

# 주제의 주요 사건을 나타내는 국문/영문 검증 후보. 회사명/검색어 분기는 없음.
# 기타/무관에는 직접 규칙을 만들지 않음. 출현만으로 정답을 확정하지 않음.
CANDIDATE_PHRASES: dict[str, tuple[str, ...]] = {
    "기술개발": ("특허 출원", "예측 모델", "연구 결과", "patent application", "research findings"),
    "제품/서비스": ("신제품 출시", "서비스 개시", "렌탈 서비스", "서비스 시작", "product launch", "service launch"),
    "기업동향": ("인수 합병", "경영진 교체", "지원 협약", "인수 추진", "merger agreement", "executive appointment"),
    "생산/공급망": ("공장 증설", "양산 시작", "양산 돌입", "생산시설 증설", "증산 계획", "생산 확대", "공급 중단", "factory expansion", "mass production", "production increase", "supply disruption"),
    "정책/규제": ("법안 통과", "규제 시행", "국회 통과", "윤리기준", "bill passed", "regulation takes effect"),
    "금융/투자": ("기업공개", "공모가", "공모주 청약", "IPO", "initial public offering"),
    "시장/산업": ("시장 규모", "산업 전망", "시장 전망", "정제마진 상승", "수요 증가", "market size", "industry outlook", "refining margins", "demand growth"),
    "노동/노사": ("직원 해고", "임금 교섭", "employee layoffs", "wage negotiations"),
    "국제/통상": ("무역 협정", "통상 협상", "무역협정", "수출 중단", "수출 제한", "trade agreement", "trade negotiations", "export suspension", "export restrictions"),
    "교육/취업": ("신입생 모집", "채용 공고", "취업박람회", "공개 채용", "신입채용", "student admissions", "job vacancies"),
    "사회": ("사건 수사", "교통 사고", "담합 재판", "형사 재판", "criminal investigation", "traffic accident", "criminal trial"),
    "정치": ("대통령 선거", "정당 공천", "presidential election", "party nomination"),
    "문화/연예": ("영화 개봉", "공연 개최", "film premiere", "concert performance"),
    "스포츠": ("리그 우승", "경기 결과", "league championship", "match result"),
    "건강/의료": ("질병 치료", "건강 검진", "disease treatment", "health screening"),
    "생활/환경": ("기상 예보", "환경 보호", "weather forecast", "environmental protection"),
}

CONTEXT_WARNING_PHRASES: dict[str, tuple[str, ...]] = {
    "cancellation": (
        "취소", "철회", "중단", "보류", "cancelled", "canceled", "cancellation",
        "scraps", "scrapped", "withdrawn", "abandoned",
    ),
    "denial": ("부인", "사실무근", "아니다", "않았다", "denies", "denied", "not", "no plans"),
    "comparison": ("비교", "대비", "versus", "vs", "compared", "comparison"),
}

HUMAN_REVIEW_CHECKLIST = (
    "후보 표현은 정답 라벨이 아님. 기사 주요 사건을 읽고 gold_label을 직접 확인",
    "동일 사건의 유사 기사는 같은 event_id로 묶고 서로 다른 사건 여부를 직접 확인",
    "review_status=confirmed, reviewed_by, split=development를 확인",
    "기존 판정이 틀렸거나 검토필요였던 서로 다른 사건의 확인된 근거 확보",
    "기존 점수 규칙과 문맥 전용 표현의 중복 및 다른 주제와의 충돌 확인",
    "취소, 부인, 비교 문맥에서 표현이 실제 주요 사건을 나타내는지 확인",
    "규칙 선정에 쓰지 않은 evaluation 사건과 검색어로 오분류 및 검토 비율 비교",
)


def article_review_id(row: Mapping[str, str]) -> tuple[str, str]:
    """검토 편의용 기사 식별자만 생성. 같은 사건의 독립성이나 event_id는 추론하지 않음."""
    supplied = str(row.get("id", "") or "").strip()
    if supplied:
        return supplied, "provided_article_id_not_verified_event"
    link = str(row.get("link", "") or "").strip()
    title = str(row.get("title", "") or "").strip()
    source = str(row.get("source", "") or "").strip()
    reference = link or f"{title}|{source}"
    digest = hashlib.sha256(reference.encode("utf-8")).hexdigest()[:16]
    return f"article-{digest}", "stable_article_reference_not_event"


def article_review_records(rows: Sequence[Mapping[str, str]]) -> list[dict]:
    """원본 예측과 후보 제안으로부터 분리한 사람이 채울 확인 양식. 원본 변경 없음."""
    records = []
    for row in rows:
        article_id, id_basis = article_review_id(row)
        fields = article_fields(row)
        records.append({
            "article_id": article_id, "article_id_basis": id_basis,
            "keyword": row.get("keyword", ""), "source": row.get("source", ""),
            "title": row.get("title", ""), "description": fields["description"],
            "content": fields["content"], "content_available": bool(fields["content"]),
            "link": row.get("link", ""),
            "provided_human_metadata": {
                key: row.get(key, "")
                for key in ("event_id", "gold_label", "split", "review_status", "reviewed_by")
            },
            "human_confirmation": {
                "event_id": "", "gold_label": "", "split": "", "review_status": "pending",
                "reviewed_by": "", "independent_event_check": "", "notes": "",
            },
            "stored_prediction_not_gold": {
                key: row.get(key, "")
                for key in ("model_category", "final_category", "rule_reason")
            },
        })
    return records


def _development_rows_only(rows: Sequence[Mapping[str, str]]) -> None:
    for row in rows:
        split = str(row.get("split", "") or "").strip().casefold()
        if split not in {"", "development"}:
            raise ValueError("평가용 기사로 규칙 후보를 선정할 수 없음: development 또는 미지정 후보만 허용")


def article_fields(row: Mapping[str, str]) -> dict[str, str]:
    """원본 필드를 변경하지 않고 기사 내용만 반환. 검색어/언론사명은 근거 제외."""
    raw_title = str(row.get("title", "") or "")
    source = str(row.get("source", "") or "")
    return {
        "title": clean_text(strip_source_suffix(raw_title, source)),
        "description": article_description(raw_title, str(row.get("description", "") or ""), source),
        "content": clean_text(str(row.get("content", "") or "")),
    }


def phrase_evidence(phrase: str, fields: Mapping[str, str]) -> list[dict]:
    """구간은 NFKC/공백 정리/casefold된 필드 기준. 원본 구간이라고 오인하지 않도록 명시."""
    evidence = []
    for field, text in fields.items():
        normalized = normalize_rule_text(text)
        for start, end in _phrase_spans(normalized, phrase):
            evidence.append({
                "field": field,
                "start": start,
                "end": end,
                "matched_text": normalized[start:end],
                "excerpt": normalized[max(0, start - 80):end + 80],
                "offset_basis": "normalized_nfkc_clean_casefold_field",
            })
    return evidence


def context_warnings(fields: Mapping[str, str], evidence: Sequence[Mapping]) -> list[dict]:
    """근거와 가까운 문맥 표현은 사람 확인용 경고일 뿐 정답/규칙 차단 판정이 아님."""
    warnings = []
    seen = set()
    for item in evidence:
        field = item["field"]
        text = normalize_rule_text(fields[field])
        for kind, phrases in CONTEXT_WARNING_PHRASES.items():
            for phrase in phrases:
                for start, end in _phrase_spans(text, phrase):
                    if end < item["start"] - 120 or start > item["end"] + 120:
                        continue
                    key = (field, kind, start, end)
                    if key in seen:
                        continue
                    seen.add(key)
                    warnings.append({
                        "kind": kind, "field": field, "matched_text": text[start:end],
                        "start": start, "end": end,
                        "offset_basis": "normalized_nfkc_clean_casefold_field",
                        "status": "human_context_check_required",
                    })
    return warnings


def audit_candidate(
    phrase: str,
    label: str,
    rows: Sequence[Mapping[str, str]],
    rule_set: RuleSet,
    *,
    candidate_phrases: Mapping[str, Sequence[str]] | None = None,
) -> dict:
    """출현/중복/충돌 감사. 규칙 활성화나 정답 라벨 생성은 수행하지 않음."""
    _development_rows_only(rows)
    if label not in rule_set.candidate_labels or label == rule_set.other_label:
        raise ValueError("candidate must use an existing non-other category")
    normalized = normalize_rule_text(phrase)
    if not normalized:
        raise ValueError("candidate phrase must not be empty")
    alias = normalized.replace(" ", "")
    existing = []
    overlaps = []
    term_audit = []
    for policy in rule_set.labels:
        for term in policy.terms:
            term_text = normalize_rule_text(term.phrase)
            relation = None
            if term_text.replace(" ", "") == alias:
                existing.append({"category": policy.label, "phrase": term.phrase})
                relation = "spacing_alias"
            elif _has_phrase(normalized, term_text) or _has_phrase(term_text, normalized):
                overlaps.append({"category": policy.label, "phrase": term.phrase})
                relation = "phrase_overlap"
            if relation:
                term_audit.append({
                    "category": policy.label, "phrase": term.phrase, "relation": relation,
                    "scoring": not term.context_only,
                    "context_only": term.context_only,
                    "strength": int(term.strength), "origin": term.origin,
                    "cross_category": policy.label != label,
                })

    catalog = candidate_phrases if candidate_phrases is not None else CANDIDATE_PHRASES
    candidate_conflicts = []
    for other_label, phrases in catalog.items():
        if other_label == label:
            continue
        for other_phrase in phrases:
            other_normalized = normalize_rule_text(other_phrase)
            if (
                other_normalized.replace(" ", "") == alias
                or _has_phrase(normalized, other_normalized)
                or _has_phrase(other_normalized, normalized)
            ):
                candidate_conflicts.append({"category": other_label, "phrase": other_phrase})

    matches = []
    matched_rows = []
    for row in rows:
        fields = article_fields(row)
        evidence = phrase_evidence(phrase, fields)
        if not evidence:
            continue
        matched_rows.append(row)
        other_mentions = []
        for other_label, phrases in catalog.items():
            if other_label == label:
                continue
            present = [other_phrase for other_phrase in phrases if phrase_evidence(other_phrase, fields)]
            if present:
                other_mentions.append({"category": other_label, "phrases": present})
        matches.append({
            "id": article_review_id(row)[0], "article_id_basis": article_review_id(row)[1],
            "event_id": row.get("event_id", ""),
            "title": row.get("title", ""), "link": row.get("link", ""),
            "keyword": row.get("keyword", ""), "source": row.get("source", ""),
            "content_available": bool(fields["content"]),
            "match_evidence": evidence,
            "context_warnings": context_warnings(fields, evidence),
            "other_candidate_topic_mentions": other_mentions,
            "topic_mentions_are_gold_labels": False,
            "association_status": "phrase_present_in_title_description_or_body_not_verified_subject",
        })

    confirmed_rows = [row for row in matched_rows if (
        str(row.get("split", "") or "").strip().casefold() == "development"
        and str(row.get("review_status", "") or "").strip().casefold() == "confirmed"
        and str(row.get("gold_label", "") or "").strip() == label
        and str(row.get("reviewed_by", "") or "").strip()
        and str(row.get("event_id", "") or "").strip()
        and not str(row.get("suggested_label", "") or "").strip()
    )]
    provided_event_ids = sorted({str(row["event_id"]).strip().casefold() for row in confirmed_rows})
    return {
        "phrase": phrase, "proposed_category": label,
        "status": "existing_or_conflicting_rule" if existing else "pending_human_review",
        "existing_aliases": existing, "overlapping_rules": overlaps,
        "existing_term_audit": term_audit,
        "scoring_conflicts": [term for term in term_audit if term["scoring"] and term["cross_category"]],
        "context_only_overlaps": [term for term in term_audit if term["context_only"]],
        "candidate_conflicts": candidate_conflicts,
        "matching_article_count": len(matches), "matches": matches,
        "human_confirmation_required": True,
        "activation_eligible": False,
        "development_evidence": {
            "confirmed_row_count": len(confirmed_rows),
            "provided_event_ids": provided_event_ids,
            "provided_distinct_event_id_count": len(provided_event_ids),
            "independent_event_count": None,
            "required_independent_event_count": rule_set.minimum_development_evidence_events,
            "independent_events_verified": False,
            "baseline_errors_verified": False,
            "note": "event_id 개수는 독립 사건의 증명이 아님. 원문과 기존 오류를 사람이 확인한 뒤 별도 활성화 필요",
        },
    }
