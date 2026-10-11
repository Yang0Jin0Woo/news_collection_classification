from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from difflib import SequenceMatcher
import re
import unicodedata

from news_classifier.models import NewsItem, NewsReference
from news_classifier.utils.text import normalize_key, strip_source_suffix


_WORDS = re.compile(r"[a-z0-9가-힣]+")
_NUMBERS = re.compile(r"\d+(?:[.,]\d+)*")
_NEWS_TAG = re.compile(r"^\s*\[(?:단독|종합|속보|포토|영상)\]\s*")
_WORD_ALIASES = {
    "키운다": "양성", "양성한다": "양성", "육성": "양성", "육성한다": "양성",
}
_FILLER_WORDS = {"본격", "정식", "등", "및"}
_PARTICLES = {"이", "가", "은", "는", "을", "를", "의", "에", "에서", "으로", "과", "와"}
_CONFLICT_CUES = (
    "취소", "철회", "중단", "부인", "무산", "연기", "실패", "폐쇄", "해체",
    "불허", "감소", "증가", "축소", "확대", "인상", "인하", "승인", "거부",
    "반박", "재개", "종료",
    "않", "아니", "없", "불가", "해제",
)


def _tokens(text: str) -> tuple[str, ...]:
    text = unicodedata.normalize("NFKC", _NEWS_TAG.sub("", text)).casefold()
    return tuple(_WORD_ALIASES.get(word, word) for word in _WORDS.findall(text))


def _similarity(left: str, right: str) -> float:
    # SequenceMatcher 점수는 비교 순서에 따라 달라질 수 있어 낮은 쪽 사용.
    return min(
        SequenceMatcher(None, left, right, autojunk=False).ratio(),
        SequenceMatcher(None, right, left, autojunk=False).ratio(),
    )


def _published_time(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _compatible_words(left_set: frozenset[str], right_set: frozenset[str]) -> bool:
    """새 기관명, 지역, 사건 단어의 차이를 단순 유사도로 무시하지 않음."""

    def equivalent(word: str, others: frozenset[str]) -> bool:
        return any(
            word == other
            or (len(other) >= 3 and word.startswith(other) and word[len(other):] in _PARTICLES)
            or (len(word) >= 3 and other.startswith(word) and other[len(word):] in _PARTICLES)
            for other in others
        )

    return (
        all(equivalent(word, right_set) for word in left_set)
        and all(equivalent(word, left_set) for word in right_set)
    )


def _independent_description(item: NewsItem, title_tokens: tuple[str, ...]) -> str:
    text = strip_source_suffix(item.description, item.source)
    words = _tokens(text)
    source_words = _tokens(item.source)
    if source_words and words[-len(source_words):] == source_words:
        words = words[:-len(source_words)]
    description = " ".join(words)
    # RSS 설명이 제목과 언론사만 반복하면 별도의 확인 근거로 사용하지 않음.
    if len(words) < 4 or _similarity(description, " ".join(title_tokens)) >= 0.9:
        return ""
    return description


@dataclass(frozen=True)
class _ComparisonInfo:
    title: str
    title_key: str
    source_key: str
    published_time: datetime | None
    title_text: str
    title_words: frozenset[str]
    title_word_count: int
    title_numbers: frozenset[str]
    conflict_cues: frozenset[str]
    description: str
    description_words: frozenset[str]
    description_numbers: frozenset[str]


def _prepare_comparison(item: NewsItem) -> _ComparisonInfo:
    title = strip_source_suffix(item.title, item.source)
    words = _tokens(title)
    word_set = frozenset(words)
    description = _independent_description(item, words)
    return _ComparisonInfo(
        title=title,
        title_key=normalize_key(title),
        source_key=normalize_key(item.source),
        published_time=_published_time(item.published_at),
        title_text=" ".join(words),
        title_words=word_set - _FILLER_WORDS,
        title_word_count=len(word_set),
        title_numbers=frozenset(_NUMBERS.findall(title)),
        conflict_cues=frozenset(
            cue for cue in _CONFLICT_CUES if cue in f"{title} {item.description}"
        ),
        description=description,
        description_words=frozenset(_tokens(description)) - _FILLER_WORDS,
        description_numbers=frozenset(_NUMBERS.findall(description)),
    )


@dataclass(frozen=True)
class _PreparedArticle:
    item: NewsItem
    originals: tuple[_ComparisonInfo, ...]


def _prepare_article(item: NewsItem) -> _PreparedArticle:
    originals = (_prepare_comparison(item),) + tuple(
        _prepare_comparison(NewsItem(
            keyword=item.keyword, title=ref.title, source=ref.source,
            published_at=ref.published_at, link=ref.link,
        ))
        for ref in item.related_articles
    )
    return _PreparedArticle(item, originals)


class TitleSourceDeduplicator:
    """완전 중복 및 보수적인 동일 사건 묶기. 첫 수집 기사 대표, 원문 정보 보존."""

    def _same_event(self, left: _PreparedArticle, right: _PreparedArticle) -> bool:
        return all(
            self._same_event_pair(a, b)
            for a in left.originals for b in right.originals
        )

    def _same_event_pair(self, left: _ComparisonInfo, right: _ComparisonInfo) -> bool:
        if not left.title or not right.title:
            return False
        left_date, right_date = left.published_time, right.published_time
        if left_date is not None and right_date is not None:
            if abs((left_date - right_date).total_seconds()) > 48 * 3600:
                return False
        exact_duplicate = (
            left.title_key == right.title_key
            and left.source_key == right.source_key
        )
        if not exact_duplicate and (left_date is None or right_date is None):
            return False

        if left.conflict_cues != right.conflict_cues:
            return False
        left_description, right_description = left.description, right.description
        if left_description and right_description:
            if not _compatible_words(left.description_words, right.description_words):
                return False
            if left.description_numbers != right.description_numbers:
                return False
            if _similarity(left_description, right_description) < 0.85:
                return False
        if exact_duplicate:
            return True  # 제목·언론사가 같아도 날짜 또는 설명이 충돌하면 위에서 유지 처리

        left_text, right_text = left.title_text, right.title_text
        if min(len(left_text), len(right_text)) < 20:
            return False
        if min(left.title_word_count, right.title_word_count) < 4:
            return False
        if left.title_numbers != right.title_numbers:
            return False
        if not _compatible_words(left.title_words, right.title_words):
            return False
        title_similarity = _similarity(left_text, right_text)
        if title_similarity < 0.88:
            return False

        if left_description and right_description:
            return True
        # 설명이 없거나 제목 반복뿐인 경우 더 엄격한 제목 비교.
        return title_similarity >= 0.95

    def deduplicate(self, items: list[NewsItem]) -> list[NewsItem]:
        # Comparison data lives only for this call; source articles remain unchanged.
        groups: list[list[_PreparedArticle]] = []
        for item in items:
            prepared = _prepare_article(item)
            for group in groups:
                # 대표와만 비교하지 않고 모든 구성원과 비교하여 연쇄 오묶음 방지.
                if all(self._same_event(member, prepared) for member in group):
                    group.append(prepared)
                    break
            else:
                groups.append([prepared])

        representatives = []
        for group in groups:
            first = group[0].item
            references = list(first.related_articles)
            for prepared in group[1:]:
                member = prepared.item
                references.append(NewsReference(
                    title=member.title, source=member.source,
                    published_at=member.published_at, link=member.link,
                ))
                references.extend(member.related_articles)
            representatives.append(replace(first, related_articles=tuple(references)))
        return representatives
