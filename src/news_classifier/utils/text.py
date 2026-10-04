import html
import re
import unicodedata
from bs4 import BeautifulSoup

_BRACKET_PATTERN = re.compile(r"\[[^\]]+\]")
_PAREN_PATTERN = re.compile(r"\([^)]*\)")
_SPACE_PATTERN = re.compile(r"\s+")
_QUOTE_PATTERN = re.compile(r"[\"'“”‘’]")
_WORD_PATTERN = re.compile(r"[A-Za-z0-9가-힣]+")


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = _SPACE_PATTERN.sub(" ", text)
    return text.strip()


def strip_html(raw_html: str | None) -> str:
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    return clean_text(soup.get_text(" "))


def normalize_description(text: str | None) -> str:
    text = clean_text(text)
    if not text:
        return ""
    text = _BRACKET_PATTERN.sub(" ", text)
    text = _PAREN_PATTERN.sub(" ", text)
    text = _QUOTE_PATTERN.sub("", text)
    text = _SPACE_PATTERN.sub(" ", text).strip()
    if len(_WORD_PATTERN.findall(text)) < 2:
        return ""
    return text


def normalize_key(text: str | None) -> str:
    text = clean_text(text).lower()
    return re.sub(r"\s+", "", text)


def strip_source_suffix(title: str | None, source: str | None) -> str:
    title = clean_text(title)
    source = clean_text(source)
    if not title or not source:
        return title

    pattern = re.compile(rf"\s+[-|–—]\s*{re.escape(source)}\s*$", flags=re.IGNORECASE)
    while True:
        cleaned = pattern.sub("", title).strip()
        if cleaned == title:
            return title
        title = cleaned


def safe_truncate(text: str, limit: int) -> str:
    text = clean_text(text)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def article_description(title: str, description: str, source: str = "") -> str:
    """언론사명과 제목만 반복하는 RSS 설명 제외. 원본 필드는 변경하지 않음."""
    description = clean_text(description)
    source = clean_text(source)
    if source:
        if normalize_key(description) == normalize_key(source):
            return ""
        description = re.sub(
            rf"\s+(?:[-|–—]\s*)?{re.escape(source)}\s*$",
            "", description, flags=re.IGNORECASE,
        ).strip()

    def comparable(value: str) -> str:
        return "".join(_WORD_PATTERN.findall(
            unicodedata.normalize("NFKC", value).casefold()
        ))

    if not comparable(description) or comparable(description) == comparable(title):
        return ""
    return description
