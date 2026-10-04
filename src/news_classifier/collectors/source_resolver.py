"""Bounded Google News URL lookup, not a general crawler or challenge bypass.

Wire protocol reference: https://github.com/SSujitX/google-news-url-decoder
The endpoint is undocumented and can change; failures retain REVIEW.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from news_classifier.utils.http import HttpFetchResult
from news_classifier.utils.urls import is_public_http_url


@dataclass(frozen=True)
class SourceResolution:
    url: str = ""
    status: str = "GOOGLE_NEWS_WRAPPER"
    http_status: int | None = None


def fetch_article_page(client, url: str) -> HttpFetchResult:
    if not is_public_http_url(url):
        return HttpFetchResult(None, "UNSAFE_URL")
    fetch = getattr(client, "get_public_with_diagnostics", None)
    if not callable(fetch):
        fetch = getattr(client, "get_with_diagnostics", None)
    if callable(fetch):
        return fetch(url)
    response = client.get(url)
    return HttpFetchResult(response, "SUCCESS" if response is not None else "REQUEST_FAILED",
                           getattr(response, "status_code", None))


def _publisher_url(url: str) -> bool:
    return is_public_http_url(url) and (urlsplit(url).hostname or "").lower() not in {
        "news.google.com", "consent.google.com", "www.google.com", "google.com",
    }


def parse_lookup_response(text: str) -> str:
    """Only accept a single garturlres from the requested RPC, not arbitrary URLs."""
    if len(text) > 2_000_000:
        return ""
    try:
        # batchexecute adds an XSSI prefix and sometimes a byte-count line.
        start = text.index("[")
        rows, _ = json.JSONDecoder().raw_decode(text[start:])
        urls = []
        for row in rows:
            if not isinstance(row, list) or len(row) < 3 or row[:2] != ["wrb.fr", "Fbv4je"]:
                continue
            payload = json.loads(row[2]) if isinstance(row[2], str) else row[2]
            if (isinstance(payload, list) and len(payload) >= 2 and payload[0] == "garturlres"
                    and isinstance(payload[1], str) and _publisher_url(payload[1])):
                urls.append(payload[1])
        return urls[0] if len(urls) == 1 else ""
    except (ValueError, TypeError, IndexError):
        return ""


class GoogleNewsSourceResolver:
    def __init__(self, client):
        self.client = client
        self._resolved: dict[str, SourceResolution] = {}

    def resolve(self, source_url: str, response) -> SourceResolution:
        parsed = urlsplit(source_url)
        match = re.fullmatch(r"/(?:rss/)?(?:articles|read)/([A-Za-z0-9_-]{16,2048})", parsed.path)
        if parsed.hostname != "news.google.com" or match is None:
            return SourceResolution()
        article_id = match[1]
        if article_id in self._resolved:
            return self._resolved[article_id]
        html = response.text
        for attempt in range(2):
            soup = BeautifulSoup(html[:2_000_000], "html.parser")
            # Explicit canonical/og:url only; menu, ad and arbitrary anchors ignored.
            canonical = soup.find("link", rel="canonical")
            og_url = soup.find("meta", property="og:url")
            for url in (canonical.get("href", "") if canonical else "",
                        og_url.get("content", "") if og_url else ""):
                if _publisher_url(url):
                    return self._remember(article_id, SourceResolution(url, "HTML_SOURCE_URL", 200))
            signed = soup.find(attrs={"data-n-a-sg": True, "data-n-a-ts": True})
            if signed is not None:
                signature, timestamp = signed.get("data-n-a-sg", ""), signed.get("data-n-a-ts", "")
                if (not 1 <= len(signature) <= 1024 or not re.fullmatch(r"\d{1,20}", timestamp)
                        or signed.get("data-n-a-id", article_id) != article_id):
                    return SourceResolution(status="SOURCE_URL_UNRESOLVED")
                lookup = getattr(self.client, "post_google_news_lookup", None)
                if not callable(lookup):
                    return SourceResolution()
                # These values form Google's article-lookup request context.
                context = [["X", "X", ["X", "X"], None, None, 1, 1, "US:en", None, 1,
                            None, None, None, None, None, 0, 1], "X", "X", 1, [1, 1, 1],
                           1, 1, None, 0, 0, None, 0]
                query = ["garturlreq", context, article_id, int(timestamp), signature]
                request = [[["Fbv4je", json.dumps(query, separators=(",", ":")), None, "0"]]]
                outcome = lookup({"f.req": json.dumps(request, separators=(",", ":"))})
                if outcome.response is None:
                    return SourceResolution(status=outcome.status, http_status=outcome.http_status)
                url = parse_lookup_response(outcome.response.text)
                if not url:
                    return SourceResolution(status="SOURCE_URL_UNRESOLVED", http_status=outcome.http_status)
                return self._remember(article_id, SourceResolution(url, "GOOGLE_RPC_SOURCE_URL", outcome.http_status))
            if attempt == 0:
                outcome = fetch_article_page(self.client,
                    f"https://news.google.com/rss/articles/{article_id}?hl=ko&gl=KR&ceid=KR%3Ako")
                if outcome.response is None:
                    return SourceResolution(status=outcome.status, http_status=outcome.http_status)
                final_url = getattr(outcome.response, "url", "")
                if _publisher_url(final_url):
                    return self._remember(article_id, SourceResolution(final_url, "HTTP_SOURCE_REDIRECT", outcome.http_status))
                if urlsplit(final_url).hostname not in {None, "news.google.com"}:
                    return SourceResolution(status="CONSENT_OR_BLOCK_PAGE", http_status=outcome.http_status)
                html = outcome.response.text
        return SourceResolution(status="SOURCE_URL_UNRESOLVED", http_status=200)

    def _remember(self, article_id, result):
        if len(self._resolved) >= 256:
            self._resolved.pop(next(iter(self._resolved)))
        self._resolved[article_id] = result
        return result
