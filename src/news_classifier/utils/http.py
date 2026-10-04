from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin

import requests

from news_classifier.utils.urls import is_public_http_url

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HttpFetchResult:
    """HTTP 응답과 실패 유형. 응답이 없는 경우에도 원인을 구분."""
    response: Optional[requests.Response]
    status: str
    http_status: int | None = None


@dataclass(frozen=True)
class HttpClient:
    headers: dict[str, str]
    timeout_seconds: int = 15

    def get(self, url: str) -> Optional[requests.Response]:
        return self.get_with_diagnostics(url).response

    def get_with_diagnostics(self, url: str) -> HttpFetchResult:
        response = None
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout_seconds,
                allow_redirects=True,
            )
            response.raise_for_status()
            return HttpFetchResult(response, "SUCCESS", response.status_code)
        except requests.Timeout as exc:
            logger.warning("request failed: %s | %s", url, exc)
            return HttpFetchResult(None, "TIMEOUT")
        except Exception as exc:
            logger.warning("request failed: %s | %s", url, exc)
            status_code = getattr(response, "status_code", None)
            if status_code is not None and status_code >= 400:
                status = "BLOCKED_HTTP" if status_code in {401, 403, 429, 451} else "HTTP_ERROR"
            else:
                status = "REQUEST_FAILED"
            return HttpFetchResult(None, status, status_code)

    def get_public_with_diagnostics(self, url: str, max_redirects: int = 4) -> HttpFetchResult:
        """New article/resolver requests validate the destination at every hop."""
        current = url
        for hop in range(max_redirects + 1):
            if not is_public_http_url(current, check_dns=True):
                return HttpFetchResult(None, "UNSAFE_URL")
            response = None
            try:
                response = requests.get(current, headers=self.headers, timeout=self.timeout_seconds,
                                        allow_redirects=False)
                response.raise_for_status()
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("Location")
                    if not location or hop == max_redirects:
                        return HttpFetchResult(None, "REDIRECT_LIMIT", response.status_code)
                    current = urljoin(current, location)
                    continue
                if not 200 <= response.status_code < 300:
                    return HttpFetchResult(None, "HTTP_ERROR", response.status_code)
                return HttpFetchResult(response, "SUCCESS", response.status_code)
            except requests.Timeout:
                return HttpFetchResult(None, "TIMEOUT")
            except Exception:
                code = getattr(response, "status_code", None)
                status = "BLOCKED_HTTP" if code in {401, 403, 429, 451} else "HTTP_ERROR" if code and code >= 400 else "REQUEST_FAILED"
                return HttpFetchResult(None, status, code)
        return HttpFetchResult(None, "REDIRECT_LIMIT")

    def post_google_news_lookup(self, data: dict[str, str]) -> HttpFetchResult:
        """Read-only article URL lookup at a fixed Google endpoint, no redirects."""
        response = None
        try:
            response = requests.post(
                "https://news.google.com/_/DotsSplashUi/data/batchexecute?rpcids=Fbv4je",
                data=data, headers=self.headers, timeout=self.timeout_seconds, allow_redirects=False,
            )
            response.raise_for_status()
            if not 200 <= response.status_code < 300:
                return HttpFetchResult(None, "HTTP_ERROR", response.status_code)
            return HttpFetchResult(response, "SUCCESS", response.status_code)
        except requests.Timeout:
            return HttpFetchResult(None, "TIMEOUT")
        except Exception:
            code = getattr(response, "status_code", None)
            status = "BLOCKED_HTTP" if code in {401, 403, 429, 451} else "HTTP_ERROR" if code and code >= 400 else "REQUEST_FAILED"
            return HttpFetchResult(None, status, code)
