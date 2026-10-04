from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests

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
