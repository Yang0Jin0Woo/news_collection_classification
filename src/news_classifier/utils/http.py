from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HttpClient:
    headers: dict[str, str]
    timeout_seconds: int = 15

    def get(self, url: str) -> Optional[requests.Response]:
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout_seconds,
                allow_redirects=True,
            )
            response.raise_for_status()
            return response
        except Exception as exc:
            logger.warning("request failed: %s | %s", url, exc)
            return None
