"""Public article URL checks; never follow credentials, local hosts or private IPs."""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit


def is_public_http_url(url: str, *, check_dns: bool = False) -> bool:
    if not isinstance(url, str) or len(url) > 8192 or any(ord(c) < 33 for c in url):
        return False
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").rstrip(".").lower()
        if (parsed.scheme not in {"http", "https"} or not host
                or parsed.username is not None or parsed.password is not None
                or parsed.port not in {None, 80, 443}
                or host == "localhost" or host.endswith((".localhost", ".local", ".internal"))):
            return False
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if "." not in host or "%" in host or "\\" in host:
                return False
        else:
            if not address.is_global:
                return False
        if check_dns:
            addresses = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80),
                                           type=socket.SOCK_STREAM)
            return bool(addresses) and all(ipaddress.ip_address(entry[4][0]).is_global for entry in addresses)
        return True
    except (ValueError, OSError, UnicodeError):
        return False
