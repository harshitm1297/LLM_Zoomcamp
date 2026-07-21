from __future__ import annotations

import json
import re
from json import JSONDecodeError
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SENSITIVE_QUERY_PARAMETER = re.compile(
    r"([?&](?:api_key|access_token|token|key)=)[^&]+",
    flags=re.IGNORECASE,
)


def redact_url(url: str) -> str:
    """Hide common query-string credentials before URLs enter logs or errors."""
    return SENSITIVE_QUERY_PARAMETER.sub(r"\1<redacted>", url)


def http_get_json(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    user_agent: str,
    timeout: int = 60,
) -> dict[str, Any]:
    if params:
        url = f"{url}?{urlencode(params)}"

    request = Request(url, headers={"User-Agent": user_agent})
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return json.loads(body)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {redact_url(url)}\n{body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error for {redact_url(url)}: {exc}") from exc
    except JSONDecodeError as exc:
        raise RuntimeError(f"Non-JSON response for {redact_url(url)}: {exc}") from exc


def http_get_text(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    user_agent: str,
    timeout: int = 60,
) -> str:
    if params:
        url = f"{url}?{urlencode(params)}"

    request = Request(url, headers={"User-Agent": user_agent})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {redact_url(url)}\n{body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error for {redact_url(url)}: {exc}") from exc


def http_download_binary(url: str, *, user_agent: str, timeout: int = 180) -> bytes:
    request = Request(url, headers={"User-Agent": user_agent})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {redact_url(url)}\n{body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error for {redact_url(url)}: {exc}") from exc


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return cleaned or "untitled"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
