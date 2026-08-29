"""Shared GBIF API client and helpers for the mg-ants POC.

All scripts go through GbifClient: one requests.Session, a project
User-Agent, a soft rate limit (<= 4 req/s) and retry with exponential
backoff on 429/5xx and connection errors.
"""
from __future__ import annotations

import hashlib
import logging
from contextlib import contextmanager
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import requests
import yaml

API_BASE = "https://api.gbif.org/v1"
USER_AGENT = "mg-ants-poc/0.1"
_MIN_INTERVAL_S = 0.25  # <= 4 requests per second
_RETRY_STATUSES = {429, 500, 502, 503, 504}

# catalogNumber suffix for the image view, e.g. "casent0214665-d01"
# only h/d/p/l are view suffixes; don't strip other dash segments
_CATALOG_SUFFIX_RE = re.compile(r"-[hdpl]\d+$", re.IGNORECASE)
# view letter inside an AntWeb image URL, e.g. ".../casent0214665_p_1_high.jpg"
_URL_VIEW_RE = re.compile(r"_([hdpl])_\d+_")

log = logging.getLogger(__name__)


class GbifClient:
    def __init__(self, max_retries: int = 5, timeout: int = 60) -> None:
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.max_retries = max_retries
        self.timeout = timeout
        self._last_request = 0.0

    def _throttle(self) -> None:
        wait = _MIN_INTERVAL_S - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def get(self, path: str, params: dict[str, Any] | None = None) -> requests.Response:
        url = path if path.startswith("http") else f"{API_BASE}/{path.lstrip('/')}"
        backoff = 1.0
        resp: requests.Response | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise
                log.warning("request error on %s: %s; retry in %.0fs", url, exc, backoff)
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code in _RETRY_STATUSES and attempt < self.max_retries:
                log.warning("HTTP %d on %s; retry in %.0fs", resp.status_code, url, backoff)
                time.sleep(backoff)
                backoff *= 2
                continue
            return resp
        assert resp is not None
        return resp

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        resp = self.get(path, params)
        resp.raise_for_status()
        return resp.json()


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return yaml.safe_load(fh)


def setup_logging(report_path: Path | None = None, level: int = logging.INFO) -> None:
    """Log to stdout, and also to a file when report_path is given."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(report_path, mode="w", encoding="utf-8"))
    logging.basicConfig(level=level, format="%(message)s", handlers=handlers)


@contextmanager
def quiet_logging(level: int = logging.WARNING) -> Iterator[None]:
    """Temporarily raise the ROOT logger level. open_clip logs its model
    loading chatter through logging.info() on the root logger, so a
    per-module logger level cannot silence it."""
    root = logging.getLogger()
    previous = root.level
    root.setLevel(level)
    try:
        yield
    finally:
        root.setLevel(previous)


def specimen_code(catalog_number: str) -> str:
    """Strip the per-view suffix: 'casent0214665-d01' -> 'casent0214665'."""
    return _CATALOG_SUFFIX_RE.sub("", catalog_number.strip())


def view_from_url(image_url: str) -> str | None:
    """Extract the view letter (h/d/p/l) from an AntWeb image URL."""
    m = _URL_VIEW_RE.search(image_url)
    return m.group(1) if m else None


def sanitize_code(code: str) -> str:
    """Filesystem-safe specimen code: chars outside [A-Za-z0-9_-] become '_',
    with runs of '_' collapsed (e.g. 'blf2102(14)-8' -> 'blf2102_14_-8')."""
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_-]", "_", code))


def is_antweb_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host == "antweb.org" or host.endswith(".antweb.org")


def cache_url(gbif_id: int | str, identifier: str, size: str = "x") -> str:
    """GBIF image-cache URL for a media item.

    size is '{W}x{H}' (max 1200x1200) or the literal 'x' for the original.
    The media path segment is the md5 hex of the original identifier URL.
    """
    md5 = hashlib.md5(identifier.encode("utf-8")).hexdigest()
    return f"{API_BASE}/image/cache/{size}/occurrence/{gbif_id}/media/{md5}"
