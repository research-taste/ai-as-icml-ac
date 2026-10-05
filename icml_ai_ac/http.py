from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from icml_ai_ac.storage import ensure_parent


DEFAULT_USER_AGENT = (
    "icml-ai-ac-crawler/0.1 "
    "(public paper metadata research; contact: project-maintainer)"
)


class AccessChallengeError(RuntimeError):
    """Raised when a public endpoint requires interactive browser verification."""

    def __init__(self, url: str, message: str) -> None:
        super().__init__(f"access challenge for {url}: {message}")
        self.url = url
        self.message = message


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    status: int
    content_type: str | None
    body: bytes
    final_url: str

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


@dataclass(slots=True)
class DownloadResult:
    url: str
    path: Path
    bytes_written: int
    sha256: str
    skipped: bool


class HttpClient:
    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_seconds: float = 30.0,
        retries: int = 3,
        backoff_seconds: float = 1.5,
        polite_delay_seconds: float = 0.0,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self.polite_delay_seconds = polite_delay_seconds
        self.default_headers = dict(default_headers or {})
        self._last_request_at = 0.0

    def fetch(self, url: str, *, accept: str | None = None) -> FetchResult:
        request = urllib.request.Request(url, headers=self._headers(accept=accept))
        return self._execute(request)

    def post_json(self, url: str, payload: dict[str, object]) -> FetchResult:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers = self._headers(accept="application/json")
        headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        return self._execute(request)

    def set_bearer_token(self, token: str) -> None:
        self.default_headers["Authorization"] = f"Bearer {token}"

    def _headers(self, *, accept: str | None = None) -> dict[str, str]:
        headers = dict(self.default_headers)
        headers.setdefault("User-Agent", self.user_agent)
        if accept:
            headers["Accept"] = accept
        return headers

    def _execute(self, request: urllib.request.Request) -> FetchResult:
        url = request.full_url
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            self._sleep_if_needed()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = response.read()
                    return FetchResult(
                        url=url,
                        status=response.status,
                        content_type=response.headers.get("content-type"),
                        body=body,
                        final_url=response.geturl(),
                    )
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                if isinstance(exc, urllib.error.HTTPError) and exc.code == 403:
                    challenge_message = _access_challenge_message(exc)
                    if challenge_message:
                        raise AccessChallengeError(url, challenge_message) from exc
                if isinstance(exc, urllib.error.HTTPError) and exc.code == 429 and attempt < self.retries:
                    retry_after = exc.headers.get("retry-after") if exc.headers else None
                    sleep_seconds = _parse_retry_after(retry_after) or max(
                        self.backoff_seconds * (2**attempt),
                        self.polite_delay_seconds,
                    )
                    time.sleep(sleep_seconds)
                    continue
                if isinstance(exc, urllib.error.HTTPError) and exc.code < 500:
                    raise
                if attempt < self.retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
        if last_error is None:
            raise RuntimeError("HTTP request failed without an exception")
        raise last_error

    def download(self, url: str, path: Path, *, overwrite: bool = False) -> DownloadResult:
        ensure_parent(path)
        if is_pdf_file(path) and not overwrite:
            digest = sha256_file(path)
            return DownloadResult(url=url, path=path, bytes_written=path.stat().st_size, sha256=digest, skipped=True)

        result = self.fetch(url, accept="application/pdf,*/*")
        if not is_pdf_response(result):
            content_type = result.content_type or "unknown content type"
            raise ValueError(f"downloaded response is not a PDF: {url} ({content_type})")
        digest = hashlib.sha256(result.body).hexdigest()
        path.write_bytes(result.body)
        return DownloadResult(url=url, path=path, bytes_written=len(result.body), sha256=digest, skipped=False)

    def _sleep_if_needed(self) -> None:
        if self.polite_delay_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.polite_delay_seconds:
            time.sleep(self.polite_delay_seconds - elapsed)
        self._last_request_at = time.monotonic()


def join_url(base_url: str, href: str) -> str:
    return urllib.parse.urljoin(base_url, href)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_pdf_response(result: FetchResult) -> bool:
    content_type = (result.content_type or "").lower()
    if "pdf" in content_type:
        return True
    return b"%PDF-" in result.body[:1024]


def is_pdf_file(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        with path.open("rb") as handle:
            return b"%PDF-" in handle.read(1024)
    except OSError:
        return False


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None


def _access_challenge_message(exc: urllib.error.HTTPError) -> str | None:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - the original HTTP error remains authoritative.
        return None
    lowered = body.lower()
    if "challengerequirederror" in lowered or "challenge verification required" in lowered:
        return "interactive challenge verification required"
    hostname = urllib.parse.urlparse(exc.url).hostname or ""
    if hostname.endswith("openreview.net") and (
        "access to this page is restricted" in lowered
        or ("error 403" in lowered and "logged in to openreview" in lowered)
    ):
        return "OpenReview returned an access-gate HTTP 403"
    return None
