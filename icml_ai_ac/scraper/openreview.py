from __future__ import annotations

import os
import urllib.parse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from icml_ai_ac.http import HttpClient
from icml_ai_ac.models import PaperRecord
from icml_ai_ac.storage import read_json


OPENREVIEW_WEB_BASE = "https://openreview.net"
OPENREVIEW_API2_BASE = "https://api2.openreview.net"


class OpenReviewAuthenticationError(RuntimeError):
    """Raised when an OpenReview API v2 session cannot be established."""


class OpenReviewMfaRequiredError(OpenReviewAuthenticationError):
    """Raised when OpenReview requires an MFA step this client cannot complete."""


class OpenReviewClient:
    """Minimal OpenReview API v2 client.

    This intentionally avoids the Python openreview package so the crawler can
    bootstrap in a fresh environment. Query parameters are exposed because venue
    visibility rules and invitation names often change across years.
    """

    def __init__(self, *, api_base: str = OPENREVIEW_API2_BASE, http_client: HttpClient | None = None) -> None:
        self.api_base = api_base.rstrip("/")
        self.http = http_client or HttpClient()
        self.authenticated = False

    def authenticate_from_env(self, *, token_expires_seconds: int = 3600) -> None:
        username = os.environ.get("OPENREVIEW_USERNAME", "")
        password = os.environ.get("OPENREVIEW_PASSWORD", "")
        if not username or not password:
            raise OpenReviewAuthenticationError(
                "OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD must both be set"
            )
        self.authenticate(username=username, password=password, token_expires_seconds=token_expires_seconds)

    def authenticate(self, *, username: str, password: str, token_expires_seconds: int = 3600) -> None:
        response = self.http.post_json(
            f"{self.api_base}/login",
            {"id": username, "password": password, "expiresIn": token_expires_seconds},
        )
        payload = read_json_from_string(response.text)
        if payload.get("mfaPending"):
            methods = payload.get("mfaMethods")
            method_text = ", ".join(str(item) for item in methods) if isinstance(methods, list) else "unknown"
            raise OpenReviewMfaRequiredError(f"OpenReview MFA is required (available methods: {method_text})")
        token = payload.get("token")
        if not isinstance(token, str) or not token:
            raise OpenReviewAuthenticationError("OpenReview login succeeded without returning an access token")
        self.http.set_bearer_token(token)
        self.authenticated = True

    def pdf_url_for_forum_id(self, forum_id: str) -> str:
        return f"{self.api_base}/pdf?id={urllib.parse.quote(forum_id)}"

    def get_group(self, group_id: str) -> dict[str, Any] | None:
        url = f"{self.api_base}/groups?{urllib.parse.urlencode({'id': group_id})}"
        payload = self.http.fetch(url, accept="application/json").text
        data = read_json_from_string(payload)
        groups = data.get("groups", [])
        if isinstance(groups, list) and groups and isinstance(groups[0], dict):
            return groups[0]
        return None

    def iter_notes(
        self,
        *,
        query: dict[str, str],
        page_size: int = 1000,
        max_notes: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        offset = 0
        yielded = 0
        while True:
            params = dict(query)
            params["limit"] = str(page_size)
            params["offset"] = str(offset)
            url = f"{self.api_base}/notes?{urllib.parse.urlencode(params)}"
            payload = self.http.fetch(url, accept="application/json").text
            data = read_json_from_string(payload)
            notes = data.get("notes", [])
            if not isinstance(notes, list) or not notes:
                break
            for note in notes:
                if isinstance(note, dict):
                    yield note
                    yielded += 1
                    if max_notes is not None and yielded >= max_notes:
                        return
            if len(notes) < page_size:
                break
            offset += len(notes)

    def count_notes(self, *, query: dict[str, str]) -> int | None:
        params = dict(query)
        params["limit"] = "1"
        params["offset"] = "0"
        url = f"{self.api_base}/notes?{urllib.parse.urlencode(params)}"
        payload = self.http.fetch(url, accept="application/json").text
        data = read_json_from_string(payload)
        count = data.get("count")
        return int(count) if isinstance(count, int) else None

    def search_notes(
        self,
        *,
        title: str,
        limit: int = 10,
        source: str = "forum",
    ) -> list[dict[str, Any]]:
        params = {
            "term": title,
            "content": "title",
            "source": source,
            "limit": str(limit),
        }
        url = f"{self.api_base}/notes/search?{urllib.parse.urlencode(params)}"
        payload = self.http.fetch(url, accept="application/json").text
        data = read_json_from_string(payload)
        notes = data.get("notes", [])
        return [note for note in notes if isinstance(note, dict)] if isinstance(notes, list) else []

    def notes_to_records(
        self,
        notes: Iterator[dict[str, Any]],
        *,
        source: str,
        decision_label: str | None = None,
    ) -> Iterator[PaperRecord]:
        for note in notes:
            yield note_to_record(note, source=source, decision_label=decision_label)


def build_query(
    *,
    venue_id: str | None = None,
    invitation: str | None = None,
    content_venue: str | None = None,
    details: str | None = None,
) -> dict[str, str]:
    query: dict[str, str] = {}
    if venue_id:
        query["content.venueid"] = venue_id
    if content_venue:
        query["content.venue"] = content_venue
    if invitation:
        query["invitation"] = invitation
    if details:
        query["details"] = details
    return query


def content_field(group: dict[str, Any], key: str) -> Any:
    content = group.get("content") if isinstance(group.get("content"), dict) else {}
    value = content.get(key)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def venue_status_report(group: dict[str, Any]) -> dict[str, Any]:
    venue_id = str(group.get("id") or "")
    submission_name = content_field(group, "submission_name") or "Submission"
    submission_invitation = content_field(group, "submission_id") or f"{venue_id}/-/{submission_name}"
    decision_heading_map = content_field(group, "decision_heading_map") or {}

    statuses = {
        "all_submissions": {
            "public": bool(content_field(group, "public_submissions")),
            "query": {"invitation": submission_invitation},
        },
        "accepted": {
            "public": bool(decision_heading_map),
            "query": {"content.venueid": venue_id},
            "fallback_queries": [
                {"invitation": submission_invitation, "content.venue": venue}
                for venue, decision in decision_heading_map.items()
                if isinstance(decision, str) and decision.startswith("Accept")
            ],
            "archive_queries": [
                {"content.venue": venue}
                for venue, decision in decision_heading_map.items()
                if isinstance(decision, str) and decision.startswith("Accept")
            ],
        },
        "submitted": {
            "public": bool(content_field(group, "public_submissions")),
            "query": {"content.venueid": content_field(group, "submission_venue_id")},
        },
        "rejected": {
            "public": bool(decision_heading_map),
            "query": {"content.venueid": content_field(group, "rejected_venue_id")},
            "fallback_queries": [
                {"invitation": submission_invitation, "content.venue": venue}
                for venue, decision in decision_heading_map.items()
                if isinstance(decision, str) and "reject" in decision.lower()
            ],
            "archive_queries": [
                {"content.venue": venue}
                for venue, decision in decision_heading_map.items()
                if isinstance(decision, str) and "reject" in decision.lower()
            ],
        },
        "desk_rejected": {
            "public": bool(content_field(group, "public_desk_rejected_submissions")),
            "query": {"content.venueid": content_field(group, "desk_rejected_venue_id")},
        },
        "withdrawn": {
            "public": bool(content_field(group, "public_withdrawn_submissions")),
            "query": {"content.venueid": content_field(group, "withdrawn_venue_id")},
        },
    }
    return {
        "venue_id": venue_id,
        "api_version": "api2" if group.get("domain") else "api1_or_unknown",
        "domain": group.get("domain"),
        "submission_invitation": submission_invitation,
        "public_submissions": bool(content_field(group, "public_submissions")),
        "public_withdrawn_submissions": bool(content_field(group, "public_withdrawn_submissions")),
        "public_desk_rejected_submissions": bool(content_field(group, "public_desk_rejected_submissions")),
        "decision_heading_map": decision_heading_map,
        "statuses": statuses,
    }


def query_for_status(group: dict[str, Any], status: str) -> dict[str, str]:
    queries = queries_for_status(group, status, include_fallbacks=False)
    return queries[0] if queries else {}


def queries_for_status(group: dict[str, Any], status: str, *, include_fallbacks: bool = True) -> list[dict[str, str]]:
    report = venue_status_report(group)
    statuses = report["statuses"]
    if status not in statuses:
        raise ValueError(f"Unknown OpenReview venue status: {status}")
    raw_queries = [statuses[status]["query"]]
    if include_fallbacks:
        raw_queries.extend(statuses[status].get("fallback_queries", []))
        raw_queries.extend(statuses[status].get("archive_queries", []))

    queries: list[dict[str, str]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for raw_query in raw_queries:
        query = {str(key): str(value) for key, value in raw_query.items() if value is not None}
        signature = tuple(sorted(query.items()))
        if query and signature not in seen:
            seen.add(signature)
            queries.append(query)
    return queries


def note_to_record(note: dict[str, Any], *, source: str, decision_label: str | None = None) -> PaperRecord:
    note_id = str(note.get("forum") or note.get("id") or "")
    content = note.get("content") if isinstance(note.get("content"), dict) else {}
    title = content_value(content, "title")
    abstract = content_value(content, "abstract")
    authors = content_list(content, "authors")
    venue = content_value(content, "venue")
    venue_id = content_value(content, "venueid")
    decision = decision_label or content_value(content, "decision") or venue
    pdf_value = content_value(content, "pdf")
    html_value = content_value(content, "html")
    pdf_url = normalize_openreview_url(pdf_value)
    if not pdf_url and html_value and "/pdf" in html_value:
        pdf_url = normalize_openreview_url(html_value)
    if not pdf_url:
        pdf_url = pdf_url_for_forum_id(note_id)
    forum_url = normalize_openreview_url(html_value) if html_value and "/forum" in html_value else None
    if not forum_url and note_id:
        forum_url = f"{OPENREVIEW_WEB_BASE}/forum?id={note_id}"

    return PaperRecord(
        paper_id=note_id,
        source=source,
        decision_label=decision,
        title=title,
        abstract=abstract,
        authors=authors,
        forum_url=forum_url,
        pdf_url=pdf_url,
        extra={
            "openreview_id": note.get("id"),
            "openreview_forum": note.get("forum"),
            "openreview_invitation": note.get("invitation"),
            "openreview_venue": venue,
            "openreview_venueid": venue_id,
            "openreview_html": html_value,
            "openreview_pdf_field": pdf_value,
            "source_parser": "openreview_api2",
        },
    )


def content_value(content: dict[str, Any], key: str) -> str | None:
    value = content.get(key)
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    if isinstance(value, str):
        return " ".join(value.split()) or None
    return None


def content_list(content: dict[str, Any], key: str) -> list[str]:
    value = content.get(key)
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return []


def normalize_openreview_url(value: str | None) -> str | None:
    if not value:
        return None
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if value.startswith("/"):
        return f"{OPENREVIEW_WEB_BASE}{value}"
    return value


def forum_id_from_url(value: str | None) -> str | None:
    """Extract an OpenReview forum/note id from a forum or PDF URL."""

    if not value:
        return None
    parsed = urllib.parse.urlparse(value)
    query_id = urllib.parse.parse_qs(parsed.query).get("id")
    if query_id and query_id[0]:
        return query_id[0]
    if parsed.path.startswith("/forum/") or parsed.path.startswith("/pdf/"):
        candidate = parsed.path.rsplit("/", 1)[-1]
        if candidate:
            return candidate.removesuffix(".pdf")
    return None


def pdf_url_for_forum_id(forum_id: str | None) -> str | None:
    if not forum_id:
        return None
    return f"{OPENREVIEW_WEB_BASE}/pdf?id={urllib.parse.quote(forum_id)}"


def read_json_from_string(payload: str) -> dict[str, Any]:
    # Kept separate so tests can patch without depending on urllib internals.
    import json

    data = json.loads(payload)
    return data if isinstance(data, dict) else {}


def read_query_file(path: str) -> dict[str, str]:
    payload = read_json(Path(path))
    if not isinstance(payload, dict):
        raise ValueError(f"OpenReview query file must contain a JSON object: {path}")
    return {str(key): str(value) for key, value in payload.items()}
