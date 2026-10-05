from __future__ import annotations

import json
import re
import urllib.parse
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from icml_ai_ac.http import HttpClient, join_url
from icml_ai_ac.models import PaperRecord
from icml_ai_ac.scraper.html_utils import clean_text, iter_json_ld_nodes, parse_html_metadata
from icml_ai_ac.storage import write_json

POSTER_RE_TEMPLATE = r"/virtual/{year}/poster/([0-9]+)"
EVENTS_URL_TEMPLATE = "https://icml.cc/static/virtual/data/icml-{year}-orals-posters.json"
ABSTRACTS_URL_TEMPLATE = "https://icml.cc/static/virtual/data/icml-{year}-abstracts.json"


@dataclass(frozen=True, slots=True)
class PosterStub:
    paper_id: str
    title: str | None
    detail_url: str


class ICMLVirtualScraper:
    """Scrape accepted papers from the public ICML virtual-site index."""

    def __init__(
        self,
        *,
        year: int = 2026,
        index_url: str | None = None,
        http_client: HttpClient | None = None,
    ) -> None:
        self.year = year
        self.index_url = index_url or f"https://icml.cc/virtual/{year}/papers.html"
        self.http = http_client or HttpClient()
        self._poster_re = re.compile(POSTER_RE_TEMPLATE.format(year=year))

    def discover_index(self) -> list[PosterStub]:
        response = self.http.fetch(self.index_url, accept="text/html,*/*")
        return list(self.parse_index(response.text, base_url=response.final_url))

    def fetch_events_payload(self, events_url: str | None = None) -> dict[str, Any]:
        url = events_url or EVENTS_URL_TEMPLATE.format(year=self.year)
        response = self.http.fetch(url, accept="application/json,*/*")
        payload = json.loads(response.text)
        return payload if isinstance(payload, dict) else {}

    def fetch_abstracts_payload(self, abstracts_url: str | None = None) -> dict[str, str]:
        url = abstracts_url or ABSTRACTS_URL_TEMPLATE.format(year=self.year)
        response = self.http.fetch(url, accept="application/json,*/*")
        payload = json.loads(response.text)
        if not isinstance(payload, dict):
            return {}
        return {str(key): str(value) for key, value in payload.items() if value is not None}

    def scrape_json_records(
        self,
        *,
        events_url: str | None = None,
        abstracts_url: str | None = None,
        event_types: set[str] | None = None,
        require_decision: bool = False,
        limit: int | None = None,
    ) -> tuple[list[PaperRecord], dict[str, Any]]:
        events_payload = self.fetch_events_payload(events_url)
        abstracts = self.fetch_abstracts_payload(abstracts_url)
        records, report = records_from_virtual_json(
            events_payload,
            abstracts=abstracts,
            year=self.year,
            event_types=event_types or {"Poster"},
            require_decision=require_decision,
            limit=limit,
        )
        report["events_url"] = events_url or EVENTS_URL_TEMPLATE.format(year=self.year)
        report["abstracts_url"] = abstracts_url or ABSTRACTS_URL_TEMPLATE.format(year=self.year)
        return records, report

    def parse_index(self, markup: str, *, base_url: str | None = None) -> Iterator[PosterStub]:
        base_url = base_url or self.index_url
        parser = parse_html_metadata(markup)
        seen: set[str] = set()
        for link in parser.links:
            detail_url = join_url(base_url, link.href)
            match = self._poster_re.fullmatch(urlparse(detail_url).path)
            if not match:
                continue
            paper_id = match.group(1)
            if paper_id in seen:
                continue
            seen.add(paper_id)
            yield PosterStub(
                paper_id=paper_id,
                title=clean_text(link.text),
                detail_url=detail_url,
            )

    def scrape_detail(self, stub: PosterStub, *, raw_html_dir: Path | None = None) -> PaperRecord:
        response = self.http.fetch(stub.detail_url, accept="text/html,*/*")
        if raw_html_dir is not None:
            write_json(
                raw_html_dir / f"{stub.paper_id}.meta.json",
                {
                    "url": stub.detail_url,
                    "final_url": response.final_url,
                    "status": response.status,
                    "content_type": response.content_type,
                },
            )
            raw_html_dir.mkdir(parents=True, exist_ok=True)
            (raw_html_dir / f"{stub.paper_id}.html").write_text(response.text, encoding="utf-8")
        return self.parse_detail(response.text, stub=stub, final_url=response.final_url)

    def parse_detail(self, markup: str, *, stub: PosterStub, final_url: str | None = None) -> PaperRecord:
        parser = parse_html_metadata(markup)
        creative_work = _find_creative_work(parser.json_ld)
        title = clean_text(_json_value(creative_work, "name")) or stub.title or _clean_page_title(parser.title)
        authors = _extract_authors(creative_work)
        abstract = parser.abstract
        links = [join_url(final_url or stub.detail_url, link.href) for link in parser.links]
        openreview_url = _first_matching_url(links, "openreview.net/forum")
        pdf_url = _first_matching_url(links, "openreview.net/pdf") or _first_matching_url(links, ".pdf")
        extra = {
            "icml_virtual_url": final_url or stub.detail_url,
            "icml_year": self.year,
            "source_parser": "icml_virtual",
            "pdf_discovery_status": "found" if pdf_url else "missing_on_virtual_page",
        }

        return PaperRecord(
            paper_id=stub.paper_id,
            source="accepted",
            decision_label="accepted",
            title=title,
            abstract=abstract,
            authors=authors,
            forum_url=openreview_url or (final_url or stub.detail_url),
            pdf_url=pdf_url,
            extra=extra,
        )

    def scrape_records(self, *, limit: int | None = None, raw_html_dir: Path | None = None) -> Iterator[PaperRecord]:
        for idx, stub in enumerate(self.discover_index()):
            if limit is not None and idx >= limit:
                break
            yield self.scrape_detail(stub, raw_html_dir=raw_html_dir)


def _find_creative_work(payloads: list[Any]) -> dict[str, Any]:
    for node in iter_json_ld_nodes(payloads):
        node_type = node.get("@type")
        if node_type == "CreativeWork" or (isinstance(node_type, list) and "CreativeWork" in node_type):
            return node
    return {}


def _json_value(node: dict[str, Any], key: str) -> str | None:
    value = node.get(key)
    if isinstance(value, str):
        return value
    return None


def _extract_authors(node: dict[str, Any]) -> list[str]:
    authors = node.get("author", [])
    if isinstance(authors, dict):
        authors = [authors]
    if not isinstance(authors, list):
        return []
    names: list[str] = []
    for author in authors:
        if isinstance(author, dict) and isinstance(author.get("name"), str):
            names.append(author["name"])
        elif isinstance(author, str):
            names.append(author)
    return [name for name in (clean_text(name) for name in names) if name]


def _first_matching_url(urls: Iterable[str], needle: str) -> str | None:
    for url in urls:
        if needle in url:
            return url
    return None


def _clean_page_title(title: str | None) -> str | None:
    if not title:
        return None
    title = re.sub(r"^ICML\s+Poster\s+", "", title).strip()
    return clean_text(title)


def hostname(url: str) -> str | None:
    parsed = urlparse(url)
    return parsed.hostname


def records_from_virtual_json(
    payload: dict[str, Any],
    *,
    abstracts: dict[str, str],
    year: int,
    event_types: set[str],
    require_decision: bool,
    limit: int | None,
) -> tuple[list[PaperRecord], dict[str, Any]]:
    rows = payload.get("results")
    if not isinstance(rows, list):
        rows = []
    raw_event_type_counter = Counter(
        str(row.get("eventtype") or row.get("event_type") or "")
        for row in rows
        if isinstance(row, dict)
    )
    raw_decision_counter = Counter(
        str(row.get("decision") or "missing")
        for row in rows
        if isinstance(row, dict)
    )
    selected: list[dict[str, Any]] = []
    excluded_by_event_type = 0
    excluded_missing_decision = 0
    seen: set[str] = set()
    duplicate_ids = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        event_type = str(row.get("eventtype") or row.get("event_type") or "")
        if event_types and event_type not in event_types:
            excluded_by_event_type += 1
            continue
        if require_decision and not row.get("decision"):
            excluded_missing_decision += 1
            continue
        paper_id = str(row.get("id") or "")
        if not paper_id:
            continue
        if paper_id in seen:
            duplicate_ids += 1
            continue
        seen.add(paper_id)
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break

    rows_by_id = {
        str(row.get("id")): row
        for row in rows
        if isinstance(row, dict) and row.get("id") is not None
    }
    oral_rows_by_poster_id = {
        str(poster_id): row
        for row in rows
        if isinstance(row, dict)
        and str(row.get("eventtype") or row.get("event_type") or "").lower() == "oral"
        for poster_id in row.get("related_events_ids", []) or []
    }
    records: list[PaperRecord] = []
    for row in selected:
        record = event_row_to_record(row, abstract=abstracts.get(str(row.get("id") or "")), year=year)
        oral_row = related_oral_row(
            row,
            rows_by_id=rows_by_id,
            oral_rows_by_poster_id=oral_rows_by_poster_id,
        )
        if oral_row:
            attach_oral_metadata(record, oral_row=oral_row, year=year)
        records.append(record)
    selected_event_type_counter = Counter(str(row.get("eventtype") or row.get("event_type") or "") for row in selected)
    selected_decision_counter = Counter(str(row.get("decision") or "missing") for row in selected)
    report = {
        "source": "icml_virtual_json",
        "year": year,
        "payload_count": payload.get("count"),
        "raw_result_count": len(rows),
        "event_type_counts": dict(sorted(selected_event_type_counter.items())),
        "decision_counts": dict(sorted(selected_decision_counter.items())),
        "raw_event_type_counts": dict(sorted(raw_event_type_counter.items())),
        "raw_decision_counts": dict(sorted(raw_decision_counter.items())),
        "selected_event_types": sorted(event_types),
        "require_decision": require_decision,
        "selected_records": len(records),
        "excluded_by_event_type": excluded_by_event_type,
        "excluded_missing_decision": excluded_missing_decision,
        "duplicate_ids_skipped": duplicate_ids,
        "abstract_count": len(abstracts),
        "records_with_abstract": sum(1 for record in records if record.abstract),
        "records_with_forum_url": sum(1 for record in records if record.forum_url and "openreview.net" in record.forum_url),
        "records_with_pdf_url": sum(1 for record in records if record.pdf_url),
        "records_missing_decision_label": sum(1 for record in records if not record.decision_label),
        "spotlight_records": sum(bool(record.extra.get("is_spotlight")) for record in records),
        "oral_presentation_records": sum(bool(record.extra.get("is_oral")) for record in records),
        "limit": limit,
    }
    return records, report


def event_row_to_record(row: dict[str, Any], *, abstract: str | None, year: int) -> PaperRecord:
    paper_id = str(row.get("id") or "")
    paper_url = clean_url(row.get("paper_url"))
    pdf_url = clean_url(row.get("paper_pdf_url")) or eventmedia_pdf_url(row)
    forum_url = paper_url or eventmedia_openreview_url(row) or virtual_url(row, year=year)
    event_type = str(row.get("eventtype") or row.get("event_type") or "")
    decision = clean_text(str(row.get("decision") or "")) or None
    authors = [
        clean_text(str(author.get("fullname") or ""))
        for author in row.get("authors", [])
        if isinstance(author, dict) and author.get("fullname")
    ]
    related_event_ids = [item for item in row.get("related_events_ids", []) if item is not None]
    extra = {
        "icml_virtual_url": virtual_url(row, year=year),
        "icml_year": year,
        "source_parser": "icml_virtual_json",
        "event_type": event_type,
        "event_type_detail": row.get("event_type"),
        "decision": row.get("decision"),
        "session": row.get("session"),
        "room_name": row.get("room_name"),
        "starttime": row.get("starttime"),
        "endtime": row.get("endtime"),
        "topic": row.get("topic"),
        "keywords": row.get("keywords") or [],
        "sourceid": row.get("sourceid"),
        "sourceurl": row.get("sourceurl"),
        "paper_url": paper_url,
        "paper_pdf_url": pdf_url,
        "openreview_forum_id": openreview_id_from_url(paper_url),
        "related_events_ids": related_event_ids,
        "parent_id": row.get("parent_id"),
        "pdf_discovery_status": "found_icml_json" if pdf_url else "missing_on_icml_json",
        "decision_label_missing": decision is None,
        "acceptance_tier": acceptance_tier(decision),
        "is_spotlight": acceptance_tier(decision) == "spotlight",
        "is_oral": event_type.lower() == "oral",
        "presentation_type": event_type.lower() or None,
    }
    return PaperRecord(
        paper_id=paper_id,
        source="accepted",
        decision_label=decision,
        title=clean_text(str(row.get("name") or "")) or None,
        abstract=clean_text(abstract) if abstract else None,
        authors=authors,
        forum_url=forum_url,
        pdf_url=pdf_url,
        topic_cluster=clean_text(str(row.get("topic") or "")) or None,
        extra=extra,
    )


def acceptance_tier(decision: str | None) -> str | None:
    normalized = (decision or "").lower()
    if "spotlight" in normalized:
        return "spotlight"
    if "regular" in normalized or "poster" in normalized:
        return "regular"
    return None


def related_oral_row(
    row: dict[str, Any],
    *,
    rows_by_id: dict[str, dict[str, Any]],
    oral_rows_by_poster_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    for related_id in row.get("related_events_ids", []) or []:
        related = rows_by_id.get(str(related_id))
        if related and str(related.get("eventtype") or related.get("event_type") or "").lower() == "oral":
            return related

    return oral_rows_by_poster_id.get(str(row.get("id") or ""))


def attach_oral_metadata(record: PaperRecord, *, oral_row: dict[str, Any], year: int) -> None:
    record.extra.update(
        {
            "is_oral": True,
            "presentation_type": "oral",
            "oral_event_id": str(oral_row.get("id") or "") or None,
            "oral_virtual_url": virtual_url(oral_row, year=year),
            "oral_session": oral_row.get("session"),
            "oral_room_name": oral_row.get("room_name"),
            "oral_starttime": oral_row.get("starttime"),
            "oral_endtime": oral_row.get("endtime"),
        }
    )


def virtual_url(row: dict[str, Any], *, year: int) -> str | None:
    value = clean_url(row.get("virtualsite_url"))
    if not value:
        return None
    return join_url(f"https://icml.cc/virtual/{year}/", value)


def clean_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def eventmedia_openreview_url(row: dict[str, Any]) -> str | None:
    for item in row.get("eventmedia", []) or []:
        if not isinstance(item, dict):
            continue
        uri = clean_url(item.get("uri"))
        if uri and "openreview.net/forum" in uri:
            return uri
    return None


def eventmedia_pdf_url(row: dict[str, Any]) -> str | None:
    for item in row.get("eventmedia", []) or []:
        if not isinstance(item, dict):
            continue
        uri = clean_url(item.get("uri"))
        name = str(item.get("name") or "").lower()
        if uri and ("paper" in name or "pdf" in name) and (".pdf" in uri or "openreview.net/pdf" in uri):
            return uri
    return None


def openreview_id_from_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urllib.parse.urlparse(url)
    values = urllib.parse.parse_qs(parsed.query).get("id")
    return values[0] if values else None
