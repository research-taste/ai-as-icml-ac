from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from icml_ai_ac.http import HttpClient
from icml_ai_ac.models import PaperRecord
from icml_ai_ac.storage import ensure_parent, write_json


ARXIV_API_BASE = "https://export.arxiv.org/api/query"
ARXIV_WEB_BASE = "https://arxiv.org"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
MIN_TITLE_TERMS = 4
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "the",
    "to",
    "via",
    "with",
}
CONFIDENCE_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3, "exact": 4}


class ArxivQueryBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ArxivEntry:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str | None
    published: str | None
    updated: str | None
    categories: list[str]
    abs_url: str
    pdf_url: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "published": self.published,
            "updated": self.updated,
            "categories": self.categories,
            "abs_url": self.abs_url,
            "pdf_url": self.pdf_url,
        }


@dataclass(slots=True)
class ArxivClientStats:
    searches: int = 0
    network_requests: int = 0
    cache_hits: int = 0
    cache_writes: int = 0
    rate_limit_cooldowns: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "searches": self.searches,
            "network_requests": self.network_requests,
            "cache_hits": self.cache_hits,
            "cache_writes": self.cache_writes,
            "rate_limit_cooldowns": self.rate_limit_cooldowns,
        }


@dataclass(frozen=True, slots=True)
class ArxivMatch:
    entry: ArxivEntry
    score: float
    confidence: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.entry.to_dict(),
            "match_score": round(self.score, 4),
            "match_confidence": self.confidence,
            "match_evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class ArxivResolution:
    paper_id: str
    title: str | None
    best_match: ArxivMatch | None
    candidates: list[ArxivMatch]
    queries: list[str]

    def candidate_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for rank, match in enumerate(self.candidates, start=1):
            rows.append(
                {
                    "paper_id": self.paper_id,
                    "paper_title": self.title,
                    "candidate_rank": rank,
                    **match.to_dict(),
                }
            )
        return rows


class ArxivClient:
    def __init__(
        self,
        *,
        api_base: str = ARXIV_API_BASE,
        http_client: HttpClient | None = None,
        cache_dir: Path | None = None,
        refresh_cache: bool = False,
        max_network_requests: int | None = None,
        rate_limit_cooldown_seconds: float = 180.0,
        rate_limit_extra_retries: int = 2,
    ) -> None:
        self.api_base = api_base
        self.http = http_client or HttpClient(polite_delay_seconds=12.0, backoff_seconds=30.0)
        self.cache_dir = cache_dir
        self.refresh_cache = refresh_cache
        self.max_network_requests = max_network_requests
        self.rate_limit_cooldown_seconds = rate_limit_cooldown_seconds
        self.rate_limit_extra_retries = rate_limit_extra_retries
        self.stats = ArxivClientStats()

    def search(self, query: str, *, max_results: int = 10) -> list[ArxivEntry]:
        self.stats.searches += 1
        params = {
            "search_query": query,
            "start": "0",
            "max_results": str(max_results),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        url = f"{self.api_base}?{urllib.parse.urlencode(params)}"
        cached = self._read_cached_search(query=query, max_results=max_results)
        if cached is not None:
            self.stats.cache_hits += 1
            return parse_arxiv_feed(cached)

        last_rate_limit: urllib.error.HTTPError | None = None
        for attempt in range(self.rate_limit_extra_retries + 1):
            self._check_network_budget()
            self.stats.network_requests += 1
            try:
                response = self.http.fetch(url, accept="application/atom+xml,application/xml,*/*")
            except urllib.error.HTTPError as exc:
                if exc.code != 429 or attempt >= self.rate_limit_extra_retries:
                    raise
                last_rate_limit = exc
                self.stats.rate_limit_cooldowns += 1
                time.sleep(max(self.rate_limit_cooldown_seconds, self.http.polite_delay_seconds))
                self.http.polite_delay_seconds = max(self.http.polite_delay_seconds, self.rate_limit_cooldown_seconds / 2)
                continue
            self._write_cached_search(query=query, max_results=max_results, payload=response.text)
            return parse_arxiv_feed(response.text)
        if last_rate_limit is not None:
            raise last_rate_limit
        raise RuntimeError("arXiv search failed without an exception")

    def _check_network_budget(self) -> None:
        if self.max_network_requests is None:
            return
        if self.stats.network_requests >= self.max_network_requests:
            raise ArxivQueryBudgetExceeded(
                f"arXiv network query budget exceeded: {self.max_network_requests}"
            )

    def _cache_path(self, *, query: str, max_results: int) -> Path | None:
        if self.cache_dir is None:
            return None
        key = json.dumps(
            {
                "api_base": self.api_base,
                "max_results": max_results,
                "query": query,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _read_cached_search(self, *, query: str, max_results: int) -> str | None:
        path = self._cache_path(query=query, max_results=max_results)
        if path is None or self.refresh_cache or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        text = payload.get("response_text") if isinstance(payload, dict) else None
        return text if isinstance(text, str) else None

    def _write_cached_search(self, *, query: str, max_results: int, payload: str) -> None:
        path = self._cache_path(query=query, max_results=max_results)
        if path is None:
            return
        ensure_parent(path)
        write_json(
            path,
            {
                "api_base": self.api_base,
                "fetched_at": datetime.now().astimezone().isoformat(),
                "max_results": max_results,
                "query": query,
                "response_text": payload,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
        )
        self.stats.cache_writes += 1

    def resolve_record(
        self,
        record: PaperRecord,
        *,
        max_results: int = 10,
        deep_search: bool = False,
        deep_max_results: int = 50,
        published_from: str | None = None,
        published_to: str | None = None,
    ) -> ArxivResolution:
        queries = build_arxiv_queries(record, published_from=published_from, published_to=published_to)
        by_id: dict[str, ArxivEntry] = {}
        executed_queries: list[str] = []

        def run_queries(query_list: list[str], *, result_limit: int) -> bool:
            for query in query_list:
                if query in executed_queries:
                    continue
                executed_queries.append(query)
                for entry in self.search(query, max_results=result_limit):
                    by_id.setdefault(entry.arxiv_id, entry)
                if any(is_confident_enough(score_arxiv_match(record, entry).confidence, "high") for entry in by_id.values()):
                    return True
            return False

        found_confident = run_queries(queries, result_limit=max_results)
        if deep_search and not found_confident:
            run_queries(
                build_deep_arxiv_queries(record, published_from=published_from, published_to=published_to),
                result_limit=deep_max_results,
            )
        matches = [score_arxiv_match(record, entry, published_from=published_from, published_to=published_to) for entry in by_id.values()]
        matches.sort(key=lambda match: (CONFIDENCE_ORDER[match.confidence], match.score), reverse=True)
        return ArxivResolution(
            paper_id=record.paper_id,
            title=record.title,
            best_match=matches[0] if matches else None,
            candidates=matches,
            queries=executed_queries,
        )


def parse_arxiv_feed(payload: str) -> list[ArxivEntry]:
    root = ET.fromstring(payload)
    entries: list[ArxivEntry] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        abs_url = text(entry, "atom:id") or ""
        arxiv_id = arxiv_id_from_abs_url(abs_url)
        title = compact_space(text(entry, "atom:title") or "")
        abstract = compact_space(text(entry, "atom:summary") or "")
        authors = [
            compact_space(name)
            for author in entry.findall("atom:author", ATOM_NS)
            if (name := text(author, "atom:name"))
        ]
        categories = [
            term
            for category in entry.findall("atom:category", ATOM_NS)
            if (term := category.attrib.get("term"))
        ]
        entries.append(
            ArxivEntry(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract or None,
                published=text(entry, "atom:published"),
                updated=text(entry, "atom:updated"),
                categories=categories,
                abs_url=abs_url or f"{ARXIV_WEB_BASE}/abs/{arxiv_id}",
                pdf_url=pdf_url_from_entry(entry, arxiv_id=arxiv_id),
            )
        )
    return entries


def text(node: ET.Element, path: str) -> str | None:
    child = node.find(path, ATOM_NS)
    return child.text.strip() if child is not None and child.text else None


def pdf_url_from_entry(entry: ET.Element, *, arxiv_id: str) -> str | None:
    for link in entry.findall("atom:link", ATOM_NS):
        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
            href = link.attrib.get("href")
            if href:
                return href
    return f"{ARXIV_WEB_BASE}/pdf/{arxiv_id}"


def arxiv_id_from_abs_url(abs_url: str) -> str:
    path = urllib.parse.urlparse(abs_url).path.rstrip("/")
    value = path.split("/abs/", 1)[-1] if "/abs/" in path else path.rsplit("/", 1)[-1]
    return re.sub(r"v\d+$", "", value)


def build_arxiv_queries(
    record: PaperRecord,
    *,
    published_from: str | None = None,
    published_to: str | None = None,
) -> list[str]:
    title = record.title or ""
    terms = title_terms(title)
    title_query = f'ti:"{escape_query_phrase(title)}"' if title else ""
    keyword_query = " AND ".join(f"ti:{term}" for term in terms[:8])
    authors = author_last_names(record.authors)
    author_query = " AND ".join(f"au:{name}" for name in authors[:2])
    date_query = submitted_date_query(published_from, published_to)
    queries = []
    fallback_query = " AND ".join(part for part in [keyword_query, author_query] if part)
    for base in [title_query, fallback_query]:
        if not base:
            continue
        query = " AND ".join(part for part in [base, date_query] if part)
        if query not in queries:
            queries.append(query)
    return queries


def build_deep_arxiv_queries(
    record: PaperRecord,
    *,
    published_from: str | None = None,
    published_to: str | None = None,
) -> list[str]:
    terms = list(dict.fromkeys(title_terms(record.title or "")))
    authors = author_last_names(record.authors)
    date_query = submitted_date_query(published_from, published_to)
    bases: list[str] = []
    if len(authors) >= 2:
        bases.append(" AND ".join(f"au:{name}" for name in authors[:2]))
    if authors and terms:
        bases.append(" AND ".join([*(f"all:{term}" for term in terms[:5]), f"au:{authors[0]}"]))
    if terms:
        bases.append(" AND ".join(f"all:{term}" for term in terms[:4]))
    queries: list[str] = []
    for base in bases:
        query = " AND ".join(part for part in [base, date_query] if part)
        if query and query not in queries:
            queries.append(query)
    return queries


def submitted_date_query(published_from: str | None, published_to: str | None) -> str:
    if not published_from and not published_to:
        return ""
    start = arxiv_date_bound(published_from, default="000101010000")
    end = arxiv_date_bound(published_to, default="999912312359")
    return f"submittedDate:[{start} TO {end}]"


def arxiv_date_bound(value: str | None, *, default: str) -> str:
    if not value:
        return default
    parsed = date.fromisoformat(value)
    suffix = "0000" if default.endswith("0000") else "2359"
    return parsed.strftime("%Y%m%d") + suffix


def score_arxiv_match(
    record: PaperRecord,
    entry: ArxivEntry,
    *,
    published_from: str | None = None,
    published_to: str | None = None,
) -> ArxivMatch:
    title_score = title_similarity(record.title or "", entry.title)
    author_score = author_overlap(record.authors, entry.authors)
    abstract_score = text_jaccard(record.abstract or "", entry.abstract or "")
    date_status = date_match_status(entry.published, published_from=published_from, published_to=published_to)
    weights = {"title": 0.68, "authors": 0.18, "abstract": 0.14}
    if not record.authors:
        weights["title"] += weights.pop("authors")
        author_score = 0.0
    if not record.abstract or not entry.abstract:
        weights["title"] += weights.pop("abstract")
        abstract_score = 0.0
    score = sum(
        {
            "title": title_score,
            "authors": author_score,
            "abstract": abstract_score,
        }[key]
        * weight
        for key, weight in weights.items()
    )
    if date_status == "outside_range":
        score *= 0.92
    confidence = confidence_for_match(
        title_exact=normalize_title(record.title or "") == normalize_title(entry.title),
        title_score=title_score,
        author_score=author_score,
        abstract_score=abstract_score,
        score=score,
        has_authors=bool(record.authors),
    )
    evidence = {
        "title_similarity": round(title_score, 4),
        "author_overlap": round(author_score, 4),
        "abstract_similarity": round(abstract_score, 4),
        "date_status": date_status,
        "record_author_last_names": author_last_names(record.authors),
        "arxiv_author_last_names": author_last_names(entry.authors),
    }
    return ArxivMatch(entry=entry, score=max(0.0, min(1.0, score)), confidence=confidence, evidence=evidence)


def confidence_for_match(
    *,
    title_exact: bool,
    title_score: float,
    author_score: float,
    abstract_score: float,
    score: float,
    has_authors: bool,
) -> str:
    author_ok = author_score >= 0.5 or not has_authors
    if title_exact and author_ok:
        return "exact"
    if has_authors and author_score >= 0.8 and abstract_score >= 0.70 and title_score >= 0.70:
        return "high"
    if score >= 0.88 and title_score >= 0.84 and (author_ok or abstract_score >= 0.25):
        return "high"
    if score >= 0.74 and title_score >= 0.68:
        return "medium"
    if score >= 0.55 and title_score >= 0.50:
        return "low"
    return "none"


def is_confident_enough(confidence: str, minimum: str) -> bool:
    return CONFIDENCE_ORDER.get(confidence, 0) >= CONFIDENCE_ORDER.get(minimum, 0)


def title_similarity(left: str, right: str) -> float:
    norm_left = normalize_title(left)
    norm_right = normalize_title(right)
    if not norm_left or not norm_right:
        return 0.0
    if norm_left == norm_right:
        return 1.0
    ratio = SequenceMatcher(None, norm_left, norm_right).ratio()
    tokens_left = set(norm_left.split())
    tokens_right = set(norm_right.split())
    jaccard = len(tokens_left & tokens_right) / len(tokens_left | tokens_right) if tokens_left and tokens_right else 0.0
    return 0.65 * ratio + 0.35 * jaccard


def author_overlap(left: Iterable[str], right: Iterable[str]) -> float:
    left_names = set(author_last_names(left))
    right_names = set(author_last_names(right))
    if not left_names or not right_names:
        return 0.0
    return len(left_names & right_names) / min(len(left_names), len(right_names))


def text_jaccard(left: str, right: str) -> float:
    left_tokens = set(content_tokens(left))
    right_tokens = set(content_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def content_tokens(text_value: str) -> list[str]:
    return [token for token in normalize_title(text_value).split() if token not in STOPWORDS and len(token) > 2]


def title_terms(title: str) -> list[str]:
    terms = [token for token in content_tokens(title) if not token.isdigit()]
    return terms[: max(MIN_TITLE_TERMS, min(len(terms), 10))]


def normalize_title(title: str) -> str:
    normalized = unicodedata.normalize("NFKD", title)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return compact_space(normalized)


def author_last_names(authors: Iterable[str]) -> list[str]:
    names: list[str] = []
    for author in authors:
        cleaned = normalize_title(str(author))
        if not cleaned:
            continue
        names.append(cleaned.split()[-1])
    return names


def compact_space(value: str) -> str:
    return " ".join(value.split())


def escape_query_phrase(value: str) -> str:
    return value.replace('"', "")


def date_match_status(arxiv_published: str | None, *, published_from: str | None, published_to: str | None) -> str:
    if not arxiv_published or (not published_from and not published_to):
        return "not_checked"
    try:
        published = datetime.fromisoformat(arxiv_published.replace("Z", "+00:00")).date()
    except ValueError:
        return "unknown"
    if published_from and published < date.fromisoformat(published_from):
        return "outside_range"
    if published_to and published > date.fromisoformat(published_to):
        return "outside_range"
    return "inside_range"


def arxiv_pdf_filename(arxiv_id: str) -> str:
    safe = arxiv_id.replace("/", "_").replace(":", "_")
    return f"{safe}.pdf"
