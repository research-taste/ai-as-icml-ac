from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ArxivAuditThresholds:
    probable_author_overlap: float = 0.8
    probable_abstract_similarity: float = 0.5
    probable_title_similarity: float = 0.5
    possible_author_overlap: float = 0.8
    possible_match_score: float = 0.6


def audit_arxiv_resolution(
    manifest_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    thresholds: ArxivAuditThresholds = ArxivAuditThresholds(),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidate_rows:
        paper_id = candidate.get("paper_id")
        if isinstance(paper_id, str) and paper_id:
            candidates_by_paper[paper_id].append(candidate)

    audit_rows: list[dict[str, Any]] = []
    for row in manifest_rows:
        paper_id = row.get("paper_id")
        if not isinstance(paper_id, str) or not paper_id:
            continue
        candidates = candidates_by_paper.get(paper_id, [])
        best_candidate = best_candidate_for_audit(row, candidates)
        audit_rows.append(
            {
                "paper_id": paper_id,
                "title": row.get("title"),
                "arxiv_audit_class": classify_arxiv_row(row, best_candidate, thresholds=thresholds),
                "resolution_status": arxiv_resolution_status(row),
                "match_confidence": arxiv_match_confidence(row),
                "arxiv_id": arxiv_id(row, best_candidate),
                "arxiv_title": best_candidate.get("title") if best_candidate else None,
                "match_score": best_candidate.get("match_score") if best_candidate else None,
                "title_similarity": match_evidence_value(best_candidate, "title_similarity"),
                "author_overlap": match_evidence_value(best_candidate, "author_overlap"),
                "abstract_similarity": match_evidence_value(best_candidate, "abstract_similarity"),
                "candidate_count": len(candidates),
            }
        )

    counts = Counter(row["arxiv_audit_class"] for row in audit_rows)
    report = {
        "records": len(audit_rows),
        "class_counts": dict(sorted(counts.items())),
        "thresholds": {
            "probable_author_overlap": thresholds.probable_author_overlap,
            "probable_abstract_similarity": thresholds.probable_abstract_similarity,
            "probable_title_similarity": thresholds.probable_title_similarity,
            "possible_author_overlap": thresholds.possible_author_overlap,
            "possible_match_score": thresholds.possible_match_score,
        },
    }
    return audit_rows, report


def classify_arxiv_row(
    manifest_row: dict[str, Any],
    best_candidate: dict[str, Any] | None,
    *,
    thresholds: ArxivAuditThresholds = ArxivAuditThresholds(),
) -> str:
    status = arxiv_resolution_status(manifest_row)
    confidence = arxiv_match_confidence(manifest_row)
    if status == "matched" and confidence in {"exact", "high"}:
        return "automatic_high_confidence"
    if best_candidate is None:
        return "likely_absent_no_candidate"

    title_similarity = float(match_evidence_value(best_candidate, "title_similarity") or 0.0)
    author_overlap = float(match_evidence_value(best_candidate, "author_overlap") or 0.0)
    abstract_similarity = float(match_evidence_value(best_candidate, "abstract_similarity") or 0.0)
    match_score = float(best_candidate.get("match_score") or 0.0)

    if (
        author_overlap >= thresholds.probable_author_overlap
        and abstract_similarity >= thresholds.probable_abstract_similarity
        and title_similarity >= thresholds.probable_title_similarity
    ):
        return "review_probable_prior_title"
    if author_overlap >= thresholds.possible_author_overlap and match_score >= thresholds.possible_match_score:
        return "review_possible_related_or_prior"
    return "likely_absent_weak_related_hits"


def best_candidate_for_audit(row: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    extra = row.get("extra")
    if isinstance(extra, dict) and isinstance(extra.get("arxiv_best_match"), dict):
        return extra["arxiv_best_match"]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: (
            confidence_rank(str(candidate.get("match_confidence") or "")),
            float(candidate.get("match_score") or 0.0),
            float(match_evidence_value(candidate, "abstract_similarity") or 0.0),
        ),
    )


def arxiv_resolution_status(row: dict[str, Any]) -> str:
    extra = row.get("extra")
    if not isinstance(extra, dict):
        return ""
    value = extra.get("arxiv_resolution_status")
    return value if isinstance(value, str) else ""


def arxiv_match_confidence(row: dict[str, Any]) -> str:
    extra = row.get("extra")
    if not isinstance(extra, dict):
        return ""
    value = extra.get("arxiv_match_confidence")
    return value if isinstance(value, str) else ""


def arxiv_id(row: dict[str, Any], best_candidate: dict[str, Any] | None) -> str | None:
    extra = row.get("extra")
    if isinstance(extra, dict) and isinstance(extra.get("arxiv_id"), str):
        return extra["arxiv_id"]
    if best_candidate and isinstance(best_candidate.get("arxiv_id"), str):
        return best_candidate["arxiv_id"]
    return None


def match_evidence_value(candidate: dict[str, Any] | None, key: str) -> float | None:
    if not candidate:
        return None
    evidence = candidate.get("match_evidence")
    if not isinstance(evidence, dict):
        return None
    value = evidence.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def confidence_rank(confidence: str) -> int:
    return {"none": 0, "low": 1, "medium": 2, "high": 3, "exact": 4}.get(confidence, 0)
