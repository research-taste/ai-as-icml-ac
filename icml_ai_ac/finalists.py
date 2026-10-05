from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from icml_ai_ac.shortlist import shortlist_sort_key
from icml_ai_ac.storage import read_jsonl, write_json, write_jsonl


@dataclass(slots=True)
class FinalistSelectionConfig:
    limit: int
    semifinal_top: int | None
    cheap_top: int
    min_per_class: int
    disagreement_saves: int
    judge_disagreement_saves: int = 0


def select_finalists(
    *,
    cheap_path: Path,
    semifinal_path: Path,
    out: Path,
    report: Path | None,
    config: FinalistSelectionConfig,
) -> dict[str, Any]:
    if config.limit < 1:
        raise ValueError("limit must be positive")
    nonnegative_fields = {
        "cheap_top": config.cheap_top,
        "min_per_class": config.min_per_class,
        "disagreement_saves": config.disagreement_saves,
        "judge_disagreement_saves": config.judge_disagreement_saves,
    }
    if config.semifinal_top is not None:
        nonnegative_fields["semifinal_top"] = config.semifinal_top
    for field_name, value in nonnegative_fields.items():
        if value < 0:
            raise ValueError(f"{field_name} must be nonnegative")
    cheap_rows = read_ranked_rows(cheap_path)
    semifinal_rows = read_ranked_rows(semifinal_path)
    cheap_by_id = {str(row.get("paper_id") or ""): row for row in cheap_rows if row.get("paper_id")}
    semifinal_by_id = {str(row.get("paper_id") or ""): row for row in semifinal_rows if row.get("paper_id")}
    candidate_ids = [paper_id for paper_id in semifinal_by_id if paper_id in cheap_by_id]
    if not candidate_ids:
        raise ValueError("no overlapping papers between cheap and semifinal rankings")

    cheap_rank = {paper_id: rank for rank, paper_id in enumerate(ranked_ids(cheap_rows), start=1)}
    semifinal_rank = {paper_id: rank for rank, paper_id in enumerate(ranked_ids(semifinal_rows), start=1)}
    cheap_count = len(cheap_rank)
    semifinal_count = len(semifinal_rank)

    selected: dict[str, dict[str, Any]] = {}
    semifinal_top = config.semifinal_top
    if semifinal_top is None:
        semifinal_top = max(1, min(config.limit, int(round(config.limit * 0.75))))

    add_by_order(
        selected,
        [paper_id for paper_id in ranked_ids(semifinal_rows) if paper_id in candidate_ids],
        limit=min(semifinal_top, config.limit),
        reason="semifinal_top",
    )
    add_by_order(
        selected,
        [paper_id for paper_id in ranked_ids(cheap_rows) if paper_id in candidate_ids],
        limit=min(config.cheap_top, config.limit),
        reason="cheap_top_save",
    )
    for contribution_class, paper_ids in class_leaders(
        candidate_ids=candidate_ids,
        cheap_by_id=cheap_by_id,
        semifinal_by_id=semifinal_by_id,
        semifinal_rank=semifinal_rank,
        cheap_rank=cheap_rank,
    ).items():
        for paper_id in paper_ids[: config.min_per_class]:
            add_selected(selected, paper_id, f"class_leader:{contribution_class}")

    disagreement_order = sorted(
        candidate_ids,
        key=lambda paper_id: (
            -disagreement_score(
                paper_id,
                cheap_rank=cheap_rank,
                semifinal_rank=semifinal_rank,
                cheap_count=cheap_count,
                semifinal_count=semifinal_count,
            ),
            cheap_rank.get(paper_id, 10**9),
            semifinal_rank.get(paper_id, 10**9),
            paper_id,
        ),
    )
    add_by_order(
        selected,
        [paper_id for paper_id in disagreement_order if disagreement_score(
            paper_id,
            cheap_rank=cheap_rank,
            semifinal_rank=semifinal_rank,
            cheap_count=cheap_count,
            semifinal_count=semifinal_count,
        ) > 0],
        limit=config.disagreement_saves,
        reason="cheap_semifinal_disagreement_save",
    )

    judge_disagreement_order = sorted(
        candidate_ids,
        key=lambda paper_id: (
            -semifinal_judge_disagreement(semifinal_by_id[paper_id]),
            source_best_rank_percentile(semifinal_by_id[paper_id]),
            semifinal_rank.get(paper_id, 10**9),
            paper_id,
        ),
    )
    add_by_order(
        selected,
        [
            paper_id
            for paper_id in judge_disagreement_order
            if semifinal_judge_disagreement(semifinal_by_id[paper_id]) > 0
        ],
        limit=config.judge_disagreement_saves,
        reason="semifinal_judge_disagreement_save",
    )

    if len(selected) > config.limit:
        raise ValueError(
            f"configured preservation rules selected {len(selected)} unique papers, "
            f"exceeding the finalist limit of {config.limit}; reduce semifinal/save/class quotas"
        )

    fill_order = sorted(
        candidate_ids,
        key=lambda paper_id: (
            semifinal_rank.get(paper_id, 10**9),
            cheap_rank.get(paper_id, 10**9),
            paper_id,
        ),
    )
    for paper_id in fill_order:
        if len(selected) >= config.limit:
            break
        if paper_id in selected:
            continue
        add_selected(selected, paper_id, "semifinal_fill")

    rows = [
        finalist_row(
            paper_id,
            cheap_row=cheap_by_id[paper_id],
            semifinal_row=semifinal_by_id[paper_id],
            cheap_rank=cheap_rank.get(paper_id, 10**9),
            semifinal_rank=semifinal_rank.get(paper_id, 10**9),
            cheap_count=cheap_count,
            semifinal_count=semifinal_count,
            reasons=payload["selection_reasons"],
        )
        for paper_id, payload in selected.items()
    ]
    rows.sort(key=finalist_sort_key)
    rows = rows[: config.limit]
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["finalist_rank"] = rank

    write_jsonl(out, rows)
    payload = {
        "cheap_path": str(cheap_path),
        "semifinal_path": str(semifinal_path),
        "out": str(out),
        "limit": config.limit,
        "semifinal_top": semifinal_top,
        "cheap_top": config.cheap_top,
        "min_per_class": config.min_per_class,
        "disagreement_saves": config.disagreement_saves,
        "judge_disagreement_saves": config.judge_disagreement_saves,
        "cheap_count": cheap_count,
        "semifinal_count": semifinal_count,
        "overlap_count": len(candidate_ids),
        "written_rows": len(rows),
        "selection_reason_counts": selection_reason_counts(rows),
        "top_paper_ids": [row["paper_id"] for row in rows[:20]],
    }
    if report is not None:
        write_json(report, payload)
    return payload


def read_ranked_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows = list(read_jsonl(path))
        if rows and not any(row.get("rank") for row in rows):
            rows = sorted(rows, key=shortlist_sort_key, reverse=True)
    else:
        if isinstance(payload, dict) and "ranking" in payload and isinstance(payload["ranking"], dict):
            payload = payload["ranking"]
        if isinstance(payload, dict) and isinstance(payload.get("ranked_papers"), list):
            rows = payload["ranked_papers"]
        elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            rows = payload["rows"]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError(f"Could not read ranked rows from {path}")
    ranked = [dict(row) for row in rows if isinstance(row, dict) and row.get("paper_id")]
    validate_unique_paper_ids(ranked, path=path)
    ranked.sort(key=lambda row: int(row.get("rank") or row.get("batch_rank") or row.get("derived_rank") or 10**9))
    for rank, row in enumerate(ranked, start=1):
        row.setdefault("rank", rank)
    return ranked


def validate_unique_paper_ids(rows: list[dict[str, Any]], *, path: Path) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        paper_id = str(row.get("paper_id") or "")
        if paper_id in seen:
            duplicates.append(paper_id)
        seen.add(paper_id)
    if duplicates:
        raise ValueError(f"{path}: duplicate paper_id values in ranking: {sorted(set(duplicates))}")


def ranked_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("paper_id")) for row in sorted(rows, key=lambda row: int(row.get("rank") or 10**9))]


def add_by_order(selected: dict[str, dict[str, Any]], paper_ids: list[str], *, limit: int, reason: str) -> None:
    for paper_id in paper_ids[:limit]:
        add_selected(selected, paper_id, reason)


def add_selected(selected: dict[str, dict[str, Any]], paper_id: str, reason: str) -> None:
    payload = selected.setdefault(paper_id, {"selection_reasons": []})
    reasons = payload["selection_reasons"]
    if reason not in reasons:
        reasons.append(reason)


def class_leaders(
    *,
    candidate_ids: list[str],
    cheap_by_id: dict[str, dict[str, Any]],
    semifinal_by_id: dict[str, dict[str, Any]],
    semifinal_rank: dict[str, int],
    cheap_rank: dict[str, int],
) -> dict[str, list[str]]:
    by_class: dict[str, list[str]] = {}
    for paper_id in candidate_ids:
        contribution_class = contribution_class_for(semifinal_by_id.get(paper_id, {}), cheap_by_id.get(paper_id, {}))
        by_class.setdefault(contribution_class, []).append(paper_id)
    for contribution_class, paper_ids in by_class.items():
        by_class[contribution_class] = sorted(
            paper_ids,
            key=lambda paper_id: (
                semifinal_rank.get(paper_id, 10**9),
                cheap_rank.get(paper_id, 10**9),
                paper_id,
            ),
        )
    return dict(sorted(by_class.items()))


def finalist_row(
    paper_id: str,
    *,
    cheap_row: dict[str, Any],
    semifinal_row: dict[str, Any],
    cheap_rank: int,
    semifinal_rank: int,
    cheap_count: int,
    semifinal_count: int,
    reasons: list[str],
) -> dict[str, Any]:
    contribution_class = contribution_class_for(semifinal_row, cheap_row)
    title = semifinal_row.get("title") or cheap_row.get("title")
    return {
        "status": "ok",
        "paper_id": paper_id,
        "title": title,
        "primary_contribution_class": contribution_class,
        "selection_reasons": sorted(reasons),
        "cheap_rank": cheap_rank,
        "semifinal_rank": semifinal_rank,
        "cheap_rank_percentile": round(rank_percentile(cheap_rank, cheap_count), 4),
        "semifinal_rank_percentile": round(rank_percentile(semifinal_rank, semifinal_count), 4),
        "cheap_semifinal_disagreement": round(
            disagreement_score(
                paper_id,
                cheap_rank={paper_id: cheap_rank},
                semifinal_rank={paper_id: semifinal_rank},
                cheap_count=cheap_count,
                semifinal_count=semifinal_count,
            ),
            4,
        ),
        "semifinal_judge_disagreement": semifinal_judge_disagreement(semifinal_row),
        "semifinal_source_ranks": semifinal_row.get("source_ranks") or {},
        "semifinal_source_rank_percentiles": semifinal_row.get("source_rank_percentiles") or {},
        "semifinal_source_judges": semifinal_row.get("source_judges") or {},
        "scores": {
            "contribution_profile": {
                "primary_contribution_class": contribution_class,
                "secondary_contribution_classes": [],
            },
            "scores": {
                "executive_ac_priority": selection_priority(cheap_rank, semifinal_rank, cheap_count, semifinal_count) * 10,
                "broader_science_impact_forecast": numeric_score(semifinal_row, "broad_scientific_impact_score")
                or numeric_score(cheap_row, "broader_science_impact_forecast"),
                "ml_field_impact_forecast": numeric_score(semifinal_row, "ml_field_impact_score")
                or numeric_score(cheap_row, "ml_field_impact_forecast"),
                "overall_significance": numeric_score(semifinal_row, "overall_priority_score")
                or numeric_score(cheap_row, "overall_significance"),
            },
            "ranking_signals": {
                "should_advance_to_strong_model": True,
            },
        },
        "semifinal_summary": {
            "ensemble_method": semifinal_row.get("ensemble_method"),
            "judge_count": semifinal_row.get("judge_count"),
            "consensus_score": semifinal_row.get("consensus_score"),
            "judge_disagreement": semifinal_judge_disagreement(semifinal_row),
            "source_judgments": summarized_source_judgments(semifinal_row),
            "why_ranked_here": semifinal_row.get("why_ranked_here"),
            "best_case_for_impact": semifinal_row.get("best_case_for_impact"),
            "main_risk": semifinal_row.get("main_risk"),
        },
    }


def finalist_sort_key(row: dict[str, Any]) -> tuple[int, int, float, str]:
    return (
        int(row.get("semifinal_rank") or 10**9),
        int(row.get("cheap_rank") or 10**9),
        -float(row.get("cheap_semifinal_disagreement") or 0.0),
        str(row.get("paper_id") or ""),
    )


def selection_reason_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for reason in row.get("selection_reasons", []):
            counts[str(reason)] = counts.get(str(reason), 0) + 1
    return dict(sorted(counts.items()))


def semifinal_judge_disagreement(row: dict[str, Any]) -> float:
    try:
        return round(float(row.get("semifinal_judge_disagreement") or 0.0), 6)
    except (TypeError, ValueError):
        return 0.0


def source_best_rank_percentile(row: dict[str, Any]) -> float:
    value = row.get("best_source_rank_percentile")
    try:
        return float(value) if value is not None else 1.0
    except (TypeError, ValueError):
        return 1.0


def summarized_source_judgments(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    source_rows = row.get("source_rows")
    if not isinstance(source_rows, dict):
        return {}
    return {
        str(label): {
            "rank": source_row.get("rank"),
            "overall_priority_score": source_row.get("overall_priority_score"),
            "broad_scientific_impact_score": source_row.get("broad_scientific_impact_score"),
            "ml_field_impact_score": source_row.get("ml_field_impact_score"),
            "why_ranked_here": source_row.get("why_ranked_here"),
            "best_case_for_impact": source_row.get("best_case_for_impact"),
            "main_risk": source_row.get("main_risk"),
        }
        for label, source_row in sorted(source_rows.items())
        if isinstance(source_row, dict)
    }


def contribution_class_for(primary_row: dict[str, Any], fallback_row: dict[str, Any]) -> str:
    for row in [primary_row, fallback_row]:
        value = row.get("primary_contribution_class") or row.get("contribution_class")
        if isinstance(value, str) and value:
            return value
        scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
        profile = scores.get("contribution_profile") if isinstance(scores.get("contribution_profile"), dict) else {}
        value = profile.get("primary_contribution_class")
        if isinstance(value, str) and value:
            return value
    return "other"


def disagreement_score(
    paper_id: str,
    *,
    cheap_rank: dict[str, int],
    semifinal_rank: dict[str, int],
    cheap_count: int,
    semifinal_count: int,
) -> float:
    cheap_pct = rank_percentile(cheap_rank.get(paper_id, cheap_count), cheap_count)
    semifinal_pct = rank_percentile(semifinal_rank.get(paper_id, semifinal_count), semifinal_count)
    return max(0.0, semifinal_pct - cheap_pct)


def rank_percentile(rank: int, count: int) -> float:
    if count <= 1:
        return 0.0
    return max(0.0, min(1.0, (rank - 1) / (count - 1)))


def selection_priority(cheap_rank: int, semifinal_rank: int, cheap_count: int, semifinal_count: int) -> float:
    cheap_quality = 1.0 - rank_percentile(cheap_rank, cheap_count)
    semifinal_quality = 1.0 - rank_percentile(semifinal_rank, semifinal_count)
    return round(max(0.0, min(1.0, 0.35 * cheap_quality + 0.65 * semifinal_quality)), 4)


def numeric_score(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    core = scores.get("scores") if isinstance(scores.get("scores"), dict) else {}
    value = core.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
