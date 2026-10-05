from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from icml_ai_ac.storage import read_jsonl, write_json, write_jsonl


@dataclass(slots=True)
class PaperAggregate:
    paper_id: str
    title: str | None = None
    primary_contribution_class: str | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)


def build_shortlist(
    *,
    scores_path: Path | list[Path],
    out: Path,
    report: Path | None,
    limit: int | None,
    min_per_class: int,
    class_path: Path | None,
) -> dict[str, Any]:
    score_paths = [scores_path] if isinstance(scores_path, Path) else scores_path
    rows = read_score_rows(score_paths)
    class_by_id = read_class_map(class_path)
    aggregates = aggregate_rows(rows, class_by_id=class_by_id)
    ranked = sorted((aggregate_to_row(aggregate) for aggregate in aggregates.values()), key=shortlist_sort_key, reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["aggregate_rank"] = rank
    selected = select_ranked_rows(ranked, limit=limit, min_per_class=min_per_class)
    write_jsonl(out, selected)
    payload = {
        "scores_path": str(score_paths[0]) if len(score_paths) == 1 else [str(path) for path in score_paths],
        "out": str(out),
        "limit": limit,
        "min_per_class": min_per_class,
        "class_path": str(class_path) if class_path else None,
        "input_rows": len(rows),
        "unique_papers": len(aggregates),
        "written_rows": len(selected),
        "source_count": len(score_paths),
        "source_models": sorted({source_model(row) for row in rows}),
        "top_paper_ids": [row["paper_id"] for row in selected[:20]],
    }
    if report is not None:
        write_json(report, payload)
    return payload


def read_score_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(path):
            row["ensemble_source_path"] = str(path)
            rows.append(row)
    return rows


def select_ranked_rows(
    ranked: list[dict[str, Any]],
    *,
    limit: int | None,
    min_per_class: int,
) -> list[dict[str, Any]]:
    if limit is None:
        return ranked
    if min_per_class <= 0:
        return ranked[:limit]
    selected_by_id: dict[str, dict[str, Any]] = {}
    classes = sorted({str(row.get("primary_contribution_class") or "other") for row in ranked})
    class_leaders: list[dict[str, Any]] = []
    for contribution_class in classes:
        class_rows = [row for row in ranked if row.get("primary_contribution_class") == contribution_class]
        class_leaders.extend(class_rows[:min_per_class])
    for row in sorted(class_leaders, key=lambda item: item["aggregate_rank"]):
        if len(selected_by_id) >= limit:
            break
        selected_by_id[row["paper_id"]] = row
    for row in ranked:
        if len(selected_by_id) >= limit:
            break
        selected_by_id.setdefault(row["paper_id"], row)
    selected = sorted(selected_by_id.values(), key=lambda row: row["aggregate_rank"])
    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank
    return selected


def read_class_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    return {
        str(row.get("paper_id")): str(row.get("primary_contribution_class") or "other")
        for row in read_jsonl(path)
        if row.get("paper_id")
    }


def aggregate_rows(rows: list[dict[str, Any]], *, class_by_id: dict[str, str]) -> dict[str, PaperAggregate]:
    aggregates: dict[str, PaperAggregate] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        paper_id = str(row.get("paper_id") or "")
        if not paper_id:
            continue
        aggregate = aggregates.setdefault(paper_id, PaperAggregate(paper_id=paper_id))
        aggregate.rows.append(row)
        if aggregate.title is None:
            aggregate.title = row.get("title")
        if aggregate.primary_contribution_class is None:
            aggregate.primary_contribution_class = class_by_id.get(paper_id) or contribution_class(row)
    return aggregates


def aggregate_to_row(aggregate: PaperAggregate) -> dict[str, Any]:
    rows = aggregate.rows
    appearances = len(rows)
    routing_class = aggregate.primary_contribution_class or "other"
    class_votes = Counter(contribution_class(row) for row in rows)
    max_class_votes = max(class_votes.values(), default=0)
    class_vote_leaders = sorted(
        contribution_class
        for contribution_class, count in class_votes.items()
        if count == max_class_votes
    )
    ensemble_class = class_vote_leaders[0] if len(class_vote_leaders) == 1 else None
    resolved_class = ensemble_class or routing_class
    local_priorities = [row_priority(row) for row in rows]
    executive_scores = [score_value(row, "executive_ac_priority") for row in rows]
    broader_scores = [score_value(row, "broader_science_impact_forecast") for row in rows]
    ml_scores = [score_value(row, "ml_field_impact_forecast") for row in rows]
    significance_scores = [score_value(row, "overall_significance") for row in rows]
    rank_percentiles = [numeric(row.get("batch_rank_percentile")) for row in rows if row.get("batch_rank_percentile") is not None]
    advance_votes = [
        bool(
            nested(row, ["scores", "ranking_signals", "should_advance_to_strong_model"])
            or row.get("should_advance_to_strong_model")
        )
        for row in rows
    ]
    source_groups = rows_by_source_model(rows)
    source_model_class_votes = {
        source: dict(sorted(Counter(contribution_class(row) for row in source_rows).items()))
        for source, source_rows in sorted(source_groups.items())
    }
    source_priorities = [
        aggregate_priority(
            local_priorities=[row_priority(row) for row in source_rows],
            advance_votes=[
                bool(
                    nested(row, ["scores", "ranking_signals", "should_advance_to_strong_model"])
                    or row.get("should_advance_to_strong_model")
                )
                for row in source_rows
            ],
        )
        for source_rows in source_groups.values()
    ]
    priority = ensemble_priority(local_priorities=local_priorities, advance_votes=advance_votes, source_priorities=source_priorities)
    executive = round(10 * priority, 4)
    percentile = round(100 * priority, 4)
    broader = round(mean(broader_scores), 4)
    ml = round(mean(ml_scores), 4)
    significance = round(mean(significance_scores), 4)
    return {
        "status": "ok",
        "paper_id": aggregate.paper_id,
        "title": aggregate.title,
        "primary_contribution_class": resolved_class,
        "routing_contribution_class": routing_class,
        "ensemble_contribution_class": ensemble_class,
        "scores": {
            "contribution_profile": {
                "primary_contribution_class": resolved_class,
                "secondary_contribution_classes": [],
            },
            "scores": {
                "executive_ac_priority": executive,
                "broader_science_impact_forecast": broader,
                "ml_field_impact_forecast": ml,
                "overall_significance": significance,
                "technical_soundness": round(mean([score_value(row, "technical_soundness") for row in rows]), 4),
            },
            "calibration": {
                "estimated_percentile_among_accepted_papers": percentile,
                "triage_bucket": aggregate_bucket(priority),
            },
            "ranking_signals": {
                "should_advance_to_strong_model": priority >= 0.65,
            },
        },
        "shortlist_metrics": {
            "appearances": appearances,
            "advance_votes": sum(advance_votes),
            "advance_rate": round(sum(advance_votes) / appearances, 4) if appearances else 0.0,
            "mean_local_priority": round(mean(local_priorities), 4),
            "max_local_priority": round(max(local_priorities), 4) if local_priorities else 0.0,
            "mean_executive_ac_priority": round(mean(executive_scores), 4),
            "max_executive_ac_priority": round(max(executive_scores), 4) if executive_scores else 0.0,
            "mean_batch_rank_percentile": round(mean(rank_percentiles), 4),
            "max_batch_rank_percentile": round(max(rank_percentiles), 4) if rank_percentiles else None,
            "source_rows": [
                {
                    "provider": row.get("provider"),
                    "model": row.get("model"),
                    "served_model": row.get("served_model"),
                    "ensemble_source_path": row.get("ensemble_source_path"),
                    "partition_index": row.get("partition_index"),
                    "batch_index": row.get("batch_index"),
                    "batch_rank": row.get("batch_rank"),
                    "forced_bucket": row.get("forced_bucket"),
                    "batch_local_priority": row.get("batch_local_priority"),
                    "primary_contribution_class": contribution_class(row),
                    "prompt_path": row.get("prompt_path"),
                }
                for row in rows
            ],
            "contribution_class_consensus": {
                "routing_class": routing_class,
                "ensemble_class": ensemble_class,
                "resolved_class": resolved_class,
                "resolution": "unique_ensemble_plurality" if ensemble_class else "routing_tiebreak",
                "vote_counts": dict(sorted(class_votes.items())),
                "vote_leaders": class_vote_leaders,
                "agreement_rate": (
                    round(max_class_votes / sum(class_votes.values()), 4)
                    if class_votes
                    else 0.0
                ),
                "source_model_vote_counts": source_model_class_votes,
            },
            "aggregate_priority": round(priority, 4),
            "source_model_count": len(source_groups),
            "source_models": sorted(source_groups),
            "source_model_priorities": {
                source: round(
                    aggregate_priority(
                        local_priorities=[row_priority(row) for row in source_rows],
                        advance_votes=[
                            bool(
                                nested(row, ["scores", "ranking_signals", "should_advance_to_strong_model"])
                                or row.get("should_advance_to_strong_model")
                            )
                            for row in source_rows
                        ],
                    ),
                    4,
                )
                for source, source_rows in sorted(source_groups.items())
            },
        },
    }


def row_priority(row: dict[str, Any]) -> float:
    batch_local = row.get("batch_local_priority")
    if isinstance(batch_local, (int, float)):
        return float(batch_local)
    executive = score_value(row, "executive_ac_priority") / 10.0
    percentile = nested(row, ["scores", "calibration", "estimated_percentile_among_accepted_papers"])
    percentile_score = numeric(percentile) / 100.0
    broader = score_value(row, "broader_science_impact_forecast") / 10.0
    ml = score_value(row, "ml_field_impact_forecast") / 10.0
    return max(0.0, min(1.0, 0.45 * executive + 0.25 * percentile_score + 0.15 * broader + 0.15 * ml))


def aggregate_priority(*, local_priorities: list[float], advance_votes: list[bool]) -> float:
    if not local_priorities:
        return 0.0
    advance_rate = sum(advance_votes) / len(advance_votes) if advance_votes else 0.0
    priority = 0.45 * mean(local_priorities) + 0.35 * max(local_priorities) + 0.20 * advance_rate
    return max(0.0, min(1.0, priority))


def ensemble_priority(
    *,
    local_priorities: list[float],
    advance_votes: list[bool],
    source_priorities: list[float],
) -> float:
    if len(source_priorities) <= 1:
        return aggregate_priority(local_priorities=local_priorities, advance_votes=advance_votes)
    model_advance_rate = sum(priority >= 0.65 for priority in source_priorities) / len(source_priorities)
    priority = 0.50 * mean(source_priorities) + 0.30 * max(source_priorities) + 0.20 * model_advance_rate
    return max(0.0, min(1.0, priority))


def shortlist_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, str]:
    metrics = row.get("shortlist_metrics") if isinstance(row.get("shortlist_metrics"), dict) else {}
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    core = scores.get("scores") if isinstance(scores.get("scores"), dict) else {}
    calibration = scores.get("calibration") if isinstance(scores.get("calibration"), dict) else {}
    return (
        numeric(metrics.get("aggregate_priority")),
        numeric(metrics.get("advance_rate")),
        numeric(core.get("executive_ac_priority")),
        numeric(calibration.get("estimated_percentile_among_accepted_papers")),
        numeric(core.get("broader_science_impact_forecast")) + numeric(core.get("ml_field_impact_forecast")),
        str(row.get("paper_id") or ""),
    )


def contribution_class(row: dict[str, Any]) -> str:
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    profile = scores.get("contribution_profile") if isinstance(scores.get("contribution_profile"), dict) else {}
    return str(row.get("primary_contribution_class") or profile.get("primary_contribution_class") or "other")


def rows_by_source_model(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(source_model(row), []).append(row)
    return grouped


def source_model(row: dict[str, Any]) -> str:
    provider = str(row.get("provider") or "unknown_provider")
    model = str(row.get("model") or "unknown_model")
    return f"{provider}:{model}"


def score_value(row: dict[str, Any], field: str) -> float:
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    core = scores.get("scores") if isinstance(scores.get("scores"), dict) else {}
    return numeric(core.get(field))


def aggregate_bucket(priority: float) -> str:
    if priority >= 0.85:
        return "top_10_percent"
    if priority >= 0.70:
        return "top_quartile"
    if priority >= 0.55:
        return "above_average"
    if priority >= 0.35:
        return "middle"
    return "lower_priority"


def nested(row: dict[str, Any], path: list[str]) -> Any:
    current: Any = row
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def mean(values: list[float]) -> float:
    values = [value for value in values if isinstance(value, (int, float))]
    return sum(values) / len(values) if values else 0.0


def numeric(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
