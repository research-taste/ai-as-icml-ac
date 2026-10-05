from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from icml_ai_ac.finalists import (
    contribution_class_for,
    rank_percentile,
    read_ranked_rows,
)
from icml_ai_ac.storage import write_json


SEMIFINAL_ENSEMBLE_VERSION = "semifinal_equal_weight_rank_v1"


@dataclass(slots=True)
class RankingSource:
    path: Path
    label: str
    metadata: dict[str, Any]
    rows: list[dict[str, Any]]
    rows_by_id: dict[str, dict[str, Any]]


def ensemble_semifinal_rankings(
    *,
    ranking_paths: list[Path],
    labels: list[str] | None,
    out: Path,
    report: Path | None,
) -> dict[str, Any]:
    if len(ranking_paths) < 2:
        raise ValueError("at least two semifinal rankings are required")
    if len(set(ranking_paths)) != len(ranking_paths):
        raise ValueError("semifinal ranking paths must be unique")
    if labels is not None and len(labels) != len(ranking_paths):
        raise ValueError("--label must be supplied once per --ranking")

    sources = [
        read_ranking_source(path, explicit_label=labels[index] if labels else None)
        for index, path in enumerate(ranking_paths)
    ]
    source_labels = [source.label for source in sources]
    if len(set(source_labels)) != len(source_labels):
        raise ValueError("semifinal judge labels must be unique; supply explicit --label values")

    validate_matching_coverage(sources)
    paper_count = len(sources[0].rows)
    working_rows = [
        aggregate_paper(paper_id, sources=sources, paper_count=paper_count)
        for paper_id in sources[0].rows_by_id
    ]
    working_rows.sort(
        key=lambda row: (
            row["mean_normalized_rank"],
            row["semifinal_judge_disagreement"],
            row["best_source_rank_percentile"],
            row["paper_id"],
        )
    )
    for rank, row in enumerate(working_rows, start=1):
        row["rank"] = rank
        row["semifinal_ensemble_rank"] = rank

    source_metadata = [source.metadata for source in sources]
    payload = {
        "status": "ok",
        "method_version": SEMIFINAL_ENSEMBLE_VERSION,
        "aggregation": "equal_weight_mean_normalized_rank",
        "tie_breaking": [
            "lower_semifinal_judge_disagreement",
            "better_best_source_rank_percentile",
            "paper_id",
        ],
        "source_count": len(sources),
        "paper_count": paper_count,
        "sources": source_metadata,
        "ranked_papers": working_rows,
    }
    write_json(out, payload)

    disagreements = [float(row["semifinal_judge_disagreement"]) for row in working_rows]
    report_payload = {
        "status": "ok",
        "method_version": SEMIFINAL_ENSEMBLE_VERSION,
        "aggregation": payload["aggregation"],
        "out": str(out),
        "source_count": len(sources),
        "paper_count": paper_count,
        "sources": source_metadata,
        "mean_judge_disagreement": round(sum(disagreements) / paper_count, 6),
        "max_judge_disagreement": round(max(disagreements), 6),
        "top_paper_ids": [row["paper_id"] for row in working_rows[:20]],
        "highest_disagreement_paper_ids": [
            row["paper_id"]
            for row in sorted(
                working_rows,
                key=lambda row: (-row["semifinal_judge_disagreement"], row["semifinal_ensemble_rank"]),
            )[:20]
        ],
    }
    if report is not None:
        write_json(report, report_payload)
    return payload


def read_ranking_source(path: Path, *, explicit_label: str | None) -> RankingSource:
    raw_metadata = read_top_level_metadata(path)
    status = raw_metadata.get("status")
    if status is not None and status != "ok":
        raise ValueError(f"{path}: ranking status must be 'ok', got {status!r}")
    rows = read_ranked_rows(path)
    if not rows:
        raise ValueError(f"{path}: ranking is empty")
    validate_contiguous_ranks(rows, path=path)
    label = (explicit_label or inferred_label(path, raw_metadata)).strip()
    if not label:
        raise ValueError(f"empty semifinal judge label for {path}")
    metadata = {
        "label": label,
        "path": str(path),
        "sha256": sha256_path(path),
        "status": status,
        "provider": raw_metadata.get("provider"),
        "requested_model": raw_metadata.get("model"),
        "served_model": raw_metadata.get("served_model"),
        "reasoning_effort": raw_metadata.get("reasoning_effort"),
        "prompt_version": raw_metadata.get("prompt_version"),
        "source_aggregation": (
            raw_metadata.get("ranking", {}).get("aggregation")
            if isinstance(raw_metadata.get("ranking"), dict)
            else None
        ),
        "source_tie_breaking": (
            raw_metadata.get("ranking", {}).get("tie_breaking")
            if isinstance(raw_metadata.get("ranking"), dict)
            else None
        ),
    }
    return RankingSource(
        path=path,
        label=label,
        metadata=metadata,
        rows=rows,
        rows_by_id={str(row["paper_id"]): row for row in rows},
    )


def read_top_level_metadata(path: Path) -> dict[str, Any]:
    if path.suffix == ".jsonl":
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inferred_label(path: Path, metadata: dict[str, Any]) -> str:
    model = metadata.get("served_model") or metadata.get("model")
    provider = metadata.get("provider")
    if provider and model:
        return f"{provider}:{model}"
    if model:
        return str(model)
    return path.stem


def validate_matching_coverage(sources: list[RankingSource]) -> None:
    reference = set(sources[0].rows_by_id)
    for source in sources[1:]:
        source_ids = set(source.rows_by_id)
        missing = sorted(reference - source_ids)
        extra = sorted(source_ids - reference)
        if missing or extra:
            raise ValueError(
                f"{source.path}: paper coverage differs from {sources[0].path}; "
                f"missing={len(missing)} {missing[:5]}, extra={len(extra)} {extra[:5]}"
            )


def validate_contiguous_ranks(rows: list[dict[str, Any]], *, path: Path) -> None:
    ranks = sorted(int(row.get("rank") or 10**9) for row in rows)
    expected = list(range(1, len(rows) + 1))
    if ranks != expected:
        raise ValueError(
            f"{path}: ranks must contain each integer from 1 through {len(rows)} exactly once"
        )


def aggregate_paper(
    paper_id: str,
    *,
    sources: list[RankingSource],
    paper_count: int,
) -> dict[str, Any]:
    source_rows = {source.label: source.rows_by_id[paper_id] for source in sources}
    source_ranks = {label: int(row.get("rank") or 10**9) for label, row in source_rows.items()}
    source_rank_percentiles = {
        label: rank_percentile(rank, paper_count) for label, rank in source_ranks.items()
    }
    normalized_ranks = list(source_rank_percentiles.values())
    mean_normalized_rank = sum(normalized_ranks) / len(normalized_ranks)
    disagreement = max(normalized_ranks) - min(normalized_ranks)

    contribution_classes = {
        label: contribution_class_for(row, {})
        for label, row in source_rows.items()
    }
    agreed_class = consensus_contribution_class(list(contribution_classes.values()))
    row: dict[str, Any] = {
        "status": "ok",
        "paper_id": paper_id,
        "title": first_nonempty(row.get("title") for row in source_rows.values()),
        "ensemble_method": SEMIFINAL_ENSEMBLE_VERSION,
        "judge_count": len(sources),
        "consensus_score": round(1.0 - mean_normalized_rank, 6),
        "mean_normalized_rank": round(mean_normalized_rank, 6),
        "best_source_rank_percentile": round(min(normalized_ranks), 6),
        "worst_source_rank_percentile": round(max(normalized_ranks), 6),
        "semifinal_judge_disagreement": round(disagreement, 6),
        "source_ranks": source_ranks,
        "source_rank_percentiles": {
            label: round(value, 6) for label, value in source_rank_percentiles.items()
        },
        "source_judges": {
            source.label: {
                key: value
                for key, value in source.metadata.items()
                if key not in {"label", "path"}
            }
            for source in sources
        },
        "source_contribution_classes": contribution_classes,
        "source_rows": source_rows,
    }
    if agreed_class is not None:
        row["primary_contribution_class"] = agreed_class
    return row


def consensus_contribution_class(values: list[str]) -> str | None:
    concrete = [value for value in values if value and value != "other"]
    if concrete and len(set(concrete)) == 1:
        return concrete[0]
    return None


def first_nonempty(values: Iterable[Any]) -> Any:
    return next((value for value in values if value), None)
