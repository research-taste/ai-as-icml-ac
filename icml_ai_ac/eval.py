from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from icml_ai_ac.storage import ensure_parent, read_jsonl, write_json


@dataclass(slots=True)
class RankingRow:
    paper_id: str
    rank: int
    title: str | None = None
    primary_contribution_class: str | None = None


def evaluate_ranking(
    *,
    gold_path: Path,
    candidate_path: Path,
    out: Path,
    k_values: list[int],
) -> dict[str, Any]:
    gold_rows = read_ranking(gold_path)
    candidate_rows = read_ranking(candidate_path)
    gold_by_id = {row.paper_id: row for row in gold_rows}
    candidate_by_id = {row.paper_id: row for row in candidate_rows}
    gold_ids = [row.paper_id for row in sorted(gold_rows, key=lambda row: row.rank)]
    candidate_ids = [row.paper_id for row in sorted(candidate_rows, key=lambda row: row.rank)]

    metrics: dict[str, Any] = {
        "gold_path": str(gold_path),
        "candidate_path": str(candidate_path),
        "gold_count": len(gold_rows),
        "candidate_count": len(candidate_rows),
        "overlap_count": len(set(gold_by_id) & set(candidate_by_id)),
        "k_metrics": {},
        "recall_matrix": recall_matrix(gold_ids, candidate_ids, k_values=k_values),
        "rank_correlation": rank_correlation(gold_by_id, candidate_by_id),
        "category_metrics": category_metrics(gold_rows, candidate_rows, k_values=k_values),
        "gold_anchored_category_recall": gold_anchored_category_recall(
            gold_rows,
            candidate_rows,
            k_values=k_values,
        ),
        "missing_from_candidate": [paper_id for paper_id in gold_ids if paper_id not in candidate_by_id],
        "extra_in_candidate": [paper_id for paper_id in candidate_ids if paper_id not in gold_by_id],
    }
    for k in k_values:
        metrics["k_metrics"][str(k)] = top_k_metrics(gold_ids, candidate_ids, k)
    ensure_parent(out)
    write_json(out, metrics)
    return metrics


def read_ranking(path: Path) -> list[RankingRow]:
    if path.suffix == ".jsonl":
        rows = list(read_jsonl(path))
        if rows and not any(has_explicit_rank(row) for row in rows):
            rows = sorted(rows, key=score_sort_key, reverse=True)
            for idx, row in enumerate(rows, start=1):
                row["derived_rank"] = idx
        return validate_ranking_rows([ranking_row_from_any(row) for row in rows], path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "ranking" in payload and isinstance(payload["ranking"], dict):
        payload = payload["ranking"]
    if isinstance(payload, dict) and "ranked_papers" in payload:
        rows = payload["ranked_papers"]
    elif isinstance(payload, dict) and "rows" in payload:
        rows = payload["rows"]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"Could not find ranking rows in {path}")
    return validate_ranking_rows([ranking_row_from_any(row) for row in rows if isinstance(row, dict)], path=path)


def validate_ranking_rows(rows: list[RankingRow], *, path: Path) -> list[RankingRow]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        if not row.paper_id:
            raise ValueError(f"{path}: ranking row is missing paper_id")
        if row.paper_id in seen:
            duplicates.append(row.paper_id)
        seen.add(row.paper_id)
    if duplicates:
        raise ValueError(f"{path}: duplicate paper_id values in ranking: {sorted(set(duplicates))}")
    return rows


def ranking_row_from_any(row: dict[str, Any]) -> RankingRow:
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    contribution_profile = scores.get("contribution_profile") if isinstance(scores.get("contribution_profile"), dict) else {}
    return RankingRow(
        paper_id=str(row.get("paper_id") or ""),
        rank=int(row.get("rank") or row.get("batch_rank") or row.get("gold_rank") or row.get("derived_rank") or 10**9),
        title=row.get("title"),
        primary_contribution_class=(
            row.get("primary_contribution_class")
            or contribution_profile.get("primary_contribution_class")
            or row.get("contribution_class")
        ),
    )


def has_explicit_rank(row: dict[str, Any]) -> bool:
    return any(row.get(key) is not None for key in ["rank", "batch_rank", "gold_rank", "derived_rank"])


def score_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    core = scores.get("scores") if isinstance(scores.get("scores"), dict) else {}
    calibration = scores.get("calibration") if isinstance(scores.get("calibration"), dict) else {}
    return (
        number(core.get("executive_ac_priority")),
        number(calibration.get("estimated_percentile_among_accepted_papers")),
        number(core.get("broader_science_impact_forecast")),
        number(core.get("ml_field_impact_forecast")),
        number(core.get("overall_significance")),
    )


def number(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def top_k_metrics(gold_ids: list[str], candidate_ids: list[str], k: int) -> dict[str, Any]:
    gold_top = set(gold_ids[:k])
    candidate_top = set(candidate_ids[:k])
    overlap = gold_top & candidate_top
    precision = len(overlap) / len(candidate_top) if candidate_top else 0.0
    recall = len(overlap) / len(gold_top) if gold_top else 0.0
    return {
        "k": k,
        "overlap": len(overlap),
        "precision_at_k": round(precision, 4),
        "recall_at_k": round(recall, 4),
        "gold_top_k": gold_ids[:k],
        "candidate_top_k": candidate_ids[:k],
        "missed_gold_top_k": [paper_id for paper_id in gold_ids[:k] if paper_id not in candidate_top],
    }


def recall_matrix(gold_ids: list[str], candidate_ids: list[str], *, k_values: list[int]) -> dict[str, Any]:
    matrix: dict[str, Any] = {}
    for gold_k in k_values:
        gold_top = set(gold_ids[:gold_k])
        row: dict[str, Any] = {}
        for candidate_k in k_values:
            candidate_top = set(candidate_ids[:candidate_k])
            overlap = gold_top & candidate_top
            row[str(candidate_k)] = {
                "gold_top_k": gold_k,
                "candidate_top_k": candidate_k,
                "overlap": len(overlap),
                "recall": round(len(overlap) / len(gold_top), 4) if gold_top else 0.0,
                "missed_gold": [paper_id for paper_id in gold_ids[:gold_k] if paper_id not in candidate_top],
            }
        matrix[str(gold_k)] = row
    return matrix


def rank_correlation(gold_by_id: dict[str, RankingRow], candidate_by_id: dict[str, RankingRow]) -> dict[str, Any]:
    shared = sorted(set(gold_by_id) & set(candidate_by_id))
    n = len(shared)
    if n < 2:
        return {"shared_count": n, "spearman": None}
    gold_shared_rank = {
        paper_id: idx
        for idx, paper_id in enumerate(
            sorted(shared, key=lambda paper_id: gold_by_id[paper_id].rank),
            start=1,
        )
    }
    candidate_shared_rank = {
        paper_id: idx
        for idx, paper_id in enumerate(
            sorted(shared, key=lambda paper_id: candidate_by_id[paper_id].rank),
            start=1,
        )
    }
    diffs_squared = [
        (gold_shared_rank[paper_id] - candidate_shared_rank[paper_id]) ** 2
        for paper_id in shared
    ]
    spearman = 1 - (6 * sum(diffs_squared)) / (n * (n * n - 1))
    return {
        "shared_count": n,
        "spearman": round(spearman, 4),
        "rank_basis": "relative_order_within_shared_papers",
    }


def category_metrics(
    gold_rows: list[RankingRow],
    candidate_rows: list[RankingRow],
    *,
    k_values: list[int],
) -> dict[str, Any]:
    gold_by_category: dict[str, list[RankingRow]] = defaultdict(list)
    candidate_by_category: dict[str, list[RankingRow]] = defaultdict(list)
    for row in gold_rows:
        gold_by_category[row.primary_contribution_class or "unknown"].append(row)
    for row in candidate_rows:
        candidate_by_category[row.primary_contribution_class or "unknown"].append(row)
    results: dict[str, Any] = {}
    for category in sorted(set(gold_by_category) | set(candidate_by_category)):
        gold_ids = [row.paper_id for row in sorted(gold_by_category[category], key=lambda row: row.rank)]
        candidate_ids = [row.paper_id for row in sorted(candidate_by_category[category], key=lambda row: row.rank)]
        results[category] = {
            str(k): top_k_metrics(gold_ids, candidate_ids, k)
            for k in k_values
            if gold_ids or candidate_ids
        }
    return results


def gold_anchored_category_recall(
    gold_rows: list[RankingRow],
    candidate_rows: list[RankingRow],
    *,
    k_values: list[int],
) -> dict[str, Any]:
    gold_by_category: dict[str, list[RankingRow]] = defaultdict(list)
    for row in gold_rows:
        gold_by_category[row.primary_contribution_class or "unknown"].append(row)
    for rows in gold_by_category.values():
        rows.sort(key=lambda row: row.rank)

    candidate_ids = [row.paper_id for row in sorted(candidate_rows, key=lambda row: row.rank)]
    by_category: dict[str, Any] = {}
    macro_by_k: dict[str, Any] = {}
    for category, rows in sorted(gold_by_category.items()):
        gold_ids = [row.paper_id for row in rows]
        gold_top3 = gold_ids[: min(3, len(gold_ids))]
        category_metrics_by_k: dict[str, Any] = {}
        for k in k_values:
            candidate_top = set(candidate_ids[:k])
            best = gold_ids[0] if gold_ids else None
            top3_overlap = [paper_id for paper_id in gold_top3 if paper_id in candidate_top]
            class_overlap = [paper_id for paper_id in gold_ids if paper_id in candidate_top]
            category_metrics_by_k[str(k)] = {
                "candidate_top_k": k,
                "gold_class_count": len(gold_ids),
                "gold_class_top_paper": best,
                "captured_gold_class_top_paper": bool(best and best in candidate_top),
                "gold_class_top3_denominator": len(gold_top3),
                "gold_class_top3_overlap": len(top3_overlap),
                "gold_class_top3_recall": round(len(top3_overlap) / len(gold_top3), 4) if gold_top3 else None,
                "gold_class_all_overlap": len(class_overlap),
                "gold_class_all_recall": round(len(class_overlap) / len(gold_ids), 4) if gold_ids else None,
                "missed_gold_class_top3": [paper_id for paper_id in gold_top3 if paper_id not in candidate_top],
            }
        by_category[category] = {
            "gold_class_count": len(gold_ids),
            "gold_class_ids_by_gold_rank": gold_ids,
            "metrics_by_candidate_k": category_metrics_by_k,
        }

    for k in k_values:
        key = str(k)
        category_values = [
            category_payload["metrics_by_candidate_k"][key]
            for category_payload in by_category.values()
        ]
        captured_best = [value["captured_gold_class_top_paper"] for value in category_values]
        top3_recalls = [
            value["gold_class_top3_recall"]
            for value in category_values
            if value["gold_class_top3_recall"] is not None
        ]
        all_recalls = [
            value["gold_class_all_recall"]
            for value in category_values
            if value["gold_class_all_recall"] is not None
        ]
        macro_by_k[key] = {
            "candidate_top_k": k,
            "category_count": len(category_values),
            "best_per_class_capture_rate": round(sum(captured_best) / len(captured_best), 4)
            if captured_best else None,
            "macro_gold_class_top3_recall": round(sum(top3_recalls) / len(top3_recalls), 4)
            if top3_recalls else None,
            "macro_gold_class_all_recall": round(sum(all_recalls) / len(all_recalls), 4)
            if all_recalls else None,
        }
    return {
        "description": "Gold-class recall uses the gold ranking's primary contribution class, not candidate-predicted classes.",
        "by_category": by_category,
        "macro_by_candidate_k": macro_by_k,
    }
