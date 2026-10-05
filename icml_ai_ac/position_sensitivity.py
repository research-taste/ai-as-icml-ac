from __future__ import annotations

import copy
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from icml_ai_ac.shortlist import (
    aggregate_rows,
    aggregate_to_row,
    read_class_map,
    shortlist_sort_key,
    source_model,
)
from icml_ai_ac.storage import read_json, read_jsonl, write_json, write_jsonl


def build_position_robust_shortlist(
    *,
    scores_paths: list[Path],
    raw_ranking_path: Path,
    class_path: Path | None,
    cutoff: int,
    adjusted_out: Path,
    shortlist_out: Path,
    report_out: Path,
) -> dict[str, Any]:
    """Build a conservative union of raw and input-position-adjusted leaders."""
    if not scores_paths:
        raise ValueError("at least one score path is required")
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")

    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    prompt_cache: dict[Path, list[str]] = {}
    for path in scores_paths:
        rows = list(read_jsonl(path))
        if not rows:
            raise ValueError(f"{path}: no score rows")
        model_keys = {source_model(row) for row in rows}
        if len(model_keys) != 1:
            raise ValueError(f"{path}: expected one provider/model stream, found {sorted(model_keys)}")
        model_key = next(iter(model_keys))
        if model_key in rows_by_model:
            raise ValueError(f"duplicate model stream: {model_key}")
        for row in rows:
            row["ensemble_source_path"] = str(path)
            row["input_slot"] = input_slot(row, prompt_cache=prompt_cache)
        rows_by_model[model_key] = rows

    adjusted_rows: list[dict[str, Any]] = []
    model_reports: dict[str, Any] = {}
    for model_key, rows in sorted(rows_by_model.items()):
        slot_effects, diagnostics = estimate_slot_effects(rows)
        model_reports[model_key] = {
            **diagnostics,
            "slot_effects": [round(effect, 6) for effect in slot_effects],
            "slot_effect_range": round(max(slot_effects) - min(slot_effects), 6),
        }
        for row in rows:
            adjusted = copy.deepcopy(row)
            slot = int(row["input_slot"])
            raw_priority = float(row["batch_local_priority"])
            adjusted_priority = clamp(raw_priority - slot_effects[slot])
            adjusted["batch_local_priority_raw"] = raw_priority
            adjusted["batch_local_priority"] = round(adjusted_priority, 6)
            adjusted["position_sensitivity"] = {
                "input_slot": slot,
                "estimated_slot_effect": round(slot_effects[slot], 6),
            }
            adjusted_rows.append(adjusted)

    class_by_id = read_class_map(class_path)
    aggregates = aggregate_rows(adjusted_rows, class_by_id=class_by_id)
    adjusted_ranking = sorted(
        (aggregate_to_row(aggregate) for aggregate in aggregates.values()),
        key=shortlist_sort_key,
        reverse=True,
    )
    for rank, row in enumerate(adjusted_ranking, start=1):
        row["rank"] = rank
        row["aggregate_rank"] = rank
        row["position_sensitivity"] = {"adjusted_rank": rank}

    raw_ranking = list(read_jsonl(raw_ranking_path))
    raw_ok = [row for row in raw_ranking if row.get("status") == "ok"]
    raw_ok.sort(key=lambda row: int(row.get("aggregate_rank") or row.get("rank") or 10**9))
    if len(raw_ok) != len(adjusted_ranking):
        raise ValueError(
            f"raw/adjusted coverage mismatch: {len(raw_ok)} != {len(adjusted_ranking)}"
        )
    raw_rank_by_id = {
        str(row["paper_id"]): int(row.get("aggregate_rank") or row.get("rank"))
        for row in raw_ok
    }
    adjusted_by_id = {str(row["paper_id"]): row for row in adjusted_ranking}
    if set(raw_rank_by_id) != set(adjusted_by_id):
        raise ValueError("raw and adjusted rankings contain different paper IDs")

    effective_cutoff = min(cutoff, len(raw_ok))
    raw_top = {str(row["paper_id"]) for row in raw_ok[:effective_cutoff]}
    adjusted_top = {str(row["paper_id"]) for row in adjusted_ranking[:effective_cutoff]}
    union_ids = raw_top | adjusted_top
    shortlist = []
    for paper_id in union_ids:
        row = copy.deepcopy(adjusted_by_id[paper_id])
        adjusted_rank = int(row["aggregate_rank"])
        raw_rank = raw_rank_by_id[paper_id]
        row["stage3_selection"] = {
            "raw_rank": raw_rank,
            "position_adjusted_rank": adjusted_rank,
            "raw_top_cutoff": raw_rank <= effective_cutoff,
            "position_adjusted_top_cutoff": adjusted_rank <= effective_cutoff,
            "cutoff": effective_cutoff,
            "selection_rule": "union_raw_and_position_adjusted_top_k",
        }
        shortlist.append(row)
    shortlist.sort(
        key=lambda row: (
            min(
                int(row["stage3_selection"]["raw_rank"]),
                int(row["stage3_selection"]["position_adjusted_rank"]),
            ),
            int(row["stage3_selection"]["raw_rank"]),
            str(row["paper_id"]),
        )
    )
    for rank, row in enumerate(shortlist, start=1):
        row["rank"] = rank

    write_jsonl(adjusted_out, adjusted_ranking)
    write_jsonl(shortlist_out, shortlist)
    overlap = len(raw_top & adjusted_top)
    adjusted_rank_by_id = {
        str(row["paper_id"]): int(row["aggregate_rank"]) for row in adjusted_ranking
    }
    overlap_at_k = {}
    for k in sorted({50, 100, 250, 500, effective_cutoff}):
        if k > len(raw_ok):
            continue
        raw_ids = {str(row["paper_id"]) for row in raw_ok[:k]}
        adjusted_ids = {str(row["paper_id"]) for row in adjusted_ranking[:k]}
        overlap_at_k[str(k)] = {
            "count": len(raw_ids & adjusted_ids),
            "fraction": round(len(raw_ids & adjusted_ids) / k, 6),
        }
    raw_ranks = [raw_rank_by_id[str(row["paper_id"])] for row in raw_ok]
    adjusted_ranks = [adjusted_rank_by_id[str(row["paper_id"])] for row in raw_ok]
    report = {
        "status": "ok",
        "method": "within_paper_paired_slot_fixed_effect_sensitivity",
        "scores_paths": [str(path) for path in scores_paths],
        "raw_ranking_path": str(raw_ranking_path),
        "class_path": str(class_path) if class_path else None,
        "adjusted_out": str(adjusted_out),
        "shortlist_out": str(shortlist_out),
        "paper_count": len(raw_ok),
        "cutoff": effective_cutoff,
        "raw_top_count": len(raw_top),
        "position_adjusted_top_count": len(adjusted_top),
        "overlap_count": overlap,
        "raw_only_count": len(raw_top - adjusted_top),
        "position_adjusted_only_count": len(adjusted_top - raw_top),
        "union_count": len(union_ids),
        "union_expansion_count": len(union_ids) - effective_cutoff,
        "union_expansion_fraction": round(
            (len(union_ids) - effective_cutoff) / effective_cutoff, 6
        ),
        "raw_adjusted_spearman": round(pearson(raw_ranks, adjusted_ranks), 6),
        "mean_absolute_rank_shift": round(
            mean([abs(raw - adjusted) for raw, adjusted in zip(raw_ranks, adjusted_ranks, strict=True)]),
            3,
        ),
        "top_k_overlap": overlap_at_k,
        "models": model_reports,
    }
    write_json(report_out, report)
    return report


def input_slot(row: dict[str, Any], *, prompt_cache: dict[Path, list[str]]) -> int:
    prompt_path_value = row.get("prompt_path")
    if not prompt_path_value:
        raise ValueError(f"paper {row.get('paper_id')}: missing prompt_path")
    prompt_path = Path(str(prompt_path_value))
    candidate_ids = prompt_cache.get(prompt_path)
    if candidate_ids is None:
        payload = read_json(prompt_path)
        candidate_ids = [str(paper_id) for paper_id in payload.get("candidate_ids", [])]
        if not candidate_ids:
            raise ValueError(f"{prompt_path}: missing candidate_ids")
        prompt_cache[prompt_path] = candidate_ids
    paper_id = str(row.get("paper_id") or "")
    matches = [index for index, candidate_id in enumerate(candidate_ids) if candidate_id == paper_id]
    if len(matches) != 1:
        raise ValueError(f"{prompt_path}: paper {paper_id} appears {len(matches)} times")
    return matches[0]


def estimate_slot_effects(rows: list[dict[str, Any]]) -> tuple[list[float], dict[str, Any]]:
    by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    max_slot = -1
    for row in rows:
        if row.get("status") != "ok":
            raise ValueError(f"non-ok row in position sensitivity input: {row.get('paper_id')}")
        if not isinstance(row.get("batch_local_priority"), (int, float)):
            raise ValueError(f"paper {row.get('paper_id')}: missing numeric batch_local_priority")
        slot = int(row["input_slot"])
        max_slot = max(max_slot, slot)
        by_paper[str(row["paper_id"])].append(row)
    slot_count = max_slot + 1
    if slot_count < 2:
        raise ValueError("position sensitivity requires at least two input slots")

    laplacian = [[0.0] * slot_count for _ in range(slot_count)]
    rhs = [0.0] * slot_count
    earlier_wins = later_wins = ties = 0
    earlier_deltas: list[float] = []
    slot_deltas: list[float] = []
    score_deltas: list[float] = []
    pair_count = 0
    for paper_id, paper_rows in by_paper.items():
        if len(paper_rows) != 2:
            raise ValueError(f"paper {paper_id}: expected two partition rows, found {len(paper_rows)}")
        first, second = paper_rows
        first_slot = int(first["input_slot"])
        second_slot = int(second["input_slot"])
        if first_slot == second_slot:
            continue
        first_score = float(first["batch_local_priority"])
        second_score = float(second["batch_local_priority"])
        difference = first_score - second_score
        laplacian[first_slot][first_slot] += 1.0
        laplacian[second_slot][second_slot] += 1.0
        laplacian[first_slot][second_slot] -= 1.0
        laplacian[second_slot][first_slot] -= 1.0
        rhs[first_slot] += difference
        rhs[second_slot] -= difference
        pair_count += 1

        earlier, later = (
            (first, second) if first_slot < second_slot else (second, first)
        )
        earlier_delta = float(earlier["batch_local_priority"]) - float(later["batch_local_priority"])
        earlier_deltas.append(earlier_delta)
        if earlier_delta > 1e-12:
            earlier_wins += 1
        elif earlier_delta < -1e-12:
            later_wins += 1
        else:
            ties += 1
        slot_deltas.append(float(first_slot - second_slot))
        score_deltas.append(difference)
    if pair_count == 0:
        raise ValueError("no cross-slot paper pairs available")

    reduced = [row[:-1] for row in laplacian[:-1]]
    solution = solve_linear_system(reduced, rhs[:-1])
    effects = solution + [0.0]
    effect_mean = sum(effects) / len(effects)
    effects = [effect - effect_mean for effect in effects]
    non_ties = earlier_wins + later_wins
    diagnostics = {
        "row_count": len(rows),
        "paper_count": len(by_paper),
        "cross_slot_pair_count": pair_count,
        "same_slot_pair_count": len(by_paper) - pair_count,
        "earlier_wins": earlier_wins,
        "later_wins": later_wins,
        "ties": ties,
        "earlier_win_rate_excluding_ties": (
            round(earlier_wins / non_ties, 6) if non_ties else None
        ),
        "mean_earlier_minus_later_priority": round(mean(earlier_deltas), 6),
        "slot_delta_score_delta_correlation": round(
            pearson(slot_deltas, score_deltas), 6
        ),
    }
    return effects, diagnostics


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [list(matrix[index]) + [float(vector[index])] for index in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("input-slot comparison graph is disconnected")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(augmented[row], augmented[column], strict=True)
                ]
    return [augmented[index][-1] for index in range(size)]


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    return numerator / (x_scale * y_scale) if x_scale and y_scale else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
