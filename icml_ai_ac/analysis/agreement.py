"""Quantitative AI-vs-human agreement metrics over the comparison join.

Two reports, both deterministic (no model calls), built on the same
``load_comparison_rows`` foundation as the divergence gallery:

- **Q-A agreement** (``build_agreement_report``): how well an AI ranking recovers
  the human presentation tier — recall@k / precision@k and ROC-AUC for surfacing
  orals+spotlights, Kendall tau-b vs the graded tier and vs reviewer means, and
  per-tier AI-score distributions. When a strong ranking is supplied it also
  reports the same metrics on the finalist subset (the "layered" view: does the
  stronger judge agree with humans more than the cheap full-coverage signal?).

- **Q-B axis decomposition** (``build_axis_decomposition_report``): which AI axis
  best predicts human honors. Ranks every stored axis by tau-b vs tier and
  highlights the impact-vs-polish contrast (``conventional_acceptance_strength``
  minus ``executive_ac_priority``).

Metric primitives are pure standard library (Kendall tau-b via concordant/
discordant counting, ROC-AUC via the rank-based Mann-Whitney identity).
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Iterable

from icml_ai_ac.analysis.human_comparison import ComparisonRow, percentile_ranks, tier_counts


HONORED_TIERS = ("oral", "spotlight")


# --- metric primitives ----------------------------------------------------


def kendall_tau_b(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if len(y) != n:
        raise ValueError("x and y must have equal length")
    if n < 2:
        return None
    n0 = n * (n - 1) // 2
    pairs = sorted(zip(x, y), key=lambda pair: (pair[0], pair[1]))
    y_values = sorted(set(y))
    y_rank = {value: index + 1 for index, value in enumerate(y_values)}
    y_counts: dict[float, int] = {}
    for value in y:
        y_counts[value] = y_counts.get(value, 0) + 1
    tie_y = sum(count * (count - 1) // 2 for count in y_counts.values())

    tree = [0] * (len(y_values) + 1)

    def tree_add(index: int) -> None:
        while index < len(tree):
            tree[index] += 1
            index += index & -index

    def tree_sum(index: int) -> int:
        total = 0
        while index > 0:
            total += tree[index]
            index -= index & -index
        return total

    tie_x = concordant = discordant = processed = 0
    start = 0
    while start < n:
        end = start + 1
        while end < n and pairs[end][0] == pairs[start][0]:
            end += 1
        group_size = end - start
        tie_x += group_size * (group_size - 1) // 2
        for _, y_value in pairs[start:end]:
            rank = y_rank[y_value]
            concordant += tree_sum(rank - 1)
            discordant += processed - tree_sum(rank)
        for _, y_value in pairs[start:end]:
            tree_add(y_rank[y_value])
        processed += group_size
        start = end
    denom = math.sqrt((n0 - tie_x) * (n0 - tie_y))
    if denom == 0:
        return 0.0
    return round((concordant - discordant) / denom, 4)


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def roc_auc(scores: list[float], labels: list[bool]) -> float | None:
    positives = sum(1 for label in labels if label)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = average_ranks(scores)
    rank_sum_positive = sum(rank for rank, label in zip(ranks, labels) if label)
    auc = (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)
    return round(auc, 4)


def distribution_summary(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    return {
        "count": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 3),
        "median": _quantile(ordered, 0.5),
        "q25": _quantile(ordered, 0.25),
        "q75": _quantile(ordered, 0.75),
    }


def _quantile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return round(ordered[0], 3)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    frac = position - low
    return round(ordered[low] * (1 - frac) + ordered[high] * frac, 3)


# --- Q-A: agreement -------------------------------------------------------


def is_honored(row: ComparisonRow, include_award: bool) -> bool:
    return row.tier in HONORED_TIERS or (include_award and row.is_award)


def agreement_metrics(
    rows: list[ComparisonRow],
    *,
    score_of: Callable[[ComparisonRow], float],
    k_values: Iterable[int],
    include_award: bool,
    include_per_tier: bool,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 20260725,
) -> dict[str, Any]:
    scored = [(score_of(row), row) for row in rows]
    scores = [score for score, _ in scored]
    tiers = [float(row.tier_rank) for _, row in scored]

    reviewer_pairs = [(score, row.reviewer_overall) for score, row in scored if row.reviewer_overall is not None]

    discrimination = [(score, row) for score, row in scored if row.tier == "poster" or is_honored(row, include_award)]

    honored_total = sum(1 for _, row in scored if is_honored(row, include_award))
    oral_total = sum(1 for _, row in scored if row.tier == "oral")
    order = sorted(scored, key=lambda pair: (-pair[0], pair[1].paper_id))

    recall_at_k: dict[str, Any] = {}
    for k in k_values:
        top = order[:k]
        captured = sum(1 for _, row in top if is_honored(row, include_award))
        captured_oral = sum(1 for _, row in top if row.tier == "oral")
        recall_at_k[str(k)] = {
            "recall_honored": round(captured / honored_total, 4) if honored_total else None,
            "precision_honored": round(captured / len(top), 4) if top else None,
            "recall_oral": round(captured_oral / oral_total, 4) if oral_total else None,
        }

    metrics: dict[str, Any] = {
        "n": len(rows),
        "honored_count": honored_total,
        "oral_count": oral_total,
        "kendall_tau_b": {
            "vs_tier": kendall_tau_b(scores, tiers),
            "vs_reviewer_overall": kendall_tau_b(
                [score for score, _ in reviewer_pairs], [value for _, value in reviewer_pairs]
            ),
        },
        "roc_auc": {
            "honored_vs_poster": roc_auc(
                [score for score, _ in discrimination],
                [is_honored(row, include_award) for _, row in discrimination],
            ),
            "oral_vs_rest": roc_auc(scores, [row.tier == "oral" for _, row in scored]),
        },
        "roc_auc_ci95": {
            "honored_vs_poster": bootstrap_roc_auc_ci(
                [score for score, _ in discrimination],
                [is_honored(row, include_award) for _, row in discrimination],
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            ),
            "oral_vs_rest": bootstrap_roc_auc_ci(
                scores,
                [row.tier == "oral" for _, row in scored],
                samples=bootstrap_samples,
                seed=bootstrap_seed + 1,
            ),
        },
        "recall_at_k": recall_at_k,
    }
    if include_per_tier:
        local_percentiles = percentile_ranks(scores)
        percentile_by_id = {
            row.paper_id: percentile
            for percentile, (_, row) in zip(local_percentiles, scored)
        }
        by_tier: dict[str, Any] = {}
        for tier in ("oral", "spotlight", "poster"):
            values = [percentile_by_id[row.paper_id] for _, row in scored if row.tier == tier]
            by_tier[tier] = distribution_summary(values)
        metrics["per_tier_ai_percentile"] = by_tier
    return metrics


def build_agreement_report(
    rows: list[ComparisonRow],
    *,
    k_values: list[int],
    include_award: bool = False,
    coverage: dict[str, Any] | None = None,
    bootstrap_samples: int = 500,
    bootstrap_seed: int = 20260725,
) -> dict[str, Any]:
    tier_rows = [row for row in rows if row.tier in {"oral", "spotlight", "poster"}]
    acceptance = acceptance_outcome_metrics(
        rows,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed + 4,
    )
    tier_usable = bool(tier_rows) and len({row.tier_rank for row in tier_rows}) >= 2
    if not tier_usable and acceptance is None:
        raise ValueError("human labels have fewer than two usable outcome levels")
    full = (
        agreement_metrics(
            tier_rows,
            score_of=lambda row: row.ai_score,
            k_values=k_values,
            include_award=include_award,
            include_per_tier=True,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        if tier_usable
        else None
    )
    strong_rows = [row for row in tier_rows if row.strong_rank is not None]
    strong = None
    if tier_usable and len(strong_rows) >= 2:
        strong = {
            "selection": advancement_metrics(
                full_rows=tier_rows,
                selected_rows=strong_rows,
                include_award=include_award,
            ),
            "cheap_on_same_subset": agreement_metrics(
                strong_rows,
                score_of=lambda row: row.ai_score,
                k_values=k_values,
                include_award=include_award,
                include_per_tier=False,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + 2,
            ),
            "strong_on_same_subset": agreement_metrics(
                strong_rows,
                score_of=lambda row: -float(row.strong_rank),
                k_values=k_values,
                include_award=include_award,
                include_per_tier=False,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + 3,
            ),
        }
    return {
        "analysis": "human_ai_agreement",
        "config": {
            "k_values": k_values,
            "honored_includes_award": include_award,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
        "coverage": coverage,
        "counts": {
            "papers": len(rows),
            "tier_labeled_papers": len(tier_rows),
            "by_tier": tier_counts(rows),
        },
        "full_coverage": full,
        "strong_subset": strong,
        "acceptance_outcome": acceptance,
    }


def advancement_metrics(
    *,
    full_rows: list[ComparisonRow],
    selected_rows: list[ComparisonRow],
    include_award: bool,
) -> dict[str, Any]:
    selected_ids = {row.paper_id for row in selected_rows}
    full_honored = [row for row in full_rows if is_honored(row, include_award)]
    full_orals = [row for row in full_rows if row.tier == "oral"]
    return {
        "selected_papers": len(selected_rows),
        "selection_rate": round(len(selected_rows) / len(full_rows), 4) if full_rows else None,
        "honored_retained": sum(row.paper_id in selected_ids for row in full_honored),
        "honored_total": len(full_honored),
        "honored_recall": (
            round(sum(row.paper_id in selected_ids for row in full_honored) / len(full_honored), 4)
            if full_honored
            else None
        ),
        "oral_retained": sum(row.paper_id in selected_ids for row in full_orals),
        "oral_total": len(full_orals),
        "oral_recall": (
            round(sum(row.paper_id in selected_ids for row in full_orals) / len(full_orals), 4)
            if full_orals
            else None
        ),
    }


def acceptance_outcome_metrics(
    rows: list[ComparisonRow],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any] | None:
    decision_rows = [row for row in rows if row.decision_status in {"accepted", "rejected"}]
    labels = [row.decision_status == "accepted" for row in decision_rows]
    if not any(labels) or all(labels):
        return None
    scores = [row.ai_score for row in decision_rows]
    local_percentiles = percentile_ranks(scores)
    percentile_by_id = {
        row.paper_id: percentile
        for row, percentile in zip(decision_rows, local_percentiles)
    }
    return {
        "n": len(decision_rows),
        "accepted_count": sum(labels),
        "rejected_count": len(labels) - sum(labels),
        "roc_auc_accepted_vs_rejected": roc_auc(scores, labels),
        "roc_auc_ci95": bootstrap_roc_auc_ci(
            scores,
            labels,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "ai_percentile_by_decision": {
            status: distribution_summary(
                [
                    percentile_by_id[row.paper_id]
                    for row in decision_rows
                    if row.decision_status == status
                ]
            )
            for status in ("accepted", "rejected")
        },
    }


def bootstrap_roc_auc_ci(
    scores: list[float],
    labels: list[bool],
    *,
    samples: int,
    seed: int,
) -> list[float] | None:
    if samples <= 0:
        return None
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        sampled_positive = [rng.choice(positives) for _ in positives]
        sampled_negative = [rng.choice(negatives) for _ in negatives]
        estimate = roc_auc(
            sampled_positive + sampled_negative,
            [True] * len(sampled_positive) + [False] * len(sampled_negative),
        )
        if estimate is not None:
            estimates.append(estimate)
    if not estimates:
        return None
    estimates.sort()
    return [_quantile(estimates, 0.025), _quantile(estimates, 0.975)]


# --- Q-B: axis decomposition ----------------------------------------------


def build_axis_decomposition_report(
    rows: list[ComparisonRow],
    *,
    include_award: bool = False,
    coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = [row for row in rows if row.tier in {"oral", "spotlight", "poster"}]
    if not rows:
        raise ValueError("no papers have accepted-presentation tier labels")
    if len({row.tier_rank for row in rows}) < 2:
        raise ValueError("accepted-presentation tier labels have fewer than two outcome levels")
    axis_names = sorted({name for row in rows for name in row.ai_axes})
    results: list[dict[str, Any]] = []
    for axis in axis_names:
        pairs = [(row.ai_axes[axis], row) for row in rows if axis in row.ai_axes]
        if len(pairs) < 2:
            continue
        scores = [value for value, _ in pairs]
        tiers = [float(row.tier_rank) for _, row in pairs]
        discrimination = [(value, row) for value, row in pairs if row.tier == "poster" or is_honored(row, include_award)]
        reviewer_pairs = [(value, row.reviewer_overall) for value, row in pairs if row.reviewer_overall is not None]
        results.append(
            {
                "axis": axis,
                "n": len(pairs),
                "kendall_tau_b_vs_tier": kendall_tau_b(scores, tiers),
                "roc_auc_honored_vs_poster": roc_auc(
                    [value for value, _ in discrimination],
                    [is_honored(row, include_award) for _, row in discrimination],
                ),
                "kendall_tau_b_vs_reviewer_overall": kendall_tau_b(
                    [value for value, _ in reviewer_pairs], [value for _, value in reviewer_pairs]
                ),
            }
        )
    results.sort(key=lambda item: (item["kendall_tau_b_vs_tier"] is not None, item["kendall_tau_b_vs_tier"] or 0.0), reverse=True)

    def tau_of(name: str) -> float | None:
        return next((item["kendall_tau_b_vs_tier"] for item in results if item["axis"] == name), None)

    executive = tau_of("executive_ac_priority")
    conventional = tau_of("conventional_acceptance_strength")
    delta = round(conventional - executive, 4) if executive is not None and conventional is not None else None
    return {
        "analysis": "ai_axis_decomposition",
        "config": {"honored_includes_award": include_award},
        "coverage": coverage,
        "axes": results,
        "impact_vs_polish": {
            "executive_ac_priority_tau_b_vs_tier": executive,
            "conventional_acceptance_strength_tau_b_vs_tier": conventional,
            "delta_conventional_minus_executive": delta,
            "note": "Positive delta means AI's model of conventional reviewer strength tracks human tier better than its impact-forward priority.",
        },
    }
