from __future__ import annotations

from typing import Any

from icml_ai_ac.eval import top_k_metrics
from icml_ai_ac.scoring.frontier_gold import (
    aggregate_sparse_or_hybrid_stats,
    aggregate_tournament_matches,
    build_swiss_round_pairs,
)


def simulate_hybrid_tournament(
    tournament: dict[str, Any],
    *,
    swiss_rounds: int,
    playoff_top_n: int,
) -> dict[str, Any]:
    """Replay a Swiss-plus-playoff schedule from complete all-pairs outcomes."""
    if swiss_rounds <= 0:
        raise ValueError("swiss_rounds must be positive")
    ranked = tournament.get("ranked_papers")
    matches = tournament.get("matches")
    if not isinstance(ranked, list) or not isinstance(matches, list):
        raise ValueError("tournament must contain ranked_papers and matches arrays")

    pool_rows = [
        row
        for row in ranked
        if isinstance(row, dict) and row.get("ranking_source") == "pairwise_tournament"
    ]
    if len(pool_rows) < 2:
        raise ValueError("tournament has fewer than two pairwise-adjudicated papers")
    if not 2 <= playoff_top_n <= len(pool_rows):
        raise ValueError("playoff_top_n must be between 2 and the tournament pool size")

    seed_rank_by_id = {
        str(row["paper_id"]): int(row.get("seed_synthesis_rank") or row.get("rank"))
        for row in pool_rows
    }
    tournament_ids = sorted(seed_rank_by_id, key=seed_rank_by_id.get)
    full_rank_ids = [
        str(row["paper_id"])
        for row in sorted(pool_rows, key=lambda row: int(row["rank"]))
    ]
    match_by_pair: dict[frozenset[str], dict[str, Any]] = {}
    for raw_match in matches:
        if not isinstance(raw_match, dict):
            continue
        paper_a = str(raw_match.get("paper_a") or "")
        paper_b = str(raw_match.get("paper_b") or "")
        if paper_a not in seed_rank_by_id or paper_b not in seed_rank_by_id:
            continue
        key = frozenset((paper_a, paper_b))
        if len(key) != 2 or key in match_by_pair:
            raise ValueError("all-pairs artifact contains an invalid or duplicate pair")
        match_by_pair[key] = raw_match
    expected_pairs = len(tournament_ids) * (len(tournament_ids) - 1) // 2
    if len(match_by_pair) != expected_pairs:
        raise ValueError(
            f"all-pairs artifact is incomplete: expected {expected_pairs}, found {len(match_by_pair)}"
        )

    standings_ids = list(tournament_ids)
    played_pairs: set[frozenset[str]] = set()
    swiss_matches: list[dict[str, Any]] = []
    round_summaries: list[dict[str, Any]] = []
    for round_index in range(swiss_rounds):
        round_pairs = build_swiss_round_pairs(standings_ids, played_pairs)
        if not round_pairs:
            break
        for paper_a, paper_b in round_pairs:
            match = dict(match_by_pair[frozenset((paper_a, paper_b))])
            match["stage"] = f"swiss_round_{round_index + 1}"
            swiss_matches.append(match)
        standings = aggregate_tournament_matches(
            swiss_matches,
            tournament_ids=tournament_ids,
            seed_rank_by_id=seed_rank_by_id,
        )
        standings_ids = [str(row["paper_id"]) for row in standings]
        round_summaries.append(
            {
                "round": round_index + 1,
                "pair_count": len(round_pairs),
                "provisional_top_ids": standings_ids[:playoff_top_n],
            }
        )

    playoff_ids = standings_ids[:playoff_top_n]
    playoff_set = set(playoff_ids)
    hybrid_by_pair = {
        frozenset((str(match["paper_a"]), str(match["paper_b"]))): match
        for match in swiss_matches
    }
    for key, raw_match in match_by_pair.items():
        if key <= playoff_set:
            match = dict(raw_match)
            match["stage"] = "playoff_all_pairs"
            hybrid_by_pair[key] = match
    hybrid_matches = list(hybrid_by_pair.values())
    hybrid_stats = aggregate_sparse_or_hybrid_stats(
        hybrid_matches,
        tournament_ids=tournament_ids,
        seed_rank_by_id=seed_rank_by_id,
        playoff_ids=playoff_ids,
        strategy="swiss_playoff",
    )
    hybrid_ids = [str(row["paper_id"]) for row in hybrid_stats]

    return {
        "evaluation_mode": "simulated_swiss_playoff_from_all_pairs",
        "source_prompt_version": tournament.get("prompt_version"),
        "pool_size": len(tournament_ids),
        "swiss_rounds_requested": swiss_rounds,
        "swiss_rounds_completed": len(round_summaries),
        "playoff_top_n": playoff_top_n,
        "all_pairs_count": expected_pairs,
        "simulated_pair_count": len(hybrid_matches),
        "pair_reduction_fraction": round(1 - len(hybrid_matches) / expected_pairs, 4),
        "playoff_ids": playoff_ids,
        "rounds": round_summaries,
        "ranked_papers": [
            {
                **next(row for row in pool_rows if str(row["paper_id"]) == paper_id),
                "rank": rank,
                "simulation_stats": next(
                    stats for stats in hybrid_stats if str(stats["paper_id"]) == paper_id
                ),
            }
            for rank, paper_id in enumerate(hybrid_ids, start=1)
        ],
        "metrics_vs_all_pairs": {
            "spearman": spearman(full_rank_ids, hybrid_ids),
            "top_10": top_k_metrics(full_rank_ids, hybrid_ids, 10),
            "gold_top_10_recall_at_15": recall_at(full_rank_ids[:10], hybrid_ids[:15]),
            "top_20": top_k_metrics(full_rank_ids, hybrid_ids, min(20, len(full_rank_ids))),
        },
    }


def spearman(reference_ids: list[str], candidate_ids: list[str]) -> float:
    if set(reference_ids) != set(candidate_ids):
        raise ValueError("rankings must cover the same paper IDs")
    n = len(reference_ids)
    if n < 2:
        return 1.0
    candidate_rank = {paper_id: rank for rank, paper_id in enumerate(candidate_ids, start=1)}
    squared_differences = sum(
        (rank - candidate_rank[paper_id]) ** 2
        for rank, paper_id in enumerate(reference_ids, start=1)
    )
    return round(1 - 6 * squared_differences / (n * (n * n - 1)), 4)


def recall_at(reference_ids: list[str], candidate_ids: list[str]) -> float:
    if not reference_ids:
        return 0.0
    return round(len(set(reference_ids) & set(candidate_ids)) / len(set(reference_ids)), 4)
