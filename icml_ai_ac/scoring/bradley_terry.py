"""Bradley-Terry aggregation over pairwise tournament matches.

The frontier card tournament (`frontier_gold.py`) produces a list of pairwise
`matches`, and ranks papers by Copeland-style win points. Win points are simple
and transparent but they do not model opponent strength or provide a
model-based uncertainty estimate.

This module fits a Bradley-Terry model to the same `matches`:

    P(i beats j) = strength_i / (strength_i + strength_j)

using the standard minorization-maximization (MM) update (Zermelo 1929;
Ford 1957; Hunter 2004). The output is a latent log-strength per paper plus an
asymptotic standard error conditional on the observed comparison outcomes.

Design choices:

- **Ties** count as half a win to each side.
- **Optional confidence weighting** lets each comparison contribute its judge
  confidence rather than a unit vote.
- **A smoothing prior** (`prior_strength`) adds virtual wins/losses against a
  reference paper of strength 1. This guarantees a unique, finite MLE even when a
  paper is undefeated, winless, or unplayed, and pulls sparse estimates toward the
  field average. It is a mild MAP regularizer, not a strong prior.
- **Standard errors are conditional and model-based**: they come from the full
  observed-information inverse in log-strength space. They quantify sampling
  uncertainty under the fitted Bradley-Terry model, not uncertainty over judge
  choice, prompts, or model families.

Pure standard library: no numpy, matching the repo's zero-dependency policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable


BRADLEY_TERRY_METHOD = "regularized_bradley_terry_mm"
BRADLEY_TERRY_EVALUATION_MODE = "bradley_terry_pairwise_ranking"

WEIGHT_MODES = ("none", "confidence")
STAGE_MODES = ("auto", "all", "swiss", "playoff")


@dataclass(slots=True)
class MatchTallies:
    players: list[str]
    wins: dict[str, float]
    opponents: dict[str, dict[str, float]]
    decisive_count: float = 0.0
    tie_count: float = 0.0
    skipped_count: int = 0


@dataclass(slots=True)
class BradleyTerryFit:
    log_strength: dict[str, float]
    iterations: int
    converged: bool
    max_delta: float
    approx_standard_error: dict[str, float] = field(default_factory=dict)


def rank_bradley_terry(
    matches: Iterable[dict[str, Any]],
    *,
    ranked_meta: dict[str, dict[str, Any]] | None = None,
    restrict_to: Iterable[str] | None = None,
    weight_mode: str = "none",
    prior_strength: float = 1.0,
    max_iter: int = 5000,
    tol: float = 1e-9,
) -> dict[str, Any]:
    """Fit Bradley-Terry over `matches` and return a serializable ranking payload.

    The payload's `ranked_papers` uses the same shape the rest of the pipeline
    consumes (`paper_id`, `rank`, plus metadata), so it can be fed straight into
    `eval-ranking`.
    """
    if weight_mode not in WEIGHT_MODES:
        raise ValueError(f"weight_mode must be one of {WEIGHT_MODES}, got {weight_mode!r}")
    _validate_fit_config(prior_strength=prior_strength, max_iter=max_iter, tol=tol)

    restrict = {str(paper_id) for paper_id in restrict_to} if restrict_to is not None else None
    tallies = tally_matches(matches, weight_mode=weight_mode, restrict_to=restrict)
    # Papers named in restrict_to but absent from matches should still appear at
    # the neutral reference strength.
    if restrict is not None:
        for paper_id in sorted(restrict):
            if paper_id not in tallies.wins:
                tallies.players.append(paper_id)
                tallies.wins[paper_id] = 0.0
                tallies.opponents[paper_id] = {}
        tallies.players = sorted(set(tallies.players))

    fit = fit_bradley_terry(
        tallies,
        prior_strength=prior_strength,
        max_iter=max_iter,
        tol=tol,
    )

    meta = ranked_meta or {}
    rows: list[dict[str, Any]] = []
    for paper_id in tallies.players:
        theta = fit.log_strength.get(paper_id, 0.0)
        strength = math.exp(theta)
        stderr = fit.approx_standard_error.get(paper_id, float("inf"))
        played = matches_played(tallies, paper_id)
        row_meta = meta.get(paper_id, {})
        rows.append(
            {
                "paper_id": paper_id,
                "title": row_meta.get("title", ""),
                "primary_contribution_class": row_meta.get("primary_contribution_class"),
                "bt_log_strength": round(theta, 6),
                "bt_strength": round(strength, 6),
                "win_prob_vs_reference": round(strength / (strength + 1.0), 6),
                "approx_standard_error": None if math.isinf(stderr) else round(stderr, 6),
                "log_strength_ci95_low": None if math.isinf(stderr) else round(theta - 1.96 * stderr, 6),
                "log_strength_ci95_high": None if math.isinf(stderr) else round(theta + 1.96 * stderr, 6),
                "matches_played": round(played, 3),
                "weighted_wins": round(tallies.wins.get(paper_id, 0.0), 3),
            }
        )

    rows.sort(
        key=lambda row: (
            -fit.log_strength[row["paper_id"]],
            -row["matches_played"],
            row["paper_id"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return {
        "evaluation_mode": BRADLEY_TERRY_EVALUATION_MODE,
        "method": BRADLEY_TERRY_METHOD,
        "ranked_papers": rows,
        "diagnostics": {
            "player_count": len(tallies.players),
            "decisive_comparison_weight": round(tallies.decisive_count, 3),
            "tie_comparison_weight": round(tallies.tie_count, 3),
            "skipped_match_count": tallies.skipped_count,
            "weight_mode": weight_mode,
            "prior_strength": prior_strength,
            "iterations": fit.iterations,
            "converged": fit.converged,
            "max_final_log_strength_delta": round(fit.max_delta, 12),
            "comparison_graph_connected": is_connected(tallies),
            "standard_error_method": "full_observed_fisher_information_inverse",
            "standard_error_note": (
                "Asymptotic and conditional on the observed pairwise judgments; "
                "does not include judge, prompt, or model-family uncertainty."
            ),
        },
    }


def tally_matches(
    matches: Iterable[dict[str, Any]],
    *,
    weight_mode: str = "none",
    restrict_to: set[str] | None = None,
) -> MatchTallies:
    wins: dict[str, float] = {}
    opponents: dict[str, dict[str, float]] = {}
    decisive = 0.0
    ties = 0.0
    skipped = 0

    def ensure(paper_id: str) -> None:
        wins.setdefault(paper_id, 0.0)
        opponents.setdefault(paper_id, {})

    def add_comparison(paper_id: str, opponent: str, weight: float) -> None:
        opponents[paper_id][opponent] = opponents[paper_id].get(opponent, 0.0) + weight

    for match in matches:
        if not isinstance(match, dict):
            skipped += 1
            continue
        paper_a = str(match.get("paper_a") or "")
        paper_b = str(match.get("paper_b") or "")
        if not paper_a or not paper_b or paper_a == paper_b:
            skipped += 1
            continue
        if restrict_to is not None and (paper_a not in restrict_to or paper_b not in restrict_to):
            skipped += 1
            continue
        weight = comparison_weight(match, weight_mode)
        if weight <= 0:
            skipped += 1
            continue
        winner = str(match.get("winner") or "").strip().lower()
        if winner not in {"paper_a", "paper_b", "tie"}:
            skipped += 1
            continue
        ensure(paper_a)
        ensure(paper_b)
        add_comparison(paper_a, paper_b, weight)
        add_comparison(paper_b, paper_a, weight)
        if winner == "paper_a":
            wins[paper_a] += weight
            decisive += weight
        elif winner == "paper_b":
            wins[paper_b] += weight
            decisive += weight
        elif winner == "tie":
            wins[paper_a] += weight / 2.0
            wins[paper_b] += weight / 2.0
            ties += weight
    players = sorted(wins)
    return MatchTallies(
        players=players,
        wins=wins,
        opponents=opponents,
        decisive_count=decisive,
        tie_count=ties,
        skipped_count=skipped,
    )


def comparison_weight(match: dict[str, Any], weight_mode: str) -> float:
    if weight_mode == "confidence":
        return _clamp(_number(match.get("confidence")), 0.0, 1.0)
    return 1.0


def fit_bradley_terry(
    tallies: MatchTallies,
    *,
    prior_strength: float = 1.0,
    max_iter: int = 5000,
    tol: float = 1e-9,
) -> BradleyTerryFit:
    _validate_fit_config(prior_strength=prior_strength, max_iter=max_iter, tol=tol)
    players = tallies.players
    if not players:
        return BradleyTerryFit(log_strength={}, iterations=0, converged=True, max_delta=0.0)

    # Work in strength space p = exp(theta); reference paper has fixed strength 1.
    strength = {paper_id: 1.0 for paper_id in players}
    alpha = prior_strength
    iterations = 0
    max_delta = 0.0
    converged = False

    for iterations in range(1, max_iter + 1):
        updated: dict[str, float] = {}
        for paper_id in players:
            p_i = strength[paper_id]
            denom = 2.0 * alpha / (p_i + 1.0)
            for opponent, weight in tallies.opponents[paper_id].items():
                denom += weight / (p_i + strength[opponent])
            numer = tallies.wins[paper_id] + alpha
            updated[paper_id] = numer / denom if denom > 0 else p_i

        max_delta = max(
            abs(math.log(updated[paper_id]) - math.log(strength[paper_id]))
            for paper_id in players
        )
        strength = updated
        if max_delta < tol:
            converged = True
            break

    log_strength = {paper_id: math.log(value) for paper_id, value in strength.items()}
    stderr = approximate_standard_errors(tallies, strength, prior_strength=alpha)
    return BradleyTerryFit(
        log_strength=log_strength,
        iterations=iterations,
        converged=converged,
        max_delta=max_delta,
        approx_standard_error=stderr,
    )


def approximate_standard_errors(
    tallies: MatchTallies,
    strength: dict[str, float],
    *,
    prior_strength: float,
) -> dict[str, float]:
    """Asymptotic log-strength SE from the full observed-information inverse.

    For a Bradley-Terry model the information contributed by a comparison between
    i and j is n_ij * p_i * p_j / (p_i + p_j)^2. The fixed-reference prior makes
    the information matrix positive definite, so its inverse identifies the
    marginal variance of every fitted log-strength.
    """
    players = list(tallies.players)
    index = {paper_id: position for position, paper_id in enumerate(players)}
    information = [[0.0 for _ in players] for _ in players]
    for paper_id in players:
        i = index[paper_id]
        p_i = strength[paper_id]
        for opponent, weight in tallies.opponents.get(paper_id, {}).items():
            if weight <= 0 or index[opponent] <= i:
                continue
            j = index[opponent]
            p_j = strength[opponent]
            contribution = weight * (p_i * p_j) / ((p_i + p_j) ** 2)
            information[i][i] += contribution
            information[j][j] += contribution
            information[i][j] -= contribution
            information[j][i] -= contribution
        information[i][i] += 2.0 * prior_strength * p_i / ((p_i + 1.0) ** 2)

    inverse_diagonal = _inverse_diagonal_positive_definite(information)
    return {
        paper_id: math.sqrt(max(inverse_diagonal[index[paper_id]], 0.0))
        for paper_id in players
    }


def matches_played(tallies: MatchTallies, paper_id: str) -> float:
    return sum(tallies.opponents.get(paper_id, {}).values())


def is_connected(tallies: MatchTallies) -> bool:
    players = tallies.players
    if len(players) <= 1:
        return True
    start = players[0]
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for opponent, weight in tallies.opponents.get(current, {}).items():
            if weight <= 0:
                continue
            if opponent not in seen:
                seen.add(opponent)
                stack.append(opponent)
    return len(seen) == len(players)


def ranked_meta_from_payload(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract title/contribution-class metadata from a tournament payload."""
    meta: dict[str, dict[str, Any]] = {}
    ranked = payload.get("ranked_papers")
    if isinstance(ranked, list):
        for row in ranked:
            if not isinstance(row, dict):
                continue
            paper_id = str(row.get("paper_id") or "")
            if not paper_id:
                continue
            meta[paper_id] = {
                "title": row.get("title") or "",
                "primary_contribution_class": row.get("primary_contribution_class"),
            }
    return meta


def tournament_pool_from_payload(payload: dict[str, Any]) -> list[str] | None:
    """Return the pairwise-adjudicated pool, if the payload records one."""
    summary = payload.get("tournament_summary")
    if isinstance(summary, dict):
        ids = [
            str(row.get("paper_id") or "")
            for row in payload.get("ranked_papers", [])
            if isinstance(row, dict) and row.get("ranking_source") == "pairwise_tournament"
        ]
        ids = [paper_id for paper_id in ids if paper_id]
        if ids:
            return ids
    return None


def select_tournament_stage(
    payload: dict[str, Any],
    *,
    stage: str = "auto",
) -> tuple[list[dict[str, Any]], list[str] | None, dict[str, Any]]:
    """Select a statistically coherent match set from a tournament payload."""
    if stage not in STAGE_MODES:
        raise ValueError(f"stage must be one of {STAGE_MODES}, got {stage!r}")
    matches = payload.get("matches")
    if not isinstance(matches, list):
        raise ValueError("tournament payload has no matches array")
    summary = payload.get("tournament_summary")
    summary = summary if isinstance(summary, dict) else {}
    strategy = str(payload.get("tournament_strategy") or summary.get("strategy") or "all_pairs")
    selected_stage = stage
    if stage == "auto":
        selected_stage = "playoff" if strategy == "swiss_playoff" else ("swiss" if strategy == "swiss" else "all")

    pool_ids = tournament_pool_from_payload(payload)
    if selected_stage == "all":
        selected = [match for match in matches if isinstance(match, dict)]
        restrict = pool_ids
    elif selected_stage == "swiss":
        schedule = summary.get("schedule") or payload.get("schedule")
        if not isinstance(schedule, list):
            if strategy == "swiss":
                selected = [match for match in matches if isinstance(match, dict)]
            else:
                raise ValueError("cannot identify Swiss matches without tournament schedule metadata")
        else:
            swiss_count = sum(
                int(row.get("pair_count") or 0)
                for row in schedule
                if isinstance(row, dict) and row.get("stage") == "swiss"
            )
            if swiss_count <= 0 or swiss_count > len(matches):
                raise ValueError("tournament schedule has an invalid Swiss match count")
            selected = [match for match in matches[:swiss_count] if isinstance(match, dict)]
        restrict = pool_ids
    else:
        playoff_ids = summary.get("playoff_ids") or payload.get("playoff_ids")
        if not isinstance(playoff_ids, list) or len(playoff_ids) < 2:
            raise ValueError("tournament payload has no valid playoff_ids")
        restrict = [str(paper_id) for paper_id in playoff_ids if paper_id]
        playoff_set = set(restrict)
        selected = [
            match
            for match in matches
            if isinstance(match, dict)
            and str(match.get("paper_a") or "") in playoff_set
            and str(match.get("paper_b") or "") in playoff_set
        ]
        if not selected:
            raise ValueError("no playoff matches found for playoff_ids")

    return selected, restrict, {
        "requested_stage": stage,
        "selected_stage": selected_stage,
        "tournament_strategy": strategy,
        "source_match_count": len(matches),
        "selected_match_count": len(selected),
    }


def _validate_fit_config(*, prior_strength: float, max_iter: int, tol: float) -> None:
    if not math.isfinite(prior_strength) or prior_strength <= 0:
        raise ValueError("prior_strength must be finite and greater than zero")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and greater than zero")


def _inverse_diagonal_positive_definite(matrix: list[list[float]]) -> list[float]:
    """Return diag(A^-1) for a symmetric positive-definite matrix via Cholesky."""
    size = len(matrix)
    if size == 0:
        return []
    lower = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            value = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if value <= 0 or not math.isfinite(value):
                    raise ValueError("observed information matrix is not positive definite")
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = value / lower[j][j]

    diagonal: list[float] = []
    for target in range(size):
        forward = [0.0] * size
        for i in range(size):
            rhs = 1.0 if i == target else 0.0
            forward[i] = (rhs - sum(lower[i][k] * forward[k] for k in range(i))) / lower[i][i]
        backward = [0.0] * size
        for i in range(size - 1, -1, -1):
            backward[i] = (
                forward[i] - sum(lower[k][i] * backward[k] for k in range(i + 1, size))
            ) / lower[i][i]
        diagonal.append(backward[target])
    return diagonal


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0
