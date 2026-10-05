"""Join canonical AI ranking signals with human outcomes.

This is the deterministic foundation of the human-vs-AI comparison. It loads:

- an enriched paper manifest (human labels: presentation tier from
  ``extra.is_oral`` / ``extra.is_spotlight``, reviewer means from
  ``extra.openreview_scores``, awards from ``extra.award_labels``), and
- a full-coverage AI ranking JSONL (preferably the deterministic cheap-ensemble
  aggregate, with exactly one row per paper), and
- optionally a strong ranking JSON (tournament / Bradley-Terry) for the finalist
  subset, shown as a cross-reference.

The loader reports exact join coverage and refuses partial populations unless
the caller explicitly lowers the coverage threshold. Duplicate AI rows are an
error rather than an invitation to select the most favorable judgment.

The divergence report supports three distinct human axes:

- presentation tier among accepted papers,
- published reviewer overall scores, and
- accepted versus publicly visible rejected papers.

The gallery is an explicit windowed view of the reported distribution (not a
hand-picked set): the same divergence score orders every paper, and the report
records how many cases exist beyond the shown window.

No model calls: this whole module is deterministic and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from icml_ai_ac.storage import read_jsonl, read_paper_records


HONORED_TIERS = ("oral", "spotlight")

# AI rationale fields surfaced verbatim on each case card (path within `scores`).
RATIONALE_PATHS: dict[str, tuple[str, ...]] = {
    "why_it_might_matter": ("executive_lens", "why_it_might_matter"),
    "sweeping_impact_scenario": ("executive_lens", "sweeping_impact_scenario"),
    "top_paper_case": ("ranking_signals", "top_paper_case"),
    "broad_scientific_impact_argument": ("ranking_signals", "broad_scientific_impact_argument"),
    "dealbreaker_risks": ("ranking_signals", "dealbreaker_risks"),
    "why_not_higher": ("calibration", "why_not_higher"),
    "why_not_lower": ("calibration", "why_not_lower"),
    "reviewer_vs_executive_delta": ("calibration", "reviewer_vs_executive_delta"),
    "human_review_alignment": ("reviewer_lens", "human_review_alignment"),
    "likely_reviewer_concerns": ("reviewer_lens", "likely_reviewer_concerns"),
    "observed_weaknesses": ("evidence_audit", "observed_weaknesses"),
    "contamination_sensitivity": ("uncertainty", "contamination_sensitivity"),
}

# Numeric axes carried on each case so the taxonomy (QL-B) can ground reason
# codes in the impact-vs-polish contrast rather than the AI's prose alone.
AXIS_SNAPSHOT_PATHS: dict[str, tuple[str, ...]] = {
    "executive_ac_priority": ("scores", "executive_ac_priority"),
    "conventional_acceptance_strength": ("scores", "conventional_acceptance_strength"),
    "technical_soundness": ("scores", "technical_soundness"),
    "novelty": ("scores", "novelty"),
    "broader_science_impact_forecast": ("scores", "broader_science_impact_forecast"),
    "ml_field_impact_forecast": ("scores", "ml_field_impact_forecast"),
    "field_building_potential": ("impact_axes", "field_building_potential"),
    "benchmark_or_dataset_value": ("impact_axes", "benchmark_or_dataset_value"),
}


@dataclass(slots=True)
class ComparisonRow:
    paper_id: str
    title: str
    abstract: str
    contribution_class: str | None
    # human outcome
    tier: str
    tier_rank: int
    decision_status: str
    is_award: bool
    award_labels: list[Any]
    reviewer_overall: float | None
    reviewer_soundness: float | None
    reviewer_confidence: float | None
    review_count: int
    # ai signal
    ai_score: float
    ai_signal_kind: str
    ai_reported_percentile: float | None
    ai_rank: int = 0
    ai_percentile: float = 0.0
    strong_rank: int | None = None
    # derived divergence (filled by build_divergence_report)
    human_percentile: float | None = None
    comparison_ai_percentile: float | None = None
    residual: float | None = None
    rationale: dict[str, Any] = field(default_factory=dict)
    ai_axes: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ComparisonDataset:
    rows: list[ComparisonRow]
    coverage: dict[str, Any]


def load_comparison_dataset(
    *,
    manifest: Path,
    ai_scores: Path,
    strong_ranking: Path | None = None,
    min_coverage: float = 1.0,
) -> ComparisonDataset:
    if not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must be in [0, 1]")
    ai_by_paper = load_ai_scores(ai_scores)
    strong_rank_by_paper = load_strong_ranks(strong_ranking) if strong_ranking else {}
    records = list(read_paper_records(manifest))
    manifest_ids = {record.paper_id for record in records}
    if len(manifest_ids) != len(records):
        raise ValueError("manifest contains duplicate paper_id rows")

    rows: list[ComparisonRow] = []
    signal_kinds: dict[str, int] = {}
    for record in records:
        ai_row = ai_by_paper.get(record.paper_id)
        if ai_row is None:
            continue
        scores = ai_row.get("scores")
        if not isinstance(scores, dict):
            continue
        tier, tier_rank = tier_of(record.extra)
        reviewer = reviewer_means(record.extra)
        ai_score, reported_percentile, signal_kind = ai_position(ai_row)
        signal_kinds[signal_kind] = signal_kinds.get(signal_kind, 0) + 1
        rows.append(
            ComparisonRow(
                paper_id=record.paper_id,
                title=record.title or "",
                abstract=record.abstract or "",
                contribution_class=contribution_class_of(scores),
                tier=tier,
                tier_rank=tier_rank,
                decision_status=decision_status_of(record.extra),
                is_award=bool(record.extra.get("is_award_paper")),
                award_labels=list(record.extra.get("award_labels") or []),
                reviewer_overall=reviewer["overall"],
                reviewer_soundness=reviewer["soundness"],
                reviewer_confidence=reviewer["confidence"],
                review_count=reviewer["count"],
                ai_score=ai_score,
                ai_signal_kind=signal_kind,
                ai_reported_percentile=reported_percentile,
                strong_rank=strong_rank_by_paper.get(record.paper_id),
                rationale=extract_rationale(scores),
                ai_axes=extract_axes(scores),
            )
        )

    joined_ids = {row.paper_id for row in rows}
    coverage = {
        "manifest_papers": len(records),
        "ai_score_papers": len(ai_by_paper),
        "joined_papers": len(rows),
        "coverage_fraction": round(len(rows) / len(records), 6) if records else 0.0,
        "missing_ai_score_count": len(manifest_ids - joined_ids),
        "missing_ai_score_ids": sorted(manifest_ids - joined_ids)[:100],
        "extra_ai_score_count": len(set(ai_by_paper) - manifest_ids),
        "extra_ai_score_ids": sorted(set(ai_by_paper) - manifest_ids)[:100],
        "ai_signal_kinds": signal_kinds,
        "unknown_tier_count": sum(1 for row in rows if row.tier == "unknown"),
        "unknown_decision_count": sum(1 for row in rows if row.decision_status == "unknown"),
    }
    if coverage["coverage_fraction"] < min_coverage:
        raise ValueError(
            "AI/manifest coverage "
            f"{coverage['joined_papers']}/{coverage['manifest_papers']} "
            f"({coverage['coverage_fraction']:.3f}) is below required {min_coverage:.3f}"
        )

    _validate_ai_signal_population(
        rows,
        require_contiguous=(
            coverage["missing_ai_score_count"] == 0
            and coverage["extra_ai_score_count"] == 0
        ),
    )
    _derive_ai_positions(rows)
    return ComparisonDataset(rows=rows, coverage=coverage)


def load_comparison_rows(
    *,
    manifest: Path,
    ai_scores: Path,
    strong_ranking: Path | None = None,
    min_coverage: float = 0.0,
) -> list[ComparisonRow]:
    return load_comparison_dataset(
        manifest=manifest,
        ai_scores=ai_scores,
        strong_ranking=strong_ranking,
        min_coverage=min_coverage,
    ).rows


def _derive_ai_positions(rows: list[ComparisonRow]) -> None:
    ai_percentiles = percentile_ranks([row.ai_score for row in rows])
    for row, percentile in zip(rows, ai_percentiles):
        row.ai_percentile = percentile
    for rank, row in enumerate(sorted(rows, key=lambda r: (-r.ai_score, r.paper_id), reverse=False), start=1):
        row.ai_rank = rank


def _validate_ai_signal_population(
    rows: list[ComparisonRow],
    *,
    require_contiguous: bool,
) -> None:
    signal_kinds = {row.ai_signal_kind for row in rows}
    if len(signal_kinds) > 1:
        raise ValueError(
            "AI score input mixes incompatible ranking signal types: "
            + ", ".join(sorted(signal_kinds))
        )
    if signal_kinds == {"ensemble_aggregate_rank"}:
        ranks = sorted(int(-row.ai_score) for row in rows)
        if any(rank < 1 for rank in ranks) or len(set(ranks)) != len(ranks):
            raise ValueError("ensemble aggregate ranks must be positive and unique")
        if require_contiguous and ranks != list(range(1, len(rows) + 1)):
            raise ValueError(
                "ensemble aggregate ranks must be unique and contiguous over the analyzed population"
            )


def build_divergence_report(
    rows: list[ComparisonRow],
    *,
    human_axis: str = "tier",
    top_frac: float = 0.10,
    cases_per_direction: int = 20,
    min_reviews: int = 0,
    coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if human_axis not in {"tier", "reviewer", "decision"}:
        raise ValueError("human_axis must be 'tier', 'reviewer', or 'decision'")
    if not 0 < top_frac < 1:
        raise ValueError("top_frac must be in (0, 1)")
    if cases_per_direction < 1:
        raise ValueError("cases_per_direction must be at least 1")
    if min_reviews < 0:
        raise ValueError("min_reviews must be non-negative")

    eligible = eligible_for_axis(rows, human_axis=human_axis, min_reviews=min_reviews)
    if not eligible:
        raise ValueError(f"no papers have usable human labels for axis {human_axis!r}")
    human_values = human_metric_values(eligible, human_axis)
    observed_human_values = {value for value in human_values if value is not None}
    if len(observed_human_values) < 2:
        raise ValueError(f"human axis {human_axis!r} has fewer than two observed outcome levels")
    scored = [row for row, value in zip(eligible, human_values) if value is not None]
    human_percentiles = percentile_ranks([value for value in human_values if value is not None])
    comparison_ai_percentiles = percentile_ranks([row.ai_score for row in scored])
    for row, human_percentile, ai_percentile in zip(
        scored,
        human_percentiles,
        comparison_ai_percentiles,
    ):
        row.human_percentile = human_percentile
        row.comparison_ai_percentile = ai_percentile
        row.residual = round(ai_percentile - human_percentile, 3)

    residual_rows = [row for row in scored if row.residual is not None]
    top_cut = 100.0 * (1.0 - top_frac)
    bottom_cut = 100.0 * top_frac
    if human_axis == "tier":
        gem_candidates = (
            row
            for row in residual_rows
            if row.tier == "poster"
            and not row.is_award
            and (row.comparison_ai_percentile or 0.0) >= top_cut
            and (row.residual or 0.0) > 0
        )
        blind_candidates = (
            row
            for row in residual_rows
            if (row.tier in HONORED_TIERS or row.is_award)
            and (row.comparison_ai_percentile or 0.0) <= bottom_cut
            and (row.residual or 0.0) < 0
        )
        gem_description = "Accepted posters in the configured AI top tail."
        blind_description = "Orals, spotlights, or award papers in the configured AI bottom tail."
    elif human_axis == "decision":
        gem_candidates = (
            row
            for row in residual_rows
            if row.decision_status == "rejected"
            and (row.comparison_ai_percentile or 0.0) >= top_cut
            and (row.residual or 0.0) > 0
        )
        blind_candidates = (
            row
            for row in residual_rows
            if row.decision_status == "accepted"
            and (row.comparison_ai_percentile or 0.0) <= bottom_cut
            and (row.residual or 0.0) < 0
        )
        gem_description = "Publicly visible rejected papers in the configured AI top tail."
        blind_description = "Accepted papers in the configured AI bottom tail."
    else:
        gem_candidates = (
            row
            for row in residual_rows
            if (row.comparison_ai_percentile or 0.0) >= top_cut
            and (row.human_percentile or 0.0) <= bottom_cut
            and (row.residual or 0.0) > 0
        )
        blind_candidates = (
            row
            for row in residual_rows
            if (row.comparison_ai_percentile or 0.0) <= bottom_cut
            and (row.human_percentile or 0.0) >= top_cut
            and (row.residual or 0.0) < 0
        )
        gem_description = "Papers in the AI top tail and published-reviewer bottom tail."
        blind_description = "Papers in the published-reviewer top tail and AI bottom tail."

    gems_all = sorted(
        gem_candidates,
        key=lambda r: (
            -(r.residual or 0.0),
            -(r.comparison_ai_percentile or 0.0),
            r.paper_id,
        ),
    )
    blind_all = sorted(
        blind_candidates,
        key=lambda r: (
            (r.residual or 0.0),
            r.comparison_ai_percentile or 0.0,
            r.paper_id,
        ),
    )
    gems = gems_all[:cases_per_direction]
    blind_spots = blind_all[:cases_per_direction]

    return {
        "analysis": "human_ai_divergence_gallery",
        "config": {
            "human_axis": human_axis,
            "top_frac": top_frac,
            "cases_per_direction": cases_per_direction,
            "min_reviews": min_reviews,
        },
        "counts": {
            "papers": len(rows),
            "eligible_papers": len(eligible),
            "scored_papers": len(residual_rows),
            "by_tier": tier_counts(rows),
            "by_decision": decision_counts(rows),
            "award_papers": sum(1 for row in rows if row.is_award),
        },
        "coverage": coverage,
        "tier_ai_decile_crosstab": tier_ai_decile_crosstab(rows, top_frac=top_frac),
        "residual_summary": residual_summary(residual_rows),
        "overlooked_gems": {
            "description": gem_description,
            "total_matching": len(gems_all),
            "shown": len(gems),
            "cases": [case_card(row) for row in gems],
            "case_pool": [case_card(row) for row in gems_all],
        },
        "blind_spots": {
            "description": blind_description,
            "total_matching": len(blind_all),
            "shown": len(blind_spots),
            "cases": [case_card(row) for row in blind_spots],
            "case_pool": [case_card(row) for row in blind_all],
        },
    }


# --- human-side helpers ---------------------------------------------------


def tier_of(extra: dict[str, Any]) -> tuple[str, int]:
    if decision_status_of(extra) == "rejected":
        return "unknown", -1
    if extra.get("is_oral"):
        return "oral", 2
    if extra.get("is_spotlight"):
        return "spotlight", 1
    if (
        "is_oral" in extra
        or "is_spotlight" in extra
        or str(extra.get("presentation_type") or "").lower() == "poster"
        or "poster" in str(extra.get("acceptance_tier") or "").lower()
    ):
        return "poster", 0
    return "unknown", -1


def decision_status_of(extra: dict[str, Any]) -> str:
    values = [
        extra.get("decision"),
        extra.get("decision_label"),
        extra.get("venue"),
        extra.get("acceptance_tier"),
    ]
    text = " ".join(str(value).lower() for value in values if value not in (None, ""))
    if any(label in text for label in ("reject", "desk reject")):
        return "rejected"
    if any(label in text for label in ("accept", "oral", "spotlight", "poster")):
        return "accepted"
    if (
        "is_oral" in extra
        or "is_spotlight" in extra
        or extra.get("is_award_paper")
        or extra.get("presentation_type")
    ):
        return "accepted"
    return "unknown"


def reviewer_means(extra: dict[str, Any]) -> dict[str, Any]:
    scores = extra.get("openreview_scores") if isinstance(extra.get("openreview_scores"), dict) else {}
    return {
        "overall": as_float(scores.get("overall_mean")),
        "soundness": as_float(scores.get("soundness_mean")),
        "confidence": as_float(scores.get("confidence_mean")),
        "count": int(scores.get("review_count") or 0),
    }


# --- ai-side helpers ------------------------------------------------------


def load_ai_scores(path: Path) -> dict[str, dict[str, Any]]:
    """Load exactly one canonical AI ranking row per paper."""
    rows_by_paper: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in read_jsonl(path):
        if row.get("status") not in {None, "ok"}:
            continue
        paper_id = str(row.get("paper_id") or "")
        scores = row.get("scores")
        if not paper_id or not isinstance(scores, dict):
            continue
        if paper_id in rows_by_paper:
            duplicates.append(paper_id)
            continue
        rows_by_paper[paper_id] = row
    if duplicates:
        examples = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(
            f"AI score input has duplicate paper rows ({len(set(duplicates))} papers; examples: {examples}). "
            "Pass the unique full-coverage ensemble aggregate, not raw repeated judgments."
        )
    return rows_by_paper


def ai_position(row: dict[str, Any]) -> tuple[float, float | None, str]:
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    core = scores.get("scores") if isinstance(scores.get("scores"), dict) else {}
    calibration = scores.get("calibration") if isinstance(scores.get("calibration"), dict) else {}
    aggregate_rank = as_float(row.get("aggregate_rank"))
    if aggregate_rank is None and isinstance(row.get("shortlist_metrics"), dict):
        aggregate_rank = as_float(row.get("rank"))
    if aggregate_rank is not None and aggregate_rank > 0:
        priority = -aggregate_rank
        signal_kind = "ensemble_aggregate_rank"
    else:
        priority = as_float(core.get("executive_ac_priority"))
        signal_kind = "executive_ac_priority"
    if priority is None:
        priority = as_float(core.get("overall_significance"))
        signal_kind = "overall_significance"
    if priority is None:
        raise ValueError(f"AI row {row.get('paper_id')!r} has no supported ranking signal")
    reported = as_float(calibration.get("estimated_percentile_among_accepted_papers"))
    return float(priority), reported, signal_kind


def contribution_class_of(scores: dict[str, Any]) -> str | None:
    profile = scores.get("contribution_profile") if isinstance(scores.get("contribution_profile"), dict) else {}
    value = profile.get("primary_contribution_class")
    return str(value) if isinstance(value, str) and value else None


def extract_rationale(scores: dict[str, Any]) -> dict[str, Any]:
    rationale: dict[str, Any] = {}
    for label, path in RATIONALE_PATHS.items():
        value = nested_get(scores, *path)
        if value not in (None, "", [], {}):
            rationale[label] = value
    return rationale


def extract_axes(scores: dict[str, Any]) -> dict[str, float]:
    axes: dict[str, float] = {}
    for label, path in AXIS_SNAPSHOT_PATHS.items():
        value = as_float(nested_get(scores, *path))
        if value is not None:
            axes[label] = value
    return axes


def load_strong_ranks(path: Path) -> dict[str, int]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    ranked = payload.get("ranked_papers") if isinstance(payload, dict) else None
    ranks: dict[str, int] = {}
    if isinstance(ranked, list):
        for row in ranked:
            if not isinstance(row, dict):
                continue
            paper_id = str(row.get("paper_id") or "")
            rank = row.get("rank")
            if paper_id and isinstance(rank, int):
                ranks[paper_id] = rank
    return ranks


# --- aggregation helpers --------------------------------------------------


def human_metric_values(rows: list[ComparisonRow], human_axis: str) -> list[float | None]:
    if human_axis == "tier":
        # Awards are the strongest human honor; rank them above orals.
        return [float(3 if row.is_award else row.tier_rank) for row in rows]
    if human_axis == "decision":
        return [
            1.0 if row.decision_status == "accepted" else 0.0 if row.decision_status == "rejected" else None
            for row in rows
        ]
    return [row.reviewer_overall for row in rows]


def eligible_for_axis(
    rows: list[ComparisonRow],
    *,
    human_axis: str,
    min_reviews: int,
) -> list[ComparisonRow]:
    if human_axis == "tier":
        return [row for row in rows if row.tier in {"oral", "spotlight", "poster"}]
    if human_axis == "decision":
        return [row for row in rows if row.decision_status in {"accepted", "rejected"}]
    return [
        row
        for row in rows
        if row.reviewer_overall is not None and row.review_count >= min_reviews
    ]


def percentile_ranks(values: list[float]) -> list[float]:
    """Mean percentile rank (0-100); higher value -> higher percentile, ties averaged."""
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [50.0]
    ordered = sorted(values)
    percentile_by_value: dict[float, float] = {}
    start = 0
    while start < n:
        end = start + 1
        while end < n and ordered[end] == ordered[start]:
            end += 1
        percentile_by_value[ordered[start]] = round(
            100.0 * (start + 0.5 * (end - start)) / n,
            3,
        )
        start = end
    return [percentile_by_value[value] for value in values]


def tier_counts(rows: list[ComparisonRow]) -> dict[str, int]:
    counts = {"oral": 0, "spotlight": 0, "poster": 0, "unknown": 0}
    for row in rows:
        counts[row.tier] = counts.get(row.tier, 0) + 1
    return counts


def decision_counts(rows: list[ComparisonRow]) -> dict[str, int]:
    counts = {"accepted": 0, "rejected": 0, "unknown": 0}
    for row in rows:
        counts[row.decision_status] = counts.get(row.decision_status, 0) + 1
    return counts


def tier_ai_decile_crosstab(rows: list[ComparisonRow], *, top_frac: float) -> dict[str, Any]:
    cut = 100.0 * (1.0 - top_frac)
    table: dict[str, dict[str, int]] = {}
    tier_rows = [row for row in rows if row.tier != "unknown"]
    local_percentiles = percentile_ranks([row.ai_score for row in tier_rows])
    for row, percentile in zip(tier_rows, local_percentiles):
        bucket = "ai_top" if percentile >= cut else "ai_rest"
        table.setdefault(row.tier, {"ai_top": 0, "ai_rest": 0})[bucket] += 1
    return {"ai_top_threshold_percentile": round(cut, 3), "table": table}


def residual_summary(rows: list[ComparisonRow]) -> dict[str, Any]:
    residuals = sorted(row.residual for row in rows if row.residual is not None)
    if not residuals:
        return {"count": 0}
    return {
        "count": len(residuals),
        "min": residuals[0],
        "q25": quantile(residuals, 0.25),
        "median": quantile(residuals, 0.50),
        "q75": quantile(residuals, 0.75),
        "max": residuals[-1],
    }


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return round(sorted_values[0], 3)
    position = q * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    frac = position - low
    return round(sorted_values[low] * (1 - frac) + sorted_values[high] * frac, 3)


def case_card(row: ComparisonRow) -> dict[str, Any]:
    return {
        "paper_id": row.paper_id,
        "title": row.title,
        "abstract": row.abstract,
        "contribution_class": row.contribution_class,
        "human": {
            "tier": row.tier,
            "decision_status": row.decision_status,
            "is_award": row.is_award,
            "award_labels": row.award_labels,
            "reviewer_overall": row.reviewer_overall,
            "reviewer_soundness": row.reviewer_soundness,
            "reviewer_confidence": row.reviewer_confidence,
            "review_count": row.review_count,
        },
        "ai": {
            "ranking_signal": round(row.ai_score, 4),
            "ranking_signal_kind": row.ai_signal_kind,
            "executive_ac_priority": row.ai_axes.get("executive_ac_priority"),
            "rank": row.ai_rank,
            "percentile": row.ai_percentile,
            "reported_percentile": row.ai_reported_percentile,
            "strong_rank": row.strong_rank,
        },
        "divergence": {
            "residual": row.residual,
            "human_percentile": row.human_percentile,
            "comparison_ai_percentile": row.comparison_ai_percentile,
        },
        "ai_axes": row.ai_axes,
        "ai_rationale": row.rationale,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = ["# AI-vs-human divergence gallery", ""]
    counts = report["counts"]
    lines.append(
        f"Papers: {counts['papers']} (oral={counts['by_tier'].get('oral', 0)}, "
        f"spotlight={counts['by_tier'].get('spotlight', 0)}, poster={counts['by_tier'].get('poster', 0)}, "
        f"awards={counts['award_papers']}). Human axis: {report['config']['human_axis']}."
    )
    lines.append("")
    for key, heading in (("overlooked_gems", "Overlooked gems"), ("blind_spots", "AI blind spots")):
        section = report[key]
        lines.append(f"## {heading} (showing {section['shown']} of {section['total_matching']})")
        lines.append(f"_{section['description']}_")
        lines.append("")
        for card in section["cases"]:
            lines.extend(render_case_markdown(card))
        lines.append("")
    return "\n".join(lines)


def render_case_markdown(card: dict[str, Any]) -> list[str]:
    human = card["human"]
    ai = card["ai"]
    lines = [
        f"### {card['title'] or card['paper_id']} (`{card['paper_id']}`)",
        f"- **Human:** decision={human['decision_status']}, tier={human['tier']}"
        + (", award" if human["is_award"] else "")
        + (f", reviewer overall {human['reviewer_overall']}" if human["reviewer_overall"] is not None else ""),
        f"- **AI:** rank {ai['rank']} (percentile {ai['percentile']}, signal={ai['ranking_signal_kind']})"
        + (f", strong rank {ai['strong_rank']}" if ai["strong_rank"] is not None else ""),
        f"- **Residual (AI - human):** {card['divergence']['residual']}",
    ]
    for label in ("sweeping_impact_scenario", "top_paper_case", "reviewer_vs_executive_delta", "why_not_higher", "dealbreaker_risks"):
        value = card["ai_rationale"].get(label)
        if value:
            lines.append(f"- _{label}:_ {inline(value)}")
    lines.append("")
    return lines


# --- small utilities ------------------------------------------------------


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def nested_get(row: dict[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def inline(value: Any, limit: int = 300) -> str:
    if isinstance(value, list):
        value = "; ".join(str(item) for item in value)
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit].rstrip() + "..."
