"""Build the static blog data bundle from frozen experiment artifacts."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARXIV_OVERRIDES = ROOT / "site/arxiv_overrides.json"
OUTPUT = ROOT / "site/data.js"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def human_tier(extra: dict[str, Any]) -> str:
    if extra.get("is_oral"):
        return "oral"
    if extra.get("is_spotlight"):
        return "spotlight"
    return "poster"


def track_name(extra: dict[str, Any]) -> str:
    source = str(extra.get("sourceurl") or "")
    if source.endswith("/Conference"):
        return "main"
    if "Position_Paper_Track" in source:
        return "position"
    return "journal"


def normalized_title(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[*_`]", "", text)
    return " ".join(text.casefold().split())


def primary_link(
    paper_id: str,
    row: dict[str, Any],
    source: dict[str, Any],
    overrides: dict[str, dict[str, str]],
) -> dict[str, str | None]:
    extra = row.get("extra") or {}
    status = extra.get("arxiv_resolution_status")
    confidence = extra.get("arxiv_match_confidence")
    arxiv_id = extra.get("arxiv_id")
    if status == "matched" and confidence in {"high", "exact"} and arxiv_id:
        url = f"https://arxiv.org/abs/{arxiv_id}"
        return {"primary": url, "primaryKind": "arxiv", "arxiv": url}

    override = overrides.get(paper_id)
    if override:
        url = f"https://arxiv.org/abs/{override['arxiv_id']}"
        return {"primary": url, "primaryKind": "arxiv", "arxiv": url}

    openreview = (source.get("extra") or {}).get("paper_url")
    if not openreview:
        raise ValueError(f"Paper {paper_id} has neither an arXiv nor official paper URL")
    return {"primary": openreview, "primaryKind": "openreview", "arxiv": None}


def outcome_counts(
    paper_ids: list[str],
    manifest: dict[str, dict[str, Any]],
    *,
    main_track_only: bool = False,
) -> dict[str, Any]:
    counts = {"papers": 0, "oral": 0, "spotlight": 0, "poster": 0, "awards": 0}
    for paper_id in paper_ids:
        row = manifest[paper_id]
        extra = row.get("extra") or {}
        if main_track_only and track_name(extra) != "main":
            continue
        counts["papers"] += 1
        counts[human_tier(extra)] += 1
        counts["awards"] += int(bool(extra.get("is_award_paper")))
    counts["honored"] = counts["oral"] + counts["spotlight"]
    counts["honoredRate"] = counts["honored"] / counts["papers"]
    return counts


def build_human_comparison(
    manifest: dict[str, dict[str, Any]],
    finalists: list[dict[str, Any]],
    tournament_seed: dict[str, Any],
    ranked: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    main_track_ids = [
        paper_id
        for paper_id, row in manifest.items()
        if track_name(row.get("extra") or {}) == "main"
    ]
    finalist_ids = [str(row["paper_id"]) for row in finalists]
    seed_rows = tournament_seed["ranked_papers"]
    swiss_ids = [
        str(row["paper_id"])
        for row in seed_rows
        if row.get("tournament_pool_reasons")
    ]
    if len(swiss_ids) != tournament_seed["tournament_pool_count"] or len(swiss_ids) != 172:
        raise ValueError("Expected the frozen 172-paper Swiss union")

    ranked_ids = [str(row["paper_id"]) for row in ranked]
    stage_specs = [
        ("main_track", "Main track", main_track_ids, True),
        ("finalists", "AI finalists", finalist_ids, False),
        ("swiss_pool", "Swiss pool", swiss_ids, False),
        ("all_pairs_playoff", "All-pairs playoff", ranked_ids[:60], False),
        ("final_top_20", "AI top 20", ranked_ids[:20], False),
        ("final_top_10", "AI top 10", ranked_ids[:10], False),
    ]
    stages = []
    for key, label, paper_ids, main_track_only in stage_specs:
        stages.append(
            {
                "key": key,
                "label": label,
                **outcome_counts(
                    paper_ids,
                    manifest,
                    main_track_only=main_track_only,
                ),
            }
        )

    expected = {
        "finalists": summary["pipeline_sets"][0],
        "swiss_pool": summary["pipeline_sets"][1],
        "all_pairs_playoff": summary["pipeline_sets"][2],
        "final_top_20": summary["pipeline_sets"][3],
        "final_top_10": summary["pipeline_sets"][4],
    }
    for stage in stages[1:]:
        reference = expected[stage["key"]]
        if stage["papers"] != reference["papers"] or stage["honored"] != reference["honored"]:
            raise ValueError(f"Human-outcome stage mismatch for {stage['key']}")
        if stage["awards"] != reference["award_papers"]:
            raise ValueError(f"Award-paper stage mismatch for {stage['key']}")

    playoff_outcomes = []
    for row in ranked[:60]:
        paper_id = str(row["paper_id"])
        source = manifest[paper_id]
        extra = source.get("extra") or {}
        playoff_outcomes.append(
            {
                "rank": row["rank"],
                "paperId": paper_id,
                "title": row["title"],
                "tier": human_tier(extra),
                "actualAward": bool(extra.get("is_award_paper")),
            }
        )

    award_papers = []
    swiss_id_set = set(swiss_ids)
    finalist_id_set = set(finalist_ids)
    rank_by_id = {paper_id: index for index, paper_id in enumerate(ranked_ids, 1)}
    for paper_id in main_track_ids:
        source = manifest[paper_id]
        extra = source.get("extra") or {}
        if not extra.get("is_award_paper"):
            continue
        rank = rank_by_id.get(paper_id)
        if rank is not None and rank <= 10:
            highest_stage = "top_10"
        elif rank is not None and rank <= 60:
            highest_stage = "playoff"
        elif paper_id in swiss_id_set:
            highest_stage = "swiss_pool"
        elif paper_id in finalist_id_set:
            highest_stage = "finalists"
        else:
            highest_stage = "main_track"
        award_papers.append(
            {
                "paperId": paper_id,
                "title": source["title"],
                "highestStage": highest_stage,
                "aiRank": rank,
            }
        )

    if len(award_papers) != 7:
        raise ValueError(f"Expected seven main-track award papers, found {len(award_papers)}")
    return {
        "stages": stages,
        "playoffOutcomes": playoff_outcomes,
        "officialAwardPapers": award_papers,
    }


FULL_COVERAGE_SIGNAL = "icml_2026_pass1_cheap_ensemble_signal.jsonl"

CLASS_LABELS = {
    "core_ml_algorithm": "Core ML algorithm",
    "theory": "Theory",
    "safety_governance_eval": "Safety / governance / eval",
    "scientific_modeling_tool": "Scientific modeling tool",
    "benchmark_dataset": "Benchmark / dataset",
    "infrastructure_systems": "Infrastructure / systems",
    "analysis_position": "Analysis / position",
    "application_method": "Application method",
    "other": "Other",
}


def class_mix(paper_ids: list[str], class_by_paper: dict[str, str | None]) -> dict[str, Any]:
    counts: dict[str, int] = {key: 0 for key in CLASS_LABELS}
    for paper_id in paper_ids:
        counts[class_by_paper.get(paper_id) or "other"] += 1
    total = len(paper_ids)
    if total == 0:
        raise ValueError("cannot summarize an empty paper set")
    return {
        "n": total,
        "counts": counts,
        "shares": {key: value / total for key, value in counts.items()},
    }


def build_preference_comparison(
    manifest: dict[str, dict[str, Any]],
    *,
    scores_dir: Path,
) -> dict[str, Any] | None:
    """Contribution-class mix of the human-honored set against a same-size AI top set.

    This is the only symmetric comparison of the two preferences: ICML never ranks
    its papers, so the AI cut is matched to the size of the human-honored set rather
    than to an invented "human top N".

    Returns None when the full-coverage AI signal is not present, so that building
    from the committed frozen data alone keeps working.
    """
    signal_path = scores_dir / FULL_COVERAGE_SIGNAL
    if not signal_path.exists():
        return None

    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from icml_ai_ac.analysis.human_comparison import (
            ai_position,
            contribution_class_of,
            load_ai_scores,
        )
    except ImportError:
        return None

    ai_rows = load_ai_scores(signal_path)
    main_track = [
        paper_id
        for paper_id, row in manifest.items()
        if track_name(row.get("extra") or {}) == "main"
    ]

    class_by_paper: dict[str, str | None] = {}
    score_by_paper: dict[str, float] = {}
    for paper_id in main_track:
        ai_row = ai_rows.get(paper_id)
        if ai_row is None:
            continue
        score, _reported, _signal = ai_position(ai_row)
        score_by_paper[paper_id] = score
        scores = ai_row.get("scores") if isinstance(ai_row.get("scores"), dict) else {}
        class_by_paper[paper_id] = contribution_class_of(scores)

    scored = [paper_id for paper_id in main_track if paper_id in score_by_paper]
    if len(scored) < len(main_track):
        raise ValueError(
            f"full-coverage signal misses {len(main_track) - len(scored)} main-track papers; "
            "refusing to publish a partial preference comparison"
        )
    unclassified = sum(1 for paper_id in scored if not class_by_paper.get(paper_id))
    if unclassified > len(scored) // 20:
        raise ValueError(
            f"{unclassified} of {len(scored)} main-track papers have no contribution class"
        )

    honored = [
        paper_id
        for paper_id in scored
        if human_tier(manifest[paper_id].get("extra") or {}) in ("oral", "spotlight")
    ]
    matched_n = len(honored)
    ai_top = sorted(scored, key=lambda paper_id: (-score_by_paper[paper_id], paper_id))[:matched_n]

    return {
        "matchedN": matched_n,
        "classLabels": CLASS_LABELS,
        "sets": [
            {"key": "corpus", "label": "All main-track papers", **class_mix(scored, class_by_paper)},
            {"key": "human", "label": "ICML orals and spotlights", **class_mix(honored, class_by_paper)},
            {"key": "ai", "label": f"AI's top {matched_n}", **class_mix(ai_top, class_by_paper)},
        ],
    }


DIVERGENCE_MAIN_TRACK = "icml_2026_human_divergence_main_track_tier.json"
STAGE02_AUDIT = "icml_2026_full_launch/stage02_final_audit.json"


def score_histogram(values: list[float]) -> list[list[float]]:
    """Reviewer means sit on a coarse discrete grid, so an exact tally beats binning."""
    tally: dict[float, int] = {}
    for value in values:
        key = round(float(value), 4)
        tally[key] = tally.get(key, 0) + 1
    return [[key, tally[key]] for key in sorted(tally)]


# Labels from the ICML 2026 reviewer form's overall recommendation field:
# https://icml.cc/Conferences/2026/ReviewerInstructions
REVIEWER_SCALE = [
    (1, "Strong reject"),
    (2, "Reject"),
    (3, "Weak reject"),
    (4, "Weak accept"),
    (5, "Accept"),
    (6, "Strong accept"),
]


def build_reviewer_scale(*, metadata_dir: Path) -> dict[str, Any] | None:
    """How often each point of the official scale was used on accepted main-track papers.

    Readers meet mean reviewer scores (3.75, 5.25) before any other context, so the
    site shows the scale's labels next to how individual reviews actually fell on it.
    """
    manifest_path = metadata_dir / "icml_2026_scoring_manifest_main_track.jsonl"
    if not manifest_path.exists():
        return None
    counts = {score: 0 for score, _ in REVIEWER_SCALE}
    papers = 0
    for row in read_jsonl(manifest_path):
        values = ((row.get("extra") or {}).get("openreview_scores") or {}).get("overall_values") or []
        if values:
            papers += 1
        for value in values:
            if float(value) not in counts:
                raise ValueError(f"Reviewer score {value} is off the 1-6 scale")
            counts[int(value)] += 1
    reviews = sum(counts.values())
    return {
        "papers": papers,
        "reviews": reviews,
        "levels": [
            {"score": score, "label": label, "reviews": counts[score], "share": counts[score] / reviews}
            for score, label in REVIEWER_SCALE
        ],
    }


def reviewer_score_vs_rank(
    *, scores_dir: Path, metadata_dir: Path, min_papers: int = 25
) -> list[dict[str, Any]]:
    """Average position in the AI ranking at each reviewer-score level.

    This is the rank correlation shown as a curve: if the two orderings were
    unrelated every level would sit at the 50th percentile.
    """
    ranks: dict[str, float] = {}
    for row in read_jsonl(scores_dir / FULL_COVERAGE_SIGNAL):
        paper_id = str(row.get("paper_id") or "")
        aggregate_rank = row.get("aggregate_rank")
        if paper_id and aggregate_rank is not None:
            ranks[paper_id] = float(aggregate_rank)

    reviewer: dict[str, float] = {}
    for row in read_jsonl(metadata_dir / "icml_2026_scoring_manifest_main_track.jsonl"):
        scores = (row.get("extra") or {}).get("openreview_scores") or {}
        value = scores.get("overall_mean")
        if value is not None:
            reviewer[str(row["paper_id"])] = float(value)

    common = sorted(set(ranks) & set(reviewer))
    if len(common) < 1000:
        raise ValueError(f"reviewer-vs-rank join covered only {len(common)} papers")
    ordered = sorted(common, key=lambda paper_id: ranks[paper_id])
    percentile = {
        paper_id: 100.0 * (1 - index / (len(ordered) - 1))
        for index, paper_id in enumerate(ordered)
    }

    grouped: dict[float, list[float]] = {}
    for paper_id in common:
        grouped.setdefault(round(reviewer[paper_id], 4), []).append(percentile[paper_id])
    return [
        {
            "reviewerScore": score,
            "papers": len(values),
            "meanAiPercentile": sum(values) / len(values),
        }
        for score, values in sorted(grouped.items())
        if len(values) >= min_papers
    ]


def stated_reasons(
    paper_ids: set[str], *, model_runs_dir: Path, field: str
) -> dict[str, dict[str, Any]]:
    """The shortest of the eight judgments' sentences for each paper, plus how many agreed.

    Shortest keeps the table readable; the count is reported alongside so the quote
    is never mistaken for the only model that said it.
    """
    run_dir = model_runs_dir / PASS1_RUN_DIR
    if not run_dir.is_dir():
        return {}
    gathered: dict[str, list[str]] = {paper_id: [] for paper_id in paper_ids}
    seen: dict[str, set[str]] = {paper_id: set() for paper_id in paper_ids}
    for path in sorted(run_dir.glob("[0-9][0-9]_*.jsonl")):
        label = MODEL_LABELS.get(path.stem)
        if label is None:
            raise ValueError(f"Unrecognized pass-1 model run: {path.stem}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                paper_id = str(row.get("paper_id") or "")
                if paper_id not in gathered:
                    continue
                text = ((row.get("scores") or {}).get("calibration") or {}).get(field) or ""
                if text.strip():
                    gathered[paper_id].append(" ".join(text.split()))
                    seen[paper_id].add(label)
    return {
        paper_id: {"reason": min(texts, key=len), "models": len(seen[paper_id])}
        for paper_id, texts in gathered.items()
        if texts
    }


def divergence_cases(
    divergence: dict[str, Any], *, model_runs_dir: Path, per_direction: int = 3
) -> dict[str, Any]:
    """The most extreme disagreement in each direction, on one shared scale.

    Both pools are pre-sorted by residual, so taking the head is the rule the
    gallery itself uses rather than a hand-picked selection. Each row carries the
    model's own argument on the side that drove it away from the human verdict:
    for a buried paper why it did not rank higher, for an elevated one why it did
    not rank lower.
    """

    def heads(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return pool[:per_direction]

    gems, blind = (
        heads(divergence["overlooked_gems"]["case_pool"]),
        heads(divergence["blind_spots"]["case_pool"]),
    )
    reasons = {
        **stated_reasons(
            {str(case["paper_id"]) for case in gems},
            model_runs_dir=model_runs_dir,
            field="why_not_lower",
        ),
        **stated_reasons(
            {str(case["paper_id"]) for case in blind},
            model_runs_dir=model_runs_dir,
            field="why_not_higher",
        ),
    }

    def rows(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "paperId": str(case["paper_id"]),
                "title": case["title"],
                "tier": case["human"]["tier"],
                "reviewerScore": case["human"].get("reviewer_overall"),
                "corpusRank": case["ai"]["rank"],
                "contributionClass": case.get("contribution_class"),
                **(
                    {"statedReason": reasons[str(case["paper_id"])]["reason"]}
                    if str(case["paper_id"]) in reasons
                    else {}
                ),
            }
            for case in pool
        ]

    return {
        "corpusSize": divergence["counts"]["papers"],
        "gems": rows(gems),
        "gemPool": divergence["overlooked_gems"]["total_matching"],
        "blindSpots": rows(blind),
        "blindSpotPool": divergence["blind_spots"]["total_matching"],
    }


def build_preference_evidence(
    *,
    evals_dir: Path,
    metadata_dir: Path,
    model_runs_dir: Path,
    scores_dir: Path,
) -> dict[str, Any] | None:
    """Corpus-scale evidence that the AI and human orderings encode different preferences.

    Returns None when the analysis bundle is absent so a build from committed data
    alone still succeeds.
    """
    divergence_path = evals_dir / DIVERGENCE_MAIN_TRACK
    audit_path = model_runs_dir / STAGE02_AUDIT
    manifest_path = metadata_dir / "icml_2026_scoring_manifest_main_track.jsonl"
    if not (divergence_path.exists() and audit_path.exists() and manifest_path.exists()):
        return None

    divergence = json.loads(divergence_path.read_text(encoding="utf-8"))
    crosstab_block = divergence["tier_ai_decile_crosstab"]
    table = crosstab_block["table"]
    crosstab = {
        tier: {"inTopDecile": table[tier]["ai_top"], "total": table[tier]["ai_top"] + table[tier]["ai_rest"]}
        for tier in ("oral", "spotlight", "poster")
    }

    def reviewer_scores(pool: list[dict[str, Any]]) -> list[float]:
        values = [(row.get("human") or {}).get("reviewer_overall") for row in pool]
        return [float(value) for value in values if value is not None]

    gems = reviewer_scores(divergence["overlooked_gems"]["case_pool"])
    blind = reviewer_scores(divergence["blind_spots"]["case_pool"])

    corpus: list[float] = []
    for row in read_jsonl(manifest_path):
        scores = (row.get("extra") or {}).get("openreview_scores") or {}
        value = scores.get("overall_mean")
        if value is not None:
            corpus.append(float(value))

    if not (corpus and gems and blind):
        raise ValueError("preference evidence requires corpus, gem, and blind-spot reviewer scores")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    top_k = audit["top_k_overlap"]["50"]

    mean = lambda values: sum(values) / len(values)
    return {
        "crosstab": {"topDecilePercentile": crosstab_block["ai_top_threshold_percentile"], "tiers": crosstab},
        "reviewerVsRank": reviewer_score_vs_rank(scores_dir=scores_dir, metadata_dir=metadata_dir),
        "divergenceCases": divergence_cases(divergence, model_runs_dir=model_runs_dir),
        "stability": {
            "topK": 50,
            "overlap": top_k["count"],
            "meanAbsoluteRankShift": audit["mean_absolute_rank_shift"],
            "addedToShortlist": audit["union_expansion_count"],
            "judges": {
                name.split("/")[-1]: {
                    "earlierSlotWinRate": block["earlier_win_rate_excluding_ties"],
                    "slotEffectRange": block["slot_effect_range"],
                }
                for name, block in audit["models"].items()
            },
        },
    }


PASS1_RUN_DIR = "icml_2026_pass1_cheap_ensemble"
BOOTSTRAP_SEED = 20260730
BOOTSTRAP_SAMPLES = 2000

MODEL_LABELS = {
    "00_nvidia_nemotron-3-ultra-550b-a55b": "Nemotron 3 Ultra",
    "01_google_gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
    "02_openai_gpt-5.6-luna": "GPT-5.6 Luna",
    "03_x-ai_grok-4.3": "Grok 4.3",
}

# The six axes the frontier panel scored on every finalist card, in the order the
# section presents them.
FRONTIER_AXES = [
    ("ml_field_impact", "Predicted ML field impact"),
    ("broad_scientific_impact", "Broad scientific impact"),
    ("evidence_confidence", "Evidence confidence"),
    ("technical_soundness", "Technical soundness"),
    ("novelty", "Novelty"),
    ("visual_evidence_importance", "Quality of figures and tables"),
]


def rank_values(values: list[float]) -> list[float]:
    """Average ranks, so tied scores do not create a spurious ordering."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        average = (start + stop) / 2 + 1
        for position in range(start, stop + 1):
            ranks[order[position]] = average
        start = stop + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("spearman needs paired inputs")
    xs, ys = rank_values(xs), rank_values(ys)
    count = len(xs)
    mean_x, mean_y = sum(xs) / count, sum(ys) / count
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    spread = (
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    ) ** 0.5
    return numerator / spread if spread else 0.0


def bootstrap_interval(
    keys: list[str],
    statistic: Any,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[float]:
    """Percentile interval over paper resamples, seeded so the build is reproducible."""
    import random

    rng = random.Random(seed)
    size = len(keys)
    draws = sorted(
        statistic([keys[rng.randrange(size)] for _ in range(size)]) for _ in range(samples)
    )
    return [draws[int(0.025 * samples)], draws[int(0.975 * samples)]]


def build_axis_weights(*, scores_dir: Path, metadata_dir: Path) -> dict[str, Any] | None:
    """What each panel axis drove: the AI's own ordering, and the reviewer score.

    Both correlations run over the same finalists, so the only thing that changes
    between the two columns is which verdict the axis is measured against.
    """
    card_paths = sorted(scores_dir.glob("icml_2026_frontier_cards_*_final.jsonl"))
    manifest_path = metadata_dir / "icml_2026_scoring_manifest_main_track.jsonl"
    if not card_paths or not manifest_path.exists():
        return None

    reviewer: dict[str, float] = {}
    for row in read_jsonl(manifest_path):
        value = ((row.get("extra") or {}).get("openreview_scores") or {}).get("overall_mean")
        if value is not None:
            reviewer[str(row["paper_id"])] = float(value)

    wanted = [key for key, _ in FRONTIER_AXES] + ["overall_gold_priority"]
    collected: dict[str, dict[str, list[float]]] = {}
    for path in card_paths:
        for row in read_jsonl(path):
            paper_id = str(row["paper_id"])
            card_scores = (row.get("card") or {}).get("scores") or {}
            for axis in wanted:
                value = card_scores.get(f"{axis}_score")
                if isinstance(value, (int, float)):
                    collected.setdefault(paper_id, {}).setdefault(axis, []).append(float(value))

    averaged = {
        paper_id: {axis: sum(values) / len(values) for axis, values in per_axis.items()}
        for paper_id, per_axis in collected.items()
    }
    papers = sorted(
        paper_id
        for paper_id, per_axis in averaged.items()
        if paper_id in reviewer and set(wanted) <= set(per_axis)
    )
    if len(papers) < 200:
        raise ValueError(f"axis weights joined only {len(papers)} finalists to reviewer scores")

    def against_ai(pool: list[str], axis: str) -> float:
        return spearman(
            [averaged[p][axis] for p in pool],
            [averaged[p]["overall_gold_priority"] for p in pool],
        )

    def against_human(pool: list[str], axis: str) -> float:
        return spearman([averaged[p][axis] for p in pool], [reviewer[p] for p in pool])

    axes = [
        {
            "key": key,
            "label": label,
            "ai": against_ai(papers, key),
            "aiInterval": bootstrap_interval(papers, lambda pool, k=key: against_ai(pool, k)),
            "human": against_human(papers, key),
            "humanInterval": bootstrap_interval(
                papers, lambda pool, k=key: against_human(pool, k)
            ),
        }
        for key, label in FRONTIER_AXES
    ]

    # The headline contrast: field impact minus novelty, measured against each verdict.
    def gap(pool: list[str], scorer: Any) -> float:
        return scorer(pool, "ml_field_impact") - scorer(pool, "novelty")

    return {
        "finalists": len(papers),
        "axes": axes,
        "contrast": {
            "ai": gap(papers, against_ai),
            "aiInterval": bootstrap_interval(papers, lambda pool: gap(pool, against_ai)),
            "human": gap(papers, against_human),
            "humanInterval": bootstrap_interval(papers, lambda pool: gap(pool, against_human)),
        },
    }


# Recurring lines of argument in the cheap stage's written rationales. Rewards are
# matched against the case a model makes for a paper, concerns against the case it
# makes about it. These match language, not adjudicated meaning.
REASON_AXES: list[tuple[str, str, str, str]] = [
    (
        "evidence",
        "reward",
        "Its evidence is strong",
        r"extensive (?:experiment|evaluation|ablation)|thorough|comprehensive (?:evaluation|experiment)"
        r"|strong (?:empirical|experimental)|rigorous|multiple (?:benchmark|dataset)|reproducib"
        r"|state[- ]of[- ]the[- ]art|significant (?:gain|improv)",
    ),
    (
        "reusable",
        "reward",
        "It gives a mechanism others can reuse",
        r"general(?:iz\w+|ity)?\b|broadly applicable|across (?:domains|tasks|architectures|settings|modalities)"
        r"|reusable|drop[- ]in|plug[- ]in|any (?:model|architecture)|model[- ]agnostic|framework",
    ),
    (
        "theory",
        "reward",
        "It is backed by theory or guarantees",
        r"theoretical (?:guarantee|analysis|foundation|result)|prov\w+|formal (?:guarantee|analysis)"
        r"|bound\w*|convergence|optimality|principled",
    ),
    (
        "complexity",
        "concern",
        "It is complex or hard to adopt",
        r"complex|many (?:moving parts|components)|hard to (?:tune|implement|adopt|reproduce)"
        r"|hyperparameter|brittle|sensitiv\w+ to|tailored|engineering effort",
    ),
    (
        "incremental",
        "concern",
        "Its gains are small or incremental",
        r"incremental|marginal|modest (?:gain|improv)|small (?:gain|improv|margin)|limited novelty"
        r"|not novel|straightforward (?:extension|application)|existing (?:techniques|components|methods)",
    ),
    (
        "narrow",
        "concern",
        "It may not generalize beyond its setting",
        # "tied to specific drone datasets" is as common as "domain-specific", so the
        # noun may be plural and may carry a qualifier in front of it.
        r"generaliz|transfer\w*|narrow|domain[- ]specific|specialized"
        r"|specific (?:[\w-]+ )?(?:dataset|domain|task|benchmark|setting|cohort|corpora|corpus)s?"
        r"|beyond (?:the|this|its)|limited (?:applicab|scope)",
    ),
]

REWARD_FIELDS = (
    ("calibration", "why_not_lower"),
    ("ranking_signals", "top_paper_case"),
    ("ranking_signals", "broad_scientific_impact_argument"),
    ("ranking_signals", "why_not_just_incremental_or_domain_specific"),
)
CONCERN_FIELDS = (
    ("calibration", "why_not_higher"),
    ("ranking_signals", "dealbreaker_risks"),
)


def kendall_tau(xs: list[float], ys: list[float]) -> float | None:
    """Tau-a over a handful of items, skipping pairs either side calls a tie."""
    concordant = discordant = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            left, right = xs[i] - xs[j], ys[i] - ys[j]
            if left == 0 or right == 0:
                continue
            if left * right > 0:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else None


def build_within_batch_agreement(
    batches: dict[tuple[Any, Any], dict[str, dict[str, float]]],
    *,
    reviewer: dict[str, float],
    models: int,
    minimum_papers: int = 4,
) -> dict[str, Any]:
    """Agreement measured inside one batch, which holds the context identical.

    Every model saw the same partition into batches of eight, and the prompt asked
    each to rank a paper against the others in its batch. Comparing two models
    across the whole corpus therefore credits them for a batch layout they shared.
    Comparing them inside a single batch removes that, and the reviewer score can
    be scored on exactly the same eight papers, so both sides face one context.
    """
    model_to_model: list[float] = []
    model_to_human: list[float] = []
    used = 0
    for block in batches.values():
        if len(block) < models:
            continue
        names = sorted(block)
        papers = [
            paper
            for paper in block[names[0]]
            if paper in reviewer and all(paper in block[name] for name in names)
        ]
        if len(papers) < minimum_papers:
            continue
        used += 1
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                value = kendall_tau(
                    [block[left][p] for p in papers], [block[right][p] for p in papers]
                )
                if value is not None:
                    model_to_model.append(value)
            value = kendall_tau(
                [block[left][p] for p in papers], [reviewer[p] for p in papers]
            )
            if value is not None:
                model_to_human.append(value)
    if not (model_to_model and model_to_human):
        raise ValueError("within-batch agreement found no comparable batches")
    return {
        "batches": used,
        "modelToModel": {
            "mean": sum(model_to_model) / len(model_to_model),
            "comparisons": len(model_to_model),
        },
        "modelToHuman": {
            "mean": sum(model_to_human) / len(model_to_human),
            "comparisons": len(model_to_human),
        },
    }


def read_pass1_judgments(
    *, model_runs_dir: Path, keep: set[str], batches: dict | None = None
) -> dict[str, dict[str, dict[str, Any]]]:
    """Per model, the priority it gave each paper and which arguments it reached for.

    Each model scored every paper twice under randomized batch orderings, so the
    priority is averaged and a flag is set when either pass raised that argument.
    """
    run_dir = model_runs_dir / PASS1_RUN_DIR
    patterns = {key: re.compile(pattern, re.I) for key, _, _, pattern in REASON_AXES}
    kinds = {key: kind for key, kind, _, _ in REASON_AXES}

    judgments: dict[str, dict[str, dict[str, Any]]] = {}
    for path in sorted(run_dir.glob("[0-9][0-9]_*.jsonl")):
        label = MODEL_LABELS.get(path.stem)
        if label is None:
            raise ValueError(f"Unrecognized pass-1 model run: {path.stem}")
        per_paper: dict[str, dict[str, Any]] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                paper_id = str(row.get("paper_id") or "")
                if paper_id not in keep:
                    continue
                blocks = row.get("scores") or {}
                priority = (blocks.get("scores") or {}).get("executive_ac_priority")
                if not isinstance(priority, (int, float)):
                    continue
                reward = " ".join(
                    (blocks.get(group) or {}).get(field) or "" for group, field in REWARD_FIELDS
                )
                concern = " ".join(
                    (blocks.get(group) or {}).get(field) or "" for group, field in CONCERN_FIELDS
                )
                entry = per_paper.setdefault(paper_id, {"priorities": [], "flags": set()})
                entry["priorities"].append(float(priority))
                if batches is not None:
                    key = (row.get("partition_index"), row.get("batch_index"))
                    batches.setdefault(key, {}).setdefault(label, {})[paper_id] = float(priority)
                for key, pattern in patterns.items():
                    if pattern.search(reward if kinds[key] == "reward" else concern):
                        entry["flags"].add(key)
        judgments[label] = {
            paper_id: {
                "priority": sum(entry["priorities"]) / len(entry["priorities"]),
                # Kept unaveraged so a model can be compared against its own second reading.
                "readings": entry["priorities"],
                "flags": entry["flags"],
            }
            for paper_id, entry in per_paper.items()
        }
    return judgments


def build_reason_axes(
    judgments: dict[str, dict[str, dict[str, Any]]],
    *,
    reviewer: dict[str, float],
    tiers: dict[str, str],
    ranks: dict[str, float],
) -> dict[str, Any]:
    """How much each line of argument moves the AI ranking, and the reviewer score."""
    papers = sorted(set(ranks) & set(reviewer) & set(tiers))
    if len(papers) < 1000:
        raise ValueError(f"reason axes joined only {len(papers)} papers")
    ordered = sorted(papers, key=lambda paper_id: ranks[paper_id])
    percentile = {
        paper_id: 100.0 * (1 - index / (len(ordered) - 1))
        for index, paper_id in enumerate(ordered)
    }

    def raising(paper_id: str, key: str) -> int:
        return sum(
            1
            for per_paper in judgments.values()
            if key in (per_paper.get(paper_id) or {}).get("flags", ())
        )

    axes = []
    for key, kind, label, _ in REASON_AXES:
        counts = [raising(paper_id, key) for paper_id in papers]
        axes.append(
            {
                "key": key,
                "kind": kind,
                "label": label,
                "coverage": sum(1 for count in counts if count) / len(counts),
                "ai": spearman(counts, [percentile[paper_id] for paper_id in papers]),
                "human": spearman(counts, [reviewer[paper_id] for paper_id in papers]),
            }
        )

    # The generality objection in detail, because it is the AI's stated test for impact.
    buckets = []
    for raised in range(len(judgments) + 1):
        pool = [paper_id for paper_id in papers if raising(paper_id, "narrow") == raised]
        if not pool:
            continue
        buckets.append(
            {
                "modelsRaising": raised,
                "papers": len(pool),
                "meanAiPercentile": sum(percentile[paper_id] for paper_id in pool) / len(pool),
                "honoredShare": sum(1 for p in pool if tiers[p] != "poster") / len(pool),
            }
        )
    majority = (len(judgments) // 2) + 1
    by_tier = {}
    for tier in ("oral", "spotlight", "poster"):
        pool = [paper_id for paper_id in papers if tiers[paper_id] == tier]
        by_tier[tier] = {
            "papers": len(pool),
            "share": (
                sum(1 for p in pool if raising(p, "narrow") >= majority) / len(pool)
                if pool
                else None
            ),
        }

    return {
        "papers": len(papers),
        "models": len(judgments),
        "axes": axes,
        "generality": {"majority": majority, "buckets": buckets, "byTier": by_tier},
    }


def build_model_agreement(
    judgments: dict[str, dict[str, dict[str, Any]]], *, reviewer: dict[str, float]
) -> dict[str, Any]:
    """Whether the models agree with each other more than any of them agrees with reviewers."""
    names = sorted(judgments)
    papers = sorted(set.intersection(*(set(per_paper) for per_paper in judgments.values())) & set(reviewer))
    if len(papers) < 1000:
        raise ValueError(f"model agreement joined only {len(papers)} papers")

    def priorities(model: str) -> list[float]:
        return [judgments[model][paper_id]["priority"] for paper_id in papers]

    pairs = [
        {"a": left, "b": right, "rho": spearman(priorities(left), priorities(right))}
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    ]
    against_human = [
        {"model": model, "rho": spearman(priorities(model), [reviewer[p] for p in papers])}
        for model in names
    ]
    return {
        "papers": len(papers),
        "models": names,
        "pairs": pairs,
        "human": against_human,
        "meanModelToModel": sum(pair["rho"] for pair in pairs) / len(pairs),
        "meanModelToHuman": sum(row["rho"] for row in against_human) / len(against_human),
        "weakestModelPair": min(pair["rho"] for pair in pairs),
        "strongestHumanPair": max(row["rho"] for row in against_human),
    }


def pairwise_agreement(scores: dict[str, dict[str, float]]) -> dict[str, Any] | None:
    """Mean, best and worst Spearman across every unordered pair of judges."""
    names = sorted(scores)
    if len(names) < 2:
        return None
    common = sorted(set.intersection(*(set(scores[name]) for name in names)))
    if not common:
        return None
    values = [
        spearman([scores[left][p] for p in common], [scores[right][p] for p in common])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    ]
    return {
        "papers": len(common),
        "judges": len(names),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def spread(values: list[float]) -> dict[str, Any]:
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "pairings": len(values),
    }


def build_judge_consistency(
    readings: dict[str, dict[str, list[float]]],
    *,
    reviewer: dict[str, float],
    scores_dir: Path,
) -> dict[str, Any]:
    """Agreement measured with a single reading on both sides of every comparison.

    Each cheap model read every paper twice, so a model against its own second
    reading is the natural ceiling: nothing can agree with a model more than the
    model agrees with itself. Reporting the finalist range alongside the corpus
    matters because inside that range even self-agreement collapses, which means
    no conclusion can be drawn from agreement measured there.
    """
    names = sorted(readings)
    repeated = [
        set(paper for paper, values in readings[name].items() if len(values) >= 2)
        for name in names
    ]
    corpus = sorted(set.intersection(*repeated) & set(reviewer))
    if len(corpus) < 1000:
        raise ValueError(f"judge consistency joined only {len(corpus)} papers")

    frontier: dict[str, dict[str, float]] = {}
    for path in sorted(scores_dir.glob("icml_2026_frontier_cards_*_final.jsonl")):
        label = path.name.split("cards_")[1].split("_final")[0]
        for row in read_jsonl(path):
            value = ((row.get("card") or {}).get("scores") or {}).get(
                "overall_gold_priority_score"
            )
            if isinstance(value, (int, float)):
                frontier.setdefault(label, {})[str(row["paper_id"])] = float(value)
    judged = set.intersection(*(set(block) for block in frontier.values())) if frontier else set()
    finalists = [paper for paper in corpus if paper in judged]

    def block(pool: list[str], key: str, label: str) -> dict[str, Any]:
        return {
            "key": key,
            "label": label,
            "papers": len(pool),
            "self": spread(
                [
                    spearman(
                        [readings[name][p][0] for p in pool],
                        [readings[name][p][1] for p in pool],
                    )
                    for name in names
                ]
            ),
            "cross": spread(
                [
                    spearman(
                        [readings[left][p][first] for p in pool],
                        [readings[right][p][second] for p in pool],
                    )
                    for index, left in enumerate(names)
                    for right in names[index + 1 :]
                    for first in (0, 1)
                    for second in (0, 1)
                ]
            ),
            "human": spread(
                [
                    spearman(
                        [readings[name][p][which] for p in pool],
                        [reviewer[p] for p in pool],
                    )
                    for name in names
                    for which in (0, 1)
                ]
            ),
        }

    ranges = [block(corpus, "corpus", "Every main-track paper")]
    if len(finalists) >= 100:
        ranges.append(block(finalists, "finalists", "Only the finalists"))

    result: dict[str, Any] = {"models": len(names), "ranges": ranges}
    if frontier and finalists:
        pool = [paper for paper in finalists if all(paper in block for block in frontier.values())]
        judges = sorted(frontier)
        result["frontier"] = {
            "papers": len(pool),
            "judges": len(judges),
            "cross": spread(
                [
                    spearman([frontier[a][p] for p in pool], [frontier[b][p] for p in pool])
                    for index, a in enumerate(judges)
                    for b in judges[index + 1 :]
                ]
            ),
            "human": spread(
                [
                    spearman([frontier[a][p] for p in pool], [reviewer[p] for p in pool])
                    for a in judges
                ]
            ),
        }
    return result


def build_judgment_basis(
    *, model_runs_dir: Path, metadata_dir: Path, scores_dir: Path, evals_dir: Path
) -> dict[str, Any] | None:
    """Section 04's evidence: what the AI weighed, and how widely that is shared.

    Returns None when the analysis bundle is absent so a build from committed data
    alone still succeeds.
    """
    manifest_path = metadata_dir / "icml_2026_scoring_manifest_main_track.jsonl"
    signal_path = scores_dir / FULL_COVERAGE_SIGNAL
    run_dir = model_runs_dir / PASS1_RUN_DIR
    if not (manifest_path.exists() and signal_path.exists() and run_dir.is_dir()):
        return None

    reviewer: dict[str, float] = {}
    tiers: dict[str, str] = {}
    for row in read_jsonl(manifest_path):
        paper_id = str(row["paper_id"])
        extra = row.get("extra") or {}
        tiers[paper_id] = human_tier(extra)
        value = (extra.get("openreview_scores") or {}).get("overall_mean")
        if value is not None:
            reviewer[paper_id] = float(value)

    ranks: dict[str, float] = {}
    for row in read_jsonl(signal_path):
        paper_id = str(row.get("paper_id") or "")
        aggregate_rank = row.get("aggregate_rank")
        if paper_id in tiers and aggregate_rank is not None:
            ranks[paper_id] = float(aggregate_rank)

    batches: dict = {}
    judgments = read_pass1_judgments(
        model_runs_dir=model_runs_dir, keep=set(tiers), batches=batches
    )
    if len(judgments) != len(MODEL_LABELS):
        raise ValueError(f"expected {len(MODEL_LABELS)} pass-1 model runs, found {len(judgments)}")

    return {
        "axisWeights": build_axis_weights(scores_dir=scores_dir, metadata_dir=metadata_dir),
        "reasonAxes": build_reason_axes(
            judgments, reviewer=reviewer, tiers=tiers, ranks=ranks
        ),
        "modelAgreement": build_model_agreement(judgments, reviewer=reviewer),
        "judgeConsistency": build_judge_consistency(
            {
                name: {paper: entry["readings"] for paper, entry in per_paper.items()}
                for name, per_paper in judgments.items()
            },
            reviewer=reviewer,
            scores_dir=scores_dir,
        ),
        "withinBatch": build_within_batch_agreement(
            batches, reviewer=reviewer, models=len(judgments)
        ),
    }


def build_method_diagram(tournament: dict[str, Any]) -> dict[str, Any]:
    schedule = tournament["tournament_summary"]["schedule"]
    swiss_pairs = sum(
        row["pair_count"] for row in schedule if row["stage"] == "swiss"
    )
    new_playoff_pairs = sum(
        row["pair_count"] for row in schedule if row["stage"] == "playoff_all_pairs"
    )
    complete_playoff_pairs = 60 * 59 // 2
    if (swiss_pairs, new_playoff_pairs, complete_playoff_pairs) != (860, 1587, 1770):
        raise ValueError("Unexpected production tournament schedule")
    return {
        "stages": [
            {
                "stage": "01",
                "count": 6617,
                "title": "Anonymize and label",
                "models": "Gemini 3.1 Flash Lite",
                "detail": "Author names removed; contribution type labeled",
                "tone": "source",
            },
            {
                "stage": "02",
                "count": 6617,
                "title": "First-pass ranking",
                "models": "Nemotron 3 Ultra · Gemini 3.5 Flash Lite · GPT-5.6 Luna · Grok 4.3",
                "detail": "Each paper ranked in 2 batches of 8, by each model",
                "tone": "cheap",
            },
            {
                "stage": "03",
                "count": 1442,
                "title": "Second-pass ranking",
                "models": "GPT-5.6 Terra + Claude Sonnet 5",
                "detail": "Same batch design, stronger models",
                "tone": "strong",
            },
            {
                "stage": "04",
                "count": 250,
                "title": "Full-paper judgments",
                "models": "GPT-5.6 Sol + Claude Fable 5 + Gemini 3.1 Pro",
                "detail": "750 written judgments, 3 per paper",
                "tone": "frontier",
            },
            {
                "stage": "05",
                "count": 172,
                "title": "Swiss-system rounds",
                "models": "GPT-5.6 Sol, pairwise",
                "detail": f"10 rounds · {swiss_pairs:,} pairs",
                "tone": "swiss",
            },
            {
                "stage": "06",
                "count": 60,
                "title": "Round robin",
                "models": "GPT-5.6 Sol, pairwise",
                "detail": f"Every pair compared · {complete_playoff_pairs:,} pairs",
                "tone": "playoff",
            },
        ],
        "swissPairs": swiss_pairs,
        "newPlayoffPairs": new_playoff_pairs,
        "completePlayoffPairs": complete_playoff_pairs,
    }


def build(
    *,
    data_root: Path | None = None,
    summary_path: Path | None = None,
    overrides_path: Path = ARXIV_OVERRIDES,
) -> dict[str, Any]:
    data_root = data_root or ROOT / "data"
    scores = data_root / "scores"
    metadata = data_root / "metadata"
    summary_path = summary_path or data_root / "evals/icml_2026_stage7_summary.json"
    tournament = json.loads(
        (scores / "icml_2026_frontier_tournament_sol56_union172_swiss10_playoff60.json").read_text(
            encoding="utf-8"
        )
    )
    tournament_seed = json.loads(
        (scores / "icml_2026_tournament_seed_union172.json").read_text(encoding="utf-8")
    )
    finalists = read_jsonl(scores / "icml_2026_finalists250.jsonl")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = {
        row["paper_id"]: row
        for row in read_jsonl(metadata / "icml_2026_scoring_manifest.jsonl")
    }
    arxiv_matches = {
        row["paper_id"]: row
        for row in read_jsonl(metadata / "icml_2026_ai_orals60_with_arxiv.jsonl")
    }
    arxiv_overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
    ranked = tournament["ranked_papers"]
    playoff = ranked[:60]

    if [row["rank"] for row in playoff] != list(range(1, 61)):
        raise ValueError("Expected contiguous final playoff ranks 1-60")
    if any(
        row.get("tournament_stats", {}).get("ranking_source_stage") != "playoff_all_pairs"
        for row in playoff
    ):
        raise ValueError("Every published AI oral must come from the dense all-pairs playoff")

    papers: list[dict[str, Any]] = []
    for row in playoff:
        paper_id = row["paper_id"]
        source = manifest[paper_id]
        arxiv_match = arxiv_matches.get(paper_id)
        if arxiv_match is None:
            raise ValueError(f"Missing arXiv resolution row for paper {paper_id}")
        if normalized_title(source["title"]) != normalized_title(row["title"]):
            raise ValueError(f"Title mismatch for paper {paper_id}")
        if normalized_title(source["title"]) != normalized_title(arxiv_match["title"]):
            raise ValueError(f"arXiv resolution title mismatch for paper {paper_id}")
        extra = source["extra"]
        review = extra.get("openreview_scores") or {}
        papers.append(
            {
                "rank": row["rank"],
                "paperId": paper_id,
                "title": row["title"],
                "primaryClass": row["primary_contribution_class"],
                "secondaryClasses": row.get("secondary_contribution_classes") or [],
                "whyRankedHere": row["why_ranked_here"],
                "bestCase": row["best_case_for_impact"],
                "mainRisk": row["main_risk"],
                "scores": {
                    "priority": row["overall_priority_score"],
                    "technical": row["technical_soundness_score"],
                    "mlImpact": row["ml_field_impact_score"],
                    "scienceImpact": row["broad_scientific_impact_score"],
                    "evidence": row["evidence_confidence_score"],
                },
                "human": {
                    "tier": human_tier(extra),
                    "reviewerMean": review.get("overall_mean"),
                    "reviewCount": review.get("review_count", 0),
                    "actualAward": bool(extra.get("is_award_paper")),
                    "track": track_name(extra),
                },
                "links": {
                    **primary_link(paper_id, arxiv_match, source, arxiv_overrides),
                    "icml": extra.get("icml_virtual_url"),
                    "openreview": extra.get("paper_url"),
                },
            }
        )

    arxiv_count = sum(paper["links"]["primaryKind"] == "arxiv" for paper in papers)
    if arxiv_count != 59:
        raise ValueError(f"Expected 59 verified arXiv links, found {arxiv_count}")

    return {
        "generatedAt": "2026-07-30",
        "title": "AI as Area Chair: What Science Would AI Send to ICML?",
        "awardRules": {
            "outstandingRanks": [1, 2],
            "honorableMentionRanks": [3, 4, 5, 6, 7],
            "oralRanks": [1, 60],
            "note": (
                "The mock program names ranks 1-2 Outstanding Papers, ranks 3-7 "
                "Honorable Mentions, and the complete 60-paper all-pairs playoff "
                "as AI Orals."
            ),
        },
        "experiment": {
            "scoredPapers": summary["populations"]["all_scored"],
            "mainTrackPapers": summary["populations"]["main_track"]["papers"],
            "finalists": 250,
            "swissPool": 172,
            "playoff": 60,
            "pairwiseComparisons": tournament["tournament_summary"]["pair_count"],
            "finalJudge": tournament["judge"],
            "humanOutcomesJoinedPostHoc": True,
            "arxivLinkedPapers": arxiv_count,
            "officialFallbackLinks": len(papers) - arxiv_count,
        },
        "method": build_method_diagram(tournament),
        "humanComparison": build_human_comparison(
            manifest,
            finalists,
            tournament_seed,
            ranked,
            summary,
        ),
        "results": summary,
        "papers": papers,
        **(
            {"reviewerScale": reviewer_scale}
            if (reviewer_scale := build_reviewer_scale(metadata_dir=metadata))
            else {}
        ),
        **(
            {"preferenceComparison": preference_comparison}
            if (preference_comparison := build_preference_comparison(manifest, scores_dir=scores))
            else {}
        ),
        **(
            {"preferenceEvidence": preference_evidence}
            if (
                preference_evidence := build_preference_evidence(
                    evals_dir=data_root / "evals",
                    metadata_dir=metadata,
                    model_runs_dir=data_root / "model_runs",
                    scores_dir=scores,
                )
            )
            else {}
        ),
        **(
            {"judgmentBasis": judgment_basis}
            if (
                judgment_basis := build_judgment_basis(
                    model_runs_dir=data_root / "model_runs",
                    metadata_dir=metadata,
                    scores_dir=scores,
                    evals_dir=data_root / "evals",
                )
            )
            else {}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--overrides", type=Path, default=ARXIV_OVERRIDES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    payload = json.dumps(
        build(
            data_root=args.data_root,
            summary_path=args.summary,
            overrides_path=args.overrides,
        ),
        ensure_ascii=False,
        indent=2,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(f"window.ICML_AI_AC_DATA = {payload};\n", encoding="utf-8")
    try:
        label = args.output.relative_to(ROOT)
    except ValueError:
        label = args.output
    print(f"Wrote {label}")


if __name__ == "__main__":
    main()
