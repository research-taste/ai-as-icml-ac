from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from icml_ai_ac.finalists import read_ranked_rows
from icml_ai_ac.position_sensitivity import estimate_slot_effects, pearson
from icml_ai_ac.storage import read_json, read_jsonl, write_json


SEMIFINAL_AUDIT_VERSION = "stage3_semifinal_audit_v1"
FORBIDDEN_STRUCTURED_KEYS = {
    "ac_comment",
    "acceptance_status",
    "award",
    "awards",
    "decision",
    "decision_label",
    "meta_review",
    "presentation_tier",
    "presentation_type",
    "review",
    "reviewer_ratings",
    "reviewer_scores",
    "reviews",
}
FORBIDDEN_PROMPT_MARKERS = tuple(
    f'"{key}":' for key in sorted(FORBIDDEN_STRUCTURED_KEYS)
)
LIVE_URL_PATTERN = re.compile(r"https?://|www\.", flags=re.IGNORECASE)
ALLOWED_BLINDED_EXTRA_KEYS = {
    "anonymization",
    "blind_manifest",
    "scoring_artifacts",
}


def audit_semifinal_rankings(
    *,
    ranking_paths: list[Path],
    labels: list[str],
    expected_models: list[str],
    shortlist_path: Path,
    manifest_path: Path,
    out: Path,
) -> dict[str, Any]:
    if len(ranking_paths) < 2:
        raise ValueError("at least two semifinal rankings are required")
    if len(labels) != len(ranking_paths):
        raise ValueError("supply one label per ranking")
    if len(expected_models) != len(ranking_paths):
        raise ValueError("supply one expected model per ranking")
    if len(set(labels)) != len(labels):
        raise ValueError("semifinal audit labels must be unique")

    shortlist_rows = list(read_jsonl(shortlist_path))
    shortlist_by_id = unique_rows_by_id(shortlist_rows, path=shortlist_path)
    expected_ids = set(shortlist_by_id)
    blocking_issues: list[str] = []
    warnings: list[str] = []

    manifest_report = audit_blinded_manifest(
        manifest_path=manifest_path,
        expected_ids=expected_ids,
    )
    blocking_issues.extend(manifest_report["blocking_issues"])

    shortlist_forbidden_keys = sorted(
        collect_forbidden_keys(shortlist_rows, forbidden=FORBIDDEN_STRUCTURED_KEYS)
    )
    if shortlist_forbidden_keys:
        blocking_issues.append(
            f"shortlist contains forbidden human-outcome keys: {shortlist_forbidden_keys}"
        )

    source_reports: dict[str, Any] = {}
    rows_by_label: dict[str, dict[str, dict[str, Any]]] = {}
    for label, ranking_path, expected_model in zip(
        labels,
        ranking_paths,
        expected_models,
        strict=True,
    ):
        source_report, source_rows = audit_ranking_source(
            label=label,
            ranking_path=ranking_path,
            expected_model=expected_model,
            expected_ids=expected_ids,
            shortlist_by_id=shortlist_by_id,
        )
        source_reports[label] = source_report
        rows_by_label[label] = source_rows
        blocking_issues.extend(
            f"{label}: {issue}" for issue in source_report["blocking_issues"]
        )
        warnings.extend(f"{label}: {warning}" for warning in source_report["warnings"])

    comparison = compare_sources(
        labels=labels,
        rows_by_label=rows_by_label,
        shortlist_by_id=shortlist_by_id,
    )
    sample = deterministic_sample(
        labels=labels,
        rows_by_label=rows_by_label,
        shortlist_by_id=shortlist_by_id,
    )
    status = "failed" if blocking_issues else ("pass_with_warnings" if warnings else "pass")
    report = {
        "status": status,
        "audit_version": SEMIFINAL_AUDIT_VERSION,
        "ranking_paths": [str(path) for path in ranking_paths],
        "labels": labels,
        "expected_models": expected_models,
        "shortlist_path": str(shortlist_path),
        "manifest_path": str(manifest_path),
        "paper_count": len(expected_ids),
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "blinded_manifest": manifest_report,
        "shortlist_forbidden_keys": shortlist_forbidden_keys,
        "sources": source_reports,
        "comparison": comparison,
        "deterministic_sample": sample,
    }
    write_json(out, report)
    return report


def audit_blinded_manifest(
    *,
    manifest_path: Path,
    expected_ids: set[str],
) -> dict[str, Any]:
    rows = list(read_jsonl(manifest_path))
    by_id = unique_rows_by_id(rows, path=manifest_path)
    blocking: list[str] = []
    missing = sorted(expected_ids - set(by_id))
    if missing:
        blocking.append(f"missing {len(missing)} shortlisted paper IDs")

    nonempty_authors = 0
    human_values = 0
    wrong_source = 0
    unexpected_extra_keys: Counter[str] = Counter()
    for paper_id in expected_ids & set(by_id):
        row = by_id[paper_id]
        if row.get("authors"):
            nonempty_authors += 1
        if any(
            row.get(key) is not None
            for key in (
                "decision_label",
                "forum_url",
                "pdf_url",
                "notes",
                "final_rank_scores",
                "pairwise_results",
                "scores_pass1",
                "scores_pass2",
            )
        ):
            human_values += 1
        if row.get("source") != "blinded_scoring":
            wrong_source += 1
        extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
        for key in set(extra) - ALLOWED_BLINDED_EXTRA_KEYS:
            unexpected_extra_keys[str(key)] += 1

    if nonempty_authors:
        blocking.append(f"{nonempty_authors} shortlisted records retain authors")
    if human_values:
        blocking.append(f"{human_values} shortlisted records retain outcome or URL values")
    if wrong_source:
        blocking.append(f"{wrong_source} shortlisted records are not marked blinded_scoring")
    if unexpected_extra_keys:
        blocking.append(
            f"unexpected blinded-manifest extra keys: {dict(unexpected_extra_keys)}"
        )
    return {
        "status": "pass" if not blocking else "failed",
        "manifest_rows": len(rows),
        "shortlisted_rows_checked": len(expected_ids & set(by_id)),
        "nonempty_author_rows": nonempty_authors,
        "rows_with_human_or_url_values": human_values,
        "wrong_source_rows": wrong_source,
        "unexpected_extra_keys": dict(unexpected_extra_keys),
        "blocking_issues": blocking,
    }


def audit_ranking_source(
    *,
    label: str,
    ranking_path: Path,
    expected_model: str,
    expected_ids: set[str],
    shortlist_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = read_json(ranking_path)
    rows = read_ranked_rows(ranking_path)
    rows_by_id = unique_rows_by_id(rows, path=ranking_path)
    blocking: list[str] = []
    warnings: list[str] = []

    if payload.get("status") != "ok":
        blocking.append(f"run status is {payload.get('status')!r}, not 'ok'")
    if payload.get("model") != expected_model:
        blocking.append(
            f"requested model {payload.get('model')!r} does not match {expected_model!r}"
        )
    served_models = sorted(str(value) for value in payload.get("served_models") or [])
    if served_models != [expected_model] or payload.get("served_model") != expected_model:
        blocking.append(
            f"served models {served_models!r}/{payload.get('served_model')!r} "
            f"do not match {expected_model!r}"
        )
    if payload.get("coverage_errors"):
        blocking.append(f"coverage errors present: {payload['coverage_errors']}")
    if set(rows_by_id) != expected_ids:
        missing = sorted(expected_ids - set(rows_by_id))
        extra = sorted(set(rows_by_id) - expected_ids)
        blocking.append(
            f"ranking coverage differs: missing={len(missing)} extra={len(extra)}"
        )
    ranks = sorted(int(row.get("rank") or 0) for row in rows)
    if ranks != list(range(1, len(rows) + 1)):
        blocking.append("global ranks are not unique and contiguous")

    batch_results = payload.get("batch_results")
    if not isinstance(batch_results, list):
        blocking.append("run is missing batch_results")
        batch_results = []
    expected_batch_count = int(payload.get("batch_count") or 0)
    if len(batch_results) != expected_batch_count:
        blocking.append(
            f"batch result count {len(batch_results)} != expected {expected_batch_count}"
        )

    batch_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    prompt_cache: dict[Path, list[str]] = {}
    artifact_missing = 0
    fingerprint_mismatches = 0
    model_mismatches = 0
    failed_batches = 0
    prompt_marker_hits: Counter[str] = Counter()
    live_url_prompt_count = 0
    recovery_batches: list[dict[str, Any]] = []
    for batch in batch_results:
        partition_index = int(batch.get("partition_index") or 0)
        batch_index = int(batch.get("batch_index") or 0)
        batch_key = (partition_index, batch_index)
        if batch_key in batch_by_key:
            blocking.append(f"duplicate batch key {batch_key}")
            continue
        batch_by_key[batch_key] = batch
        if batch.get("status") != "ok":
            failed_batches += 1
        if batch.get("served_model") != expected_model:
            model_mismatches += 1
        if batch.get("recovered_from_content_filter"):
            recovery_batches.append(
                {
                    "partition_index": partition_index,
                    "batch_index": batch_index,
                    "candidate_ids": [str(value) for value in batch.get("candidate_ids") or []],
                    "recovery_kind": batch.get("recovery_kind"),
                    "effective_prompt_path": batch.get("effective_prompt_path"),
                }
            )

        artifact_keys = ("prompt_path", "raw_response_path", "parsed_response_path")
        for key in artifact_keys:
            value = batch.get(key)
            if not value or not Path(str(value)).exists():
                artifact_missing += 1
        prompt_value = batch.get("effective_prompt_path") or batch.get("prompt_path")
        if not prompt_value:
            continue
        prompt_path = Path(str(prompt_value))
        if not prompt_path.exists():
            continue
        prompt = read_json(prompt_path)
        prompt_ids = [str(value) for value in prompt.get("candidate_ids") or []]
        prompt_cache[prompt_path] = prompt_ids
        if prompt_ids != [str(value) for value in batch.get("candidate_ids") or []]:
            blocking.append(f"{prompt_path}: candidate IDs differ from batch state")
        if not batch.get("recovered_from_content_filter"):
            if prompt.get("fingerprint") != batch.get("fingerprint"):
                fingerprint_mismatches += 1
        message_text = "\n".join(
            str(message.get("content") or "")
            for message in prompt.get("messages") or []
            if isinstance(message, dict)
        )
        for marker in FORBIDDEN_PROMPT_MARKERS:
            if marker in message_text:
                prompt_marker_hits[marker] += 1
        if LIVE_URL_PATTERN.search(message_text):
            live_url_prompt_count += 1

    if failed_batches:
        blocking.append(f"{failed_batches} batches are not status=ok")
    if artifact_missing:
        blocking.append(f"{artifact_missing} required batch artifacts are missing")
    if fingerprint_mismatches:
        blocking.append(f"{fingerprint_mismatches} prompt fingerprints do not match")
    if model_mismatches:
        blocking.append(f"{model_mismatches} batches used an unexpected served model")
    if prompt_marker_hits:
        blocking.append(f"forbidden structured prompt markers: {dict(prompt_marker_hits)}")
    if live_url_prompt_count:
        blocking.append(f"{live_url_prompt_count} prompts contain a live URL")
    if recovery_batches:
        warnings.append(f"{len(recovery_batches)} content-filter recovery batch(es)")

    judgment_rows: list[dict[str, Any]] = []
    judgment_coverage_errors = 0
    scale_by_batch: dict[tuple[int, int], str] = {}
    for paper_id, row in rows_by_id.items():
        sources = row.get("source_batch_judgments")
        if not isinstance(sources, list) or len(sources) != 2:
            judgment_coverage_errors += 1
            continue
        partitions = {int(source.get("partition_index") or 0) for source in sources}
        if partitions != {0, 1}:
            judgment_coverage_errors += 1
        scales = row.get("overall_priority_score_source_scales")
        if not isinstance(scales, list) or len(scales) != len(sources):
            judgment_coverage_errors += 1
            scales = ["unknown"] * len(sources)
        for source, source_scale in zip(sources, scales, strict=True):
            partition_index = int(source["partition_index"])
            batch_index = int(source["batch_index"])
            batch_key = (partition_index, batch_index)
            batch = batch_by_key.get(batch_key)
            if batch is None or paper_id not in {
                str(value) for value in batch.get("candidate_ids") or []
            }:
                judgment_coverage_errors += 1
                continue
            prior_scale = scale_by_batch.setdefault(batch_key, str(source_scale))
            if prior_scale != str(source_scale):
                judgment_coverage_errors += 1
            prompt_path = Path(str(source["prompt_path"]))
            candidate_ids = prompt_cache.get(prompt_path)
            if candidate_ids is None and prompt_path.exists():
                candidate_ids = [
                    str(value) for value in read_json(prompt_path).get("candidate_ids") or []
                ]
                prompt_cache[prompt_path] = candidate_ids
            if not candidate_ids or candidate_ids.count(paper_id) != 1:
                judgment_coverage_errors += 1
                continue
            slot = candidate_ids.index(paper_id)
            judgment_rows.append(
                {
                    "status": "ok",
                    "paper_id": paper_id,
                    "partition_index": partition_index,
                    "batch_index": batch_index,
                    "input_slot": slot,
                    "batch_local_priority": 1.0 - float(source["normalized_local_rank"]),
                }
            )
    if judgment_coverage_errors:
        blocking.append(f"{judgment_coverage_errors} judgment provenance errors")

    scale_counts = Counter(scale_by_batch.values())
    if len(scale_counts) > 1:
        warnings.append(f"multiple raw priority-score scales: {dict(scale_counts)}")
    aggregation = payload.get("ranking", {}).get("aggregation")
    if "0_100" in scale_counts and aggregation != "pass2_batch_rank_with_calibrated_priority_v2":
        blocking.append("0-100 priority batches are present without calibrated aggregation")

    position = (
        position_report(judgment_rows, global_rows_by_id=rows_by_id)
        if len(judgment_rows) == 2 * len(expected_ids)
        else {"status": "unavailable"}
    )
    score_fields = (
        "overall_priority_score",
        "broad_scientific_impact_score",
        "ml_field_impact_score",
        "technical_soundness_score",
        "mean_normalized_local_rank",
        "local_rank_range",
    )
    score_distributions = {
        field: distribution([numeric(row.get(field)) for row in rows])
        for field in score_fields
    }
    class_distribution = class_rank_distribution(
        rows_by_id=rows_by_id,
        shortlist_by_id=shortlist_by_id,
    )
    report = {
        "status": "failed" if blocking else ("pass_with_warnings" if warnings else "pass"),
        "ranking_path": str(ranking_path),
        "requested_model": payload.get("model"),
        "served_models": served_models,
        "provider": payload.get("provider"),
        "paper_count": len(rows),
        "batch_count": len(batch_results),
        "judgment_count": payload.get("judgment_count"),
        "expected_judgment_count": payload.get("expected_judgment_count"),
        "usage": payload.get("usage"),
        "aggregation": aggregation,
        "tie_breaking": payload.get("ranking", {}).get("tie_breaking"),
        "failed_batch_count": failed_batches,
        "missing_artifact_count": artifact_missing,
        "fingerprint_mismatch_count": fingerprint_mismatches,
        "model_mismatch_count": model_mismatches,
        "prompt_forbidden_marker_hits": dict(prompt_marker_hits),
        "prompts_with_live_urls": live_url_prompt_count,
        "recovery_batches": recovery_batches,
        "priority_score_batch_scales": dict(sorted(scale_counts.items())),
        "score_distributions": score_distributions,
        "advance_to_final_review_count": sum(
            1 for row in rows if row.get("advance_to_final_review") is True
        ),
        "position_sensitivity": position,
        "class_rank_distribution": class_distribution,
        "blocking_issues": blocking,
        "warnings": warnings,
    }
    return report, rows_by_id


def position_report(
    rows: list[dict[str, Any]],
    *,
    global_rows_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    effects, diagnostics = estimate_slot_effects(rows)
    by_slot: dict[int, list[float]] = {}
    by_paper: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_slot.setdefault(int(row["input_slot"]), []).append(
            float(row["batch_local_priority"])
        )
        by_paper.setdefault(str(row["paper_id"]), []).append(row)
    partition_zero = []
    partition_one = []
    for paper_rows in by_paper.values():
        ordered = sorted(paper_rows, key=lambda row: int(row["partition_index"]))
        partition_zero.append(float(ordered[0]["batch_local_priority"]))
        partition_one.append(float(ordered[1]["batch_local_priority"]))
    adjusted_priority = {
        paper_id: sum(
            float(row["batch_local_priority"]) - effects[int(row["input_slot"])]
            for row in paper_rows
        )
        / len(paper_rows)
        for paper_id, paper_rows in by_paper.items()
    }
    adjusted_order = sorted(
        adjusted_priority,
        key=lambda paper_id: (
            -adjusted_priority[paper_id],
            int(global_rows_by_id[paper_id]["rank"]),
            paper_id,
        ),
    )
    adjusted_rank = {
        paper_id: rank for rank, paper_id in enumerate(adjusted_order, start=1)
    }
    raw_ranks = [
        int(global_rows_by_id[paper_id]["rank"]) for paper_id in sorted(by_paper)
    ]
    adjusted_ranks = [adjusted_rank[paper_id] for paper_id in sorted(by_paper)]
    absolute_shifts = [
        abs(raw - adjusted)
        for raw, adjusted in zip(raw_ranks, adjusted_ranks, strict=True)
    ]
    top_k_overlap = {}
    for k in (50, 100, 250, 500):
        raw_top = {
            paper_id
            for paper_id, row in global_rows_by_id.items()
            if int(row["rank"]) <= k
        }
        adjusted_top = {
            paper_id for paper_id, rank in adjusted_rank.items() if rank <= k
        }
        overlap = len(raw_top & adjusted_top)
        top_k_overlap[str(k)] = {
            "count": overlap,
            "recall_at_k": round(overlap / k, 6),
        }
    return {
        "status": "ok",
        **diagnostics,
        "slot_effects": [round(value, 6) for value in effects],
        "slot_effect_range": round(max(effects) - min(effects), 6),
        "mean_priority_by_slot": {
            str(slot): round(sum(values) / len(values), 6)
            for slot, values in sorted(by_slot.items())
        },
        "partition_priority_correlation": round(
            pearson(partition_zero, partition_one),
            6,
        ),
        "adjusted_ranking_sensitivity": {
            "method": "within_paper_paired_slot_fixed_effect",
            "raw_adjusted_spearman": round(pearson(raw_ranks, adjusted_ranks), 6),
            "mean_absolute_rank_shift": round(
                sum(absolute_shifts) / len(absolute_shifts),
                3,
            ),
            "max_absolute_rank_shift": max(absolute_shifts),
            "top_k_overlap": top_k_overlap,
        },
    }


def compare_sources(
    *,
    labels: list[str],
    rows_by_label: dict[str, dict[str, dict[str, Any]]],
    shortlist_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if len(labels) != 2:
        return {"status": "not_implemented", "reason": "pairwise report expects two judges"}
    first_label, second_label = labels
    first = rows_by_label[first_label]
    second = rows_by_label[second_label]
    common_ids = sorted(set(first) & set(second))
    first_ranks = [int(first[paper_id]["rank"]) for paper_id in common_ids]
    second_ranks = [int(second[paper_id]["rank"]) for paper_id in common_ids]
    absolute_differences = [
        abs(first_rank - second_rank)
        for first_rank, second_rank in zip(first_ranks, second_ranks, strict=True)
    ]
    top_k_overlap = {}
    for k in (25, 50, 100, 250, 500, 1000):
        if k > len(common_ids):
            continue
        first_top = {
            paper_id for paper_id, row in first.items() if int(row["rank"]) <= k
        }
        second_top = {
            paper_id for paper_id, row in second.items() if int(row["rank"]) <= k
        }
        overlap = len(first_top & second_top)
        top_k_overlap[str(k)] = {
            "count": overlap,
            "recall_at_k": round(overlap / k, 6),
            "jaccard": round(overlap / len(first_top | second_top), 6),
        }
    class_agreement = sum(
        1
        for paper_id in common_ids
        if contribution_class(first[paper_id], shortlist_by_id[paper_id])
        == contribution_class(second[paper_id], shortlist_by_id[paper_id])
    )
    return {
        "status": "ok",
        "paper_count": len(common_ids),
        "spearman_rank_correlation": round(pearson(first_ranks, second_ranks), 6),
        "mean_absolute_rank_difference": round(
            sum(absolute_differences) / len(absolute_differences),
            3,
        ),
        "absolute_rank_difference": distribution(
            [float(value) for value in absolute_differences]
        ),
        "top_k_overlap": top_k_overlap,
        "contribution_class_agreement": {
            "count": class_agreement,
            "fraction": round(class_agreement / len(common_ids), 6),
        },
    }


def deterministic_sample(
    *,
    labels: list[str],
    rows_by_label: dict[str, dict[str, dict[str, Any]]],
    shortlist_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if len(labels) != 2:
        return {}
    first_label, second_label = labels
    first = rows_by_label[first_label]
    second = rows_by_label[second_label]
    rank_positions = (1, 10, 25, 50, 100, 250, 500, 750, 1000, 1250, len(first))
    first_by_rank = {int(row["rank"]): paper_id for paper_id, row in first.items()}
    selected_ids = {
        first_by_rank[rank] for rank in rank_positions if rank in first_by_rank
    }
    disagreement_ids = sorted(
        first,
        key=lambda paper_id: (
            -abs(int(first[paper_id]["rank"]) - int(second[paper_id]["rank"])),
            paper_id,
        ),
    )[:20]
    selected_ids.update(disagreement_ids)

    def sample_row(paper_id: str) -> dict[str, Any]:
        return {
            "paper_id": paper_id,
            "title": shortlist_by_id[paper_id].get("title"),
            "contribution_class": contribution_class(
                first[paper_id],
                shortlist_by_id[paper_id],
            ),
            "ranks": {
                first_label: int(first[paper_id]["rank"]),
                second_label: int(second[paper_id]["rank"]),
            },
            "rank_difference": abs(
                int(first[paper_id]["rank"]) - int(second[paper_id]["rank"])
            ),
            "why_ranked_here": {
                first_label: first[paper_id].get("why_ranked_here"),
                second_label: second[paper_id].get("why_ranked_here"),
            },
        }

    return {
        "cross_section": [
            sample_row(paper_id)
            for paper_id in sorted(
                selected_ids,
                key=lambda paper_id: (
                    min(
                        int(first[paper_id]["rank"]),
                        int(second[paper_id]["rank"]),
                    ),
                    paper_id,
                ),
            )
        ],
        "highest_rank_disagreement": [
            sample_row(paper_id) for paper_id in disagreement_ids
        ],
    }


def class_rank_distribution(
    *,
    rows_by_id: dict[str, dict[str, Any]],
    shortlist_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    shortlist_baseline = Counter(
        shortlist_class(shortlist_by_id[paper_id])
        for paper_id in rows_by_id
    )
    judge_baseline = Counter(
        str(rows_by_id[paper_id].get("primary_contribution_class") or "other")
        for paper_id in rows_by_id
    )
    report: dict[str, Any] = {
        "shortlist_class_baseline": dict(sorted(shortlist_baseline.items())),
        "judge_assigned_class_baseline": dict(sorted(judge_baseline.items())),
    }
    for k in (50, 100, 250, 500):
        if k > len(rows_by_id):
            continue
        top_ids = [
            paper_id
            for paper_id, row in sorted(
                rows_by_id.items(),
                key=lambda item: int(item[1]["rank"]),
            )[:k]
        ]
        counts = Counter(
            shortlist_class(shortlist_by_id[paper_id])
            for paper_id in top_ids
        )
        report[f"top_{k}"] = dict(sorted(counts.items()))
    return report


def shortlist_class(row: dict[str, Any]) -> str:
    return str(
        row.get("ensemble_contribution_class")
        or row.get("routing_contribution_class")
        or row.get("primary_contribution_class")
        or "other"
    )


def contribution_class(
    ranking_row: dict[str, Any],
    shortlist_row: dict[str, Any],
) -> str:
    return str(
        ranking_row.get("primary_contribution_class")
        or shortlist_row.get("ensemble_contribution_class")
        or shortlist_row.get("routing_contribution_class")
        or shortlist_row.get("primary_contribution_class")
        or "other"
    )


def unique_rows_by_id(
    rows: list[dict[str, Any]],
    *,
    path: Path,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        paper_id = str(row.get("paper_id") or "")
        if not paper_id:
            raise ValueError(f"{path}: row missing paper_id")
        if paper_id in result:
            raise ValueError(f"{path}: duplicate paper_id {paper_id}")
        result[paper_id] = row
    return result


def collect_forbidden_keys(
    value: Any,
    *,
    forbidden: set[str],
) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in forbidden:
                found.add(str(key))
            found.update(collect_forbidden_keys(child, forbidden=forbidden))
    elif isinstance(value, list):
        for child in value:
            found.update(collect_forbidden_keys(child, forbidden=forbidden))
    return found


def distribution(values: list[float]) -> dict[str, Any]:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return {"count": 0}
    return {
        "count": len(ordered),
        "min": round(ordered[0], 6),
        "q10": quantile(ordered, 0.10),
        "q25": quantile(ordered, 0.25),
        "median": quantile(ordered, 0.50),
        "mean": round(sum(ordered) / len(ordered), 6),
        "q75": quantile(ordered, 0.75),
        "q90": quantile(ordered, 0.90),
        "max": round(ordered[-1], 6),
    }


def quantile(ordered: list[float], probability: float) -> float:
    if len(ordered) == 1:
        return round(ordered[0], 6)
    position = probability * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return round(
        ordered[low] * (1.0 - fraction) + ordered[high] * fraction,
        6,
    )


def numeric(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan
