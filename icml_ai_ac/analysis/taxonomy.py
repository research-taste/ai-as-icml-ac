"""LLM taxonomy of AI-vs-human divergence reasons (QL-B).

Consumes the deterministic divergence gallery (QL-A) and codes *why* the
impact-forward AI diverged from human ICML outcomes, in two stages:

1. ``induce`` — sample the most divergent cases and propose a compact codebook of
   reason codes. A human then freezes/edits that codebook file.
2. ``classify`` — assign codes (multi-label) to every case against the frozen
   codebook, then aggregate the distribution across gems vs blind spots and
   contribution classes.

Guardrails against "laundering the AI's own reasons":

- Each case brief includes the *numeric* signals (executive_ac_priority vs
  conventional_acceptance_strength, reviewer soundness, residual, impact axes),
  and the coder is told to ground codes in that impact-vs-polish contrast, not to
  restate the AI's self-justification.
- Coding with a different model family than the one that wrote the rationale is
  supported (just point ``--provider/--model`` at another lab).
- ``--second-model`` re-codes the cases with an independent judge and reports a
  per-code Cohen's kappa so low-reliability codes are visible.

Run functions take a ``client`` (None for ``dry_run``) so the prompt-building,
validation, and aggregation are all unit-testable without network access.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from icml_ai_ac.analysis.human_comparison import inline
from icml_ai_ac.scoring.schema import parse_json_response
from icml_ai_ac.storage import read_json, write_json


TAXONOMY_INDUCE_PROMPT_VERSION = "divergence_taxonomy_induce_v1"
TAXONOMY_CLASSIFY_PROMPT_VERSION = "divergence_taxonomy_classify_v1"

RATIONALE_SNIPPETS = (
    "sweeping_impact_scenario",
    "reviewer_vs_executive_delta",
    "human_review_alignment",
    "why_not_higher",
    "why_not_lower",
    "dealbreaker_risks",
)


@dataclass(slots=True)
class TaxonomyConfig:
    provider: str
    model: str
    reasoning_effort: str | None
    temperature: float
    max_output_tokens: int
    seed: int | None
    dry_run: bool


# --- case assembly --------------------------------------------------------


def collect_cases(report: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for key, direction in (("overlooked_gems", "overlooked_gem"), ("blind_spots", "blind_spot")):
        section = report.get(key) if isinstance(report.get(key), dict) else {}
        pool = section.get("case_pool")
        if not isinstance(pool, list):
            pool = section.get("cases", []) if isinstance(section.get("cases"), list) else []
        for card in pool:
            if isinstance(card, dict) and card.get("paper_id"):
                cases.append({**card, "direction": direction})
    return cases


def balanced_sample(cases: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    if sample_size <= 0:
        return []
    gems = [case for case in cases if case["direction"] == "overlooked_gem"]
    blind = [case for case in cases if case["direction"] == "blind_spot"]
    first_count = min(len(gems), (sample_size + 1) // 2)
    second_count = min(len(blind), sample_size // 2)
    sampled = gems[:first_count] + blind[:second_count]
    if len(sampled) < sample_size:
        used = {case["paper_id"] for case in sampled}
        sampled.extend(
            case
            for case in cases
            if case["paper_id"] not in used
        )
    return sampled[:sample_size]


def case_brief(case: dict[str, Any]) -> str:
    human = case.get("human", {})
    ai = case.get("ai", {})
    axes = case.get("ai_axes", {})
    lines = [
        f"PAPER {case['paper_id']} | title={inline(case.get('title'), 240)}",
        f"  Direction={case['direction']} | class={case.get('contribution_class')}",
        f"  Abstract: {inline(case.get('abstract'), 900)}",
        f"  Human: decision={human.get('decision_status')} tier={human.get('tier')} award={human.get('is_award')} "
        f"reviewer_overall={human.get('reviewer_overall')} soundness={human.get('reviewer_soundness')}",
        f"  AI: signal={ai.get('ranking_signal')} signal_kind={ai.get('ranking_signal_kind')} "
        f"executive_priority={ai.get('executive_ac_priority')} percentile={ai.get('percentile')} "
        f"residual={case.get('divergence', {}).get('residual')}",
        "  AI axes: " + ", ".join(f"{key}={value}" for key, value in axes.items()),
    ]
    rationale = case.get("ai_rationale", {})
    for label in RATIONALE_SNIPPETS:
        value = rationale.get(label)
        if value:
            lines.append(f"  {label}: {inline(value, 240)}")
    return "\n".join(lines)


# --- stage 1: induce ------------------------------------------------------


def build_induce_messages(cases: list[dict[str, Any]], *, max_codes: int) -> list[dict[str, str]]:
    system = """You are analyzing where an impact-forward AI area chair disagreed with the human decisions of an ML conference.

Propose a compact codebook of reason codes describing observable patterns associated with the AI-human divergence. Ground each code in the paper abstract and measurable contrast (which AI axes are high or low relative to the human tier and reviewer scores). Do NOT simply restate the AI's self-justification, and do not make causal claims about private reviewer deliberation. Codes must be distinct and reusable.

Return only valid JSON."""
    user = f"""Here are divergent cases. Each lists the paper's contribution class, the human outcome, the AI's placement and axis scores, and snippets of the AI's stated reasons.

Propose at most {max_codes} reason codes.

Required JSON:
{{
  "codes": [
    {{
      "code": "short_snake_case_name",
      "definition": "one sentence, grounded in the axis/human contrast",
      "applies_when": "observable signature (e.g. executive_ac_priority high but conventional_acceptance_strength/soundness low)",
      "example_paper_ids": ["id"]
    }}
  ]
}}

Cases:
{chr(10).join(case_brief(case) for case in cases)}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate_codebook(
    payload: dict[str, Any],
    *,
    max_codes: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    codes = payload.get("codes")
    if not isinstance(codes, list) or not codes:
        return [], ["codes must be a non-empty list"]
    if max_codes is not None and len(codes) > max_codes:
        errors.append(f"codebook has {len(codes)} codes; maximum is {max_codes}")
    clean: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, code in enumerate(codes):
        if not isinstance(code, dict):
            errors.append(f"codes[{index}] must be an object")
            continue
        name = str(code.get("code") or "").strip()
        if not name:
            errors.append(f"codes[{index}] missing code name")
            continue
        if name in seen:
            errors.append(f"duplicate code: {name}")
            continue
        seen.add(name)
        clean.append(
            {
                "code": name,
                "definition": str(code.get("definition") or ""),
                "applies_when": str(code.get("applies_when") or ""),
                "example_paper_ids": [str(pid) for pid in code.get("example_paper_ids", []) if pid],
            }
        )
    return clean, errors


def run_taxonomy_induction(
    *,
    report_path: Path,
    out: Path,
    run_dir: Path | None,
    config: TaxonomyConfig,
    client: Any,
    sample_size: int,
    max_codes: int,
) -> dict[str, Any]:
    report = read_json(report_path)
    cases = balanced_sample(collect_cases(report), sample_size)
    messages = build_induce_messages(cases, max_codes=max_codes)
    base = {
        "analysis": "divergence_taxonomy_induction",
        "prompt_version": TAXONOMY_INDUCE_PROMPT_VERSION,
        "provider": config.provider,
        "model": config.model,
        "report_path": str(report_path),
        "sampled_case_count": len(cases),
        "max_codes": max_codes,
    }
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "induce_prompt.json", {**base, "messages": messages})
    if config.dry_run:
        result = {**base, "status": "dry_run", "messages": messages}
        write_json(out, result)
        return result
    result = _complete_json(client, messages, config)
    if result["status"] != "ok":
        write_json(out, {**base, **result})
        return {**base, **result}
    codes, errors = validate_codebook(result["parsed"], max_codes=max_codes)
    payload = {
        **base,
        "status": "ok" if not errors else "validation_error",
        "validation_errors": errors,
        "codebook_note": "Review/edit this codebook, then pass it to divergence-taxonomy-classify.",
        "sampled_case_ids": [case["paper_id"] for case in cases],
        "codes": codes,
        "served_model": result.get("served_model"),
        "usage": result.get("usage"),
    }
    write_json(out, payload)
    return payload


# --- stage 2: classify ----------------------------------------------------


def build_classify_messages(
    cases: list[dict[str, Any]],
    codebook: list[dict[str, Any]],
    *,
    max_codes_per_case: int,
) -> list[dict[str, str]]:
    code_lines = "\n".join(f"- {code['code']}: {code['definition']}" for code in codebook)
    system = """You are coding why an impact-forward AI area chair diverged from human ML-conference decisions, using a fixed codebook.

Assign codes by grounding them in the paper abstract and observable contrast between the AI's axis scores and the human outcome (decision, tier, reviewer soundness/overall), not by echoing the AI's own justification. Treat codes as descriptive associations, not causal claims about private reviewer deliberation. Use only codebook codes. Prefer 1-2 codes; add a third only if clearly warranted.

Return only valid JSON."""
    user = f"""Codebook:
{code_lines}

For each paper, assign up to {max_codes_per_case} codes with a one-line evidence grounded in its numbers/outcome.

Required JSON:
{{
  "assignments": [
    {{"paper_id": "id", "codes": [{{"code": "codebook_code", "evidence": "grounded one-liner"}}]}}
  ]
}}

Papers that must each appear exactly once: {", ".join(case["paper_id"] for case in cases)}

Cases:
{chr(10).join(case_brief(case) for case in cases)}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate_classification(
    payload: dict[str, Any],
    *,
    expected_ids: list[str],
    code_names: set[str],
    max_codes_per_case: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    rows = payload.get("assignments")
    if not isinstance(rows, list):
        return [], ["assignments must be a list"]
    expected = set(expected_ids)
    by_id: dict[str, list[dict[str, str]]] = {}
    invalid_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        paper_id = str(row.get("paper_id") or "")
        if paper_id not in expected:
            errors.append(f"unexpected paper_id: {paper_id}")
            continue
        if paper_id in by_id:
            errors.append(f"duplicate paper_id: {paper_id}")
            invalid_ids.add(paper_id)
            continue
        assigned: list[dict[str, Any]] = []
        seen_codes: set[str] = set()
        for entry in row.get("codes", []) if isinstance(row.get("codes"), list) else []:
            code = str((entry or {}).get("code") or "") if isinstance(entry, dict) else str(entry)
            if code not in code_names:
                errors.append(f"{paper_id}: unknown code {code}")
                invalid_ids.add(paper_id)
                continue
            if code in seen_codes:
                errors.append(f"{paper_id}: duplicate code {code}")
                invalid_ids.add(paper_id)
                continue
            seen_codes.add(code)
            evidence = str((entry or {}).get("evidence") or "").strip() if isinstance(entry, dict) else ""
            if not evidence:
                errors.append(f"{paper_id}: code {code} has no evidence")
                invalid_ids.add(paper_id)
            assigned.append({"code": code, "evidence": evidence})
        if not assigned:
            errors.append(f"{paper_id}: no valid codes assigned")
            invalid_ids.add(paper_id)
        if len(assigned) > max_codes_per_case:
            errors.append(
                f"{paper_id}: assigned {len(assigned)} codes; maximum is {max_codes_per_case}"
            )
            invalid_ids.add(paper_id)
            assigned = assigned[:max_codes_per_case]
        by_id[paper_id] = assigned
    assignments = [
        {
            "paper_id": paper_id,
            "status": "ok" if paper_id in by_id and paper_id not in invalid_ids else "validation_error",
            "codes": by_id.get(paper_id, []),
        }
        for paper_id in expected_ids
    ]
    missing = [paper_id for paper_id in expected_ids if paper_id not in by_id]
    if missing:
        errors.append("missing assignments: " + ", ".join(missing))
    return assignments, errors


def classify_cases(
    cases: list[dict[str, Any]],
    codebook: list[dict[str, Any]],
    *,
    config: TaxonomyConfig,
    client: Any,
    batch_size: int,
    max_codes_per_case: int,
    run_dir: Path | None,
    tag: str,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    code_names = {code["code"] for code in codebook}
    assignments: list[dict[str, Any]] = []
    errors: list[str] = []
    batch_meta: list[dict[str, Any]] = []
    for index in range(0, len(cases), max(1, batch_size)):
        batch = cases[index : index + max(1, batch_size)]
        messages = build_classify_messages(batch, codebook, max_codes_per_case=max_codes_per_case)
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(run_dir / f"classify_{tag}_prompt_{index:04d}.json", {"messages": messages})
        result = _complete_json(client, messages, config)
        batch_row = {
            "batch_start": index,
            "paper_ids": [case["paper_id"] for case in batch],
            "status": result["status"],
            "served_model": result.get("served_model"),
            "usage": result.get("usage"),
            "elapsed_seconds": result.get("elapsed_seconds"),
        }
        if result.get("error"):
            batch_row["error"] = result["error"]
        batch_meta.append(batch_row)
        if run_dir is not None:
            write_json(run_dir / f"classify_{tag}_result_{index:04d}.json", result)
        if result["status"] != "ok":
            errors.append(f"batch {index}: {result.get('error') or result['status']}")
            for case in batch:
                assignments.append(
                    {
                        "paper_id": case["paper_id"],
                        "status": "failed",
                        "codes": [],
                        "error": result.get("error") or result["status"],
                    }
                )
            continue
        batch_assignments, batch_errors = validate_classification(
            result["parsed"],
            expected_ids=[case["paper_id"] for case in batch],
            code_names=code_names,
            max_codes_per_case=max_codes_per_case,
        )
        assignments.extend(batch_assignments)
        errors.extend(batch_errors)
    return assignments, errors, batch_meta


def run_taxonomy_classification(
    *,
    report_path: Path,
    codebook_path: Path,
    out: Path,
    run_dir: Path | None,
    config: TaxonomyConfig,
    client: Any,
    batch_size: int,
    max_codes_per_case: int,
    second_client: Any | None = None,
    second_label: str | None = None,
    second_config: TaxonomyConfig | None = None,
) -> dict[str, Any]:
    report = read_json(report_path)
    cases = collect_cases(report)
    codebook_payload = read_json(codebook_path)
    codebook = codebook_payload.get("codes") if isinstance(codebook_payload, dict) else None
    if not isinstance(codebook, list) or not codebook:
        result = {"analysis": "divergence_taxonomy_classification", "status": "failed", "error": "codebook has no codes"}
        write_json(out, result)
        return result
    direction_by_id = {case["paper_id"]: case["direction"] for case in cases}
    class_by_id = {case["paper_id"]: case.get("contribution_class") for case in cases}
    base = {
        "analysis": "divergence_taxonomy_classification",
        "prompt_version": TAXONOMY_CLASSIFY_PROMPT_VERSION,
        "provider": config.provider,
        "model": config.model,
        "report_path": str(report_path),
        "codebook_path": str(codebook_path),
        "case_count": len(cases),
        "code_count": len(codebook),
    }
    if config.dry_run:
        messages = build_classify_messages(cases[: max(1, batch_size)], codebook, max_codes_per_case=max_codes_per_case)
        result = {**base, "status": "dry_run", "messages": messages}
        write_json(out, result)
        return result

    assignments, errors, batch_meta = classify_cases(
        cases, codebook, config=config, client=client, batch_size=batch_size,
        max_codes_per_case=max_codes_per_case, run_dir=run_dir, tag="primary",
    )
    enriched = [
        {**row, "direction": direction_by_id.get(row["paper_id"]), "contribution_class": class_by_id.get(row["paper_id"])}
        for row in assignments
    ]
    payload = {
        **base,
        "status": "ok" if not errors else "validation_error",
        "validation_errors": errors,
        "batches": batch_meta,
        "assignments": enriched,
        "aggregate": aggregate_codes(enriched, codebook),
    }
    sampled_ids = set(codebook_payload.get("sampled_case_ids") or [])
    if sampled_ids:
        payload["held_out_aggregate"] = aggregate_codes(
            [row for row in enriched if row["paper_id"] not in sampled_ids],
            codebook,
        )
    if second_client is not None:
        effective_second_config = second_config or config
        second_assignments, second_errors, second_batches = classify_cases(
            cases, codebook, config=effective_second_config, client=second_client, batch_size=batch_size,
            max_codes_per_case=max_codes_per_case, run_dir=run_dir, tag="second",
        )
        second_enriched = [
            {
                **row,
                "direction": direction_by_id.get(row["paper_id"]),
                "contribution_class": class_by_id.get(row["paper_id"]),
            }
            for row in second_assignments
        ]
        payload["reliability"] = {
            "second_label": second_label or "second_model",
            "second_provider": effective_second_config.provider,
            "second_model": effective_second_config.model,
            "second_validation_errors": second_errors,
            "second_batches": second_batches,
            "second_assignments": second_enriched,
            "per_code_cohens_kappa": cohens_kappa_per_code(
                assignments, second_assignments, [code["code"] for code in codebook]
            ),
            "exact_multilabel_agreement": exact_multilabel_agreement(
                assignments,
                second_assignments,
            ),
        }
    write_json(out, payload)
    return payload


# --- aggregation & reliability --------------------------------------------


def aggregate_codes(assignments: list[dict[str, Any]], codebook: list[dict[str, Any]]) -> dict[str, Any]:
    code_names = [code["code"] for code in codebook]
    totals = {name: 0 for name in code_names}
    by_direction: dict[str, dict[str, int]] = {}
    by_class: dict[str, dict[str, int]] = {}
    valid_assignments = [row for row in assignments if row.get("status", "ok") == "ok"]
    for row in valid_assignments:
        direction = str(row.get("direction") or "unknown")
        klass = str(row.get("contribution_class") or "unknown")
        for code in assigned_code_names(row):
            if code in totals:
                totals[code] += 1
            by_direction.setdefault(direction, {})[code] = by_direction.setdefault(direction, {}).get(code, 0) + 1
            by_class.setdefault(klass, {})[code] = by_class.setdefault(klass, {}).get(code, 0) + 1
    return {
        "cases_requested": len(assignments),
        "cases_coded": len(valid_assignments),
        "cases_failed": len(assignments) - len(valid_assignments),
        "code_totals": totals,
        "by_direction": by_direction,
        "by_contribution_class": by_class,
    }


def cohens_kappa_per_code(
    assignments_a: list[dict[str, Any]],
    assignments_b: list[dict[str, Any]],
    code_names: list[str],
) -> dict[str, dict[str, Any]]:
    a_by_id = {
        row["paper_id"]: set(assigned_code_names(row))
        for row in assignments_a
        if row.get("status", "ok") == "ok"
    }
    b_by_id = {
        row["paper_id"]: set(assigned_code_names(row))
        for row in assignments_b
        if row.get("status", "ok") == "ok"
    }
    shared = [paper_id for paper_id in a_by_id if paper_id in b_by_id]
    result: dict[str, dict[str, Any]] = {}
    for code in code_names:
        a_labels = [code in a_by_id[paper_id] for paper_id in shared]
        b_labels = [code in b_by_id[paper_id] for paper_id in shared]
        agreement = (
            sum(a == b for a, b in zip(a_labels, b_labels)) / len(shared)
            if shared
            else None
        )
        result[code] = {
            "n": len(shared),
            "kappa": cohens_kappa(a_labels, b_labels),
            "observed_agreement": round(agreement, 4) if agreement is not None else None,
            "prevalence_primary": round(sum(a_labels) / len(shared), 4) if shared else None,
            "prevalence_second": round(sum(b_labels) / len(shared), 4) if shared else None,
        }
    return result


def cohens_kappa(a_labels: list[bool], b_labels: list[bool]) -> float | None:
    if len(a_labels) != len(b_labels):
        raise ValueError("label arrays must have equal length")
    n = len(a_labels)
    if n == 0:
        return None
    agree = sum(1 for a, b in zip(a_labels, b_labels) if a == b) / n
    pa = sum(a_labels) / n
    pb = sum(b_labels) / n
    chance = pa * pb + (1 - pa) * (1 - pb)
    if chance >= 1.0:
        return None
    return round((agree - chance) / (1 - chance), 4)


def assigned_code_names(row: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for entry in row.get("codes", []) if isinstance(row.get("codes"), list) else []:
        if isinstance(entry, dict):
            name = str(entry.get("code") or "")
        else:
            name = str(entry)
        if name:
            names.append(name)
    return names


def exact_multilabel_agreement(
    assignments_a: list[dict[str, Any]],
    assignments_b: list[dict[str, Any]],
) -> dict[str, Any]:
    a_by_id = {
        row["paper_id"]: set(assigned_code_names(row))
        for row in assignments_a
        if row.get("status", "ok") == "ok"
    }
    b_by_id = {
        row["paper_id"]: set(assigned_code_names(row))
        for row in assignments_b
        if row.get("status", "ok") == "ok"
    }
    shared = sorted(set(a_by_id) & set(b_by_id))
    exact = sum(a_by_id[paper_id] == b_by_id[paper_id] for paper_id in shared)
    return {
        "n": len(shared),
        "exact_matches": exact,
        "rate": round(exact / len(shared), 4) if shared else None,
    }


# --- provider call --------------------------------------------------------


def _complete_json(client: Any, messages: list[dict[str, str]], config: TaxonomyConfig) -> dict[str, Any]:
    if client is None:
        raise ValueError("client is required unless dry_run=True")
    started = time.monotonic()
    try:
        chat = client.complete(
            messages=messages,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            seed=config.seed,
            response_format={"type": "json_object"},
        )
        parsed = parse_json_response(chat.content)
        return {
            "status": "ok",
            "parsed": parsed,
            "served_model": getattr(chat, "served_model", None),
            "usage": getattr(chat, "usage", {}),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    except Exception as exc:  # noqa: BLE001 - surface provider/JSON failures to the caller.
        return {"status": "failed", "error": repr(exc), "elapsed_seconds": round(time.monotonic() - started, 3)}
