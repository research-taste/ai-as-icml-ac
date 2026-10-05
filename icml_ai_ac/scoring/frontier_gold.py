from __future__ import annotations

import base64
import hashlib
import itertools
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from icml_ai_ac.models import PaperRecord
from icml_ai_ac.pdf_utils import write_pdf_page_excerpt
from icml_ai_ac.scoring.providers import ChatCompletionClient, is_batch_blocking_provider_error
from icml_ai_ac.scoring.runner import (
    estimate_tokens,
    read_text_path,
    resolve_record_pdf_path,
    resolve_record_text_path,
)
from icml_ai_ac.scoring.schema import CONTRIBUTION_CLASSES, parse_json_response
from icml_ai_ac.scoring.usage import sum_usage
from icml_ai_ac.storage import read_paper_records, write_json, write_jsonl


FRONTIER_CARD_PROMPT_VERSION = "frontier_pdf_paper_card_v2"
FRONTIER_CARD_ENSEMBLE_VERSION = "frontier_pdf_card_ensemble_v2"
FRONTIER_SYNTHESIS_PROMPT_VERSION = "frontier_pdf_card_synthesis_rank_v2"
FRONTIER_TOURNAMENT_PROMPT_VERSION = "frontier_pdf_card_pairwise_tournament_v4"


@dataclass(slots=True)
class FrontierCardConfig:
    provider: str
    model: str
    reasoning_effort: str | None
    prompt_version: str
    paper_set_name: str
    text_source: str
    limit: int | None
    paper_ids: set[str]
    per_paper_char_budget: int
    pdf_excerpt_pages: int
    pdf_excerpt_dir: Path
    pdf_optimize_threshold_bytes: int | None
    pdf_settings: str
    temperature: float
    max_output_tokens: int
    seed: int | None
    dry_run: bool
    overwrite: bool
    openrouter_pdf_engine: str | None
    fallback_model: str | None
    fallback_reasoning_effort: str | None


@dataclass(slots=True)
class FrontierCardCandidate:
    record: PaperRecord
    text_path: str
    resolved_text_source: str
    paper_text: str
    pdf_path: Path
    pdf_excerpt_path: Path


@dataclass(slots=True)
class FrontierSynthesisConfig:
    provider: str
    model: str
    reasoning_effort: str | None
    prompt_version: str
    paper_set_name: str
    temperature: float
    max_output_tokens: int
    seed: int | None
    dry_run: bool


@dataclass(slots=True)
class FrontierTournamentConfig:
    provider: str
    model: str
    reasoning_effort: str | None
    prompt_version: str
    paper_set_name: str
    strategy: str
    top_n: int
    pairs_per_batch: int
    swiss_rounds: int
    playoff_top_n: int | None
    temperature: float
    max_output_tokens: int
    seed: int | None
    dry_run: bool
    overwrite: bool


def run_frontier_pdf_cards(
    *,
    manifest: Path,
    out: Path,
    run_dir: Path,
    config: FrontierCardConfig,
    client: ChatCompletionClient | None,
    fallback_client: ChatCompletionClient | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    candidates = select_frontier_card_candidates(manifest=manifest, config=config)
    for child in ["prompts", "responses", "parsed", "cards", "errors"]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    blocked_reason: str | None = None
    for index, candidate in enumerate(candidates, start=1):
        existing_completed: dict[str, Any] | None = None
        existing_failure: dict[str, Any] | None = None
        primary_card_path = run_dir / "cards" / f"{candidate.record.paper_id}.json"
        if config.fallback_model and primary_card_path.exists() and not config.overwrite:
            existing = json.loads(primary_card_path.read_text(encoding="utf-8"))
            fallback_config = replace(
                config,
                model=config.fallback_model,
                reasoning_effort=config.fallback_reasoning_effort,
                fallback_model=None,
                fallback_reasoning_effort=None,
                overwrite=False,
            )
            if (
                existing.get("status") == "ok"
                and existing.get("fallback_from_model") == config.model
                and existing.get("fingerprint")
                == frontier_card_request_fingerprint(candidate, config=fallback_config)
            ):
                existing_completed = {**existing, "resumed_from": str(primary_card_path)}
            if (
                existing.get("status") not in {"ok", "dry_run"}
                and not existing.get("blocking_provider_error")
                and existing.get("fingerprint") == frontier_card_request_fingerprint(candidate, config=config)
            ):
                existing_failure = existing
        row = existing_completed or existing_failure or run_frontier_pdf_card(
            candidate,
            index=index,
            total=len(candidates),
            run_dir=run_dir,
            config=config,
            client=client,
        )
        if (
            row.get("status") not in {"ok", "dry_run"}
            and not row.get("blocking_provider_error")
            and config.fallback_model
            and fallback_client is not None
        ):
            primary_row = row
            fallback_config = replace(
                config,
                model=config.fallback_model,
                reasoning_effort=config.fallback_reasoning_effort,
                fallback_model=None,
                fallback_reasoning_effort=None,
                overwrite=False,
            )
            fallback_row = run_frontier_pdf_card(
                candidate,
                index=index,
                total=len(candidates),
                run_dir=run_dir / "fallbacks" / candidate.record.paper_id,
                config=fallback_config,
                client=fallback_client,
            )
            fallback_row["fallback_from_model"] = config.model
            fallback_row["fallback_reason"] = primary_row.get("error") or primary_row.get("validation_errors")
            fallback_row["primary_attempt"] = {
                key: primary_row.get(key)
                for key in (
                    "status",
                    "model",
                    "served_model",
                    "error",
                    "error_path",
                    "validation_errors",
                    "usage",
                )
                if primary_row.get(key) is not None
            }
            fallback_row["attempt_usage"] = sum_usage(
                (primary_row.get("usage"), fallback_row.get("usage"))
            )
            row = fallback_row
            if row.get("status") == "ok":
                write_json(run_dir / "cards" / f"{candidate.record.paper_id}.json", row)
        rows.append(row)
        write_jsonl(out, rows)
        if row.get("blocking_provider_error"):
            blocked_reason = str(row.get("blocked_reason") or row.get("error") or "provider-wide error")
            break
    summary = {
        "status": "dry_run" if config.dry_run else "ok",
        "provider": config.provider,
        "model": config.model,
        "fallback_model": config.fallback_model,
        "reasoning_effort": config.reasoning_effort,
        "prompt_version": config.prompt_version,
        "paper_set_name": config.paper_set_name,
        "manifest": str(manifest),
        "out": str(out),
        "run_dir": str(run_dir),
        "candidate_count": len(candidates),
        "attempted_count": len(rows),
        "skipped_count": len(candidates) - len(rows),
        "ok_count": sum(1 for row in rows if row.get("status") == "ok"),
        "failed_count": sum(1 for row in rows if row.get("status") == "failed"),
        "validation_error_count": sum(1 for row in rows if row.get("status") == "validation_error"),
        "dry_run_count": sum(1 for row in rows if row.get("status") == "dry_run"),
        "fallback_count": sum(1 for row in rows if row.get("fallback_from_model")),
        "blocked_reason": blocked_reason,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "usage": sum_usage(row.get("attempt_usage") or row.get("usage") for row in rows),
    }
    if summary["failed_count"] or summary["validation_error_count"] or summary["skipped_count"]:
        summary["status"] = "partial" if not config.dry_run else "dry_run"
    write_json(run_dir / "run.json", summary)
    return summary


def run_frontier_pdf_card(
    candidate: FrontierCardCandidate,
    *,
    index: int,
    total: int,
    run_dir: Path,
    config: FrontierCardConfig,
    client: ChatCompletionClient | None,
) -> dict[str, Any]:
    paper_id = candidate.record.paper_id
    card_path = run_dir / "cards" / f"{paper_id}.json"
    request_fingerprint = frontier_card_request_fingerprint(candidate, config=config)
    if card_path.exists() and not config.overwrite:
        existing = json.loads(card_path.read_text(encoding="utf-8"))
        if existing.get("status") == "ok" and existing.get("fingerprint") == request_fingerprint:
            return {**existing, "resumed_from": str(card_path)}

    base_result: dict[str, Any] = {
        "status": "dry_run" if config.dry_run else "pending",
        "paper_id": paper_id,
        "title": candidate.record.title,
        "provider": config.provider,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "prompt_version": config.prompt_version,
        "paper_set_name": config.paper_set_name,
        "text_source": config.text_source,
        "resolved_text_source": candidate.resolved_text_source,
        "text_path": candidate.text_path,
        "pdf_path": str(candidate.pdf_path),
        "pdf_excerpt_path": str(candidate.pdf_excerpt_path),
        "candidate_index": index,
        "candidate_total": total,
        "fingerprint": request_fingerprint,
        "raw_response_path": None,
        "parsed_response_path": None,
        "prompt_path": None,
        "validation_errors": [],
    }
    chat = None
    try:
        excerpt_info = write_pdf_page_excerpt(
            source_pdf=candidate.pdf_path,
            out_pdf=candidate.pdf_excerpt_path,
            max_pages=config.pdf_excerpt_pages,
            overwrite=False,
            optimize_if_larger_than_bytes=config.pdf_optimize_threshold_bytes,
            pdf_settings=config.pdf_settings,
        )
        messages = build_frontier_card_messages(candidate, config=config)
        prompt_payload = {
            **base_result,
            "status": "prompt_ready",
            "pdf_excerpt": excerpt_info,
            "messages": redact_file_data(messages),
            "prompt_tokens_estimate_without_pdf": sum(
                estimate_tokens(_message_text_for_estimate(message)) for message in messages
            ),
        }
        prompt_path = run_dir / "prompts" / f"{paper_id}.json"
        write_json(prompt_path, prompt_payload)
        base_result["prompt_path"] = str(prompt_path)
        if config.dry_run:
            row = {**base_result, "status": "dry_run", "pdf_excerpt": excerpt_info}
            write_json(card_path, row)
            return row
        if client is None:
            raise ValueError("client is required unless dry_run=True")
        chat = client.complete(
            messages=messages,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            seed=config.seed,
            response_format={"type": "json_object"},
        )
        raw_response_path = run_dir / "responses" / f"{paper_id}.json"
        write_json(raw_response_path, chat.response)
        parsed = parse_json_response(chat.content)
        parsed.setdefault("paper_id", paper_id)
        parsed.setdefault("title", candidate.record.title or "")
        validation_errors = validate_frontier_card(parsed, expected_paper_id=paper_id)
        parsed_response_path = run_dir / "parsed" / f"{paper_id}.json"
        write_json(parsed_response_path, parsed)
        row = {
            **base_result,
            "status": "ok" if not validation_errors else "validation_error",
            "pdf_excerpt": excerpt_info,
            "raw_response_path": str(raw_response_path),
            "parsed_response_path": str(parsed_response_path),
            "validation_errors": validation_errors,
            "served_model": chat.served_model,
            "usage": chat.usage,
            "provider_elapsed_seconds": round(chat.elapsed_seconds, 3),
            "card": parsed,
        }
    except Exception as exc:  # noqa: BLE001 - persist row-local failures for retry.
        response = getattr(exc, "response", None)
        raw_response_path = run_dir / "responses" / f"{paper_id}.json"
        if isinstance(response, dict) and not raw_response_path.exists():
            write_json(raw_response_path, response)
        usage = chat.usage if chat is not None else getattr(exc, "usage", {})
        served_model = chat.served_model if chat is not None else getattr(exc, "served_model", None)
        error_path = run_dir / "errors" / f"{paper_id}.json"
        write_json(error_path, {"paper_id": paper_id, "error": repr(exc)})
        row = {
            **base_result,
            "status": "failed",
            "error_path": str(error_path),
            "error": repr(exc),
            "blocking_provider_error": is_batch_blocking_provider_error(exc),
        }
        if row["blocking_provider_error"]:
            row["blocked_reason"] = repr(exc)
        if raw_response_path.exists():
            row["raw_response_path"] = str(raw_response_path)
        if isinstance(usage, dict) and usage:
            row["usage"] = usage
        if served_model:
            row["served_model"] = served_model
    write_json(card_path, row)
    return row


def frontier_card_request_fingerprint(
    candidate: FrontierCardCandidate,
    *,
    config: FrontierCardConfig,
) -> str:
    anonymization = candidate.record.extra.get("anonymization")
    pdf_sha256 = (
        anonymization.get("anonymized_pdf_sha256")
        if isinstance(anonymization, dict)
        else candidate.record.extra.get("pdf_sha256")
    )
    if not isinstance(pdf_sha256, str) or not pdf_sha256:
        stat = candidate.pdf_path.stat()
        pdf_sha256 = f"stat:{stat.st_size}:{stat.st_mtime_ns}"
    payload = {
        "paper_id": candidate.record.paper_id,
        "title": candidate.record.title,
        "provider": config.provider,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "prompt_version": config.prompt_version,
        "paper_set_name": config.paper_set_name,
        "text_source": config.text_source,
        "resolved_text_source": candidate.resolved_text_source,
        "paper_text_sha256": hashlib.sha256(candidate.paper_text.encode("utf-8")).hexdigest(),
        "pdf_sha256": pdf_sha256,
        "pdf_excerpt_pages": config.pdf_excerpt_pages,
        "pdf_optimize_threshold_bytes": config.pdf_optimize_threshold_bytes,
        "pdf_settings": config.pdf_settings,
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
        "seed": config.seed,
        "openrouter_pdf_engine": config.openrouter_pdf_engine,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def select_frontier_card_candidates(*, manifest: Path, config: FrontierCardConfig) -> list[FrontierCardCandidate]:
    candidates: list[FrontierCardCandidate] = []
    for record in read_paper_records(manifest):
        if config.paper_ids and record.paper_id not in config.paper_ids:
            continue
        if record.parse_status != "ok":
            continue
        try:
            text_path, resolved_text_source = resolve_record_text_path(record, config.text_source)
            pdf_path, _ = resolve_record_pdf_path(record)
        except ValueError:
            continue
        if not pdf_path.exists():
            continue
        paper_text = truncate_text(read_text_path(text_path), config.per_paper_char_budget)
        candidates.append(
            FrontierCardCandidate(
                record=record,
                text_path=text_path,
                resolved_text_source=resolved_text_source,
                paper_text=paper_text,
                pdf_path=pdf_path,
                pdf_excerpt_path=config.pdf_excerpt_dir / f"{record.paper_id}.first{config.pdf_excerpt_pages}.pdf",
            )
        )
        if config.limit is not None and len(candidates) >= config.limit:
            break
    return candidates


def build_frontier_card_messages(
    candidate: FrontierCardCandidate,
    *,
    config: FrontierCardConfig,
) -> list[dict[str, Any]]:
    system = f"""You are a frontier reference judge for an ICML paper impact study.

Use the supplied first-{config.pdf_excerpt_pages}-page PDF and extracted main-paper text. The PDF is included so you can inspect figures, tables, diagrams, experimental displays, and visual evidence that may not survive text extraction. Do not use web search, citation memory, author identity, institution, or external knowledge.
No reviews, reviewer scores, area-chair comments, decisions, presentation tiers, or awards are provided. Do not infer paper quality from possible venue or outcome cues in the document.

Your task is to create a calibrated paper judgment card that will later feed an ensemble/tournament gold ranking. Be strict: broad impact requires a credible path to changing scientific practice, ML practice, theory, evaluation, safety, infrastructure, or reusable community resources. Technical soundness is a gate, not the only objective.

Return only valid JSON."""
    user_text = f"""Evaluate this paper as a candidate for the strongest papers in {config.paper_set_name}.

Paper id: {candidate.record.paper_id}
Title: {candidate.record.title}

Use both inputs:
- First-{config.pdf_excerpt_pages}-page PDF attachment for figures/tables/layout/evidence.
- Extracted text below for searchable details.

Contribution classes:
{", ".join(CONTRIBUTION_CLASSES)}

Required JSON shape:
{{
  "evaluation_mode": "frontier_pdf_paper_card",
  "prompt_version": "{config.prompt_version}",
  "judge": "{config.provider}:{config.model}",
  "paper_id": "{candidate.record.paper_id}",
  "title": "title",
  "primary_contribution_class": "core_ml_algorithm",
  "secondary_contribution_classes": ["scientific_modeling_tool"],
  "scores": {{
    "overall_gold_priority_score": 1,
    "broad_scientific_impact_score": 1,
    "ml_field_impact_score": 1,
    "technical_soundness_score": 1,
    "novelty_score": 1,
    "evidence_confidence_score": 1,
    "visual_evidence_importance_score": 1
  }},
  "impact_assessment": {{
    "two_year_adoption_path": "who uses it and what changes",
    "five_year_field_effect": "what durable effect could remain",
    "best_case_for_impact": "strongest plausible impact route",
    "main_risk": "main reason this may not matter or hold up",
    "why_not_higher": "calibration reason",
    "why_not_lower": "calibration reason"
  }},
  "evidence_from_pdf": {{
    "figures_or_tables_that_matter": ["specific observed evidence from the PDF"],
    "visual_or_tabular_evidence_changes_judgment": "yes/no/maybe plus why",
    "missing_or_weak_visual_evidence": ["important caveats"]
  }},
  "ranking_hooks": {{
    "beats_papers_when": "conditions under which this should rank above peers",
    "loses_to_papers_when": "conditions under which this should rank below peers",
    "category_standout": true,
    "top_10_case": "case for or against being top 10 in this 50-paper diagnostic"
  }},
  "one_sentence_summary": "dense impact summary"
}}

Scoring calibration:
- 10: plausible best-of-conference or field-shaping impact.
- 8-9: clear high-impact finalist with strong evidence.
- 6-7: solid accepted paper, useful but impact route is narrower or evidence-limited.
- 4-5: incremental, fragile, or mainly local interest.
- 1-3: serious soundness/relevance concerns for broad-impact ranking.

Extracted main-paper text:
<<<PAPER_TEXT
{candidate.paper_text}
PAPER_TEXT
"""
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "file",
                    "file": {
                        "filename": candidate.pdf_excerpt_path.name,
                        "file_data": encode_pdf_data_url(candidate.pdf_excerpt_path),
                        "path": str(candidate.pdf_excerpt_path),
                    },
                },
            ],
        },
    ]


def build_frontier_card_ensemble(
    *,
    card_paths: list[Path],
    labels: list[str] | None = None,
    out: Path,
    paper_set_name: str,
    prompt_version: str = FRONTIER_CARD_ENSEMBLE_VERSION,
) -> dict[str, Any]:
    if not card_paths:
        raise ValueError("at least one frontier card path is required")
    stream_labels = labels or [path.stem for path in card_paths]
    if len(stream_labels) != len(card_paths):
        raise ValueError("supply exactly one --label per --cards path")
    if len(set(stream_labels)) != len(stream_labels):
        raise ValueError("frontier card stream labels must be unique")

    stream_rows: dict[str, dict[str, dict[str, Any]]] = {}
    stream_metadata: list[dict[str, Any]] = []
    expected_ids: set[str] | None = None
    for label, path in zip(stream_labels, card_paths, strict=True):
        rows_by_paper = load_complete_card_stream(path=path, label=label)
        paper_ids = set(rows_by_paper)
        if expected_ids is None:
            expected_ids = paper_ids
        elif paper_ids != expected_ids:
            missing = sorted(expected_ids - paper_ids)
            extras = sorted(paper_ids - expected_ids)
            raise ValueError(
                f"frontier card coverage mismatch for {label}: missing={missing}, extras={extras}"
            )
        ranked_rows = rank_frontier_card_stream(rows_by_paper)
        stream_rows[label] = ranked_rows
        stream_metadata.append(
            {
                "label": label,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "row_count": len(ranked_rows),
                "served_models": dict(
                    sorted(
                        _counts(
                            str(row.get("served_model") or "")
                            for row in ranked_rows.values()
                            if row.get("served_model")
                        ).items()
                    )
                ),
                "fallback_count": sum(
                    1 for row in ranked_rows.values() if row.get("fallback_from_model")
                ),
            }
        )

    paper_ids = sorted(expected_ids or set())
    ranked_items = [
        ensemble_item(
            paper_id,
            {label: stream_rows[label][paper_id] for label in stream_labels},
        )
        for paper_id in paper_ids
    ]
    ranked_items.sort(key=frontier_ensemble_sort_key)
    for rank, item in enumerate(ranked_items, start=1):
        item["rank"] = rank
    payload = {
        "evaluation_mode": "frontier_pdf_card_ensemble",
        "prompt_version": prompt_version,
        "paper_set": paper_set_name,
        "card_paths": [str(path) for path in card_paths],
        "card_streams": stream_metadata,
        "candidate_count": len(ranked_items),
        "judge_count": len(stream_labels),
        "judge_count_by_paper": {paper_id: len(stream_labels) for paper_id in paper_ids},
        "ranked_papers": ranked_items,
        "category_rankings": build_category_rankings(ranked_items),
        "method_notes": {
            "modality": "Each judge card saw the configured first-page PDF excerpt plus extracted main-paper text.",
            "ranking_method": (
                "Equal-weight mean of tie-aware within-judge rank percentiles. "
                "Overall gold priority is primary; broad impact, ML impact, technical "
                "soundness, evidence confidence, and novelty break exact score ties."
            ),
            "scale_handling": (
                "Raw scores are retained for analysis but never averaged to determine "
                "the ensemble rank."
            ),
            "fallback_handling": (
                "Each input path is one judge stream, so an explicitly recorded fallback "
                "remains part of its requested panel stream rather than becoming an extra judge."
            ),
            "limitation": "This is a PDF-aware ensemble prior, not the final pairwise tournament ranking.",
        },
    }
    write_json(out, payload)
    return payload


def run_frontier_card_synthesis(
    *,
    card_paths: list[Path],
    out: Path,
    run_dir: Path,
    config: FrontierSynthesisConfig,
    client: ChatCompletionClient | None,
) -> dict[str, Any]:
    started = time.monotonic()
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = load_card_rows(card_paths)
    rows_by_paper: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        paper_id = str(row.get("paper_id") or "")
        if paper_id:
            rows_by_paper.setdefault(paper_id, []).append(row)
    paper_ids = sorted(rows_by_paper)
    messages = build_frontier_synthesis_messages(rows_by_paper, config=config)
    prompt_payload = {
        "prompt_version": config.prompt_version,
        "provider": config.provider,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
        "seed": config.seed,
        "paper_set_name": config.paper_set_name,
        "card_paths": [str(path) for path in card_paths],
        "card_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in card_paths
        },
        "candidate_count": len(paper_ids),
        "candidate_ids": paper_ids,
        "messages": messages,
        "prompt_tokens_estimate": sum(estimate_tokens(_message_text_for_estimate(message)) for message in messages),
    }
    prompt_path = run_dir / "prompt.json"
    write_json(prompt_path, prompt_payload)
    base_result: dict[str, Any] = {
        "status": "dry_run" if config.dry_run else "pending",
        "provider": config.provider,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
        "seed": config.seed,
        "prompt_version": config.prompt_version,
        "paper_set_name": config.paper_set_name,
        "card_paths": [str(path) for path in card_paths],
        "out": str(out),
        "run_dir": str(run_dir),
        "candidate_count": len(paper_ids),
        "candidate_ids": paper_ids,
        "prompt_path": str(prompt_path),
        "raw_response_path": None,
        "parsed_response_path": None,
        "validation_errors": [],
    }
    if config.dry_run:
        result = {**base_result, "elapsed_seconds": round(time.monotonic() - started, 3)}
        write_json(out, result)
        write_json(run_dir / "run.json", result)
        return result
    if client is None:
        raise ValueError("client is required unless dry_run=True")
    chat = None
    try:
        chat = client.complete(
            messages=messages,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            seed=config.seed,
            response_format={"type": "json_object"},
        )
        raw_response_path = run_dir / "response.json"
        write_json(raw_response_path, chat.response)
        parsed = parse_json_response(chat.content)
        validation_errors = validate_synthesis_ranking(parsed, expected_ids=paper_ids)
        parsed_response_path = run_dir / "parsed.json"
        write_json(parsed_response_path, parsed)
        result = {
            **base_result,
            "status": "ok" if not validation_errors else "validation_error",
            "raw_response_path": str(raw_response_path),
            "parsed_response_path": str(parsed_response_path),
            "validation_errors": validation_errors,
            "served_model": chat.served_model,
            "usage": chat.usage,
            "provider_elapsed_seconds": round(chat.elapsed_seconds, 3),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "ranking": parsed,
        }
    except Exception as exc:  # noqa: BLE001 - persist failed synthesis attempts.
        response = getattr(exc, "response", None)
        raw_response_path = run_dir / "response.json"
        if isinstance(response, dict) and not raw_response_path.exists():
            write_json(raw_response_path, response)
        usage = chat.usage if chat is not None else getattr(exc, "usage", {})
        served_model = chat.served_model if chat is not None else getattr(exc, "served_model", None)
        error_path = run_dir / "error.json"
        write_json(error_path, {"error": repr(exc)})
        result = {
            **base_result,
            "status": "failed",
            "error_path": str(error_path),
            "error": repr(exc),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        if raw_response_path.exists():
            result["raw_response_path"] = str(raw_response_path)
        if isinstance(usage, dict) and usage:
            result["usage"] = usage
        if served_model:
            result["served_model"] = served_model
    write_json(out, result if result["status"] == "failed" else result["ranking"])
    write_json(run_dir / "run.json", result)
    return result


def run_frontier_card_tournament(
    *,
    card_paths: list[Path],
    seed_ranking_path: Path,
    out: Path,
    run_dir: Path,
    config: FrontierTournamentConfig,
    client: ChatCompletionClient | None,
) -> dict[str, Any]:
    started = time.monotonic()
    if config.top_n < 2:
        raise ValueError("top_n must be at least 2")
    if config.pairs_per_batch < 1:
        raise ValueError("pairs_per_batch must be at least 1")
    if config.strategy not in {"all_pairs", "swiss", "swiss_playoff"}:
        raise ValueError(f"unsupported tournament strategy: {config.strategy}")
    if config.strategy in {"swiss", "swiss_playoff"} and config.swiss_rounds < 1:
        raise ValueError("swiss_rounds must be at least 1 for Swiss strategies")
    if config.strategy == "swiss_playoff":
        if config.playoff_top_n is None or config.playoff_top_n < 2:
            raise ValueError("playoff_top_n must be at least 2 for swiss_playoff")
        if config.playoff_top_n > config.top_n:
            raise ValueError("playoff_top_n cannot exceed top_n")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "batches").mkdir(parents=True, exist_ok=True)

    seed_payload = json.loads(seed_ranking_path.read_text(encoding="utf-8"))
    seed_ranked = seed_payload.get("ranked_papers")
    if not isinstance(seed_ranked, list):
        raise ValueError(f"seed ranking missing ranked_papers: {seed_ranking_path}")
    seed_ranked = [row for row in seed_ranked if isinstance(row, dict)]
    seed_ids = [str(row.get("paper_id") or "") for row in sorted(seed_ranked, key=lambda row: number(row.get("rank")))]
    seed_ids = [paper_id for paper_id in seed_ids if paper_id]
    tournament_ids = seed_ids[: config.top_n]

    rows = load_card_rows(card_paths)
    rows_by_paper: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        paper_id = str(row.get("paper_id") or "")
        if paper_id:
            rows_by_paper.setdefault(paper_id, []).append(row)
    missing_cards = [paper_id for paper_id in tournament_ids if paper_id not in rows_by_paper]
    if missing_cards:
        raise ValueError(f"missing card rows for tournament papers: {missing_cards}")

    planned_pairs = build_tournament_pairs(tournament_ids) if config.strategy == "all_pairs" else []
    batch_count_estimate = estimate_tournament_batch_count(config)
    run_payload = {
        "prompt_version": config.prompt_version,
        "provider": config.provider,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
        "seed": config.seed,
        "overwrite": config.overwrite,
        "paper_set_name": config.paper_set_name,
        "strategy": config.strategy,
        "card_paths": [str(path) for path in card_paths],
        "card_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in card_paths
        },
        "seed_ranking_path": str(seed_ranking_path),
        "seed_ranking_sha256": hashlib.sha256(
            seed_ranking_path.read_bytes()
        ).hexdigest(),
        "out": str(out),
        "run_dir": str(run_dir),
        "top_n": config.top_n,
        "swiss_rounds": config.swiss_rounds,
        "playoff_top_n": config.playoff_top_n,
        "pairs_per_batch": config.pairs_per_batch,
        "pair_schedule": "deterministic_random",
        "pair_schedule_seed": config.seed or 0,
        "pair_orientation": "deterministic_random",
        "pair_orientation_seed": config.seed or 0,
        "tournament_ids": tournament_ids,
        "pair_count": len(planned_pairs),
        "batch_count_estimate": batch_count_estimate,
        "dry_run": config.dry_run,
    }
    write_json(run_dir / "run.request.json", run_payload)

    seed_rank_by_id = {paper_id: rank for rank, paper_id in enumerate(seed_ids, start=1)}
    schedule_result = run_tournament_schedule(
        tournament_ids=tournament_ids,
        rows_by_paper=rows_by_paper,
        seed_ranked=seed_ranked,
        seed_rank_by_id=seed_rank_by_id,
        run_dir=run_dir,
        config=config,
        client=client,
        batch_count_estimate=batch_count_estimate,
    )
    batch_results = schedule_result["batch_results"]
    matches = schedule_result["matches"]
    tournament_stats = schedule_result["tournament_stats"]
    tournament_order_ids = [str(row.get("paper_id")) for row in tournament_stats]

    failed_batches = [row for row in batch_results if row.get("status") not in {"ok", "dry_run"}]
    if config.dry_run:
        result = {
            **run_payload,
            "status": "dry_run",
            "batch_results": batch_results,
            "schedule": schedule_result["schedule"],
            "pair_count": schedule_result["scheduled_pair_count"],
            "batch_count": len(batch_results),
            "playoff_ids": schedule_result.get("playoff_ids", []),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        write_json(out, result)
        write_json(run_dir / "run.json", result)
        return result
    if failed_batches:
        blocked_reason = next(
            (
                str(row.get("blocked_reason") or row.get("error"))
                for row in failed_batches
                if row.get("blocking_provider_error")
            ),
            None,
        )
        result = {
            **run_payload,
            "status": "partial",
            "ok_batch_count": len(batch_results) - len(failed_batches),
            "failed_batch_count": len(failed_batches),
            "attempted_batch_count": len(batch_results),
            "estimated_skipped_batch_count": max(0, batch_count_estimate - len(batch_results)),
            "blocked_reason": blocked_reason,
            "failed_batches": failed_batches,
            "match_count": len(matches),
            "pair_count": schedule_result["scheduled_pair_count"],
            "batch_count": len(batch_results),
            "schedule": schedule_result["schedule"],
            "playoff_ids": schedule_result.get("playoff_ids", []),
            "usage": sum_usage(row.get("usage") for row in batch_results),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        write_json(out, result)
        write_json(run_dir / "run.json", result)
        return result

    payload = build_tournament_gold_payload(
        seed_payload=seed_payload,
        seed_ranked=seed_ranked,
        tournament_ids=tournament_ids,
        tournament_stats=tournament_stats,
        config=config,
        card_paths=card_paths,
        seed_ranking_path=seed_ranking_path,
        run_dir=run_dir,
        matches=matches,
        batch_results=batch_results,
        elapsed_seconds=round(time.monotonic() - started, 3),
        schedule=schedule_result["schedule"],
        playoff_ids=schedule_result.get("playoff_ids", []),
    )
    write_json(out, payload)
    result = {
        **run_payload,
        "status": "ok",
        "ok_batch_count": len(batch_results),
        "failed_batch_count": 0,
        "match_count": len(matches),
        "pair_count": schedule_result["scheduled_pair_count"],
        "batch_count": len(batch_results),
        "playoff_ids": schedule_result.get("playoff_ids", []),
        "top_ranked_ids": tournament_order_ids[: min(20, len(tournament_order_ids))],
        "usage": sum_usage(row.get("usage") for row in batch_results),
        "elapsed_seconds": payload["tournament_summary"]["elapsed_seconds"],
    }
    write_json(run_dir / "run.json", result)
    return result


def run_tournament_schedule(
    *,
    tournament_ids: list[str],
    rows_by_paper: dict[str, list[dict[str, Any]]],
    seed_ranked: list[dict[str, Any]],
    seed_rank_by_id: dict[str, int],
    run_dir: Path,
    config: FrontierTournamentConfig,
    client: ChatCompletionClient | None,
    batch_count_estimate: int,
) -> dict[str, Any]:
    if config.strategy == "all_pairs":
        return run_all_pairs_schedule(
            tournament_ids=tournament_ids,
            rows_by_paper=rows_by_paper,
            seed_ranked=seed_ranked,
            seed_rank_by_id=seed_rank_by_id,
            run_dir=run_dir,
            config=config,
            client=client,
        )
    return run_swiss_schedule(
        tournament_ids=tournament_ids,
        rows_by_paper=rows_by_paper,
        seed_ranked=seed_ranked,
        seed_rank_by_id=seed_rank_by_id,
        run_dir=run_dir,
        config=config,
        client=client,
        batch_count_estimate=batch_count_estimate,
    )


def run_all_pairs_schedule(
    *,
    tournament_ids: list[str],
    rows_by_paper: dict[str, list[dict[str, Any]]],
    seed_ranked: list[dict[str, Any]],
    seed_rank_by_id: dict[str, int],
    run_dir: Path,
    config: FrontierTournamentConfig,
    client: ChatCompletionClient | None,
) -> dict[str, Any]:
    pairs = randomize_tournament_pairs(build_tournament_pairs(tournament_ids), seed=config.seed or 0)
    pair_batches = batch_tournament_pairs(pairs, config.pairs_per_batch)
    batch_results, matches = run_pair_batches(
        pair_batches=pair_batches,
        rows_by_paper=rows_by_paper,
        seed_ranked=seed_ranked,
        start_batch_index=0,
        batch_count=len(pair_batches),
        run_dir=run_dir,
        config=config,
        client=client,
    )
    tournament_stats = aggregate_tournament_matches(
        matches,
        tournament_ids=tournament_ids,
        seed_rank_by_id=seed_rank_by_id,
    )
    return {
        "batch_results": batch_results,
        "matches": matches,
        "tournament_stats": tournament_stats,
        "scheduled_pair_count": len(pairs),
        "schedule": [
            {
                "stage": "all_pairs",
                "pair_count": len(pairs),
                "batch_count": len(pair_batches),
            }
        ],
        "playoff_ids": [],
    }


def run_swiss_schedule(
    *,
    tournament_ids: list[str],
    rows_by_paper: dict[str, list[dict[str, Any]]],
    seed_ranked: list[dict[str, Any]],
    seed_rank_by_id: dict[str, int],
    run_dir: Path,
    config: FrontierTournamentConfig,
    client: ChatCompletionClient | None,
    batch_count_estimate: int,
) -> dict[str, Any]:
    batch_results: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    standings_matches: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []
    played_pair_keys: set[frozenset[str]] = set()
    standings_ids = list(tournament_ids)
    batch_index = 0

    for round_index in range(config.swiss_rounds):
        round_pairs = build_swiss_round_pairs(standings_ids, played_pair_keys)
        round_pairs = randomize_tournament_pairs(
            round_pairs,
            seed=(config.seed or 0) + round_index,
        )
        pair_batches = batch_tournament_pairs(round_pairs, config.pairs_per_batch)
        round_batch_results, round_matches = run_pair_batches(
            pair_batches=pair_batches,
            rows_by_paper=rows_by_paper,
            seed_ranked=seed_ranked,
            start_batch_index=batch_index,
            batch_count=batch_count_estimate,
            run_dir=run_dir,
            config=config,
            client=client,
        )
        batch_index += len(pair_batches)
        batch_results.extend(round_batch_results)
        matches.extend(round_matches)
        schedule.append(
            {
                "stage": "swiss",
                "round": round_index + 1,
                "pair_count": len(round_pairs),
                "batch_count": len(pair_batches),
                "standings_before_round": standings_ids,
            }
        )
        if any(row.get("status") not in {"ok", "dry_run"} for row in round_batch_results):
            break
        standings_input = matches
        if config.dry_run:
            standings_matches.extend(synthetic_seed_matches(round_pairs, seed_rank_by_id=seed_rank_by_id))
            standings_input = standings_matches
        standings = aggregate_tournament_matches(
            standings_input,
            tournament_ids=tournament_ids,
            seed_rank_by_id=seed_rank_by_id,
        )
        standings_ids = [str(row.get("paper_id")) for row in standings]

    playoff_ids: list[str] = []
    if (
        config.strategy == "swiss_playoff"
        and not config.dry_run
        and not any(row.get("status") not in {"ok", "dry_run"} for row in batch_results)
    ):
        swiss_stats = aggregate_tournament_matches(
            matches,
            tournament_ids=tournament_ids,
            seed_rank_by_id=seed_rank_by_id,
        )
        playoff_ids = [str(row.get("paper_id")) for row in swiss_stats[: config.playoff_top_n]]
        playoff_pairs = [
            pair
            for pair in build_tournament_pairs(playoff_ids)
            if frozenset(pair) not in {frozenset((str(match.get("paper_a")), str(match.get("paper_b")))) for match in matches}
        ]
        playoff_pairs = randomize_tournament_pairs(playoff_pairs, seed=(config.seed or 0) + 10_000)
        pair_batches = batch_tournament_pairs(playoff_pairs, config.pairs_per_batch)
        playoff_batch_results, playoff_matches = run_pair_batches(
            pair_batches=pair_batches,
            rows_by_paper=rows_by_paper,
            seed_ranked=seed_ranked,
            start_batch_index=batch_index,
            batch_count=batch_count_estimate,
            run_dir=run_dir,
            config=config,
            client=client,
        )
        batch_results.extend(playoff_batch_results)
        matches.extend(playoff_matches)
        schedule.append(
            {
                "stage": "playoff_all_pairs",
                "playoff_top_n": config.playoff_top_n,
                "playoff_ids": playoff_ids,
                "pair_count": len(playoff_pairs),
                "batch_count": len(pair_batches),
            }
        )
    elif config.strategy == "swiss_playoff" and config.dry_run:
        playoff_ids = standings_ids[: config.playoff_top_n]
        playoff_pairs = [
            pair for pair in build_tournament_pairs(playoff_ids) if frozenset(pair) not in played_pair_keys
        ]
        playoff_pairs = randomize_tournament_pairs(playoff_pairs, seed=(config.seed or 0) + 10_000)
        pair_batches = batch_tournament_pairs(playoff_pairs, config.pairs_per_batch)
        playoff_batch_results, _ = run_pair_batches(
            pair_batches=pair_batches,
            rows_by_paper=rows_by_paper,
            seed_ranked=seed_ranked,
            start_batch_index=batch_index,
            batch_count=batch_count_estimate,
            run_dir=run_dir,
            config=config,
            client=client,
        )
        batch_results.extend(playoff_batch_results)
        schedule.append(
            {
                "stage": "playoff_all_pairs",
                "playoff_top_n": config.playoff_top_n,
                "playoff_ids": playoff_ids,
                "pair_count": len(playoff_pairs),
                "batch_count": len(pair_batches),
                "dry_run_note": "Dry run uses seed-order synthetic match results to plan Swiss standings.",
            }
        )

    tournament_stats = aggregate_sparse_or_hybrid_stats(
        matches,
        tournament_ids=tournament_ids,
        seed_rank_by_id=seed_rank_by_id,
        playoff_ids=playoff_ids,
        strategy=config.strategy,
    )
    return {
        "batch_results": batch_results,
        "matches": matches,
        "tournament_stats": tournament_stats,
        "scheduled_pair_count": scheduled_pair_count(schedule),
        "schedule": schedule,
        "playoff_ids": playoff_ids,
    }


def run_pair_batches(
    *,
    pair_batches: list[list[tuple[str, str]]],
    rows_by_paper: dict[str, list[dict[str, Any]]],
    seed_ranked: list[dict[str, Any]],
    start_batch_index: int,
    batch_count: int,
    run_dir: Path,
    config: FrontierTournamentConfig,
    client: ChatCompletionClient | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    batch_results: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    for offset, pairs in enumerate(pair_batches):
        batch_result = run_frontier_tournament_batch(
            pairs=pairs,
            rows_by_paper=rows_by_paper,
            seed_ranked=seed_ranked,
            batch_index=start_batch_index + offset,
            batch_count=max(batch_count, start_batch_index + len(pair_batches)),
            run_dir=run_dir,
            config=config,
            client=client,
        )
        batch_results.append(batch_result)
        if batch_result.get("status") == "ok":
            parsed = batch_result.get("parsed")
            if isinstance(parsed, dict) and isinstance(parsed.get("matches"), list):
                matches.extend(match for match in parsed["matches"] if isinstance(match, dict))
        if batch_result.get("blocking_provider_error"):
            break
    return batch_results, matches


def run_frontier_tournament_batch(
    *,
    pairs: list[tuple[str, str]],
    rows_by_paper: dict[str, list[dict[str, Any]]],
    seed_ranked: list[dict[str, Any]],
    batch_index: int,
    batch_count: int,
    run_dir: Path,
    config: FrontierTournamentConfig,
    client: ChatCompletionClient | None,
) -> dict[str, Any]:
    batch_dir = run_dir / "batches" / f"batch_{batch_index:03d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    parsed_path = batch_dir / "parsed.json"
    expected = {pair_id_for(a, b): (a, b) for a, b in pairs}
    messages = build_frontier_tournament_messages(
        pairs=pairs,
        rows_by_paper=rows_by_paper,
        seed_ranked=seed_ranked,
        batch_index=batch_index,
        batch_count=batch_count,
        config=config,
    )
    prompt_path = batch_dir / "prompt.json"
    prompt_payload = {
        "prompt_version": config.prompt_version,
        "provider": config.provider,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "paper_set_name": config.paper_set_name,
        "batch_index": batch_index,
        "batch_count": batch_count,
        "pair_count": len(pairs),
        "pairs": [{"pair_id": pair_id_for(a, b), "paper_a": a, "paper_b": b} for a, b in pairs],
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
        "seed": config.seed,
        "messages": messages,
        "prompt_tokens_estimate": sum(estimate_tokens(_message_text_for_estimate(message)) for message in messages),
    }
    fingerprint = tournament_prompt_fingerprint(prompt_payload)
    prompt_payload["fingerprint"] = fingerprint
    cache_reuse_error: str | None = None
    if parsed_path.exists() and not config.overwrite:
        try:
            existing_prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
            if not isinstance(existing_prompt, dict):
                raise ValueError("cached prompt is not a JSON object")
            existing_fingerprint = str(existing_prompt.get("fingerprint") or "")
            if existing_fingerprint:
                if existing_fingerprint != fingerprint:
                    raise ValueError("cached prompt fingerprint does not match this request")
            else:
                legacy_prompt = {
                    key: value
                    for key, value in existing_prompt.items()
                    if key != "fingerprint"
                }
                expected_legacy_prompt = {
                    key: prompt_payload.get(key)
                    for key in legacy_prompt
                }
                if legacy_prompt != expected_legacy_prompt:
                    raise ValueError("cached legacy prompt does not match this request")
            parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
            parsed = normalize_tournament_batch(parsed, expected_pairs=expected)
            validation_errors = validate_tournament_batch(parsed, expected_pairs=expected)
            if not validation_errors:
                write_json(prompt_path, prompt_payload)
                cached_result = {
                    "status": "ok",
                    "batch_index": batch_index,
                    "batch_count": batch_count,
                    "pair_count": len(pairs),
                    "fingerprint": fingerprint,
                    "prompt_path": str(prompt_path),
                    "parsed_response_path": str(parsed_path),
                    "validation_errors": [],
                    "parsed": parsed,
                    "resumed_from": str(parsed_path),
                }
                run_state_path = batch_dir / "run.json"
                if run_state_path.exists():
                    run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
                    if isinstance(run_state.get("usage"), dict):
                        cached_result["usage"] = run_state["usage"]
                    if run_state.get("served_model"):
                        cached_result["served_model"] = run_state["served_model"]
                return cached_result
            cache_reuse_error = "; ".join(validation_errors)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            cache_reuse_error = repr(exc)

    write_json(prompt_path, prompt_payload)
    base_result: dict[str, Any] = {
        "status": "dry_run" if config.dry_run else "pending",
        "batch_index": batch_index,
        "batch_count": batch_count,
        "pair_count": len(pairs),
        "fingerprint": fingerprint,
        "prompt_path": str(prompt_path),
        "raw_response_path": None,
        "parsed_response_path": None,
        "validation_errors": [],
        "cache_reuse_error": cache_reuse_error,
    }
    if config.dry_run:
        write_json(batch_dir / "run.json", base_result)
        return base_result
    if client is None:
        raise ValueError("client is required unless dry_run=True")
    chat = None
    try:
        chat = client.complete(
            messages=messages,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            seed=config.seed,
            response_format={"type": "json_object"},
        )
        raw_response_path = batch_dir / "response.json"
        write_json(raw_response_path, chat.response)
        parsed = parse_json_response(chat.content)
        parsed = normalize_tournament_batch(parsed, expected_pairs=expected)
        validation_errors = validate_tournament_batch(parsed, expected_pairs=expected)
        write_json(parsed_path, parsed)
        result = {
            **base_result,
            "status": "ok" if not validation_errors else "validation_error",
            "raw_response_path": str(raw_response_path),
            "parsed_response_path": str(parsed_path),
            "validation_errors": validation_errors,
            "served_model": chat.served_model,
            "usage": chat.usage,
            "provider_elapsed_seconds": round(chat.elapsed_seconds, 3),
            "parsed": parsed,
        }
    except Exception as exc:  # noqa: BLE001 - batch-local persistence enables retry.
        response = getattr(exc, "response", None)
        raw_response_path = batch_dir / "response.json"
        if isinstance(response, dict) and not raw_response_path.exists():
            write_json(raw_response_path, response)
        usage = chat.usage if chat is not None else getattr(exc, "usage", {})
        served_model = chat.served_model if chat is not None else getattr(exc, "served_model", None)
        error_path = batch_dir / "error.json"
        write_json(error_path, {"error": repr(exc)})
        result = {
            **base_result,
            "status": "failed",
            "error_path": str(error_path),
            "error": repr(exc),
            "blocking_provider_error": is_batch_blocking_provider_error(exc),
        }
        if result["blocking_provider_error"]:
            result["blocked_reason"] = repr(exc)
        if raw_response_path.exists():
            result["raw_response_path"] = str(raw_response_path)
        if isinstance(usage, dict) and usage:
            result["usage"] = usage
        if served_model:
            result["served_model"] = served_model
    write_json(batch_dir / "run.json", {key: value for key, value in result.items() if key != "parsed"})
    return result


def tournament_prompt_fingerprint(payload: dict[str, Any]) -> str:
    canonical = {key: value for key, value in payload.items() if key != "fingerprint"}
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_frontier_tournament_messages(
    *,
    pairs: list[tuple[str, str]],
    rows_by_paper: dict[str, list[dict[str, Any]]],
    seed_ranked: list[dict[str, Any]],
    batch_index: int,
    batch_count: int,
    config: FrontierTournamentConfig,
) -> list[dict[str, Any]]:
    system = """You are a frontier pairwise judge finalizing an ICML paper impact ranking.

You are comparing papers using independent PDF-aware evidence cards created by frontier judges after each saw the configured first-page paper PDF excerpt plus extracted main-paper text. Treat those cards as evidence. Do not use web search, author identity, institution, citation memory, or outside knowledge.
No reviews, reviewer scores, area-chair comments, decisions, presentation tiers, or awards are included in the cards. Do not infer paper quality from possible venue or outcome cues.

For every pair, choose the paper with the stronger expected broad scientific and ML-field impact, using technical soundness and evidence quality as gates. Prefer concrete durable contributions over paper polish. Benchmarks, datasets, infrastructure, safety/eval resources, scientific tools, theory, and algorithms can all be highly impactful, but their impact route should be explicit and credible.

Return only valid JSON."""
    seed_by_id = {str(row.get("paper_id") or ""): row for row in seed_ranked if str(row.get("paper_id") or "")}
    batch_ids = sorted({paper_id for pair in pairs for paper_id in pair})
    random.Random((config.seed or 0) + batch_index).shuffle(batch_ids)
    paper_blocks = "\n\n".join(
        format_tournament_paper_block(paper_id, rows_by_paper[paper_id], seed_item=seed_by_id.get(paper_id, {}))
        for paper_id in batch_ids
    )
    pair_lines = "\n".join(
        f"- pair_id={pair_id_for(a, b)} | paper_a={a} | paper_b={b}"
        for a, b in pairs
    )
    user = f"""Adjudicate tournament batch {batch_index + 1} of {batch_count} for {config.paper_set_name}.

Evaluate each pair independently. A "tie" is allowed only when the expected impact difference is genuinely indistinguishable from the card evidence. For close but real differences, choose a winner with impact_delta="small".

Required JSON shape:
{{
  "evaluation_mode": "frontier_pdf_card_pairwise_tournament_batch",
  "prompt_version": "{config.prompt_version}",
  "judge": "{config.provider}:{config.model}",
  "paper_set": "{config.paper_set_name}",
  "batch_index": {batch_index},
  "matches": [
    {{
      "pair_id": "paperA__paperB",
      "paper_a": "paper id",
      "paper_b": "paper id",
      "winner": "paper_a",
      "winner_paper_id": "paper id",
      "confidence": 0.72,
      "impact_delta": "small",
      "reason": "One concise comparative reason grounded in the card evidence."
    }}
  ]
}}

Winner rules:
- winner must be one of: "paper_a", "paper_b", "tie".
- winner_paper_id must equal paper_a, paper_b, or "" for a tie.
- confidence must be between 0 and 1.
- impact_delta must be one of: "large", "medium", "small", "tie".
- Return exactly these {len(pairs)} pair decisions and no others:
{pair_lines}

PDF-aware evidence cards:
{paper_blocks}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def format_tournament_paper_block(paper_id: str, rows: list[dict[str, Any]], *, seed_item: dict[str, Any]) -> str:
    cards = [row.get("card") for row in rows if isinstance(row.get("card"), dict)]
    title = first_nonempty([*(card.get("title") for card in cards), seed_item.get("title")])
    primary_class = majority(
        str(card.get("primary_contribution_class") or "other")
        for card in cards
        if isinstance(card, dict)
    )
    lines = [f"<PAPER id=\"{paper_id}\">", f"Title: {title}", f"Contribution class: {primary_class}"]
    for card in cards:
        scores = card.get("scores") if isinstance(card.get("scores"), dict) else {}
        lines.extend(
            [
                (
                    "Judge {judge}: overall={overall} broad={broad} ml={ml} soundness={soundness} "
                    "novelty={novelty} evidence={evidence} visual={visual}"
                ).format(
                    judge=truncate_inline(str(card.get("judge") or "unknown"), 72),
                    overall=scores.get("overall_gold_priority_score"),
                    broad=scores.get("broad_scientific_impact_score"),
                    ml=scores.get("ml_field_impact_score"),
                    soundness=scores.get("technical_soundness_score"),
                    novelty=scores.get("novelty_score"),
                    evidence=scores.get("evidence_confidence_score"),
                    visual=scores.get("visual_evidence_importance_score"),
                ),
                f"  Summary: {truncate_inline(card.get('one_sentence_summary'), 260)}",
                f"  Best case: {truncate_inline(nested_get(card, 'impact_assessment', 'best_case_for_impact'), 320)}",
                f"  Main risk: {truncate_inline(nested_get(card, 'impact_assessment', 'main_risk'), 260)}",
                f"  Top-10 case: {truncate_inline(nested_get(card, 'ranking_hooks', 'top_10_case'), 320)}",
                f"  Visual/table effect: {truncate_inline(nested_get(card, 'evidence_from_pdf', 'visual_or_tabular_evidence_changes_judgment'), 260)}",
            ]
        )
    lines.append("</PAPER>")
    return "\n".join(str(line) for line in lines)


def build_tournament_pairs(paper_ids: list[str]) -> list[tuple[str, str]]:
    return list(itertools.combinations(paper_ids, 2))


def randomize_tournament_pairs(
    pairs: list[tuple[str, str]],
    *,
    seed: int,
) -> list[tuple[str, str]]:
    """Randomize pair batching and A/B presentation without changing the schedule."""
    rng = random.Random(seed)
    randomized = list(pairs)
    rng.shuffle(randomized)
    return [
        (paper_b, paper_a) if rng.random() < 0.5 else (paper_a, paper_b)
        for paper_a, paper_b in randomized
    ]


def build_swiss_round_pairs(standings_ids: list[str], played_pair_keys: set[frozenset[str]]) -> list[tuple[str, str]]:
    if len(set(standings_ids)) != len(standings_ids):
        raise ValueError("Swiss standings contain duplicate paper IDs")

    position = {paper_id: index for index, paper_id in enumerate(standings_ids)}
    failed_states: set[frozenset[str]] = set()

    def find_matching(remaining: tuple[str, ...]) -> list[tuple[str, str]] | None:
        if not remaining:
            return []
        state = frozenset(remaining)
        if state in failed_states:
            return None

        available_by_paper = {
            paper_id: [
                candidate_id
                for candidate_id in remaining
                if candidate_id != paper_id
                and frozenset((paper_id, candidate_id)) not in played_pair_keys
            ]
            for paper_id in remaining
        }
        paper_id = min(
            remaining,
            key=lambda value: (
                len(available_by_paper[value]),
                position[value],
            ),
        )
        opponents = sorted(
            available_by_paper[paper_id],
            key=lambda value: (
                abs(position[value] - position[paper_id]),
                position[value],
            ),
        )
        for opponent_id in opponents:
            next_remaining = tuple(
                value
                for value in remaining
                if value not in {paper_id, opponent_id}
            )
            suffix = find_matching(next_remaining)
            if suffix is not None:
                return [(paper_id, opponent_id), *suffix]

        failed_states.add(state)
        return None

    if len(standings_ids) % 2 == 0:
        pairs = find_matching(tuple(standings_ids))
    else:
        pairs = None
        for bye_id in reversed(standings_ids):
            remaining = tuple(
                paper_id for paper_id in standings_ids if paper_id != bye_id
            )
            pairs = find_matching(remaining)
            if pairs is not None:
                break

    if pairs is None:
        raise ValueError(
            "Swiss scheduler could not construct a complete round without repeated pairs"
        )
    expected_pair_count = len(standings_ids) // 2
    if len(pairs) != expected_pair_count:
        raise ValueError(
            f"Swiss scheduler produced {len(pairs)} of {expected_pair_count} required pairs"
        )
    played_pair_keys.update(frozenset(pair) for pair in pairs)
    return pairs


def synthetic_seed_matches(
    pairs: list[tuple[str, str]],
    *,
    seed_rank_by_id: dict[str, int],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for paper_a, paper_b in pairs:
        winner_paper_id = paper_a if seed_rank_by_id.get(paper_a, 10**9) <= seed_rank_by_id.get(paper_b, 10**9) else paper_b
        matches.append(
            {
                "pair_id": pair_id_for(paper_a, paper_b),
                "paper_a": paper_a,
                "paper_b": paper_b,
                "winner": "paper_a" if winner_paper_id == paper_a else "paper_b",
                "winner_paper_id": winner_paper_id,
                "confidence": 0.5,
                "impact_delta": "small",
                "reason": "synthetic seed-order result for dry-run schedule planning",
            }
        )
    return matches


def batch_tournament_pairs(pairs: list[tuple[str, str]], pairs_per_batch: int) -> list[list[tuple[str, str]]]:
    if pairs_per_batch < 1:
        raise ValueError("pairs_per_batch must be positive")
    return [pairs[index : index + pairs_per_batch] for index in range(0, len(pairs), pairs_per_batch)]


def pair_id_for(paper_a: str, paper_b: str) -> str:
    return f"{paper_a}__{paper_b}"


def estimate_tournament_batch_count(config: FrontierTournamentConfig) -> int:
    if config.strategy == "all_pairs":
        pair_count = config.top_n * (config.top_n - 1) // 2
    else:
        swiss_pairs = config.swiss_rounds * (config.top_n // 2)
        playoff_pairs = 0
        if config.strategy == "swiss_playoff" and config.playoff_top_n:
            playoff_pairs = config.playoff_top_n * (config.playoff_top_n - 1) // 2
        pair_count = swiss_pairs + playoff_pairs
    return math.ceil(pair_count / config.pairs_per_batch) if pair_count else 0


def scheduled_pair_count(schedule: list[dict[str, Any]]) -> int:
    return sum(int(number(row.get("pair_count"))) for row in schedule)


def normalize_tournament_batch(
    parsed: dict[str, Any],
    *,
    expected_pairs: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    matches = parsed.get("matches")
    if not isinstance(matches, list):
        return parsed
    normalized: list[dict[str, Any]] = []
    expected_by_unordered = {frozenset((a, b)): (pair_id, a, b) for pair_id, (a, b) in expected_pairs.items()}
    for raw_match in matches:
        if not isinstance(raw_match, dict):
            continue
        match = dict(raw_match)
        pair_id = str(match.get("pair_id") or "")
        paper_a = str(match.get("paper_a") or "")
        paper_b = str(match.get("paper_b") or "")
        if pair_id not in expected_pairs and paper_a and paper_b:
            expected = expected_by_unordered.get(frozenset((paper_a, paper_b)))
            if expected:
                pair_id, canonical_a, canonical_b = expected
                match["pair_id"] = pair_id
                match["paper_a"] = canonical_a
                match["paper_b"] = canonical_b
                paper_a, paper_b = canonical_a, canonical_b
        if pair_id in expected_pairs:
            paper_a, paper_b = expected_pairs[pair_id]
            match["pair_id"] = pair_id
            match["paper_a"] = paper_a
            match["paper_b"] = paper_b

        winner = str(match.get("winner") or "").strip().lower()
        winner_paper_id = str(match.get("winner_paper_id") or "").strip()
        if winner_paper_id == paper_a:
            winner = "paper_a"
        elif winner_paper_id == paper_b:
            winner = "paper_b"
        elif winner in {"a", "paper a", "paper_a", "first"}:
            winner = "paper_a"
            winner_paper_id = paper_a
        elif winner in {"b", "paper b", "paper_b", "second"}:
            winner = "paper_b"
            winner_paper_id = paper_b
        elif winner in {"tie", "draw", "equal"}:
            winner = "tie"
            winner_paper_id = ""
        match["winner"] = winner
        match["winner_paper_id"] = winner_paper_id
        match["confidence"] = clamp(number(match.get("confidence")), 0.0, 1.0)
        impact_delta = str(match.get("impact_delta") or "").strip().lower()
        if impact_delta not in {"large", "medium", "small", "tie"}:
            impact_delta = "tie" if winner == "tie" else "small"
        if winner == "tie":
            impact_delta = "tie"
        match["impact_delta"] = impact_delta
        normalized.append(match)
    return {**parsed, "matches": normalized}


def validate_tournament_batch(
    parsed: dict[str, Any],
    *,
    expected_pairs: dict[str, tuple[str, str]],
) -> list[str]:
    matches = parsed.get("matches")
    if not isinstance(matches, list):
        return ["matches must be a list"]
    errors: list[str] = []
    pair_ids = [str(match.get("pair_id") or "") for match in matches if isinstance(match, dict)]
    expected_set = set(expected_pairs)
    actual_set = set(pair_ids)
    missing = sorted(expected_set - actual_set)
    extras = sorted(actual_set - expected_set)
    duplicates = sorted({pair_id for pair_id in pair_ids if pair_ids.count(pair_id) > 1})
    if missing:
        errors.append(f"missing pair_ids: {missing}")
    if extras:
        errors.append(f"extra pair_ids: {extras}")
    if duplicates:
        errors.append(f"duplicate pair_ids: {duplicates}")
    for match in matches:
        if not isinstance(match, dict):
            errors.append("each match must be an object")
            continue
        pair_id = str(match.get("pair_id") or "")
        if pair_id not in expected_pairs:
            continue
        paper_a, paper_b = expected_pairs[pair_id]
        if str(match.get("paper_a") or "") != paper_a:
            errors.append(f"{pair_id}: paper_a mismatch")
        if str(match.get("paper_b") or "") != paper_b:
            errors.append(f"{pair_id}: paper_b mismatch")
        winner = str(match.get("winner") or "")
        winner_paper_id = str(match.get("winner_paper_id") or "")
        if winner not in {"paper_a", "paper_b", "tie"}:
            errors.append(f"{pair_id}: winner must be paper_a, paper_b, or tie")
        elif winner == "paper_a" and winner_paper_id != paper_a:
            errors.append(f"{pair_id}: winner_paper_id must equal paper_a")
        elif winner == "paper_b" and winner_paper_id != paper_b:
            errors.append(f"{pair_id}: winner_paper_id must equal paper_b")
        elif winner == "tie" and winner_paper_id not in {"", paper_a, paper_b}:
            errors.append(f"{pair_id}: tie winner_paper_id must be empty or one of the pair ids")
        confidence = number(match.get("confidence"))
        if confidence < 0 or confidence > 1:
            errors.append(f"{pair_id}: confidence must be 0..1")
        if str(match.get("impact_delta") or "") not in {"large", "medium", "small", "tie"}:
            errors.append(f"{pair_id}: invalid impact_delta")
    return errors


def aggregate_tournament_matches(
    matches: list[dict[str, Any]],
    *,
    tournament_ids: list[str],
    seed_rank_by_id: dict[str, int],
) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {
        paper_id: {
            "paper_id": paper_id,
            "pairwise_points": 0.0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "match_count": 0,
            "confidence_margin": 0.0,
            "large_wins": 0,
            "medium_wins": 0,
            "small_wins": 0,
            "large_losses": 0,
            "medium_losses": 0,
            "small_losses": 0,
            "seed_synthesis_rank": seed_rank_by_id.get(paper_id, 10**9),
        }
        for paper_id in tournament_ids
    }
    delta_weight = {"large": 1.0, "medium": 0.67, "small": 0.33, "tie": 0.0}
    for match in matches:
        pair_id = str(match.get("pair_id") or "")
        paper_a = str(match.get("paper_a") or "")
        paper_b = str(match.get("paper_b") or "")
        if paper_a not in stats or paper_b not in stats:
            continue
        confidence = clamp(number(match.get("confidence")), 0.0, 1.0)
        impact_delta = str(match.get("impact_delta") or "small")
        margin = confidence * delta_weight.get(impact_delta, 0.33)
        winner = str(match.get("winner") or "")
        for paper_id in [paper_a, paper_b]:
            stats[paper_id]["match_count"] += 1
        if winner == "tie":
            stats[paper_a]["pairwise_points"] += 0.5
            stats[paper_b]["pairwise_points"] += 0.5
            stats[paper_a]["ties"] += 1
            stats[paper_b]["ties"] += 1
            continue
        winner_id = paper_a if winner == "paper_a" else paper_b if winner == "paper_b" else ""
        loser_id = paper_b if winner == "paper_a" else paper_a if winner == "paper_b" else ""
        if not winner_id or not loser_id:
            continue
        if impact_delta == "tie":
            impact_delta = "small"
            margin = confidence * delta_weight[impact_delta]
        stats[winner_id]["pairwise_points"] += 1.0
        stats[winner_id]["wins"] += 1
        stats[loser_id]["losses"] += 1
        stats[winner_id]["confidence_margin"] += margin
        stats[loser_id]["confidence_margin"] -= margin
        stats[winner_id][f"{impact_delta}_wins"] += 1
        stats[loser_id][f"{impact_delta}_losses"] += 1
        stats[winner_id].setdefault("notable_wins", []).append({"opponent": loser_id, "pair_id": pair_id, "impact_delta": impact_delta})
        stats[loser_id].setdefault("notable_losses", []).append({"opponent": winner_id, "pair_id": pair_id, "impact_delta": impact_delta})

    for item in stats.values():
        match_count = max(number(item.get("match_count")), 1.0)
        item["pairwise_point_rate"] = number(item.get("pairwise_points")) / match_count
        item["confidence_margin_rate"] = number(item.get("confidence_margin")) / match_count

    ordered = sorted(
        stats.values(),
        key=lambda item: (
            -number(item.get("pairwise_point_rate")),
            -number(item.get("pairwise_points")),
            -number(item.get("wins")),
            -number(item.get("confidence_margin_rate")),
            number(item.get("seed_synthesis_rank")),
            str(item.get("paper_id")),
        ),
    )
    for rank, item in enumerate(ordered, start=1):
        item["tournament_rank"] = rank
        item["pairwise_points"] = round(number(item.get("pairwise_points")), 3)
        item["pairwise_point_rate"] = round(number(item.get("pairwise_point_rate")), 4)
        item["confidence_margin"] = round(number(item.get("confidence_margin")), 3)
        item["confidence_margin_rate"] = round(number(item.get("confidence_margin_rate")), 4)
        for key in ["notable_wins", "notable_losses"]:
            if isinstance(item.get(key), list):
                item[key] = item[key][:5]
    return ordered


def aggregate_sparse_or_hybrid_stats(
    matches: list[dict[str, Any]],
    *,
    tournament_ids: list[str],
    seed_rank_by_id: dict[str, int],
    playoff_ids: list[str],
    strategy: str,
) -> list[dict[str, Any]]:
    if strategy != "swiss_playoff" or not playoff_ids:
        return aggregate_tournament_matches(
            matches,
            tournament_ids=tournament_ids,
            seed_rank_by_id=seed_rank_by_id,
        )
    playoff_set = set(playoff_ids)
    playoff_matches = [
        match
        for match in matches
        if str(match.get("paper_a")) in playoff_set and str(match.get("paper_b")) in playoff_set
    ]
    playoff_stats = aggregate_tournament_matches(
        playoff_matches,
        tournament_ids=playoff_ids,
        seed_rank_by_id=seed_rank_by_id,
    )
    swiss_stats = aggregate_tournament_matches(
        matches,
        tournament_ids=tournament_ids,
        seed_rank_by_id=seed_rank_by_id,
    )
    swiss_by_id = {str(item.get("paper_id")): item for item in swiss_stats}
    remaining_stats = [dict(swiss_by_id[paper_id]) for paper_id in tournament_ids if paper_id not in playoff_set]
    remaining_stats.sort(
        key=lambda item: (
            -number(item.get("pairwise_point_rate")),
            -number(item.get("pairwise_points")),
            -number(item.get("wins")),
            number(item.get("seed_synthesis_rank")),
            str(item.get("paper_id")),
        )
    )
    ordered = [dict(item, ranking_source_stage="playoff_all_pairs") for item in playoff_stats] + [
        dict(item, ranking_source_stage="swiss_only") for item in remaining_stats
    ]
    for rank, item in enumerate(ordered, start=1):
        item["tournament_rank"] = rank
    return ordered


def build_tournament_gold_payload(
    *,
    seed_payload: dict[str, Any],
    seed_ranked: list[dict[str, Any]],
    tournament_ids: list[str],
    tournament_stats: list[dict[str, Any]],
    config: FrontierTournamentConfig,
    card_paths: list[Path],
    seed_ranking_path: Path,
    run_dir: Path,
    matches: list[dict[str, Any]],
    batch_results: list[dict[str, Any]],
    elapsed_seconds: float,
    schedule: list[dict[str, Any]],
    playoff_ids: list[str],
) -> dict[str, Any]:
    seed_items = [dict(row) for row in sorted(seed_ranked, key=lambda row: number(row.get("rank")))]
    seed_by_id = {str(row.get("paper_id") or ""): row for row in seed_items}
    stats_by_id = {str(item.get("paper_id") or ""): item for item in tournament_stats}
    tournament_order = [str(item.get("paper_id")) for item in tournament_stats]
    remaining_order = [str(row.get("paper_id") or "") for row in seed_items if str(row.get("paper_id") or "") not in tournament_ids]
    final_order = tournament_order + [paper_id for paper_id in remaining_order if paper_id]
    ranked_papers: list[dict[str, Any]] = []
    for rank, paper_id in enumerate(final_order, start=1):
        item = dict(seed_by_id[paper_id])
        seed_rank = int(number(item.get("rank")))
        item["rank"] = rank
        item["seed_synthesis_rank"] = seed_rank
        if paper_id in stats_by_id:
            item["tournament_stats"] = stats_by_id[paper_id]
            item["ranking_source"] = "pairwise_tournament"
        else:
            item["ranking_source"] = "seed_synthesis_outside_tournament_pool"
        ranked_papers.append(item)
    usage = sum_usage(row.get("usage") for row in batch_results)
    payload = {
        "evaluation_mode": "frontier_pdf_card_pairwise_tournament_ranking",
        "prompt_version": config.prompt_version,
        "judge": f"{config.provider}:{config.model}",
        "paper_set": config.paper_set_name,
        "modality": "pairwise_tournament_over_pdf_aware_frontier_cards",
        "tournament_strategy": config.strategy,
        "source_seed": {
            "path": str(seed_ranking_path),
            "evaluation_mode": seed_payload.get("evaluation_mode"),
            "prompt_version": seed_payload.get("prompt_version"),
            "note": (
                "Seed ranking selects the tournament pool and orders papers outside it. "
                "Pair batching, A/B orientation, and evidence-block order are separately "
                "and deterministically randomized."
            ),
        },
        "card_paths": [str(path) for path in card_paths],
        "ranked_papers": ranked_papers,
        "category_rankings": build_category_rankings(ranked_papers),
        "tournament_summary": {
            "strategy": config.strategy,
            "top_n": config.top_n,
            "swiss_rounds": config.swiss_rounds,
            "playoff_top_n": config.playoff_top_n,
            "playoff_ids": playoff_ids,
            "pair_count": len(matches),
            "scheduled_pair_count": scheduled_pair_count(schedule),
            "all_pairs_equivalent_pair_count": len(build_tournament_pairs(tournament_ids)),
            "batch_count": len(batch_results),
            "pairs_per_batch": config.pairs_per_batch,
            "pair_schedule": "deterministic_random",
            "pair_schedule_seed": config.seed or 0,
            "pair_orientation": "deterministic_random",
            "pair_orientation_seed": config.seed or 0,
            "schedule": schedule,
            "usage": usage,
            "elapsed_seconds": elapsed_seconds,
            "ranking_method": tournament_ranking_method_note(config),
            "limitation": tournament_limitation_note(config),
        },
        "matches": matches,
    }
    return payload


def tournament_ranking_method_note(config: FrontierTournamentConfig) -> str:
    if config.strategy == "all_pairs":
        return (
            "All-pairs Copeland-style points over the seed top-N: win=1, tie=0.5, loss=0; "
            "ties broken by wins, confidence-weighted margin, then seed synthesis rank."
        )
    if config.strategy == "swiss_playoff":
        return (
            "Swiss rounds identify a provisional playoff set, then all-pairs adjudication ranks that playoff set. "
            "Papers outside the playoff are ordered by normalized Swiss win points."
        )
    return "Swiss rounds rank papers by normalized win points, with seed rank as the final tie-breaker."


def tournament_limitation_note(config: FrontierTournamentConfig) -> str:
    if config.strategy == "all_pairs":
        return (
            "Ranks below the tournament pool remain from the previous seed ranking and should not be "
            "overinterpreted as pairwise-adjudicated."
        )
    if config.strategy == "swiss_playoff":
        return (
            "Only the playoff subset receives dense all-pairs adjudication. Swiss-only ranks are suitable for "
            "candidate selection and broad ordering, not final headline claims."
        )
    return "Swiss-only ranks are suitable for reranking/candidate selection, not final headline claims."


def build_frontier_synthesis_messages(
    rows_by_paper: dict[str, list[dict[str, Any]]],
    *,
    config: FrontierSynthesisConfig,
) -> list[dict[str, Any]]:
    system = """You are the final synthesis judge for an ICML paper impact set.

You do not have direct PDF attachments in this call. Instead, you have three independent frontier-model judgment cards per paper. Those cards were produced after each judge saw the configured first-page PDF excerpt and extracted main-paper text. Treat the cards as evidence, not as votes to average blindly.
No reviews, reviewer scores, area-chair comments, decisions, presentation tiers, or awards are included in the cards. Do not infer paper quality from possible venue or outcome cues.

Rank by likely broad scientific and machine-learning impact, with technical soundness and evidence quality as gates. Penalize narrow toy-only results, weak evidence, incremental polish, and unclear adoption paths. Elevate papers with credible field-shaping ideas, durable datasets/benchmarks, infrastructure, safety/eval resources, scientific tools, theory, or algorithms.

Return only valid JSON."""
    paper_blocks = "\n\n".join(format_synthesis_paper_block(paper_id, rows) for paper_id, rows in sorted(rows_by_paper.items()))
    paper_ids = sorted(rows_by_paper)
    user = f"""Create a forced 1..{len(paper_ids)} reference ranking for {config.paper_set_name}.

Use the three PDF-aware frontier judge cards for each paper. Do not use web search, author identity, institution, citation memory, or outside knowledge.

Ranking requirements:
- Return exactly {len(paper_ids)} ranked_papers entries.
- Every paper_id must appear exactly once: {", ".join(paper_ids)}
- Scores should be comparative and spread out; do not let conservative per-paper scores collapse the ranking.
- When judges disagree, prefer the evidence-backed argument over majority vote.
- Preserve contribution diversity in category rankings, but the overall ranking should still be a single best-paper ordering.

Contribution classes:
{", ".join(CONTRIBUTION_CLASSES)}

Required JSON shape:
{{
  "evaluation_mode": "frontier_pdf_card_synthesis_ranking",
  "prompt_version": "{config.prompt_version}",
  "judge": "{config.provider}:{config.model}",
  "paper_set": "{config.paper_set_name}",
  "modality": "synthesis_of_three_pdf_aware_frontier_cards",
  "ranked_papers": [
    {{
      "rank": 1,
      "paper_id": "paper id",
      "title": "title",
      "primary_contribution_class": "core_ml_algorithm",
      "secondary_contribution_classes": ["scientific_modeling_tool"],
      "category_rank": 1,
      "overall_priority_score": 10,
      "broad_scientific_impact_score": 10,
      "ml_field_impact_score": 10,
      "technical_soundness_score": 9,
      "evidence_confidence_score": 8,
      "advance_to_final_review": true,
      "why_ranked_here": "comparative reason grounded in the judge cards",
      "best_case_for_impact": "strongest plausible impact route",
      "main_risk": "main reason this may not matter or hold up",
      "judge_disagreement_note": "where relevant, explain disagreement or uncertainty"
    }}
  ],
  "category_rankings": {{
    "core_ml_algorithm": ["paper_id"],
    "theory": ["paper_id"],
    "benchmark_dataset": ["paper_id"],
    "infrastructure_systems": ["paper_id"],
    "scientific_modeling_tool": ["paper_id"],
    "safety_governance_eval": ["paper_id"],
    "application_method": ["paper_id"],
    "analysis_position": ["paper_id"],
    "other": ["paper_id"]
  }},
  "calibration_notes": {{
    "top_10_threshold": "what distinguished the top 10",
    "major_disagreements": ["paper_id plus explanation"],
    "visual_evidence_effect": "how PDF visual/table evidence changed the reference ranking"
  }},
  "overall_assessment": "short assessment"
}}

PDF-aware judge cards:
{paper_blocks}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def format_synthesis_paper_block(paper_id: str, rows: list[dict[str, Any]]) -> str:
    cards = [row.get("card") for row in rows if isinstance(row.get("card"), dict)]
    title = first_nonempty(card.get("title") for card in cards)
    lines = [f"<PAPER id=\"{paper_id}\">", f"Title: {title}"]
    for card in cards:
        scores = card.get("scores") if isinstance(card.get("scores"), dict) else {}
        lines.extend(
            [
                f"Judge: {card.get('judge')}",
                f"Class: {card.get('primary_contribution_class')} secondary={card.get('secondary_contribution_classes')}",
                (
                    "Scores: overall={overall} broad={broad} ml={ml} soundness={soundness} "
                    "novelty={novelty} evidence={evidence} visual={visual}"
                ).format(
                    overall=scores.get("overall_gold_priority_score"),
                    broad=scores.get("broad_scientific_impact_score"),
                    ml=scores.get("ml_field_impact_score"),
                    soundness=scores.get("technical_soundness_score"),
                    novelty=scores.get("novelty_score"),
                    evidence=scores.get("evidence_confidence_score"),
                    visual=scores.get("visual_evidence_importance_score"),
                ),
                f"Summary: {card.get('one_sentence_summary')}",
                f"Best case: {nested_get(card, 'impact_assessment', 'best_case_for_impact')}",
                f"Main risk: {nested_get(card, 'impact_assessment', 'main_risk')}",
                f"Visual evidence effect: {nested_get(card, 'evidence_from_pdf', 'visual_or_tabular_evidence_changes_judgment')}",
                f"Top-10 case: {nested_get(card, 'ranking_hooks', 'top_10_case')}",
            ]
        )
    lines.append("</PAPER>")
    return "\n".join(str(line) for line in lines)


def validate_synthesis_ranking(parsed: dict[str, Any], *, expected_ids: list[str]) -> list[str]:
    errors: list[str] = []
    ranked = parsed.get("ranked_papers")
    if not isinstance(ranked, list):
        return ["ranked_papers must be a list"]
    row_ids = [str(row.get("paper_id") or "") for row in ranked if isinstance(row, dict)]
    expected_set = set(expected_ids)
    row_id_set = set(row_ids)
    missing = sorted(expected_set - row_id_set)
    extras = sorted(row_id_set - expected_set)
    duplicates = sorted({paper_id for paper_id in row_ids if row_ids.count(paper_id) > 1})
    if missing:
        errors.append(f"missing paper_ids: {missing}")
    if extras:
        errors.append(f"extra paper_ids: {extras}")
    if duplicates:
        errors.append(f"duplicate paper_ids: {duplicates}")
    ranks = [int(number(row.get("rank"))) for row in ranked if isinstance(row, dict)]
    if sorted(ranks) != list(range(1, len(expected_ids) + 1)):
        errors.append("ranks must be exactly 1..N")
    return errors


FRONTIER_RANK_SCORE_KEYS = (
    "overall_gold_priority_score",
    "broad_scientific_impact_score",
    "ml_field_impact_score",
    "technical_soundness_score",
    "evidence_confidence_score",
    "novelty_score",
)

FRONTIER_ALL_SCORE_KEYS = FRONTIER_RANK_SCORE_KEYS + ("visual_evidence_importance_score",)


def load_complete_card_stream(*, path: Path, label: str) -> dict[str, dict[str, Any]]:
    rows = load_card_rows([path])
    if not rows:
        raise ValueError(f"frontier card stream is empty: {label} ({path})")
    by_paper: dict[str, dict[str, Any]] = {}
    for row in rows:
        paper_id = str(row.get("paper_id") or "")
        if not paper_id:
            raise ValueError(f"frontier card stream {label} contains a row without paper_id")
        if paper_id in by_paper:
            raise ValueError(f"frontier card stream {label} has duplicate paper_id {paper_id}")
        if row.get("status") != "ok":
            raise ValueError(
                f"frontier card stream {label} has non-ok paper {paper_id}: {row.get('status')}"
            )
        card = row.get("card")
        if not isinstance(card, dict):
            raise ValueError(f"frontier card stream {label} has no card for {paper_id}")
        validation_errors = validate_frontier_card(card, expected_paper_id=paper_id)
        if validation_errors:
            raise ValueError(
                f"frontier card stream {label} has invalid paper {paper_id}: {validation_errors}"
            )
        by_paper[paper_id] = dict(row)
    return by_paper


def rank_frontier_card_stream(
    rows_by_paper: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    paper_ids = list(rows_by_paper)
    ranking_values = {
        paper_id: tuple(
            card_score(rows_by_paper[paper_id]["card"], key)
            for key in FRONTIER_RANK_SCORE_KEYS
        )
        for paper_id in paper_ids
    }
    ranks = tie_aware_descending_ranks(ranking_values)
    axis_percentiles = {
        key: rank_percentiles(
            tie_aware_descending_ranks(
                {
                    paper_id: (card_score(rows_by_paper[paper_id]["card"], key),)
                    for paper_id in paper_ids
                }
            )
        )
        for key in FRONTIER_ALL_SCORE_KEYS
    }
    percentiles = rank_percentiles(ranks)
    ranked: dict[str, dict[str, Any]] = {}
    for paper_id, row in rows_by_paper.items():
        ranked[paper_id] = {
            **row,
            "within_judge_rank": round(ranks[paper_id], 6),
            "within_judge_percentile": round(percentiles[paper_id], 9),
            "within_judge_axis_percentiles": {
                key: round(values[paper_id], 9)
                for key, values in axis_percentiles.items()
            },
        }
    return ranked


def tie_aware_descending_ranks(
    values_by_paper: dict[str, tuple[float, ...]],
) -> dict[str, float]:
    ordered = sorted(
        values_by_paper,
        key=lambda paper_id: values_by_paper[paper_id],
        reverse=True,
    )
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        value = values_by_paper[ordered[index]]
        while end < len(ordered) and values_by_paper[ordered[end]] == value:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for tied_index in range(index, end):
            ranks[ordered[tied_index]] = average_rank
        index = end
    return ranks


def rank_percentiles(ranks_by_paper: dict[str, float]) -> dict[str, float]:
    count = len(ranks_by_paper)
    if count <= 1:
        return {paper_id: 1.0 for paper_id in ranks_by_paper}
    return {
        paper_id: (count - rank) / (count - 1)
        for paper_id, rank in ranks_by_paper.items()
    }


def ensemble_item(
    paper_id: str,
    rows_by_label: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = list(rows_by_label.values())
    cards = [row["card"] for row in rows]
    score_keys = [
        "overall_gold_priority_score",
        "broad_scientific_impact_score",
        "ml_field_impact_score",
        "technical_soundness_score",
        "novelty_score",
        "evidence_confidence_score",
        "visual_evidence_importance_score",
    ]
    means = {key: mean([card_score(card, key) for card in cards]) for key in score_keys}
    overall_values = [card_score(card, "overall_gold_priority_score") for card in cards]
    judge_percentiles = {
        label: number(row.get("within_judge_percentile"))
        for label, row in rows_by_label.items()
    }
    rank_score = statistics.mean(judge_percentiles.values())
    consensus_axis_percentiles = {
        key: statistics.mean(
            number(row.get("within_judge_axis_percentiles", {}).get(key))
            for row in rows
        )
        for key in FRONTIER_ALL_SCORE_KEYS
    }
    primary_class = majority(
        str(card.get("primary_contribution_class") or "other")
        for card in cards
        if isinstance(card, dict)
    )
    if primary_class not in CONTRIBUTION_CLASSES:
        primary_class = "other"
    title = next((str(card.get("title")) for card in cards if card.get("title")), "")
    return {
        "rank": None,
        "paper_id": paper_id,
        "title": title,
        "primary_contribution_class": primary_class,
        "secondary_contribution_classes": sorted(
            {
                str(value)
                for card in cards
                for value in card.get("secondary_contribution_classes", [])
                if isinstance(value, str) and value in CONTRIBUTION_CLASSES
            }
        ),
        "overall_priority_score": round(means["overall_gold_priority_score"], 3),
        "broad_scientific_impact_score": round(means["broad_scientific_impact_score"], 3),
        "ml_field_impact_score": round(means["ml_field_impact_score"], 3),
        "technical_soundness_score": round(means["technical_soundness_score"], 3),
        "novelty_score": round(means["novelty_score"], 3),
        "evidence_confidence_score": round(means["evidence_confidence_score"], 3),
        "visual_evidence_importance_score": round(means["visual_evidence_importance_score"], 3),
        "ensemble_rank_score": round(rank_score, 9),
        "ensemble_gold_priority_score": round(rank_score, 9),
        "ensemble_broad_scientific_impact_score": round(
            consensus_axis_percentiles["broad_scientific_impact_score"],
            9,
        ),
        "ensemble_technical_soundness_score": round(
            consensus_axis_percentiles["technical_soundness_score"],
            9,
        ),
        "consensus_axis_percentiles": {
            key: round(value, 9)
            for key, value in consensus_axis_percentiles.items()
        },
        "raw_mean_scores": {key: round(value, 3) for key, value in means.items()},
        "judge_count": len(cards),
        "judge_score_stddev": round(statistics.pstdev(overall_values), 3) if len(overall_values) > 1 else 0.0,
        "judge_rank_stddev": round(
            statistics.pstdev(judge_percentiles.values()),
            9,
        )
        if len(judge_percentiles) > 1
        else 0.0,
        "judge_rank_range": round(
            max(judge_percentiles.values()) - min(judge_percentiles.values()),
            9,
        )
        if judge_percentiles
        else 0.0,
        "source_judge_percentiles": {
            label: round(value, 9)
            for label, value in judge_percentiles.items()
        },
        "source_judge_ranks": {
            label: round(number(row.get("within_judge_rank")), 6)
            for label, row in rows_by_label.items()
        },
        "advance_to_final_review": True,
        "judge_summaries": [
            {
                "stream_label": label,
                "judge": card.get("judge"),
                "requested_model": row.get("fallback_from_model") or row.get("model"),
                "served_model": row.get("served_model"),
                "fallback_from_model": row.get("fallback_from_model"),
                "within_judge_rank": round(number(row.get("within_judge_rank")), 6),
                "within_judge_percentile": round(
                    number(row.get("within_judge_percentile")),
                    9,
                ),
                "overall_gold_priority_score": card_score(card, "overall_gold_priority_score"),
                "broad_scientific_impact_score": card_score(card, "broad_scientific_impact_score"),
                "technical_soundness_score": card_score(card, "technical_soundness_score"),
                "best_case_for_impact": nested_get(card, "impact_assessment", "best_case_for_impact"),
                "main_risk": nested_get(card, "impact_assessment", "main_risk"),
                "visual_or_tabular_evidence_changes_judgment": nested_get(
                    card,
                    "evidence_from_pdf",
                    "visual_or_tabular_evidence_changes_judgment",
                ),
                "one_sentence_summary": card.get("one_sentence_summary"),
            }
            for label, row in rows_by_label.items()
            for card in [row["card"]]
        ],
        "why_ranked_here": synthesize_why_ranked(cards),
        "best_case_for_impact": first_nonempty(
            nested_get(card, "impact_assessment", "best_case_for_impact") for card in cards
        ),
        "main_risk": first_nonempty(nested_get(card, "impact_assessment", "main_risk") for card in cards),
    }


def frontier_ensemble_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    axis = item.get("consensus_axis_percentiles")
    axis = axis if isinstance(axis, dict) else {}
    return (
        -number(item.get("ensemble_rank_score")),
        -number(axis.get("broad_scientific_impact_score")),
        -number(axis.get("ml_field_impact_score")),
        -number(axis.get("technical_soundness_score")),
        -number(axis.get("evidence_confidence_score")),
        -number(axis.get("novelty_score")),
        number(item.get("judge_rank_stddev")),
        str(item.get("paper_id") or ""),
    )


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def validate_frontier_card(parsed: dict[str, Any], *, expected_paper_id: str) -> list[str]:
    errors: list[str] = []
    if str(parsed.get("paper_id") or "") != expected_paper_id:
        errors.append(f"paper_id mismatch: expected {expected_paper_id}, got {parsed.get('paper_id')}")
    if parsed.get("evaluation_mode") != "frontier_pdf_paper_card":
        errors.append("evaluation_mode must be frontier_pdf_paper_card")
    if not isinstance(parsed.get("title"), str) or not parsed["title"].strip():
        errors.append("title must be a non-empty string")
    scores = parsed.get("scores")
    if not isinstance(scores, dict):
        errors.append("missing scores object")
    else:
        for key in [
            "overall_gold_priority_score",
            "broad_scientific_impact_score",
            "ml_field_impact_score",
            "technical_soundness_score",
            "novelty_score",
            "evidence_confidence_score",
            "visual_evidence_importance_score",
        ]:
            value = number(scores.get(key))
            if value < 1 or value > 10:
                errors.append(f"{key} must be in 1..10")
    contribution_class = str(parsed.get("primary_contribution_class") or "")
    if contribution_class not in CONTRIBUTION_CLASSES:
        errors.append(f"unknown primary_contribution_class: {contribution_class}")
    secondary_classes = parsed.get("secondary_contribution_classes")
    if not isinstance(secondary_classes, list):
        errors.append("secondary_contribution_classes must be a list")
    elif any(str(value) not in CONTRIBUTION_CLASSES for value in secondary_classes):
        errors.append("secondary_contribution_classes contains an unknown class")

    _require_string_fields(
        parsed,
        section="impact_assessment",
        fields=(
            "two_year_adoption_path",
            "five_year_field_effect",
            "best_case_for_impact",
            "main_risk",
            "why_not_higher",
            "why_not_lower",
        ),
        errors=errors,
    )
    _require_string_fields(
        parsed,
        section="ranking_hooks",
        fields=("beats_papers_when", "loses_to_papers_when", "top_10_case"),
        errors=errors,
    )
    ranking_hooks = parsed.get("ranking_hooks")
    if isinstance(ranking_hooks, dict) and not isinstance(ranking_hooks.get("category_standout"), bool):
        errors.append("ranking_hooks.category_standout must be a boolean")
    evidence = parsed.get("evidence_from_pdf")
    if not isinstance(evidence, dict):
        errors.append("missing evidence_from_pdf object")
    else:
        for key in ("figures_or_tables_that_matter", "missing_or_weak_visual_evidence"):
            if not isinstance(evidence.get(key), list):
                errors.append(f"evidence_from_pdf.{key} must be a list")
        visual_judgment = evidence.get("visual_or_tabular_evidence_changes_judgment")
        if not isinstance(visual_judgment, str) or not visual_judgment.strip():
            errors.append(
                "evidence_from_pdf.visual_or_tabular_evidence_changes_judgment "
                "must be a non-empty string"
            )
    summary = parsed.get("one_sentence_summary")
    if not isinstance(summary, str) or not summary.strip():
        errors.append("one_sentence_summary must be a non-empty string")
    return errors


def _require_string_fields(
    payload: dict[str, Any],
    *,
    section: str,
    fields: tuple[str, ...],
    errors: list[str],
) -> None:
    value = payload.get(section)
    if not isinstance(value, dict):
        errors.append(f"missing {section} object")
        return
    for field in fields:
        child = value.get(field)
        if not isinstance(child, str) or not child.strip():
            errors.append(f"{section}.{field} must be a non-empty string")


def load_card_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def encode_pdf_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:application/pdf;base64,{data}"


def redact_file_data(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_redact_value(message) for message in messages]


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, child in value.items():
            if key == "file_data" and isinstance(child, str):
                redacted[key] = f"[base64 file data redacted, {len(child)} chars]"
            else:
                redacted[key] = _redact_value(child)
        return redacted
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


def _message_text_for_estimate(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


def truncate_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n\n[TRUNCATED for frontier PDF card at {limit} characters]"


def truncate_inline(value: Any, limit: int) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def number(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            return float(value)
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def card_score(card: dict[str, Any], key: str) -> float:
    scores = card.get("scores") if isinstance(card.get("scores"), dict) else {}
    return number(scores.get(key))


def mean(values: list[float]) -> float:
    filtered = [value for value in values if value > 0]
    return sum(filtered) / len(filtered) if filtered else 0.0


def majority(values: Any) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return "other"
    return sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)[0][0]


def nested_get(row: dict[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def first_nonempty(values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def synthesize_why_ranked(cards: list[dict[str, Any]]) -> str:
    summaries = [
        str(card.get("one_sentence_summary") or "").strip()
        for card in cards
        if str(card.get("one_sentence_summary") or "").strip()
    ]
    return " | ".join(summaries[:3])


def build_category_rankings(ranked_items: list[dict[str, Any]]) -> dict[str, list[str]]:
    category_rankings: dict[str, list[str]] = {category: [] for category in CONTRIBUTION_CLASSES}
    for item in ranked_items:
        category = str(item.get("primary_contribution_class") or "other")
        if category not in category_rankings:
            category = "other"
        category_rankings[category].append(str(item.get("paper_id")))
    return category_rankings
