from __future__ import annotations

import hashlib
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from icml_ai_ac.models import PaperRecord
from icml_ai_ac.scoring.providers import ChatCompletionClient, is_batch_blocking_provider_error
from icml_ai_ac.scoring.runner import estimate_tokens, read_text_path, resolve_record_text_path
from icml_ai_ac.scoring.schema import CONTRIBUTION_CLASSES, parse_json_response
from icml_ai_ac.scoring.usage import sum_usage
from icml_ai_ac.storage import ensure_parent, read_json, read_paper_records, write_json, write_jsonl


PASS2_PROMPT_VERSION = "pass2_strong_batch_rank_v4"
PASS2_BATCHED_PROMPT_VERSION = "pass2_strong_resumable_batches_v2"
PASS2_BATCHED_AGGREGATION_VERSION = "pass2_batch_rank_with_calibrated_priority_v2"
PASS2_STAGE1_PROMPT_VERSION = "pass2_stage1_strong_semifinal_v2"
PASS2_FINAL_PROMPT_VERSION = "pass2_stage2_strong_final_v2"
REFERENCE_PROMPT_VERSION = "reference_gold_rank_v2"
REFERENCE_REPAIR_PROMPT_VERSION = "reference_gold_rank_repair_v2"


@dataclass(slots=True)
class RankingCandidate:
    record: PaperRecord
    pass1_row: dict[str, Any]
    text_path: str
    resolved_text_source: str
    paper_text: str


@dataclass(slots=True)
class ReferenceCandidate:
    record: PaperRecord
    text_path: str
    resolved_text_source: str
    paper_text: str


@dataclass(slots=True)
class Pass2RankingConfig:
    provider: str
    model: str
    reasoning_effort: str | None
    prompt_version: str
    text_source: str
    top_fraction: float
    limit: int | None
    per_paper_char_budget: int
    temperature: float
    max_output_tokens: int
    seed: int | None
    dry_run: bool


@dataclass(slots=True)
class Pass2BatchedRankingConfig:
    provider: str
    model: str
    reasoning_effort: str | None
    prompt_version: str
    text_source: str
    top_fraction: float
    limit: int | None
    per_paper_char_budget: int
    batch_size: int
    partitions: int
    strategy: str
    class_path: Path | None
    request_delay_seconds: float
    temperature: float
    max_output_tokens: int
    seed: int | None
    dry_run: bool
    openrouter_provider_only: tuple[str, ...] = ()


@dataclass(slots=True)
class Pass2TwoStageConfig:
    provider: str
    model: str
    reasoning_effort: str | None
    stage1_prompt_version: str
    stage2_prompt_version: str
    text_source: str
    stage1_limit: int | None
    stage2_limit: int
    stage1_per_paper_char_budget: int
    stage2_per_paper_char_budget: int
    temperature: float
    stage1_max_output_tokens: int
    stage2_max_output_tokens: int
    seed: int | None
    dry_run: bool


@dataclass(slots=True)
class ReferenceRankingConfig:
    provider: str
    model: str
    reasoning_effort: str | None
    prompt_version: str
    paper_set_name: str
    text_source: str
    limit: int | None
    paper_ids: set[str]
    per_paper_char_budget: int
    temperature: float
    max_output_tokens: int
    seed: int | None
    dry_run: bool


@dataclass(slots=True)
class ReferenceRepairConfig:
    provider: str
    model: str
    reasoning_effort: str | None
    prompt_version: str
    paper_set_name: str
    text_source: str
    per_paper_char_budget: int
    temperature: float
    max_output_tokens: int
    seed: int | None
    dry_run: bool


def run_pass2_ranking(
    *,
    manifest: Path,
    pass1_path: Path,
    out: Path,
    run_dir: Path,
    config: Pass2RankingConfig,
    client: ChatCompletionClient | None,
) -> dict[str, Any]:
    started = time.monotonic()
    prepare_ranking_run_dir(run_dir)
    candidates = select_candidates(
        manifest=manifest,
        pass1_path=pass1_path,
        text_source=config.text_source,
        top_fraction=config.top_fraction,
        limit=config.limit,
        per_paper_char_budget=config.per_paper_char_budget,
    )
    messages = build_pass2_ranking_messages(candidates, config=config)
    prompt_payload = {
        "prompt_version": config.prompt_version,
        "provider": config.provider,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "candidate_count": len(candidates),
        "candidate_ids": [candidate.record.paper_id for candidate in candidates],
        "text_source": config.text_source,
        "messages": messages,
        "prompt_tokens_estimate": sum(estimate_tokens(message["content"]) for message in messages),
    }
    prompt_path = run_dir / "prompt.json"
    write_json(prompt_path, prompt_payload)

    base_result: dict[str, Any] = {
        "status": "dry_run" if config.dry_run else "pending",
        "provider": config.provider,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "prompt_version": config.prompt_version,
        "manifest": str(manifest),
        "pass1_path": str(pass1_path),
        "out": str(out),
        "run_dir": str(run_dir),
        "candidate_count": len(candidates),
        "candidate_ids": [candidate.record.paper_id for candidate in candidates],
        "candidate_titles": {candidate.record.paper_id: candidate.record.title for candidate in candidates},
        "prompt_path": str(prompt_path),
        "raw_response_path": None,
        "parsed_response_path": None,
        "elapsed_seconds": None,
    }
    if config.dry_run:
        result = {**base_result, "elapsed_seconds": round(time.monotonic() - started, 3)}
        write_json(out, result)
        write_json(run_dir / "run.json", result)
        return result
    if client is None:
        raise ValueError("client is required unless dry_run=True")

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
        expected_ids = [candidate.record.paper_id for candidate in candidates]
        canonicalize_ranked_paper_ids(parsed, expected_ids)
        validation_errors = validate_ranked_paper_coverage(parsed, expected_ids)
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
    except Exception as exc:  # noqa: BLE001 - persist the failed ranking attempt.
        error_path = run_dir / "error.json"
        write_json(error_path, {"error": repr(exc)})
        result = {
            **base_result,
            "status": "failed",
            "error_path": str(error_path),
            "error": repr(exc),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    write_json(out, result)
    write_json(run_dir / "run.json", result)
    return result


def run_pass2_batched_ranking(
    *,
    manifest: Path,
    pass1_path: Path,
    out: Path,
    run_dir: Path,
    config: Pass2BatchedRankingConfig,
    client: ChatCompletionClient | None,
) -> dict[str, Any]:
    """Rank a large semifinal pool through resumable, repeated listwise batches."""
    started = time.monotonic()
    prepare_ranking_run_dir(run_dir)
    if config.batch_size < 2:
        raise ValueError("batch_size must be at least 2")
    if config.partitions <= 0:
        raise ValueError("partitions must be positive")
    candidates = select_candidates(
        manifest=manifest,
        pass1_path=pass1_path,
        text_source=config.text_source,
        top_fraction=config.top_fraction,
        limit=config.limit,
        per_paper_char_budget=config.per_paper_char_budget,
    )
    class_by_id = read_contribution_class_map(config.class_path)
    partitions = make_pass2_partitions(
        candidates,
        batch_size=config.batch_size,
        partition_count=config.partitions,
        strategy=config.strategy,
        class_by_id=class_by_id,
        seed=config.seed or 0,
    )
    planned_connectivity_errors = validate_pass2_comparison_connectivity(
        [
            {
                "paper_id": candidate.record.paper_id,
                "partition_index": partition_index,
                "batch_index": batch_index,
            }
            for partition_index, batches in enumerate(partitions)
            for batch_index, batch_candidates in enumerate(batches)
            for candidate in batch_candidates
        ],
        candidate_ids=[candidate.record.paper_id for candidate in candidates],
    )
    if planned_connectivity_errors:
        raise ValueError("; ".join(planned_connectivity_errors))
    planned_batch_count = sum(len(partition) for partition in partitions)
    base_result: dict[str, Any] = {
        "status": "dry_run" if config.dry_run else "pending",
        "provider": config.provider,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "prompt_version": config.prompt_version,
        "manifest": str(manifest),
        "pass1_path": str(pass1_path),
        "out": str(out),
        "run_dir": str(run_dir),
        "candidate_count": len(candidates),
        "candidate_ids": [candidate.record.paper_id for candidate in candidates],
        "candidate_titles": {candidate.record.paper_id: candidate.record.title for candidate in candidates},
        "batch_size": config.batch_size,
        "partition_count": config.partitions,
        "strategy": config.strategy,
        "class_path": str(config.class_path) if config.class_path else None,
        "openrouter_provider_only": list(config.openrouter_provider_only),
        "batch_count": planned_batch_count,
    }
    if not config.dry_run and client is None:
        raise ValueError("client is required unless dry_run=True")

    single_config = Pass2RankingConfig(
        provider=config.provider,
        model=config.model,
        reasoning_effort=config.reasoning_effort,
        prompt_version=config.prompt_version,
        text_source=config.text_source,
        top_fraction=1.0,
        limit=None,
        per_paper_char_budget=config.per_paper_char_budget,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        seed=config.seed,
        dry_run=config.dry_run,
    )
    judgments: list[dict[str, Any]] = []
    batch_results: list[dict[str, Any]] = []
    failure_count = 0
    resumed_batch_count = 0
    blocked_reason: str | None = None
    for partition_index, batches in enumerate(partitions):
        if blocked_reason:
            break
        for batch_index, batch_candidates in enumerate(batches):
            batch_dir = (
                run_dir
                / "batches"
                / f"partition_{partition_index:02d}_batch_{batch_index:04d}"
            )
            batch_dir.mkdir(parents=True, exist_ok=True)
            messages = build_pass2_ranking_messages(batch_candidates, config=single_config)
            prompt_path = batch_dir / "prompt.json"
            prompt_payload = {
                "prompt_version": config.prompt_version,
                "provider": config.provider,
                "model": config.model,
                "reasoning_effort": config.reasoning_effort,
                "partition_index": partition_index,
                "batch_index": batch_index,
                "candidate_count": len(batch_candidates),
                "candidate_ids": [candidate.record.paper_id for candidate in batch_candidates],
                "text_source": config.text_source,
                "temperature": config.temperature,
                "max_output_tokens": config.max_output_tokens,
                "seed": config.seed,
                "messages": messages,
                "prompt_tokens_estimate": sum(estimate_tokens(message["content"]) for message in messages),
            }
            fingerprint = ranking_prompt_fingerprint(prompt_payload)
            prompt_payload["fingerprint"] = fingerprint
            write_json(prompt_path, prompt_payload)

            if config.dry_run:
                batch_results.append(
                    {
                        "status": "dry_run",
                        "partition_index": partition_index,
                        "batch_index": batch_index,
                        "candidate_count": len(batch_candidates),
                        "candidate_ids": prompt_payload["candidate_ids"],
                        "fingerprint": fingerprint,
                        "prompt_path": str(prompt_path),
                    }
                )
                continue

            cached = load_cached_pass2_batch(
                batch_dir=batch_dir,
                fingerprint=fingerprint,
                candidates=batch_candidates,
                prompt_path=prompt_path,
                partition_index=partition_index,
                batch_index=batch_index,
            )
            if cached is not None:
                batch_judgments, batch_result = cached
                judgments.extend(batch_judgments)
                batch_results.append(batch_result)
                resumed_batch_count += 1
                continue

            recovered = recover_saved_pass2_batch(
                batch_dir=batch_dir,
                fingerprint=fingerprint,
                candidates=batch_candidates,
                prompt_path=prompt_path,
                partition_index=partition_index,
                batch_index=batch_index,
            )
            if recovered is not None:
                batch_judgments, batch_result = recovered
                judgments.extend(batch_judgments)
                batch_results.append(batch_result)
                resumed_batch_count += 1
                continue

            archived_attempt_count = archive_failed_pass2_attempt(batch_dir)
            clear_failed_pass2_attempt(batch_dir)
            provider_attempt_index = archived_attempt_count + 1
            chat = None
            try:
                if client is None:
                    raise RuntimeError("live pass-2 batches require a provider client")
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
                expected_ids = [candidate.record.paper_id for candidate in batch_candidates]
                canonicalize_ranked_paper_ids(parsed, expected_ids)
                validation_errors = validate_ranked_paper_coverage(parsed, expected_ids)
                if validation_errors:
                    raise ValueError("; ".join(validation_errors))
                parsed_response_path = batch_dir / "parsed.json"
                write_json(parsed_response_path, parsed)
                batch_judgments = pass2_payload_to_judgments(
                    parsed,
                    partition_index=partition_index,
                    batch_index=batch_index,
                    batch_size=len(batch_candidates),
                    prompt_path=prompt_path,
                    raw_response_path=raw_response_path,
                    parsed_response_path=parsed_response_path,
                )
                batch_result = {
                    "status": "ok",
                    "partition_index": partition_index,
                    "batch_index": batch_index,
                    "candidate_count": len(batch_candidates),
                    "candidate_ids": expected_ids,
                    "fingerprint": fingerprint,
                    "prompt_path": str(prompt_path),
                    "raw_response_path": str(raw_response_path),
                    "parsed_response_path": str(parsed_response_path),
                    "served_model": chat.served_model,
                    "usage": chat.usage,
                    "provider_elapsed_seconds": round(chat.elapsed_seconds, 3),
                    "provider_attempt_index": provider_attempt_index,
                    "provider_preferences": getattr(chat, "request", {}).get("provider"),
                    "resumed": False,
                }
                write_json(batch_dir / "batch.json", batch_result)
                judgments.extend(batch_judgments)
                batch_results.append(batch_result)
            except Exception as exc:  # noqa: BLE001 - persist and retry only this batch later.
                failure_count += 1
                response = getattr(exc, "response", None)
                raw_response_path = batch_dir / "response.json"
                if isinstance(response, dict) and not raw_response_path.exists():
                    write_json(raw_response_path, response)
                usage = chat.usage if chat is not None else getattr(exc, "usage", {})
                served_model = chat.served_model if chat is not None else getattr(exc, "served_model", None)
                error_path = batch_dir / "error.json"
                write_json(error_path, {"error": repr(exc)})
                batch_result = {
                    "status": "failed",
                    "partition_index": partition_index,
                    "batch_index": batch_index,
                    "candidate_count": len(batch_candidates),
                    "candidate_ids": prompt_payload["candidate_ids"],
                    "fingerprint": fingerprint,
                    "prompt_path": str(prompt_path),
                    "error_path": str(error_path),
                    "error": repr(exc),
                    "provider_attempt_index": provider_attempt_index,
                }
                if raw_response_path.exists():
                    batch_result["raw_response_path"] = str(raw_response_path)
                if isinstance(usage, dict) and usage:
                    batch_result["usage"] = usage
                if served_model:
                    batch_result["served_model"] = served_model
                provider_preferences = getattr(client, "openrouter_provider_preferences", None)
                if provider_preferences:
                    batch_result["provider_preferences"] = provider_preferences
                write_json(batch_dir / "batch.json", batch_result)
                batch_results.append(batch_result)
                if is_batch_blocking_provider_error(exc):
                    blocked_reason = repr(exc)
            if config.request_delay_seconds:
                time.sleep(config.request_delay_seconds)
            if blocked_reason:
                break

    coverage_errors = validate_pass2_judgment_coverage(
        judgments,
        candidate_ids=[candidate.record.paper_id for candidate in candidates],
        partition_count=config.partitions,
    )
    coverage_errors.extend(
        validate_pass2_comparison_connectivity(
            judgments,
            candidate_ids=[candidate.record.paper_id for candidate in candidates],
        )
    )
    complete = (
        not config.dry_run
        and failure_count == 0
        and len(batch_results) == planned_batch_count
        and not coverage_errors
    )
    ranking = aggregate_pass2_judgments(candidates, judgments) if complete else None
    served_models = sorted(
        {
            str(result["served_model"])
            for result in batch_results
            if result.get("status") == "ok" and result.get("served_model")
        }
    )
    result = {
        **base_result,
        "status": "dry_run" if config.dry_run else ("ok" if complete else "partial_failed"),
        "ranking": ranking,
        "served_model": served_models[0] if len(served_models) == 1 else None,
        "served_models": served_models,
        "usage": sum_usage(result.get("usage") for result in batch_results),
        "judgment_count": len(judgments),
        "expected_judgment_count": len(candidates) * config.partitions,
        "coverage_errors": coverage_errors,
        "attempted_batch_count": len(batch_results),
        "skipped_batch_count": planned_batch_count - len(batch_results),
        "failure_count": failure_count,
        "resumed_batch_count": resumed_batch_count,
        "blocked_reason": blocked_reason,
        "batch_results": batch_results,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(out, result)
    write_json(run_dir / "run.json", result)
    return result


def archive_failed_pass2_attempt(batch_dir: Path) -> int:
    state_path = batch_dir / "batch.json"
    if not state_path.exists():
        return existing_attempt_count(batch_dir)
    try:
        state = read_json(state_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        state = None
    if not isinstance(state, dict) or state.get("status") != "failed":
        return existing_attempt_count(batch_dir)
    attempts_dir = batch_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    attempt_index = existing_attempt_count(batch_dir) + 1
    attempt_dir = attempts_dir / f"attempt_{attempt_index:04d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    for name in ("prompt.json", "response.json", "parsed.json", "error.json", "batch.json"):
        source = batch_dir / name
        if source.exists():
            shutil.copy2(source, attempt_dir / name)
    return attempt_index


def existing_attempt_count(batch_dir: Path) -> int:
    attempts_dir = batch_dir / "attempts"
    if not attempts_dir.exists():
        return 0
    return sum(1 for path in attempts_dir.glob("attempt_*") if path.is_dir())


def clear_failed_pass2_attempt(batch_dir: Path) -> None:
    state_path = batch_dir / "batch.json"
    if not state_path.exists():
        return
    try:
        state = read_json(state_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        state = None
    if not isinstance(state, dict) or state.get("status") != "failed":
        return
    for name in ("response.json", "parsed.json", "error.json", "batch.json"):
        path = batch_dir / name
        if path.exists():
            path.unlink()


def ranking_prompt_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_cached_pass2_batch(
    *,
    batch_dir: Path,
    fingerprint: str,
    candidates: list[RankingCandidate],
    prompt_path: Path,
    partition_index: int,
    batch_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    state_path = batch_dir / "batch.json"
    raw_response_path = batch_dir / "response.json"
    parsed_response_path = batch_dir / "parsed.json"
    if not state_path.exists() or not raw_response_path.exists() or not parsed_response_path.exists():
        return None
    try:
        state = read_json(state_path)
        if (
            not isinstance(state, dict)
            or state.get("status") != "ok"
            or state.get("fingerprint") != fingerprint
        ):
            return None
        parsed = read_json(parsed_response_path)
        if not isinstance(parsed, dict):
            return None
        expected_ids = [candidate.record.paper_id for candidate in candidates]
        canonicalize_ranked_paper_ids(parsed, expected_ids)
        errors = validate_ranked_paper_coverage(parsed, expected_ids)
        if errors:
            return None
        effective_prompt_path = Path(str(state.get("effective_prompt_path") or prompt_path))
        judgments = pass2_payload_to_judgments(
            parsed,
            partition_index=partition_index,
            batch_index=batch_index,
            batch_size=len(candidates),
            prompt_path=effective_prompt_path,
            raw_response_path=raw_response_path,
            parsed_response_path=parsed_response_path,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return judgments, {**state, "resumed": True}


def recover_saved_pass2_batch(
    *,
    batch_dir: Path,
    fingerprint: str,
    candidates: list[RankingCandidate],
    prompt_path: Path,
    partition_index: int,
    batch_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    state_path = batch_dir / "batch.json"
    raw_response_path = batch_dir / "response.json"
    if not state_path.exists() or not raw_response_path.exists():
        return None
    try:
        state = read_json(state_path)
        if (
            not isinstance(state, dict)
            or state.get("status") != "failed"
            or state.get("fingerprint") != fingerprint
        ):
            return None
        response = read_json(raw_response_path)
        content = response["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            return None
        parsed = parse_json_response(content)
        expected_ids = [candidate.record.paper_id for candidate in candidates]
        canonicalize_ranked_paper_ids(parsed, expected_ids)
        errors = validate_ranked_paper_coverage(parsed, expected_ids)
        if errors:
            return None
        archived_attempt_count = archive_failed_pass2_attempt(batch_dir)
        parsed_response_path = batch_dir / "parsed.json"
        write_json(parsed_response_path, parsed)
        judgments = pass2_payload_to_judgments(
            parsed,
            partition_index=partition_index,
            batch_index=batch_index,
            batch_size=len(candidates),
            prompt_path=prompt_path,
            raw_response_path=raw_response_path,
            parsed_response_path=parsed_response_path,
        )
        recovered_state = {
            **state,
            "status": "ok",
            "validation_errors": [],
            "raw_response_path": str(raw_response_path),
            "parsed_response_path": str(parsed_response_path),
            "recovered_from_saved_response": True,
            "archived_attempt_count": archived_attempt_count,
            "resumed": True,
        }
        recovered_state.pop("error", None)
        recovered_state.pop("error_path", None)
        write_json(state_path, recovered_state)
        error_path = batch_dir / "error.json"
        if error_path.exists():
            error_path.unlink()
    except (IndexError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return judgments, recovered_state


def pass2_payload_to_judgments(
    payload: dict[str, Any],
    *,
    partition_index: int,
    batch_index: int,
    batch_size: int,
    prompt_path: Path,
    raw_response_path: Path,
    parsed_response_path: Path,
) -> list[dict[str, Any]]:
    rows = payload.get("ranked_papers")
    if not isinstance(rows, list):
        raise ValueError("ranking payload missing ranked_papers")
    judgments: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("ranked_papers contains a non-object item")
        local_rank = int(row["rank"])
        normalized_rank = (local_rank - 1) / max(1, batch_size - 1)
        judgments.append(
            {
                "paper_id": str(row["paper_id"]),
                "partition_index": partition_index,
                "batch_index": batch_index,
                "batch_size": batch_size,
                "local_rank": local_rank,
                "normalized_local_rank": normalized_rank,
                "prompt_path": str(prompt_path),
                "raw_response_path": str(raw_response_path),
                "parsed_response_path": str(parsed_response_path),
                "judgment": row,
            }
        )
    return judgments


def validate_pass2_judgment_coverage(
    judgments: list[dict[str, Any]],
    *,
    candidate_ids: list[str],
    partition_count: int,
) -> list[str]:
    errors: list[str] = []
    expected = set(candidate_ids)
    observed = {str(row.get("paper_id") or "") for row in judgments}
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing:
        errors.append(f"missing judgments for {len(missing)} papers: {missing[:10]}")
    if extra:
        errors.append(f"unexpected judgments for {len(extra)} papers: {extra[:10]}")
    for paper_id in candidate_ids:
        rows = [row for row in judgments if row.get("paper_id") == paper_id]
        partitions = [int(row.get("partition_index", -1)) for row in rows]
        if len(rows) != partition_count or sorted(partitions) != list(range(partition_count)):
            errors.append(
                f"paper_id={paper_id} expected one judgment in each of {partition_count} partitions; "
                f"observed={partitions}"
            )
    return errors


def validate_pass2_comparison_connectivity(
    judgments: list[dict[str, Any]],
    *,
    candidate_ids: list[str],
) -> list[str]:
    """Require overlapping batches to connect every paper to one comparison graph."""
    if len(candidate_ids) <= 1:
        return []
    expected = set(candidate_ids)
    adjacency = {paper_id: set() for paper_id in candidate_ids}
    by_batch: dict[tuple[int, int], list[str]] = {}
    for row in judgments:
        paper_id = str(row.get("paper_id") or "")
        if paper_id not in expected:
            continue
        key = (
            int(row.get("partition_index", -1)),
            int(row.get("batch_index", -1)),
        )
        by_batch.setdefault(key, []).append(paper_id)
    for paper_ids in by_batch.values():
        for paper_id in paper_ids:
            adjacency[paper_id].update(other for other in paper_ids if other != paper_id)
    visited: set[str] = set()
    stack = [candidate_ids[0]]
    while stack:
        paper_id = stack.pop()
        if paper_id in visited:
            continue
        visited.add(paper_id)
        stack.extend(adjacency[paper_id] - visited)
    missing = sorted(expected - visited)
    if missing:
        return [
            "listwise comparison graph is disconnected; "
            f"{len(missing)} papers are outside the first component: {missing[:10]}"
        ]
    return []


def aggregate_pass2_judgments(
    candidates: list[RankingCandidate],
    judgments: list[dict[str, Any]],
) -> dict[str, Any]:
    input_rank = {
        candidate.record.paper_id: rank
        for rank, candidate in enumerate(candidates, start=1)
    }
    by_id: dict[str, list[dict[str, Any]]] = {}
    for judgment in judgments:
        by_id.setdefault(str(judgment["paper_id"]), []).append(judgment)
    priority_metrics = calibrate_pass2_batch_priority_scores(judgments)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        paper_id = candidate.record.paper_id
        source_judgments = sorted(
            by_id[paper_id],
            key=lambda row: (row["partition_index"], row["batch_index"]),
        )
        normalized_ranks = [float(row["normalized_local_rank"]) for row in source_judgments]
        mean_normalized_rank = sum(normalized_ranks) / len(normalized_ranks)
        representative = min(
            source_judgments,
            key=lambda row: (
                abs(float(row["normalized_local_rank"]) - mean_normalized_rank),
                row["partition_index"],
                row["batch_index"],
            ),
        )
        row = dict(representative["judgment"])
        for field in (
            "broad_scientific_impact_score",
            "ml_field_impact_score",
            "technical_soundness_score",
        ):
            values = [number(source["judgment"].get(field)) for source in source_judgments]
            row[field] = round(sum(values) / len(values), 4)
        paper_priority_metrics = [
            priority_metrics[
                (
                    int(source["partition_index"]),
                    int(source["batch_index"]),
                    paper_id,
                )
            ]
            for source in source_judgments
        ]
        canonical_priority_scores = [
            float(metric["canonical_score"]) for metric in paper_priority_metrics
        ]
        normalized_priority_scores = [
            float(metric["within_batch_normalized_score"]) for metric in paper_priority_metrics
        ]
        row.update(
            {
                "paper_id": paper_id,
                "title": candidate.record.title,
                "input_cheap_rank": input_rank[paper_id],
                "mean_normalized_local_rank": round(mean_normalized_rank, 6),
                "overall_priority_score": round(
                    sum(canonical_priority_scores) / len(canonical_priority_scores),
                    4,
                ),
                "mean_batch_normalized_priority_score": round(
                    sum(normalized_priority_scores) / len(normalized_priority_scores),
                    6,
                ),
                "raw_overall_priority_scores": [
                    number(source["judgment"].get("overall_priority_score"))
                    for source in source_judgments
                ],
                "overall_priority_score_source_scales": [
                    str(metric["source_scale"]) for metric in paper_priority_metrics
                ],
                "local_rank_range": round(max(normalized_ranks) - min(normalized_ranks), 6),
                "batch_judgment_count": len(source_judgments),
                "source_batch_judgments": source_judgments,
                "advance_to_final_review": any(
                    bool(source["judgment"].get("advance_to_final_review"))
                    for source in source_judgments
                ),
            }
        )
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["mean_normalized_local_rank"],
            -number(row.get("overall_priority_score")),
            -number(row.get("mean_batch_normalized_priority_score")),
            row["input_cheap_rank"],
            row["paper_id"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {
        "evaluation_mode": "pass2_strong_resumable_batch_ranking",
        "aggregation": PASS2_BATCHED_AGGREGATION_VERSION,
        "tie_breaking": [
            "higher_calibrated_overall_priority_score",
            "higher_mean_batch_normalized_priority_score",
            "better_input_cheap_rank",
            "paper_id",
        ],
        "priority_score_calibration": {
            "reported_score": "canonical 0-10 score; 0-100 batches are divided by 10",
            "ranking_tie_break": (
                "mean canonical score, then per-batch min-max normalized canonical score"
            ),
            "raw_scores_preserved": True,
        },
        "ranked_papers": rows,
        "category_rankings": build_pass2_category_rankings(rows),
    }


def calibrate_pass2_batch_priority_scores(
    judgments: list[dict[str, Any]],
) -> dict[tuple[int, int, str], dict[str, float | str]]:
    by_batch: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for judgment in judgments:
        batch_key = (int(judgment["partition_index"]), int(judgment["batch_index"]))
        by_batch.setdefault(batch_key, []).append(judgment)

    calibrated: dict[tuple[int, int, str], dict[str, float | str]] = {}
    for (partition_index, batch_index), batch_judgments in by_batch.items():
        raw_scores = [
            number(judgment["judgment"].get("overall_priority_score"))
            for judgment in batch_judgments
        ]
        minimum = min(raw_scores)
        maximum = max(raw_scores)
        if 0.0 <= minimum and maximum <= 10.0:
            source_scale = "0_10"
            canonical_scores = raw_scores
        elif 10.0 < minimum and maximum <= 100.0:
            source_scale = "0_100"
            canonical_scores = [score / 10.0 for score in raw_scores]
        else:
            source_scale = "batch_minmax_fallback"
            span = maximum - minimum
            canonical_scores = [
                5.0 if span == 0.0 else 10.0 * (score - minimum) / span
                for score in raw_scores
            ]

        canonical_minimum = min(canonical_scores)
        canonical_maximum = max(canonical_scores)
        canonical_span = canonical_maximum - canonical_minimum
        for judgment, raw_score, canonical_score in zip(
            batch_judgments,
            raw_scores,
            canonical_scores,
            strict=True,
        ):
            normalized_score = (
                0.5
                if canonical_span == 0.0
                else (canonical_score - canonical_minimum) / canonical_span
            )
            key = (
                partition_index,
                batch_index,
                str(judgment["paper_id"]),
            )
            calibrated[key] = {
                "raw_score": raw_score,
                "canonical_score": canonical_score,
                "within_batch_normalized_score": normalized_score,
                "source_scale": source_scale,
            }
    return calibrated


def build_pass2_category_rankings(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    rankings = {contribution_class: [] for contribution_class in CONTRIBUTION_CLASSES}
    for row in rows:
        contribution_class = str(row.get("primary_contribution_class") or "other")
        if contribution_class not in rankings:
            contribution_class = "other"
        rankings[contribution_class].append(str(row["paper_id"]))
    return rankings


def read_contribution_class_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        str(row["paper_id"]): str(row.get("primary_contribution_class") or "other")
        for row in rows
        if row.get("paper_id")
    }


def make_pass2_partitions(
    candidates: list[RankingCandidate],
    *,
    batch_size: int,
    partition_count: int,
    strategy: str,
    class_by_id: dict[str, str],
    seed: int,
) -> list[list[list[RankingCandidate]]]:
    partitions: list[list[list[RankingCandidate]]] = []
    for partition_index in range(partition_count):
        ordered = order_pass2_candidates(
            candidates,
            strategy=strategy,
            class_by_id=class_by_id,
            seed=seed + partition_index,
            force_shuffle=partition_index > 0,
        )
        partitions.append(balance_pass2_batches(ordered, batch_size))
    return partitions


def order_pass2_candidates(
    candidates: list[RankingCandidate],
    *,
    strategy: str,
    class_by_id: dict[str, str],
    seed: int,
    force_shuffle: bool,
) -> list[RankingCandidate]:
    rng = random.Random(seed)
    ordered = list(candidates)
    if strategy == "sequential" and not force_shuffle:
        return ordered
    if strategy in {"sequential", "shuffled"}:
        rng.shuffle(ordered)
        return ordered
    if strategy != "class_round_robin":
        raise ValueError(f"unsupported batch strategy: {strategy}")
    by_class: dict[str, list[RankingCandidate]] = {}
    for candidate in ordered:
        contribution_class = class_by_id.get(
            candidate.record.paper_id,
            contribution_class_from_pass1(candidate.pass1_row),
        )
        by_class.setdefault(contribution_class, []).append(candidate)
    for rows in by_class.values():
        rng.shuffle(rows)
    class_names = sorted(by_class)
    rng.shuffle(class_names)
    result: list[RankingCandidate] = []
    while any(by_class.values()):
        for class_name in class_names:
            rows = by_class[class_name]
            if rows:
                result.append(rows.pop())
    return result


def contribution_class_from_pass1(row: dict[str, Any]) -> str:
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    profile = scores.get("contribution_profile") if isinstance(scores.get("contribution_profile"), dict) else {}
    return str(profile.get("primary_contribution_class") or row.get("primary_contribution_class") or "other")


def balance_pass2_batches(
    candidates: list[RankingCandidate],
    batch_size: int,
) -> list[list[RankingCandidate]]:
    if not candidates:
        return []
    batch_count = max(1, (len(candidates) + batch_size - 1) // batch_size)
    if batch_count > 1 and len(candidates) // batch_count < 2:
        batch_count -= 1
    base_size, larger_count = divmod(len(candidates), batch_count)
    batches: list[list[RankingCandidate]] = []
    offset = 0
    for batch_index in range(batch_count):
        size = base_size + (1 if batch_index < larger_count else 0)
        batches.append(candidates[offset : offset + size])
        offset += size
    return batches


def run_pass2_two_stage(
    *,
    manifest: Path,
    pass1_path: Path,
    out: Path,
    run_dir: Path,
    config: Pass2TwoStageConfig,
    client: ChatCompletionClient | None,
) -> dict[str, Any]:
    started = time.monotonic()
    run_dir.mkdir(parents=True, exist_ok=True)
    stage1_out = run_dir / "stage1_ranking.json"
    stage1_rows = run_dir / "stage1_candidates_for_stage2.jsonl"
    stage2_out = run_dir / "stage2_final_ranking.json"
    stage1_result = run_pass2_ranking(
        manifest=manifest,
        pass1_path=pass1_path,
        out=stage1_out,
        run_dir=run_dir / "stage1",
        config=Pass2RankingConfig(
            provider=config.provider,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            prompt_version=config.stage1_prompt_version,
            text_source=config.text_source,
            top_fraction=1.0,
            limit=config.stage1_limit,
            per_paper_char_budget=config.stage1_per_paper_char_budget,
            temperature=config.temperature,
            max_output_tokens=config.stage1_max_output_tokens,
            seed=config.seed,
            dry_run=config.dry_run,
        ),
        client=client,
    )
    if stage1_result["status"] not in {"ok", "dry_run"}:
        result = {
            "status": "stage1_failed",
            "manifest": str(manifest),
            "pass1_path": str(pass1_path),
            "out": str(out),
            "run_dir": str(run_dir),
            "stage1_out": str(stage1_out),
            "stage1_status": stage1_result["status"],
            "stage1_result": stage1_result,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        write_json(out, result)
        write_json(run_dir / "run.json", result)
        return result
    if config.dry_run:
        result = {
            "status": "dry_run",
            "manifest": str(manifest),
            "pass1_path": str(pass1_path),
            "out": str(out),
            "run_dir": str(run_dir),
            "provider": config.provider,
            "model": config.model,
            "reasoning_effort": config.reasoning_effort,
            "stage1_prompt_version": config.stage1_prompt_version,
            "stage2_prompt_version": config.stage2_prompt_version,
            "stage1_limit": config.stage1_limit,
            "stage2_limit": config.stage2_limit,
            "stage1_out": str(stage1_out),
            "stage1_status": stage1_result["status"],
            "stage1_result": stage1_result,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        write_json(out, result)
        write_json(run_dir / "run.json", result)
        return result

    converted_count = write_ranking_score_rows(stage1_result, stage1_rows)
    stage2_result = run_pass2_ranking(
        manifest=manifest,
        pass1_path=stage1_rows,
        out=stage2_out,
        run_dir=run_dir / "stage2",
        config=Pass2RankingConfig(
            provider=config.provider,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            prompt_version=config.stage2_prompt_version,
            text_source=config.text_source,
            top_fraction=1.0,
            limit=config.stage2_limit,
            per_paper_char_budget=config.stage2_per_paper_char_budget,
            temperature=config.temperature,
            max_output_tokens=config.stage2_max_output_tokens,
            seed=config.seed,
            dry_run=config.dry_run,
        ),
        client=client,
    )
    result = {
        "status": "ok" if stage2_result["status"] in {"ok", "dry_run"} else "stage2_failed",
        "manifest": str(manifest),
        "pass1_path": str(pass1_path),
        "out": str(out),
        "run_dir": str(run_dir),
        "provider": config.provider,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "stage1_prompt_version": config.stage1_prompt_version,
        "stage2_prompt_version": config.stage2_prompt_version,
        "stage1_limit": config.stage1_limit,
        "stage2_limit": config.stage2_limit,
        "stage1_out": str(stage1_out),
        "stage1_rows": str(stage1_rows),
        "stage1_converted_rows": converted_count,
        "stage1_status": stage1_result["status"],
        "stage2_out": str(stage2_out),
        "stage2_status": stage2_result["status"],
        "stage1_result": stage1_result,
        "stage2_result": stage2_result,
        "ranking": stage2_result.get("ranking"),
        "usage": {
            "stage1": stage1_result.get("usage", {}),
            "stage2": stage2_result.get("usage", {}),
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(out, result)
    write_json(run_dir / "run.json", result)
    return result


def write_ranking_score_rows(result: dict[str, Any], out: Path) -> int:
    ranking = result.get("ranking")
    if not isinstance(ranking, dict):
        ranking = load_ranking_payload(Path(result["out"]))
    ranked = ranking.get("ranked_papers") if isinstance(ranking, dict) else None
    if not isinstance(ranked, list):
        raise ValueError("ranking result missing ranked_papers")
    rows = [
        ranking_item_to_score_row(
            item,
            provider=str(result.get("provider") or ""),
            model=str(result.get("model") or ""),
            prompt_version=str(result.get("prompt_version") or ""),
            source_ranking_path=str(result.get("out") or ""),
        )
        for item in ranked
        if isinstance(item, dict)
    ]
    return write_jsonl(out, rows)


def ranking_item_to_score_row(
    item: dict[str, Any],
    *,
    provider: str,
    model: str,
    prompt_version: str,
    source_ranking_path: str,
) -> dict[str, Any]:
    rank = int(number(item.get("rank")) or 10**9)
    overall = number(item.get("overall_priority_score"))
    if overall <= 0:
        overall = max(0.0, 10.0 - (rank - 1) * 0.25)
    broad = number(item.get("broad_scientific_impact_score"))
    ml = number(item.get("ml_field_impact_score"))
    soundness = number(item.get("technical_soundness_score"))
    contribution_class = str(item.get("primary_contribution_class") or "other")
    if contribution_class not in CONTRIBUTION_CLASSES:
        contribution_class = "other"
    percentile = max(0.0, min(100.0, 100.0 - (rank - 1) * 2.0))
    return {
        "status": "ok",
        "paper_id": str(item.get("paper_id") or ""),
        "title": item.get("title") or "",
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
        "rank": rank,
        "source_ranking_path": source_ranking_path,
        "scores": {
            "contribution_profile": {
                "primary_contribution_class": contribution_class,
                "secondary_contribution_classes": item.get("secondary_contribution_classes") or [],
            },
            "scores": {
                "executive_ac_priority": overall,
                "broader_science_impact_forecast": broad,
                "ml_field_impact_forecast": ml,
                "overall_significance": overall,
                "technical_soundness": soundness,
            },
            "calibration": {
                "estimated_percentile_among_accepted_papers": percentile,
                "triage_bucket": bucket_from_rank(rank),
                "why_not_higher": item.get("main_risk") or "",
                "why_not_lower": item.get("why_ranked_here") or "",
            },
            "ranking_signals": {
                "should_advance_to_strong_model": bool(item.get("advance_to_final_review")) or rank <= 25,
                "top_paper_case": item.get("top_paper_case") or item.get("best_case_for_impact") or "",
                "broad_scientific_impact_argument": item.get("why_ranked_here") or "",
                "ml_field_impact_argument": item.get("why_ranked_here") or "",
                "dealbreaker_risks": item.get("main_risk") or "",
            },
        },
    }


def bucket_from_rank(rank: int) -> str:
    if rank <= 5:
        return "top_10_percent"
    if rank <= 15:
        return "top_quartile"
    if rank <= 25:
        return "above_average"
    if rank <= 40:
        return "middle"
    return "lower_priority"


def run_reference_ranking(
    *,
    manifest: Path,
    out: Path,
    run_dir: Path,
    config: ReferenceRankingConfig,
    client: ChatCompletionClient | None,
) -> dict[str, Any]:
    started = time.monotonic()
    prepare_ranking_run_dir(run_dir)
    candidates = select_reference_candidates(
        manifest=manifest,
        text_source=config.text_source,
        paper_ids=config.paper_ids,
        limit=config.limit,
        per_paper_char_budget=config.per_paper_char_budget,
    )
    messages = build_reference_ranking_messages(candidates, config=config)
    prompt_payload = {
        "prompt_version": config.prompt_version,
        "provider": config.provider,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "paper_set_name": config.paper_set_name,
        "candidate_count": len(candidates),
        "candidate_ids": [candidate.record.paper_id for candidate in candidates],
        "text_source": config.text_source,
        "messages": messages,
        "prompt_tokens_estimate": sum(estimate_tokens(message["content"]) for message in messages),
    }
    prompt_path = run_dir / "prompt.json"
    write_json(prompt_path, prompt_payload)

    base_result: dict[str, Any] = {
        "status": "dry_run" if config.dry_run else "pending",
        "provider": config.provider,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "prompt_version": config.prompt_version,
        "paper_set_name": config.paper_set_name,
        "manifest": str(manifest),
        "out": str(out),
        "run_dir": str(run_dir),
        "candidate_count": len(candidates),
        "candidate_ids": [candidate.record.paper_id for candidate in candidates],
        "candidate_titles": {candidate.record.paper_id: candidate.record.title for candidate in candidates},
        "prompt_path": str(prompt_path),
        "raw_response_path": None,
        "parsed_response_path": None,
        "validation_errors": [],
        "elapsed_seconds": None,
    }
    if config.dry_run:
        result = {**base_result, "elapsed_seconds": round(time.monotonic() - started, 3)}
        write_json(out, result)
        write_json(run_dir / "run.json", result)
        return result
    if client is None:
        raise ValueError("client is required unless dry_run=True")

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
        canonicalize_reference_ids(parsed, candidates)
        validation_errors = validate_reference_ranking(parsed, candidates)
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
    except Exception as exc:  # noqa: BLE001 - persist the failed ranking attempt.
        error_path = run_dir / "error.json"
        write_json(error_path, {"error": repr(exc)})
        result = {
            **base_result,
            "status": "failed",
            "error_path": str(error_path),
            "error": repr(exc),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    write_json(out, result)
    write_json(run_dir / "run.json", result)
    return result


def run_reference_repair(
    *,
    manifest: Path,
    reference_path: Path,
    out: Path,
    run_dir: Path,
    config: ReferenceRepairConfig,
    client: ChatCompletionClient | None,
) -> dict[str, Any]:
    started = time.monotonic()
    prepare_ranking_run_dir(run_dir)
    candidates = select_reference_candidates(
        manifest=manifest,
        text_source=config.text_source,
        paper_ids=set(),
        limit=None,
        per_paper_char_budget=config.per_paper_char_budget,
    )
    source_payload = load_ranking_payload(reference_path)
    canonicalize_reference_ids(source_payload, candidates)
    source_validation_errors = validate_reference_ranking(source_payload, candidates)
    messages = build_reference_repair_messages(
        source_payload,
        candidates=candidates,
        validation_errors=source_validation_errors,
        config=config,
    )
    prompt_payload = {
        "prompt_version": config.prompt_version,
        "provider": config.provider,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "paper_set_name": config.paper_set_name,
        "candidate_count": len(candidates),
        "candidate_ids": [candidate.record.paper_id for candidate in candidates],
        "text_source": config.text_source,
        "reference_path": str(reference_path),
        "source_validation_errors": source_validation_errors,
        "messages": messages,
        "prompt_tokens_estimate": sum(estimate_tokens(message["content"]) for message in messages),
    }
    prompt_path = run_dir / "prompt.json"
    write_json(prompt_path, prompt_payload)
    base_result: dict[str, Any] = {
        "status": "dry_run" if config.dry_run else "pending",
        "provider": config.provider,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "prompt_version": config.prompt_version,
        "paper_set_name": config.paper_set_name,
        "manifest": str(manifest),
        "reference_path": str(reference_path),
        "out": str(out),
        "run_dir": str(run_dir),
        "candidate_count": len(candidates),
        "candidate_ids": [candidate.record.paper_id for candidate in candidates],
        "source_validation_errors": source_validation_errors,
        "prompt_path": str(prompt_path),
        "raw_response_path": None,
        "parsed_response_path": None,
        "validation_errors": [],
        "elapsed_seconds": None,
    }
    if config.dry_run:
        result = {**base_result, "elapsed_seconds": round(time.monotonic() - started, 3)}
        write_json(out, result)
        write_json(run_dir / "run.json", result)
        return result
    if client is None:
        raise ValueError("client is required unless dry_run=True")

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
        canonicalize_reference_ids(parsed, candidates)
        validation_errors = validate_reference_ranking(parsed, candidates)
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
    except Exception as exc:  # noqa: BLE001 - persist failed repair attempts.
        error_path = run_dir / "error.json"
        write_json(error_path, {"error": repr(exc)})
        result = {
            **base_result,
            "status": "failed",
            "error_path": str(error_path),
            "error": repr(exc),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    write_json(out, result)
    write_json(run_dir / "run.json", result)
    return result


def prepare_ranking_run_dir(run_dir: Path) -> None:
    ensure_parent(run_dir / ".keep")
    run_dir.mkdir(parents=True, exist_ok=True)


def select_candidates(
    *,
    manifest: Path,
    pass1_path: Path,
    text_source: str,
    top_fraction: float,
    limit: int | None,
    per_paper_char_budget: int,
) -> list[RankingCandidate]:
    records_by_id = {record.paper_id: record for record in read_paper_records(manifest)}
    rows = [json.loads(line) for line in pass1_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ok_rows = [row for row in rows if row.get("status") == "ok" and isinstance(row.get("scores"), dict)]
    ok_rows.sort(key=pass1_sort_key, reverse=True)
    if limit is None:
        limit = max(1, round(len(ok_rows) * top_fraction))
    selected_rows = ok_rows[:limit]
    candidates: list[RankingCandidate] = []
    for row in selected_rows:
        paper_id = str(row.get("paper_id") or "")
        record = records_by_id.get(paper_id)
        if record is None:
            continue
        text_path, resolved_text_source = resolve_record_text_path(record, text_source)
        paper_text = truncate_candidate_text(read_text_path(text_path), per_paper_char_budget)
        candidates.append(
            RankingCandidate(
                record=record,
                pass1_row=row,
                text_path=text_path,
                resolved_text_source=resolved_text_source,
                paper_text=paper_text,
            )
        )
    return candidates


def select_reference_candidates(
    *,
    manifest: Path,
    text_source: str,
    paper_ids: set[str],
    limit: int | None,
    per_paper_char_budget: int,
) -> list[ReferenceCandidate]:
    candidates: list[ReferenceCandidate] = []
    for record in read_paper_records(manifest):
        if paper_ids and record.paper_id not in paper_ids:
            continue
        if record.parse_status != "ok":
            continue
        text_path, resolved_text_source = resolve_record_text_path(record, text_source)
        paper_text = truncate_reference_text(read_text_path(text_path), per_paper_char_budget)
        candidates.append(ReferenceCandidate(record, text_path, resolved_text_source, paper_text))
        if limit is not None and len(candidates) >= limit:
            break
    return candidates


def pass1_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    core = scores.get("scores") if isinstance(scores.get("scores"), dict) else {}
    calibration = scores.get("calibration") if isinstance(scores.get("calibration"), dict) else {}
    return (
        number(core.get("executive_ac_priority")),
        number(calibration.get("estimated_percentile_among_accepted_papers")) / 10.0,
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


def truncate_candidate_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n\n[TRUNCATED for pass-2 ranking at {limit} characters]"


def truncate_reference_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n\n[TRUNCATED for reference ranking at {limit} characters]"


def build_pass2_ranking_messages(
    candidates: list[RankingCandidate],
    *,
    config: Pass2RankingConfig,
) -> list[dict[str, str]]:
    system = """You are the strong-model second-stage judge for a machine learning conference paper study.

Rank candidate papers by likely broad scientific and machine-learning impact. Use only the supplied paper text and first-pass outputs. Ignore author identity, institution, venue prestige, and outside knowledge.
No reviews, reviewer scores, area-chair comments, decisions, presentation tiers, or awards are provided. Do not infer paper quality from possible venue or outcome cues in the document.

This run is text-only and retrieval-free: you do not have PDF page images, figure images, table images, file search, web search, vector stores, or external tools. Treat missing visual/table detail as an uncertainty, not as a factual flaw.

Your job is comparative ranking, not independent accept/reject reviewing. A technically clean but incremental paper should rank below a paper with a plausible path to broad scientific influence, even if both are acceptable ICML papers.

Return only valid JSON."""
    user = f"""Rank the candidate papers selected by the prior pipeline stage.

Primary objective:
- Identify which papers best deserve expensive human/strong-model attention for broad scientific impact, sweeping ML-field importance, and long-run contribution.
- Return exactly {len(candidates)} `ranked_papers` entries: one for every candidate paper_id, with unique ranks 1..{len(candidates)}.
- Candidate paper_ids that must appear exactly once: {", ".join(candidate.record.paper_id for candidate in candidates)}

Secondary objective:
- Audit the prior-stage shortlist. Say whether it surfaced genuinely strong broad-impact candidates or mostly reviewer-polished incremental work.
- Note any paper that the prior stage may have overpromoted or underpromoted from the evidence provided here.

Ranking rubric:
- broad_scientific_impact: could this enable scientific discovery, measurement, modeling, safety, or methodology beyond a narrow subcommunity?
- ml_field_impact: could this reshape common ML practice, theory, evaluation, systems, or benchmarks?
- soundness_gate: if core evidence is weak, cap rank even when the idea is attractive.
- novelty_gate: incremental combinations should not outrank field-shaping ideas unless they are exceptionally useful.
- evidence_gate: distinguish visible evidence from merely plausible impact.
- contribution_class: classify the main impact route. Benchmarks, datasets, infrastructure, safety/evals, scientific tools, theory, and algorithms can all be high-impact, but should also be compared within class.

Contribution classes:
{", ".join(CONTRIBUTION_CLASSES)}

Required JSON shape:
{{
  "evaluation_mode": "pass2_strong_batch_ranking",
  "prompt_version": "{config.prompt_version}",
  "ranked_papers": [
    {{
      "rank": 1,
      "paper_id": "paper id",
      "title": "title",
      "broad_scientific_impact_score": 1,
      "ml_field_impact_score": 1,
      "technical_soundness_score": 1,
      "overall_priority_score": 1,
      "primary_contribution_class": "core_ml_algorithm",
      "secondary_contribution_classes": ["scientific_modeling_tool"],
      "advance_to_final_review": true,
      "why_ranked_here": "evidence-grounded comparative reason",
      "top_paper_case": "best case for broad importance",
      "main_risk": "main reason this might not matter"
    }}
  ],
  "first_pass_quality_assessment": {{
    "did_prior_stage_surface_broad_impact_candidates": "yes/no/mixed plus explanation",
    "prior_stage_overpromoted": ["paper_id with reason"],
    "prior_stage_underpromoted": ["paper_id with reason"],
    "recommended_first_pass_changes": ["prompt or representation changes"]
  }},
  "category_rankings": {{
    "core_ml_algorithm": ["paper_id", "paper_id"],
    "benchmark_dataset": ["paper_id"],
    "infrastructure_systems": ["paper_id"],
    "scientific_modeling_tool": ["paper_id"],
    "safety_governance_eval": ["paper_id"]
  }},
  "modality_limitations": {{
    "text_only_risks": ["what may be missed without PDF page images, figures, or table images"],
    "would_pdf_visual_review_change_any_rankings": "yes/no/maybe plus explanation"
  }},
  "overall_assessment": "short comparative assessment"
}}

Candidates:
{format_candidates(candidates)}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_reference_ranking_messages(
    candidates: list[ReferenceCandidate],
    *,
    config: ReferenceRankingConfig,
) -> list[dict[str, str]]:
    n = len(candidates)
    system = """You are the strongest available reference judge for a machine learning conference paper study.

Create an independent reference ranking for evaluation of a cheaper first-pass -> strong-model pipeline. Use only the supplied extracted paper text. Do not use web search, retrieval, citation memory, author identity, institution, venue prestige, or outside knowledge.
No reviews, reviewer scores, area-chair comments, decisions, presentation tiers, or awards are provided. Do not infer paper quality from possible venue or outcome cues in the document.

This run is text-only and retrieval-free: you do not have PDF page images, figure images, table images, file search, vector stores, or external tools. Treat missing visual/table detail as an uncertainty, not as a factual flaw.

Rank by likely broad scientific and machine-learning impact, with technical soundness as a gate. Your job is comparative prioritization among accepted papers, not accept/reject reviewing.

Return only valid JSON."""
    user = f"""Create a reference gold ranking for these {n} ICML accepted papers.

This artifact will be used to evaluate whether cheaper batch triage and strong-model reranking preserve the papers a stronger judge would prioritize. It is not a claim of objective truth; it is a reproducible strong-judge reference.

Primary objective:
- Rank every candidate from 1 to {n}; no ties, no missing papers.
- Identify papers with the strongest expected broad scientific impact and ML-field impact.
- Give higher ranks to work that could reshape methods, evaluation, theory, scientific practice, safety/evals, infrastructure, or durable community resources.

Ranking rules:
- Use only the text provided here.
- Do not infer merit from author identity, institution, citation memory, web knowledge, or ICML acceptance itself.
- Apply a soundness gate: weak or under-evidenced technical claims should cap the rank even when the idea is attractive.
- Apply an impact gate: polished but incremental papers should not outrank papers with a plausible path to field-shaping influence.
- Score spread matters: use the full 1-10 scale where justified, and make ranks meaningfully comparative.
- Benchmarks, datasets, infrastructure, safety/eval work, scientific tools, theory, and algorithms can all be high-impact, but also rank each paper within its primary contribution class.

Contribution classes:
{", ".join(CONTRIBUTION_CLASSES)}

Required JSON shape:
{{
  "evaluation_mode": "reference_gold_ranking",
  "prompt_version": "{config.prompt_version}",
  "judge": "{config.provider}:{config.model}",
  "paper_set": "{config.paper_set_name}",
  "modality": "text_only_extracted_main_paper",
  "retrieval": "off",
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
      "novelty_score": 9,
      "evidence_confidence_score": 8,
      "advance_to_final_review": true,
      "two_year_adoption_path": "who uses it and what changes",
      "why_ranked_here": "evidence-grounded comparative reason",
      "best_case_for_impact": "strongest plausible impact route",
      "main_risk": "main reason this may not matter or may not hold up"
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
    "top_20_threshold": "what distinguished the top 20",
    "common_overranking_risks": ["risks that cheaper models may overvalue"],
    "common_underranking_risks": ["risks that cheaper models may miss"]
  }},
  "modality_limitations": {{
    "text_only_risks": ["what may be missed without PDF page images, figures, or table images"],
    "would_pdf_visual_review_change_any_rankings": "yes/no/maybe plus explanation"
  }},
  "overall_assessment": "short comparative assessment of this 50-paper set"
}}

Candidate paper_ids that must appear exactly once:
{", ".join(candidate.record.paper_id for candidate in candidates)}

Papers:
{format_reference_candidates(candidates)}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_reference_repair_messages(
    source_payload: dict[str, Any],
    *,
    candidates: list[ReferenceCandidate],
    validation_errors: list[str],
    config: ReferenceRepairConfig,
) -> list[dict[str, str]]:
    expected_ids = [candidate.record.paper_id for candidate in candidates]
    rows = source_payload.get("ranked_papers")
    rows = rows if isinstance(rows, list) else []
    row_ids = [str(row.get("paper_id") or "") for row in rows if isinstance(row, dict)]
    row_id_set = set(row_ids)
    expected_id_set = set(expected_ids)
    missing_ids = sorted(expected_id_set - row_id_set)
    duplicate_ids = sorted({paper_id for paper_id in row_ids if row_ids.count(paper_id) > 1})
    source_excerpt = {
        "evaluation_mode": source_payload.get("evaluation_mode"),
        "prompt_version": source_payload.get("prompt_version"),
        "judge": source_payload.get("judge"),
        "paper_set": source_payload.get("paper_set"),
        "modality": source_payload.get("modality"),
        "retrieval": source_payload.get("retrieval"),
        "ranked_papers": rows,
        "category_rankings": source_payload.get("category_rankings"),
        "calibration_notes": source_payload.get("calibration_notes"),
        "modality_limitations": source_payload.get("modality_limitations"),
        "overall_assessment": source_payload.get("overall_assessment"),
    }
    missing_texts = [
        format_reference_candidate(candidate)
        for candidate in candidates
        if candidate.record.paper_id in missing_ids
    ]
    duplicate_texts = [
        format_reference_candidate(candidate)
        for candidate in candidates
        if candidate.record.paper_id in duplicate_ids
    ]
    system = """You are repairing a structurally invalid reference ranking for a machine learning conference paper study.

Use only the supplied existing ranking and supplied paper text. Do not use web search, retrieval, citation memory, author identity, institution, venue prestige, or outside knowledge.

Your task is to fix exact paper coverage while preserving the ranking's comparative intent where possible. Return only valid JSON."""
    user = f"""Repair this reference ranking for {config.paper_set_name}.

The previous model response failed structural validation:
{json.dumps(validation_errors, ensure_ascii=False, indent=2)}

Expected candidate paper_ids, each exactly once:
{", ".join(expected_ids)}

Repair rules:
- Return the full reference ranking JSON, not a patch.
- `ranked_papers` must contain exactly {len(candidates)} entries.
- Every expected paper_id must appear exactly once; no extras, no duplicates.
- Ranks must be exactly 1..{len(candidates)} with no ties.
- Insert each missing paper at the rank justified by its text and the current ranking ladder.
- Remove duplicate rows. If a duplicated paper appears twice, keep the entry whose rank/reason is better supported by the current ranking context unless the lower entry clearly belongs to a missing paper.
- Preserve the existing order of unrelated papers unless the missing paper's placement requires renumbering.
- Keep the same schema as the source ranking and update category rankings to match the repaired `ranked_papers`.
- Mark `"prompt_version": "{config.prompt_version}"`, `"judge": "{config.provider}:{config.model}"`, `"paper_set": "{config.paper_set_name}"`, `"modality": "text_only_extracted_main_paper"`, and `"retrieval": "off"`.

Contribution classes:
{", ".join(CONTRIBUTION_CLASSES)}

Current invalid ranking:
<<<CURRENT_RANKING_JSON
{json.dumps(source_excerpt, ensure_ascii=False, indent=2)}
CURRENT_RANKING_JSON

Missing paper text:
{chr(10).join(missing_texts) if missing_texts else "[none]"}

Duplicated paper text:
{chr(10).join(duplicate_texts) if duplicate_texts else "[none]"}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def format_candidates(candidates: list[RankingCandidate]) -> str:
    blocks = []
    for candidate in candidates:
        blocks.append(format_candidate(candidate))
    return "\n\n".join(blocks)


def format_candidate(candidate: RankingCandidate) -> str:
    first_pass = compact_first_pass(candidate.pass1_row)
    return f"""<CANDIDATE paper_id="{candidate.record.paper_id}">
Title: {candidate.record.title or ""}
Text source: {candidate.resolved_text_source}
Text path: {candidate.text_path}

First-pass cheap-model output summary:
{json.dumps(first_pass, ensure_ascii=False, indent=2, sort_keys=True)}

Paper text:
<<<PAPER_TEXT
{candidate.paper_text}
PAPER_TEXT
</CANDIDATE>"""


def format_reference_candidates(candidates: list[ReferenceCandidate]) -> str:
    return "\n\n".join(format_reference_candidate(candidate) for candidate in candidates)


def format_reference_candidate(candidate: ReferenceCandidate) -> str:
    return f"""<PAPER paper_id="{candidate.record.paper_id}">
Title: {candidate.record.title or ""}
Text source: {candidate.resolved_text_source}
Text path: {candidate.text_path}

Paper text:
<<<PAPER_TEXT
{candidate.paper_text}
PAPER_TEXT
</PAPER>"""


def compact_first_pass(row: dict[str, Any]) -> dict[str, Any]:
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    keep = {
        "summary": scores.get("summary"),
        "scores": scores.get("scores"),
        "impact_axes": scores.get("impact_axes"),
        "calibration": scores.get("calibration"),
        "ranking_signals": scores.get("ranking_signals"),
        "evidence": scores.get("evidence"),
        "evidence_audit": scores.get("evidence_audit"),
        "uncertainty": scores.get("uncertainty"),
    }
    return {key: value for key, value in keep.items() if value is not None}


def canonicalize_ranked_paper_ids(payload: dict[str, Any], expected_ids: list[str]) -> None:
    canonical_by_lower = {paper_id.lower(): paper_id for paper_id in expected_ids}
    rows = payload.get("ranked_papers")
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        paper_id = row.get("paper_id")
        if isinstance(paper_id, str):
            canonical = canonical_by_lower.get(paper_id.lower())
            if canonical is not None:
                row["paper_id"] = canonical


def validate_ranked_paper_coverage(payload: dict[str, Any], expected_ids: list[str]) -> list[str]:
    errors: list[str] = []
    rows = payload.get("ranked_papers")
    if not isinstance(rows, list):
        return ["ranked_papers must be a list"]
    expected_id_set = set(expected_ids)
    row_ids = [str(row.get("paper_id") or "") for row in rows if isinstance(row, dict)]
    row_id_set = set(row_ids)
    duplicates = sorted({paper_id for paper_id in row_ids if row_ids.count(paper_id) > 1})
    missing = sorted(expected_id_set - row_id_set)
    extra = sorted(row_id_set - expected_id_set)
    if len(rows) != len(expected_ids):
        errors.append(f"ranked_papers length {len(rows)} does not match candidate count {len(expected_ids)}")
    if duplicates:
        errors.append("duplicate paper_ids: " + ", ".join(duplicates))
    if missing:
        errors.append("missing paper_ids: " + ", ".join(missing))
    if extra:
        errors.append("extra paper_ids: " + ", ".join(extra))
    ranks: list[int] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"ranked_papers[{idx}] must be an object")
            continue
        rank = row.get("rank")
        if not isinstance(rank, int):
            errors.append(f"ranked_papers[{idx}].rank must be an integer")
        else:
            ranks.append(rank)
    expected_ranks = list(range(1, len(expected_ids) + 1))
    if sorted(ranks) != expected_ranks:
        errors.append(f"ranks must be exactly 1..{len(expected_ids)} with no ties")
    return errors


def canonicalize_reference_ids(payload: dict[str, Any], candidates: list[ReferenceCandidate]) -> None:
    canonical_by_lower = {candidate.record.paper_id.lower(): candidate.record.paper_id for candidate in candidates}
    rows = payload.get("ranked_papers")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            paper_id = row.get("paper_id")
            if isinstance(paper_id, str):
                canonical = canonical_by_lower.get(paper_id.lower())
                if canonical is not None:
                    row["paper_id"] = canonical
    category_rankings = payload.get("category_rankings")
    if isinstance(category_rankings, dict):
        for key, values in category_rankings.items():
            if not isinstance(values, list):
                continue
            category_rankings[key] = [
                canonical_by_lower.get(value.lower(), value) if isinstance(value, str) else value
                for value in values
            ]


def validate_reference_ranking(payload: dict[str, Any], candidates: list[ReferenceCandidate]) -> list[str]:
    errors: list[str] = []
    expected_ids = [candidate.record.paper_id for candidate in candidates]
    expected_id_set = set(expected_ids)
    rows = payload.get("ranked_papers")
    if not isinstance(rows, list):
        return ["ranked_papers must be a list"]
    if len(rows) != len(candidates):
        errors.append(f"ranked_papers length {len(rows)} does not match candidate count {len(candidates)}")

    row_ids = [str(row.get("paper_id") or "") for row in rows if isinstance(row, dict)]
    row_id_set = set(row_ids)
    duplicates = sorted({paper_id for paper_id in row_ids if row_ids.count(paper_id) > 1})
    missing = sorted(expected_id_set - row_id_set)
    extra = sorted(row_id_set - expected_id_set)
    if duplicates:
        errors.append("duplicate paper_ids: " + ", ".join(duplicates))
    if missing:
        errors.append("missing paper_ids: " + ", ".join(missing))
    if extra:
        errors.append("extra paper_ids: " + ", ".join(extra))

    ranks: list[int] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"ranked_papers[{idx}] must be an object")
            continue
        rank = row.get("rank")
        if not isinstance(rank, int):
            errors.append(f"ranked_papers[{idx}].rank must be an integer")
        else:
            ranks.append(rank)
        contribution_class = row.get("primary_contribution_class")
        if contribution_class is not None and contribution_class not in CONTRIBUTION_CLASSES:
            errors.append(
                f"ranked_papers[{idx}].primary_contribution_class must be one of: "
                + ", ".join(CONTRIBUTION_CLASSES)
            )
    expected_ranks = list(range(1, len(candidates) + 1))
    if sorted(ranks) != expected_ranks:
        errors.append(f"ranks must be exactly 1..{len(candidates)} with no ties")
    return errors


def load_ranking_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("ranking"), dict):
        return payload["ranking"]
    if not isinstance(payload, dict):
        raise ValueError(f"ranking payload must be an object: {path}")
    return payload
