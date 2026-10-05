from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from icml_ai_ac.models import PaperRecord
from icml_ai_ac.scoring.providers import (
    ChatCompletionClient,
    effective_openrouter_reasoning_effort,
    extract_content,
    is_batch_blocking_provider_error,
    response_model,
)
from icml_ai_ac.scoring.runner import estimate_tokens, read_text_path, resolve_record_text_path
from icml_ai_ac.scoring.schema import CONTRIBUTION_CLASSES, parse_json_response
from icml_ai_ac.scoring.usage import sum_usage
from icml_ai_ac.storage import read_json, read_paper_records, write_json, write_jsonl


PASS1_BATCH_PROMPT_VERSION = "pass1_cheap_forced_batch_rank_v2"


@dataclass(slots=True)
class BatchCandidate:
    record: PaperRecord
    text_path: str
    resolved_text_source: str
    paper_text: str


@dataclass(slots=True)
class Pass1BatchConfig:
    provider: str
    model: str
    prompt_version: str
    text_source: str
    limit: int | None
    paper_ids: set[str]
    per_paper_char_budget: int
    temperature: float
    max_output_tokens: int
    seed: int | None
    dry_run: bool


@dataclass(slots=True)
class Pass1BatchSuiteConfig:
    provider: str
    model: str
    reasoning_effort: str | None
    prompt_version: str
    text_source: str
    limit: int | None
    paper_ids: set[str]
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


def run_pass1_batch_rank(
    *,
    manifest: Path,
    out: Path,
    run_dir: Path,
    config: Pass1BatchConfig,
    client: ChatCompletionClient | None,
) -> dict[str, Any]:
    started = time.monotonic()
    run_dir.mkdir(parents=True, exist_ok=True)
    candidates = select_batch_candidates(
        manifest=manifest,
        text_source=config.text_source,
        paper_ids=config.paper_ids,
        limit=config.limit,
        per_paper_char_budget=config.per_paper_char_budget,
    )
    messages = build_pass1_batch_messages(candidates, config=config)
    prompt_payload = {
        "prompt_version": config.prompt_version,
        "provider": config.provider,
        "model": config.model,
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
        "prompt_version": config.prompt_version,
        "manifest": str(manifest),
        "out": str(out),
        "run_dir": str(run_dir),
        "candidate_count": len(candidates),
        "candidate_ids": [candidate.record.paper_id for candidate in candidates],
        "prompt_path": str(prompt_path),
        "elapsed_seconds": None,
    }
    if config.dry_run:
        rows = build_dry_run_rows(candidates, config=config, prompt_path=prompt_path)
        write_jsonl(out, rows)
        result = {**base_result, "status": "dry_run", "elapsed_seconds": round(time.monotonic() - started, 3)}
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
        parsed = parse_batch_json_response(chat.content, candidates=candidates)
        parsed_response_path = run_dir / "parsed.json"
        write_json(parsed_response_path, parsed)
        rows = batch_response_to_rows(
            parsed,
            candidates=candidates,
            config=config,
            prompt_path=prompt_path,
            raw_response_path=raw_response_path,
            parsed_response_path=parsed_response_path,
            usage=chat.usage,
        )
        for row in rows:
            row["served_model"] = chat.served_model
        write_jsonl(out, rows)
        result = {
            **base_result,
            "status": "ok",
            "raw_response_path": str(raw_response_path),
            "parsed_response_path": str(parsed_response_path),
            "served_model": chat.served_model,
            "usage": chat.usage,
            "provider_elapsed_seconds": round(chat.elapsed_seconds, 3),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "rows": len(rows),
        }
    except Exception as exc:  # noqa: BLE001 - batch failure is persisted.
        error_path = run_dir / "error.json"
        write_json(error_path, {"error": repr(exc)})
        result = {**base_result, "status": "failed", "error": repr(exc), "error_path": str(error_path)}
    write_json(run_dir / "run.json", result)
    return result


def run_pass1_batch_suite(
    *,
    manifest: Path,
    out: Path,
    run_dir: Path,
    config: Pass1BatchSuiteConfig,
    client: ChatCompletionClient | None,
) -> dict[str, Any]:
    started = time.monotonic()
    run_dir.mkdir(parents=True, exist_ok=True)
    candidates = select_batch_candidates(
        manifest=manifest,
        text_source=config.text_source,
        paper_ids=config.paper_ids,
        limit=config.limit,
        per_paper_char_budget=config.per_paper_char_budget,
    )
    class_by_id = read_class_map(config.class_path)
    partitions = make_batch_partitions(candidates, config=config, class_by_id=class_by_id)
    effective_reasoning_effort = (
        effective_openrouter_reasoning_effort(
            model=config.model,
            requested_effort=config.reasoning_effort,
        )
        if config.provider == "openrouter"
        else config.reasoning_effort
    )
    base_result: dict[str, Any] = {
        "status": "dry_run" if config.dry_run else "pending",
        "provider": config.provider,
        "model": config.model,
        "prompt_version": config.prompt_version,
        "manifest": str(manifest),
        "out": str(out),
        "run_dir": str(run_dir),
        "candidate_count": len(candidates),
        "candidate_ids": [candidate.record.paper_id for candidate in candidates],
        "batch_size": config.batch_size,
        "reasoning_effort": config.reasoning_effort,
        "effective_reasoning_effort": effective_reasoning_effort,
        "partitions": config.partitions,
        "strategy": config.strategy,
        "class_path": str(config.class_path) if config.class_path else None,
        "elapsed_seconds": None,
    }
    if not config.dry_run and client is None:
        raise ValueError("client is required unless dry_run=True")

    rows: list[dict[str, Any]] = []
    batch_results: list[dict[str, Any]] = []
    failures = 0
    blocked_reason: str | None = None
    batch_config = Pass1BatchConfig(
        provider=config.provider,
        model=config.model,
        prompt_version=config.prompt_version,
        text_source=config.text_source,
        limit=None,
        paper_ids=set(),
        per_paper_char_budget=config.per_paper_char_budget,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        seed=config.seed,
        dry_run=config.dry_run,
    )
    for partition_index, batches in enumerate(partitions):
        if blocked_reason:
            break
        for batch_index, batch_candidates in enumerate(batches):
            batch_dir = run_dir / "batches" / f"partition_{partition_index:02d}_batch_{batch_index:02d}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            messages = build_pass1_batch_messages(batch_candidates, config=batch_config)
            prompt_path = batch_dir / "prompt.json"
            prompt_payload = {
                "prompt_version": config.prompt_version,
                "provider": config.provider,
                "model": config.model,
                "partition_index": partition_index,
                "batch_index": batch_index,
                "batch_size": len(batch_candidates),
                "candidate_ids": [candidate.record.paper_id for candidate in batch_candidates],
                "text_source": config.text_source,
                "reasoning_effort": config.reasoning_effort,
                "effective_reasoning_effort": effective_reasoning_effort,
                "temperature": config.temperature,
                "max_output_tokens": config.max_output_tokens,
                "seed": config.seed,
                "messages": messages,
                "prompt_tokens_estimate": sum(estimate_tokens(message["content"]) for message in messages),
            }
            fingerprint = batch_prompt_fingerprint(prompt_payload)
            prompt_payload["fingerprint"] = fingerprint
            write_json(prompt_path, prompt_payload)
            if config.dry_run:
                batch_rows = build_dry_run_rows(batch_candidates, config=batch_config, prompt_path=prompt_path)
                annotate_suite_rows(batch_rows, partition_index=partition_index, batch_index=batch_index)
                rows.extend(batch_rows)
                batch_results.append(
                    {
                        "status": "dry_run",
                        "partition_index": partition_index,
                        "batch_index": batch_index,
                        "candidate_count": len(batch_candidates),
                        "fingerprint": fingerprint,
                        "prompt_path": str(prompt_path),
                    }
                )
                continue
            if client is None:
                raise RuntimeError("live batch suite requires a provider client")
            cached = load_cached_pass1_batch(
                batch_dir=batch_dir,
                fingerprint=fingerprint,
                candidates=batch_candidates,
                config=batch_config,
                prompt_path=prompt_path,
                partition_index=partition_index,
                batch_index=batch_index,
            )
            if cached is not None:
                batch_rows, batch_result = cached
                rows.extend(batch_rows)
                batch_results.append(batch_result)
                continue
            recovered = recover_saved_pass1_batch(
                batch_dir=batch_dir,
                fingerprint=fingerprint,
                candidates=batch_candidates,
                config=batch_config,
                prompt_path=prompt_path,
                partition_index=partition_index,
                batch_index=batch_index,
                effective_reasoning_effort=effective_reasoning_effort,
            )
            if recovered is not None:
                batch_rows, batch_result = recovered
                rows.extend(batch_rows)
                batch_results.append(batch_result)
                continue
            archived_attempt_count = archive_failed_pass1_attempt(
                batch_dir=batch_dir,
                fingerprint=fingerprint,
            )
            clear_pass1_attempt_artifacts(batch_dir)
            attempt_index = archived_attempt_count + 1
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
                parsed = parse_batch_json_response(chat.content, candidates=batch_candidates)
                parsed_response_path = batch_dir / "parsed.json"
                write_json(parsed_response_path, parsed)
                batch_rows = batch_response_to_rows(
                    parsed,
                    candidates=batch_candidates,
                    config=batch_config,
                    prompt_path=prompt_path,
                    raw_response_path=raw_response_path,
                    parsed_response_path=parsed_response_path,
                    usage=chat.usage,
                )
                annotate_suite_rows(batch_rows, partition_index=partition_index, batch_index=batch_index)
                for row in batch_rows:
                    row["served_model"] = chat.served_model
                rows.extend(batch_rows)
                batch_results.append(
                    {
                        "status": "ok",
                        "partition_index": partition_index,
                        "batch_index": batch_index,
                        "candidate_count": len(batch_candidates),
                        "fingerprint": fingerprint,
                        "prompt_path": str(prompt_path),
                        "raw_response_path": str(raw_response_path),
                        "parsed_response_path": str(parsed_response_path),
                        "usage": chat.usage,
                        "served_model": chat.served_model,
                        "effective_reasoning": getattr(chat, "request", {}).get("reasoning"),
                        "provider_elapsed_seconds": round(chat.elapsed_seconds, 3),
                        "resumed": False,
                        "attempt_index": attempt_index,
                    }
                )
                write_json(batch_dir / "batch.json", batch_results[-1])
            except Exception as exc:  # noqa: BLE001 - keep running later batches.
                failures += 1
                response = getattr(exc, "response", None)
                raw_response_path = batch_dir / "response.json"
                if isinstance(response, dict) and not raw_response_path.exists():
                    write_json(raw_response_path, response)
                usage = chat.usage if chat is not None else getattr(exc, "usage", {})
                served_model = chat.served_model if chat is not None else getattr(exc, "served_model", None)
                error_path = batch_dir / "error.json"
                write_json(error_path, {"error": repr(exc)})
                batch_results.append(
                    {
                        "status": "failed",
                        "partition_index": partition_index,
                        "batch_index": batch_index,
                        "candidate_count": len(batch_candidates),
                        "fingerprint": fingerprint,
                        "prompt_path": str(prompt_path),
                        "error_path": str(error_path),
                        "error": repr(exc),
                        "attempt_index": attempt_index,
                    }
                )
                if raw_response_path.exists():
                    batch_results[-1]["raw_response_path"] = str(raw_response_path)
                if isinstance(usage, dict) and usage:
                    batch_results[-1]["usage"] = usage
                if served_model:
                    batch_results[-1]["served_model"] = served_model
                write_json(batch_dir / "batch.json", batch_results[-1])
                if is_batch_blocking_provider_error(exc):
                    blocked_reason = repr(exc)
            if config.request_delay_seconds:
                time.sleep(config.request_delay_seconds)
            if blocked_reason:
                break

    write_jsonl(out, rows)
    planned_batches = sum(len(partition) for partition in partitions)
    result = {
        **base_result,
        "status": "dry_run" if config.dry_run else ("ok" if failures == 0 else "partial_failed"),
        "rows": len(rows),
        "batch_count": planned_batches,
        "attempted_batch_count": len(batch_results),
        "skipped_batch_count": planned_batches - len(batch_results),
        "failure_count": failures,
        "resumed_batch_count": sum(bool(result.get("resumed")) for result in batch_results),
        "blocked_reason": blocked_reason,
        "usage": sum_usage(result.get("usage") for result in batch_results),
        "batch_results": batch_results,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(run_dir / "run.json", result)
    return result


def batch_prompt_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def archive_failed_pass1_attempt(*, batch_dir: Path, fingerprint: str) -> int:
    attempts_dir = batch_dir / "attempts"
    existing_indices = [
        int(path.name.removeprefix("attempt_"))
        for path in attempts_dir.glob("attempt_[0-9][0-9][0-9][0-9]")
        if path.is_dir()
    ]
    archived_count = max(existing_indices, default=0)
    state_path = batch_dir / "batch.json"
    if not state_path.exists():
        return archived_count
    try:
        state = read_json(state_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return archived_count
    if (
        not isinstance(state, dict)
        or state.get("status") != "failed"
        or state.get("fingerprint") != fingerprint
    ):
        return archived_count

    archived_count += 1
    archive_dir = attempts_dir / f"attempt_{archived_count:04d}"
    archive_dir.mkdir(parents=True, exist_ok=False)
    for name in ("prompt.json", "response.json", "parsed.json", "error.json", "batch.json"):
        source = batch_dir / name
        if source.exists():
            shutil.copy2(source, archive_dir / name)
            if name in {"response.json", "parsed.json", "error.json"}:
                source.unlink()
    write_json(
        archive_dir / "archive.json",
        {
            "attempt_index": state.get("attempt_index") or archived_count,
            "fingerprint": fingerprint,
            "status": "failed",
        },
    )
    write_json(
        state_path,
        {
            **state,
            "status": "retry_pending",
            "archived_attempt_index": archived_count,
        },
    )
    return archived_count


def clear_pass1_attempt_artifacts(batch_dir: Path) -> None:
    for name in ("response.json", "parsed.json", "error.json"):
        path = batch_dir / name
        if path.exists():
            path.unlink()


def load_cached_pass1_batch(
    *,
    batch_dir: Path,
    fingerprint: str,
    candidates: list[BatchCandidate],
    config: Pass1BatchConfig,
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
        rows = batch_response_to_rows(
            parsed,
            candidates=candidates,
            config=config,
            prompt_path=prompt_path,
            raw_response_path=raw_response_path,
            parsed_response_path=parsed_response_path,
            usage=state.get("usage") if isinstance(state.get("usage"), dict) else {},
        )
        annotate_suite_rows(rows, partition_index=partition_index, batch_index=batch_index)
        for row in rows:
            row["served_model"] = state.get("served_model")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return rows, {**state, "resumed": True}


def recover_saved_pass1_batch(
    *,
    batch_dir: Path,
    fingerprint: str,
    candidates: list[BatchCandidate],
    config: Pass1BatchConfig,
    prompt_path: Path,
    partition_index: int,
    batch_index: int,
    effective_reasoning_effort: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    state_path = batch_dir / "batch.json"
    raw_response_path = batch_dir / "response.json"
    parsed_response_path = batch_dir / "parsed.json"
    if not state_path.exists() or not raw_response_path.exists():
        return None
    try:
        state = read_json(state_path)
        if not isinstance(state, dict) or state.get("fingerprint") != fingerprint:
            return None
        response = read_json(raw_response_path)
        if not isinstance(response, dict):
            return None
        parsed = parse_batch_json_response(extract_content(response), candidates=candidates)
        write_json(parsed_response_path, parsed)
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        rows = batch_response_to_rows(
            parsed,
            candidates=candidates,
            config=config,
            prompt_path=prompt_path,
            raw_response_path=raw_response_path,
            parsed_response_path=parsed_response_path,
            usage=usage,
        )
        annotate_suite_rows(rows, partition_index=partition_index, batch_index=batch_index)
        served_model = response_model(response)
        for row in rows:
            row["served_model"] = served_model
        recovered_state = {
            "status": "ok",
            "partition_index": partition_index,
            "batch_index": batch_index,
            "candidate_count": len(candidates),
            "fingerprint": fingerprint,
            "prompt_path": str(prompt_path),
            "raw_response_path": str(raw_response_path),
            "parsed_response_path": str(parsed_response_path),
            "usage": usage,
            "served_model": served_model,
            "effective_reasoning": (
                {"effort": effective_reasoning_effort, "exclude": True}
                if effective_reasoning_effort
                else None
            ),
            "provider_elapsed_seconds": state.get("provider_elapsed_seconds"),
            "resumed": True,
            "recovered_from_raw_response": True,
        }
        write_json(state_path, recovered_state)
        return rows, recovered_state
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def select_batch_candidates(
    *,
    manifest: Path,
    text_source: str,
    paper_ids: set[str],
    limit: int | None,
    per_paper_char_budget: int,
) -> list[BatchCandidate]:
    candidates: list[BatchCandidate] = []
    for record in read_paper_records(manifest):
        if paper_ids and record.paper_id not in paper_ids:
            continue
        if record.parse_status != "ok":
            continue
        text_path, resolved_text_source = resolve_record_text_path(record, text_source)
        paper_text = truncate_candidate_text(read_text_path(text_path), per_paper_char_budget)
        candidates.append(BatchCandidate(record, text_path, resolved_text_source, paper_text))
        if limit is not None and len(candidates) >= limit:
            break
    return candidates


def parse_batch_json_response(text: str, *, candidates: list[BatchCandidate]) -> dict[str, Any]:
    try:
        return parse_json_response(text)
    except json.JSONDecodeError as original_error:
        repaired = repair_candidate_title_lines(text, candidates=candidates)
        if repaired == text:
            raise original_error
        return parse_json_response(repaired)


def repair_candidate_title_lines(text: str, *, candidates: list[BatchCandidate]) -> str:
    repaired = text
    for candidate in candidates:
        paper_id = re.escape(candidate.record.paper_id)
        title = candidate.record.title
        if not title:
            continue
        pattern = re.compile(
            rf'("paper_id"\s*:\s*"{paper_id}"\s*,[ \t]*\r?\n)'
            r'([ \t]*)"title"\s*:[^\r\n]*(\r?\n)'
        )
        repaired = pattern.sub(
            lambda match: (
                f'{match.group(1)}{match.group(2)}"title": '
                f"{json.dumps(title, ensure_ascii=False)},{match.group(3)}"
            ),
            repaired,
        )
    return repaired


def read_class_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        str(row.get("paper_id")): str(row.get("primary_contribution_class") or "other")
        for row in rows
        if row.get("paper_id")
    }


def make_batch_partitions(
    candidates: list[BatchCandidate],
    *,
    config: Pass1BatchSuiteConfig,
    class_by_id: dict[str, str],
) -> list[list[list[BatchCandidate]]]:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.partitions <= 0:
        raise ValueError("partitions must be positive")
    partitions: list[list[list[BatchCandidate]]] = []
    for partition_index in range(config.partitions):
        ordered = order_candidates_for_partition(
            candidates,
            strategy=config.strategy,
            class_by_id=class_by_id,
            seed=(config.seed or 0) + partition_index,
            force_shuffle=partition_index > 0,
        )
        partitions.append(chunk_candidates(ordered, config.batch_size))
    return partitions


def order_candidates_for_partition(
    candidates: list[BatchCandidate],
    *,
    strategy: str,
    class_by_id: dict[str, str],
    seed: int,
    force_shuffle: bool,
) -> list[BatchCandidate]:
    rng = random.Random(seed)
    ordered = list(candidates)
    if strategy == "sequential" and not force_shuffle:
        return ordered
    if strategy in {"sequential", "shuffled"}:
        rng.shuffle(ordered)
        return ordered
    if strategy == "class_round_robin":
        by_class: dict[str, list[BatchCandidate]] = {}
        for candidate in ordered:
            contribution_class = class_by_id.get(candidate.record.paper_id, "other")
            by_class.setdefault(contribution_class, []).append(candidate)
        for rows in by_class.values():
            rng.shuffle(rows)
        class_names = sorted(by_class)
        rng.shuffle(class_names)
        round_robin: list[BatchCandidate] = []
        while any(by_class.values()):
            for class_name in class_names:
                rows = by_class[class_name]
                if rows:
                    round_robin.append(rows.pop())
        return round_robin
    raise ValueError(f"Unsupported batch strategy: {strategy}")


def chunk_candidates(candidates: list[BatchCandidate], batch_size: int) -> list[list[BatchCandidate]]:
    return [candidates[idx : idx + batch_size] for idx in range(0, len(candidates), batch_size)]


def truncate_candidate_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n\n[TRUNCATED for cheap-model batch ranking at {limit} characters]"


def build_pass1_batch_messages(candidates: list[BatchCandidate], *, config: Pass1BatchConfig) -> list[dict[str, str]]:
    n = len(candidates)
    top_10 = max(1, math.ceil(n * 0.10))
    top_25 = max(top_10, math.ceil(n * 0.25))
    system = """You are a first-pass triage judge for a machine learning conference paper study.

Use only the provided paper text. Do not use web search, retrieval, citation memory, author identity, venue prestige, institution, or outside knowledge.
No reviews, reviewer scores, area-chair comments, decisions, presentation tiers, or awards are provided. Do not infer paper quality from possible venue or outcome cues in the document.

Your job is recall-oriented comparative triage: rank papers so the best broad-impact candidates move to a stronger OpenAI model. Your ranking is not final; missing a plausible high-impact candidate is worse than advancing a few extra papers. Do not give every paper the same score. Return only valid JSON."""
    user = f"""Rank these {n} papers by evidence-grounded priority for strong-model review.

Forced ranking rules:
- Assign a unique integer rank from 1 to {n}; no ties.
- Return exactly one `ranked_papers` entry for every candidate paper_id below.
- Do not invent, drop, rename, or duplicate paper_ids.
- Candidate paper_ids: {", ".join(candidate.record.paper_id for candidate in candidates)}
- Treat each `<PAPER paper_id="...">` block as sealed. Do not attribute methods, datasets, domains, claims, or risks from one paper to another.
- If you cannot verify a detail inside the current paper block, omit it or say it is uncertain. The `paper_isolation_check` must name one detail unique to that same paper.
- Exactly {top_10} paper(s) should be marked `forced_bucket: "top_10_percent"`.
- Exactly {top_25 - top_10} additional paper(s) should be marked `forced_bucket: "top_quartile"`.
- The remaining papers must be spread across `above_average`, `middle`, and `lower_priority`.
- `executive_ac_priority` must be derived from the rank: top_10_percent=9, top_quartile=8, above_average=6-7, middle=4-5, lower_priority=1-3.
- `should_advance_to_strong_model` should be true for the top quartile and may be true for lower-ranked papers only when `category_standout` is true or the paper has a clear route to field-level or scientific impact.

Contribution classes:
{", ".join(CONTRIBUTION_CLASSES)}

Ranking criteria:
- Prefer papers with a plausible path to broad scientific or ML-field impact, not merely accepted-paper polish.
- Separate strong accepted-paper polish from sweeping importance.
- Benchmarks, datasets, infrastructure, safety/eval work, and scientific modeling tools can rank highly if they are likely to become durable community resources, change practice, or unlock important downstream work.
- Do not penalize a paper merely for being applied or domain-targeted. Penalize it only when the impact route is narrow, local, or not transferable.
- Reward papers that create or clarify a new evaluation target, measurement capability, workflow, scientific modeling approach, safety protocol, or reusable method, even if they are not new core algorithms.
- Penalize benchmark-local wins when adoption path, generality, and field-building value are weak.
- Require a concrete two-year adoption path: who uses this, and what changes?
- Separate observed weaknesses from unverified risks.
- For each contribution class present in the batch, consider whether the strongest paper in that class deserves `category_standout: true`, even if it is not top-ranked overall.
- Before finalizing, check whether any applied/scientific/safety/resource paper with a strong adoption path is being suppressed only because it is not a core ML algorithm.

Required JSON:
{{
  "evaluation_mode": "pass1_cheap_forced_batch_ranking",
  "prompt_version": "{config.prompt_version}",
  "ranked_papers": [
    {{
      "rank": 1,
      "paper_id": "paper id",
      "title": "title",
      "forced_bucket": "top_10_percent",
      "estimated_percentile_among_batch": 95,
      "primary_contribution_class": "core_ml_algorithm",
      "secondary_contribution_classes": ["scientific_modeling_tool"],
      "impact_route_type": "core_ml_algorithm",
      "executive_ac_priority": 9,
      "broader_science_impact_forecast": 8,
      "ml_field_impact_forecast": 8,
      "technical_soundness": 7,
      "overall_significance": 8,
      "should_advance_to_strong_model": true,
      "category_standout": true,
      "category_rank_within_batch": 1,
      "two_year_adoption_path": "who uses it and what changes",
      "why_not_just_incremental_or_domain_specific": "why the contribution has reach beyond a local result",
      "why_ranked_here": "comparative, evidence-grounded reason",
      "main_risk": "main risk or limitation",
      "paper_isolation_check": "one paper-specific method/dataset/domain detail used for this judgment"
    }}
  ],
  "batch_assessment": "short assessment of score spread and shortlist quality"
}}

Papers:
{format_batch_candidates(candidates)}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def format_batch_candidates(candidates: list[BatchCandidate]) -> str:
    return "\n\n".join(format_batch_candidate(candidate) for candidate in candidates)


def format_batch_candidate(candidate: BatchCandidate) -> str:
    return f"""<PAPER paper_id="{candidate.record.paper_id}">
Title: {candidate.record.title or ""}
Text source: {candidate.resolved_text_source}
<<<PAPER_TEXT
{candidate.paper_text}
PAPER_TEXT
</PAPER>"""


def build_dry_run_rows(
    candidates: list[BatchCandidate],
    *,
    config: Pass1BatchConfig,
    prompt_path: Path,
) -> list[dict[str, Any]]:
    return [
        {
            "status": "dry_run",
            "paper_id": candidate.record.paper_id,
            "title": candidate.record.title,
            "provider": config.provider,
            "model": config.model,
            "prompt_version": config.prompt_version,
            "text_source": config.text_source,
            "resolved_text_source": candidate.resolved_text_source,
            "text_path": candidate.text_path,
            "prompt_path": str(prompt_path),
            "scores": None,
        }
        for candidate in candidates
    ]


def batch_response_to_rows(
    parsed: dict[str, Any],
    *,
    candidates: list[BatchCandidate],
    config: Pass1BatchConfig,
    prompt_path: Path,
    raw_response_path: Path,
    parsed_response_path: Path,
    usage: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates_by_id = {candidate.record.paper_id: candidate for candidate in candidates}
    ranked = parsed.get("ranked_papers")
    if not isinstance(ranked, list):
        raise ValueError("batch response missing ranked_papers array")
    ranked_objects = [item for item in ranked if isinstance(item, dict) and item.get("paper_id")]
    canonicalize_ranked_ids(ranked_objects, expected_ids=set(candidates_by_id))
    repair_duplicate_ids_from_titles(ranked_objects, candidates_by_id=candidates_by_id)
    validate_ranked_items(ranked_objects, expected_ids=set(candidates_by_id), expected_count=len(candidates))
    rows: list[dict[str, Any]] = []
    for item in ranked_objects:
        paper_id = str(item.get("paper_id") or "")
        candidate = candidates_by_id.get(paper_id)
        if candidate is None:
            continue
        rows.append(build_score_row(item, candidate=candidate, config=config, prompt_path=prompt_path,
                                    raw_response_path=raw_response_path, parsed_response_path=parsed_response_path,
                                    usage=usage))
    return rows


def repair_duplicate_ids_from_titles(
    ranked: list[dict[str, Any]],
    *,
    candidates_by_id: dict[str, BatchCandidate],
) -> None:
    output_ids = [str(item.get("paper_id") or "") for item in ranked]
    missing_ids = set(candidates_by_id) - set(output_ids)
    duplicate_ids = {paper_id for paper_id in output_ids if output_ids.count(paper_id) > 1}
    if not missing_ids or not duplicate_ids:
        return

    missing_by_title: dict[str, list[str]] = {}
    for paper_id in missing_ids:
        title = normalize_title(candidates_by_id[paper_id].record.title)
        if title:
            missing_by_title.setdefault(title, []).append(paper_id)

    for item in ranked:
        paper_id = str(item.get("paper_id") or "")
        if paper_id not in duplicate_ids:
            continue
        matching_missing = missing_by_title.get(normalize_title(item.get("title")), [])
        if len(matching_missing) != 1:
            continue
        repaired_id = matching_missing[0]
        item["paper_id_original_model_output"] = paper_id
        item["paper_id"] = repaired_id
        missing_ids.remove(repaired_id)
        missing_by_title.pop(normalize_title(item.get("title")), None)
        if output_ids.count(paper_id) - sum(
            str(row.get("paper_id") or "") == paper_id for row in ranked
        ) >= 1:
            duplicate_ids.discard(paper_id)


def normalize_title(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def canonicalize_ranked_ids(ranked: list[Any], *, expected_ids: set[str]) -> None:
    lower_to_id: dict[str, str] = {}
    for paper_id in expected_ids:
        lowered = paper_id.lower()
        if lowered in lower_to_id:
            return
        lower_to_id[lowered] = paper_id
    for item in ranked:
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("paper_id") or "")
        if paper_id in expected_ids:
            continue
        canonical = lower_to_id.get(paper_id.lower())
        if canonical:
            item["paper_id_original_model_output"] = paper_id
            item["paper_id"] = canonical
    repair_single_near_match(ranked, expected_ids=expected_ids)


def repair_single_near_match(ranked: list[Any], *, expected_ids: set[str]) -> None:
    output_ids = [str(item.get("paper_id") or "") for item in ranked if isinstance(item, dict)]
    seen_ids = set(output_ids)
    missing = sorted(expected_ids - seen_ids)
    extra = sorted(seen_ids - expected_ids)
    if len(missing) != 1 or len(extra) != 1:
        return
    if edit_distance(missing[0], extra[0]) > 2:
        return
    for item in ranked:
        if isinstance(item, dict) and item.get("paper_id") == extra[0]:
            item["paper_id_original_model_output"] = extra[0]
            item["paper_id"] = missing[0]
            return


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_ch in enumerate(left, start=1):
        current = [i]
        for j, right_ch in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_ch != right_ch),
                )
            )
        previous = current
    return previous[-1]


def validate_ranked_items(ranked: list[Any], *, expected_ids: set[str], expected_count: int) -> None:
    ids: list[str] = []
    ranks: list[int] = []
    for item in ranked:
        if not isinstance(item, dict):
            raise ValueError("ranked_papers contains a non-object item")
        paper_id = str(item.get("paper_id") or "")
        ids.append(paper_id)
        try:
            ranks.append(int(item.get("rank")))
        except (TypeError, ValueError):
            raise ValueError(f"invalid rank for paper_id={paper_id}") from None
    seen_ids = set(ids)
    duplicate_ids = sorted({paper_id for paper_id in ids if ids.count(paper_id) > 1})
    missing_ids = sorted(expected_ids - seen_ids)
    extra_ids = sorted(seen_ids - expected_ids)
    if len(ranked) != expected_count or duplicate_ids or missing_ids or extra_ids:
        raise ValueError(
            "batch response paper_id coverage error: "
            f"expected_count={expected_count}, actual_count={len(ranked)}, "
            f"duplicates={duplicate_ids}, missing={missing_ids}, extra={extra_ids}"
        )
    expected_ranks = set(range(1, expected_count + 1))
    rank_set = set(ranks)
    if rank_set != expected_ranks or len(ranks) != len(rank_set):
        raise ValueError(f"batch response rank coverage error: expected={sorted(expected_ranks)}, actual={sorted(ranks)}")


def build_score_row(
    item: dict[str, Any],
    *,
    candidate: BatchCandidate,
    config: Pass1BatchConfig,
    prompt_path: Path,
    raw_response_path: Path,
    parsed_response_path: Path,
    usage: dict[str, Any],
) -> dict[str, Any]:
    contribution_class = str(item.get("primary_contribution_class") or "other")
    if contribution_class not in CONTRIBUTION_CLASSES:
        contribution_class = "other"
    forced_bucket = str(item.get("forced_bucket") or "middle")
    category_standout = boolish(item.get("category_standout"))
    should_advance = boolish(item.get("should_advance_to_strong_model")) or (
        category_standout and numeric(item.get("executive_ac_priority")) >= 6
    )
    scores = {
        "summary": {
            "one_sentence_contribution": "",
            "main_claims": [],
            "primary_area": "",
        },
        "contribution_profile": {
            "contribution_types": [],
            "primary_contribution_class": contribution_class,
            "secondary_contribution_classes": item.get("secondary_contribution_classes") or [],
            "impact_route": item.get("two_year_adoption_path") or "",
            "impact_route_type": item.get("impact_route_type") or contribution_class,
            "category_rank_within_batch": numeric(item.get("category_rank_within_batch")),
            "main_audience": "",
            "artifact_types": [],
        },
        "scores": {
            "executive_ac_priority": numeric(item.get("executive_ac_priority")),
            "broader_science_impact_forecast": numeric(item.get("broader_science_impact_forecast")),
            "ml_field_impact_forecast": numeric(item.get("ml_field_impact_forecast")),
            "technical_soundness": numeric(item.get("technical_soundness")),
            "overall_significance": numeric(item.get("overall_significance")),
        },
        "calibration": {
            "estimated_percentile_among_accepted_papers": numeric(item.get("estimated_percentile_among_batch")),
            "triage_bucket": forced_bucket,
            "why_not_higher": item.get("main_risk") or "",
            "why_not_lower": item.get("why_ranked_here") or "",
            "reviewer_vs_executive_delta": "",
        },
        "ranking_signals": {
            "broad_scientific_impact_argument": item.get("why_ranked_here") or "",
            "ml_field_impact_argument": item.get("why_ranked_here") or "",
            "top_paper_case": item.get("two_year_adoption_path") or "",
            "dealbreaker_risks": item.get("main_risk") or "",
            "best_for_categories": [contribution_class],
            "should_advance_to_strong_model": should_advance,
            "category_standout": category_standout,
            "why_not_just_incremental_or_domain_specific": item.get("why_not_just_incremental_or_domain_specific") or "",
            "paper_isolation_check": item.get("paper_isolation_check") or "",
        },
        "evidence_audit": {
            "visible_sections_used": [],
            "observed_weaknesses": [],
            "not_visible_or_unverified_risks": [item.get("main_risk") or ""],
            "unsupported_inferences_to_avoid": [],
        },
    }
    if item.get("paper_id_original_model_output"):
        scores["ranking_signals"]["paper_id_original_model_output"] = item[
            "paper_id_original_model_output"
        ]
    return {
        "status": "ok",
        "paper_id": candidate.record.paper_id,
        "title": candidate.record.title,
        "source": candidate.record.source,
        "provider": config.provider,
        "model": config.model,
        "prompt_version": config.prompt_version,
        "text_source": config.text_source,
        "resolved_text_source": candidate.resolved_text_source,
        "text_path": candidate.text_path,
        "batch_rank": numeric(item.get("rank")),
        "forced_bucket": forced_bucket,
        "scores": scores,
        "validation_errors": [],
        "prompt_path": str(prompt_path),
        "raw_response_path": str(raw_response_path),
        "parsed_response_path": str(parsed_response_path),
        "usage": usage,
    }


def annotate_suite_rows(rows: list[dict[str, Any]], *, partition_index: int, batch_index: int) -> None:
    batch_size = len(rows)
    for row in rows:
        row["partition_index"] = partition_index
        row["batch_index"] = batch_index
        row["batch_size"] = batch_size
        rank = numeric(row.get("batch_rank"))
        row["batch_rank_percentile"] = batch_rank_percentile(rank=rank, batch_size=batch_size)
        row["batch_bucket_priority"] = bucket_priority(str(row.get("forced_bucket") or "middle"))
        row["batch_local_priority"] = round(row["batch_bucket_priority"] / 5 * 0.55 + row["batch_rank_percentile"] / 100 * 0.45, 4)


def batch_rank_percentile(*, rank: float, batch_size: int) -> float:
    if batch_size <= 1:
        return 100.0
    if rank <= 0:
        return 0.0
    return round(max(0.0, min(100.0, 100.0 * (batch_size - rank) / (batch_size - 1))), 4)


def bucket_priority(bucket: str) -> int:
    return {
        "top_10_percent": 5,
        "top_quartile": 4,
        "above_average": 3,
        "middle": 2,
        "lower_priority": 1,
    }.get(bucket, 2)


def numeric(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False
