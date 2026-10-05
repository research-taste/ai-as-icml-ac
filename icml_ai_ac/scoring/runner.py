from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from icml_ai_ac.models import PaperRecord
from icml_ai_ac.scoring.prompts import build_pass1_prompt
from icml_ai_ac.scoring.providers import ChatCompletionClient
from icml_ai_ac.scoring.schema import parse_json_response, validate_scoring_output
from icml_ai_ac.storage import ensure_parent, write_json


@dataclass(slots=True)
class ScoreRunConfig:
    provider: str
    model: str
    prompt_version: str
    temperature: float
    max_output_tokens: int
    seed: int | None
    text_source: str
    dry_run: bool
    input_cost_per_mtok: float | None
    output_cost_per_mtok: float | None


def score_record(
    record: PaperRecord,
    *,
    run_dir: Path,
    config: ScoreRunConfig,
    client: ChatCompletionClient | None = None,
) -> dict[str, Any]:
    text_path, resolved_text_source = resolve_record_text_path(record, config.text_source)
    paper_text = read_text_path(text_path)
    prompt = build_pass1_prompt(
        record,
        paper_text,
        prompt_version=config.prompt_version,
        text_source=resolved_text_source,
    )
    prompt_tokens_estimate = estimate_tokens(prompt.system) + estimate_tokens(prompt.user)
    prompt_payload = {
        "paper_id": record.paper_id,
        "title": record.title,
        "prompt_version": prompt.prompt_version,
        "text_source": config.text_source,
        "resolved_text_source": resolved_text_source,
        "text_path": str(text_path),
        "messages": prompt.messages(),
        "prompt_tokens_estimate": prompt_tokens_estimate,
    }
    write_json(run_dir / "prompts" / f"{record.paper_id}.json", prompt_payload)
    base_result: dict[str, Any] = {
        "paper_id": record.paper_id,
        "title": record.title,
        "source": record.source,
        "provider": config.provider,
        "model": config.model,
        "prompt_version": config.prompt_version,
        "text_source": config.text_source,
        "resolved_text_source": resolved_text_source,
        "text_path": str(text_path),
        "temperature": config.temperature,
        "seed": config.seed,
        "dry_run": config.dry_run,
        "input_token_estimate": prompt_tokens_estimate,
        "output_token_estimate": None,
        "cost_estimate": None,
        "status": "dry_run" if config.dry_run else "pending",
        "scores": None,
        "validation_errors": [],
        "raw_response_path": None,
        "parsed_response_path": None,
        "prompt_path": str(run_dir / "prompts" / f"{record.paper_id}.json"),
    }
    if config.dry_run:
        return base_result
    if client is None:
        raise ValueError("client is required unless dry_run=True")
    try:
        chat = client.complete(
            messages=prompt.messages(),
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            seed=config.seed,
            response_format={"type": "json_object"},
        )
        raw_response_path = run_dir / "responses" / f"{record.paper_id}.json"
        write_json(raw_response_path, chat.response)
        parsed = parse_json_response(chat.content)
        parsed.setdefault("paper_id", record.paper_id)
        parsed.setdefault("title", record.title or "")
        validation_errors = validate_scoring_output(parsed)
        parsed_response_path = run_dir / "parsed" / f"{record.paper_id}.json"
        write_json(parsed_response_path, parsed)
        output_tokens_estimate = estimate_tokens(chat.content)
        usage = chat.usage
        result = {
            **base_result,
            "status": "ok" if not validation_errors else "validation_error",
            "scores": parsed,
            "validation_errors": validation_errors,
            "raw_response_path": str(raw_response_path),
            "parsed_response_path": str(parsed_response_path),
            "served_model": chat.served_model,
            "output_token_estimate": output_tokens_estimate,
            "usage": usage,
            "provider_elapsed_seconds": round(chat.elapsed_seconds, 3),
        }
        result["cost_estimate"] = estimate_cost(
            input_tokens=usage.get("prompt_tokens") or prompt_tokens_estimate,
            output_tokens=usage.get("completion_tokens") or output_tokens_estimate,
            input_cost_per_mtok=config.input_cost_per_mtok,
            output_cost_per_mtok=config.output_cost_per_mtok,
        )
        return result
    except Exception as exc:  # noqa: BLE001 - row-local scoring failure is a valid artifact.
        error_path = run_dir / "errors" / f"{record.paper_id}.json"
        write_json(error_path, {"paper_id": record.paper_id, "error": repr(exc)})
        return {
            **base_result,
            "status": "failed",
            "validation_errors": [repr(exc)],
            "error_path": str(error_path),
        }


def prepare_run_dir(run_dir: Path) -> None:
    for child in ["prompts", "responses", "parsed", "errors"]:
        ensure_parent(run_dir / child / ".keep")
        (run_dir / child).mkdir(parents=True, exist_ok=True)


def read_text_path(path: str | None) -> str:
    if not path:
        raise ValueError("record has no text path")
    return Path(path).read_text(encoding="utf-8")


def resolve_record_text_path(record: PaperRecord, text_source: str) -> tuple[str, str]:
    artifacts = record.extra.get("text_artifacts") if isinstance(record.extra, dict) else None
    artifacts = artifacts if isinstance(artifacts, dict) else {}
    scoring_artifacts = record.extra.get("scoring_artifacts") if isinstance(record.extra, dict) else None
    scoring_artifacts = scoring_artifacts if isinstance(scoring_artifacts, dict) else {}
    anonymization = record.extra.get("anonymization") if isinstance(record.extra, dict) else None
    if isinstance(anonymization, dict) and anonymization.get("status") != "ok":
        raise ValueError(
            f"record {record.paper_id} has non-passing anonymization status: "
            f"{anonymization.get('status')}"
        )
    if text_source == "compact":
        path = scoring_artifacts.get("compact_repr") or record.text_compact or artifacts.get("compact_repr")
        resolved = "anonymized_compact" if scoring_artifacts.get("compact_repr") else "compact"
    elif text_source == "full":
        path = scoring_artifacts.get("full_repr") or record.text_full or artifacts.get("full_repr")
        resolved = "anonymized_full" if scoring_artifacts.get("full_repr") else "full"
    elif text_source == "scoring":
        path = scoring_artifacts.get("scoring_repr") or record.text_scoring or artifacts.get("scoring_repr")
        resolved = "anonymized_scoring" if scoring_artifacts.get("scoring_repr") else "scoring"
        if not path:
            path = scoring_artifacts.get("full_repr") or record.text_full or artifacts.get("full_repr")
            resolved = (
                "anonymized_full_fallback_from_scoring"
                if scoring_artifacts.get("full_repr")
                else "full_fallback_from_scoring"
            )
    else:
        raise ValueError(f"Unsupported text_source: {text_source}")
    if not path:
        raise ValueError(f"record has no {text_source} text path")
    return str(path), resolved


def resolve_record_pdf_path(record: PaperRecord) -> tuple[Path, str]:
    scoring_artifacts = record.extra.get("scoring_artifacts") if isinstance(record.extra, dict) else None
    scoring_artifacts = scoring_artifacts if isinstance(scoring_artifacts, dict) else {}
    anonymization = record.extra.get("anonymization") if isinstance(record.extra, dict) else None
    if isinstance(anonymization, dict):
        if anonymization.get("status") != "ok":
            raise ValueError(
                f"record {record.paper_id} has non-passing anonymization status: "
                f"{anonymization.get('status')}"
            )
        path = scoring_artifacts.get("pdf")
        if not path:
            raise ValueError(f"record {record.paper_id} is missing its anonymized scoring PDF")
        return Path(path), "anonymized_pdf"
    if not record.pdf_path:
        raise ValueError(f"record {record.paper_id} has no PDF path")
    return Path(record.pdf_path), "canonical_pdf"


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def estimate_cost(
    *,
    input_tokens: int | float,
    output_tokens: int | float,
    input_cost_per_mtok: float | None,
    output_cost_per_mtok: float | None,
) -> float | None:
    if input_cost_per_mtok is None or output_cost_per_mtok is None:
        return None
    return (float(input_tokens) / 1_000_000 * input_cost_per_mtok) + (
        float(output_tokens) / 1_000_000 * output_cost_per_mtok
    )


def write_run_metadata(path: Path, *, config: ScoreRunConfig, records: int, started_at: float, extra: dict[str, Any]) -> None:
    write_json(
        path,
        {
            "provider": config.provider,
            "model": config.model,
            "prompt_version": config.prompt_version,
            "text_source": config.text_source,
            "temperature": config.temperature,
            "seed": config.seed,
            "dry_run": config.dry_run,
            "records": records,
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
            **extra,
        },
    )
