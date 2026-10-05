from __future__ import annotations

from dataclasses import dataclass


DEFAULT_CHEAP_MODEL = "qwen/qwen3.7-plus"


@dataclass(frozen=True, slots=True)
class CheapModelPreset:
    name: str
    models: tuple[str, ...]
    notes: str


CHEAP_MODEL_PRESETS: dict[str, CheapModelPreset] = {
    "production_2026_v2": CheapModelPreset(
        name="production_2026_v2",
        models=(
            "nvidia/nemotron-3-ultra-550b-a55b",
            "google/gemini-3.5-flash-lite",
            "openai/gpt-5.6-luna",
            "x-ai/grok-4.3",
        ),
        notes=(
            "Gold-50 validated high-recall ensemble. The four families were selected "
            "after a fourteen-model screen and all 330 subsets of the eleven usable "
            "rankings; a later matched bakeoff upgraded its Gemini stream from 3.1 "
            "to 3.5 Flash Lite for stronger top-tail recall."
        ),
    ),
    "production_2026_v1": CheapModelPreset(
        name="production_2026_v1",
        models=(
            DEFAULT_CHEAP_MODEL,
            "nvidia/nemotron-3-ultra-550b-a55b",
            "google/gemini-3.1-flash-lite",
        ),
        notes=(
            "Historical June 2026 production ensemble retained for reproducibility: "
            "Qwen, NVIDIA Nemotron, and Gemini Flash Lite families."
        ),
    ),
    "broad_2026_probe": CheapModelPreset(
        name="broad_2026_probe",
        models=(
            DEFAULT_CHEAP_MODEL,
            "nvidia/nemotron-3-ultra-550b-a55b",
            "google/gemini-3.1-flash-lite",
            "google/gemini-3.5-flash-lite",
            "stepfun/step-3.7-flash",
            "inclusionai/ring-2.6-1t",
        ),
        notes=(
            "Broader cheap-model probe for accepted-50 evaluation before a full run. "
            "Includes extra low-cost candidates that may be less proven for strict JSON."
        ),
    ),
    "prelaunch_2026_refresh_probe": CheapModelPreset(
        name="prelaunch_2026_refresh_probe",
        models=(
            DEFAULT_CHEAP_MODEL,
            "nvidia/nemotron-3-ultra-550b-a55b",
            "google/gemini-3.1-flash-lite",
            "google/gemini-3.5-flash-lite",
            "openai/gpt-5.6-luna",
            "deepseek/deepseek-v4-pro",
            "minimax/minimax-m3",
            "stepfun/step-3.7-flash",
            "z-ai/glm-5.2",
            "x-ai/grok-4.3",
            "google/gemini-3.5-flash",
        ),
        notes=(
            "One-time accepted-50 bakeoff for the July 2026 catalog refresh. "
            "Select a smaller production ensemble by marginal recall, reliability, "
            "cost, and model-family diversity."
        ),
    ),
    "legacy_2025_gold": CheapModelPreset(
        name="legacy_2025_gold",
        models=(
            "qwen/qwen3-32b",
            "qwen/qwen3.6-35b-a3b",
            "qwen/qwen3-235b-a22b-2507",
        ),
        notes="Historical accepted-50 probe set retained for reproducibility.",
    ),
}


def resolve_cheap_models(*, preset: str | None, explicit_models: list[str] | None) -> list[str]:
    models: list[str] = []
    if preset:
        if preset not in CHEAP_MODEL_PRESETS:
            valid = ", ".join(sorted(CHEAP_MODEL_PRESETS))
            raise ValueError(f"Unknown cheap-model preset: {preset}. Valid presets: {valid}")
        models.extend(CHEAP_MODEL_PRESETS[preset].models)
    models.extend(explicit_models or [])
    deduped: list[str] = []
    seen: set[str] = set()
    for model in models:
        if model and model not in seen:
            deduped.append(model)
            seen.add(model)
    return deduped
