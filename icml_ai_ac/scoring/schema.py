from __future__ import annotations

import json
import re
from typing import Any


CORE_SCORE_FIELDS = [
    "technical_soundness",
    "methodological_rigor",
    "empirical_credibility",
    "theoretical_depth",
    "novelty",
    "clarity",
    "reproducibility",
    "ml_field_impact_forecast",
    "broader_science_impact_forecast",
    "conceptual_importance",
    "practical_usefulness",
    "overall_significance",
    "conventional_acceptance_strength",
    "executive_ac_priority",
]

IMPACT_AXIS_FIELDS = [
    "foundational_theory",
    "methodological_tooling",
    "benchmark_or_dataset_value",
    "systems_scalability",
    "scientific_discovery_enablement",
    "safety_alignment_or_governance",
    "interdisciplinary_reach",
    "field_building_potential",
]

CONTRIBUTION_CLASSES = [
    "core_ml_algorithm",
    "theory",
    "benchmark_dataset",
    "infrastructure_systems",
    "scientific_modeling_tool",
    "safety_governance_eval",
    "application_method",
    "analysis_position",
    "other",
]
INVALID_JSON_ESCAPE_PATTERN = re.compile(r'\\(?!["\\/bfnrtu])')
UNQUOTED_OBJECT_KEY_PATTERN = re.compile(
    r'(?m)^([ \t]*)([A-Za-z_$][A-Za-z0-9_$ -]*?)"?[ \t]*:'
)

TOP_LEVEL_REQUIRED = [
    "paper_id",
    "title",
    "evaluation_mode",
    "summary",
    "contribution_profile",
    "scores",
    "impact_axes",
    "reviewer_lens",
    "executive_lens",
    "calibration",
    "visibility_limits",
    "ranking_signals",
    "evidence_audit",
    "false_negative_likelihood_if_rejected",
    "evidence",
    "uncertainty",
]

CORE_VALIDATION_REQUIRED = [
    "paper_id",
    "title",
    "scores",
    "impact_axes",
]


SCORING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": TOP_LEVEL_REQUIRED,
    "properties": {
        "paper_id": {"type": "string"},
        "title": {"type": "string"},
        "evaluation_mode": {"type": "string"},
        "summary": {
            "type": "object",
            "required": ["one_sentence_contribution", "main_claims", "primary_area"],
        },
        "contribution_profile": {
            "type": "object",
            "required": [
                "contribution_types",
                "primary_contribution_class",
                "secondary_contribution_classes",
                "impact_route",
                "main_audience",
                "artifact_types",
            ],
        },
        "scores": {"type": "object", "required": CORE_SCORE_FIELDS},
        "impact_axes": {"type": "object", "required": IMPACT_AXIS_FIELDS},
        "reviewer_lens": {
            "type": "object",
            "required": ["likely_reviewer_strengths", "likely_reviewer_concerns", "human_review_alignment"],
        },
        "executive_lens": {
            "type": "object",
            "required": ["why_it_might_matter", "sweeping_impact_scenario", "barriers_to_impact"],
        },
        "calibration": {
            "type": "object",
            "required": [
                "estimated_percentile_among_accepted_papers",
                "triage_bucket",
                "why_not_higher",
                "why_not_lower",
                "reviewer_vs_executive_delta",
            ],
        },
        "visibility_limits": {
            "type": "object",
            "required": [
                "what_was_visible",
                "not_visible_in_provided_repr",
                "claims_requiring_full_paper_check",
            ],
        },
        "ranking_signals": {
            "type": "object",
            "required": [
                "broad_scientific_impact_argument",
                "ml_field_impact_argument",
                "top_paper_case",
                "dealbreaker_risks",
                "best_for_categories",
                "should_advance_to_strong_model",
            ],
        },
        "evidence_audit": {
            "type": "object",
            "required": [
                "visible_sections_used",
                "observed_weaknesses",
                "not_visible_or_unverified_risks",
                "unsupported_inferences_to_avoid",
            ],
        },
        "false_negative_likelihood_if_rejected": {
            "type": "object",
            "required": ["score", "rationale"],
        },
        "evidence": {
            "type": "object",
            "required": ["key_positive_evidence", "key_negative_evidence", "missing_information"],
        },
        "uncertainty": {
            "type": "object",
            "required": ["confidence", "confidence_rationale", "contamination_sensitivity"],
        },
    },
}


def schema_for_prompt() -> str:
    return json.dumps(SCORING_SCHEMA, indent=2, sort_keys=True)


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = load_json_with_invalid_escape_repair(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        parsed = load_json_with_invalid_escape_repair(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("model response JSON must be an object")
    return parsed


def load_json_with_invalid_escape_repair(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as original_error:
        candidates = [INVALID_JSON_ESCAPE_PATTERN.sub(r"\\\\", text)]
        candidates.append(quote_unquoted_object_keys(text))
        candidates.append(quote_unquoted_object_keys(candidates[0]))
        candidates.extend(remove_trailing_json_commas(candidate) for candidate in list(candidates))
        candidates.append(remove_trailing_json_commas(text))
        for candidate in [text, *candidates]:
            if candidate != text:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
            try:
                return json.loads(candidate, strict=False)
            except json.JSONDecodeError:
                pass
        raise original_error


def quote_unquoted_object_keys(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return f"{match.group(1)}{json.dumps(match.group(2).rstrip())}:"

    return UNQUOTED_OBJECT_KEY_PATTERN.sub(replace, text)


def remove_trailing_json_commas(text: str) -> str:
    output: list[str] = []
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if in_string:
            output.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            output.append(character)
            continue
        if character == ",":
            next_index = index + 1
            while next_index < len(text) and text[next_index].isspace():
                next_index += 1
            if next_index < len(text) and text[next_index] in "}]":
                continue
        output.append(character)
    return "".join(output)


def validate_scoring_output(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in CORE_VALIDATION_REQUIRED:
        if field not in payload:
            errors.append(f"missing top-level field: {field}")
    scores = payload.get("scores")
    if isinstance(scores, dict):
        for field in CORE_SCORE_FIELDS:
            validate_score(scores, field, errors, path=f"scores.{field}")
    else:
        errors.append("scores must be an object")
    impact_axes = payload.get("impact_axes")
    if isinstance(impact_axes, dict):
        for field in IMPACT_AXIS_FIELDS:
            validate_score(impact_axes, field, errors, path=f"impact_axes.{field}")
    else:
        errors.append("impact_axes must be an object")
    contribution_profile = payload.get("contribution_profile")
    if isinstance(contribution_profile, dict):
        contribution_class = contribution_profile.get("primary_contribution_class")
        if contribution_class is not None and contribution_class not in CONTRIBUTION_CLASSES:
            errors.append(
                "contribution_profile.primary_contribution_class must be one of: "
                + ", ".join(CONTRIBUTION_CLASSES)
            )
    false_negative = payload.get("false_negative_likelihood_if_rejected")
    if isinstance(false_negative, dict):
        validate_score(false_negative, "score", errors, path="false_negative_likelihood_if_rejected.score")
    uncertainty = payload.get("uncertainty")
    if isinstance(uncertainty, dict):
        validate_score(uncertainty, "confidence", errors, path="uncertainty.confidence")
    calibration = payload.get("calibration")
    if isinstance(calibration, dict):
        value = calibration.get("estimated_percentile_among_accepted_papers")
        if not isinstance(value, (int, float)):
            errors.append("calibration.estimated_percentile_among_accepted_papers must be numeric")
        elif not 0 <= value <= 100:
            errors.append("calibration.estimated_percentile_among_accepted_papers must be between 0 and 100")
    return errors


def validate_score(container: dict[str, Any], field: str, errors: list[str], *, path: str) -> None:
    value = container.get(field)
    if not isinstance(value, (int, float)):
        errors.append(f"{path} must be numeric")
        return
    if not 1 <= value <= 10:
        errors.append(f"{path} must be between 1 and 10")
