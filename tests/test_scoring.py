import json
import tempfile
import unittest
from pathlib import Path

from icml_ai_ac.cheap_models import CHEAP_MODEL_PRESETS, DEFAULT_CHEAP_MODEL
from icml_ai_ac.models import PaperRecord
from icml_ai_ac.model_presets import PRODUCTION_FRONTIER_JUDGES
from icml_ai_ac.scoring.batch import Pass1BatchSuiteConfig, run_pass1_batch_suite
from icml_ai_ac.scoring.prompts import build_pass1_prompt
from icml_ai_ac.scoring.runner import ScoreRunConfig, score_record
from icml_ai_ac.scoring.schema import (
    CORE_SCORE_FIELDS,
    IMPACT_AXIS_FIELDS,
    parse_json_response,
    validate_scoring_output,
)


class ScoringTests(unittest.TestCase):
    def test_parse_json_repairs_unescaped_latex_commands(self) -> None:
        parsed = parse_json_response('{"title":"Fast Min-$\\epsilon$ Regression"}')

        self.assertEqual(parsed["title"], "Fast Min-$\\epsilon$ Regression")

    def test_parse_json_quotes_unquoted_object_keys(self) -> None:
        parsed = parse_json_response(
            """{
  "paper_id": "p1",
  main_risk": "limited validation",
  vulnerability or risk: "narrow adoption",
  $paper_isolation_check$: "specific method"
}"""
        )

        self.assertEqual(parsed["main_risk"], "limited validation")
        self.assertEqual(parsed["vulnerability or risk"], "narrow adoption")
        self.assertEqual(parsed["$paper_isolation_check$"], "specific method")

    def test_parse_json_accepts_unescaped_control_character_in_string(self) -> None:
        parsed = parse_json_response('{"rationale":"first line\nsecond line"}')

        self.assertEqual(parsed["rationale"], "first line\nsecond line")

    def test_parse_json_removes_trailing_comma_without_changing_string_content(self) -> None:
        parsed = parse_json_response(
            '{"rationale":"literal comma before brace,}", "scores":[1, 2,],}'
        )

        self.assertEqual(parsed["rationale"], "literal comma before brace,}")
        self.assertEqual(parsed["scores"], [1, 2])

    def test_production_cheap_preset_is_the_validated_four_model_panel(self) -> None:
        self.assertEqual(
            CHEAP_MODEL_PRESETS["production_2026_v2"].models,
            (
                "nvidia/nemotron-3-ultra-550b-a55b",
                "google/gemini-3.5-flash-lite",
                "openai/gpt-5.6-luna",
                "x-ai/grok-4.3",
            ),
        )

    def test_production_frontier_panel_spans_three_model_families(self) -> None:
        self.assertEqual(
            [
                (
                    judge.model,
                    judge.reasoning_effort,
                    judge.fallback_model,
                    judge.fallback_reasoning_effort,
                )
                for judge in PRODUCTION_FRONTIER_JUDGES
            ],
            [
                ("openai/gpt-5.6-sol", "xhigh", None, None),
                (
                    "anthropic/claude-fable-5",
                    "high",
                    "anthropic/claude-opus-4.8",
                    "high",
                ),
                ("google/gemini-3.1-pro-preview", "high", None, None),
            ],
        )

    def test_prompt_omits_author_and_human_outcome_fields(self) -> None:
        record = PaperRecord(
            paper_id="p1",
            source="SENTINEL_HUMAN_SOURCE",
            title="Test Paper",
            decision_label="SENTINEL_HUMAN_DECISION",
        )
        compact = "Title: Test Paper\nAuthors: A, B\n\nAbstract:\nA contribution."

        prompt = build_pass1_prompt(record, compact)

        self.assertIn("Required JSON schema", prompt.user)
        self.assertNotIn("Authors: A, B", prompt.user)
        self.assertNotIn("SENTINEL_HUMAN_SOURCE", prompt.user)
        self.assertNotIn("SENTINEL_HUMAN_DECISION", prompt.user)
        self.assertIn("Do not use web search", prompt.system)

    def test_validate_scoring_output_accepts_required_scores(self) -> None:
        payload = {
            "paper_id": "p1",
            "title": "Test",
            "evaluation_mode": "paper_only_executive_ac",
            "summary": {"one_sentence_contribution": "x", "main_claims": ["x"], "primary_area": "ml"},
            "contribution_profile": {
                "contribution_types": ["algorithm"],
                "primary_contribution_class": "core_ml_algorithm",
                "secondary_contribution_classes": [],
                "impact_route": "better model",
                "main_audience": "ml",
                "artifact_types": [],
            },
            "scores": {field: 5 for field in CORE_SCORE_FIELDS},
            "impact_axes": {field: 5 for field in IMPACT_AXIS_FIELDS},
            "reviewer_lens": {
                "likely_reviewer_strengths": ["x"],
                "likely_reviewer_concerns": ["y"],
                "human_review_alignment": "likely aligned",
            },
            "executive_lens": {
                "why_it_might_matter": "x",
                "sweeping_impact_scenario": "y",
                "barriers_to_impact": ["z"],
            },
            "calibration": {
                "estimated_percentile_among_accepted_papers": 75,
                "triage_bucket": "top_quartile",
                "why_not_higher": "x",
                "why_not_lower": "y",
                "reviewer_vs_executive_delta": "aligned",
            },
            "visibility_limits": {
                "what_was_visible": ["abstract", "intro"],
                "not_visible_in_provided_repr": ["full experiments"],
                "claims_requiring_full_paper_check": ["baselines"],
            },
            "ranking_signals": {
                "broad_scientific_impact_argument": "x",
                "ml_field_impact_argument": "y",
                "top_paper_case": "z",
                "dealbreaker_risks": ["none"],
                "best_for_categories": ["method"],
                "should_advance_to_strong_model": True,
            },
            "evidence_audit": {
                "visible_sections_used": ["abstract"],
                "observed_weaknesses": ["x"],
                "not_visible_or_unverified_risks": ["y"],
                "unsupported_inferences_to_avoid": ["z"],
            },
            "false_negative_likelihood_if_rejected": {"score": 5, "rationale": "x"},
            "evidence": {"key_positive_evidence": ["x"], "key_negative_evidence": ["y"], "missing_information": ["z"]},
            "uncertainty": {"confidence": 6, "confidence_rationale": "x", "contamination_sensitivity": "low"},
        }

        self.assertEqual(validate_scoring_output(payload), [])

    def test_score_record_dry_run_writes_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            compact = root / "compact.txt"
            compact.write_text("Title: Test\n\nAbstract:\nA test paper.", encoding="utf-8")
            record = PaperRecord(paper_id="p1", source="accepted", title="Test", text_compact=str(compact))
            config = ScoreRunConfig(
                provider="openrouter",
                model=DEFAULT_CHEAP_MODEL,
                prompt_version="pass1_executive_ac_v5",
                temperature=0.2,
                max_output_tokens=1000,
                seed=None,
                text_source="compact",
                dry_run=True,
                input_cost_per_mtok=None,
                output_cost_per_mtok=None,
            )

            result = score_record(record, run_dir=root / "run", config=config)

            self.assertEqual(result["status"], "dry_run")
            prompt_path = Path(result["prompt_path"])
            self.assertTrue(prompt_path.exists())
            prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
            self.assertEqual(prompt["paper_id"], "p1")

    def test_batch_suite_stops_after_request_wide_http_400(self) -> None:
        class RejectingClient:
            calls = 0

            def complete(self, **kwargs):
                self.calls += 1
                raise RuntimeError("HTTP 400 from provider: incompatible reasoning configuration")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text_path = root / "paper.txt"
            text_path.write_text("A paper body.", encoding="utf-8")
            records = [
                PaperRecord(
                    paper_id=f"p{index}",
                    source="accepted",
                    title=f"Paper {index}",
                    parse_status="ok",
                    text_scoring=str(text_path),
                )
                for index in range(4)
            ]
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "".join(json.dumps(record.to_dict()) + "\n" for record in records),
                encoding="utf-8",
            )
            client = RejectingClient()
            config = Pass1BatchSuiteConfig(
                provider="openrouter",
                model="model/requiring-reasoning",
                reasoning_effort=None,
                prompt_version="test",
                text_source="scoring",
                limit=None,
                paper_ids=set(),
                per_paper_char_budget=1000,
                batch_size=2,
                partitions=1,
                strategy="sequential",
                class_path=None,
                request_delay_seconds=0,
                temperature=0,
                max_output_tokens=100,
                seed=None,
                dry_run=False,
            )

            result = run_pass1_batch_suite(
                manifest=manifest,
                out=root / "scores.jsonl",
                run_dir=root / "run",
                config=config,
                client=client,  # type: ignore[arg-type]
            )

            self.assertEqual(client.calls, 1)
            self.assertEqual(result["attempted_batch_count"], 1)
            self.assertEqual(result["skipped_batch_count"], 1)
            self.assertIn("HTTP 400", result["blocked_reason"])
