import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from icml_ai_ac.analysis.taxonomy import (
    TaxonomyConfig,
    aggregate_codes,
    balanced_sample,
    build_classify_messages,
    build_induce_messages,
    case_brief,
    cohens_kappa,
    cohens_kappa_per_code,
    collect_cases,
    run_taxonomy_classification,
    run_taxonomy_induction,
    validate_classification,
    validate_codebook,
)


def case(pid: str, direction_class: str = "theory") -> dict:
    return {
        "paper_id": pid,
        "title": pid.title(),
        "abstract": f"Abstract for {pid}.",
        "contribution_class": direction_class,
        "human": {
            "decision_status": "accepted",
            "tier": "poster",
            "is_award": False,
            "reviewer_overall": 3.0,
            "reviewer_soundness": 2.5,
        },
        "ai": {
            "ranking_signal": -1.0,
            "ranking_signal_kind": "ensemble_aggregate_rank",
            "executive_ac_priority": 9.0,
            "percentile": 92.0,
        },
        "divergence": {"residual": 60.0},
        "ai_axes": {"executive_ac_priority": 9.0, "conventional_acceptance_strength": 3.0, "technical_soundness": 4.0},
        "ai_rationale": {"sweeping_impact_scenario": f"scenario {pid}", "reviewer_vs_executive_delta": "AI higher"},
    }


def report_fixture() -> dict:
    return {
        "overlooked_gems": {"cases": [case("g1"), case("g2")]},
        "blind_spots": {"cases": [case("b1"), case("b2")]},
    }


CODEBOOK = {"codes": [{"code": "impact_over_rigor", "definition": "d1"}, {"code": "missed_rigor", "definition": "d2"}]}


def assignments_json(pairs: list[tuple[str, list[str]]]) -> str:
    return json.dumps(
        {"assignments": [{"paper_id": pid, "codes": [{"code": c, "evidence": "e"} for c in codes]} for pid, codes in pairs]}
    )


class FakeClient:
    def __init__(self, content) -> None:
        self.content = content
        self.calls = 0

    def complete(self, *, messages, temperature, max_output_tokens, seed=None, response_format=None):
        self.calls += 1
        text = self.content(messages) if callable(self.content) else self.content
        return SimpleNamespace(
            provider="p", model="m", served_model="m", request={}, response={}, content=text, usage={}, elapsed_seconds=0.0
        )


def config(dry_run: bool = False) -> TaxonomyConfig:
    return TaxonomyConfig(
        provider="openrouter", model="x/y", reasoning_effort="high",
        temperature=0.0, max_output_tokens=1000, seed=None, dry_run=dry_run,
    )


class TaxonomyHelperTests(unittest.TestCase):
    def test_collect_and_sample(self) -> None:
        cases = collect_cases(report_fixture())
        self.assertEqual([c["paper_id"] for c in cases], ["g1", "g2", "b1", "b2"])
        self.assertEqual([c["direction"] for c in cases], ["overlooked_gem", "overlooked_gem", "blind_spot", "blind_spot"])
        sampled = balanced_sample(cases, 2)
        self.assertEqual({c["direction"] for c in sampled}, {"overlooked_gem", "blind_spot"})

    def test_case_brief_includes_numeric_grounding(self) -> None:
        brief = case_brief(collect_cases(report_fixture())[0])
        self.assertIn("Abstract for g1", brief)
        self.assertIn("executive_priority=9.0", brief)
        self.assertIn("conventional_acceptance_strength=3.0", brief)
        self.assertIn("residual=60.0", brief)

    def test_prompts_build(self) -> None:
        cases = collect_cases(report_fixture())
        induce = build_induce_messages(cases, max_codes=5)
        self.assertIn("codebook", induce[0]["content"].lower())
        classify = build_classify_messages(cases, CODEBOOK["codes"], max_codes_per_case=2)
        self.assertIn("impact_over_rigor", classify[1]["content"])
        self.assertIn("g1, g2, b1, b2", classify[1]["content"])

    def test_validate_codebook(self) -> None:
        clean, errors = validate_codebook(CODEBOOK, max_codes=2)
        self.assertEqual([c["code"] for c in clean], ["impact_over_rigor", "missed_rigor"])
        self.assertEqual(errors, [])
        _, bad = validate_codebook({"codes": []}, max_codes=2)
        self.assertTrue(bad)

    def test_validate_classification_filters_unknown_and_flags_missing(self) -> None:
        payload = {"assignments": [
            {
                "paper_id": "g1",
                "codes": [
                    {"code": "impact_over_rigor", "evidence": "grounded evidence"},
                    {"code": "not_a_code", "evidence": "bad"},
                ],
            },
        ]}
        assignments, errors = validate_classification(
            payload,
            expected_ids=["g1", "g2"],
            code_names={"impact_over_rigor", "missed_rigor"},
            max_codes_per_case=2,
        )
        by_id = {a["paper_id"]: a["codes"] for a in assignments}
        self.assertEqual(
            by_id["g1"],
            [{"code": "impact_over_rigor", "evidence": "grounded evidence"}],
        )
        self.assertEqual(by_id["g2"], [])
        self.assertTrue(any("unknown code" in e for e in errors))
        self.assertTrue(any("missing assignments" in e for e in errors))

    def test_aggregate_codes(self) -> None:
        assignments = [
            {"paper_id": "g1", "direction": "overlooked_gem", "contribution_class": "theory", "codes": ["impact_over_rigor"]},
            {"paper_id": "b1", "direction": "blind_spot", "contribution_class": "benchmark_dataset", "codes": ["missed_rigor"]},
        ]
        agg = aggregate_codes(assignments, CODEBOOK["codes"])
        self.assertEqual(agg["code_totals"], {"impact_over_rigor": 1, "missed_rigor": 1})
        self.assertEqual(agg["by_direction"]["overlooked_gem"], {"impact_over_rigor": 1})

    def test_cohens_kappa(self) -> None:
        self.assertIsNone(cohens_kappa([True, True], [True, True]))
        self.assertIsNone(cohens_kappa([False, False], [False, False]))
        self.assertIsNone(cohens_kappa([], []))
        self.assertEqual(cohens_kappa([True, False, True, False], [True, False, True, True]), 0.5)


class TaxonomyRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.report = self.tmp / "report.json"
        self.report.write_text(json.dumps(report_fixture()), encoding="utf-8")
        self.codebook = self.tmp / "codebook.json"
        self.codebook.write_text(json.dumps(CODEBOOK), encoding="utf-8")

    def test_induction_dry_run(self) -> None:
        result = run_taxonomy_induction(
            report_path=self.report, out=self.tmp / "cb.json", run_dir=None,
            config=config(dry_run=True), client=None, sample_size=40, max_codes=8,
        )
        self.assertEqual(result["status"], "dry_run")
        self.assertIn("messages", result)

    def test_induction_with_client(self) -> None:
        client = FakeClient(json.dumps(CODEBOOK))
        out = self.tmp / "cb.json"
        result = run_taxonomy_induction(
            report_path=self.report, out=out, run_dir=self.tmp / "run",
            config=config(), client=client, sample_size=40, max_codes=8,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["codes"]), 2)
        self.assertTrue(out.exists())

    def test_classification_with_reliability(self) -> None:
        primary = FakeClient(assignments_json([("g1", ["impact_over_rigor"]), ("g2", ["impact_over_rigor"]), ("b1", ["missed_rigor"]), ("b2", ["missed_rigor"])]))
        # Second coder disagrees on b2.
        second = FakeClient(assignments_json([("g1", ["impact_over_rigor"]), ("g2", ["impact_over_rigor"]), ("b1", ["missed_rigor"]), ("b2", ["impact_over_rigor"])]))
        out = self.tmp / "classified.json"
        result = run_taxonomy_classification(
            report_path=self.report, codebook_path=self.codebook, out=out, run_dir=None,
            config=config(), client=primary, batch_size=10, max_codes_per_case=3,
            second_client=second, second_label="second/model",
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["aggregate"]["cases_coded"], 4)
        self.assertEqual(result["aggregate"]["code_totals"]["impact_over_rigor"], 2)
        kappa = result["reliability"]["per_code_cohens_kappa"]
        self.assertEqual(kappa["impact_over_rigor"]["kappa"], 0.5)
        self.assertEqual(kappa["missed_rigor"]["kappa"], 0.5)
        self.assertEqual(result["assignments"][0]["codes"][0]["evidence"], "e")
        self.assertEqual(result["reliability"]["exact_multilabel_agreement"]["rate"], 0.75)
        self.assertTrue(out.exists())

    def test_failed_batch_is_not_counted_as_coded(self) -> None:
        def fail(_messages):
            raise RuntimeError("provider failed")

        result = run_taxonomy_classification(
            report_path=self.report,
            codebook_path=self.codebook,
            out=self.tmp / "failed.json",
            run_dir=None,
            config=config(),
            client=FakeClient(fail),
            batch_size=10,
            max_codes_per_case=3,
        )
        self.assertEqual(result["status"], "validation_error")
        self.assertEqual(result["aggregate"]["cases_requested"], 4)
        self.assertEqual(result["aggregate"]["cases_coded"], 0)
        self.assertEqual(result["aggregate"]["cases_failed"], 4)

    def test_classification_dry_run(self) -> None:
        result = run_taxonomy_classification(
            report_path=self.report, codebook_path=self.codebook, out=self.tmp / "c.json", run_dir=None,
            config=config(dry_run=True), client=None, batch_size=10, max_codes_per_case=3,
        )
        self.assertEqual(result["status"], "dry_run")


if __name__ == "__main__":
    unittest.main()
