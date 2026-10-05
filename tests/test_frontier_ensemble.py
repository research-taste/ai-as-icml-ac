import json
import tempfile
import unittest
from pathlib import Path

from icml_ai_ac.scoring.frontier_gold import (
    build_frontier_card_ensemble,
    rank_frontier_card_stream,
)
from icml_ai_ac.storage import write_jsonl


def card_row(
    paper_id: str,
    *,
    overall: float,
    broad: float | None = None,
    model: str = "model",
    fallback_from_model: str | None = None,
) -> dict:
    broad = overall if broad is None else broad
    card = {
        "evaluation_mode": "frontier_pdf_paper_card",
        "prompt_version": "frontier_pdf_paper_card_v2",
        "judge": f"openrouter:{model}",
        "paper_id": paper_id,
        "title": f"Paper {paper_id}",
        "primary_contribution_class": "core_ml_algorithm",
        "secondary_contribution_classes": [],
        "scores": {
            "overall_gold_priority_score": overall,
            "broad_scientific_impact_score": broad,
            "ml_field_impact_score": overall,
            "technical_soundness_score": overall,
            "novelty_score": overall,
            "evidence_confidence_score": overall,
            "visual_evidence_importance_score": 5,
        },
        "impact_assessment": {
            "two_year_adoption_path": "Adoption.",
            "five_year_field_effect": "Field effect.",
            "best_case_for_impact": "Best case.",
            "main_risk": "Risk.",
            "why_not_higher": "Ceiling.",
            "why_not_lower": "Floor.",
        },
        "evidence_from_pdf": {
            "figures_or_tables_that_matter": [],
            "visual_or_tabular_evidence_changes_judgment": "no",
            "missing_or_weak_visual_evidence": [],
        },
        "ranking_hooks": {
            "beats_papers_when": "Condition.",
            "loses_to_papers_when": "Condition.",
            "category_standout": False,
            "top_10_case": "Case.",
        },
        "one_sentence_summary": "Summary.",
    }
    row = {
        "status": "ok",
        "paper_id": paper_id,
        "title": card["title"],
        "model": model,
        "served_model": model,
        "card": card,
    }
    if fallback_from_model:
        row["fallback_from_model"] = fallback_from_model
    return row


class FrontierCardEnsembleTests(unittest.TestCase):
    def test_ensemble_is_invariant_to_monotonic_raw_score_scale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            transformed = root / "transformed.jsonl"
            write_jsonl(
                first,
                [
                    card_row("p1", overall=9, model="first"),
                    card_row("p2", overall=7, model="first"),
                    card_row("p3", overall=5, model="first"),
                ],
            )
            write_jsonl(
                second,
                [
                    card_row("p1", overall=8, model="second"),
                    card_row("p2", overall=6, model="second"),
                    card_row("p3", overall=4, model="second"),
                ],
            )
            write_jsonl(
                transformed,
                [
                    card_row("p1", overall=10, model="second"),
                    card_row("p2", overall=3, model="second"),
                    card_row("p3", overall=1, model="second"),
                ],
            )

            baseline = build_frontier_card_ensemble(
                card_paths=[first, second],
                labels=["first", "second"],
                out=root / "baseline.json",
                paper_set_name="test",
            )
            rescaled = build_frontier_card_ensemble(
                card_paths=[first, transformed],
                labels=["first", "second"],
                out=root / "rescaled.json",
                paper_set_name="test",
            )

            self.assertEqual(
                [row["paper_id"] for row in baseline["ranked_papers"]],
                ["p1", "p2", "p3"],
            )
            self.assertEqual(
                [row["paper_id"] for row in baseline["ranked_papers"]],
                [row["paper_id"] for row in rescaled["ranked_papers"]],
            )
            self.assertEqual(
                [row["ensemble_rank_score"] for row in baseline["ranked_papers"]],
                [1.0, 0.5, 0.0],
            )

    def test_score_axes_break_overall_score_ties(self) -> None:
        rows = {
            "p1": card_row("p1", overall=8, broad=9),
            "p2": card_row("p2", overall=8, broad=7),
            "p3": card_row("p3", overall=7, broad=10),
        }

        ranked = rank_frontier_card_stream(rows)

        self.assertEqual(ranked["p1"]["within_judge_rank"], 1.0)
        self.assertEqual(ranked["p2"]["within_judge_rank"], 2.0)
        self.assertEqual(ranked["p3"]["within_judge_rank"], 3.0)

    def test_fallback_remains_in_its_requested_panel_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sol = root / "sol.jsonl"
            fable = root / "fable.jsonl"
            write_jsonl(sol, [card_row("p1", overall=8, model="sol")])
            write_jsonl(
                fable,
                [
                    card_row(
                        "p1",
                        overall=7,
                        model="opus",
                        fallback_from_model="fable",
                    )
                ],
            )

            result = build_frontier_card_ensemble(
                card_paths=[sol, fable],
                labels=["sol", "fable_panel"],
                out=root / "ensemble.json",
                paper_set_name="test",
            )

            self.assertEqual(result["judge_count"], 2)
            self.assertEqual(result["card_streams"][1]["fallback_count"], 1)
            summaries = result["ranked_papers"][0]["judge_summaries"]
            self.assertEqual([row["stream_label"] for row in summaries], ["sol", "fable_panel"])
            self.assertEqual(summaries[1]["requested_model"], "fable")
            self.assertEqual(summaries[1]["served_model"], "opus")

    def test_mismatched_stream_coverage_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            write_jsonl(first, [card_row("p1", overall=8), card_row("p2", overall=7)])
            write_jsonl(second, [card_row("p1", overall=8)])

            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                build_frontier_card_ensemble(
                    card_paths=[first, second],
                    labels=["first", "second"],
                    out=root / "ensemble.json",
                    paper_set_name="test",
                )

    def test_labels_must_match_card_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cards.jsonl"
            write_jsonl(path, [card_row("p1", overall=8)])

            with self.assertRaisesRegex(ValueError, "exactly one --label"):
                build_frontier_card_ensemble(
                    card_paths=[path],
                    labels=["first", "extra"],
                    out=Path(tmp) / "ensemble.json",
                    paper_set_name="test",
                )


if __name__ == "__main__":
    unittest.main()
