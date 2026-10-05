import json
import tempfile
import unittest
from pathlib import Path

from icml_ai_ac.finalists import FinalistSelectionConfig, read_ranked_rows, select_finalists


def cheap_row(paper_id: str, rank: int, contribution_class: str = "core_ml_algorithm") -> dict:
    return {
        "status": "ok",
        "rank": rank,
        "paper_id": paper_id,
        "title": paper_id,
        "primary_contribution_class": contribution_class,
        "scores": {
            "contribution_profile": {"primary_contribution_class": contribution_class},
            "scores": {"executive_ac_priority": 10 - rank / 10},
        },
    }


def semifinal_row(paper_id: str, rank: int, contribution_class: str = "core_ml_algorithm") -> dict:
    return {
        "rank": rank,
        "paper_id": paper_id,
        "title": paper_id,
        "primary_contribution_class": contribution_class,
        "overall_priority_score": 10 - rank / 10,
        "why_ranked_here": "reason",
    }


class FinalistSelectionTests(unittest.TestCase):
    def test_select_finalists_adds_class_and_disagreement_saves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cheap = root / "cheap.jsonl"
            semifinal = root / "semifinal.json"
            cheap_rows = [
                cheap_row("p1", 1),
                cheap_row("p2", 2),
                cheap_row("p3", 3, "theory"),
                cheap_row("p4", 4, "theory"),
                cheap_row("p5", 5, "benchmark_dataset"),
            ]
            semifinal_rows = [
                semifinal_row("p1", 1),
                semifinal_row("p2", 2),
                semifinal_row("p3", 5, "theory"),
                semifinal_row("p4", 4, "theory"),
                semifinal_row("p5", 3, "benchmark_dataset"),
            ]
            cheap.write_text("\n".join(json.dumps(row) for row in cheap_rows) + "\n", encoding="utf-8")
            semifinal.write_text(json.dumps({"ranked_papers": semifinal_rows}), encoding="utf-8")
            out = root / "finalists.jsonl"
            report = root / "report.json"

            payload = select_finalists(
                cheap_path=cheap,
                semifinal_path=semifinal,
                out=out,
                report=report,
                config=FinalistSelectionConfig(
                    limit=5,
                    semifinal_top=2,
                    cheap_top=0,
                    min_per_class=1,
                    disagreement_saves=1,
                ),
            )

            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            selected_ids = {row["paper_id"] for row in rows}
            self.assertEqual(payload["written_rows"], 5)
            self.assertIn("p5", selected_ids)
            self.assertIn("p3", selected_ids)
            self.assertTrue(any("cheap_semifinal_disagreement_save" in row["selection_reasons"] for row in rows))
            self.assertNotIn("semifinal_fill", rows[0]["selection_reasons"])

    def test_read_ranked_rows_rejects_duplicate_paper_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ranking.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        cheap_row("p1", 1),
                        cheap_row("p1", 2),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate paper_id"):
                read_ranked_rows(path)

    def test_read_ranked_rows_uses_content_not_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "semifinal.jsonl"
            path.write_text(
                json.dumps({"ranked_papers": [semifinal_row("p1", 1)]}),
                encoding="utf-8",
            )

            rows = read_ranked_rows(path)

            self.assertEqual([row["paper_id"] for row in rows], ["p1"])

    def test_select_finalists_preserves_semifinal_judge_disagreements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cheap = root / "cheap.jsonl"
            semifinal = root / "semifinal.json"
            cheap_rows = [cheap_row(f"p{index}", index) for index in range(1, 5)]
            semifinal_rows = [semifinal_row(f"p{index}", index) for index in range(1, 5)]
            semifinal_rows[3].update(
                {
                    "semifinal_judge_disagreement": 1.0,
                    "best_source_rank_percentile": 0.0,
                    "source_ranks": {"terra": 4, "sonnet": 1},
                    "source_rank_percentiles": {"terra": 1.0, "sonnet": 0.0},
                    "source_judges": {
                        "terra": {"requested_model": "gpt-5.6-terra"},
                        "sonnet": {"requested_model": "anthropic/claude-sonnet-5"},
                    },
                }
            )
            cheap.write_text("\n".join(json.dumps(row) for row in cheap_rows) + "\n", encoding="utf-8")
            semifinal.write_text(json.dumps({"ranked_papers": semifinal_rows}), encoding="utf-8")

            out = root / "finalists.jsonl"
            select_finalists(
                cheap_path=cheap,
                semifinal_path=semifinal,
                out=out,
                report=None,
                config=FinalistSelectionConfig(
                    limit=3,
                    semifinal_top=1,
                    cheap_top=0,
                    min_per_class=0,
                    disagreement_saves=0,
                    judge_disagreement_saves=1,
                ),
            )

            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            saved = next(row for row in rows if row["paper_id"] == "p4")
            self.assertIn("semifinal_judge_disagreement_save", saved["selection_reasons"])
            self.assertEqual(saved["semifinal_source_ranks"], {"terra": 4, "sonnet": 1})
            self.assertEqual(
                saved["semifinal_source_judges"]["sonnet"]["requested_model"],
                "anthropic/claude-sonnet-5",
            )

    def test_select_finalists_rejects_negative_save_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cheap = root / "cheap.jsonl"
            semifinal = root / "semifinal.json"
            cheap.write_text(json.dumps(cheap_row("p1", 1)) + "\n", encoding="utf-8")
            semifinal.write_text(
                json.dumps({"ranked_papers": [semifinal_row("p1", 1)]}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "judge_disagreement_saves must be nonnegative"):
                select_finalists(
                    cheap_path=cheap,
                    semifinal_path=semifinal,
                    out=root / "out.jsonl",
                    report=None,
                    config=FinalistSelectionConfig(
                        limit=1,
                        semifinal_top=1,
                        cheap_top=0,
                        min_per_class=0,
                        disagreement_saves=0,
                        judge_disagreement_saves=-1,
                    ),
                )

    def test_select_finalists_rejects_preservation_rules_over_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cheap = root / "cheap.jsonl"
            semifinal = root / "semifinal.json"
            cheap_rows = [cheap_row(f"p{index}", index) for index in range(1, 5)]
            semifinal_rows = [semifinal_row(f"p{index}", index) for index in range(1, 5)]
            semifinal_rows[2]["semifinal_judge_disagreement"] = 1.0
            semifinal_rows[3]["semifinal_judge_disagreement"] = 0.9
            cheap.write_text("\n".join(json.dumps(row) for row in cheap_rows) + "\n", encoding="utf-8")
            semifinal.write_text(json.dumps({"ranked_papers": semifinal_rows}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "preservation rules selected 4 unique papers"):
                select_finalists(
                    cheap_path=cheap,
                    semifinal_path=semifinal,
                    out=root / "out.jsonl",
                    report=None,
                    config=FinalistSelectionConfig(
                        limit=3,
                        semifinal_top=2,
                        cheap_top=0,
                        min_per_class=0,
                        disagreement_saves=0,
                        judge_disagreement_saves=2,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
