import json
import tempfile
import unittest
from pathlib import Path

from icml_ai_ac.ranking_ensemble import ensemble_semifinal_rankings


def ranking_row(paper_id: str, rank: int, *, contribution_class: str = "core_ml_algorithm") -> dict:
    return {
        "paper_id": paper_id,
        "title": f"Paper {paper_id}",
        "rank": rank,
        "primary_contribution_class": contribution_class,
        "overall_priority_score": 10 - rank,
        "why_ranked_here": f"Reason from rank {rank}",
    }


def write_ranking(path: Path, *, provider: str, model: str, ordered_ids: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "provider": provider,
                "model": model,
                "served_model": model,
                "reasoning_effort": "high",
                "ranking": {
                    "ranked_papers": [
                        ranking_row(paper_id, rank)
                        for rank, paper_id in enumerate(ordered_ids, start=1)
                    ]
                },
            }
        ),
        encoding="utf-8",
    )


class SemifinalRankingEnsembleTests(unittest.TestCase):
    def test_equal_weight_consensus_preserves_source_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            terra = root / "terra.json"
            sonnet = root / "sonnet.json"
            write_ranking(
                terra,
                provider="openai",
                model="gpt-5.6-terra",
                ordered_ids=["p1", "p2", "p3", "p4"],
            )
            write_ranking(
                sonnet,
                provider="openrouter",
                model="anthropic/claude-sonnet-5",
                ordered_ids=["p2", "p3", "p1", "p4"],
            )
            out = root / "ensemble.json"
            report = root / "report.json"

            payload = ensemble_semifinal_rankings(
                ranking_paths=[terra, sonnet],
                labels=["terra", "sonnet"],
                out=out,
                report=report,
            )

            rows = payload["ranked_papers"]
            self.assertEqual([row["paper_id"] for row in rows], ["p2", "p1", "p3", "p4"])
            self.assertEqual(rows[0]["source_ranks"], {"terra": 2, "sonnet": 1})
            self.assertEqual(rows[1]["semifinal_judge_disagreement"], 0.666667)
            self.assertEqual(rows[1]["source_rows"]["terra"]["why_ranked_here"], "Reason from rank 1")
            self.assertEqual(rows[1]["source_judges"]["sonnet"]["served_model"], "anthropic/claude-sonnet-5")
            self.assertEqual(payload["sources"][1]["requested_model"], "anthropic/claude-sonnet-5")
            self.assertEqual(len(payload["sources"][1]["sha256"]), 64)
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["paper_count"], 4)

    def test_rejects_mismatched_paper_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.json"
            second = root / "second.json"
            write_ranking(first, provider="openai", model="terra", ordered_ids=["p1", "p2"])
            write_ranking(second, provider="openrouter", model="sonnet", ordered_ids=["p1", "p3"])

            with self.assertRaisesRegex(ValueError, "paper coverage differs"):
                ensemble_semifinal_rankings(
                    ranking_paths=[first, second],
                    labels=["terra", "sonnet"],
                    out=root / "out.json",
                    report=None,
                )

    def test_rejects_duplicate_judge_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.json"
            second = root / "second.json"
            write_ranking(first, provider="openai", model="terra", ordered_ids=["p1", "p2"])
            write_ranking(second, provider="openrouter", model="sonnet", ordered_ids=["p1", "p2"])

            with self.assertRaisesRegex(ValueError, "labels must be unique"):
                ensemble_semifinal_rankings(
                    ranking_paths=[first, second],
                    labels=["same", "same"],
                    out=root / "out.json",
                    report=None,
                )

    def test_rejects_noncontiguous_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.json"
            second = root / "second.json"
            write_ranking(first, provider="openai", model="terra", ordered_ids=["p1", "p2"])
            write_ranking(second, provider="openrouter", model="sonnet", ordered_ids=["p1", "p2"])
            payload = json.loads(second.read_text(encoding="utf-8"))
            payload["ranking"]["ranked_papers"][1]["rank"] = 1
            second.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "ranks must contain"):
                ensemble_semifinal_rankings(
                    ranking_paths=[first, second],
                    labels=["terra", "sonnet"],
                    out=root / "out.json",
                    report=None,
                )


if __name__ == "__main__":
    unittest.main()
