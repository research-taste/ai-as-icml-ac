import unittest

from icml_ai_ac.analysis.tournament_simulation import simulate_hybrid_tournament


class TournamentSimulationTests(unittest.TestCase):
    def test_simulates_hybrid_schedule_from_complete_all_pairs(self) -> None:
        paper_ids = [f"p{index}" for index in range(6)]
        matches = []
        for left_index, paper_a in enumerate(paper_ids):
            for paper_b in paper_ids[left_index + 1 :]:
                matches.append(
                    {
                        "pair_id": f"{paper_a}__{paper_b}",
                        "paper_a": paper_a,
                        "paper_b": paper_b,
                        "winner": "paper_a",
                        "winner_paper_id": paper_a,
                        "confidence": 0.8,
                        "impact_delta": "medium",
                    }
                )
        tournament = {
            "prompt_version": "test",
            "matches": matches,
            "ranked_papers": [
                {
                    "rank": rank,
                    "paper_id": paper_id,
                    "seed_synthesis_rank": rank,
                    "ranking_source": "pairwise_tournament",
                }
                for rank, paper_id in enumerate(paper_ids, start=1)
            ],
        }

        result = simulate_hybrid_tournament(
            tournament,
            swiss_rounds=2,
            playoff_top_n=4,
        )

        self.assertEqual(result["pool_size"], 6)
        self.assertEqual(result["all_pairs_count"], 15)
        self.assertLess(result["simulated_pair_count"], 15)
        self.assertEqual(result["metrics_vs_all_pairs"]["spearman"], 1.0)
        self.assertEqual(
            [row["paper_id"] for row in result["ranked_papers"]],
            paper_ids,
        )

    def test_rejects_incomplete_all_pairs_artifact(self) -> None:
        tournament = {
            "matches": [],
            "ranked_papers": [
                {
                    "rank": rank,
                    "paper_id": paper_id,
                    "seed_synthesis_rank": rank,
                    "ranking_source": "pairwise_tournament",
                }
                for rank, paper_id in enumerate(["p1", "p2"], start=1)
            ],
        }

        with self.assertRaisesRegex(ValueError, "incomplete"):
            simulate_hybrid_tournament(
                tournament,
                swiss_rounds=1,
                playoff_top_n=2,
            )


if __name__ == "__main__":
    unittest.main()
