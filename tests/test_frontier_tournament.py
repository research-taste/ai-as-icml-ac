import unittest

from icml_ai_ac.scoring.frontier_gold import (
    aggregate_tournament_matches,
    batch_tournament_pairs,
    build_swiss_round_pairs,
    build_tournament_pairs,
    normalize_tournament_batch,
    pair_id_for,
    randomize_tournament_pairs,
    tournament_prompt_fingerprint,
    validate_tournament_batch,
)


class FrontierTournamentTests(unittest.TestCase):
    def test_pair_generation_and_batching(self) -> None:
        pairs = build_tournament_pairs(["p1", "p2", "p3", "p4"])

        self.assertEqual(
            pairs,
            [
                ("p1", "p2"),
                ("p1", "p3"),
                ("p1", "p4"),
                ("p2", "p3"),
                ("p2", "p4"),
                ("p3", "p4"),
            ],
        )
        self.assertEqual(batch_tournament_pairs(pairs, 4), [pairs[:4], pairs[4:]])

    def test_pair_batch_size_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "pairs_per_batch must be positive"):
            batch_tournament_pairs([("p1", "p2")], 0)

    def test_pair_schedule_and_orientation_are_deterministically_randomized(self) -> None:
        pairs = build_tournament_pairs(["p1", "p2", "p3", "p4"])
        randomized = randomize_tournament_pairs(pairs, seed=17)

        self.assertEqual(randomized, randomize_tournament_pairs(pairs, seed=17))
        self.assertEqual(
            {frozenset(pair) for pair in randomized},
            {frozenset(pair) for pair in pairs},
        )
        self.assertNotEqual(
            [frozenset(pair) for pair in randomized],
            [frozenset(pair) for pair in pairs],
        )
        original_orientation = {
            frozenset(pair): pair
            for pair in pairs
        }
        self.assertTrue(
            any(pair != original_orientation[frozenset(pair)] for pair in randomized)
        )
        self.assertTrue(
            any(pair == original_orientation[frozenset(pair)] for pair in randomized)
        )

    def test_swiss_scheduler_completes_ten_rounds_for_172_papers(self) -> None:
        paper_ids = [f"p{index:03d}" for index in range(172)]
        played: set[frozenset[str]] = set()

        for _ in range(10):
            pairs = build_swiss_round_pairs(paper_ids, played)
            self.assertEqual(len(pairs), 86)
            self.assertEqual(
                len({paper_id for pair in pairs for paper_id in pair}),
                172,
            )

        self.assertEqual(len(played), 860)

    def test_tournament_prompt_fingerprint_covers_decoding_configuration(self) -> None:
        payload = {
            "model": "judge",
            "pairs": [{"paper_a": "p1", "paper_b": "p2"}],
            "temperature": 0.0,
            "messages": [{"role": "user", "content": "compare"}],
        }

        self.assertEqual(
            tournament_prompt_fingerprint(payload),
            tournament_prompt_fingerprint({**payload, "fingerprint": "ignored"}),
        )
        self.assertNotEqual(
            tournament_prompt_fingerprint(payload),
            tournament_prompt_fingerprint({**payload, "temperature": 0.2}),
        )

    def test_normalize_and_validate_pairwise_batch(self) -> None:
        expected = {pair_id_for("p1", "p2"): ("p1", "p2")}
        parsed = {
            "matches": [
                {
                    "paper_a": "p2",
                    "paper_b": "p1",
                    "winner": "B",
                    "confidence": 1.2,
                    "impact_delta": "unclear",
                }
            ]
        }

        normalized = normalize_tournament_batch(parsed, expected_pairs=expected)

        self.assertEqual(normalized["matches"][0]["pair_id"], "p1__p2")
        self.assertEqual(normalized["matches"][0]["paper_a"], "p1")
        self.assertEqual(normalized["matches"][0]["paper_b"], "p2")
        self.assertEqual(normalized["matches"][0]["winner"], "paper_b")
        self.assertEqual(normalized["matches"][0]["winner_paper_id"], "p2")
        self.assertEqual(normalized["matches"][0]["confidence"], 1.0)
        self.assertEqual(validate_tournament_batch(normalized, expected_pairs=expected), [])

    def test_aggregate_tournament_matches_uses_confidence_margin_after_wins(self) -> None:
        matches = [
            {
                "pair_id": "p1__p2",
                "paper_a": "p1",
                "paper_b": "p2",
                "winner": "paper_a",
                "winner_paper_id": "p1",
                "confidence": 0.8,
                "impact_delta": "medium",
            },
            {
                "pair_id": "p1__p3",
                "paper_a": "p1",
                "paper_b": "p3",
                "winner": "paper_b",
                "winner_paper_id": "p3",
                "confidence": 0.7,
                "impact_delta": "small",
            },
            {
                "pair_id": "p2__p3",
                "paper_a": "p2",
                "paper_b": "p3",
                "winner": "paper_a",
                "winner_paper_id": "p2",
                "confidence": 0.9,
                "impact_delta": "small",
            },
        ]

        ordered = aggregate_tournament_matches(
            matches,
            tournament_ids=["p1", "p2", "p3"],
            seed_rank_by_id={"p1": 2, "p2": 1, "p3": 3},
        )

        self.assertEqual([row["paper_id"] for row in ordered], ["p1", "p3", "p2"])
        self.assertEqual(ordered[0]["pairwise_points"], 1.0)
        self.assertEqual(ordered[0]["wins"], 1)


if __name__ == "__main__":
    unittest.main()
