import math
import unittest

from icml_ai_ac.scoring.bradley_terry import (
    is_connected,
    rank_bradley_terry,
    ranked_meta_from_payload,
    select_tournament_stage,
    tally_matches,
    tournament_pool_from_payload,
)


def decisive(paper_a: str, paper_b: str, winner_id: str, *, confidence: float = 0.8) -> dict:
    return {
        "pair_id": f"{paper_a}__{paper_b}",
        "paper_a": paper_a,
        "paper_b": paper_b,
        "winner": "paper_a" if winner_id == paper_a else "paper_b",
        "winner_paper_id": winner_id,
        "confidence": confidence,
        "impact_delta": "small",
    }


def tie(paper_a: str, paper_b: str, *, confidence: float = 0.5) -> dict:
    return {
        "pair_id": f"{paper_a}__{paper_b}",
        "paper_a": paper_a,
        "paper_b": paper_b,
        "winner": "tie",
        "winner_paper_id": "",
        "confidence": confidence,
        "impact_delta": "tie",
    }


def strengths_by_id(payload: dict) -> dict[str, float]:
    return {row["paper_id"]: row["bt_log_strength"] for row in payload["ranked_papers"]}


class BradleyTerryTests(unittest.TestCase):
    def test_transitive_order_recovered(self) -> None:
        matches = [
            decisive("p1", "p2", "p1"),
            decisive("p1", "p3", "p1"),
            decisive("p2", "p3", "p2"),
        ]
        payload = rank_bradley_terry(matches)
        order = [row["paper_id"] for row in payload["ranked_papers"]]
        self.assertEqual(order, ["p1", "p2", "p3"])
        self.assertEqual([row["rank"] for row in payload["ranked_papers"]], [1, 2, 3])
        theta = strengths_by_id(payload)
        self.assertGreater(theta["p1"], theta["p2"])
        self.assertGreater(theta["p2"], theta["p3"])

    def test_win_prob_in_unit_interval_and_ordered(self) -> None:
        matches = [decisive("p1", "p2", "p1"), decisive("p1", "p3", "p1"), decisive("p2", "p3", "p2")]
        payload = rank_bradley_terry(matches)
        probs = [row["win_prob_vs_reference"] for row in payload["ranked_papers"]]
        for prob in probs:
            self.assertGreater(prob, 0.0)
            self.assertLess(prob, 1.0)
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_balanced_pair_is_symmetric(self) -> None:
        matches = [decisive("p1", "p2", "p1"), decisive("p1", "p2", "p2")]
        payload = rank_bradley_terry(matches)
        theta = strengths_by_id(payload)
        self.assertAlmostEqual(theta["p1"], theta["p2"], places=5)
        self.assertAlmostEqual(theta["p1"], 0.0, places=5)

    def test_single_tie_is_symmetric(self) -> None:
        payload = rank_bradley_terry([tie("p1", "p2")])
        theta = strengths_by_id(payload)
        self.assertAlmostEqual(theta["p1"], theta["p2"], places=6)
        row = payload["ranked_papers"][0]
        self.assertAlmostEqual(row["win_prob_vs_reference"], 0.5, places=3)
        self.assertEqual(payload["diagnostics"]["tie_comparison_weight"], 1.0)

    def test_confidence_weighting_widens_gap_with_higher_confidence(self) -> None:
        high = rank_bradley_terry([decisive("p1", "p2", "p1", confidence=0.9)], weight_mode="confidence")
        low = rank_bradley_terry([decisive("p1", "p2", "p1", confidence=0.2)], weight_mode="confidence")
        high_gap = strengths_by_id(high)["p1"] - strengths_by_id(high)["p2"]
        low_gap = strengths_by_id(low)["p1"] - strengths_by_id(low)["p2"]
        self.assertGreater(high_gap, low_gap)

    def test_regularization_keeps_undefeated_finite(self) -> None:
        matches = [decisive("p1", "p2", "p1"), decisive("p1", "p3", "p1")]
        payload = rank_bradley_terry(matches, prior_strength=1.0)
        theta = strengths_by_id(payload)
        self.assertTrue(math.isfinite(theta["p1"]))
        self.assertEqual(payload["ranked_papers"][0]["paper_id"], "p1")
        top = payload["ranked_papers"][0]
        self.assertIsNotNone(top["approx_standard_error"])
        self.assertTrue(math.isfinite(top["approx_standard_error"]))

    def test_standard_error_decreases_with_more_data(self) -> None:
        one_each = rank_bradley_terry([decisive("p1", "p2", "p1"), decisive("p1", "p2", "p2")])
        five_each = rank_bradley_terry(
            [decisive("p1", "p2", "p1")] * 5 + [decisive("p1", "p2", "p2")] * 5
        )
        se_one = one_each["ranked_papers"][0]["approx_standard_error"]
        se_five = five_each["ranked_papers"][0]["approx_standard_error"]
        self.assertLess(se_five, se_one)

    def test_disconnected_graph_is_flagged_but_still_fits(self) -> None:
        matches = [decisive("p1", "p2", "p1"), decisive("p3", "p4", "p3")]
        payload = rank_bradley_terry(matches)
        self.assertFalse(payload["diagnostics"]["comparison_graph_connected"])
        for row in payload["ranked_papers"]:
            self.assertTrue(math.isfinite(row["bt_log_strength"]))
        # Winners of each isolated component outrank the losers.
        theta = strengths_by_id(payload)
        self.assertGreater(theta["p1"], theta["p2"])
        self.assertGreater(theta["p3"], theta["p4"])

    def test_restrict_to_includes_unplayed_paper_as_neutral(self) -> None:
        payload = rank_bradley_terry([decisive("p1", "p2", "p1")], restrict_to=["p1", "p2", "p3"])
        theta = strengths_by_id(payload)
        self.assertIn("p3", theta)
        self.assertAlmostEqual(theta["p3"], 0.0, places=6)
        self.assertEqual(payload["diagnostics"]["player_count"], 3)

    def test_restrict_to_drops_out_of_pool_matches(self) -> None:
        matches = [decisive("p1", "p2", "p1"), decisive("p1", "pX", "pX")]
        payload = rank_bradley_terry(matches, restrict_to=["p1", "p2"])
        ids = {row["paper_id"] for row in payload["ranked_papers"]}
        self.assertEqual(ids, {"p1", "p2"})
        self.assertEqual(payload["diagnostics"]["skipped_match_count"], 1)

    def test_deterministic(self) -> None:
        matches = [decisive("p1", "p2", "p1"), decisive("p2", "p3", "p2"), tie("p1", "p3")]
        first = rank_bradley_terry(matches)
        second = rank_bradley_terry(matches)
        self.assertEqual(first, second)

    def test_ranked_meta_is_carried(self) -> None:
        meta = {"p1": {"title": "Paper One", "primary_contribution_class": "theory"}}
        payload = rank_bradley_terry([decisive("p1", "p2", "p1")], ranked_meta=meta)
        row = next(row for row in payload["ranked_papers"] if row["paper_id"] == "p1")
        self.assertEqual(row["title"], "Paper One")
        self.assertEqual(row["primary_contribution_class"], "theory")

    def test_unknown_winner_label_is_skipped(self) -> None:
        matches = [
            decisive("p1", "p2", "p1"),
            {"paper_a": "p1", "paper_b": "p2", "winner": "banana", "confidence": 0.5},
        ]
        tallies = tally_matches(matches)
        self.assertEqual(tallies.skipped_count, 1)
        self.assertEqual(tallies.wins["p1"], 1.0)
        # The skipped match must not inflate the comparison count.
        self.assertEqual(tallies.opponents["p1"]["p2"], 1.0)

    def test_only_invalid_match_does_not_connect_players(self) -> None:
        tallies = tally_matches(
            [{"paper_a": "p1", "paper_b": "p2", "winner": "invalid", "confidence": 0.5}]
        )
        self.assertEqual(tallies.players, [])
        self.assertTrue(is_connected(tallies))

    def test_invalid_weight_mode_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "weight_mode"):
            rank_bradley_terry([decisive("p1", "p2", "p1")], weight_mode="bogus")

    def test_fit_configuration_must_be_positive(self) -> None:
        matches = [decisive("p1", "p2", "p1")]
        for kwargs in (
            {"prior_strength": 0},
            {"prior_strength": float("nan")},
            {"max_iter": 0},
            {"tol": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                rank_bradley_terry(matches, **kwargs)

    def test_payload_helpers_from_tournament_shape(self) -> None:
        payload = {
            "ranked_papers": [
                {"paper_id": "p1", "title": "One", "primary_contribution_class": "theory", "ranking_source": "pairwise_tournament"},
                {"paper_id": "p2", "title": "Two", "primary_contribution_class": "theory", "ranking_source": "pairwise_tournament"},
                {"paper_id": "p9", "title": "Nine", "ranking_source": "seed_synthesis_outside_tournament_pool"},
            ],
            "tournament_summary": {"top_n": 2},
        }
        meta = ranked_meta_from_payload(payload)
        self.assertEqual(meta["p1"]["title"], "One")
        self.assertEqual(tournament_pool_from_payload(payload), ["p1", "p2"])

    def test_stage_selection_keeps_swiss_and_playoff_evidence_separate(self) -> None:
        swiss_matches = [
            decisive("p1", "p3", "p1"),
            decisive("p2", "p4", "p2"),
        ]
        playoff_new = [decisive("p1", "p2", "p1")]
        payload = {
            "tournament_strategy": "swiss_playoff",
            "matches": swiss_matches + playoff_new,
            "ranked_papers": [
                {"paper_id": paper_id, "ranking_source": "pairwise_tournament"}
                for paper_id in ("p1", "p2", "p3", "p4")
            ],
            "tournament_summary": {
                "strategy": "swiss_playoff",
                "playoff_ids": ["p1", "p2"],
                "schedule": [
                    {"stage": "swiss", "pair_count": 2},
                    {"stage": "playoff_all_pairs", "pair_count": 1},
                ],
            },
        }

        swiss, swiss_ids, swiss_meta = select_tournament_stage(payload, stage="swiss")
        self.assertEqual(swiss, swiss_matches)
        self.assertEqual(set(swiss_ids or []), {"p1", "p2", "p3", "p4"})
        self.assertEqual(swiss_meta["selected_stage"], "swiss")

        playoff, playoff_ids, playoff_meta = select_tournament_stage(payload, stage="auto")
        self.assertEqual(playoff, [playoff_new[0]])
        self.assertEqual(playoff_ids, ["p1", "p2"])
        self.assertEqual(playoff_meta["selected_stage"], "playoff")

    def test_playoff_selection_includes_prior_swiss_pair_between_finalists(self) -> None:
        payload = {
            "tournament_strategy": "swiss_playoff",
            "matches": [
                decisive("p1", "p2", "p1"),
                decisive("p1", "p3", "p1"),
                decisive("p2", "p3", "p2"),
            ],
            "tournament_summary": {
                "strategy": "swiss_playoff",
                "playoff_ids": ["p1", "p2"],
                "schedule": [
                    {"stage": "swiss", "pair_count": 2},
                    {"stage": "playoff_all_pairs", "pair_count": 1},
                ],
            },
        }
        matches, ids, _ = select_tournament_stage(payload, stage="playoff")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["pair_id"], "p1__p2")
        self.assertEqual(ids, ["p1", "p2"])

    def test_empty_matches_returns_empty_ranking(self) -> None:
        payload = rank_bradley_terry([])
        self.assertEqual(payload["ranked_papers"], [])
        self.assertTrue(is_connected(tally_matches([])))


if __name__ == "__main__":
    unittest.main()
