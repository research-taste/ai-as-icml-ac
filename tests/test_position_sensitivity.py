import unittest

from icml_ai_ac.position_sensitivity import estimate_slot_effects, solve_linear_system


def row(paper_id: str, slot: int, priority: float) -> dict:
    return {
        "status": "ok",
        "paper_id": paper_id,
        "input_slot": slot,
        "batch_local_priority": priority,
    }


class PositionSensitivityTests(unittest.TestCase):
    def test_estimates_centered_slot_effects_from_within_paper_pairs(self) -> None:
        rows = [
            row("p1", 0, 0.8),
            row("p1", 1, 0.6),
            row("p2", 1, 0.7),
            row("p2", 2, 0.6),
            row("p3", 0, 0.9),
            row("p3", 2, 0.6),
        ]

        effects, report = estimate_slot_effects(rows)

        self.assertAlmostEqual(sum(effects), 0.0)
        self.assertAlmostEqual(effects[0] - effects[1], 0.2)
        self.assertAlmostEqual(effects[1] - effects[2], 0.1)
        self.assertEqual(report["earlier_wins"], 3)
        self.assertEqual(report["later_wins"], 0)
        self.assertEqual(report["cross_slot_pair_count"], 3)

    def test_rejects_incomplete_partition_coverage(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected two partition rows"):
            estimate_slot_effects(
                [
                    row("p1", 0, 0.8),
                    row("p2", 0, 0.7),
                    row("p2", 1, 0.6),
                ]
            )

    def test_linear_solver_rejects_disconnected_design(self) -> None:
        with self.assertRaisesRegex(ValueError, "disconnected"):
            solve_linear_system([[1.0, 0.0], [0.0, 0.0]], [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
