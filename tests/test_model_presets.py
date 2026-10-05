import unittest

from icml_ai_ac.cli import build_parser
from icml_ai_ac.model_presets import PRODUCTION_SEMIFINAL_JUDGES


class ModelPresetTests(unittest.TestCase):
    def test_cli_defaults_match_production_model_configuration(self) -> None:
        parser = build_parser()
        pass1 = parser.parse_args(
            ["score-pass1-ensemble", "--manifest", "manifest.jsonl", "--out-dir", "runs"]
        )
        pass2 = parser.parse_args(
            [
                "rank-pass2",
                "--manifest",
                "manifest.jsonl",
                "--pass1",
                "pass1.jsonl",
                "--out",
                "pass2.json",
                "--run-dir",
                "runs",
            ]
        )
        pass2_batches = parser.parse_args(
            [
                "rank-pass2-batches",
                "--manifest",
                "manifest.jsonl",
                "--pass1",
                "pass1.jsonl",
                "--out",
                "pass2.json",
                "--run-dir",
                "runs",
            ]
        )
        classify = parser.parse_args(
            [
                "classify-contributions",
                "--manifest",
                "manifest.jsonl",
                "--out",
                "classes.jsonl",
                "--run-dir",
                "runs",
            ]
        )
        tournament = parser.parse_args(
            [
                "rank-frontier-card-tournament",
                "--cards",
                "cards.jsonl",
                "--seed-ranking",
                "seed.json",
                "--out",
                "ranking.json",
                "--run-dir",
                "runs",
            ]
        )

        self.assertEqual(pass1.model_preset, "production_2026_v2")
        self.assertEqual(pass1.reasoning_effort, "none")
        self.assertEqual(pass1.model_workers, 1)
        self.assertEqual(pass1.timeout, 180.0)
        self.assertEqual(classify.model, "google/gemini-3.1-flash-lite")
        self.assertEqual(classify.batch_size, 16)
        self.assertEqual(pass2.model, "gpt-5.6-terra")
        self.assertEqual(pass2.reasoning_effort, "high")
        self.assertEqual(pass2.timeout, 600.0)
        self.assertEqual(pass2_batches.model, "gpt-5.6-terra")
        self.assertEqual(pass2_batches.batch_size, 8)
        self.assertEqual(pass2_batches.partitions, 2)
        self.assertEqual(pass2_batches.max_output_tokens, 12_000)
        self.assertEqual(pass2_batches.timeout, 600.0)
        self.assertEqual(tournament.model, "openai/gpt-5.6-sol")
        self.assertEqual(tournament.reasoning_effort, "xhigh")
        self.assertEqual(tournament.timeout, 600.0)
        self.assertEqual(
            [(judge.provider, judge.model, judge.reasoning_effort) for judge in PRODUCTION_SEMIFINAL_JUDGES],
            [
                ("openrouter", "openai/gpt-5.6-terra", "high"),
                ("openrouter", "anthropic/claude-sonnet-5", "high"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
