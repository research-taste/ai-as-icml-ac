from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("icml_site_build_data", ROOT / "site/build_data.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load site/build_data.py")
BUILD_DATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD_DATA)


class SpearmanTests(unittest.TestCase):
    def test_perfect_and_inverted_orderings(self) -> None:
        self.assertAlmostEqual(BUILD_DATA.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(BUILD_DATA.spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)

    def test_monotone_but_nonlinear_still_reads_as_perfect(self) -> None:
        # Rank correlation must not be fooled by scale, which is why it is used here.
        self.assertAlmostEqual(BUILD_DATA.spearman([1, 2, 3, 4], [1, 4, 9, 16]), 1.0)

    def test_ties_share_an_average_rank_rather_than_an_arbitrary_order(self) -> None:
        self.assertEqual(BUILD_DATA.rank_values([5.0, 5.0, 9.0]), [1.5, 1.5, 3.0])
        # A constant column has no ordering to correlate with, so the result is 0 not NaN.
        self.assertEqual(BUILD_DATA.spearman([1, 2, 3], [7, 7, 7]), 0.0)

    def test_rejects_unpaired_inputs(self) -> None:
        with self.assertRaises(ValueError):
            BUILD_DATA.spearman([1, 2], [1, 2, 3])


class BootstrapTests(unittest.TestCase):
    def test_interval_is_reproducible_across_calls(self) -> None:
        keys = [str(index) for index in range(80)]
        statistic = lambda pool: sum(int(key) for key in pool) / len(pool)
        first = BUILD_DATA.bootstrap_interval(keys, statistic, samples=200)
        second = BUILD_DATA.bootstrap_interval(keys, statistic, samples=200)
        self.assertEqual(first, second, "a published interval must not move between builds")
        self.assertLess(first[0], first[1])


class ReasonAxisTests(unittest.TestCase):
    """The regexes decide what the published figure claims, so they are pinned here."""

    def patterns(self) -> dict[str, object]:
        import re

        return {key: re.compile(pattern, re.I) for key, _, _, pattern in BUILD_DATA.REASON_AXES}

    def test_generality_objection_matches_the_language_models_actually_used(self) -> None:
        narrow = self.patterns()["narrow"]
        for sentence in [
            "Limited transferability beyond aerial multi-modal object detection benchmarks.",
            "Specialized domain focus with narrow adoption outside of clinical neuroimaging.",
            "Performance gains tied to specific drone datasets and modalities.",
            "may not generalize beyond the specific brain network analysis task",
        ]:
            self.assertRegex(sentence, narrow, f"should read as a generality objection: {sentence}")

    def test_generality_objection_does_not_fire_on_unrelated_praise(self) -> None:
        narrow = self.patterns()["narrow"]
        self.assertIsNone(narrow.search("Convergence proof is clean and the ablations are thorough."))

    def test_every_axis_declares_a_side(self) -> None:
        for key, kind, label, _ in BUILD_DATA.REASON_AXES:
            self.assertIn(kind, {"reward", "concern"}, key)
            self.assertTrue(label and label[0].isupper(), key)


class ModelAgreementTests(unittest.TestCase):
    def judgments(self) -> dict[str, dict[str, dict[str, object]]]:
        # Two models that rank papers identically, and a third that inverts them.
        papers = [str(index) for index in range(1200)]
        agreeing = {paper: {"priority": float(paper), "flags": set()} for paper in papers}
        inverted = {paper: {"priority": -float(paper), "flags": set()} for paper in papers}
        return {"A": agreeing, "B": dict(agreeing), "C": inverted}

    def test_reports_every_pair_and_the_human_column(self) -> None:
        judgments = self.judgments()
        reviewer = {paper: float(paper) for paper in judgments["A"]}
        result = BUILD_DATA.build_model_agreement(judgments, reviewer=reviewer)
        self.assertEqual(result["papers"], 1200)
        self.assertEqual(len(result["pairs"]), 3, "three models make three unordered pairs")
        self.assertEqual(len(result["human"]), 3)
        pairs = {(row["a"], row["b"]): row["rho"] for row in result["pairs"]}
        self.assertAlmostEqual(pairs[("A", "B")], 1.0)
        self.assertAlmostEqual(pairs[("A", "C")], -1.0)
        self.assertAlmostEqual(result["weakestModelPair"], -1.0)

    def test_refuses_to_publish_a_thin_join(self) -> None:
        judgments = self.judgments()
        with self.assertRaises(ValueError) as caught:
            BUILD_DATA.build_model_agreement(judgments, reviewer={"1": 4.0})
        self.assertIn("only 1 papers", str(caught.exception))


class BuildJudgmentBasisTests(unittest.TestCase):
    def test_returns_none_when_the_analysis_bundle_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metadata").mkdir()
            (root / "scores").mkdir()
            result = BUILD_DATA.build_judgment_basis(
                model_runs_dir=root / "model_runs",
                metadata_dir=root / "metadata",
                scores_dir=root / "scores",
                evals_dir=root / "evals",
            )
        self.assertIsNone(result, "a clone without the bundle must still build")


class AgreementByRangeTests(unittest.TestCase):
    def test_pairwise_agreement_reports_the_spread_not_just_the_mean(self) -> None:
        papers = [str(index) for index in range(50)]
        scores = {
            "a": {paper: float(paper) for paper in papers},
            "b": {paper: float(paper) for paper in papers},
            "c": {paper: -float(paper) for paper in papers},
        }
        result = BUILD_DATA.pairwise_agreement(scores)
        assert result is not None
        self.assertEqual(result["judges"], 3)
        self.assertEqual(result["papers"], 50)
        self.assertAlmostEqual(result["max"], 1.0)
        self.assertAlmostEqual(result["min"], -1.0)

    def test_pairwise_agreement_needs_two_judges_and_shared_papers(self) -> None:
        self.assertIsNone(BUILD_DATA.pairwise_agreement({"a": {"1": 1.0}}))
        self.assertIsNone(
            BUILD_DATA.pairwise_agreement({"a": {"1": 1.0}, "b": {"2": 1.0}}),
            "judges with no papers in common cannot be correlated",
        )


class ReasonAxisScaleTests(unittest.TestCase):
    def test_generality_buckets_and_tier_shares_partition_the_corpus(self) -> None:
        papers = [str(index) for index in range(1500)]
        # Every third paper draws the objection from both models.
        def per_model(flagged_every: int) -> dict[str, dict[str, object]]:
            return {
                paper: {
                    "priority": 1.0,
                    "flags": {"narrow"} if int(paper) % flagged_every == 0 else set(),
                }
                for paper in papers
            }

        judgments = {"A": per_model(3), "B": per_model(3)}
        tiers = {paper: ("oral" if int(paper) % 10 == 0 else "poster") for paper in papers}
        result = BUILD_DATA.build_reason_axes(
            judgments,
            reviewer={paper: float(int(paper) % 7) for paper in papers},
            tiers=tiers,
            ranks={paper: float(paper) for paper in papers},
        )
        self.assertEqual(result["papers"], 1500)
        self.assertEqual(result["models"], 2)
        self.assertEqual(
            sum(bucket["papers"] for bucket in result["generality"]["buckets"]),
            1500,
            "every paper must land in exactly one bucket",
        )
        self.assertEqual(result["generality"]["majority"], 2, "2 of 2 models is the majority")
        self.assertEqual(
            {bucket["modelsRaising"] for bucket in result["generality"]["buckets"]},
            {0, 2},
            "both models agree by construction, so no paper has exactly one",
        )
        self.assertEqual(
            sum(block["papers"] for block in result["generality"]["byTier"].values()),
            1500,
        )
        for axis in result["axes"]:
            self.assertGreaterEqual(axis["coverage"], 0.0)
            self.assertLessEqual(axis["coverage"], 1.0)

    def test_refuses_to_publish_a_thin_join(self) -> None:
        with self.assertRaises(ValueError) as caught:
            BUILD_DATA.build_reason_axes(
                {"A": {"1": {"priority": 1.0, "flags": set()}}},
                reviewer={"1": 4.0},
                tiers={"1": "oral"},
                ranks={"1": 1.0},
            )
        self.assertIn("only 1 papers", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
