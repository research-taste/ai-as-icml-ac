import json
import tempfile
import unittest
from pathlib import Path

from icml_ai_ac.analysis.human_comparison import (
    build_divergence_report,
    load_ai_scores,
    load_comparison_dataset,
    load_comparison_rows,
    percentile_ranks,
    render_markdown,
    tier_of,
)


def manifest_row(pid: str, title: str, tier: str, overall: float, *, is_award: bool = False) -> dict:
    extra = {
        "is_oral": tier == "oral",
        "is_spotlight": tier == "spotlight",
        "is_award_paper": is_award,
        "decision": "Accept",
        "openreview_scores": {
            "overall_mean": overall,
            "soundness_mean": overall - 0.5,
            "confidence_mean": 4.0,
            "review_count": 4,
        },
    }
    if is_award:
        extra["award_labels"] = [{"award_type": "honorable_mention"}]
    return {"paper_id": pid, "source": "openreview", "title": title, "extra": extra}


def ai_row(pid: str, priority: float, cls: str = "theory") -> dict:
    return {
        "paper_id": pid,
        "status": "ok",
        "scores": {
            "scores": {"executive_ac_priority": priority, "overall_significance": priority},
            "calibration": {
                "estimated_percentile_among_accepted_papers": min(99.0, priority * 10),
                "why_not_higher": "wh",
                "why_not_lower": "wl",
                "reviewer_vs_executive_delta": "delta",
            },
            "contribution_profile": {"primary_contribution_class": cls},
            "executive_lens": {"why_it_might_matter": "matters", "sweeping_impact_scenario": f"scenario {pid}"},
            "ranking_signals": {"top_paper_case": "case", "broad_scientific_impact_argument": "arg", "dealbreaker_risks": "risk"},
            "reviewer_lens": {"human_review_alignment": "would disagree", "likely_reviewer_concerns": "concern"},
            "evidence_audit": {"observed_weaknesses": ["weak"]},
            "uncertainty": {"contamination_sensitivity": "low"},
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


PAPERS = [
    ("p_gem", "Gem Paper", "poster", 3.0, 9.0, False),
    ("p_oral_low", "Oral Low", "oral", 7.0, 2.0, False),
    ("p_aligned_top", "Aligned Top", "oral", 8.0, 8.5, False),
    ("p_aligned_bottom", "Aligned Bottom", "poster", 4.0, 2.5, False),
    ("p_spot_mid", "Spotlight Mid", "spotlight", 6.0, 5.0, False),
    ("p_award_low", "Award Low", "poster", 5.0, 2.2, True),
]


class HumanComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.manifest = self.tmp / "manifest.jsonl"
        self.ai_scores = self.tmp / "ai_scores.jsonl"
        write_jsonl(self.manifest, [manifest_row(p, t, tier, ov, is_award=aw) for p, t, tier, ov, _, aw in PAPERS])
        write_jsonl(self.ai_scores, [ai_row(p, pr) for p, _, _, _, pr, _ in PAPERS])

    def rows(self):
        return load_comparison_rows(manifest=self.manifest, ai_scores=self.ai_scores)

    def by_id(self, rows):
        return {row.paper_id: row for row in rows}

    def test_tier_of(self) -> None:
        self.assertEqual(tier_of({"is_oral": True}), ("oral", 2))
        self.assertEqual(tier_of({"is_spotlight": True}), ("spotlight", 1))
        self.assertEqual(tier_of({"is_oral": False, "is_spotlight": False}), ("poster", 0))
        self.assertEqual(tier_of({}), ("unknown", -1))
        self.assertEqual(tier_of({"decision": "Reject", "is_oral": False}), ("unknown", -1))

    def test_percentile_ranks_handles_ties(self) -> None:
        self.assertEqual(percentile_ranks([1.0, 1.0]), [50.0, 50.0])
        self.assertEqual(percentile_ranks([]), [])
        result = percentile_ranks([5.0, 1.0, 3.0])
        self.assertEqual(result[0], round(100 * 2.5 / 3, 3))  # highest value

    def test_join_extracts_human_and_ai_signals(self) -> None:
        rows = self.by_id(self.rows())
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows["p_gem"].tier, "poster")
        self.assertEqual(rows["p_oral_low"].tier, "oral")
        self.assertEqual(rows["p_spot_mid"].tier, "spotlight")
        self.assertTrue(rows["p_award_low"].is_award)
        self.assertEqual(rows["p_gem"].decision_status, "accepted")
        self.assertEqual(rows["p_gem"].reviewer_overall, 3.0)
        self.assertEqual(rows["p_gem"].contribution_class, "theory")
        self.assertEqual(rows["p_gem"].rationale["sweeping_impact_scenario"], "scenario p_gem")

    def test_ai_rank_derivation(self) -> None:
        rows = self.by_id(self.rows())
        self.assertEqual(rows["p_gem"].ai_rank, 1)
        self.assertEqual(rows["p_oral_low"].ai_rank, 6)

    def test_gems_and_blind_spots_tier_axis(self) -> None:
        report = build_divergence_report(self.rows(), human_axis="tier", cases_per_direction=10)
        gem_ids = [c["paper_id"] for c in report["overlooked_gems"]["cases"]]
        blind_ids = [c["paper_id"] for c in report["blind_spots"]["cases"]]

        self.assertEqual(gem_ids[0], "p_gem")
        self.assertNotIn("p_award_low", gem_ids)  # award papers are not "overlooked"
        self.assertEqual(blind_ids, ["p_oral_low"])
        self.assertNotIn("p_aligned_top", blind_ids)
        self.assertEqual(report["overlooked_gems"]["total_matching"], 1)
        self.assertEqual(report["blind_spots"]["total_matching"], 1)

        rows = self.by_id(self.rows())
        build_divergence_report([rows[p] for p in rows], human_axis="tier")
        self.assertGreater(rows["p_gem"].residual, 0)
        self.assertLess(rows["p_oral_low"].residual, 0)

    def test_crosstab_and_summary(self) -> None:
        report = build_divergence_report(self.rows(), human_axis="tier")
        table = report["tier_ai_decile_crosstab"]["table"]
        total = sum(cell for tier in table.values() for cell in tier.values())
        self.assertEqual(total, 6)
        self.assertEqual(report["residual_summary"]["count"], 6)
        self.assertEqual(
            report["counts"]["by_tier"],
            {"oral": 2, "spotlight": 1, "poster": 3, "unknown": 0},
        )

    def test_reviewer_axis(self) -> None:
        report = build_divergence_report(
            self.rows(),
            human_axis="reviewer",
            top_frac=0.25,
            cases_per_direction=10,
        )
        gem_ids = [c["paper_id"] for c in report["overlooked_gems"]["cases"]]
        blind_ids = [c["paper_id"] for c in report["blind_spots"]["cases"]]
        self.assertIn("p_gem", gem_ids)
        self.assertIn("p_oral_low", blind_ids)

    def test_markdown_renders(self) -> None:
        report = build_divergence_report(self.rows(), human_axis="tier")
        md = render_markdown(report)
        self.assertIn("Overlooked gems", md)
        self.assertIn("Gem Paper", md)

    def test_strong_ranking_merged(self) -> None:
        strong = self.tmp / "strong.json"
        strong.write_text(json.dumps({"ranked_papers": [{"paper_id": "p_gem", "rank": 1}, {"paper_id": "p_aligned_top", "rank": 2}]}))
        rows = self.by_id(load_comparison_rows(manifest=self.manifest, ai_scores=self.ai_scores, strong_ranking=strong))
        self.assertEqual(rows["p_gem"].strong_rank, 1)
        self.assertIsNone(rows["p_oral_low"].strong_rank)

    def test_unjoined_manifest_paper_is_skipped(self) -> None:
        write_jsonl(self.manifest, [manifest_row("only_in_manifest", "X", "poster", 5.0)])
        self.assertEqual(self.rows(), [])

    def test_load_ai_scores_rejects_duplicate_judgments(self) -> None:
        write_jsonl(self.ai_scores, [ai_row("dup", 3.0), ai_row("dup", 7.0)])
        with self.assertRaisesRegex(ValueError, "duplicate paper rows"):
            load_ai_scores(self.ai_scores)

    def test_ensemble_aggregate_rank_is_the_canonical_signal(self) -> None:
        first = ai_row("p_gem", 1.0)
        first["aggregate_rank"] = 1
        second = ai_row("p_oral_low", 10.0)
        second["aggregate_rank"] = 2
        write_jsonl(self.ai_scores, [first, second])
        write_jsonl(
            self.manifest,
            [
                manifest_row("p_gem", "Gem", "poster", 3.0),
                manifest_row("p_oral_low", "Oral", "oral", 7.0),
            ],
        )
        rows = {row.paper_id: row for row in self.rows()}
        self.assertGreater(rows["p_gem"].ai_score, rows["p_oral_low"].ai_score)
        self.assertEqual(rows["p_gem"].ai_signal_kind, "ensemble_aggregate_rank")

    def test_strict_coverage_rejects_partial_shortlist(self) -> None:
        write_jsonl(self.ai_scores, [ai_row("p_gem", 9.0)])
        with self.assertRaisesRegex(ValueError, "coverage"):
            load_comparison_dataset(
                manifest=self.manifest,
                ai_scores=self.ai_scores,
                min_coverage=1.0,
            )
        dataset = load_comparison_dataset(
            manifest=self.manifest,
            ai_scores=self.ai_scores,
            min_coverage=0.1,
        )
        self.assertEqual(dataset.coverage["joined_papers"], 1)
        self.assertEqual(dataset.coverage["missing_ai_score_count"], 5)

    def test_decision_axis_keeps_rejected_papers_out_of_poster_tier(self) -> None:
        rejected = manifest_row("rejected", "Rejected", "poster", 2.0)
        rejected["extra"]["decision"] = "Reject"
        accepted = manifest_row("accepted", "Accepted", "poster", 7.0)
        write_jsonl(self.manifest, [rejected, accepted])
        write_jsonl(self.ai_scores, [ai_row("rejected", 9.0), ai_row("accepted", 1.0)])
        rows = self.by_id(self.rows())
        self.assertEqual(rows["rejected"].tier, "unknown")
        self.assertEqual(rows["rejected"].decision_status, "rejected")
        report = build_divergence_report(
            list(rows.values()),
            human_axis="decision",
            top_frac=0.25,
        )
        self.assertEqual(
            [case["paper_id"] for case in report["overlooked_gems"]["cases"]],
            ["rejected"],
        )
        self.assertEqual(
            [case["paper_id"] for case in report["blind_spots"]["cases"]],
            ["accepted"],
        )

    def test_invalid_config_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "human_axis"):
            build_divergence_report(self.rows(), human_axis="bogus")
        with self.assertRaisesRegex(ValueError, "top_frac"):
            build_divergence_report(self.rows(), top_frac=1.5)
        with self.assertRaisesRegex(ValueError, "cases_per_direction"):
            build_divergence_report(self.rows(), cases_per_direction=0)


if __name__ == "__main__":
    unittest.main()
