import json
import tempfile
import unittest
from pathlib import Path

from icml_ai_ac.analysis.semifinal_audit import audit_semifinal_rankings
from icml_ai_ac.storage import write_jsonl


class SemifinalAuditTests(unittest.TestCase):
    def test_complete_blinded_two_judge_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            shortlist = root / "shortlist.jsonl"
            write_jsonl(
                manifest,
                [
                    {
                        "paper_id": paper_id,
                        "source": "blinded_scoring",
                        "title": f"Paper {paper_id}",
                        "authors": [],
                        "decision_label": None,
                        "forum_url": None,
                        "pdf_url": None,
                        "notes": None,
                        "final_rank_scores": None,
                        "pairwise_results": None,
                        "scores_pass1": None,
                        "scores_pass2": None,
                        "extra": {
                            "anonymization": {"status": "ok"},
                            "blind_manifest": {
                                "human_outcomes_available_to_models": False,
                            },
                            "scoring_artifacts": {},
                        },
                    }
                    for paper_id in ("p1", "p2")
                ],
            )
            write_jsonl(
                shortlist,
                [
                    {
                        "paper_id": paper_id,
                        "title": f"Paper {paper_id}",
                        "rank": rank,
                        "ensemble_contribution_class": "theory",
                    }
                    for rank, paper_id in enumerate(("p1", "p2"), start=1)
                ],
            )

            rankings = []
            models = ("test/terra", "test/sonnet")
            for label, model in zip(("terra", "sonnet"), models, strict=True):
                run_dir = root / label
                batch_results = []
                source_judgments = {"p1": [], "p2": []}
                for partition_index, candidate_ids in enumerate(
                    (["p1", "p2"], ["p2", "p1"])
                ):
                    batch_dir = run_dir / f"partition_{partition_index}"
                    batch_dir.mkdir(parents=True)
                    prompt_path = batch_dir / "prompt.json"
                    response_path = batch_dir / "response.json"
                    parsed_path = batch_dir / "parsed.json"
                    fingerprint = f"{label}-{partition_index}"
                    prompt_path.write_text(
                        json.dumps(
                            {
                                "fingerprint": fingerprint,
                                "candidate_ids": candidate_ids,
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": "Rank anonymized paper content.",
                                    }
                                ],
                            }
                        ),
                        encoding="utf-8",
                    )
                    response_path.write_text("{}", encoding="utf-8")
                    parsed_path.write_text("{}", encoding="utf-8")
                    batch_results.append(
                        {
                            "status": "ok",
                            "partition_index": partition_index,
                            "batch_index": 0,
                            "candidate_ids": candidate_ids,
                            "fingerprint": fingerprint,
                            "prompt_path": str(prompt_path),
                            "raw_response_path": str(response_path),
                            "parsed_response_path": str(parsed_path),
                            "served_model": model,
                        }
                    )
                    for paper_id in candidate_ids:
                        local_rank = candidate_ids.index(paper_id) + 1
                        source_judgments[paper_id].append(
                            {
                                "paper_id": paper_id,
                                "partition_index": partition_index,
                                "batch_index": 0,
                                "batch_size": 2,
                                "local_rank": local_rank,
                                "normalized_local_rank": float(local_rank - 1),
                                "prompt_path": str(prompt_path),
                                "raw_response_path": str(response_path),
                                "parsed_response_path": str(parsed_path),
                                "judgment": {
                                    "paper_id": paper_id,
                                    "rank": local_rank,
                                    "overall_priority_score": 9 - local_rank,
                                },
                            }
                        )

                ranked_papers = []
                for rank, paper_id in enumerate(("p1", "p2"), start=1):
                    ranked_papers.append(
                        {
                            "paper_id": paper_id,
                            "title": f"Paper {paper_id}",
                            "rank": rank,
                            "primary_contribution_class": "theory",
                            "overall_priority_score": 7.5,
                            "broad_scientific_impact_score": 7,
                            "ml_field_impact_score": 7,
                            "technical_soundness_score": 7,
                            "mean_normalized_local_rank": 0.5,
                            "local_rank_range": 1.0,
                            "advance_to_final_review": True,
                            "source_batch_judgments": source_judgments[paper_id],
                            "overall_priority_score_source_scales": [
                                "0_10",
                                "0_10",
                            ],
                        }
                    )
                ranking_path = root / f"{label}.json"
                ranking_path.write_text(
                    json.dumps(
                        {
                            "status": "ok",
                            "provider": "openrouter",
                            "model": model,
                            "served_model": model,
                            "served_models": [model],
                            "candidate_count": 2,
                            "candidate_ids": ["p1", "p2"],
                            "batch_count": 2,
                            "judgment_count": 4,
                            "expected_judgment_count": 4,
                            "coverage_errors": [],
                            "batch_results": batch_results,
                            "ranking": {
                                "aggregation": "pass2_batch_rank_with_calibrated_priority_v2",
                                "tie_breaking": [
                                    "higher_calibrated_overall_priority_score",
                                ],
                                "ranked_papers": ranked_papers,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                rankings.append(ranking_path)

            report = audit_semifinal_rankings(
                ranking_paths=rankings,
                labels=["terra", "sonnet"],
                expected_models=list(models),
                shortlist_path=shortlist,
                manifest_path=manifest,
                out=root / "audit.json",
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["blocking_issues"], [])
            self.assertEqual(report["comparison"]["spearman_rank_correlation"], 1.0)


if __name__ == "__main__":
    unittest.main()
