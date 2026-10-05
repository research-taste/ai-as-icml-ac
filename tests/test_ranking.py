import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from icml_ai_ac.models import PaperRecord
from icml_ai_ac.scoring.ranking import write_ranking_score_rows
from icml_ai_ac.scoring.ranking import (
    Pass2BatchedRankingConfig,
    Pass2RankingConfig,
    RankingCandidate,
    aggregate_pass2_judgments,
    archive_failed_pass2_attempt,
    balance_pass2_batches,
    build_pass2_ranking_messages,
    clear_failed_pass2_attempt,
    load_cached_pass2_batch,
    run_pass2_batched_ranking,
    validate_pass2_comparison_connectivity,
)
from icml_ai_ac.eval import read_ranking
from icml_ai_ac.storage import write_jsonl


class FakeRankingClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, messages, temperature, max_output_tokens, seed=None, response_format=None):
        self.calls += 1
        paper_ids = re.findall(r'<CANDIDATE paper_id="([^"]+)">', messages[-1]["content"])
        content = json.dumps(
            {
                "ranked_papers": [
                    {
                        "rank": rank,
                        "paper_id": paper_id,
                        "title": paper_id,
                        "primary_contribution_class": "theory",
                        "overall_priority_score": 10 - rank,
                        "broad_scientific_impact_score": 9 - rank,
                        "ml_field_impact_score": 8 - rank,
                        "technical_soundness_score": 7,
                        "advance_to_final_review": rank == 1,
                        "why_ranked_here": "test",
                    }
                    for rank, paper_id in enumerate(paper_ids, start=1)
                ]
            }
        )
        return SimpleNamespace(
            response={"model": "served"},
            content=content,
            usage={"prompt_tokens": 20, "cost": 0.02},
            served_model="served",
            elapsed_seconds=0.01,
        )


class RankingTests(unittest.TestCase):
    def test_failed_batch_is_archived_before_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            batch_dir = Path(tmp)
            for name, payload in (
                ("prompt.json", {"fingerprint": "abc"}),
                ("response.json", {"model": "served"}),
                ("error.json", {"error": "bad output"}),
                ("batch.json", {"status": "failed", "fingerprint": "abc"}),
            ):
                (batch_dir / name).write_text(json.dumps(payload), encoding="utf-8")

            attempt_count = archive_failed_pass2_attempt(batch_dir)
            clear_failed_pass2_attempt(batch_dir)

            self.assertEqual(attempt_count, 1)
            archived = batch_dir / "attempts" / "attempt_0001"
            self.assertTrue((archived / "response.json").exists())
            self.assertTrue((archived / "error.json").exists())
            self.assertTrue((archived / "batch.json").exists())
            self.assertTrue((batch_dir / "prompt.json").exists())
            self.assertFalse((batch_dir / "batch.json").exists())

    def test_pass2_prompt_excludes_human_outcome_metadata(self) -> None:
        record = PaperRecord(
            paper_id="p1",
            source="SENTINEL_HUMAN_SOURCE",
            title="Paper",
            decision_label="SENTINEL_HUMAN_DECISION",
        )
        candidate = RankingCandidate(
            record=record,
            pass1_row={"scores": {"summary": {"one_sentence_contribution": "test"}}},
            text_path="paper.txt",
            resolved_text_source="scoring",
            paper_text="Paper content only.",
        )
        config = Pass2RankingConfig(
            provider="openrouter",
            model="test",
            reasoning_effort=None,
            prompt_version="test",
            text_source="scoring",
            top_fraction=1,
            limit=None,
            per_paper_char_budget=1000,
            temperature=0,
            max_output_tokens=1000,
            seed=None,
            dry_run=True,
        )

        messages = build_pass2_ranking_messages([candidate], config=config)
        prompt_text = "\n".join(str(message["content"]) for message in messages)

        self.assertNotIn("SENTINEL_HUMAN_SOURCE", prompt_text)
        self.assertNotIn("SENTINEL_HUMAN_DECISION", prompt_text)
        self.assertIn("No reviews, reviewer scores", prompt_text)

    def test_batched_ranking_has_exact_coverage_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            pass1_rows = []
            for index in range(6):
                paper_id = f"p{index}"
                text_path = root / f"{paper_id}.txt"
                text_path.write_text(f"Scoring text for {paper_id}", encoding="utf-8")
                records.append(
                    PaperRecord(
                        paper_id=paper_id,
                        source="accepted",
                        title=paper_id,
                        text_scoring=str(text_path),
                        parse_status="ok",
                    ).to_dict()
                )
                pass1_rows.append(
                    {
                        "status": "ok",
                        "paper_id": paper_id,
                        "scores": {
                            "scores": {"executive_ac_priority": 10 - index},
                            "contribution_profile": {"primary_contribution_class": "theory"},
                        },
                    }
                )
            manifest = root / "manifest.jsonl"
            pass1 = root / "pass1.jsonl"
            out = root / "ranking.json"
            run_dir = root / "run"
            write_jsonl(manifest, records)
            write_jsonl(pass1, pass1_rows)
            config = Pass2BatchedRankingConfig(
                provider="openrouter",
                model="test/model",
                reasoning_effort="high",
                prompt_version="test",
                text_source="scoring",
                top_fraction=1,
                limit=None,
                per_paper_char_budget=1000,
                batch_size=3,
                partitions=2,
                strategy="shuffled",
                class_path=None,
                request_delay_seconds=0,
                temperature=0,
                max_output_tokens=1000,
                seed=11,
                dry_run=False,
            )
            client = FakeRankingClient()

            first = run_pass2_batched_ranking(
                manifest=manifest,
                pass1_path=pass1,
                out=out,
                run_dir=run_dir,
                config=config,
                client=client,
            )
            second = run_pass2_batched_ranking(
                manifest=manifest,
                pass1_path=pass1,
                out=out,
                run_dir=run_dir,
                config=config,
                client=client,
            )

            ranked = first["ranking"]["ranked_papers"]
            self.assertEqual(first["status"], "ok")
            self.assertEqual(first["judgment_count"], 12)
            self.assertEqual(sorted(row["rank"] for row in ranked), list(range(1, 7)))
            self.assertTrue(all(row["batch_judgment_count"] == 2 for row in ranked))
            self.assertEqual(first["usage"]["cost"], 0.08)
            self.assertEqual(client.calls, 4)
            self.assertEqual(second["resumed_batch_count"], 4)

    def test_cached_recovery_uses_effective_prompt_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            batch_dir = Path(tmp)
            original_prompt = batch_dir / "prompt.json"
            recovery_prompt = batch_dir / "recovery" / "prompt.json"
            recovery_prompt.parent.mkdir()
            original_prompt.write_text("{}", encoding="utf-8")
            recovery_prompt.write_text("{}", encoding="utf-8")
            (batch_dir / "response.json").write_text("{}", encoding="utf-8")
            (batch_dir / "parsed.json").write_text(
                json.dumps(
                    {
                        "ranked_papers": [
                            {
                                "rank": 1,
                                "paper_id": "p1",
                                "title": "Paper",
                                "primary_contribution_class": "theory",
                                "overall_priority_score": 8,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (batch_dir / "batch.json").write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "fingerprint": "source-fingerprint",
                        "effective_prompt_path": str(recovery_prompt),
                    }
                ),
                encoding="utf-8",
            )
            candidate = RankingCandidate(
                record=PaperRecord(paper_id="p1", source="blinded_scoring", title="Paper"),
                pass1_row={},
                text_path="paper.txt",
                resolved_text_source="scoring",
                paper_text="Paper content.",
            )

            cached = load_cached_pass2_batch(
                batch_dir=batch_dir,
                fingerprint="source-fingerprint",
                candidates=[candidate],
                prompt_path=original_prompt,
                partition_index=1,
                batch_index=2,
            )

            self.assertIsNotNone(cached)
            judgments, state = cached
            self.assertEqual(judgments[0]["prompt_path"], str(recovery_prompt))
            self.assertTrue(state["resumed"])

    def test_batched_aggregation_is_invariant_to_priority_score_scale(self) -> None:
        candidates = [
            RankingCandidate(
                record=PaperRecord(
                    paper_id=paper_id,
                    source="blinded_scoring",
                    title=f"Paper {paper_id}",
                ),
                pass1_row={},
                text_path=f"{paper_id}.txt",
                resolved_text_source="scoring",
                paper_text="Paper content.",
            )
            for paper_id in ("p1", "p2")
        ]

        def build_judgments(second_batch_scale: float) -> list[dict]:
            rows = []
            for partition_index, values in enumerate(
                (
                    (("p1", 1, 9.0), ("p2", 2, 7.0)),
                    (
                        ("p1", 2, 7.0 * second_batch_scale),
                        ("p2", 1, 9.0 * second_batch_scale),
                    ),
                )
            ):
                for paper_id, local_rank, score in values:
                    rows.append(
                        {
                            "paper_id": paper_id,
                            "partition_index": partition_index,
                            "batch_index": 0,
                            "batch_size": 2,
                            "local_rank": local_rank,
                            "normalized_local_rank": float(local_rank - 1),
                            "judgment": {
                                "paper_id": paper_id,
                                "rank": local_rank,
                                "overall_priority_score": score,
                                "broad_scientific_impact_score": 8,
                                "ml_field_impact_score": 8,
                                "technical_soundness_score": 8,
                            },
                        }
                    )
            return rows

        canonical = aggregate_pass2_judgments(candidates, build_judgments(1.0))
        mixed_scale = aggregate_pass2_judgments(candidates, build_judgments(10.0))

        self.assertEqual(
            [row["paper_id"] for row in canonical["ranked_papers"]],
            [row["paper_id"] for row in mixed_scale["ranked_papers"]],
        )
        self.assertEqual(
            [row["overall_priority_score"] for row in canonical["ranked_papers"]],
            [row["overall_priority_score"] for row in mixed_scale["ranked_papers"]],
        )
        self.assertEqual(
            mixed_scale["ranked_papers"][0]["overall_priority_score_source_scales"],
            ["0_10", "0_100"],
        )

    def test_balanced_batches_avoid_singleton_tail(self) -> None:
        self.assertEqual([len(batch) for batch in balance_pass2_batches(list(range(5)), 2)], [3, 2])
        self.assertEqual([len(batch) for batch in balance_pass2_batches(list(range(11)), 8)], [6, 5])

    def test_batched_ranking_requires_connected_comparison_graph(self) -> None:
        judgments = [
            {"paper_id": paper_id, "partition_index": 0, "batch_index": batch_index}
            for batch_index, paper_ids in enumerate((["p1", "p2"], ["p3", "p4"]))
            for paper_id in paper_ids
        ]

        errors = validate_pass2_comparison_connectivity(
            judgments,
            candidate_ids=["p1", "p2", "p3", "p4"],
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("disconnected", errors[0])

    def test_batched_ranking_rejects_disconnected_plan_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            pass1_rows = []
            for index in range(4):
                paper_id = f"p{index}"
                text_path = root / f"{paper_id}.txt"
                text_path.write_text(paper_id, encoding="utf-8")
                records.append(
                    PaperRecord(
                        paper_id=paper_id,
                        source="accepted",
                        title=paper_id,
                        text_scoring=str(text_path),
                        parse_status="ok",
                    ).to_dict()
                )
                pass1_rows.append(
                    {
                        "status": "ok",
                        "paper_id": paper_id,
                        "scores": {"scores": {"executive_ac_priority": 10 - index}},
                    }
                )
            manifest = root / "manifest.jsonl"
            pass1 = root / "pass1.jsonl"
            write_jsonl(manifest, records)
            write_jsonl(pass1, pass1_rows)
            client = FakeRankingClient()
            config = Pass2BatchedRankingConfig(
                provider="openrouter",
                model="test/model",
                reasoning_effort="high",
                prompt_version="test",
                text_source="scoring",
                top_fraction=1,
                limit=None,
                per_paper_char_budget=1000,
                batch_size=2,
                partitions=1,
                strategy="sequential",
                class_path=None,
                request_delay_seconds=0,
                temperature=0,
                max_output_tokens=1000,
                seed=11,
                dry_run=False,
            )

            with self.assertRaisesRegex(ValueError, "disconnected"):
                run_pass2_batched_ranking(
                    manifest=manifest,
                    pass1_path=pass1,
                    out=root / "out.json",
                    run_dir=root / "run",
                    config=config,
                    client=client,
                )

            self.assertEqual(client.calls, 0)

    def test_write_ranking_score_rows_converts_pass2_json_for_stage2(self) -> None:
        result = {
            "provider": "openai",
            "model": "gpt-5.4",
            "prompt_version": "test_prompt",
            "out": "stage1.json",
            "ranking": {
                "ranked_papers": [
                    {
                        "rank": 1,
                        "paper_id": "p1",
                        "title": "Paper 1",
                        "primary_contribution_class": "theory",
                        "overall_priority_score": 9,
                        "broad_scientific_impact_score": 8,
                        "ml_field_impact_score": 9,
                        "technical_soundness_score": 8,
                        "advance_to_final_review": True,
                        "why_ranked_here": "strong",
                    }
                ]
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rows.jsonl"
            count = write_ranking_score_rows(result, out)
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, 1)
        self.assertEqual(rows[0]["paper_id"], "p1")
        self.assertEqual(rows[0]["scores"]["contribution_profile"]["primary_contribution_class"], "theory")
        self.assertEqual(rows[0]["scores"]["scores"]["executive_ac_priority"], 9.0)

    def test_eval_ranking_rejects_duplicate_paper_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ranking.json"
            path.write_text(
                json.dumps(
                    {
                        "ranked_papers": [
                            {"rank": 1, "paper_id": "p1"},
                            {"rank": 2, "paper_id": "p1"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate paper_id"):
                read_ranking(path)


if __name__ == "__main__":
    unittest.main()
