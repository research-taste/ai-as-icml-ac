import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from icml_ai_ac.models import PaperRecord
from icml_ai_ac.scoring.batch import (
    BatchCandidate,
    Pass1BatchConfig,
    Pass1BatchSuiteConfig,
    build_pass1_batch_messages,
    batch_response_to_rows,
    parse_batch_json_response,
    run_pass1_batch_suite,
)
from icml_ai_ac.storage import write_jsonl


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, messages, temperature, max_output_tokens, seed=None, response_format=None):
        self.calls += 1
        paper_ids = re.findall(r'<PAPER paper_id="(p\d+)">', messages[-1]["content"])
        content = json.dumps(
            {
                "ranked_papers": [
                    {
                        "rank": rank,
                        "paper_id": paper_id,
                        "forced_bucket": "top_10_percent" if rank == 1 else "middle",
                        "estimated_percentile_among_batch": 100 - rank * 10,
                        "primary_contribution_class": "theory",
                        "executive_ac_priority": 10 - rank,
                        "broader_science_impact_forecast": 9 - rank,
                        "ml_field_impact_forecast": 8 - rank,
                        "technical_soundness": 7,
                        "overall_significance": 8,
                        "should_advance_to_strong_model": rank == 1,
                        "category_standout": rank == 1,
                    }
                    for rank, paper_id in enumerate(paper_ids, start=1)
                ]
            }
        )
        return SimpleNamespace(
            response={
                "model": "served",
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 20, "cost": 0.02},
            },
            request={"reasoning": {"effort": "none", "exclude": True}},
            content=content,
            usage={"prompt_tokens": 20, "cost": 0.02},
            served_model="served",
            elapsed_seconds=0.01,
        )


class MalformedThenValidClient(FakeClient):
    def complete(self, **kwargs):
        result = super().complete(**kwargs)
        if self.calls == 1:
            result.content = '{"ranked_papers": ['
            result.response["choices"][0]["message"]["content"] = result.content
        return result


class Pass1BatchResumeTests(unittest.TestCase):
    def test_batch_parser_repairs_unescaped_quotes_in_canonical_title(self) -> None:
        candidate = BatchCandidate(
            PaperRecord(
                paper_id="p1",
                source="accepted",
                title='How Text Becomes The "Jailbreak Key"',
            ),
            "paper.txt",
            "scoring",
            "Paper text.",
        )
        content = """{
  "ranked_papers": [
    {
      "rank": 1,
      "paper_id": "p1",
      "title": "How Text Becomes The "Jailbreak Key"",
      "forced_bucket": "top_10_percent"
    }
  ]
}"""

        parsed = parse_batch_json_response(content, candidates=[candidate])

        self.assertEqual(
            parsed["ranked_papers"][0]["title"],
            'How Text Becomes The "Jailbreak Key"',
        )

    def test_batch_rows_repair_duplicate_id_using_exact_canonical_title(self) -> None:
        candidates = [
            BatchCandidate(
                PaperRecord(paper_id="p1", source="accepted", title="First Paper"),
                "p1.txt",
                "scoring",
                "First.",
            ),
            BatchCandidate(
                PaperRecord(paper_id="p2", source="accepted", title="Second Paper"),
                "p2.txt",
                "scoring",
                "Second.",
            ),
        ]
        parsed = {
            "ranked_papers": [
                {"rank": 1, "paper_id": "p1", "title": "First Paper"},
                {"rank": 2, "paper_id": "p1", "title": "Second Paper"},
            ]
        }
        config = Pass1BatchConfig(
            provider="openrouter",
            model="test/model",
            prompt_version="test",
            text_source="scoring",
            limit=None,
            paper_ids=set(),
            per_paper_char_budget=1000,
            temperature=0,
            max_output_tokens=1000,
            seed=None,
            dry_run=False,
        )

        rows = batch_response_to_rows(
            parsed,
            candidates=candidates,
            config=config,
            prompt_path=Path("prompt.json"),
            raw_response_path=Path("response.json"),
            parsed_response_path=Path("parsed.json"),
            usage={},
        )

        self.assertEqual({row["paper_id"] for row in rows}, {"p1", "p2"})
        repaired = next(row for row in rows if row["paper_id"] == "p2")
        self.assertEqual(
            repaired["scores"]["ranking_signals"]["paper_id_original_model_output"],
            "p1",
        )

    def test_batch_prompt_excludes_human_outcome_metadata(self) -> None:
        record = PaperRecord(
            paper_id="p1",
            source="SENTINEL_HUMAN_SOURCE",
            title="Paper",
            decision_label="SENTINEL_HUMAN_DECISION",
        )
        candidate = BatchCandidate(record, "paper.txt", "scoring", "Paper content.")
        config = Pass1BatchConfig(
            provider="openrouter",
            model="test",
            prompt_version="test",
            text_source="scoring",
            limit=None,
            paper_ids=set(),
            per_paper_char_budget=1000,
            temperature=0,
            max_output_tokens=1000,
            seed=None,
            dry_run=True,
        )

        messages = build_pass1_batch_messages([candidate], config=config)
        prompt_text = "\n".join(message["content"] for message in messages)

        self.assertNotIn("SENTINEL_HUMAN_SOURCE", prompt_text)
        self.assertNotIn("SENTINEL_HUMAN_DECISION", prompt_text)
        self.assertIn("No reviews, reviewer scores", prompt_text)

    def test_suite_resumes_only_matching_valid_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for index in range(4):
                text_path = root / f"p{index}.txt"
                text_path.write_text(f"Paper {index}", encoding="utf-8")
                records.append(
                    PaperRecord(
                        paper_id=f"p{index}",
                        source="accepted",
                        title=f"Paper {index}",
                        text_scoring=str(text_path),
                        parse_status="ok",
                    ).to_dict()
                )
            manifest = root / "manifest.jsonl"
            out = root / "scores.jsonl"
            run_dir = root / "run"
            write_jsonl(manifest, records)
            config = Pass1BatchSuiteConfig(
                provider="openrouter",
                model="test/model",
                reasoning_effort="none",
                prompt_version="test",
                text_source="scoring",
                limit=None,
                paper_ids=set(),
                per_paper_char_budget=1000,
                batch_size=2,
                partitions=2,
                strategy="shuffled",
                class_path=None,
                request_delay_seconds=0,
                temperature=0,
                max_output_tokens=1000,
                seed=3,
                dry_run=False,
            )
            client = FakeClient()

            first = run_pass1_batch_suite(
                manifest=manifest,
                out=out,
                run_dir=run_dir,
                config=config,
                client=client,
            )
            failed_batch = run_dir / "batches" / "partition_00_batch_00" / "batch.json"
            failed_state = json.loads(failed_batch.read_text(encoding="utf-8"))
            failed_state["status"] = "failed"
            failed_batch.write_text(json.dumps(failed_state), encoding="utf-8")
            (failed_batch.parent / "parsed.json").unlink()
            second = run_pass1_batch_suite(
                manifest=manifest,
                out=out,
                run_dir=run_dir,
                config=config,
                client=client,
            )

            self.assertEqual(first["status"], "ok", first)
            self.assertEqual(first["rows"], 8)
            self.assertEqual(first["usage"]["cost"], 0.08)
            self.assertEqual(client.calls, 4)
            self.assertEqual(second["resumed_batch_count"], 4)
            recovered = json.loads(failed_batch.read_text(encoding="utf-8"))
            self.assertTrue(recovered["recovered_from_raw_response"])
            self.assertEqual(len(out.read_text(encoding="utf-8").splitlines()), 8)

    def test_suite_archives_failed_response_before_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text_path = root / "paper.txt"
            text_path.write_text("Paper text", encoding="utf-8")
            manifest = root / "manifest.jsonl"
            write_jsonl(
                manifest,
                [
                    PaperRecord(
                        paper_id="p0",
                        source="accepted",
                        title="Paper 0",
                        text_scoring=str(text_path),
                        parse_status="ok",
                    ).to_dict()
                ],
            )
            config = Pass1BatchSuiteConfig(
                provider="openrouter",
                model="test/model",
                reasoning_effort="none",
                prompt_version="test",
                text_source="scoring",
                limit=None,
                paper_ids=set(),
                per_paper_char_budget=1000,
                batch_size=1,
                partitions=1,
                strategy="sequential",
                class_path=None,
                request_delay_seconds=0,
                temperature=0,
                max_output_tokens=1000,
                seed=3,
                dry_run=False,
            )
            client = MalformedThenValidClient()
            run_dir = root / "run"

            first = run_pass1_batch_suite(
                manifest=manifest,
                out=root / "scores.jsonl",
                run_dir=run_dir,
                config=config,
                client=client,
            )
            second = run_pass1_batch_suite(
                manifest=manifest,
                out=root / "scores.jsonl",
                run_dir=run_dir,
                config=config,
                client=client,
            )

            batch_dir = run_dir / "batches" / "partition_00_batch_00"
            archived = batch_dir / "attempts" / "attempt_0001"
            current = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
            self.assertEqual(first["status"], "partial_failed")
            self.assertEqual(second["status"], "ok")
            self.assertEqual(client.calls, 2)
            self.assertTrue((archived / "response.json").exists())
            self.assertEqual(current["attempt_index"], 2)


if __name__ == "__main__":
    unittest.main()
