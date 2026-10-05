import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from icml_ai_ac.models import PaperRecord
from icml_ai_ac.scoring.classification import (
    ContributionClassificationConfig,
    build_classification_messages,
    classification_response_to_rows,
    recover_complete_classification_prefix,
    run_contribution_classification,
)
from icml_ai_ac.storage import write_jsonl


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, messages, temperature, max_output_tokens, seed=None, response_format=None):
        self.calls += 1
        paper_ids = re.findall(r'<PAPER paper_id="([^"]+)">', messages[-1]["content"])
        content = json.dumps(
            {
                "classifications": [
                    {
                        "paper_id": paper_id,
                        "primary_contribution_class": "theory",
                        "secondary_contribution_classes": [],
                        "confidence": 4,
                        "rationale": "test",
                    }
                    for paper_id in paper_ids
                ]
            }
        )
        return SimpleNamespace(
            response={
                "model": "served",
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "cost": 0.01},
            },
            request={"reasoning": {"effort": "none", "exclude": True}},
            content=content,
            usage={"prompt_tokens": 10, "cost": 0.01},
            served_model="served",
            elapsed_seconds=0.01,
        )


def classification_config() -> ContributionClassificationConfig:
    return ContributionClassificationConfig(
        provider="openrouter",
        model="test/model",
        prompt_version="test",
        text_source="compact",
        limit=None,
        paper_ids=set(),
        per_paper_char_budget=1000,
        batch_size=2,
        request_delay_seconds=0,
        temperature=0,
        max_output_tokens=1000,
        seed=7,
        dry_run=False,
    )


class ContributionClassificationTests(unittest.TestCase):
    def test_classification_prompt_has_no_human_outcome_inputs(self) -> None:
        candidates = [
            {
                "paper_id": "p1",
                "title": "Paper",
                "resolved_text_source": "compact",
                "text": "Paper content.",
                "decision_label": "SENTINEL_HUMAN_DECISION",
                "openreview_scores": "SENTINEL_HUMAN_SCORES",
            }
        ]

        messages = build_classification_messages(candidates, config=classification_config())
        prompt_text = "\n".join(message["content"] for message in messages)

        self.assertNotIn("SENTINEL_HUMAN_DECISION", prompt_text)
        self.assertNotIn("SENTINEL_HUMAN_SCORES", prompt_text)
        self.assertIn("No reviews, reviewer scores", prompt_text)

    def test_failed_parsing_retains_billed_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text_path = root / "p1.txt"
            text_path.write_text("Paper", encoding="utf-8")
            manifest = root / "manifest.jsonl"
            write_jsonl(
                manifest,
                [
                    PaperRecord(
                        paper_id="p1",
                        source="accepted",
                        title="Paper",
                        text_compact=str(text_path),
                        parse_status="ok",
                    ).to_dict()
                ],
            )

            class MalformedClient:
                def complete(self, **kwargs):
                    return SimpleNamespace(
                        response={"model": "served", "usage": {"cost": 0.25}},
                        content="{}",
                        usage={"cost": 0.25},
                        served_model="served",
                        elapsed_seconds=0.01,
                    )

            result = run_contribution_classification(
                manifest=manifest,
                out=root / "out.jsonl",
                run_dir=root / "run",
                config=classification_config(),
                client=MalformedClient(),
            )

            self.assertEqual(result["status"], "partial_failed")
            self.assertEqual(result["usage"]["cost"], 0.25)
            self.assertEqual(result["batch_results"][0]["served_model"], "served")

    def test_batches_and_resumes_valid_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for index in range(5):
                text_path = root / f"p{index}.txt"
                text_path.write_text(f"Paper {index}", encoding="utf-8")
                records.append(
                    PaperRecord(
                        paper_id=f"p{index}",
                        source="accepted",
                        title=f"Paper {index}",
                        text_compact=str(text_path),
                        parse_status="ok",
                    ).to_dict()
                )
            manifest = root / "manifest.jsonl"
            out = root / "classes.jsonl"
            run_dir = root / "run"
            write_jsonl(manifest, records)
            client = FakeClient()

            first = run_contribution_classification(
                manifest=manifest,
                out=out,
                run_dir=run_dir,
                config=classification_config(),
                client=client,
            )
            second = run_contribution_classification(
                manifest=manifest,
                out=out,
                run_dir=run_dir,
                config=classification_config(),
                client=client,
            )

            self.assertEqual(first["status"], "ok")
            self.assertEqual(first["batch_count"], 3)
            self.assertEqual(first["rows"], 5)
            self.assertEqual(first["usage"]["cost"], 0.03)
            self.assertEqual(client.calls, 3)
            self.assertEqual(second["resumed_batch_count"], 3)
            self.assertEqual(len(out.read_text(encoding="utf-8").splitlines()), 5)

    def test_retry_archives_failed_attempt_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text_path = root / "p1.txt"
            text_path.write_text("Paper", encoding="utf-8")
            manifest = root / "manifest.jsonl"
            write_jsonl(
                manifest,
                [
                    PaperRecord(
                        paper_id="p1",
                        source="accepted",
                        title="Paper",
                        text_compact=str(text_path),
                        parse_status="ok",
                    ).to_dict()
                ],
            )

            class FlakyClient(FakeClient):
                def complete(self, **kwargs):
                    if self.calls == 0:
                        self.calls += 1
                        return SimpleNamespace(
                            response={"model": "served", "usage": {"cost": 0}},
                            request={"reasoning": {"effort": "none", "exclude": True}},
                            content="{}",
                            usage={"cost": 0},
                            served_model="served",
                            elapsed_seconds=0.01,
                        )
                    return super().complete(**kwargs)

            client = FlakyClient()
            run_dir = root / "run"
            first = run_contribution_classification(
                manifest=manifest,
                out=root / "out.jsonl",
                run_dir=run_dir,
                config=classification_config(),
                client=client,
            )
            second = run_contribution_classification(
                manifest=manifest,
                out=root / "out.jsonl",
                run_dir=run_dir,
                config=classification_config(),
                client=client,
            )

            archive_dir = run_dir / "batches" / "batch_0000" / "attempts" / "attempt_0001"
            self.assertEqual(first["status"], "partial_failed")
            self.assertEqual(second["status"], "ok")
            self.assertEqual(second["batch_results"][0]["attempt_index"], 2)
            self.assertEqual(client.calls, 2)
            self.assertTrue((archive_dir / "response.json").exists())
            self.assertTrue((archive_dir / "error.json").exists())
            self.assertFalse((run_dir / "batches" / "batch_0000" / "error.json").exists())
            self.assertEqual(
                json.loads((archive_dir / "batch.json").read_text(encoding="utf-8"))["status"],
                "failed",
            )

    def test_retry_does_not_reuse_stale_response_after_transport_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text_path = root / "p1.txt"
            text_path.write_text("Paper", encoding="utf-8")
            manifest = root / "manifest.jsonl"
            write_jsonl(
                manifest,
                [
                    PaperRecord(
                        paper_id="p1",
                        source="accepted",
                        title="Paper",
                        text_compact=str(text_path),
                        parse_status="ok",
                    ).to_dict()
                ],
            )

            class ParseThenTransportFailure:
                def __init__(self) -> None:
                    self.calls = 0

                def complete(self, **kwargs):
                    self.calls += 1
                    if self.calls == 1:
                        return SimpleNamespace(
                            response={"model": "served", "usage": {"cost": 0.1}},
                            request={"reasoning": {"effort": "none", "exclude": True}},
                            content="{}",
                            usage={"cost": 0.1},
                            served_model="served",
                            elapsed_seconds=0.01,
                        )
                    raise OSError("transport failed before response")

            client = ParseThenTransportFailure()
            run_dir = root / "run"
            run_contribution_classification(
                manifest=manifest,
                out=root / "out.jsonl",
                run_dir=run_dir,
                config=classification_config(),
                client=client,
            )
            second = run_contribution_classification(
                manifest=manifest,
                out=root / "out.jsonl",
                run_dir=run_dir,
                config=classification_config(),
                client=client,
            )

            batch_dir = run_dir / "batches" / "batch_0000"
            self.assertEqual(second["status"], "partial_failed")
            self.assertNotIn("raw_response_path", second["batch_results"][0])
            self.assertFalse((batch_dir / "response.json").exists())
            self.assertTrue(
                (batch_dir / "attempts" / "attempt_0001" / "response.json").exists()
            )

    def test_recovers_complete_expected_prefix_before_truncated_duplicate(self) -> None:
        content = """{
  "classifications": [
    {"paper_id": "p1", "primary_contribution_class": "theory"},
    {"paper_id": "p2", "primary_contribution_class": "benchmark_dataset"},
    {"paper_id": "p2", "title": "truncated
"""

        parsed = recover_complete_classification_prefix(content, expected_count=2)

        self.assertIsNotNone(parsed)
        self.assertEqual(
            [row["paper_id"] for row in parsed["classifications"]],
            ["p1", "p2"],
        )
        self.assertEqual(
            parsed["_response_recovery"]["kind"],
            "complete_expected_classification_prefix",
        )

    def test_does_not_recover_incomplete_expected_prefix(self) -> None:
        content = """{
  "classifications": [
    {"paper_id": "p1", "primary_contribution_class": "theory"},
    {"paper_id": "p2", "title": "truncated
"""

        self.assertIsNone(
            recover_complete_classification_prefix(content, expected_count=2)
        )

    def test_rejects_duplicate_and_missing_ids(self) -> None:
        candidates = [
            {"paper_id": "p1", "title": "One", "resolved_text_source": "compact", "text_path": "one"},
            {"paper_id": "p2", "title": "Two", "resolved_text_source": "compact", "text_path": "two"},
        ]
        parsed = {
            "classifications": [
                {"paper_id": "p1", "primary_contribution_class": "theory"},
                {"paper_id": "p1", "primary_contribution_class": "theory"},
            ]
        }
        with self.assertRaisesRegex(ValueError, "coverage error"):
            classification_response_to_rows(
                parsed,
                candidates=candidates,
                config=classification_config(),
                prompt_path=Path("prompt"),
                raw_response_path=Path("response"),
                parsed_response_path=Path("parsed"),
                usage={},
            )

    def test_repairs_one_near_match_paper_id(self) -> None:
        candidates = [
            {"paper_id": "aqZKgwf7Cc", "title": "One", "resolved_text_source": "compact", "text_path": "one"},
            {"paper_id": "p2", "title": "Two", "resolved_text_source": "compact", "text_path": "two"},
        ]
        parsed = {
            "classifications": [
                {"paper_id": "aqZKgwf7Sp", "primary_contribution_class": "theory"},
                {"paper_id": "p2", "primary_contribution_class": "theory"},
            ]
        }

        rows = classification_response_to_rows(
            parsed,
            candidates=candidates,
            config=classification_config(),
            prompt_path=Path("prompt"),
            raw_response_path=Path("response"),
            parsed_response_path=Path("parsed"),
            usage={},
        )

        self.assertEqual({row["paper_id"] for row in rows}, {"aqZKgwf7Cc", "p2"})

    def test_repairs_duplicate_id_from_unique_expected_title(self) -> None:
        candidates = [
            {"paper_id": "p1", "title": "One", "resolved_text_source": "compact", "text_path": "one"},
            {"paper_id": "p2", "title": "Two", "resolved_text_source": "compact", "text_path": "two"},
        ]
        parsed = {
            "classifications": [
                {"paper_id": "p1", "title": "One", "primary_contribution_class": "theory"},
                {"paper_id": "p1", "title": "Two", "primary_contribution_class": "benchmark_dataset"},
            ]
        }

        rows = classification_response_to_rows(
            parsed,
            candidates=candidates,
            config=classification_config(),
            prompt_path=Path("prompt"),
            raw_response_path=Path("response"),
            parsed_response_path=Path("parsed"),
            usage={},
        )

        self.assertEqual(
            [(row["paper_id"], row["primary_contribution_class"]) for row in rows],
            [("p1", "theory"), ("p2", "benchmark_dataset")],
        )
        repaired = parsed["classifications"][1]
        self.assertEqual(repaired["paper_id_original_model_output"], "p1")
        self.assertEqual(
            repaired["paper_id_repair_reason"],
            "unique_expected_title_match",
        )


if __name__ == "__main__":
    unittest.main()
