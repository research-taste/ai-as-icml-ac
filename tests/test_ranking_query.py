from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from icml_ai_ac.ranking_query import (
    RankingQuery,
    query_ranked_papers,
    resolve_ranking_category,
    top_ranked_papers,
    write_query_results,
)
from icml_ai_ac.storage import write_json, write_jsonl


class RankingQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.metadata_path = self.root / "metadata.jsonl"
        self.cheap_path = self.root / "cheap.jsonl"
        self.final_path = self.root / "final.json"

        write_jsonl(
            self.metadata_path,
            [
                {
                    "paper_id": "data-topic",
                    "title": "Learning from Structured Records",
                    "abstract": "A method for tabular prediction.",
                    "topic_cluster": "General Machine Learning->Data",
                },
                {
                    "paper_id": "benchmark",
                    "title": "A Reliable Evaluation Suite",
                    "abstract": "We release a broad evaluation resource.",
                    "topic_cluster": "General Machine Learning->Evaluation",
                },
                {
                    "paper_id": "pretraining",
                    "title": "Scaling Better Language Models",
                    "abstract": "We study pre-training data mixtures and model quality.",
                    "topic_cluster": "Deep Learning->Large Language Models",
                },
                {
                    "paper_id": "unrelated",
                    "title": "Convex Optimization Guarantees",
                    "abstract": "A convergence proof.",
                    "topic_cluster": "Optimization->Convex",
                },
            ],
        )
        write_jsonl(
            self.cheap_path,
            [
                cheap_row("data-topic", 40, "application_method"),
                cheap_row("benchmark", 80, "benchmark_dataset"),
                cheap_row("pretraining", 20, "core_ml_algorithm"),
                cheap_row("unrelated", 1, "theory"),
            ],
        )
        write_json(
            self.final_path,
            {
                "ranked_papers": [
                    final_row("pretraining", 2, "playoff_all_pairs"),
                    final_row("data-topic", 70, "swiss_only"),
                    final_row("benchmark", 200, None),
                ]
            },
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_data_pretraining_preset_is_transparent_and_final_ranked(self) -> None:
        rows = self.query(RankingQuery(preset="data-pretraining"))

        self.assertEqual([row["paper_id"] for row in rows], ["pretraining", "data-topic", "benchmark"])
        self.assertEqual([row["final_rank"] for row in rows], [2, 70, 200])
        self.assertEqual(rows[0]["final_stage"], "playoff_all_pairs")
        self.assertIn("content:pretraining", rows[0]["match_reasons"])
        self.assertEqual(rows[1]["match_reasons"], ["official_topic:data"])
        self.assertEqual(rows[2]["match_reasons"], ["contribution:benchmark_dataset"])
        self.assertEqual(rows[2]["final_stage"], "frontier_finalist_outside_tournament")

    def test_filters_combine_and_scope_excludes_non_playoff_papers(self) -> None:
        rows = self.query(
            RankingQuery(
                preset="pretraining",
                query="language models",
                topic="Deep Learning->*",
                contribution_class="core_ml_algorithm",
                scope="playoff",
            )
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["paper_id"], "pretraining")
        self.assertEqual(
            rows[0]["match_reasons"],
            [
                "content:pretraining",
                "query:language models",
                "topic:Deep Learning->Large Language Models",
                "contribution:core_ml_algorithm",
            ],
        )

    def test_nonfinalist_has_cheap_rank_but_no_final_rank(self) -> None:
        rows = self.query(RankingQuery(query="convex"))

        self.assertEqual(rows[0]["paper_id"], "unrelated")
        self.assertEqual(rows[0]["cheap_rank"], 1)
        self.assertIsNone(rows[0]["final_rank"])
        self.assertIsNone(rows[0]["final_stage"])
        self.assertEqual(self.query(RankingQuery(query="convex", scope="finalists")), [])

    def test_csv_output_serializes_match_reasons(self) -> None:
        rows = self.query(RankingQuery(preset="pretraining"))
        output = self.root / "results.csv"
        write_query_results(rows, path=output, output_format="csv")

        with output.open(newline="", encoding="utf-8") as handle:
            parsed = list(csv.DictReader(handle))
        self.assertEqual(parsed[0]["paper_id"], "pretraining")
        self.assertEqual(parsed[0]["match_reasons"], "content:pretraining")

    def test_top_ranked_api_accepts_human_category_name(self) -> None:
        rows = top_ranked_papers(
            "Data and Pretraining",
            top_n=2,
            metadata_path=self.metadata_path,
            cheap_ranking_path=self.cheap_path,
            final_ranking_path=self.final_path,
        )

        self.assertEqual([row["paper_id"] for row in rows], ["pretraining", "data-topic"])
        self.assertTrue(all(row["final_rank"] is not None for row in rows))

    def test_category_resolution_covers_contributions_and_topic_families(self) -> None:
        self.assertEqual(
            resolve_ranking_category("datasets").value,
            "benchmark_dataset",
        )
        topic = resolve_ranking_category("Deep Learning")
        self.assertEqual((topic.kind, topic.value), ("topic", "Deep Learning->*"))
        explicit = resolve_ranking_category("topic:Theory")
        self.assertEqual((explicit.kind, explicit.value), ("topic", "Theory->*"))

    def test_jsonl_stdout_is_machine_readable(self) -> None:
        rows = self.query(RankingQuery(query="convex"))
        output = io.StringIO()
        with redirect_stdout(output):
            write_query_results(rows, path=None, output_format="jsonl")

        parsed = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(parsed[0]["paper_id"], "unrelated")

    def query(self, query: RankingQuery) -> list[dict]:
        return query_ranked_papers(
            metadata_path=self.metadata_path,
            cheap_ranking_path=self.cheap_path,
            final_ranking_path=self.final_path,
            query=query,
        )


def cheap_row(paper_id: str, rank: int, contribution_class: str) -> dict:
    return {
        "paper_id": paper_id,
        "aggregate_rank": rank,
        "primary_contribution_class": contribution_class,
    }


def final_row(paper_id: str, rank: int, stage: str | None) -> dict:
    row = {"paper_id": paper_id, "rank": rank, "ranking_source": "test"}
    if stage is not None:
        row["tournament_stats"] = {"ranking_source_stage": stage}
    return row


if __name__ == "__main__":
    unittest.main()
