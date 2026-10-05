import tempfile
import unittest
from pathlib import Path

from icml_ai_ac.metadata import enrich_metadata_rows, normalize_title, summarize_openreview_scores
from icml_ai_ac.storage import write_json, write_jsonl


class MetadataTests(unittest.TestCase):
    def test_summarize_openreview_scores_preserves_review_fields(self) -> None:
        note = {
            "id": "forum-1",
            "forum": "forum-1",
            "content": {
                "title": {"value": "A Paper"},
                "venue": {"value": "ICML 2026 spotlight"},
                "venueid": {"value": "ICML.cc/2026/Conference"},
            },
            "details": {
                "directReplies": [
                    {
                        "id": "review-1",
                        "invitations": ["ICML.cc/2026/Conference/-/Official_Review"],
                        "tmdate": 100,
                        "content": {
                            "overall_assessment": {"value": "6: Strong accept"},
                            "soundness": {"value": "4: excellent"},
                            "confidence": {"value": "5: very confident"},
                            "presentation": {"value": 3},
                        },
                    },
                    {
                        "id": "review-2",
                        "invitation": "ICML.cc/2026/Conference/-/Official_Review",
                        "content": {
                            "overall_assessment": {"value": "5: Weak accept"},
                            "soundness": {"value": 3},
                            "confidence": {"value": 4},
                        },
                    },
                    {
                        "id": "comment-1",
                        "invitation": "ICML.cc/2026/Conference/-/Official_Comment",
                        "content": {"comment": {"value": "No score."}},
                    },
                ]
            },
        }

        summary = summarize_openreview_scores(note)

        self.assertEqual(summary["review_count"], 2)
        self.assertEqual(summary["reply_count"], 3)
        self.assertEqual(summary["overall_values"], [6.0, 5.0])
        self.assertEqual(summary["overall_mean"], 5.5)
        self.assertEqual(summary["soundness_mean"], 3.5)
        self.assertEqual(summary["presentation_values"], [3.0])
        self.assertEqual(summary["reviews"][0]["tmdate"], 100)

    def test_enrich_metadata_matches_scores_by_forum_and_awards_by_normalized_title(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ratings_path = root / "ratings.jsonl"
            awards_path = root / "awards.json"
            write_jsonl(
                ratings_path,
                [
                    {
                        "openreview_forum_id": "forum-1",
                        "title": "High-accuracy sampling",
                        "overall_values": [6.0, 5.0],
                        "overall_mean": 5.5,
                    }
                ],
            )
            write_json(
                awards_path,
                {
                    "awards": [
                        {
                            "title": "High-Accuracy Sampling",
                            "award_type": "outstanding_paper",
                            "track": "main",
                        }
                    ]
                },
            )
            rows = [
                {
                    "paper_id": "paper-1",
                    "title": "High-accuracy sampling",
                    "forum_url": "https://openreview.net/forum?id=forum-1",
                    "extra": {"openreview_forum_id": "forum-1"},
                }
            ]

            enriched, report = enrich_metadata_rows(
                rows,
                ratings_path=ratings_path,
                awards_path=awards_path,
            )

        extra = enriched[0]["extra"]
        self.assertEqual(extra["openreview_scores"]["overall_mean"], 5.5)
        self.assertTrue(extra["is_award_paper"])
        self.assertTrue(extra["is_outstanding_paper"])
        self.assertEqual(report["records_with_ratings"], 1)
        self.assertEqual(report["records_with_awards"], 1)
        self.assertEqual(report["unmatched_awards"], [])

    def test_normalize_title_ignores_case_punctuation_and_accents(self) -> None:
        self.assertEqual(normalize_title("High-Accuracy: Sampling"), normalize_title("high accuracy sampling"))
        self.assertEqual(normalize_title("Leal-Taixe"), normalize_title("Leal-Taixe"))


if __name__ == "__main__":
    unittest.main()
