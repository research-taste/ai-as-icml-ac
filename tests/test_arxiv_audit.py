import unittest

from icml_ai_ac.scraper.arxiv_audit import audit_arxiv_resolution


class ArxivAuditTests(unittest.TestCase):
    def test_audit_splits_automatic_review_and_likely_absent_rows(self) -> None:
        manifest_rows = [
            {
                "paper_id": "matched",
                "title": "Matched Paper",
                "extra": {
                    "arxiv_resolution_status": "matched",
                    "arxiv_match_confidence": "exact",
                    "arxiv_id": "2601.00001",
                    "arxiv_best_match": {
                        "arxiv_id": "2601.00001",
                        "title": "Matched Paper",
                        "match_score": 1.0,
                        "match_confidence": "exact",
                        "match_evidence": {
                            "title_similarity": 1.0,
                            "author_overlap": 1.0,
                            "abstract_similarity": 1.0,
                        },
                    },
                },
            },
            {"paper_id": "review", "title": "New Title", "extra": {"arxiv_resolution_status": "unresolved"}},
            {"paper_id": "absent", "title": "Absent Paper", "extra": {"arxiv_resolution_status": "unresolved"}},
        ]
        candidate_rows = [
            {
                "paper_id": "review",
                "arxiv_id": "2601.00002",
                "title": "Older Title",
                "match_score": 0.67,
                "match_confidence": "low",
                "match_evidence": {
                    "title_similarity": 0.58,
                    "author_overlap": 1.0,
                    "abstract_similarity": 0.68,
                },
            }
        ]

        rows, report = audit_arxiv_resolution(manifest_rows, candidate_rows)

        classes = {row["paper_id"]: row["arxiv_audit_class"] for row in rows}
        self.assertEqual(classes["matched"], "automatic_high_confidence")
        self.assertEqual(classes["review"], "review_probable_prior_title")
        self.assertEqual(classes["absent"], "likely_absent_no_candidate")
        self.assertEqual(report["class_counts"]["automatic_high_confidence"], 1)
        self.assertEqual(report["class_counts"]["review_probable_prior_title"], 1)
        self.assertEqual(report["class_counts"]["likely_absent_no_candidate"], 1)


if __name__ == "__main__":
    unittest.main()
