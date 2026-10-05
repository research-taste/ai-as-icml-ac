import tempfile
import unittest
import urllib.error
from pathlib import Path

from icml_ai_ac.cli import (
    _existing_openreview_pdf_state,
    _merge_openreview_pdf_state,
    _openreview_halt_reason,
    _openreview_resolution_complete,
)
from icml_ai_ac.models import PaperRecord


class OpenReviewPdfResumeTests(unittest.TestCase):
    def test_auth_and_rate_limit_statuses_halt_the_batch(self) -> None:
        for status in (401, 403, 429):
            error = urllib.error.HTTPError("https://api2.openreview.net/pdf?id=x", status, "error", {}, None)
            self.assertIn("batch halted", _openreview_halt_reason(error) or "")

        server_error = urllib.error.HTTPError("https://api2.openreview.net/pdf?id=x", 500, "error", {}, None)
        self.assertIsNone(_openreview_halt_reason(server_error))

    def test_merge_preserves_openreview_provenance_without_secrets(self) -> None:
        merged = _merge_openreview_pdf_state(
            {"paper_id": "paper-1", "extra": {}},
            {
                "paper_id": "paper-1",
                "extra": {
                    "openreview_authenticated": True,
                    "openreview_forum_id": "forum-1",
                    "source_pdf_kind": "openreview_official",
                    "unrelated_transient_value": "drop-me",
                },
            },
        )

        self.assertTrue(merged["extra"]["openreview_authenticated"])
        self.assertEqual(merged["extra"]["openreview_forum_id"], "forum-1")
        self.assertEqual(merged["extra"]["source_pdf_kind"], "openreview_official")
        self.assertNotIn("unrelated_transient_value", merged["extra"])

    def test_existing_pdf_is_recovered_without_network_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf_dir = Path(tmp)
            path = pdf_dir / "paper-1.pdf"
            path.write_bytes(b"%PDF-1.7\nexisting paper")
            record = PaperRecord(
                paper_id="paper-1",
                source="accepted",
                title="Paper",
                forum_url="https://openreview.net/forum?id=forum-1",
                extra={"openreview_forum_id": "forum-1"},
            )

            row = _existing_openreview_pdf_state(record, previous=None, pdf_dir=pdf_dir)

            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row["pdf_path"], str(path))
            self.assertEqual(row["extra"]["pdf_download_status"], "ok")
            self.assertEqual(row["extra"]["pdf_recovery_source"], "existing_local_file")
            self.assertEqual(row["extra"]["pdf_path_source"], "openreview_official")
            self.assertEqual(row["extra"]["source_pdf_kind"], "openreview_official")
            self.assertTrue(_openreview_resolution_complete(row, download_pdfs=True))

    def test_non_pdf_file_is_not_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf_dir = Path(tmp)
            (pdf_dir / "paper-1.pdf").write_bytes(b"not a pdf")
            record = PaperRecord(
                paper_id="paper-1",
                source="accepted",
                extra={"openreview_forum_id": "forum-1"},
            )

            row = _existing_openreview_pdf_state(record, previous=None, pdf_dir=pdf_dir)

            self.assertIsNone(row)


if __name__ == "__main__":
    unittest.main()
