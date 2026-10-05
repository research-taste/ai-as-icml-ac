import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from icml_ai_ac.cli import cmd_download_arxiv_pdfs


class _FakeHttpClient:
    def download(self, url: str, path: Path, *, overwrite: bool):
        return SimpleNamespace(path=path, sha256="abc123", bytes_written=42, skipped=False)


class ArxivDownloadTests(unittest.TestCase):
    def test_downloaded_only_output_excludes_ineligible_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            out = root / "out.jsonl"
            downloaded_only = root / "downloaded.jsonl"
            rows = [
                {
                    "paper_id": "matched",
                    "title": "Matched paper",
                    "source": "test",
                    "extra": {
                        "arxiv_id": "2607.00001",
                        "arxiv_pdf_url": "https://arxiv.org/pdf/2607.00001",
                        "arxiv_match_confidence": "exact",
                    },
                },
                {"paper_id": "unresolved", "title": "Unresolved paper", "source": "test", "extra": {}},
            ]
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            args = SimpleNamespace(
                manifest=manifest,
                out=out,
                downloaded_only_out=downloaded_only,
                pdf_dir=root / "pdfs",
                limit=None,
                min_confidence="high",
                overwrite=False,
                no_set_pdf_path=False,
            )

            with patch("icml_ai_ac.cli.make_http", return_value=_FakeHttpClient()):
                self.assertEqual(cmd_download_arxiv_pdfs(args), 0)

            full_rows = [json.loads(line) for line in out.read_text().splitlines()]
            downloaded_rows = [json.loads(line) for line in downloaded_only.read_text().splitlines()]
            self.assertEqual(len(full_rows), 2)
            self.assertEqual([row["paper_id"] for row in downloaded_rows], ["matched"])
            self.assertEqual(downloaded_rows[0]["extra"]["source_pdf_kind"], "arxiv_preprint")


if __name__ == "__main__":
    unittest.main()
