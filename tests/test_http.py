import json
import io
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from icml_ai_ac.http import AccessChallengeError, FetchResult, HttpClient, is_pdf_file


class HttpClientTests(unittest.TestCase):
    def test_post_json_sends_payload_and_bearer_token(self) -> None:
        class FakeResponse:
            status = 200
            headers = {"content-type": "application/json"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self) -> bytes:
                return b'{"ok":true}'

            def geturl(self) -> str:
                return "https://example.test/login"

        client = HttpClient(retries=0)
        client.set_bearer_token("test-token")

        with patch("urllib.request.urlopen", return_value=FakeResponse()) as urlopen:
            result = client.post_json("https://example.test/login", {"id": "user", "password": "pass"})

        request = urlopen.call_args.args[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-token")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(json.loads(request.data), {"id": "user", "password": "pass"})
        self.assertEqual(result.text, '{"ok":true}')

    def test_download_rejects_non_pdf_success_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "paper.pdf"
            client = HttpClient()

            def fake_fetch(url: str, *, accept: str | None = None) -> FetchResult:
                return FetchResult(
                    url=url,
                    status=200,
                    content_type="text/html",
                    body=b"<html>not a pdf</html>",
                    final_url=url,
                )

            setattr(client, "fetch", fake_fetch)

            with self.assertRaisesRegex(ValueError, "not a PDF"):
                client.download("https://example.test/paper.pdf", out)

            self.assertFalse(out.exists())

    def test_download_accepts_pdf_magic_without_content_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "paper.pdf"
            client = HttpClient()

            def fake_fetch(url: str, *, accept: str | None = None) -> FetchResult:
                return FetchResult(
                    url=url,
                    status=200,
                    content_type=None,
                    body=b"%PDF-1.7\nbody",
                    final_url=url,
                )

            setattr(client, "fetch", fake_fetch)

            result = client.download("https://example.test/paper.pdf", out)

            self.assertFalse(result.skipped)
            self.assertEqual(result.bytes_written, len(b"%PDF-1.7\nbody"))
            self.assertEqual(out.read_bytes(), b"%PDF-1.7\nbody")

    def test_download_replaces_invalid_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "paper.pdf"
            out.write_bytes(b"<html>temporary access error</html>")
            client = HttpClient()

            def fake_fetch(url: str, *, accept: str | None = None) -> FetchResult:
                return FetchResult(
                    url=url,
                    status=200,
                    content_type="application/pdf",
                    body=b"%PDF-1.7\nreplacement",
                    final_url=url,
                )

            setattr(client, "fetch", fake_fetch)
            result = client.download("https://example.test/paper.pdf", out)

            self.assertFalse(result.skipped)
            self.assertTrue(is_pdf_file(out))

    def test_fetch_identifies_interactive_access_challenge(self) -> None:
        url = "https://openreview.net/pdf?id=paper"
        error = urllib.error.HTTPError(
            url,
            403,
            "Forbidden",
            {},
            io.BytesIO(b'{"name":"ChallengeRequiredError","message":"Challenge verification required"}'),
        )
        client = HttpClient(retries=0)

        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(AccessChallengeError):
                client.fetch(url)

    def test_fetch_identifies_openreview_access_gate_html(self) -> None:
        url = "https://openreview.net/pdf?id=paper"
        error = urllib.error.HTTPError(
            url,
            403,
            "Forbidden",
            {},
            io.BytesIO(b"<h1>Error 403</h1><p>Access to this page is restricted.</p> Please check that you are logged in to OpenReview"),
        )
        client = HttpClient(retries=0)

        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(AccessChallengeError):
                client.fetch(url)


if __name__ == "__main__":
    unittest.main()
