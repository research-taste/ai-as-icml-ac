import unittest
from unittest.mock import patch

from icml_ai_ac.http import FetchResult
from icml_ai_ac.scraper.openreview import (
    OpenReviewAuthenticationError,
    build_query,
    OpenReviewClient,
    OpenReviewMfaRequiredError,
    forum_id_from_url,
    note_to_record,
    pdf_url_for_forum_id,
    queries_for_status,
    query_for_status,
    venue_status_report,
)


class FakeAuthHttp:
    def __init__(self, response_body: bytes) -> None:
        self.response_body = response_body
        self.login_url = None
        self.login_payload = None
        self.bearer_token = None

    def post_json(self, url: str, payload: dict[str, object]) -> FetchResult:
        self.login_url = url
        self.login_payload = payload
        return FetchResult(
            url=url,
            status=200,
            content_type="application/json",
            body=self.response_body,
            final_url=url,
        )

    def set_bearer_token(self, token: str) -> None:
        self.bearer_token = token


class OpenReviewTests(unittest.TestCase):
    def test_authenticate_uses_api2_login_and_installs_bearer_token(self) -> None:
        http = FakeAuthHttp(b'{"token":"session-token","user":{"id":"user@example.com"}}')
        client = OpenReviewClient(http_client=http)

        client.authenticate(username="user@example.com", password="secret", token_expires_seconds=600)

        self.assertTrue(client.authenticated)
        self.assertEqual(http.login_url, "https://api2.openreview.net/login")
        self.assertEqual(
            http.login_payload,
            {"id": "user@example.com", "password": "secret", "expiresIn": 600},
        )
        self.assertEqual(http.bearer_token, "session-token")
        self.assertEqual(
            client.pdf_url_for_forum_id("forum/id"),
            "https://api2.openreview.net/pdf?id=forum/id",
        )

    def test_authenticate_from_env_requires_both_credentials(self) -> None:
        client = OpenReviewClient(http_client=FakeAuthHttp(b"{}"))
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(OpenReviewAuthenticationError, "must both be set"):
                client.authenticate_from_env()

    def test_authenticate_stops_when_mfa_is_required(self) -> None:
        http = FakeAuthHttp(b'{"mfaPending":true,"mfaMethods":["totp"]}')
        client = OpenReviewClient(http_client=http)

        with self.assertRaisesRegex(OpenReviewMfaRequiredError, "totp"):
            client.authenticate(username="user", password="secret")

        self.assertFalse(client.authenticated)
        self.assertIsNone(http.bearer_token)

    def test_build_query_skips_unset_values(self) -> None:
        self.assertEqual(
            build_query(venue_id="ICML.cc/2026/Conference", invitation=None, content_venue=None, details="replyCount"),
            {
                "content.venueid": "ICML.cc/2026/Conference",
                "details": "replyCount",
            },
        )

    def test_forum_pdf_helpers_use_openreview_id_query(self) -> None:
        self.assertEqual(forum_id_from_url("https://openreview.net/forum?id=WtgQOtmw9N"), "WtgQOtmw9N")
        self.assertEqual(forum_id_from_url("https://openreview.net/pdf?id=WtgQOtmw9N"), "WtgQOtmw9N")
        self.assertEqual(forum_id_from_url("https://openreview.net/pdf/WtgQOtmw9N.pdf"), "WtgQOtmw9N")
        self.assertEqual(pdf_url_for_forum_id("WtgQOtmw9N"), "https://openreview.net/pdf?id=WtgQOtmw9N")

    def test_note_to_record_supports_api2_content_shape(self) -> None:
        note = {
            "id": "note-id",
            "forum": "forum-id",
            "invitation": "ICML.cc/2026/Conference/-/Submission",
            "content": {
                "title": {"value": "Paper Title"},
                "abstract": {"value": "Paper abstract"},
                "authors": {"value": ["Author One", "Author Two"]},
                "pdf": {"value": "/pdf/forum-id.pdf"},
                "venue": {"value": "Submitted to ICML 2026"},
                "venueid": {"value": "ICML.cc/2026/Conference"},
            },
        }

        record = note_to_record(note, source="openreview_public")

        self.assertEqual(record.paper_id, "forum-id")
        self.assertEqual(record.title, "Paper Title")
        self.assertEqual(record.abstract, "Paper abstract")
        self.assertEqual(record.authors, ["Author One", "Author Two"])
        self.assertEqual(record.forum_url, "https://openreview.net/forum?id=forum-id")
        self.assertEqual(record.pdf_url, "https://openreview.net/pdf/forum-id.pdf")
        self.assertEqual(record.extra["openreview_invitation"], "ICML.cc/2026/Conference/-/Submission")

    def test_note_to_record_infers_pdf_url_from_forum_id(self) -> None:
        note = {
            "id": "note-id",
            "forum": "forum-id",
            "content": {
                "title": {"value": "Paper Title"},
                "venue": {"value": "ICML 2026 regular"},
            },
        }

        record = note_to_record(note, source="openreview_public")

        self.assertEqual(record.forum_url, "https://openreview.net/forum?id=forum-id")
        self.assertEqual(record.pdf_url, "https://openreview.net/pdf?id=forum-id")

    def test_venue_status_report_uses_group_content(self) -> None:
        group = {
            "id": "ICML.cc/2026/Conference",
            "domain": "ICML.cc/2026/Conference",
            "content": {
                "submission_id": {"value": "ICML.cc/2026/Conference/-/Submission"},
                "submission_venue_id": {"value": "ICML.cc/2026/Conference/Submission"},
                "rejected_venue_id": {"value": "ICML.cc/2026/Conference/Rejected_Submission"},
                "public_submissions": {"value": False},
                "public_withdrawn_submissions": {"value": False},
                "public_desk_rejected_submissions": {"value": False},
                "decision_heading_map": {
                    "value": {
                        "ICML 2026 spotlight": "Accept (spotlight)",
                        "ICML 2026 regular": "Accept (regular)",
                        "Submitted to ICML 2026": "Reject",
                    }
                },
            },
        }

        report = venue_status_report(group)

        self.assertEqual(report["api_version"], "api2")
        self.assertFalse(report["public_submissions"])
        self.assertEqual(query_for_status(group, "rejected"), {"content.venueid": "ICML.cc/2026/Conference/Rejected_Submission"})
        self.assertEqual(
            report["statuses"]["accepted"]["fallback_queries"],
            [
                {"invitation": "ICML.cc/2026/Conference/-/Submission", "content.venue": "ICML 2026 spotlight"},
                {"invitation": "ICML.cc/2026/Conference/-/Submission", "content.venue": "ICML 2026 regular"},
            ],
        )
        self.assertEqual(
            queries_for_status(group, "accepted"),
            [
                {"content.venueid": "ICML.cc/2026/Conference"},
                {"content.venue": "ICML 2026 spotlight", "invitation": "ICML.cc/2026/Conference/-/Submission"},
                {"content.venue": "ICML 2026 regular", "invitation": "ICML.cc/2026/Conference/-/Submission"},
                {"content.venue": "ICML 2026 spotlight"},
                {"content.venue": "ICML 2026 regular"},
            ],
        )
