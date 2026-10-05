import unittest

from icml_ai_ac.scraper.icml_virtual import ICMLVirtualScraper, PosterStub, records_from_virtual_json


class ICMLVirtualScraperTests(unittest.TestCase):
    def test_parse_index_dedupes_poster_links(self) -> None:
        html = """
        <html><body>
          <a href="/virtual/2026/poster/123">First Paper</a>
          <a href="/virtual/2026/poster/123">First Paper Duplicate</a>
          <a href="https://icml.cc/virtual/2026/poster/456">Second Paper</a>
          <a href="/accounts/login?nextp=/virtual/2026/poster/789">Login</a>
          <a href="/virtual/2025/poster/999">Wrong Year</a>
        </body></html>
        """
        scraper = ICMLVirtualScraper(year=2026, index_url="https://icml.cc/virtual/2026/papers.html")

        stubs = list(scraper.parse_index(html))

        self.assertEqual([stub.paper_id for stub in stubs], ["123", "456"])
        self.assertEqual(stubs[0].title, "First Paper")
        self.assertEqual(stubs[0].detail_url, "https://icml.cc/virtual/2026/poster/123")

    def test_parse_detail_extracts_json_ld_authors_and_links(self) -> None:
        html = """
        <html>
          <head>
            <title>ICML Poster Ignored Fallback Title</title>
            <script type="application/ld+json">
            {
              "@context": "https://schema.org/",
              "@type": "CreativeWork",
              "name": "A Strong Paper",
              "author": [{"@type": "Person", "name": "Ada Lovelace"}, {"name": "Grace Hopper"}]
            }
            </script>
          </head>
          <body>
            <div id="abstractText"><p>This is the abstract.</p></div>
            <a href="https://openreview.net/forum?id=abc123">OpenReview</a>
            <a href="https://openreview.net/pdf?id=abc123">PDF</a>
          </body>
        </html>
        """
        scraper = ICMLVirtualScraper(year=2026)
        stub = PosterStub(paper_id="63334", title="Index Title", detail_url="https://icml.cc/virtual/2026/poster/63334")

        record = scraper.parse_detail(html, stub=stub, final_url=stub.detail_url)

        self.assertEqual(record.paper_id, "63334")
        self.assertEqual(record.source, "accepted")
        self.assertEqual(record.decision_label, "accepted")
        self.assertEqual(record.title, "A Strong Paper")
        self.assertEqual(record.abstract, "This is the abstract.")
        self.assertEqual(record.authors, ["Ada Lovelace", "Grace Hopper"])
        self.assertEqual(record.forum_url, "https://openreview.net/forum?id=abc123")
        self.assertEqual(record.pdf_url, "https://openreview.net/pdf?id=abc123")
        self.assertEqual(record.extra["pdf_discovery_status"], "found")

    def test_records_from_virtual_json_uses_poster_metadata_and_abstracts(self) -> None:
        payload = {
            "count": 2,
            "results": [
                {
                    "id": 63468,
                    "name": "A Paper",
                    "authors": [{"fullname": "Ada Lovelace"}],
                    "decision": "Accept (regular)",
                    "eventtype": "Poster",
                    "event_type": "Poster",
                    "topic": "Theory->Optimization",
                    "virtualsite_url": "/virtual/2026/poster/63468",
                    "paper_url": "https://openreview.net/forum?id=Wtjwjn1uak",
                    "paper_pdf_url": None,
                    "sourceid": 18357,
                    "sourceurl": "https://openreview.net/group?id=ICML.cc/2026/Conference",
                    "related_events_ids": [70000],
                },
                {
                    "id": 70000,
                    "name": "A Paper Oral",
                    "authors": [{"fullname": "Ada Lovelace"}],
                    "decision": "Accept (regular)",
                    "eventtype": "Oral",
                    "virtualsite_url": "/virtual/2026/oral/70000",
                    "paper_url": "",
                },
            ],
        }

        records, report = records_from_virtual_json(
            payload,
            abstracts={"63468": "This is the abstract."},
            year=2026,
            event_types={"Poster"},
            require_decision=False,
            limit=None,
        )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.paper_id, "63468")
        self.assertEqual(record.title, "A Paper")
        self.assertEqual(record.abstract, "This is the abstract.")
        self.assertEqual(record.decision_label, "Accept (regular)")
        self.assertEqual(record.authors, ["Ada Lovelace"])
        self.assertEqual(record.forum_url, "https://openreview.net/forum?id=Wtjwjn1uak")
        self.assertIsNone(record.pdf_url)
        self.assertEqual(record.topic_cluster, "Theory->Optimization")
        self.assertEqual(record.extra["openreview_forum_id"], "Wtjwjn1uak")
        self.assertEqual(record.extra["pdf_discovery_status"], "missing_on_icml_json")
        self.assertTrue(record.extra["is_oral"])
        self.assertEqual(record.extra["presentation_type"], "oral")
        self.assertEqual(record.extra["oral_event_id"], "70000")
        self.assertEqual(report["selected_records"], 1)
        self.assertEqual(report["excluded_by_event_type"], 1)
        self.assertEqual(report["event_type_counts"], {"Poster": 1})
        self.assertEqual(report["decision_counts"], {"Accept (regular)": 1})
        self.assertEqual(report["raw_event_type_counts"], {"Oral": 1, "Poster": 1})
        self.assertEqual(report["oral_presentation_records"], 1)

    def test_records_from_virtual_json_finds_eventmedia_pdf_fallback(self) -> None:
        payload = {
            "results": [
                {
                    "id": "p1",
                    "name": "Fallback PDF",
                    "eventtype": "Poster",
                    "virtualsite_url": "/virtual/2026/poster/p1",
                    "eventmedia": [
                        {"name": "OpenReview", "uri": "https://openreview.net/forum?id=abc"},
                        {"name": "Paper PDF", "uri": "https://openreview.net/pdf?id=abc"},
                    ],
                }
            ]
        }

        records, report = records_from_virtual_json(
            payload,
            abstracts={},
            year=2026,
            event_types={"Poster"},
            require_decision=False,
            limit=None,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].forum_url, "https://openreview.net/forum?id=abc")
        self.assertEqual(records[0].pdf_url, "https://openreview.net/pdf?id=abc")
        self.assertEqual(records[0].extra["pdf_discovery_status"], "found_icml_json")
        self.assertEqual(report["records_with_pdf_url"], 1)
