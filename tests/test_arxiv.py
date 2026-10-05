import tempfile
import unittest
from pathlib import Path

from icml_ai_ac.http import FetchResult, HttpClient
from icml_ai_ac.models import PaperRecord
from icml_ai_ac.scraper.arxiv import (
    ArxivEntry,
    ArxivClient,
    ArxivQueryBudgetExceeded,
    arxiv_id_from_abs_url,
    build_deep_arxiv_queries,
    build_arxiv_queries,
    is_confident_enough,
    parse_arxiv_feed,
    score_arxiv_match,
    title_similarity,
)


ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2601.01234v2</id>
    <updated>2026-02-02T00:00:00Z</updated>
    <published>2026-01-20T00:00:00Z</published>
    <title>Annotations Mitigate Post-Training Mode Collapse</title>
    <summary>
      We study post-training mode collapse and show that annotations mitigate it.
    </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Grace Hopper</name></author>
    <category term="cs.LG" />
    <link href="http://arxiv.org/abs/2601.01234v2" rel="alternate" type="text/html" />
    <link title="pdf" href="http://arxiv.org/pdf/2601.01234v2" rel="related" type="application/pdf" />
  </entry>
</feed>
"""


class ArxivTests(unittest.TestCase):
    def test_parse_arxiv_feed_extracts_latest_entry_metadata(self) -> None:
        entries = parse_arxiv_feed(ATOM_FEED)

        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry.arxiv_id, "2601.01234")
        self.assertEqual(entry.title, "Annotations Mitigate Post-Training Mode Collapse")
        self.assertEqual(entry.authors, ["Ada Lovelace", "Grace Hopper"])
        self.assertEqual(entry.categories, ["cs.LG"])
        self.assertEqual(entry.pdf_url, "http://arxiv.org/pdf/2601.01234v2")

    def test_arxiv_id_from_abs_url_strips_version(self) -> None:
        self.assertEqual(arxiv_id_from_abs_url("https://arxiv.org/abs/cs/0503069v1"), "cs/0503069")
        self.assertEqual(arxiv_id_from_abs_url("https://arxiv.org/abs/2601.01234v3"), "2601.01234")

    def test_build_queries_include_title_keywords_author_and_date(self) -> None:
        record = PaperRecord(
            paper_id="p1",
            source="accepted",
            title="Annotations Mitigate Post-Training Mode Collapse",
            authors=["Ada Lovelace", "Grace Hopper"],
        )

        queries = build_arxiv_queries(record, published_from="2025-01-01", published_to="2026-12-31")

        self.assertIn('ti:"Annotations Mitigate Post-Training Mode Collapse"', queries[0])
        self.assertTrue(any("au:lovelace" in query for query in queries))
        self.assertTrue(all("submittedDate:[202501010000 TO 202612312359]" in query for query in queries))
        self.assertLessEqual(len(queries), 2)

    def test_resolve_record_reports_only_executed_queries(self) -> None:
        record = PaperRecord(
            paper_id="p1",
            source="accepted",
            title="Annotations Mitigate Post-Training Mode Collapse",
            abstract="We study post-training mode collapse and show that annotations mitigate it.",
            authors=["Ada Lovelace", "Grace Hopper"],
        )
        entry = ArxivEntry(
            arxiv_id="2601.01234",
            title="Annotations Mitigate Post-Training Mode Collapse",
            abstract="We study post-training mode collapse and show that annotations mitigate it.",
            authors=["Ada Lovelace", "Grace Hopper"],
            published="2026-01-20T00:00:00Z",
            updated="2026-02-02T00:00:00Z",
            categories=["cs.LG"],
            abs_url="https://arxiv.org/abs/2601.01234",
            pdf_url="https://arxiv.org/pdf/2601.01234",
        )
        client = ArxivClient()
        calls: list[str] = []

        def fake_search(query: str, *, max_results: int = 10) -> list[ArxivEntry]:
            calls.append(query)
            return [entry]

        setattr(client, "search", fake_search)

        resolution = client.resolve_record(record, published_from="2025-01-01", published_to="2026-12-31")

        self.assertEqual(len(calls), 1)
        self.assertEqual(resolution.queries, calls)

    def test_deep_queries_include_author_pair_and_all_field_fallbacks(self) -> None:
        record = PaperRecord(
            paper_id="p1",
            source="accepted",
            title="Multimodal Function Vectors for Spatial Relations",
            authors=["Ada Lovelace", "Grace Hopper"],
        )

        queries = build_deep_arxiv_queries(record, published_from="2025-01-01", published_to="2026-12-31")

        self.assertIn("au:lovelace AND au:hopper", queries[0])
        self.assertTrue(any("all:multimodal" in query and "au:lovelace" in query for query in queries))
        self.assertTrue(any("all:multimodal" in query and "all:function" in query for query in queries))
        self.assertTrue(all("submittedDate:[202501010000 TO 202612312359]" in query for query in queries))

    def test_search_uses_persistent_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "arxiv-cache"
            http = HttpClient()
            calls = 0

            def fake_fetch(url: str, *, accept: str | None = None) -> FetchResult:
                nonlocal calls
                calls += 1
                return FetchResult(
                    url=url,
                    status=200,
                    content_type="application/atom+xml",
                    body=ATOM_FEED.encode("utf-8"),
                    final_url=url,
                )

            setattr(http, "fetch", fake_fetch)
            client = ArxivClient(http_client=http, cache_dir=cache_dir)

            first = client.search('ti:"Annotations Mitigate Post-Training Mode Collapse"', max_results=5)
            second = client.search('ti:"Annotations Mitigate Post-Training Mode Collapse"', max_results=5)

            self.assertEqual(len(first), 1)
            self.assertEqual(len(second), 1)
            self.assertEqual(calls, 1)
            self.assertEqual(client.stats.network_requests, 1)
            self.assertEqual(client.stats.cache_hits, 1)
            self.assertEqual(client.stats.cache_writes, 1)

    def test_search_respects_network_query_budget(self) -> None:
        client = ArxivClient(max_network_requests=0)

        with self.assertRaises(ArxivQueryBudgetExceeded):
            client.search('ti:"Annotations Mitigate Post-Training Mode Collapse"', max_results=5)

    def test_score_arxiv_match_exact_title_and_author_overlap(self) -> None:
        record = PaperRecord(
            paper_id="p1",
            source="accepted",
            title="Annotations Mitigate Post-Training Mode Collapse",
            abstract="We study post-training mode collapse and show that annotations mitigate it.",
            authors=["Ada Lovelace", "Grace Hopper"],
        )
        entry = ArxivEntry(
            arxiv_id="2601.01234",
            title="Annotations Mitigate Post-Training Mode Collapse",
            abstract="We study post-training mode collapse and show that annotations mitigate it.",
            authors=["Ada Lovelace", "Someone Else"],
            published="2026-01-20T00:00:00Z",
            updated="2026-02-02T00:00:00Z",
            categories=["cs.LG"],
            abs_url="https://arxiv.org/abs/2601.01234",
            pdf_url="https://arxiv.org/pdf/2601.01234",
        )

        match = score_arxiv_match(record, entry, published_from="2025-01-01", published_to="2026-12-31")

        self.assertEqual(match.confidence, "exact")
        self.assertGreaterEqual(match.score, 0.9)
        self.assertEqual(match.evidence["date_status"], "inside_range")
        self.assertTrue(is_confident_enough(match.confidence, "high"))

    def test_score_arxiv_match_accepts_reordered_title_with_strong_author_and_abstract_evidence(self) -> None:
        abstract = "We transfer functional knowledge for low-resource code generation with test-time scaling."
        record = PaperRecord(
            paper_id="p1",
            source="accepted",
            title="CodeChemist: Test-Time Scaling for Low-Resource Code Generation via Functional Knowledge Transfer",
            abstract=abstract,
            authors=["Ada Lovelace", "Grace Hopper"],
        )
        entry = ArxivEntry(
            arxiv_id="2601.01234",
            title="CodeChemist: Functional Knowledge Transfer for Low-Resource Code Generation via Test-Time Scaling",
            abstract=abstract,
            authors=["Ada Lovelace", "Grace Hopper"],
            published="2026-01-20T00:00:00Z",
            updated="2026-02-02T00:00:00Z",
            categories=["cs.LG"],
            abs_url="https://arxiv.org/abs/2601.01234",
            pdf_url="https://arxiv.org/pdf/2601.01234",
        )

        match = score_arxiv_match(record, entry, published_from="2025-01-01", published_to="2026-12-31")

        self.assertEqual(match.confidence, "high")

    def test_title_similarity_penalizes_different_titles(self) -> None:
        self.assertGreater(title_similarity("A Robust Optimization Method", "A Robust Optimisation Method"), 0.8)
        self.assertLess(title_similarity("A Robust Optimization Method", "Completely Different Vision Benchmark"), 0.5)


if __name__ == "__main__":
    unittest.main()
