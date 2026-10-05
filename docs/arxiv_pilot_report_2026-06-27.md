# ICML 2026 arXiv Fallback Pilot, 2026-06-27

This pilot tested whether arXiv could safely provide a temporary PDF fallback
at a time when official ICML/OpenReview PDFs were not yet public.

Supersession note: on 2026-06-28, official ICML 2026 PDFs became publicly
resolvable through OpenReview forum IDs. This report remains useful for arXiv
fallback coverage, match precision, revision timing, and responsible-querying
behavior, but arXiv is no longer the primary PDF source.

## Metadata Resolution

Input: first 500 rows from `data/metadata/icml_2026_accepted_2026-06-17.jsonl`.

Command family:

```bash
python3 -m icml_ai_ac.cli resolve-arxiv \
  --manifest data/metadata/icml_2026_accepted_2026-06-17.jsonl \
  --out data/metadata/icml_2026_accepted_with_arxiv_pilot500.jsonl \
  --candidates data/metadata/icml_2026_arxiv_candidates_pilot500.jsonl \
  --unresolved data/metadata/icml_2026_arxiv_unresolved_pilot500.jsonl \
  --limit 500 \
  --search-limit 5 \
  --min-confidence high \
  --published-from 2024-01-01 \
  --published-to 2026-12-31 \
  --max-network-queries 250
```

The run completed through budgeted resumable chunks. Final status:

- 500 total rows, 500 unique paper IDs, 0 missing statuses.
- 319 high-confidence arXiv matches.
- 181 unresolved rows.
- 298 exact-confidence matches and 21 high-confidence matches.
- 0 rate-limit cooldowns observed.
- 555 total arXiv metadata network requests across the three chunks.
- Approximate metadata wall time: 6,854 seconds under conservative throttling.

For the 319 high-confidence matches, arXiv first-post years were:

- 2024: 5
- 2025: 82
- 2026: 232

Among matched papers, 29 were first posted on or after 2026-06-01 and 96 were
updated on or after 2026-06-01. This confirms that arXiv PDFs are useful, but
they must remain explicitly marked as preprints or revised preprints rather than
canonical ICML camera-ready PDFs.

## Audit Results

Audit command:

```bash
python3 -m icml_ai_ac.cli audit-arxiv-resolution \
  --manifest data/metadata/icml_2026_accepted_with_arxiv_pilot500.jsonl \
  --candidates data/metadata/icml_2026_arxiv_candidates_pilot500.jsonl \
  --out data/metadata/icml_2026_arxiv_audit_pilot500.jsonl \
  --report data/metadata/icml_2026_arxiv_audit_pilot500.report.json
```

Audit buckets:

- `automatic_high_confidence`: 319
- `likely_absent_no_candidate`: 176
- `review_probable_prior_title`: 3
- `review_possible_related_or_prior`: 1
- `likely_absent_weak_related_hits`: 1

This is the right shape for production: automatic rows can flow to PDF download,
and ambiguous rows are explicit review queues rather than silent false positives.

## PDF Download And Parse

PDF command:

```bash
python3 -m icml_ai_ac.cli download-arxiv-pdfs \
  --manifest data/metadata/icml_2026_accepted_with_arxiv_pilot500.jsonl \
  --out data/metadata/icml_2026_accepted_with_arxiv_pilot500_pdfs200.jsonl \
  --pdf-dir data/pdfs/arxiv_icml_2026_pilot500 \
  --min-confidence high \
  --limit 200 \
  --delay 5 \
  --backoff 15
```

Download result:

- 200/200 selected high-confidence PDFs downloaded.
- 0 failed downloads.
- 0 skipped downloads.
- PDF sizes ranged from 491,819 bytes to 39,286,249 bytes.

The downloaded-only manifest was parsed with:

```bash
python3 -m icml_ai_ac.cli parse-pdfs \
  --manifest data/metadata/icml_2026_arxiv_pdfs200_downloaded_only.jsonl \
  --out data/metadata/icml_2026_arxiv_pdfs200_parsed.jsonl \
  --text-dir data/extracted_text/icml_2026_arxiv_pdfs200 \
  --report data/metadata/icml_2026_arxiv_pdfs200_parse_report.json \
  --main-paper-max-pages 9
```

Parse result:

- 200 PDFs parsed.
- 196 `ok`.
- 4 `needs_review`.
- 0 `parse_failed`.
- Median scoring representation length: 40,042 characters.
- Median limited main-body length: 39,730 characters.

The four `needs_review` rows still produced substantial text representations
and were flagged for missing abstract/conclusion headings, not extraction
failure.

## Operational Verdict

The arXiv fallback is ready for controlled production enrichment. It should not
be treated as canonical conference PDF acquisition. The production run should
continue to:

- prefer official ICML/OpenReview PDFs when they become public;
- mark arXiv PDFs with `extra.pdf_path_source=arxiv_preprint`;
- use budgeted `resolve-arxiv` chunks with persistent query caching;
- run `audit-arxiv-resolution` before downloading;
- parse only downloaded rows; and
- keep ambiguous/older-title arXiv candidates as explicit review queues.
