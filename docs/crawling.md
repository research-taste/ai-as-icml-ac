# Crawling Infrastructure

The first-phase crawler is intentionally stdlib-only. It can run in a fresh
Python environment before parser/model dependencies are added.

## Release Status Audit

Before a full crawl, check what ICML/OpenReview currently exposes:

```bash
python3 -m icml_ai_ac.cli release-status \
  --year 2026 \
  --out data/metadata/icml_2026_release_status.json
```

As of 2026-07-13, the canonical snapshot contains:

- 6,628 accepted poster-paper records from the ICML JSON feed
- 574 spotlights and 168 papers with oral presentations
- 6,613 OpenReview forum IDs and 0 explicit PDF URLs in the ICML feed
- 6,341 accepted records with published OpenReview reviewer ratings
- 214 public opt-in rejected notes in the latest successful snapshot
- nine 2026 award papers linked from the official ICML awards announcement

OpenReview's unauthenticated API/PDF endpoints may return an interactive access
gate from the production host. Production therefore uses an explicitly enabled
authenticated API v2 session. Treat any challenge, authentication error, or
rate limit as a batch-wide pause condition, not as thousands of paper-specific
failures; the production commands below are resumable.

The PDFs are present on OpenReview; file availability and access mode are
separate questions. OpenReview's documented API path is
`GET /attachment?id=<note_id>&name=pdf` (or `GET /pdf?id=<note_id>`), using an
authenticated API v2 client. Use arXiv only as a provenance-labeled fallback
when an official camera-ready PDF cannot be resolved.

## Accepted ICML Virtual-Site Metadata Crawl

Prefer the structured ICML JSON feed for metadata. It is one request for the
event table plus one request for abstracts, so it is much gentler than crawling
thousands of detail pages:

```bash
python3 -m icml_ai_ac.cli crawl-accepted-virtual-json \
  --year 2026 \
  --out data/metadata/icml_2026_accepted.jsonl \
  --report data/metadata/icml_2026_accepted.report.json
```

The default emits one canonical record per `Poster` row. It indexes linked
`Oral` rows first, then merges oral schedule and presentation metadata into the
matching poster records. The report records event and decision counts, oral and
spotlight counts, abstract coverage, forum coverage, and PDF coverage.

## Scores and Award Enrichment

OpenReview API v2 returns current direct replies with
`details=directReplies`. This command extracts only structured fields from
official reviews; it does not retain review text:

```bash
python3 -m icml_ai_ac.cli crawl-openreview-scores \
  --venue-id ICML.cc/2026/Conference \
  --from-venue-status accepted \
  --out data/metadata/icml_2026_openreview_accepted_ratings.jsonl \
  --delay 2
```

The extracted snapshot preserves per-review values and timestamps plus means
for overall assessment, soundness, and confidence. These are published reviewer
ratings, not a private AC score. The 2026 awards are maintained as a small,
auditable file transcribed from the official ICML announcement.

Merge ratings and awards into the canonical ICML inventory:

```bash
python3 -m icml_ai_ac.cli enrich-metadata \
  --manifest data/metadata/icml_2026_accepted_2026-07-13.jsonl \
  --ratings data/metadata/icml_2026_openreview_accepted_ratings_2026-06-28.jsonl \
  --awards data/metadata/icml_2026_awards_2026-07-05.json \
  --out data/metadata/icml_2026_production_manifest.jsonl
```

Ratings join by OpenReview forum ID, with unique exact-title fallback. Awards
join by normalized exact title. The report records unmatched rows; the only
unmatched award is the historical Test of Time paper, which is intentionally
outside the 2026 accepted corpus.

## Accepted ICML Detail-Page Crawl

The HTML detail-page crawler is retained for smoke tests and parser debugging.
It should not be the primary full-metadata crawl unless the JSON feed is missing
or inconsistent.

Enumerate accepted-paper poster pages and write a manifest:

```bash
python3 -m icml_ai_ac.cli crawl-accepted-virtual \
  --year 2026 \
  --out data/metadata/icml_2026_accepted.jsonl \
  --delay 0.25
```

Run a small smoke crawl and preserve raw HTML for parser debugging:

```bash
python3 -m icml_ai_ac.cli crawl-accepted-virtual \
  --limit 20 \
  --raw-html-dir data/raw_html/icml_2026_accepted \
  --out data/metadata/icml_2026_accepted_sample.jsonl
```

Download PDFs while crawling:

```bash
python3 -m icml_ai_ac.cli crawl-accepted-virtual \
  --download-pdfs \
  --pdf-dir data/pdfs/icml_2026 \
  --out data/metadata/icml_2026_accepted.jsonl \
  --delay 0.25
```

Official PDFs are resolved from OpenReview forum IDs. This is preferred over
arXiv preprints because it fills the canonical `pdf_url` and `pdf_path` fields:

Store `OPENREVIEW_USERNAME` and `OPENREVIEW_PASSWORD` in the ignored `.env`
file and restrict it with `chmod 600 .env`. Authentication is opt-in: the CLI
exchanges those credentials for a one-hour API v2 bearer token only when
`--use-openreview-auth` is present. Credentials and tokens are never written to
manifests, reports, journals, or command arguments.

```bash
python3 -m icml_ai_ac.cli resolve-openreview-pdfs \
  --manifest data/metadata/icml_2026_production_manifest.jsonl \
  --out data/metadata/icml_2026_production_with_pdfs.jsonl \
  --unresolved data/metadata/icml_2026_openreview_pdf_unresolved.jsonl \
  --pdf-dir data/pdfs/icml_2026_openreview \
  --use-openreview-auth \
  --download-pdfs \
  --delay 2 \
  --progress-every 25
```

Rerun the same command after an interruption. It resumes from the output and
`<out>.partial.jsonl`, recognizes valid PDFs already in the destination, and
never accepts an HTML error page as a PDF. An OpenReview challenge writes a
complete state snapshot and exits after the first blocked request. The run
also halts before issuing another request after HTTP 401, 403, or 429. The run
report records completed, pending, failed, recovered-local, and blocked counts.
Authenticated downloads use `api2.openreview.net/pdf`, retain checksums, and
set `source_pdf_kind=openreview_official` and
`pdf_path_source=openreview_official`.

The 2026-07-15 staged production rollout completed 25/25, 100/100, 498/500,
997/1,000, 1,496/1,500, 1,996/2,000, 2,496/2,500, 2,993/3,000, 3,493/3,500,
3,992/4,000, 4,492/4,500, 4,989/5,000, 5,487/5,500, 5,986/6,000,
6,486/6,500, and finally 6,614/6,628 selected papers with a two-second delay
and no access challenge or rate limit.
Two isolated requests timed out and each succeeded on the first later
invocation; the 6,001--6,500 tranche downloaded 500/500 without retry. This
leaves zero outstanding download failures. The fourteen incomplete records
have no OpenReview forum ID, did not expose an official PDF, and remain
explicitly unresolved. All
6,614 completed files passed
signature, checksum, uniqueness, and provenance validation. With the nine-page
main-body cap, 6,611 parsed `ok`, three usable extractions were marked
`needs_review`, and none
failed.

## Identity-Redacted Scoring Artifacts

Keep canonical PDFs and parsed text immutable. Before any model call, create a
separate fingerprinted scoring manifest whose PDF and text paths resolve to
identity-redacted derivatives:

```bash
python3 -m icml_ai_ac.cli anonymize-papers \
  --manifest data/metadata/icml_2026_scoring_manifest.jsonl \
  --out data/metadata/icml_2026_scoring_anonymized.jsonl \
  --pdf-dir data/pdfs/icml_2026_anonymized \
  --text-dir data/extracted_text/icml_2026_anonymized \
  --report data/metadata/icml_2026_scoring_anonymized.report.json \
  --max-pages 9
```

The command removes a first-page identity band established from the title,
abstract boundary, and author/email/affiliation/anonymous-byline signals, plus
known author-name occurrences, correspondence lines, emails, acknowledgements,
mail links, source/reviewer footers, and PDF author/XML metadata. It tolerates
inserted middle names and fragmented superscript text, while ignoring
implausibly short uppercase metadata identities that would over-redact notation.
It validates every derivative and omits
non-passing rows from the output manifest. Its append-only journal supports
exact resume; rerunning the unchanged 103-paper production-layout probe reused
103/103 rows in under half a second.

That probe passed 103/103 PDFs in about two minutes. It included the first 100
scoring records plus three unusual layouts without an `Abstract` heading. Twelve
metadata author strings differed from the camera-ready byline; the complete
author bands were still detected and removed, and the drift remains in the
audit. Two malformed source PDFs emitted six repair diagnostics, but their
saved derivatives passed page-count, identity, email, and metadata validation.
A separate 50-paper ICML 2025 run also passed 50/50. This is direct identity
redaction, not guaranteed anonymity: titles, self-citations, project names, and
model memory can still identify papers.

The final production gate completed on 2026-07-27 with 6,617/6,617 records,
zero exclusions, 6,617 validated PDFs, and 19,851 validated text artifacts.
The v25 base avoided a blanket rerun by promoting checksum-verified derivatives
and regenerating only incomplete or anomalous rows. No later text correction
rewrote a PDF. Selective v26-v33 passes removed hyphen-wrapped names,
correspondence lines, institution blocks displaced below the Abstract by
two-column extraction, accented and acronym-only affiliations, and wrapped
employment/contribution notes. The final version counts are 4,236 v25, 153
v26, 1,265 v27, 492 v28, 362 v29, 27 v30, 14 v31, 60 v32, and eight v33.

The final audit is
`data/model_runs/icml_2026_full_launch/stage00_v33_final_audit.json`; it reports
zero errors after checking exact coverage, blind record structure, text
idempotence, source and derivative checksums, signatures, page counts,
residual identities/contact cues, links, and PDF metadata. A separate broad
layout diagnostic leaves only ten inspected equation/introduction cases and no
affiliation-like fragment. The title-region audit retained all 6,049 metadata
titles actually present above the source Abstract; 568 metadata titles differ
from the camera-ready PDF and remain recorded as provenance drift. Every
failed gate and selective checkpoint is preserved.

If a manifest already contains explicit PDF URLs, download them directly:

```bash
python3 -m icml_ai_ac.cli download-pdfs \
  --manifest data/metadata/icml_2026_accepted.jsonl \
  --pdf-dir data/pdfs/icml_2026 \
  --out data/metadata/icml_2026_accepted_with_pdfs.jsonl
```

## arXiv Preprint Fallback

arXiv is a fallback/enrichment source for papers whose official ICML/OpenReview
PDF cannot be resolved. It should not replace the canonical official PDF fields.
The resolver stores likely preprints under `extra.arxiv_*` and marks
`extra.source_pdf_kind=arxiv_preprint`; it does not overwrite `pdf_url` or
`pdf_path`.

Run metadata-only resolution first:

```bash
python3 -m icml_ai_ac.cli resolve-arxiv \
  --manifest data/metadata/icml_2026_accepted.jsonl \
  --out data/metadata/icml_2026_accepted_with_arxiv.jsonl \
  --candidates data/metadata/icml_2026_arxiv_candidates.jsonl \
  --unresolved data/metadata/icml_2026_arxiv_unresolved.jsonl \
  --min-confidence high \
  --published-from 2024-01-01 \
  --published-to 2026-12-31
```

Only after inspecting candidate diagnostics, download high-confidence arXiv
PDFs into a separate directory:

```bash
python3 -m icml_ai_ac.cli download-arxiv-pdfs \
  --manifest data/metadata/icml_2026_accepted_with_arxiv.jsonl \
  --out data/metadata/icml_2026_accepted_with_arxiv_pdfs.jsonl \
  --downloaded-only-out data/metadata/icml_2026_arxiv_downloaded_only.jsonl \
  --pdf-dir data/pdfs/arxiv_icml_2026 \
  --min-confidence high \
  --limit 200
```

The download command is restartable from files on disk: existing valid PDFs are
hashed and skipped, so increasing `--limit` advances the same ordered tranche.
For example, rerunning with `--limit 300` after a completed 200-paper batch
verifies the first 200 locally and downloads 100 more. The 2026-07-13 run did
exactly this with a five-second delay: 100 new downloads, 200 local skips, zero
failures, and 300/300 valid PDF signatures. The 100-paper delta parsed as 96
`ok`, 4 `needs_review` for missing section headings, and 0 failed.

The bounded 2026-07-14 pilot refresh completed at 329/500 high-confidence
matches, up from 319/500. A final catch-up downloaded 19 previously matched
tail records, leaving all 329 matches locally validated. The 19-file catch-up
parsed as 17 `ok`, 2 `needs_review`, and 0 failed. Retry state retains every
campaign ID attempted for each paper, so targeted network recovery cannot make
a row eligible again in an overlapping bulk campaign. A parse-ledger audit found
and repaired one omitted batch entry; all 329 pilot PDFs are now represented as
318 `ok`, 11 `needs_review`, and 0 failed.

The production expansion excludes pilot records by exact paper ID rather than
position because the canonical 6,628-paper manifest uses a different order. Its
6,128-row source started with 13 exact matches from 19 processed papers; all 13
PDFs validated. After 706 processed papers, the expansion has 447 high-confidence
matches and 259 unresolved rows. All 447 PDFs validated without overlap with the
pilot set; parsing produced 437 `ok`, 10 `needs_review`, and 0 failed. One metadata
read timeout was recovered through a targeted campaign without repeating other
rows.

This command does not query the arXiv metadata API. It downloads known
high-confidence `extra.arxiv_pdf_url` values, stores checksums under
`extra.arxiv_pdf_*`, and sets `pdf_path` only when the canonical `pdf_path` is
empty. The row keeps `extra.pdf_path_source=arxiv_preprint`.

Set `--limit` to the number of matched records to guarantee that every current
match is local. A limit based only on the previous local-file count can leave
matched tail records outside the ordered tranche after new matches are inserted.

Parse the downloaded-only output. This keeps intentional download caps and
unresolved rows from appearing as parse-missing failures:

```bash
python3 -m icml_ai_ac.cli parse-pdfs \
  --manifest data/metadata/icml_2026_arxiv_downloaded_only.jsonl \
  --out data/metadata/icml_2026_arxiv_parsed.jsonl \
  --text-dir data/extracted_text/icml_2026_arxiv \
  --report data/metadata/icml_2026_arxiv_parse_report.json \
  --main-paper-max-pages 9
```

The command defaults to a 12-second delay between arXiv API requests, a
30-second retry backoff, and a persistent API cache at `data/cache/arxiv_api`.
The arXiv API is intentionally treated as a slow, shared service. Re-running the
same query should normally be a cache hit, not another request to arXiv. Use
`--max-network-queries` as a hard budget for probes or production batches; cache
hits do not count against this limit.

The resolver uses exact-title search plus one author-constrained fallback query,
optional date bounds, and lightweight abstract overlap to assign `exact`,
`high`, `medium`, `low`, or `none` confidence. Treat `medium`/`low` rows as
review queues, not automatic PDF sources. Interrupted runs leave a
`.partial.jsonl` journal next to the output manifest; rerun the same command to
resume without re-querying completed paper IDs. Use repeated `--paper-id` values
for targeted retry, add `--retry-failed` to re-attempt rows that previously
failed because of transient network/API errors, and add `--retry-unresolved`
when a resolver scoring change should be applied to specific previously
unresolved rows.

For a retry campaign split across bounded invocations, also pass one stable
`--resolution-run-id`. Each unresolved or failed row is then retried at most
once for that campaign, even when `--max-network-queries` stops and resumes the
run. Changing the run ID intentionally starts a new refresh campaign. `Ctrl-C`
writes the current snapshot and stop reason without discarding completed rows.

For audit-oriented recovery of unresolved rows, use `--deep-search`. This adds
author-pair and all-field title-term fallback queries. It is intentionally not
the default because it is slower and can retrieve related prior work by the same
authors; keep `--min-confidence high` and inspect any remaining low-confidence
candidates manually.

Use `--refresh-arxiv-cache` only when intentionally revalidating stale arXiv
responses, and `--no-arxiv-cache` only for debugging. The run report records
`arxiv_client_stats` with search count, network requests, cache hits, cache
writes, and long cooldowns after rate limits.

After a resolver run, generate explicit review queues:

```bash
python3 -m icml_ai_ac.cli audit-arxiv-resolution \
  --manifest data/metadata/icml_2026_accepted_with_arxiv.jsonl \
  --candidates data/metadata/icml_2026_arxiv_candidates.jsonl \
  --out data/metadata/icml_2026_arxiv_audit.jsonl
```

The audit classes are:

- `automatic_high_confidence`: safe for automatic downstream PDF retrieval.
- `review_probable_prior_title`: likely same paper with an older or reworded
  title; review before automatic use.
- `review_possible_related_or_prior`: same-author related work or possible
  earlier version; manual review required.
- `likely_absent_no_candidate` / `likely_absent_weak_related_hits`: do not use as
  an arXiv source unless later evidence appears.

## OpenReview Crawl

OpenReview invitation and visibility conventions can change by venue/year. Start
by inspecting the venue group, which reports the API version, submission
invitation, public visibility flags, and status-specific query IDs:

```bash
python3 -m icml_ai_ac.cli inspect-openreview-venue \
  --venue-id ICML.cc/2026/Conference
```

The ICML 2026 venue uses API v2, but group visibility flags are not enough to
infer corpus availability. Status-aware public queries successfully returned
accepted notes, rejected-note rows, reviews/ratings, and official PDFs on
2026-06-28; on 2026-07-13, the same host required interactive verification.
The ICML JSON feed remains the canonical accepted inventory. OpenReview adds
ratings, public rejected notes, and official PDFs when access is available.

Use the status-aware query builder when possible:

```bash
python3 -m icml_ai_ac.cli crawl-openreview \
  --venue-id ICML.cc/2026/Conference \
  --from-venue-status accepted \
  --source accepted \
  --out data/metadata/openreview_icml_2026_accepted_public_archive.jsonl
```

The default direct venue-id query is:

```bash
python3 -m icml_ai_ac.cli crawl-openreview \
  --venue-id ICML.cc/2026/Conference \
  --source openreview_public \
  --out data/metadata/openreview_icml_2026_public.jsonl
```

If the public rejected subset uses a different invitation or venue label, pass
it explicitly after inspecting OpenReview:

```bash
python3 -m icml_ai_ac.cli crawl-openreview \
  --invitation 'ICML.cc/2026/Conference/-/Submission' \
  --source rejected_public \
  --decision-label rejected_public \
  --out data/metadata/icml_2026_rejected_public.jsonl
```

For fully custom API parameters, create a JSON object and use `--query-file`.

## ICML 2025 Test Bed

ICML 2025 is useful as a released-year stand-in for the ICML 2026 pipeline. The
accepted records and PDFs are available through OpenReview API v2.

Inspect the venue:

```bash
python3 -m icml_ai_ac.cli inspect-openreview-venue \
  --venue-id ICML.cc/2025/Conference \
  --out data/metadata/openreview_icml_2025_venue_report.json
```

Crawl 50 accepted metadata records:

```bash
python3 -m icml_ai_ac.cli crawl-openreview \
  --venue-id ICML.cc/2025/Conference \
  --from-venue-status accepted \
  --source accepted \
  --out data/metadata/icml_2025_accepted_50.jsonl \
  --limit 50 \
  --delay 1
```

Download their PDFs:

```bash
python3 -m icml_ai_ac.cli download-pdfs \
  --manifest data/metadata/icml_2025_accepted_50.jsonl \
  --pdf-dir data/pdfs/icml_2025_accepted_50 \
  --out data/metadata/icml_2025_accepted_50_with_pdfs.jsonl \
  --delay 1
```

Parse the PDF text and build paper-only representations:

```bash
python3 -m icml_ai_ac.cli parse-pdfs \
  --manifest data/metadata/icml_2025_accepted_50_with_pdfs.jsonl \
  --out data/metadata/icml_2025_accepted_50_parsed.jsonl \
  --text-dir data/extracted_text/icml_2025_accepted_50 \
  --report data/metadata/icml_2025_accepted_50_parse_report.json
```

The parser uses Poppler `pdftotext -raw` and `pdfinfo`. For each paper it writes
`raw.txt`, `clean.txt`, `compact_repr.txt`, `full_repr.txt`, and
`sections.json`; the parsed manifest stores paths to the compact and full
representations plus parse diagnostics.
