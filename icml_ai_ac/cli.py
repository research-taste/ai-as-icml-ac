from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import traceback
import urllib.error
from pathlib import Path
from typing import Any

from icml_ai_ac.cheap_models import CHEAP_MODEL_PRESETS, DEFAULT_CHEAP_MODEL, resolve_cheap_models
from icml_ai_ac.env import load_dotenv
from icml_ai_ac.eval import evaluate_ranking
from icml_ai_ac.analysis.human_comparison import (
    build_divergence_report,
    load_comparison_dataset,
    render_markdown,
)
from icml_ai_ac.analysis.taxonomy import (
    TaxonomyConfig,
    run_taxonomy_classification,
    run_taxonomy_induction,
)
from icml_ai_ac.analysis.tournament_simulation import simulate_hybrid_tournament
from icml_ai_ac.analysis.agreement import build_agreement_report, build_axis_decomposition_report
from icml_ai_ac.analysis.semifinal_audit import audit_semifinal_rankings
from icml_ai_ac.finalists import FinalistSelectionConfig, read_ranked_rows, select_finalists
from icml_ai_ac.http import AccessChallengeError, HttpClient, is_pdf_file, sha256_file
from icml_ai_ac.metadata import enrich_metadata_rows, summarize_openreview_scores
from icml_ai_ac.model_presets import (
    DEFAULT_CLASSIFICATION_MODEL,
    DEFAULT_FRONTIER_MODEL,
    DEFAULT_STRONG_MODEL,
)
from icml_ai_ac.position_sensitivity import build_position_robust_shortlist
from icml_ai_ac.models import PaperRecord
from icml_ai_ac.ranking_ensemble import ensemble_semifinal_rankings
from icml_ai_ac.ranking_query import (
    DEFAULT_CHEAP_RANKING_PATH,
    DEFAULT_FINAL_RANKING_PATH,
    DEFAULT_METADATA_PATH,
    QUERY_PRESETS,
    QUERY_SCOPES,
    RankingQuery,
    available_ranking_categories,
    query_ranked_papers,
    top_ranked_papers,
    write_query_results,
)
from icml_ai_ac.scraper.arxiv import ArxivClient, ArxivQueryBudgetExceeded, arxiv_pdf_filename, is_confident_enough
from icml_ai_ac.scraper.arxiv_audit import ArxivAuditThresholds, audit_arxiv_resolution
from icml_ai_ac.parser import parse_pdf_record, write_parse_report
from icml_ai_ac.scraper.icml_virtual import ICMLVirtualScraper
from icml_ai_ac.scraper.openreview import (
    OpenReviewClient,
    build_query,
    content_value,
    forum_id_from_url,
    note_to_record,
    pdf_url_for_forum_id,
    queries_for_status,
    read_query_file,
    venue_status_report,
)
from icml_ai_ac.scoring.prompts import PROMPT_VERSION
from icml_ai_ac.scoring.providers import ChatCompletionClient
from icml_ai_ac.scoring.batch import (
    PASS1_BATCH_PROMPT_VERSION,
    Pass1BatchConfig,
    Pass1BatchSuiteConfig,
    run_pass1_batch_rank,
    run_pass1_batch_suite,
)
from icml_ai_ac.scoring.classification import (
    CLASSIFY_PROMPT_VERSION,
    ContributionClassificationConfig,
    run_contribution_classification,
)
from icml_ai_ac.scoring.frontier_gold import (
    FRONTIER_CARD_ENSEMBLE_VERSION,
    FRONTIER_CARD_PROMPT_VERSION,
    FRONTIER_SYNTHESIS_PROMPT_VERSION,
    FRONTIER_TOURNAMENT_PROMPT_VERSION,
    FrontierCardConfig,
    FrontierSynthesisConfig,
    FrontierTournamentConfig,
    build_frontier_card_ensemble,
    run_frontier_card_tournament,
    run_frontier_card_synthesis,
    run_frontier_pdf_cards,
)
from icml_ai_ac.scoring.bradley_terry import (
    STAGE_MODES,
    WEIGHT_MODES,
    rank_bradley_terry,
    ranked_meta_from_payload,
    select_tournament_stage,
)
from icml_ai_ac.scoring.ranking import (
    PASS2_BATCHED_PROMPT_VERSION,
    PASS2_PROMPT_VERSION,
    PASS2_FINAL_PROMPT_VERSION,
    PASS2_STAGE1_PROMPT_VERSION,
    REFERENCE_PROMPT_VERSION,
    REFERENCE_REPAIR_PROMPT_VERSION,
    Pass2BatchedRankingConfig,
    Pass2RankingConfig,
    Pass2TwoStageConfig,
    ReferenceRepairConfig,
    ReferenceRankingConfig,
    run_pass2_batched_ranking,
    run_pass2_ranking,
    run_pass2_two_stage,
    run_reference_repair,
    run_reference_ranking,
)
from icml_ai_ac.scoring.runner import ScoreRunConfig, prepare_run_dir, score_record, write_run_metadata
from icml_ai_ac.storage import append_jsonl, read_json, read_jsonl, read_jsonl_if_exists, read_paper_records, write_json, write_jsonl
from icml_ai_ac.shortlist import build_shortlist


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="icml-ai-ac")
    subparsers = parser.add_subparsers(required=True)

    accepted = subparsers.add_parser("crawl-accepted-virtual", help="Crawl accepted papers from the ICML virtual site.")
    add_http_args(accepted)
    accepted.add_argument("--year", type=int, default=2026)
    accepted.add_argument("--index-url", default=None)
    accepted.add_argument("--out", type=Path, default=Path("data/metadata/icml_2026_accepted.jsonl"))
    accepted.add_argument("--raw-html-dir", type=Path, default=None)
    accepted.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs"))
    accepted.add_argument("--download-pdfs", action="store_true")
    accepted.add_argument("--limit", type=int, default=None)
    accepted.set_defaults(func=cmd_crawl_accepted_virtual)

    accepted_json = subparsers.add_parser(
        "crawl-accepted-virtual-json",
        help="Crawl accepted-paper metadata from the ICML virtual-site JSON data feed.",
    )
    add_http_args(accepted_json)
    accepted_json.add_argument("--year", type=int, default=2026)
    accepted_json.add_argument("--events-url", default=None)
    accepted_json.add_argument("--abstracts-url", default=None)
    accepted_json.add_argument("--out", type=Path, default=Path("data/metadata/icml_2026_accepted.jsonl"))
    accepted_json.add_argument("--report", type=Path, default=None)
    accepted_json.add_argument("--event-type", action="append", default=None, help="ICML event type to include. Can be repeated.")
    accepted_json.add_argument("--require-decision", action="store_true")
    accepted_json.add_argument("--limit", type=int, default=None)
    accepted_json.set_defaults(func=cmd_crawl_accepted_virtual_json)

    batch = subparsers.add_parser("crawl-accepted-batch", help="Resumable accepted-paper metadata/PDF batch crawl.")
    add_http_args(batch)
    batch.add_argument("--year", type=int, default=2026)
    batch.add_argument("--index-url", default=None)
    batch.add_argument("--manifest", type=Path, default=Path("data/metadata/icml_2026_accepted_batch.jsonl"))
    batch.add_argument("--failures", type=Path, default=Path("data/metadata/icml_2026_accepted_failures.jsonl"))
    batch.add_argument("--run-report", type=Path, default=Path("data/metadata/icml_2026_accepted_batch.run.json"))
    batch.add_argument("--raw-html-dir", type=Path, default=None)
    batch.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs/icml_2026"))
    batch.add_argument("--download-pdfs", action="store_true")
    batch.add_argument("--limit", type=int, default=100, help="Number of new successful metadata records to write.")
    batch.add_argument("--max-attempts", type=int, default=None, help="Stop after this many new detail-page attempts.")
    batch.set_defaults(func=cmd_crawl_accepted_batch)

    openreview = subparsers.add_parser("crawl-openreview", help="Crawl public OpenReview notes using configurable API query params.")
    add_http_args(openreview)
    openreview.add_argument("--api-base", default="https://api2.openreview.net")
    openreview.add_argument("--venue-id", default="ICML.cc/2026/Conference")
    openreview.add_argument(
        "--from-venue-status",
        choices=["all_submissions", "accepted", "submitted", "rejected", "desk_rejected", "withdrawn"],
        default=None,
        help="Inspect the venue group and derive the recommended query for this status.",
    )
    openreview.add_argument("--invitation", default=None)
    openreview.add_argument("--content-venue", default=None)
    openreview.add_argument("--details", default=None)
    openreview.add_argument("--query-file", default=None, help="JSON object of raw OpenReview API query parameters.")
    openreview.add_argument("--source", default="accepted", choices=["accepted", "rejected_public", "openreview_public"])
    openreview.add_argument("--decision-label", default=None)
    openreview.add_argument("--out", type=Path, default=Path("data/metadata/openreview_notes.jsonl"))
    openreview.add_argument("--limit", type=int, default=None)
    openreview.set_defaults(func=cmd_crawl_openreview)

    openreview_scores = subparsers.add_parser(
        "crawl-openreview-scores",
        help="Crawl published OpenReview review-score fields without storing review text.",
    )
    add_http_args(openreview_scores)
    openreview_scores.add_argument("--api-base", default="https://api2.openreview.net")
    openreview_scores.add_argument("--venue-id", default="ICML.cc/2026/Conference")
    openreview_scores.add_argument(
        "--from-venue-status",
        choices=["accepted", "rejected"],
        default="accepted",
    )
    openreview_scores.add_argument("--out", type=Path, required=True)
    openreview_scores.add_argument("--report", type=Path, default=None)
    openreview_scores.add_argument("--limit", type=int, default=None)
    openreview_scores.set_defaults(func=cmd_crawl_openreview_scores)

    resolve = subparsers.add_parser(
        "resolve-openreview-pdfs",
        help="Resolve/download official ICML PDFs for an existing manifest via OpenReview forum IDs or title search.",
    )
    add_http_args(resolve)
    resolve.add_argument("--api-base", default="https://api2.openreview.net")
    resolve.add_argument("--manifest", type=Path, required=True)
    resolve.add_argument("--out", type=Path, required=True)
    resolve.add_argument("--unresolved", type=Path, required=True)
    resolve.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs/icml_2026"))
    resolve.add_argument("--limit", type=int, default=None)
    resolve.add_argument("--search-limit", type=int, default=10)
    resolve.add_argument("--download-pdfs", action="store_true")
    resolve.add_argument(
        "--use-openreview-auth",
        action="store_true",
        help="Authenticate with OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD before resolving PDFs.",
    )
    resolve.add_argument("--no-resume", action="store_true", help="Ignore existing output/journal state and retry from the input manifest.")
    resolve.add_argument("--progress-every", type=int, default=25, help="Print one progress line after this many processed rows.")
    resolve.add_argument(
        "--accepted-venue",
        action="append",
        default=["ICML 2026 regular", "ICML 2026 spotlight"],
        help="Accepted OpenReview venue label to trust. Can be repeated.",
    )
    resolve.set_defaults(func=cmd_resolve_openreview_pdfs)

    resolve_arxiv = subparsers.add_parser(
        "resolve-arxiv",
        help="Resolve likely arXiv preprints for an existing manifest without replacing official PDFs.",
    )
    add_http_args(resolve_arxiv, default_delay=12.0, default_backoff=30.0)
    resolve_arxiv.add_argument("--api-base", default="https://export.arxiv.org/api/query")
    resolve_arxiv.add_argument("--manifest", type=Path, required=True)
    resolve_arxiv.add_argument("--out", type=Path, required=True)
    resolve_arxiv.add_argument("--candidates", type=Path, default=None)
    resolve_arxiv.add_argument("--unresolved", type=Path, default=None)
    resolve_arxiv.add_argument("--arxiv-cache-dir", type=Path, default=Path("data/cache/arxiv_api"))
    resolve_arxiv.add_argument("--no-arxiv-cache", action="store_true")
    resolve_arxiv.add_argument("--refresh-arxiv-cache", action="store_true")
    resolve_arxiv.add_argument(
        "--max-network-queries",
        type=int,
        default=None,
        help="Optional hard cap on arXiv API requests. Cache hits do not count.",
    )
    resolve_arxiv.add_argument(
        "--rate-limit-cooldown",
        type=float,
        default=180.0,
        help="Extra cooldown in seconds after arXiv still returns HTTP 429 after normal retries.",
    )
    resolve_arxiv.add_argument(
        "--rate-limit-extra-retries",
        type=int,
        default=2,
        help="Additional long-cooldown retries for HTTP 429 responses from arXiv.",
    )
    resolve_arxiv.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs/arxiv"))
    resolve_arxiv.add_argument("--limit", type=int, default=None)
    resolve_arxiv.add_argument(
        "--paper-id",
        action="append",
        default=[],
        help="Resolve only this paper id. Can be repeated for targeted retry.",
    )
    resolve_arxiv.add_argument("--search-limit", type=int, default=10)
    resolve_arxiv.add_argument(
        "--deep-search",
        action="store_true",
        help="After the normal high-precision search fails, run slower author-pair/all-field fallback queries.",
    )
    resolve_arxiv.add_argument(
        "--deep-search-limit",
        type=int,
        default=50,
        help="Maximum arXiv results to inspect per deep fallback query.",
    )
    resolve_arxiv.add_argument("--min-confidence", choices=["low", "medium", "high", "exact"], default="high")
    resolve_arxiv.add_argument("--published-from", default=None, help="Optional YYYY-MM-DD submitted-date lower bound.")
    resolve_arxiv.add_argument("--published-to", default=None, help="Optional YYYY-MM-DD submitted-date upper bound.")
    resolve_arxiv.add_argument("--download-pdfs", action="store_true")
    resolve_arxiv.add_argument("--overwrite", action="store_true")
    resolve_arxiv.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing arXiv resolver journal/output and start a fresh run.",
    )
    resolve_arxiv.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-attempt rows whose previous arXiv resolver status was failed.",
    )
    resolve_arxiv.add_argument(
        "--retry-unresolved",
        action="store_true",
        help="Re-attempt rows whose previous arXiv resolver status was unresolved.",
    )
    resolve_arxiv.add_argument(
        "--resolution-run-id",
        default=None,
        help="Stable retry campaign id; completed rows are processed at most once for this id across resumptions.",
    )
    resolve_arxiv.set_defaults(func=cmd_resolve_arxiv)

    audit_arxiv = subparsers.add_parser(
        "audit-arxiv-resolution",
        help="Classify arXiv resolver outputs into automatic, review, and likely-absent queues.",
    )
    audit_arxiv.add_argument("--manifest", type=Path, required=True)
    audit_arxiv.add_argument("--candidates", type=Path, required=True)
    audit_arxiv.add_argument("--out", type=Path, required=True)
    audit_arxiv.add_argument("--report", type=Path, default=None)
    audit_arxiv.set_defaults(func=cmd_audit_arxiv_resolution)

    inspect = subparsers.add_parser("inspect-openreview-venue", help="Inspect OpenReview venue API version and public status flags.")
    add_http_args(inspect)
    inspect.add_argument("--api-base", default="https://api2.openreview.net")
    inspect.add_argument("--venue-id", default="ICML.cc/2026/Conference")
    inspect.add_argument("--out", type=Path, default=None)
    inspect.set_defaults(func=cmd_inspect_openreview_venue)

    release = subparsers.add_parser(
        "release-status",
        help="Audit current ICML/OpenReview public release status for metadata, PDFs, and review artifacts.",
    )
    add_http_args(release)
    release.add_argument("--year", type=int, default=2026)
    release.add_argument("--events-url", default=None)
    release.add_argument("--abstracts-url", default=None)
    release.add_argument("--api-base", default="https://api2.openreview.net")
    release.add_argument("--venue-id", default="ICML.cc/2026/Conference")
    release.add_argument("--out", type=Path, default=Path("data/metadata/icml_2026_release_status.json"))
    release.set_defaults(func=cmd_release_status)

    enrich = subparsers.add_parser(
        "enrich-metadata",
        help="Merge published OpenReview score summaries and official award labels into a paper manifest.",
    )
    enrich.add_argument("--manifest", type=Path, required=True)
    enrich.add_argument("--ratings", type=Path, default=None)
    enrich.add_argument("--awards", type=Path, default=None)
    enrich.add_argument("--out", type=Path, required=True)
    enrich.add_argument("--report", type=Path, default=None)
    enrich.set_defaults(func=cmd_enrich_metadata)

    download = subparsers.add_parser("download-pdfs", help="Download PDFs listed in a manifest JSONL.")
    add_http_args(download)
    download.add_argument("--manifest", type=Path, required=True)
    download.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs"))
    download.add_argument("--out", type=Path, default=None, help="Optional updated manifest path with pdf_path and checksum metadata.")
    download.add_argument("--limit", type=int, default=None)
    download.add_argument("--overwrite", action="store_true")
    download.set_defaults(func=cmd_download_pdfs)

    download_arxiv = subparsers.add_parser(
        "download-arxiv-pdfs",
        help="Download high-confidence arXiv preprint PDFs stored under extra.arxiv_* fields.",
    )
    add_http_args(download_arxiv, default_delay=5.0, default_backoff=15.0)
    download_arxiv.add_argument("--manifest", type=Path, required=True)
    download_arxiv.add_argument("--out", type=Path, required=True)
    download_arxiv.add_argument(
        "--downloaded-only-out",
        type=Path,
        default=None,
        help="Optional JSONL containing only successfully downloaded or verified rows.",
    )
    download_arxiv.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs/arxiv"))
    download_arxiv.add_argument("--limit", type=int, default=None, help="Maximum number of PDFs to download or verify.")
    download_arxiv.add_argument("--min-confidence", choices=["low", "medium", "high", "exact"], default="high")
    download_arxiv.add_argument("--overwrite", action="store_true")
    download_arxiv.add_argument(
        "--no-set-pdf-path",
        action="store_true",
        help="Do not set record.pdf_path to the arXiv PDF path for downstream parsing.",
    )
    download_arxiv.set_defaults(func=cmd_download_arxiv_pdfs)

    parse = subparsers.add_parser("parse-pdfs", help="Extract paper text and build compact/full representations.")
    parse.add_argument("--manifest", type=Path, required=True)
    parse.add_argument("--out", type=Path, required=True)
    parse.add_argument("--text-dir", type=Path, required=True)
    parse.add_argument("--report", type=Path, default=None)
    parse.add_argument("--limit", type=int, default=None)
    parse.add_argument("--full-char-budget", type=int, default=120_000)
    parse.add_argument("--compact-char-budget", type=int, default=24_000)
    parse.add_argument("--scoring-char-budget", type=int, default=80_000)
    parse.add_argument("--main-paper-max-pages", type=int, default=9)
    parse.add_argument("--keep-references", action="store_true")
    parse.set_defaults(func=cmd_parse_pdfs)

    anonymize = subparsers.add_parser(
        "anonymize-papers",
        help="Create validated, identity-redacted PDF and text derivatives for model scoring.",
    )
    anonymize.add_argument("--manifest", type=Path, required=True)
    anonymize.add_argument("--out", type=Path, required=True)
    anonymize.add_argument("--pdf-dir", type=Path, required=True)
    anonymize.add_argument("--text-dir", type=Path, required=True)
    anonymize.add_argument("--report", type=Path, default=None)
    anonymize.add_argument("--max-pages", type=int, default=9)
    anonymize.add_argument("--limit", type=int, default=None)
    anonymize.add_argument("--overwrite", action="store_true")
    anonymize.set_defaults(func=cmd_anonymize_papers)

    score = subparsers.add_parser("score-pass1", help="Run or dry-run first-pass executive AC scoring.")
    add_http_args(score, default_timeout=180.0)
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    score.add_argument("--run-dir", type=Path, required=True)
    score.add_argument("--provider", choices=["openrouter", "openai"], default="openrouter")
    score.add_argument("--model", default=DEFAULT_CHEAP_MODEL)
    score.add_argument("--prompt-version", default=PROMPT_VERSION)
    score.add_argument("--text-source", choices=["scoring", "full", "compact"], default="scoring")
    score.add_argument("--limit", type=int, default=None)
    score.add_argument("--paper-id", action="append", default=None, help="Restrict scoring to one paper id. Can be repeated.")
    score.add_argument("--temperature", type=float, default=0.2)
    score.add_argument("--max-output-tokens", type=int, default=2400)
    score.add_argument("--seed", type=int, default=None)
    score.add_argument("--dry-run", action="store_true")
    score.add_argument("--input-cost-per-mtok", type=float, default=None)
    score.add_argument("--output-cost-per-mtok", type=float, default=None)
    score.add_argument("--include-needs-review", action="store_true")
    score.set_defaults(func=cmd_score_pass1)

    score_batch = subparsers.add_parser(
        "score-pass1-batch",
        help="Run or dry-run listwise first-pass cheap-model triage with forced ranks and buckets.",
    )
    add_http_args(score_batch, default_timeout=180.0)
    score_batch.add_argument("--manifest", type=Path, required=True)
    score_batch.add_argument("--out", type=Path, required=True)
    score_batch.add_argument("--run-dir", type=Path, required=True)
    score_batch.add_argument("--provider", choices=["openrouter", "openai"], default="openrouter")
    score_batch.add_argument("--model", default=DEFAULT_CHEAP_MODEL)
    score_batch.add_argument("--prompt-version", default=PASS1_BATCH_PROMPT_VERSION)
    score_batch.add_argument("--text-source", choices=["scoring", "full", "compact"], default="scoring")
    score_batch.add_argument("--limit", type=int, default=10)
    score_batch.add_argument("--paper-id", action="append", default=None)
    score_batch.add_argument("--per-paper-char-budget", type=int, default=12_000)
    score_batch.add_argument("--temperature", type=float, default=0.1)
    score_batch.add_argument("--max-output-tokens", type=int, default=6000)
    score_batch.add_argument("--seed", type=int, default=None)
    score_batch.add_argument("--dry-run", action="store_true")
    score_batch.set_defaults(func=cmd_score_pass1_batch)

    score_batches = subparsers.add_parser(
        "score-pass1-batches",
        help="Run first-pass cheap-model ranking across a full paper set in multiple small batches.",
    )
    add_http_args(score_batches, default_timeout=180.0)
    score_batches.add_argument("--manifest", type=Path, required=True)
    score_batches.add_argument("--out", type=Path, required=True)
    score_batches.add_argument("--run-dir", type=Path, required=True)
    score_batches.add_argument("--provider", choices=["openrouter", "openai"], default="openrouter")
    score_batches.add_argument("--model", default=DEFAULT_CHEAP_MODEL)
    score_batches.add_argument("--prompt-version", default=PASS1_BATCH_PROMPT_VERSION)
    score_batches.add_argument("--text-source", choices=["scoring", "full", "compact"], default="scoring")
    score_batches.add_argument("--limit", type=int, default=None)
    score_batches.add_argument("--paper-id", action="append", default=None)
    score_batches.add_argument("--per-paper-char-budget", type=int, default=12_000)
    score_batches.add_argument("--batch-size", type=int, default=8)
    score_batches.add_argument("--partitions", type=int, default=1)
    score_batches.add_argument(
        "--strategy",
        choices=["sequential", "shuffled", "class_round_robin"],
        default="sequential",
    )
    score_batches.add_argument("--class-path", type=Path, default=None)
    score_batches.add_argument(
        "--reasoning-effort",
        default="none",
        help="Optional OpenRouter reasoning effort. Omit for the reasoning-off default.",
    )
    score_batches.add_argument("--temperature", type=float, default=0.1)
    score_batches.add_argument("--max-output-tokens", type=int, default=6000)
    score_batches.add_argument("--seed", type=int, default=None)
    score_batches.add_argument("--dry-run", action="store_true")
    score_batches.set_defaults(func=cmd_score_pass1_batches)

    score_ensemble = subparsers.add_parser(
        "score-pass1-ensemble",
        help="Run first-pass listwise ranking across a reproducible cheap-model ensemble.",
    )
    add_http_args(score_ensemble, default_timeout=180.0)
    score_ensemble.add_argument("--manifest", type=Path, required=True)
    score_ensemble.add_argument("--out-dir", type=Path, required=True)
    score_ensemble.add_argument("--provider", choices=["openrouter", "openai"], default="openrouter")
    score_ensemble.add_argument(
        "--model-preset",
        choices=sorted(CHEAP_MODEL_PRESETS),
        default="production_2026_v2",
        help="Named cheap-model preset. Add --model to append or use --no-preset for only explicit models.",
    )
    score_ensemble.add_argument("--no-preset", action="store_true")
    score_ensemble.add_argument("--model", action="append", default=None, help="Model to include. Can be repeated.")
    score_ensemble.add_argument(
        "--model-workers",
        type=int,
        default=1,
        help="Concurrent model streams. Use 4 only after a bounded provider preflight.",
    )
    score_ensemble.add_argument("--prompt-version", default=PASS1_BATCH_PROMPT_VERSION)
    score_ensemble.add_argument("--text-source", choices=["scoring", "full", "compact"], default="scoring")
    score_ensemble.add_argument("--limit", type=int, default=None)
    score_ensemble.add_argument("--paper-id", action="append", default=None)
    score_ensemble.add_argument("--per-paper-char-budget", type=int, default=12_000)
    score_ensemble.add_argument("--batch-size", type=int, default=8)
    score_ensemble.add_argument("--partitions", type=int, default=1)
    score_ensemble.add_argument(
        "--strategy",
        choices=["sequential", "shuffled", "class_round_robin"],
        default="class_round_robin",
    )
    score_ensemble.add_argument("--class-path", type=Path, default=None)
    score_ensemble.add_argument(
        "--reasoning-effort",
        default="none",
        help="Optional reasoning effort applied to every model in the ensemble.",
    )
    score_ensemble.add_argument("--temperature", type=float, default=0.1)
    score_ensemble.add_argument("--max-output-tokens", type=int, default=6000)
    score_ensemble.add_argument("--seed", type=int, default=None)
    score_ensemble.add_argument("--dry-run", action="store_true")
    score_ensemble.add_argument("--aggregate-out", type=Path, default=None, help="Optional aggregate shortlist/signal JSONL path.")
    score_ensemble.add_argument("--aggregate-report", type=Path, default=None)
    score_ensemble.add_argument("--aggregate-limit", type=int, default=None)
    score_ensemble.add_argument("--aggregate-min-per-class", type=int, default=0)
    score_ensemble.set_defaults(func=cmd_score_pass1_ensemble)

    classify = subparsers.add_parser(
        "classify-contributions",
        help="Classify papers by contribution route using compact title/abstract/intro/conclusion text.",
    )
    add_http_args(classify, default_timeout=180.0)
    classify.add_argument("--manifest", type=Path, required=True)
    classify.add_argument("--out", type=Path, required=True)
    classify.add_argument("--run-dir", type=Path, required=True)
    classify.add_argument("--provider", choices=["openrouter", "openai"], default="openrouter")
    classify.add_argument("--model", default=DEFAULT_CLASSIFICATION_MODEL)
    classify.add_argument("--prompt-version", default=CLASSIFY_PROMPT_VERSION)
    classify.add_argument("--text-source", choices=["compact", "scoring", "full"], default="compact")
    classify.add_argument("--limit", type=int, default=None)
    classify.add_argument("--paper-id", action="append", default=None)
    classify.add_argument("--per-paper-char-budget", type=int, default=12_000)
    classify.add_argument("--batch-size", type=int, default=16)
    classify.add_argument("--temperature", type=float, default=0.0)
    classify.add_argument("--max-output-tokens", type=int, default=4000)
    classify.add_argument("--seed", type=int, default=None)
    classify.add_argument("--dry-run", action="store_true")
    classify.set_defaults(func=cmd_classify_contributions)

    rank = subparsers.add_parser("rank-pass2", help="Run or dry-run strong-model ranking over top first-pass papers.")
    add_http_args(rank, default_timeout=600.0)
    rank.add_argument("--manifest", type=Path, required=True)
    rank.add_argument("--pass1", type=Path, required=True)
    rank.add_argument("--out", type=Path, required=True)
    rank.add_argument("--run-dir", type=Path, required=True)
    rank.add_argument("--provider", choices=["openai", "openrouter"], default="openai")
    rank.add_argument("--model", default=DEFAULT_STRONG_MODEL)
    rank.add_argument("--reasoning-effort", default="high")
    rank.add_argument("--prompt-version", default=PASS2_PROMPT_VERSION)
    rank.add_argument("--text-source", choices=["scoring", "full", "compact"], default="scoring")
    rank.add_argument("--top-fraction", type=float, default=0.25)
    rank.add_argument("--limit", type=int, default=None)
    rank.add_argument("--per-paper-char-budget", type=int, default=45_000)
    rank.add_argument("--temperature", type=float, default=0.1)
    rank.add_argument("--max-output-tokens", type=int, default=6000)
    rank.add_argument("--seed", type=int, default=None)
    rank.add_argument("--dry-run", action="store_true")
    rank.set_defaults(func=cmd_rank_pass2)

    rank_batches = subparsers.add_parser(
        "rank-pass2-batches",
        help="Run a strong semifinal judge in deterministic, resumable listwise batches.",
    )
    add_http_args(rank_batches, default_timeout=600.0)
    rank_batches.add_argument("--manifest", type=Path, required=True)
    rank_batches.add_argument("--pass1", type=Path, required=True)
    rank_batches.add_argument("--out", type=Path, required=True)
    rank_batches.add_argument("--run-dir", type=Path, required=True)
    rank_batches.add_argument("--provider", choices=["openai", "openrouter"], default="openai")
    rank_batches.add_argument("--model", default=DEFAULT_STRONG_MODEL)
    rank_batches.add_argument(
        "--openrouter-provider-only",
        action="append",
        default=None,
        help="Restrict an OpenRouter retry to these provider endpoint slugs.",
    )
    rank_batches.add_argument("--reasoning-effort", default="high")
    rank_batches.add_argument("--prompt-version", default=PASS2_BATCHED_PROMPT_VERSION)
    rank_batches.add_argument("--text-source", choices=["scoring", "full", "compact"], default="scoring")
    rank_batches.add_argument("--top-fraction", type=float, default=0.25)
    rank_batches.add_argument("--limit", type=int, default=None)
    rank_batches.add_argument("--per-paper-char-budget", type=int, default=24_000)
    rank_batches.add_argument("--batch-size", type=int, default=8)
    rank_batches.add_argument("--partitions", type=int, default=2)
    rank_batches.add_argument(
        "--strategy",
        choices=["sequential", "shuffled", "class_round_robin"],
        default="class_round_robin",
    )
    rank_batches.add_argument("--class-path", type=Path, default=None)
    rank_batches.add_argument("--temperature", type=float, default=0.1)
    rank_batches.add_argument("--max-output-tokens", type=int, default=12_000)
    rank_batches.add_argument("--seed", type=int, default=17)
    rank_batches.add_argument("--dry-run", action="store_true")
    rank_batches.set_defaults(func=cmd_rank_pass2_batches)

    rank_two_stage = subparsers.add_parser(
        "rank-pass2-two-stage",
        help="Run a strong-model semifinal rank followed by a smaller final ranking.",
    )
    add_http_args(rank_two_stage, default_timeout=600.0)
    rank_two_stage.add_argument("--manifest", type=Path, required=True)
    rank_two_stage.add_argument("--pass1", type=Path, required=True)
    rank_two_stage.add_argument("--out", type=Path, required=True)
    rank_two_stage.add_argument("--run-dir", type=Path, required=True)
    rank_two_stage.add_argument("--provider", choices=["openai", "openrouter"], default="openai")
    rank_two_stage.add_argument("--model", default=DEFAULT_STRONG_MODEL)
    rank_two_stage.add_argument("--reasoning-effort", default="high")
    rank_two_stage.add_argument("--stage1-prompt-version", default=PASS2_STAGE1_PROMPT_VERSION)
    rank_two_stage.add_argument("--stage2-prompt-version", default=PASS2_FINAL_PROMPT_VERSION)
    rank_two_stage.add_argument("--text-source", choices=["scoring", "full", "compact"], default="scoring")
    rank_two_stage.add_argument("--stage1-limit", type=int, default=None)
    rank_two_stage.add_argument("--stage2-limit", type=int, default=25)
    rank_two_stage.add_argument("--stage1-per-paper-char-budget", type=int, default=32_000)
    rank_two_stage.add_argument("--stage2-per-paper-char-budget", type=int, default=45_000)
    rank_two_stage.add_argument("--temperature", type=float, default=0.1)
    rank_two_stage.add_argument("--stage1-max-output-tokens", type=int, default=8000)
    rank_two_stage.add_argument("--stage2-max-output-tokens", type=int, default=7000)
    rank_two_stage.add_argument("--seed", type=int, default=None)
    rank_two_stage.add_argument("--dry-run", action="store_true")
    rank_two_stage.set_defaults(func=cmd_rank_pass2_two_stage)

    reference = subparsers.add_parser(
        "rank-reference",
        help="Run or dry-run an independent strong-model reference ranking over a paper set.",
    )
    add_http_args(reference, default_timeout=600.0)
    reference.add_argument("--manifest", type=Path, required=True)
    reference.add_argument("--out", type=Path, required=True)
    reference.add_argument("--run-dir", type=Path, required=True)
    reference.add_argument("--provider", choices=["openai", "openrouter"], default="openai")
    reference.add_argument("--model", default="gpt-5.6-sol")
    reference.add_argument("--reasoning-effort", default="xhigh")
    reference.add_argument("--prompt-version", default=REFERENCE_PROMPT_VERSION)
    reference.add_argument("--paper-set-name", default="icml_2025_accepted_50")
    reference.add_argument("--text-source", choices=["scoring", "full", "compact"], default="scoring")
    reference.add_argument("--limit", type=int, default=None)
    reference.add_argument("--paper-id", action="append", default=None)
    reference.add_argument("--per-paper-char-budget", type=int, default=24_000)
    reference.add_argument("--temperature", type=float, default=0.1)
    reference.add_argument("--max-output-tokens", type=int, default=20_000)
    reference.add_argument("--seed", type=int, default=None)
    reference.add_argument("--dry-run", action="store_true")
    reference.set_defaults(func=cmd_rank_reference)

    reference_repair = subparsers.add_parser(
        "repair-reference",
        help="Repair a structurally invalid strong-model reference ranking.",
    )
    add_http_args(reference_repair, default_timeout=600.0)
    reference_repair.add_argument("--manifest", type=Path, required=True)
    reference_repair.add_argument("--reference", type=Path, required=True)
    reference_repair.add_argument("--out", type=Path, required=True)
    reference_repair.add_argument("--run-dir", type=Path, required=True)
    reference_repair.add_argument("--provider", choices=["openai", "openrouter"], default="openai")
    reference_repair.add_argument("--model", default="gpt-5.6-sol")
    reference_repair.add_argument("--reasoning-effort", default="xhigh")
    reference_repair.add_argument("--prompt-version", default=REFERENCE_REPAIR_PROMPT_VERSION)
    reference_repair.add_argument("--paper-set-name", default="icml_2025_accepted_50")
    reference_repair.add_argument("--text-source", choices=["scoring", "full", "compact"], default="scoring")
    reference_repair.add_argument("--per-paper-char-budget", type=int, default=24_000)
    reference_repair.add_argument("--temperature", type=float, default=0.1)
    reference_repair.add_argument("--max-output-tokens", type=int, default=20_000)
    reference_repair.add_argument("--seed", type=int, default=None)
    reference_repair.add_argument("--dry-run", action="store_true")
    reference_repair.set_defaults(func=cmd_repair_reference)

    frontier_cards = subparsers.add_parser(
        "rank-frontier-pdf-cards",
        help="Run PDF-aware frontier judge cards over a paper set, one paper per resumable call.",
    )
    add_http_args(frontier_cards, default_timeout=600.0)
    frontier_cards.add_argument("--manifest", type=Path, required=True)
    frontier_cards.add_argument("--out", type=Path, required=True)
    frontier_cards.add_argument("--run-dir", type=Path, required=True)
    frontier_cards.add_argument("--provider", choices=["openai", "openrouter"], default="openrouter")
    frontier_cards.add_argument("--model", required=True)
    frontier_cards.add_argument("--reasoning-effort", default=None)
    frontier_cards.add_argument("--prompt-version", default=FRONTIER_CARD_PROMPT_VERSION)
    frontier_cards.add_argument("--paper-set-name", default="icml_2026")
    frontier_cards.add_argument("--text-source", choices=["scoring", "full", "compact"], default="scoring")
    frontier_cards.add_argument("--limit", type=int, default=None)
    frontier_cards.add_argument("--paper-id", action="append", default=None)
    frontier_cards.add_argument("--paper-list", type=Path, default=None, help="Ranked JSON/JSONL file of paper_ids to score.")
    frontier_cards.add_argument("--per-paper-char-budget", type=int, default=45_000)
    frontier_cards.add_argument("--pdf-excerpt-pages", type=int, default=9)
    frontier_cards.add_argument(
        "--pdf-excerpt-dir",
        type=Path,
        default=Path("data/pdf_excerpts/icml_2026_first9"),
    )
    frontier_cards.add_argument(
        "--pdf-optimize-threshold-bytes",
        type=int,
        default=4_000_000,
        help="Compress first-page PDF excerpts larger than this before upload. Use 0 to disable.",
    )
    frontier_cards.add_argument("--pdf-settings", default="/ebook", help="Ghostscript PDFSETTINGS value for oversized excerpts.")
    frontier_cards.add_argument("--temperature", type=float, default=0.0)
    frontier_cards.add_argument("--max-output-tokens", type=int, default=12_000)
    frontier_cards.add_argument("--seed", type=int, default=None)
    frontier_cards.add_argument("--dry-run", action="store_true")
    frontier_cards.add_argument("--overwrite", action="store_true")
    frontier_cards.add_argument(
        "--openrouter-pdf-engine",
        default="native",
        help="OpenRouter file-parser PDF engine. Use native for models with file input.",
    )
    frontier_cards.add_argument(
        "--fallback-model",
        default=None,
        help="Optional model used only when the primary card fails or is structurally invalid.",
    )
    frontier_cards.add_argument("--fallback-reasoning-effort", default=None)
    frontier_cards.set_defaults(func=cmd_rank_frontier_pdf_cards)

    frontier_ensemble = subparsers.add_parser(
        "rank-frontier-card-ensemble",
        help="Aggregate multiple PDF-aware frontier card JSONL files into a reference ranking prior.",
    )
    frontier_ensemble.add_argument("--cards", type=Path, required=True, action="append")
    frontier_ensemble.add_argument(
        "--label",
        action="append",
        default=None,
        help="Stable judge-stream label. Supply once per --cards path in the same order.",
    )
    frontier_ensemble.add_argument("--out", type=Path, required=True)
    frontier_ensemble.add_argument("--paper-set-name", default="icml_2025_accepted_50")
    frontier_ensemble.add_argument("--prompt-version", default=FRONTIER_CARD_ENSEMBLE_VERSION)
    frontier_ensemble.set_defaults(func=cmd_rank_frontier_card_ensemble)

    frontier_synthesis = subparsers.add_parser(
        "rank-frontier-card-synthesis",
        help="Synthesize multiple PDF-aware frontier card sets into a forced reference ranking with a strong model.",
    )
    add_http_args(frontier_synthesis, default_timeout=600.0)
    frontier_synthesis.add_argument("--cards", type=Path, required=True, action="append")
    frontier_synthesis.add_argument("--out", type=Path, required=True)
    frontier_synthesis.add_argument("--run-dir", type=Path, required=True)
    frontier_synthesis.add_argument("--provider", choices=["openai", "openrouter"], default="openrouter")
    frontier_synthesis.add_argument("--model", default=DEFAULT_FRONTIER_MODEL)
    frontier_synthesis.add_argument("--reasoning-effort", default="xhigh")
    frontier_synthesis.add_argument("--paper-set-name", default="icml_2025_accepted_50")
    frontier_synthesis.add_argument("--prompt-version", default=FRONTIER_SYNTHESIS_PROMPT_VERSION)
    frontier_synthesis.add_argument("--temperature", type=float, default=0.0)
    frontier_synthesis.add_argument("--max-output-tokens", type=int, default=22000)
    frontier_synthesis.add_argument("--seed", type=int, default=None)
    frontier_synthesis.add_argument("--dry-run", action="store_true")
    frontier_synthesis.set_defaults(func=cmd_rank_frontier_card_synthesis)

    frontier_tournament = subparsers.add_parser(
        "rank-frontier-card-tournament",
        help="Run a resumable pairwise tournament over PDF-aware frontier cards.",
    )
    add_http_args(frontier_tournament, default_timeout=600.0)
    frontier_tournament.add_argument("--cards", type=Path, required=True, action="append")
    frontier_tournament.add_argument("--seed-ranking", type=Path, required=True)
    frontier_tournament.add_argument("--out", type=Path, required=True)
    frontier_tournament.add_argument("--run-dir", type=Path, required=True)
    frontier_tournament.add_argument("--provider", choices=["openai", "openrouter"], default="openrouter")
    frontier_tournament.add_argument("--model", default=DEFAULT_FRONTIER_MODEL)
    frontier_tournament.add_argument("--reasoning-effort", default="xhigh")
    frontier_tournament.add_argument("--paper-set-name", default="icml_2025_accepted_50")
    frontier_tournament.add_argument("--prompt-version", default=FRONTIER_TOURNAMENT_PROMPT_VERSION)
    frontier_tournament.add_argument(
        "--strategy",
        choices=["all_pairs", "swiss", "swiss_playoff"],
        default="all_pairs",
        help="Tournament schedule. Use swiss_playoff for broad cheap rerank plus dense final playoff.",
    )
    frontier_tournament.add_argument("--top-n", type=int, default=30)
    frontier_tournament.add_argument("--pairs-per-batch", type=int, default=25)
    frontier_tournament.add_argument("--swiss-rounds", type=int, default=10)
    frontier_tournament.add_argument("--playoff-top-n", type=int, default=None)
    frontier_tournament.add_argument("--temperature", type=float, default=0.0)
    frontier_tournament.add_argument("--max-output-tokens", type=int, default=16000)
    frontier_tournament.add_argument("--seed", type=int, default=None)
    frontier_tournament.add_argument("--dry-run", action="store_true")
    frontier_tournament.add_argument("--overwrite", action="store_true")
    frontier_tournament.set_defaults(func=cmd_rank_frontier_card_tournament)

    bradley_terry = subparsers.add_parser(
        "rank-bradley-terry",
        help="Fit a Bradley-Terry model over a frontier tournament's pairwise matches.",
    )
    bradley_terry.add_argument(
        "--tournament",
        type=Path,
        required=True,
        help="Tournament output JSON with a `matches` array (from rank-frontier-card-tournament).",
    )
    bradley_terry.add_argument("--out", type=Path, required=True)
    bradley_terry.add_argument(
        "--weight",
        choices=list(WEIGHT_MODES),
        default="none",
        help="Comparison weighting: `none` (unit votes) or `confidence` (judge confidence per match).",
    )
    bradley_terry.add_argument(
        "--prior-strength",
        type=float,
        default=1.0,
        help="Positive virtual wins/losses vs a reference paper; keeps undefeated/winless estimates finite.",
    )
    bradley_terry.add_argument(
        "--stage",
        choices=list(STAGE_MODES),
        default="auto",
        help="Match stage to fit. Auto uses playoff for hybrid tournaments, Swiss for Swiss-only, and all otherwise.",
    )
    bradley_terry.add_argument("--max-iter", type=int, default=5000)
    bradley_terry.add_argument("--tol", type=float, default=1e-9)
    bradley_terry.set_defaults(func=cmd_rank_bradley_terry)

    tournament_simulation = subparsers.add_parser(
        "simulate-hybrid-tournament",
        help="Replay a Swiss-plus-playoff schedule from a complete all-pairs artifact.",
    )
    tournament_simulation.add_argument("--tournament", type=Path, required=True)
    tournament_simulation.add_argument("--out", type=Path, required=True)
    tournament_simulation.add_argument("--swiss-rounds", type=int, default=10)
    tournament_simulation.add_argument("--playoff-top-n", type=int, required=True)
    tournament_simulation.set_defaults(func=cmd_simulate_hybrid_tournament)

    shortlist = subparsers.add_parser("build-shortlist", help="Aggregate score rows into a unique ranked shortlist JSONL.")
    shortlist.add_argument("--scores", type=Path, required=True, action="append")
    shortlist.add_argument("--out", type=Path, required=True)
    shortlist.add_argument("--report", type=Path, default=None)
    shortlist.add_argument("--limit", type=int, default=None)
    shortlist.add_argument("--min-per-class", type=int, default=0)
    shortlist.add_argument("--class-path", type=Path, default=None)
    shortlist.set_defaults(func=cmd_build_shortlist)

    position_shortlist = subparsers.add_parser(
        "build-position-robust-shortlist",
        help="Union raw and within-paper position-adjusted first-pass leaders.",
    )
    position_shortlist.add_argument("--scores", type=Path, required=True, action="append")
    position_shortlist.add_argument("--raw-ranking", type=Path, required=True)
    position_shortlist.add_argument("--class-path", type=Path, default=None)
    position_shortlist.add_argument("--cutoff", type=int, required=True)
    position_shortlist.add_argument("--adjusted-out", type=Path, required=True)
    position_shortlist.add_argument("--out", type=Path, required=True)
    position_shortlist.add_argument("--report", type=Path, required=True)
    position_shortlist.set_defaults(func=cmd_build_position_robust_shortlist)

    semifinal_ensemble = subparsers.add_parser(
        "ensemble-semifinal-rankings",
        help="Aggregate complete independent semifinal rankings with equal-weight normalized ranks.",
    )
    semifinal_ensemble.add_argument("--ranking", type=Path, required=True, action="append")
    semifinal_ensemble.add_argument(
        "--label",
        required=False,
        action="append",
        help="Stable judge label. When used, supply once per --ranking in the same order.",
    )
    semifinal_ensemble.add_argument("--out", type=Path, required=True)
    semifinal_ensemble.add_argument("--report", type=Path, default=None)
    semifinal_ensemble.set_defaults(func=cmd_ensemble_semifinal_rankings)

    semifinal_audit = subparsers.add_parser(
        "audit-semifinal-rankings",
        help="Audit complete strong-semifinal ranking streams before local aggregation.",
    )
    semifinal_audit.add_argument("--ranking", type=Path, required=True, action="append")
    semifinal_audit.add_argument("--label", required=True, action="append")
    semifinal_audit.add_argument("--expected-model", required=True, action="append")
    semifinal_audit.add_argument("--shortlist", type=Path, required=True)
    semifinal_audit.add_argument("--manifest", type=Path, required=True)
    semifinal_audit.add_argument("--out", type=Path, required=True)
    semifinal_audit.set_defaults(func=cmd_audit_semifinal_rankings)

    finalists = subparsers.add_parser(
        "select-finalists",
        help="Select a conservative finalist pool from cheap-ensemble and strong-semifinal rankings.",
    )
    finalists.add_argument("--cheap", type=Path, required=True)
    finalists.add_argument("--semifinal", type=Path, required=True)
    finalists.add_argument("--out", type=Path, required=True)
    finalists.add_argument("--report", type=Path, default=None)
    finalists.add_argument("--limit", type=int, required=True)
    finalists.add_argument(
        "--semifinal-top",
        type=int,
        default=None,
        help="Always include this many top semifinal papers. Defaults to 75%% of --limit.",
    )
    finalists.add_argument("--cheap-top", type=int, default=0, help="Always include this many cheap-ensemble leaders.")
    finalists.add_argument("--min-per-class", type=int, default=0)
    finalists.add_argument("--disagreement-saves", type=int, default=0)
    finalists.add_argument(
        "--judge-disagreement-saves",
        type=int,
        default=0,
        help="Retain this many papers with the largest disagreement among semifinal judges.",
    )
    finalists.set_defaults(func=cmd_select_finalists)

    evaluate = subparsers.add_parser("eval-ranking", help="Evaluate a candidate ranking against a reference/gold ranking.")
    evaluate.add_argument("--gold", type=Path, required=True)
    evaluate.add_argument("--candidate", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--k", type=int, action="append", default=None, help="Top-k value to evaluate. Can be repeated.")
    evaluate.set_defaults(func=cmd_eval_ranking)

    divergence = subparsers.add_parser(
        "human-divergence-report",
        help="Build an AI-vs-human divergence gallery (overlooked gems and AI blind spots).",
    )
    divergence.add_argument("--manifest", type=Path, required=True, help="Enriched paper manifest JSONL (human tier/ratings/awards in extra).")
    divergence.add_argument("--ai-scores", type=Path, required=True, help="First-pass AI score JSONL (per-paper `scores` schema).")
    divergence.add_argument(
        "--strong-ranking",
        type=Path,
        default=None,
        help="Optional tournament/Bradley-Terry ranking JSON for finalist strong ranks.",
    )
    divergence.add_argument("--out", type=Path, required=True)
    divergence.add_argument("--markdown", type=Path, default=None, help="Optional path for a human-readable gallery.")
    divergence.add_argument("--human-axis", choices=["tier", "reviewer", "decision"], default="tier")
    divergence.add_argument("--top-frac", type=float, default=0.10)
    divergence.add_argument("--cases-per-direction", type=int, default=20)
    divergence.add_argument("--min-reviews", type=int, default=0)
    divergence.add_argument(
        "--min-coverage",
        type=float,
        default=1.0,
        help="Minimum manifest/AI join fraction; defaults to complete coverage.",
    )
    divergence.set_defaults(func=cmd_human_divergence_report)

    taxonomy_induce = subparsers.add_parser(
        "divergence-taxonomy-induce",
        help="Propose a codebook of AI-vs-human divergence reason codes from the gallery (QL-B stage 1).",
    )
    add_http_args(taxonomy_induce, default_timeout=600.0)
    taxonomy_induce.add_argument("--report", type=Path, required=True, help="human-divergence-report JSON.")
    taxonomy_induce.add_argument("--out", type=Path, required=True, help="Candidate codebook JSON (review/edit before classify).")
    taxonomy_induce.add_argument("--run-dir", type=Path, default=None)
    taxonomy_induce.add_argument("--provider", choices=["openai", "openrouter"], default="openrouter")
    taxonomy_induce.add_argument("--model", default=DEFAULT_FRONTIER_MODEL)
    taxonomy_induce.add_argument("--reasoning-effort", default="high")
    taxonomy_induce.add_argument("--sample-size", type=int, default=40)
    taxonomy_induce.add_argument("--max-codes", type=int, default=12)
    taxonomy_induce.add_argument("--temperature", type=float, default=0.2)
    taxonomy_induce.add_argument("--max-output-tokens", type=int, default=6000)
    taxonomy_induce.add_argument("--seed", type=int, default=None)
    taxonomy_induce.add_argument("--dry-run", action="store_true")
    taxonomy_induce.set_defaults(func=cmd_divergence_taxonomy_induce)

    taxonomy_classify = subparsers.add_parser(
        "divergence-taxonomy-classify",
        help="Classify divergence cases against a frozen codebook and aggregate (QL-B stage 2).",
    )
    add_http_args(taxonomy_classify, default_timeout=600.0)
    taxonomy_classify.add_argument("--report", type=Path, required=True)
    taxonomy_classify.add_argument("--codebook", type=Path, required=True)
    taxonomy_classify.add_argument("--out", type=Path, required=True)
    taxonomy_classify.add_argument("--run-dir", type=Path, default=None)
    taxonomy_classify.add_argument("--provider", choices=["openai", "openrouter"], default="openrouter")
    taxonomy_classify.add_argument("--model", default=DEFAULT_FRONTIER_MODEL)
    taxonomy_classify.add_argument("--reasoning-effort", default="high")
    taxonomy_classify.add_argument("--batch-size", type=int, default=20)
    taxonomy_classify.add_argument("--max-codes-per-case", type=int, default=3)
    taxonomy_classify.add_argument("--temperature", type=float, default=0.0)
    taxonomy_classify.add_argument("--max-output-tokens", type=int, default=8000)
    taxonomy_classify.add_argument("--seed", type=int, default=None)
    taxonomy_classify.add_argument("--second-provider", choices=["openai", "openrouter"], default=None)
    taxonomy_classify.add_argument(
        "--second-model",
        default=None,
        help="Optional independent coder (different family) for per-code Cohen's kappa reliability.",
    )
    taxonomy_classify.add_argument("--second-reasoning-effort", default="high")
    taxonomy_classify.add_argument("--dry-run", action="store_true")
    taxonomy_classify.set_defaults(func=cmd_divergence_taxonomy_classify)

    agreement = subparsers.add_parser(
        "human-agreement-report",
        help="AI-vs-human agreement: recall@k / ROC-AUC for surfacing orals+spotlights, Kendall tau-b, per-tier distributions.",
    )
    agreement.add_argument("--manifest", type=Path, required=True, help="Enriched paper manifest JSONL.")
    agreement.add_argument("--ai-scores", type=Path, required=True, help="First-pass AI score JSONL.")
    agreement.add_argument(
        "--strong-ranking",
        type=Path,
        default=None,
        help="Optional strong ranking JSON for the finalist-subset (layered) view.",
    )
    agreement.add_argument("--out", type=Path, required=True)
    agreement.add_argument("--k", type=int, action="append", default=None, help="Top-k for recall/precision. Repeatable (default 10 20 50 100).")
    agreement.add_argument("--honored-includes-award", action="store_true", help="Count award papers as honored alongside orals/spotlights.")
    agreement.add_argument("--min-coverage", type=float, default=1.0)
    agreement.add_argument("--bootstrap-samples", type=int, default=500)
    agreement.add_argument("--bootstrap-seed", type=int, default=20260725)
    agreement.set_defaults(func=cmd_human_agreement_report)

    axis_decomp = subparsers.add_parser(
        "ai-axis-decomposition",
        help="Rank AI axes by how well they predict human honors (impact-vs-polish decomposition).",
    )
    axis_decomp.add_argument("--manifest", type=Path, required=True)
    axis_decomp.add_argument("--ai-scores", type=Path, required=True)
    axis_decomp.add_argument("--out", type=Path, required=True)
    axis_decomp.add_argument("--honored-includes-award", action="store_true")
    axis_decomp.add_argument("--min-coverage", type=float, default=1.0)
    axis_decomp.set_defaults(func=cmd_ai_axis_decomposition)

    query_papers = subparsers.add_parser(
        "query-ranked-papers",
        help="Search paper metadata and join matches to the frozen ICML 2026 rankings.",
    )
    query_papers.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
    )
    query_papers.add_argument(
        "--cheap-ranking",
        type=Path,
        default=DEFAULT_CHEAP_RANKING_PATH,
    )
    query_papers.add_argument(
        "--final-ranking",
        type=Path,
        default=DEFAULT_FINAL_RANKING_PATH,
    )
    query_papers.add_argument("--preset", choices=QUERY_PRESETS)
    query_papers.add_argument("--query", help="Case-insensitive substring across title, abstract, topic, and contribution class.")
    query_papers.add_argument("--topic", help="Case-insensitive official-topic glob, such as 'Deep Learning->*'.")
    query_papers.add_argument("--contribution-class")
    query_papers.add_argument("--scope", choices=QUERY_SCOPES, default="all")
    query_papers.add_argument("--limit", type=int)
    query_papers.add_argument("--include-abstract", action="store_true")
    query_papers.add_argument("--out", type=Path, help="Optional .csv, .jsonl, or .tsv output path. Defaults to TSV on stdout.")
    query_papers.add_argument("--format", choices=["csv", "jsonl", "tsv"], default=None)
    query_papers.set_defaults(func=cmd_query_ranked_papers)

    top_papers = subparsers.add_parser(
        "top-ranked-papers",
        help="Return the top frozen-ranking papers in a named topic, contribution class, or preset.",
    )
    top_papers.add_argument(
        "--category",
        required=True,
        help="For example: data-pretraining, theory, benchmark_dataset, or 'Deep Learning'.",
    )
    top_papers.add_argument("--top", type=int, default=20)
    top_papers.add_argument("--scope", choices=QUERY_SCOPES, default="finalists")
    top_papers.add_argument("--include-abstract", action="store_true")
    top_papers.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    top_papers.add_argument("--cheap-ranking", type=Path, default=DEFAULT_CHEAP_RANKING_PATH)
    top_papers.add_argument("--final-ranking", type=Path, default=DEFAULT_FINAL_RANKING_PATH)
    top_papers.add_argument("--out", type=Path)
    top_papers.add_argument("--format", choices=["csv", "jsonl", "tsv"], default="jsonl")
    top_papers.set_defaults(func=cmd_top_ranked_papers)

    list_categories = subparsers.add_parser(
        "list-ranking-categories",
        help="List categories accepted by top-ranked-papers.",
    )
    list_categories.set_defaults(func=cmd_list_ranking_categories)

    return parser


def add_http_args(
    parser: argparse.ArgumentParser,
    *,
    default_timeout: float = 30.0,
    default_delay: float = 0.0,
    default_backoff: float = 1.5,
) -> None:
    parser.add_argument("--timeout", type=float, default=default_timeout)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff", type=float, default=default_backoff)
    parser.add_argument("--delay", type=float, default=default_delay, help="Polite delay between HTTP requests in seconds.")


def make_http(args: argparse.Namespace) -> HttpClient:
    return HttpClient(
        timeout_seconds=args.timeout,
        retries=args.retries,
        backoff_seconds=args.backoff,
        polite_delay_seconds=args.delay,
    )


def cmd_crawl_accepted_virtual(args: argparse.Namespace) -> int:
    scraper = ICMLVirtualScraper(year=args.year, index_url=args.index_url, http_client=make_http(args))
    records: list[PaperRecord] = []
    for record in scraper.scrape_records(limit=args.limit, raw_html_dir=args.raw_html_dir):
        if args.download_pdfs and record.pdf_url:
            result = scraper.http.download(record.pdf_url, args.pdf_dir / record.pdf_filename())
            record.pdf_path = str(result.path)
            record.extra["pdf_sha256"] = result.sha256
            record.extra["pdf_bytes"] = result.bytes_written
            record.extra["pdf_download_skipped"] = result.skipped
        records.append(record)
        print(f"[accepted] {len(records)} {record.paper_id} {record.title or ''}", file=sys.stderr)
    count = write_jsonl(args.out, (record.to_dict() for record in records))
    write_json(
        args.out.with_suffix(args.out.suffix + ".run.json"),
        {
            "command": "crawl-accepted-virtual",
            "year": args.year,
            "index_url": scraper.index_url,
            "records": count,
            "download_pdfs": args.download_pdfs,
            "limit": args.limit,
        },
    )
    print(f"Wrote {count} records to {args.out}")
    return 0


def cmd_crawl_accepted_virtual_json(args: argparse.Namespace) -> int:
    started = time.monotonic()
    scraper = ICMLVirtualScraper(year=args.year, http_client=make_http(args))
    records, report = scraper.scrape_json_records(
        events_url=args.events_url,
        abstracts_url=args.abstracts_url,
        event_types=set(args.event_type or ["Poster"]),
        require_decision=args.require_decision,
        limit=args.limit,
    )
    count = write_jsonl(args.out, (record.to_dict() for record in records))
    report_path = args.report or args.out.with_suffix(args.out.suffix + ".report.json")
    run_report = {
        **report,
        "command": "crawl-accepted-virtual-json",
        "out": str(args.out),
        "records": count,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(report_path, run_report)
    write_json(args.out.with_suffix(args.out.suffix + ".run.json"), run_report)
    print(
        "Wrote "
        f"{count} ICML {args.year} metadata records to {args.out} "
        f"(forum_urls={report.get('records_with_forum_url')}, pdf_urls={report.get('records_with_pdf_url')})"
    )
    return 0


def cmd_crawl_accepted_batch(args: argparse.Namespace) -> int:
    started = time.monotonic()
    scraper = ICMLVirtualScraper(year=args.year, index_url=args.index_url, http_client=make_http(args))
    existing_ids = {str(row.get("paper_id")) for row in read_jsonl_if_exists(args.manifest) if row.get("paper_id")}
    existing_at_start = len(existing_ids)
    records_written = 0
    attempts = 0
    pdf_downloaded = 0
    pdf_missing = 0
    failures = 0
    last_error: str | None = None

    stubs = scraper.discover_index()
    for stub in stubs:
        if stub.paper_id in existing_ids:
            continue
        if records_written >= args.limit:
            break
        if args.max_attempts is not None and attempts >= args.max_attempts:
            break
        attempts += 1
        try:
            record = scraper.scrape_detail(stub, raw_html_dir=args.raw_html_dir)
            if args.download_pdfs:
                if record.pdf_url:
                    result = scraper.http.download(record.pdf_url, args.pdf_dir / record.pdf_filename())
                    record.pdf_path = str(result.path)
                    record.extra["pdf_sha256"] = result.sha256
                    record.extra["pdf_bytes"] = result.bytes_written
                    record.extra["pdf_download_skipped"] = result.skipped
                    pdf_downloaded += 0 if result.skipped else 1
                else:
                    pdf_missing += 1
            append_jsonl(args.manifest, [record.to_dict()])
            existing_ids.add(record.paper_id)
            records_written += 1
            print(f"[ok] {records_written}/{args.limit} {record.paper_id} {record.title or ''}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - failure details are persisted for retry.
            failures += 1
            last_error = repr(exc)
            append_jsonl(
                args.failures,
                [
                    {
                        "paper_id": stub.paper_id,
                        "title": stub.title,
                        "detail_url": stub.detail_url,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                        "attempt_index": attempts,
                        "monotonic_elapsed_seconds": round(time.monotonic() - started, 3),
                    }
                ],
            )
            print(f"[fail] {stub.paper_id} {exc!r}", file=sys.stderr)

    elapsed = time.monotonic() - started
    report: dict[str, Any] = {
        "command": "crawl-accepted-batch",
        "year": args.year,
        "index_url": scraper.index_url,
        "manifest": str(args.manifest),
        "failures": str(args.failures),
        "records_written": records_written,
        "detail_attempts": attempts,
        "existing_records_skipped": existing_at_start,
        "pdf_downloaded": pdf_downloaded,
        "pdf_missing": pdf_missing,
        "failure_count": failures,
        "last_error": last_error,
        "limit": args.limit,
        "max_attempts": args.max_attempts,
        "download_pdfs": args.download_pdfs,
        "elapsed_seconds": round(elapsed, 3),
        "seconds_per_successful_record": round(elapsed / records_written, 3) if records_written else None,
    }
    write_json(args.run_report, report)
    print(f"Wrote {records_written} new records to {args.manifest} in {elapsed:.1f}s")
    return 0 if failures == 0 else 2


def cmd_crawl_openreview(args: argparse.Namespace) -> int:
    http = make_http(args)
    client = OpenReviewClient(api_base=args.api_base, http_client=http)
    if args.query_file:
        query = read_query_file(args.query_file)
    elif args.from_venue_status:
        group = client.get_group(args.venue_id)
        if group is None:
            raise RuntimeError(f"OpenReview venue group not found: {args.venue_id}")
        queries = queries_for_status(group, args.from_venue_status)
    else:
        queries = [build_query(
            venue_id=args.venue_id,
            invitation=args.invitation,
            content_venue=args.content_venue,
            details=args.details,
        )]
    if args.query_file:
        queries = [query]
    records_by_id: dict[str, PaperRecord] = {}
    remaining = args.limit
    for query in queries:
        notes = client.iter_notes(query=query, max_notes=remaining)
        for record in client.notes_to_records(notes, source=args.source, decision_label=args.decision_label):
            records_by_id.setdefault(record.paper_id, record)
            if args.limit is not None and len(records_by_id) >= args.limit:
                break
        if args.limit is not None:
            remaining = max(args.limit - len(records_by_id), 0)
            if remaining == 0:
                break
    records = list(records_by_id.values())
    count = write_jsonl(args.out, (record.to_dict() for record in records))
    write_json(
        args.out.with_suffix(args.out.suffix + ".run.json"),
        {
            "command": "crawl-openreview",
            "api_base": args.api_base,
            "queries": queries,
            "source": args.source,
            "decision_label": args.decision_label,
            "records": count,
            "limit": args.limit,
        },
    )
    print(f"Wrote {count} records to {args.out}")
    return 0


def cmd_crawl_openreview_scores(args: argparse.Namespace) -> int:
    started = time.monotonic()
    client = OpenReviewClient(api_base=args.api_base, http_client=make_http(args))
    group = client.get_group(args.venue_id)
    if group is None:
        raise RuntimeError(f"OpenReview venue group not found: {args.venue_id}")
    queries = []
    for query in queries_for_status(group, args.from_venue_status):
        query_with_details = dict(query)
        query_with_details["details"] = "directReplies"
        queries.append(query_with_details)

    summaries_by_id: dict[str, dict[str, Any]] = {}
    remaining = args.limit
    for query in queries:
        for note in client.iter_notes(query=query, max_notes=remaining):
            summary = summarize_openreview_scores(note)
            forum_id = str(summary.get("openreview_forum_id") or "")
            if not forum_id:
                continue
            previous = summaries_by_id.get(forum_id)
            if previous is None or int(summary.get("review_count") or 0) > int(previous.get("review_count") or 0):
                summaries_by_id[forum_id] = summary
            if args.limit is not None and len(summaries_by_id) >= args.limit:
                break
        if args.limit is not None:
            remaining = max(args.limit - len(summaries_by_id), 0)
            if remaining == 0:
                break

    summaries = list(summaries_by_id.values())
    count = write_jsonl(args.out, summaries)
    report_path = args.report or args.out.with_suffix(args.out.suffix + ".report.json")
    write_json(
        report_path,
        {
            "command": "crawl-openreview-scores",
            "venue_id": args.venue_id,
            "status": args.from_venue_status,
            "queries": queries,
            "rows": count,
            "with_reviews": sum(bool(row.get("review_count")) for row in summaries),
            "with_overall": sum(row.get("overall_mean") is not None for row in summaries),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
    )
    print(f"Wrote {count} OpenReview score summaries to {args.out}")
    return 0


def cmd_resolve_openreview_pdfs(args: argparse.Namespace) -> int:
    started = time.monotonic()
    client = OpenReviewClient(api_base=args.api_base, http_client=make_http(args))
    if args.use_openreview_auth:
        client.authenticate_from_env()
    accepted_venues = set(args.accepted_venue)
    journal_path = args.out.with_suffix(args.out.suffix + ".partial.jsonl")
    if args.no_resume:
        write_jsonl(journal_path, [])
        resume_rows: dict[str, dict[str, Any]] = {}
    else:
        resume_rows = _latest_rows_by_paper_id(args.out)
        resume_rows.update(_latest_rows_by_paper_id(journal_path))

    progress_rows: dict[str, dict[str, Any]] = {}
    searched = 0
    title_searches = 0
    resolved = 0
    resolved_by_forum_pdf = 0
    resolved_by_title_search = 0
    downloaded = 0
    download_failed = 0
    skipped_downloads = 0
    resumed_completed = 0
    blocked_reason: str | None = None
    records = list(read_paper_records(args.manifest))
    if args.limit is not None:
        records = records[: args.limit]

    preseeded_local = 0
    if args.download_pdfs:
        for record in records:
            previous = resume_rows.get(record.paper_id)
            if previous and _openreview_resolution_complete(previous, download_pdfs=True):
                continue
            local_state = _existing_openreview_pdf_state(record, previous=previous, pdf_dir=args.pdf_dir)
            if local_state:
                resume_rows[record.paper_id] = local_state
                preseeded_local += 1

    for index, record in enumerate(records, start=1):
        previous = resume_rows.get(record.paper_id)
        if previous and _openreview_resolution_complete(previous, download_pdfs=args.download_pdfs):
            progress_rows[record.paper_id] = _merge_openreview_pdf_state(record.to_dict(), previous)
            resumed_completed += 1
            continue

        row = _merge_openreview_pdf_state(record.to_dict(), previous) if previous else record.to_dict()
        row["extra"] = dict(row.get("extra")) if isinstance(row.get("extra"), dict) else {}
        title = record.title or ""
        searched += 1
        match: PaperRecord | None = None
        try:
            direct_forum_id = (
                str(record.extra.get("openreview_forum_id") or "")
                or forum_id_from_url(record.forum_url)
                or forum_id_from_url(row.get("forum_url"))
            )
            direct_pdf_url = (
                client.pdf_url_for_forum_id(direct_forum_id)
                if direct_forum_id and client.authenticated
                else pdf_url_for_forum_id(direct_forum_id)
            )
            matches: list[PaperRecord] = []
            if direct_forum_id and direct_pdf_url:
                row["forum_url"] = row.get("forum_url") or f"https://openreview.net/forum?id={direct_forum_id}"
                row["pdf_url"] = direct_pdf_url
                row["extra"]["pdf_discovery_status"] = "found_openreview_official_forum_pdf"
                row["extra"]["openreview_forum_id"] = direct_forum_id
                resolved_by_forum_pdf += 1
                match = PaperRecord(
                    paper_id=record.paper_id,
                    source=record.source,
                    decision_label=record.decision_label,
                    title=record.title,
                    abstract=record.abstract,
                    authors=record.authors,
                    forum_url=row["forum_url"],
                    pdf_url=direct_pdf_url,
                    extra={"openreview_forum_id": direct_forum_id},
                )
            elif title:
                title_searches += 1
                for note in client.search_notes(title=title, limit=args.search_limit):
                    content = note.get("content") if isinstance(note.get("content"), dict) else {}
                    note_title = content_value(content, "title")
                    venue = content_value(content, "venue")
                    candidate = note_to_record(note, source=record.source, decision_label=venue)
                    if note_title == title and venue in accepted_venues and candidate.pdf_url:
                        if client.authenticated:
                            candidate.pdf_url = client.pdf_url_for_forum_id(candidate.paper_id)
                        matches.append(candidate)
                if matches:
                    match = matches[0]
                    row["forum_url"] = match.forum_url
                    row["pdf_url"] = match.pdf_url
                    row["decision_label"] = match.decision_label or row.get("decision_label")
                    row["extra"]["pdf_discovery_status"] = "found_openreview_official_title_match"
                    row["extra"]["openreview_match"] = match.extra
                    row["extra"]["openreview_match_count"] = len(matches)
                    resolved_by_title_search += 1
            if match and match.pdf_url:
                resolved += 1
                row["extra"]["openreview_authenticated"] = client.authenticated
                row["extra"]["pdf_path_source"] = "openreview_official"
                row["extra"]["source_pdf_kind"] = "openreview_official"
                if args.download_pdfs:
                    result = client.http.download(match.pdf_url, args.pdf_dir / record.pdf_filename())
                    row["pdf_path"] = str(result.path)
                    row["extra"]["pdf_sha256"] = result.sha256
                    row["extra"]["pdf_bytes"] = result.bytes_written
                    row["extra"]["pdf_download_skipped"] = result.skipped
                    row["extra"]["pdf_download_status"] = "ok"
                    row["extra"].pop("pdf_download_error", None)
                    if result.skipped:
                        skipped_downloads += 1
                    else:
                        downloaded += 1
            else:
                reason = (
                    "no OpenReview forum id and no exact-title OpenReview result with accepted ICML 2026 venue label"
                    if title
                    else "no OpenReview forum id and no title available for OpenReview title search"
                )
                row["extra"]["pdf_discovery_status"] = "not_found_in_openreview_official_title_search"
                row["extra"]["pdf_resolution_error"] = reason
                print(f"[unresolved] {record.paper_id} {title}", file=sys.stderr)
        except AccessChallengeError as exc:
            blocked_reason = str(exc)
            row["extra"]["pdf_download_status"] = "blocked_access_challenge"
            row["extra"]["pdf_download_error"] = blocked_reason
            progress_rows[record.paper_id] = row
            append_jsonl(journal_path, [row])
            print(f"[paused] {blocked_reason}", file=sys.stderr)
            break
        except urllib.error.HTTPError as exc:
            halt_reason = _openreview_halt_reason(exc)
            if halt_reason:
                blocked_reason = halt_reason
                row["extra"]["pdf_download_status"] = "blocked_http_status"
                row["extra"]["pdf_download_error"] = blocked_reason
                progress_rows[record.paper_id] = row
                append_jsonl(journal_path, [row])
                print(f"[paused] {blocked_reason}", file=sys.stderr)
                break
            download_failed += 1
            row["extra"]["pdf_download_status"] = "failed"
            row["extra"]["pdf_download_error"] = repr(exc)
            print(f"[pdf-fail] {record.paper_id} {exc!r}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - preserve row state and retry on a later invocation.
            download_failed += 1
            row["extra"]["pdf_download_status"] = "failed"
            row["extra"]["pdf_download_error"] = repr(exc)
            print(f"[pdf-fail] {record.paper_id} {exc!r}", file=sys.stderr)

        progress_rows[record.paper_id] = row
        append_jsonl(journal_path, [row])
        if args.progress_every > 0 and index % args.progress_every == 0:
            print(
                f"[progress] {index}/{len(records)} downloaded={downloaded} "
                f"resumed={resumed_completed} failed={download_failed}",
                file=sys.stderr,
            )

    snapshot: list[dict[str, Any]] = []
    for record in records:
        previous = progress_rows.get(record.paper_id) or resume_rows.get(record.paper_id)
        snapshot.append(_merge_openreview_pdf_state(record.to_dict(), previous) if previous else record.to_dict())

    unresolved = []
    for row in snapshot:
        extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
        if extra.get("pdf_discovery_status") == "not_found_in_openreview_official_title_search":
            unresolved.append(
                {
                    "paper_id": row.get("paper_id"),
                    "title": row.get("title"),
                    "forum_url": row.get("forum_url"),
                    "reason": extra.get("pdf_resolution_error"),
                }
            )

    count = write_jsonl(args.out, snapshot)
    write_jsonl(args.unresolved, unresolved)
    completed = sum(_openreview_resolution_complete(row, download_pdfs=args.download_pdfs) for row in snapshot)
    report = {
        "command": "resolve-openreview-pdfs",
        "manifest": str(args.manifest),
        "out": str(args.out),
        "journal": str(journal_path),
        "unresolved": str(args.unresolved),
        "records": count,
        "searched": searched,
        "title_searches": title_searches,
        "resolved": resolved,
        "resolved_by_forum_pdf": resolved_by_forum_pdf,
        "resolved_by_title_search": resolved_by_title_search,
        "unresolved_count": len(unresolved),
        "download_pdfs": args.download_pdfs,
        "authenticated": client.authenticated,
        "downloaded": downloaded,
        "download_failed": download_failed,
        "skipped_downloads": skipped_downloads,
        "preseeded_local": preseeded_local,
        "resumed_completed": resumed_completed,
        "completed": completed,
        "pending": len(snapshot) - completed,
        "blocked_reason": blocked_reason,
        "accepted_venues": sorted(accepted_venues),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(args.out.with_suffix(args.out.suffix + ".run.json"), report)
    print(
        f"OpenReview PDF state: complete={completed}/{len(snapshot)} "
        f"downloaded_now={downloaded} pending={len(snapshot) - completed}"
    )
    if blocked_reason:
        return 4
    return 0 if completed else 3


def _merge_openreview_pdf_state(base: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    if not previous:
        return base
    merged = dict(base)
    for key in ("forum_url", "pdf_url", "pdf_path"):
        if previous.get(key):
            merged[key] = previous[key]
    extra = dict(base.get("extra")) if isinstance(base.get("extra"), dict) else {}
    previous_extra = previous.get("extra") if isinstance(previous.get("extra"), dict) else {}
    for key, value in previous_extra.items():
        if key.startswith("pdf_") or key.startswith("openreview_") or key == "source_pdf_kind":
            extra[key] = value
    merged["extra"] = extra
    return merged


def _openreview_halt_reason(exc: urllib.error.HTTPError) -> str | None:
    reasons = {
        401: "authentication rejected or token expired",
        403: "access forbidden",
        429: "rate limited",
    }
    reason = reasons.get(exc.code)
    return f"OpenReview HTTP {exc.code}: {reason}; batch halted without further requests" if reason else None


def _openreview_resolution_complete(row: dict[str, Any], *, download_pdfs: bool) -> bool:
    if not row.get("pdf_url"):
        return False
    if not download_pdfs:
        return True
    pdf_path = row.get("pdf_path")
    if not isinstance(pdf_path, str) or not pdf_path:
        return False
    return is_pdf_file(Path(pdf_path))


def _existing_openreview_pdf_state(
    record: PaperRecord,
    *,
    previous: dict[str, Any] | None,
    pdf_dir: Path,
) -> dict[str, Any] | None:
    path = pdf_dir / record.pdf_filename()
    if not is_pdf_file(path):
        return None

    row = _merge_openreview_pdf_state(record.to_dict(), previous)
    row["extra"] = dict(row.get("extra")) if isinstance(row.get("extra"), dict) else {}
    forum_id = (
        str(record.extra.get("openreview_forum_id") or "")
        or forum_id_from_url(record.forum_url)
        or forum_id_from_url(row.get("forum_url"))
    )
    pdf_url = row.get("pdf_url") or pdf_url_for_forum_id(forum_id)
    if not pdf_url:
        return None

    row["pdf_url"] = pdf_url
    row["pdf_path"] = str(path)
    row["extra"]["pdf_path_source"] = "openreview_official"
    row["extra"]["source_pdf_kind"] = "openreview_official"
    if forum_id:
        row["forum_url"] = row.get("forum_url") or f"https://openreview.net/forum?id={forum_id}"
        row["extra"]["openreview_forum_id"] = forum_id
        row["extra"]["pdf_discovery_status"] = "found_openreview_official_forum_pdf"
    row["extra"]["pdf_sha256"] = sha256_file(path)
    row["extra"]["pdf_bytes"] = path.stat().st_size
    row["extra"]["pdf_download_skipped"] = True
    row["extra"]["pdf_download_status"] = "ok"
    row["extra"]["pdf_recovery_source"] = "existing_local_file"
    row["extra"].pop("pdf_download_error", None)
    return row


def _arxiv_journal_path(out: Path) -> Path:
    return out.with_suffix(out.suffix + ".partial.jsonl")


def _latest_rows_by_paper_id(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl_if_exists(path):
        paper_id = row.get("paper_id")
        if isinstance(paper_id, str) and paper_id:
            latest[paper_id] = row
    return latest


def _arxiv_resolution_run_ids(row: dict[str, Any]) -> set[str]:
    extra = row.get("extra")
    if not isinstance(extra, dict):
        return set()
    run_ids = {
        value
        for value in extra.get("arxiv_resolution_run_ids", [])
        if isinstance(value, str) and value
    }
    current = extra.get("arxiv_resolution_run_id")
    if isinstance(current, str) and current:
        run_ids.add(current)
    return run_ids


def _arxiv_resolution_history(*paths: Path) -> dict[str, set[str]]:
    history: dict[str, set[str]] = {}
    for path in paths:
        for row in read_jsonl_if_exists(path):
            paper_id = row.get("paper_id")
            if not isinstance(paper_id, str) or not paper_id:
                continue
            history.setdefault(paper_id, set()).update(_arxiv_resolution_run_ids(row))
    return history


def _arxiv_resolution_status(row: dict[str, Any]) -> str:
    extra = row.get("extra")
    if not isinstance(extra, dict):
        return ""
    status = extra.get("arxiv_resolution_status")
    return status if isinstance(status, str) else ""


def _arxiv_match_confidence(row: dict[str, Any]) -> str:
    extra = row.get("extra")
    if not isinstance(extra, dict):
        return ""
    confidence = extra.get("arxiv_match_confidence")
    return confidence if isinstance(confidence, str) else ""


def _arxiv_row_needs_retry(
    row: dict[str, Any],
    *,
    retry_failed: bool,
    retry_unresolved: bool,
    resolution_run_id: str | None,
) -> bool:
    status = _arxiv_resolution_status(row)
    retry_requested = (status == "failed" and retry_failed) or (status == "unresolved" and retry_unresolved)
    if not retry_requested:
        return False
    if not resolution_run_id:
        return True
    return resolution_run_id not in _arxiv_resolution_run_ids(row)


def _jsonl_row_count(path: Path) -> int:
    return sum(1 for _ in read_jsonl_if_exists(path))


def _dedupe_jsonl_rows(path: Path, key_fields: tuple[str, ...]) -> int:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    passthrough: list[dict[str, Any]] = []
    for row in read_jsonl_if_exists(path):
        key = tuple(row.get(field) for field in key_fields)
        if all(part is not None for part in key):
            latest[key] = row
        else:
            passthrough.append(row)
    rows = [*passthrough, *latest.values()]
    return write_jsonl(path, rows)


def _current_arxiv_unresolved_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unresolved: list[dict[str, Any]] = []
    for row in rows:
        status = _arxiv_resolution_status(row)
        if not status or status == "matched":
            continue
        extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
        candidate_count = extra.get("arxiv_candidate_count")
        reason = "resolver_error" if status == "failed" else "below_high_confidence"
        if status == "unresolved" and candidate_count == 0:
            reason = "no_candidates"
        unresolved_row: dict[str, Any] = {
            "paper_id": row.get("paper_id"),
            "title": row.get("title"),
            "reason": reason,
            "queries": extra.get("arxiv_queries"),
        }
        if extra.get("arxiv_best_match"):
            unresolved_row["best_match"] = extra["arxiv_best_match"]
        if extra.get("arxiv_resolution_error"):
            unresolved_row["error"] = extra["arxiv_resolution_error"]
        unresolved.append(unresolved_row)
    return unresolved


def cmd_audit_arxiv_resolution(args: argparse.Namespace) -> int:
    manifest_rows = list(read_jsonl(args.manifest))
    candidate_rows = list(read_jsonl(args.candidates))
    audit_rows, report = audit_arxiv_resolution(
        manifest_rows,
        candidate_rows,
        thresholds=ArxivAuditThresholds(),
    )
    count = write_jsonl(args.out, audit_rows)
    report_path = args.report or args.out.with_suffix(args.out.suffix + ".report.json")
    report = {
        **report,
        "command": "audit-arxiv-resolution",
        "manifest": str(args.manifest),
        "candidates": str(args.candidates),
        "out": str(args.out),
        "rows_written": count,
    }
    write_json(report_path, report)
    print(f"Wrote {count} arXiv audit rows to {args.out}")
    return 0


def cmd_resolve_arxiv(args: argparse.Namespace) -> int:
    started = time.monotonic()
    client = ArxivClient(
        api_base=args.api_base,
        http_client=make_http(args),
        cache_dir=None if args.no_arxiv_cache else args.arxiv_cache_dir,
        refresh_cache=args.refresh_arxiv_cache,
        max_network_requests=args.max_network_queries,
        rate_limit_cooldown_seconds=args.rate_limit_cooldown,
        rate_limit_extra_retries=args.rate_limit_extra_retries,
    )
    candidates_path = args.candidates or args.out.with_suffix(args.out.suffix + ".candidates.jsonl")
    unresolved_path = args.unresolved or args.out.with_suffix(args.out.suffix + ".unresolved.jsonl")
    journal_path = _arxiv_journal_path(args.out)
    all_records = list(read_paper_records(args.manifest))
    if args.limit is not None:
        all_records = all_records[: args.limit]
    if args.paper_id:
        requested_ids = set(args.paper_id)
        records = [record for record in all_records if record.paper_id in requested_ids]
    else:
        records = all_records

    if args.no_resume:
        write_jsonl(args.out, [])
        write_jsonl(journal_path, [])
        write_jsonl(candidates_path, [])
        write_jsonl(unresolved_path, [])

    prior_by_id: dict[str, dict[str, Any]] = {}
    resolution_history: dict[str, set[str]] = {}
    if not args.no_resume:
        resolution_history = _arxiv_resolution_history(args.out, journal_path)
        prior_by_id.update(_latest_rows_by_paper_id(args.out))
        prior_by_id.update(_latest_rows_by_paper_id(journal_path))
        for paper_id, row in prior_by_id.items():
            run_ids = resolution_history.get(paper_id)
            if not run_ids:
                continue
            row = dict(row)
            row["extra"] = dict(row.get("extra")) if isinstance(row.get("extra"), dict) else {}
            row["extra"]["arxiv_resolution_run_ids"] = sorted(run_ids)
            prior_by_id[paper_id] = row

    completed_by_id: dict[str, dict[str, Any]] = {}
    for paper_id, row in prior_by_id.items():
        status = _arxiv_resolution_status(row)
        if not status:
            continue
        if _arxiv_row_needs_retry(
            row,
            retry_failed=args.retry_failed,
            retry_unresolved=args.retry_unresolved,
            resolution_run_id=args.resolution_run_id,
        ):
            continue
        completed_by_id[paper_id] = row

    processed = 0
    skipped_existing = 0
    resolved_new = 0
    downloaded = 0
    skipped_downloads = 0
    candidate_rows_written = 0
    unresolved_rows_written = 0
    stopped_reason: str | None = None
    skip_run_start: int | None = None
    skip_run_end: int | None = None
    skip_run_count = 0
    skip_run_last_id: str | None = None

    def flush_skip_run() -> None:
        nonlocal skip_run_start, skip_run_end, skip_run_count, skip_run_last_id
        if skip_run_count == 0:
            return
        if skip_run_count == 1:
            print(
                f"[arxiv-skip] {skip_run_start}/{len(records)} {skip_run_last_id} already resolved",
                file=sys.stderr,
            )
        else:
            print(
                f"[arxiv-skip] {skip_run_count} already resolved records "
                f"from {skip_run_start}/{len(records)} through {skip_run_end}/{len(records)} "
                f"(last {skip_run_last_id})",
                file=sys.stderr,
            )
        skip_run_start = None
        skip_run_end = None
        skip_run_count = 0
        skip_run_last_id = None

    for idx, record in enumerate(records, start=1):
        if record.paper_id in completed_by_id:
            skipped_existing += 1
            if skip_run_start is None:
                skip_run_start = idx
            skip_run_end = idx
            skip_run_count += 1
            skip_run_last_id = record.paper_id
            continue

        flush_skip_run()
        row = record.to_dict()
        row.setdefault("extra", {})
        if args.resolution_run_id:
            attempted_run_ids = set(resolution_history.get(record.paper_id, set()))
            attempted_run_ids.add(args.resolution_run_id)
            row["extra"]["arxiv_resolution_run_id"] = args.resolution_run_id
            row["extra"]["arxiv_resolution_run_ids"] = sorted(attempted_run_ids)
        try:
            resolution = client.resolve_record(
                record,
                max_results=args.search_limit,
                deep_search=args.deep_search,
                deep_max_results=args.deep_search_limit,
                published_from=args.published_from,
                published_to=args.published_to,
            )
        except ArxivQueryBudgetExceeded as exc:
            stopped_reason = str(exc)
            print(f"[arxiv-stop] {idx}/{len(records)} {record.paper_id} {exc}", file=sys.stderr)
            break
        except KeyboardInterrupt:
            stopped_reason = "interrupted"
            print(f"[arxiv-stop] {idx}/{len(records)} interrupted; writing resumable state", file=sys.stderr)
            break
        except Exception as exc:  # noqa: BLE001 - row-local resolver failures should be retryable.
            row["extra"]["arxiv_resolution_status"] = "failed"
            row["extra"]["arxiv_resolution_error"] = repr(exc)
            append_jsonl(
                unresolved_path,
                [
                    {
                        "paper_id": record.paper_id,
                        "title": record.title,
                        "reason": "resolver_error",
                        "error": repr(exc),
                    }
                ],
            )
            unresolved_rows_written += 1
            append_jsonl(journal_path, [row])
            processed += 1
            print(f"[arxiv-fail] {idx}/{len(records)} {record.paper_id} {exc!r}", file=sys.stderr)
            continue

        candidate_rows = resolution.candidate_rows()
        if candidate_rows:
            candidate_rows_written += append_jsonl(candidates_path, candidate_rows)
        best = resolution.best_match
        acceptable = bool(best and is_confident_enough(best.confidence, args.min_confidence))
        row["extra"]["arxiv_queries"] = resolution.queries
        row["extra"]["arxiv_candidate_count"] = len(resolution.candidates)
        row["extra"]["arxiv_resolution_status"] = "matched" if acceptable else "unresolved"
        if best:
            row["extra"]["arxiv_best_match"] = best.to_dict()
        if acceptable and best:
            resolved_new += 1
            row["extra"]["arxiv_id"] = best.entry.arxiv_id
            row["extra"]["arxiv_url"] = best.entry.abs_url
            row["extra"]["arxiv_pdf_url"] = best.entry.pdf_url
            row["extra"]["arxiv_match_confidence"] = best.confidence
            row["extra"]["arxiv_match_score"] = round(best.score, 4)
            row["extra"]["source_pdf_kind"] = "arxiv_preprint"
            if args.download_pdfs and best.entry.pdf_url:
                try:
                    result = client.http.download(
                        best.entry.pdf_url,
                        args.pdf_dir / arxiv_pdf_filename(best.entry.arxiv_id),
                        overwrite=args.overwrite,
                    )
                    row["extra"]["arxiv_pdf_path"] = str(result.path)
                    row["extra"]["arxiv_pdf_sha256"] = result.sha256
                    row["extra"]["arxiv_pdf_bytes"] = result.bytes_written
                    row["extra"]["arxiv_pdf_download_skipped"] = result.skipped
                    if result.skipped:
                        skipped_downloads += 1
                    else:
                        downloaded += 1
                except Exception as exc:  # noqa: BLE001 - keep metadata match even if PDF download fails.
                    row["extra"]["arxiv_pdf_download_status"] = "failed"
                    row["extra"]["arxiv_pdf_download_error"] = repr(exc)
                    print(f"[arxiv-pdf-fail] {idx}/{len(records)} {record.paper_id} {exc!r}", file=sys.stderr)
            print(
                f"[arxiv] {idx}/{len(records)} {record.paper_id} {best.confidence} "
                f"{best.score:.3f} {best.entry.arxiv_id}",
                file=sys.stderr,
            )
        else:
            reason = "no_candidates" if not resolution.candidates else f"below_{args.min_confidence}_confidence"
            append_jsonl(
                unresolved_path,
                [
                    {
                        "paper_id": record.paper_id,
                        "title": record.title,
                        "reason": reason,
                        "best_match": best.to_dict() if best else None,
                        "queries": resolution.queries,
                    }
                ],
            )
            unresolved_rows_written += 1
            print(f"[arxiv-unresolved] {idx}/{len(records)} {record.paper_id} {reason}", file=sys.stderr)

        append_jsonl(journal_path, [row])
        processed += 1

    flush_skip_run()
    final_by_id = _latest_rows_by_paper_id(args.out)
    final_by_id.update(_latest_rows_by_paper_id(journal_path))
    final_resolution_history = _arxiv_resolution_history(args.out, journal_path)
    for paper_id, row in final_by_id.items():
        run_ids = final_resolution_history.get(paper_id)
        if not run_ids:
            continue
        row = dict(row)
        row["extra"] = dict(row.get("extra")) if isinstance(row.get("extra"), dict) else {}
        row["extra"]["arxiv_resolution_run_ids"] = sorted(run_ids)
        final_by_id[paper_id] = row
    final_rows = [final_by_id.get(record.paper_id, record.to_dict()) for record in all_records]
    count = write_jsonl(args.out, final_rows)
    current_unresolved_rows = _current_arxiv_unresolved_rows(final_rows)
    write_jsonl(unresolved_path, current_unresolved_rows)
    compact_candidate_rows = _dedupe_jsonl_rows(candidates_path, ("paper_id", "arxiv_id"))
    status_counts: dict[str, int] = {}
    confidence_counts: dict[str, int] = {}
    for row in final_rows:
        status = _arxiv_resolution_status(row)
        if status:
            status_counts[status] = status_counts.get(status, 0) + 1
        confidence = _arxiv_match_confidence(row)
        if confidence:
            confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
    target_missing_status = 0
    for record in records:
        row = final_by_id.get(record.paper_id)
        if not row or not _arxiv_resolution_status(row):
            target_missing_status += 1

    report = {
        "command": "resolve-arxiv",
        "manifest": str(args.manifest),
        "out": str(args.out),
        "candidates": str(candidates_path),
        "unresolved": str(unresolved_path),
        "journal": str(journal_path),
        "api_base": args.api_base,
        "records": count,
        "target_records": len(records),
        "processed_this_run": processed,
        "skipped_existing": skipped_existing,
        "target_missing_status": target_missing_status,
        "target_run_completed": stopped_reason is None and target_missing_status == 0,
        "resolved_this_run": resolved_new,
        "resolved": status_counts.get("matched", 0),
        "status_counts": dict(sorted(status_counts.items())),
        "unresolved_count": status_counts.get("unresolved", 0) + status_counts.get("failed", 0),
        "candidate_rows": compact_candidate_rows,
        "candidate_rows_written_this_run": candidate_rows_written,
        "unresolved_rows": len(current_unresolved_rows),
        "unresolved_rows_written_this_run": unresolved_rows_written,
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "min_confidence": args.min_confidence,
        "search_limit": args.search_limit,
        "deep_search": args.deep_search,
        "deep_search_limit": args.deep_search_limit,
        "arxiv_cache_dir": None if args.no_arxiv_cache else str(args.arxiv_cache_dir),
        "refresh_arxiv_cache": args.refresh_arxiv_cache,
        "resolution_run_id": args.resolution_run_id,
        "max_network_queries": args.max_network_queries,
        "rate_limit_cooldown": args.rate_limit_cooldown,
        "rate_limit_extra_retries": args.rate_limit_extra_retries,
        "arxiv_client_stats": client.stats.to_dict(),
        "stopped_reason": stopped_reason,
        "published_from": args.published_from,
        "published_to": args.published_to,
        "download_pdfs": args.download_pdfs,
        "downloaded": downloaded,
        "skipped_downloads": skipped_downloads,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(args.out.with_suffix(args.out.suffix + ".run.json"), report)
    print(
        f"Resolved {report['resolved']}/{len(all_records)} records through arXiv candidates "
        f"({processed} processed, {skipped_existing} skipped)",
    )
    return 0


def cmd_inspect_openreview_venue(args: argparse.Namespace) -> int:
    client = OpenReviewClient(api_base=args.api_base, http_client=make_http(args))
    group = client.get_group(args.venue_id)
    if group is None:
        raise RuntimeError(f"OpenReview venue group not found: {args.venue_id}")
    report = venue_status_report(group)
    if args.out:
        write_json(args.out, report)
        print(f"Wrote OpenReview venue report to {args.out}")
    else:
        import json

        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_release_status(args: argparse.Namespace) -> int:
    started = time.monotonic()
    http = make_http(args)
    scraper = ICMLVirtualScraper(year=args.year, http_client=http)
    records, icml_report = scraper.scrape_json_records(
        events_url=args.events_url,
        abstracts_url=args.abstracts_url,
        event_types={"Poster"},
        require_decision=False,
        limit=None,
    )
    client = OpenReviewClient(api_base=args.api_base, http_client=http)
    group = client.get_group(args.venue_id)
    venue_report = venue_status_report(group) if group else {"error": f"OpenReview venue group not found: {args.venue_id}"}
    openreview_counts = openreview_public_counts(client, group) if group else {}
    sample_record = next((record for record in records if record.extra.get("openreview_forum_id")), None)
    sample_forum_id = str(sample_record.extra.get("openreview_forum_id")) if sample_record else None
    sample_forum_pdf_url = pdf_url_for_forum_id(sample_forum_id)
    api_note_probe = (
        probe_url(http, f"{args.api_base.rstrip('/')}/notes?id={sample_forum_id}&details=directReplies", accept="application/json")
        if sample_forum_id
        else None
    )
    pdf_probe = (
        probe_url(http, sample_forum_pdf_url, accept="application/pdf,*/*")
        if sample_forum_pdf_url
        else None
    )
    explicit_pdf_record = next((record for record in records if record.pdf_url), None)
    explicit_pdf_probe = probe_url(http, explicit_pdf_record.pdf_url, accept="application/pdf,*/*") if explicit_pdf_record and explicit_pdf_record.pdf_url else None
    payload = {
        "command": "release-status",
        "year": args.year,
        "venue_id": args.venue_id,
        "icml_virtual": icml_report,
        "openreview_venue": venue_report,
        "openreview_counts": openreview_counts,
        "sample_access": {
            "sample_paper_id": sample_record.paper_id if sample_record else None,
            "sample_openreview_forum_id": sample_forum_id,
            "sample_openreview_forum_pdf_url": sample_forum_pdf_url,
            "api_note_probe": api_note_probe,
            "pdf_by_forum_probe": pdf_probe,
            "explicit_pdf_sample_paper_id": explicit_pdf_record.paper_id if explicit_pdf_record else None,
            "explicit_pdf_probe": explicit_pdf_probe,
        },
        "readiness": {
            "accepted_metadata_crawlable": bool(records),
            "accepted_metadata_records": len(records),
            "accepted_forum_url_coverage": ratio(icml_report.get("records_with_forum_url"), len(records)),
            "accepted_pdf_url_coverage": ratio(icml_report.get("records_with_pdf_url"), len(records)),
            "accepted_explicit_pdf_urls_present": bool(icml_report.get("records_with_pdf_url")),
            "accepted_forum_pdf_probe_crawlable": is_success_probe(pdf_probe),
            "accepted_pdfs_crawlable_now": bool(icml_report.get("records_with_pdf_url")) or is_success_probe(pdf_probe),
            "accepted_pdf_source": "icml_json_pdf_url"
            if icml_report.get("records_with_pdf_url")
            else ("openreview_forum_pdf" if is_success_probe(pdf_probe) else None),
            "openreview_submission_notes_visible_to_guest": is_success_probe(api_note_probe),
            "reviews_or_ratings_visible_to_guest": is_success_probe(api_note_probe),
            "rejected_public_notes_found": any(
                count_entry.get("status_name") == "rejected" and (count_entry.get("count") or 0) > 0
                for count_entry in openreview_counts.get("queries", [])
            ),
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(args.out, payload)
    readiness = payload["readiness"]
    print(
        "Release status: "
        f"metadata_records={readiness['accepted_metadata_records']} "
        f"pdf_urls={icml_report.get('records_with_pdf_url')} "
        f"forum_pdf_probe={readiness['accepted_forum_pdf_probe_crawlable']} "
        f"reviews_visible={readiness['reviews_or_ratings_visible_to_guest']} "
        f"wrote={args.out}"
    )
    return 0


def cmd_enrich_metadata(args: argparse.Namespace) -> int:
    rows = list(read_jsonl(args.manifest))
    enriched, report = enrich_metadata_rows(
        rows,
        ratings_path=args.ratings,
        awards_path=args.awards,
    )
    report.update(
        {
            "command": "enrich-metadata",
            "manifest": str(args.manifest),
            "out": str(args.out),
        }
    )
    count = write_jsonl(args.out, enriched)
    report_path = args.report or args.out.with_suffix(args.out.suffix + ".report.json")
    write_json(report_path, report)
    print(
        f"Enriched {count} records: "
        f"ratings={report['records_with_ratings']} awards={report['records_with_awards']}"
    )
    return 0


def cmd_download_pdfs(args: argparse.Namespace) -> int:
    http = make_http(args)
    updated: list[PaperRecord] = []
    downloaded = 0
    for idx, record in enumerate(read_paper_records(args.manifest)):
        if args.limit is not None and idx >= args.limit:
            break
        if record.pdf_url:
            result = http.download(record.pdf_url, args.pdf_dir / record.pdf_filename(), overwrite=args.overwrite)
            record.pdf_path = str(result.path)
            record.extra["pdf_sha256"] = result.sha256
            record.extra["pdf_bytes"] = result.bytes_written
            record.extra["pdf_download_skipped"] = result.skipped
            downloaded += 1
            print(f"[pdf] {downloaded} {record.paper_id} {result.path}", file=sys.stderr)
        updated.append(record)
    if args.out:
        write_jsonl(args.out, (record.to_dict() for record in updated))
        print(f"Wrote updated manifest to {args.out}")
    else:
        print(f"Downloaded or verified {downloaded} PDFs")
    return 0


def cmd_download_arxiv_pdfs(args: argparse.Namespace) -> int:
    http = make_http(args)
    updated: list[PaperRecord] = []
    downloaded_records: list[PaperRecord] = []
    eligible = 0
    downloaded_or_verified = 0
    downloaded = 0
    skipped = 0
    failed = 0
    for record in read_paper_records(args.manifest):
        extra = record.extra
        confidence = str(extra.get("arxiv_match_confidence") or "")
        arxiv_id = str(extra.get("arxiv_id") or "")
        arxiv_pdf_url = str(extra.get("arxiv_pdf_url") or "")
        if (
            arxiv_id
            and arxiv_pdf_url
            and is_confident_enough(confidence, args.min_confidence)
            and (args.limit is None or downloaded_or_verified < args.limit)
        ):
            eligible += 1
            try:
                result = http.download(
                    arxiv_pdf_url,
                    args.pdf_dir / arxiv_pdf_filename(arxiv_id),
                    overwrite=args.overwrite,
                )
                record.extra["arxiv_pdf_path"] = str(result.path)
                record.extra["arxiv_pdf_sha256"] = result.sha256
                record.extra["arxiv_pdf_bytes"] = result.bytes_written
                record.extra["arxiv_pdf_download_skipped"] = result.skipped
                record.extra["source_pdf_kind"] = "arxiv_preprint"
                record.extra["pdf_path_source"] = "arxiv_preprint"
                if not args.no_set_pdf_path and not record.pdf_path:
                    record.pdf_path = str(result.path)
                if result.skipped:
                    skipped += 1
                else:
                    downloaded += 1
                downloaded_or_verified += 1
                downloaded_records.append(record)
                print(f"[arxiv-pdf] {downloaded_or_verified} {record.paper_id} {result.path}", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001 - row-local download failures should be retryable.
                failed += 1
                record.extra["arxiv_pdf_download_status"] = "failed"
                record.extra["arxiv_pdf_download_error"] = repr(exc)
                print(f"[arxiv-pdf-fail] {record.paper_id} {exc!r}", file=sys.stderr)
        updated.append(record)

    count = write_jsonl(args.out, (record.to_dict() for record in updated))
    downloaded_only_count = None
    if args.downloaded_only_out:
        downloaded_only_count = write_jsonl(
            args.downloaded_only_out,
            (record.to_dict() for record in downloaded_records),
        )
    write_json(
        args.out.with_suffix(args.out.suffix + ".run.json"),
        {
            "command": "download-arxiv-pdfs",
            "manifest": str(args.manifest),
            "out": str(args.out),
            "downloaded_only_out": str(args.downloaded_only_out) if args.downloaded_only_out else None,
            "downloaded_only_records": downloaded_only_count,
            "pdf_dir": str(args.pdf_dir),
            "records": count,
            "eligible_considered": eligible,
            "downloaded_or_verified": downloaded_or_verified,
            "downloaded": downloaded,
            "skipped_existing": skipped,
            "failed": failed,
            "min_confidence": args.min_confidence,
            "limit": args.limit,
        },
    )
    if args.downloaded_only_out:
        print(f"Wrote {downloaded_only_count} downloaded-only rows to {args.downloaded_only_out}")
    print(f"Downloaded or verified {downloaded_or_verified} arXiv PDFs")
    return 0 if downloaded_or_verified or not failed else 3


def cmd_parse_pdfs(args: argparse.Namespace) -> int:
    started = time.monotonic()
    parsed_records: list[PaperRecord] = []
    ok = 0
    needs_review = 0
    failed = 0
    for idx, record in enumerate(read_paper_records(args.manifest)):
        if args.limit is not None and idx >= args.limit:
            break
        try:
            parsed = parse_pdf_record(
                record,
                text_dir=args.text_dir,
                full_char_budget=args.full_char_budget,
                compact_char_budget=args.compact_char_budget,
                scoring_char_budget=args.scoring_char_budget,
                main_paper_max_pages=args.main_paper_max_pages,
                strip_references=not args.keep_references,
            )
        except Exception as exc:  # noqa: BLE001 - parse failures should stay row-local.
            record.parse_status = "parse_failed"
            record.extra["parse_diagnostics"] = {
                "parse_status": "parse_failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            parsed = record
        if parsed.parse_status == "ok":
            ok += 1
        elif parsed.parse_status == "parse_failed":
            failed += 1
        else:
            needs_review += 1
        parsed_records.append(parsed)
        print(f"[parse] {idx + 1} {parsed.paper_id} {parsed.parse_status}", file=sys.stderr)

    count = write_jsonl(args.out, (record.to_dict() for record in parsed_records))
    report_path = args.report or args.out.with_suffix(args.out.suffix + ".report.json")
    write_parse_report(report_path, parsed_records)
    write_json(
        args.out.with_suffix(args.out.suffix + ".run.json"),
        {
            "command": "parse-pdfs",
            "manifest": str(args.manifest),
            "out": str(args.out),
            "text_dir": str(args.text_dir),
            "report": str(report_path),
            "records": count,
            "ok": ok,
            "needs_review": needs_review,
            "failed": failed,
            "full_char_budget": args.full_char_budget,
            "compact_char_budget": args.compact_char_budget,
            "scoring_char_budget": args.scoring_char_budget,
            "main_paper_max_pages": args.main_paper_max_pages,
            "keep_references": args.keep_references,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
    )
    print(f"Parsed {count} PDFs: ok={ok}, needs_review={needs_review}, failed={failed}")
    return 0 if failed == 0 else 2


def cmd_anonymize_papers(args: argparse.Namespace) -> int:
    from icml_ai_ac.anonymize import AnonymizationConfig, anonymize_manifest

    started = time.monotonic()
    report_path = args.report or args.out.with_suffix(args.out.suffix + ".report.json")
    report = anonymize_manifest(
        records=list(read_paper_records(args.manifest)),
        out=args.out,
        report_path=report_path,
        config=AnonymizationConfig(
            pdf_dir=args.pdf_dir,
            text_dir=args.text_dir,
            max_pages=args.max_pages,
            limit=args.limit,
            overwrite=args.overwrite,
        ),
    )
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    write_json(report_path, report)
    write_json(
        args.out.with_suffix(args.out.suffix + ".run.json"),
        {
            "command": "anonymize-papers",
            "manifest": str(args.manifest),
            "out": str(args.out),
            "report": str(report_path),
            **{
                key: report[key]
                for key in (
                    "anonymization_version",
                    "input_records",
                    "output_records",
                    "excluded_records",
                    "resumed_records",
                    "max_pages",
                    "elapsed_seconds",
                    "status_counts",
                )
            },
        },
    )
    print(
        f"Anonymized {report['output_records']}/{report['input_records']} papers; "
        f"excluded={report['excluded_records']} resumed={report['resumed_records']}"
    )
    return 0 if report["excluded_records"] == 0 else 2


def cmd_score_pass1(args: argparse.Namespace) -> int:
    started = time.monotonic()
    prepare_run_dir(args.run_dir)
    config = ScoreRunConfig(
        provider=args.provider,
        model=args.model,
        prompt_version=args.prompt_version,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
        text_source=args.text_source,
        dry_run=args.dry_run,
        input_cost_per_mtok=args.input_cost_per_mtok,
        output_cost_per_mtok=args.output_cost_per_mtok,
    )
    client = None if args.dry_run else ChatCompletionClient(
        provider=args.provider,
        model=args.model,
        timeout_seconds=args.timeout,
        retries=args.retries,
        backoff_seconds=args.backoff,
    )
    records = []
    paper_ids = set(args.paper_id or [])
    for record in read_paper_records(args.manifest):
        if paper_ids and record.paper_id not in paper_ids:
            continue
        if record.parse_status != "ok" and not args.include_needs_review:
            continue
        records.append(record)
        if args.limit is not None and len(records) >= args.limit:
            break

    results = []
    status_counts: dict[str, int] = {}
    for idx, record in enumerate(records, start=1):
        result = score_record(record, run_dir=args.run_dir, config=config, client=client)
        results.append(result)
        status = str(result.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
        print(f"[score] {idx}/{len(records)} {record.paper_id} {status}", file=sys.stderr)
        if args.delay:
            time.sleep(args.delay)

    count = write_jsonl(args.out, results)
    write_run_metadata(
        args.run_dir / "run.json",
        config=config,
        records=count,
        started_at=started,
        extra={
            "manifest": str(args.manifest),
            "out": str(args.out),
            "run_dir": str(args.run_dir),
            "limit": args.limit,
            "status_counts": status_counts,
            "sources": {
                "openrouter_chat_completion_docs": "https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request",
                "openai_chat_completion_docs": "https://developers.openai.com/api/reference/resources/chat",
            },
        },
    )
    print(f"Scored {count} records: {status_counts}")
    return 0 if not any(status in status_counts for status in ["failed", "validation_error"]) else 2


def cmd_score_pass1_batch(args: argparse.Namespace) -> int:
    config = Pass1BatchConfig(
        provider=args.provider,
        model=args.model,
        prompt_version=args.prompt_version,
        text_source=args.text_source,
        limit=args.limit,
        paper_ids=set(args.paper_id or []),
        per_paper_char_budget=args.per_paper_char_budget,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    client = None if args.dry_run else ChatCompletionClient(
        provider=args.provider,
        model=args.model,
        timeout_seconds=args.timeout,
        retries=args.retries,
        backoff_seconds=args.backoff,
    )
    result = run_pass1_batch_rank(
        manifest=args.manifest,
        out=args.out,
        run_dir=args.run_dir,
        config=config,
        client=client,
    )
    print(f"Batch-ranked {result['candidate_count']} candidates: {result['status']}")
    return 0 if result["status"] in {"ok", "dry_run"} else 2


def cmd_score_pass1_batches(args: argparse.Namespace) -> int:
    config = Pass1BatchSuiteConfig(
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        prompt_version=args.prompt_version,
        text_source=args.text_source,
        limit=args.limit,
        paper_ids=set(args.paper_id or []),
        per_paper_char_budget=args.per_paper_char_budget,
        batch_size=args.batch_size,
        partitions=args.partitions,
        strategy=args.strategy,
        class_path=args.class_path,
        request_delay_seconds=args.delay,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    client = None if args.dry_run else ChatCompletionClient(
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        timeout_seconds=args.timeout,
        retries=args.retries,
        backoff_seconds=args.backoff,
    )
    result = run_pass1_batch_suite(
        manifest=args.manifest,
        out=args.out,
        run_dir=args.run_dir,
        config=config,
        client=client,
    )
    print(
        f"Batch-ranked {result['candidate_count']} candidates across "
        f"{result['batch_count']} batches: {result['status']}"
    )
    return 0 if result["status"] in {"ok", "dry_run"} else 2


def cmd_score_pass1_ensemble(args: argparse.Namespace) -> int:
    started = time.monotonic()
    preset_name = None if args.no_preset else args.model_preset
    models = resolve_cheap_models(preset=preset_name, explicit_models=args.model)
    if not models:
        raise ValueError("No cheap models selected. Use --model-preset or at least one --model.")
    if args.model_workers < 1:
        raise ValueError("--model-workers must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def run_model(job: tuple[int, str]) -> tuple[Path, dict[str, Any]]:
        model_index, model = job
        safe_name = safe_model_name(model)
        score_path = args.out_dir / f"{model_index:02d}_{safe_name}.jsonl"
        run_dir = args.out_dir / "runs" / f"{model_index:02d}_{safe_name}"
        seed = None if args.seed is None else args.seed + model_index
        config = Pass1BatchSuiteConfig(
            provider=args.provider,
            model=model,
            reasoning_effort=args.reasoning_effort,
            prompt_version=args.prompt_version,
            text_source=args.text_source,
            limit=args.limit,
            paper_ids=set(args.paper_id or []),
            per_paper_char_budget=args.per_paper_char_budget,
            batch_size=args.batch_size,
            partitions=args.partitions,
            strategy=args.strategy,
            class_path=args.class_path,
            request_delay_seconds=args.delay,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            seed=seed,
            dry_run=args.dry_run,
        )
        client = None if args.dry_run else ChatCompletionClient(
            provider=args.provider,
            model=model,
            reasoning_effort=args.reasoning_effort,
            timeout_seconds=args.timeout,
            retries=args.retries,
            backoff_seconds=args.backoff,
        )
        result = run_pass1_batch_suite(
            manifest=args.manifest,
            out=score_path,
            run_dir=run_dir,
            config=config,
            client=client,
        )
        print(f"[cheap-ensemble] {model} {result['status']} rows={result.get('rows')}", file=sys.stderr)
        return score_path, result

    jobs = list(enumerate(models))
    if args.model_workers == 1:
        completed = [run_model(job) for job in jobs]
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.model_workers, len(jobs)),
            thread_name_prefix="cheap-model",
        ) as executor:
            completed = list(executor.map(run_model, jobs))
    score_paths = [score_path for score_path, _ in completed]
    results = [result for _, result in completed]
    failures = sum(result["status"] not in {"ok", "dry_run"} for result in results)

    aggregate_payload = None
    if args.aggregate_out is not None:
        aggregate_payload = build_shortlist(
            scores_path=score_paths,
            out=args.aggregate_out,
            report=args.aggregate_report,
            limit=args.aggregate_limit,
            min_per_class=args.aggregate_min_per_class,
            class_path=args.class_path,
        )

    report = {
        "command": "score-pass1-ensemble",
        "manifest": str(args.manifest),
        "out_dir": str(args.out_dir),
        "provider": args.provider,
        "model_preset": preset_name,
        "preset_notes": CHEAP_MODEL_PRESETS[preset_name].notes if preset_name else None,
        "models": models,
        "model_workers": args.model_workers,
        "score_paths": [str(path) for path in score_paths],
        "results": results,
        "aggregate_out": str(args.aggregate_out) if args.aggregate_out else None,
        "aggregate_report": str(args.aggregate_report) if args.aggregate_report else None,
        "aggregate_payload": aggregate_payload,
        "status": "dry_run" if args.dry_run else ("ok" if failures == 0 else "partial_failed"),
        "failure_count": failures,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(args.out_dir / "run.json", report)
    print(f"Cheap ensemble {report['status']}: models={len(models)} score_paths={len(score_paths)}")
    return 0 if report["status"] in {"ok", "dry_run"} else 2


def cmd_classify_contributions(args: argparse.Namespace) -> int:
    config = ContributionClassificationConfig(
        provider=args.provider,
        model=args.model,
        prompt_version=args.prompt_version,
        text_source=args.text_source,
        limit=args.limit,
        paper_ids=set(args.paper_id or []),
        per_paper_char_budget=args.per_paper_char_budget,
        batch_size=args.batch_size,
        request_delay_seconds=args.delay,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    client = None if args.dry_run else ChatCompletionClient(
        provider=args.provider,
        model=args.model,
        timeout_seconds=args.timeout,
        retries=args.retries,
        backoff_seconds=args.backoff,
    )
    result = run_contribution_classification(
        manifest=args.manifest,
        out=args.out,
        run_dir=args.run_dir,
        config=config,
        client=client,
    )
    print(f"Classified {result['candidate_count']} candidates: {result['status']}")
    return 0 if result["status"] in {"ok", "dry_run"} else 2


def cmd_rank_pass2(args: argparse.Namespace) -> int:
    config = Pass2RankingConfig(
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        prompt_version=args.prompt_version,
        text_source=args.text_source,
        top_fraction=args.top_fraction,
        limit=args.limit,
        per_paper_char_budget=args.per_paper_char_budget,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    client = None if args.dry_run else ChatCompletionClient(
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        timeout_seconds=args.timeout,
        retries=args.retries,
        backoff_seconds=args.backoff,
    )
    result = run_pass2_ranking(
        manifest=args.manifest,
        pass1_path=args.pass1,
        out=args.out,
        run_dir=args.run_dir,
        config=config,
        client=client,
    )
    print(f"Ranked {result['candidate_count']} candidates: {result['status']}")
    return 0 if result["status"] in {"ok", "dry_run"} else 2


def cmd_rank_pass2_batches(args: argparse.Namespace) -> int:
    config = Pass2BatchedRankingConfig(
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        prompt_version=args.prompt_version,
        text_source=args.text_source,
        top_fraction=args.top_fraction,
        limit=args.limit,
        per_paper_char_budget=args.per_paper_char_budget,
        batch_size=args.batch_size,
        partitions=args.partitions,
        strategy=args.strategy,
        class_path=args.class_path,
        request_delay_seconds=args.delay,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
        dry_run=args.dry_run,
        openrouter_provider_only=tuple(args.openrouter_provider_only or ()),
    )
    provider_preferences = None
    if args.openrouter_provider_only:
        if args.provider != "openrouter":
            raise ValueError("--openrouter-provider-only requires --provider openrouter")
        provider_preferences = {
            "only": list(args.openrouter_provider_only),
            "allow_fallbacks": False,
        }
    client = None if args.dry_run else ChatCompletionClient(
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        openrouter_provider_preferences=provider_preferences,
        timeout_seconds=args.timeout,
        retries=args.retries,
        backoff_seconds=args.backoff,
    )
    result = run_pass2_batched_ranking(
        manifest=args.manifest,
        pass1_path=args.pass1,
        out=args.out,
        run_dir=args.run_dir,
        config=config,
        client=client,
    )
    print(
        f"Batched strong ranking {result['status']}: candidates={result['candidate_count']} "
        f"batches={result['attempted_batch_count']}/{result['batch_count']} "
        f"resumed={result['resumed_batch_count']}"
    )
    return 0 if result["status"] in {"ok", "dry_run"} else 2


def cmd_rank_pass2_two_stage(args: argparse.Namespace) -> int:
    config = Pass2TwoStageConfig(
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        stage1_prompt_version=args.stage1_prompt_version,
        stage2_prompt_version=args.stage2_prompt_version,
        text_source=args.text_source,
        stage1_limit=args.stage1_limit,
        stage2_limit=args.stage2_limit,
        stage1_per_paper_char_budget=args.stage1_per_paper_char_budget,
        stage2_per_paper_char_budget=args.stage2_per_paper_char_budget,
        temperature=args.temperature,
        stage1_max_output_tokens=args.stage1_max_output_tokens,
        stage2_max_output_tokens=args.stage2_max_output_tokens,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    client = None if args.dry_run else ChatCompletionClient(
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        timeout_seconds=args.timeout,
        retries=args.retries,
        backoff_seconds=args.backoff,
    )
    result = run_pass2_two_stage(
        manifest=args.manifest,
        pass1_path=args.pass1,
        out=args.out,
        run_dir=args.run_dir,
        config=config,
        client=client,
    )
    print(
        f"Two-stage ranked stage1={result.get('stage1_status')} "
        f"stage2={result.get('stage2_status')} overall={result['status']}"
    )
    return 0 if result["status"] in {"ok", "dry_run"} else 2


def cmd_rank_reference(args: argparse.Namespace) -> int:
    config = ReferenceRankingConfig(
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        prompt_version=args.prompt_version,
        paper_set_name=args.paper_set_name,
        text_source=args.text_source,
        limit=args.limit,
        paper_ids=set(args.paper_id or []),
        per_paper_char_budget=args.per_paper_char_budget,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    client = None if args.dry_run else ChatCompletionClient(
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        timeout_seconds=args.timeout,
        retries=args.retries,
        backoff_seconds=args.backoff,
    )
    result = run_reference_ranking(
        manifest=args.manifest,
        out=args.out,
        run_dir=args.run_dir,
        config=config,
        client=client,
    )
    print(f"Reference-ranked {result['candidate_count']} candidates: {result['status']}")
    if result.get("validation_errors"):
        print(f"Validation errors: {result['validation_errors']}", file=sys.stderr)
    return 0 if result["status"] in {"ok", "dry_run"} else 2


def cmd_repair_reference(args: argparse.Namespace) -> int:
    config = ReferenceRepairConfig(
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        prompt_version=args.prompt_version,
        paper_set_name=args.paper_set_name,
        text_source=args.text_source,
        per_paper_char_budget=args.per_paper_char_budget,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    client = None if args.dry_run else ChatCompletionClient(
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        timeout_seconds=args.timeout,
        retries=args.retries,
        backoff_seconds=args.backoff,
    )
    result = run_reference_repair(
        manifest=args.manifest,
        reference_path=args.reference,
        out=args.out,
        run_dir=args.run_dir,
        config=config,
        client=client,
    )
    print(f"Repaired reference ranking for {result['candidate_count']} candidates: {result['status']}")
    if result.get("validation_errors"):
        print(f"Validation errors: {result['validation_errors']}", file=sys.stderr)
    return 0 if result["status"] in {"ok", "dry_run"} else 2


def cmd_rank_frontier_pdf_cards(args: argparse.Namespace) -> int:
    paper_ids = set(args.paper_id or [])
    if args.paper_list is not None:
        paper_ids.update(str(row.get("paper_id")) for row in read_ranked_rows(args.paper_list) if row.get("paper_id"))
    config = FrontierCardConfig(
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        prompt_version=args.prompt_version,
        paper_set_name=args.paper_set_name,
        text_source=args.text_source,
        limit=args.limit,
        paper_ids=paper_ids,
        per_paper_char_budget=args.per_paper_char_budget,
        pdf_excerpt_pages=args.pdf_excerpt_pages,
        pdf_excerpt_dir=args.pdf_excerpt_dir,
        pdf_optimize_threshold_bytes=args.pdf_optimize_threshold_bytes or None,
        pdf_settings=args.pdf_settings,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        openrouter_pdf_engine=args.openrouter_pdf_engine,
        fallback_model=args.fallback_model,
        fallback_reasoning_effort=args.fallback_reasoning_effort,
    )
    client = None if args.dry_run else ChatCompletionClient(
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        openrouter_pdf_engine=args.openrouter_pdf_engine if args.provider == "openrouter" else None,
        timeout_seconds=args.timeout,
        retries=args.retries,
        backoff_seconds=args.backoff,
    )
    fallback_client = None
    if not args.dry_run and args.fallback_model:
        fallback_client = ChatCompletionClient(
            provider=args.provider,
            model=args.fallback_model,
            reasoning_effort=args.fallback_reasoning_effort,
            openrouter_pdf_engine=args.openrouter_pdf_engine if args.provider == "openrouter" else None,
            timeout_seconds=args.timeout,
            retries=args.retries,
            backoff_seconds=args.backoff,
        )
    result = run_frontier_pdf_cards(
        manifest=args.manifest,
        out=args.out,
        run_dir=args.run_dir,
        config=config,
        client=client,
        fallback_client=fallback_client,
    )
    print(
        f"Frontier PDF cards {result['ok_count']}/{result['candidate_count']} ok "
        f"for {result['model']}: {result['status']}"
    )
    return 0 if result["status"] in {"ok", "dry_run"} else 2


def cmd_rank_frontier_card_ensemble(args: argparse.Namespace) -> int:
    result = build_frontier_card_ensemble(
        card_paths=args.cards,
        labels=args.label,
        out=args.out,
        paper_set_name=args.paper_set_name,
        prompt_version=args.prompt_version,
    )
    print(f"Built frontier card ensemble ranking for {result['candidate_count']} candidates")
    return 0


def cmd_rank_frontier_card_synthesis(args: argparse.Namespace) -> int:
    config = FrontierSynthesisConfig(
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        prompt_version=args.prompt_version,
        paper_set_name=args.paper_set_name,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    client = None if args.dry_run else ChatCompletionClient(
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        timeout_seconds=args.timeout,
        retries=args.retries,
        backoff_seconds=args.backoff,
    )
    result = run_frontier_card_synthesis(
        card_paths=args.cards,
        out=args.out,
        run_dir=args.run_dir,
        config=config,
        client=client,
    )
    print(f"Frontier card synthesis ranked {result['candidate_count']} candidates: {result['status']}")
    if result.get("validation_errors"):
        print(f"Validation errors: {result['validation_errors']}", file=sys.stderr)
    return 0 if result["status"] in {"ok", "dry_run"} else 2


def cmd_rank_frontier_card_tournament(args: argparse.Namespace) -> int:
    config = FrontierTournamentConfig(
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        prompt_version=args.prompt_version,
        paper_set_name=args.paper_set_name,
        strategy=args.strategy,
        top_n=args.top_n,
        pairs_per_batch=args.pairs_per_batch,
        swiss_rounds=args.swiss_rounds,
        playoff_top_n=args.playoff_top_n,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    client = None if args.dry_run else ChatCompletionClient(
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        timeout_seconds=args.timeout,
        retries=args.retries,
        backoff_seconds=args.backoff,
    )
    result = run_frontier_card_tournament(
        card_paths=args.cards,
        seed_ranking_path=args.seed_ranking,
        out=args.out,
        run_dir=args.run_dir,
        config=config,
        client=client,
    )
    print(
        f"Frontier card tournament top-{result['top_n']} "
        f"{result.get('match_count', 0)}/{result['pair_count']} matches: {result['status']}"
    )
    return 0 if result["status"] in {"ok", "dry_run"} else 2


def cmd_rank_bradley_terry(args: argparse.Namespace) -> int:
    payload = read_json(args.tournament)
    if not isinstance(payload, dict):
        print(f"Tournament payload must be a JSON object: {args.tournament}")
        return 2
    matches = payload.get("matches")
    if not isinstance(matches, list):
        print(f"Tournament payload has no `matches` array: {args.tournament}")
        return 2
    try:
        selected_matches, restrict, stage_meta = select_tournament_stage(payload, stage=args.stage)
        result = rank_bradley_terry(
            selected_matches,
            ranked_meta=ranked_meta_from_payload(payload),
            restrict_to=restrict,
            weight_mode=args.weight,
            prior_strength=args.prior_strength,
            max_iter=args.max_iter,
            tol=args.tol,
        )
    except ValueError as exc:
        print(f"Bradley-Terry configuration error: {exc}")
        return 2
    result["source_tournament"] = str(args.tournament)
    result["stage_selection"] = stage_meta
    write_json(args.out, result)
    diagnostics = result["diagnostics"]
    top = result["ranked_papers"][0]["paper_id"] if result["ranked_papers"] else "-"
    print(
        f"Bradley-Terry ranked {diagnostics['player_count']} papers "
        f"(converged={diagnostics['converged']} in {diagnostics['iterations']} iters, "
        f"connected={diagnostics['comparison_graph_connected']}); top={top}"
    )
    return 0 if diagnostics["converged"] else 2


def cmd_simulate_hybrid_tournament(args: argparse.Namespace) -> int:
    payload = read_json(args.tournament)
    if not isinstance(payload, dict):
        print(f"Tournament payload must be a JSON object: {args.tournament}")
        return 2
    try:
        result = simulate_hybrid_tournament(
            payload,
            swiss_rounds=args.swiss_rounds,
            playoff_top_n=args.playoff_top_n,
        )
    except ValueError as exc:
        print(f"Tournament simulation error: {exc}")
        return 2
    result["source_tournament"] = str(args.tournament)
    write_json(args.out, result)
    metrics = result["metrics_vs_all_pairs"]
    print(
        f"Simulated {result['simulated_pair_count']}/{result['all_pairs_count']} pairs: "
        f"spearman={metrics['spearman']} "
        f"top10_recall={metrics['top_10']['recall_at_k']}"
    )
    return 0


def cmd_build_shortlist(args: argparse.Namespace) -> int:
    report = build_shortlist(
        scores_path=args.scores,
        out=args.out,
        report=args.report,
        limit=args.limit,
        min_per_class=args.min_per_class,
        class_path=args.class_path,
    )
    print(
        f"Built shortlist {report['written_rows']}/{report['unique_papers']} "
        f"from {report['input_rows']} score rows"
    )
    return 0


def cmd_build_position_robust_shortlist(args: argparse.Namespace) -> int:
    report = build_position_robust_shortlist(
        scores_paths=args.scores,
        raw_ranking_path=args.raw_ranking,
        class_path=args.class_path,
        cutoff=args.cutoff,
        adjusted_out=args.adjusted_out,
        shortlist_out=args.out,
        report_out=args.report,
    )
    print(
        f"Built position-robust Stage 3 shortlist: {report['union_count']} papers "
        f"from raw/adjusted top {report['cutoff']} "
        f"(overlap={report['overlap_count']})"
    )
    return 0


def cmd_ensemble_semifinal_rankings(args: argparse.Namespace) -> int:
    payload = ensemble_semifinal_rankings(
        ranking_paths=args.ranking,
        labels=args.label,
        out=args.out,
        report=args.report,
    )
    print(
        f"Ensembled {payload['source_count']} semifinal rankings over "
        f"{payload['paper_count']} papers"
    )
    return 0


def cmd_audit_semifinal_rankings(args: argparse.Namespace) -> int:
    payload = audit_semifinal_rankings(
        ranking_paths=args.ranking,
        labels=args.label,
        expected_models=args.expected_model,
        shortlist_path=args.shortlist,
        manifest_path=args.manifest,
        out=args.out,
    )
    print(
        f"Stage 3 audit {payload['status']}: papers={payload['paper_count']} "
        f"blocking={len(payload['blocking_issues'])} warnings={len(payload['warnings'])}"
    )
    return 1 if payload["blocking_issues"] else 0


def cmd_select_finalists(args: argparse.Namespace) -> int:
    payload = select_finalists(
        cheap_path=args.cheap,
        semifinal_path=args.semifinal,
        out=args.out,
        report=args.report,
        config=FinalistSelectionConfig(
            limit=args.limit,
            semifinal_top=args.semifinal_top,
            cheap_top=args.cheap_top,
            min_per_class=args.min_per_class,
            disagreement_saves=args.disagreement_saves,
            judge_disagreement_saves=args.judge_disagreement_saves,
        ),
    )
    print(
        f"Selected {payload['written_rows']} finalists from "
        f"{payload['overlap_count']} cheap/semifinal overlapping papers"
    )
    return 0


def cmd_eval_ranking(args: argparse.Namespace) -> int:
    metrics = evaluate_ranking(
        gold_path=args.gold,
        candidate_path=args.candidate,
        out=args.out,
        k_values=args.k or [5, 10, 20],
    )
    print(
        f"Evaluated ranking: overlap={metrics['overlap_count']}/{metrics['gold_count']} "
        f"spearman={metrics['rank_correlation']['spearman']}"
    )
    return 0


def cmd_human_divergence_report(args: argparse.Namespace) -> int:
    try:
        dataset = load_comparison_dataset(
            manifest=args.manifest,
            ai_scores=args.ai_scores,
            strong_ranking=args.strong_ranking,
            min_coverage=args.min_coverage,
        )
    except ValueError as exc:
        print(f"Cannot build human comparison: {exc}")
        return 2
    if not dataset.rows:
        print("No papers joined between manifest and AI scores.")
        return 2
    try:
        report = build_divergence_report(
            dataset.rows,
            human_axis=args.human_axis,
            top_frac=args.top_frac,
            cases_per_direction=args.cases_per_direction,
            min_reviews=args.min_reviews,
            coverage=dataset.coverage,
        )
    except ValueError as exc:
        print(f"Cannot build divergence report: {exc}")
        return 2
    write_json(args.out, report)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"Divergence report: {report['counts']['papers']} papers; "
        f"{report['overlooked_gems']['shown']}/{report['overlooked_gems']['total_matching']} gems, "
        f"{report['blind_spots']['shown']}/{report['blind_spots']['total_matching']} blind spots"
    )
    return 0


def cmd_divergence_taxonomy_induce(args: argparse.Namespace) -> int:
    config = TaxonomyConfig(
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    client = None if args.dry_run else ChatCompletionClient(
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        timeout_seconds=args.timeout,
        retries=args.retries,
        backoff_seconds=args.backoff,
    )
    result = run_taxonomy_induction(
        report_path=args.report,
        out=args.out,
        run_dir=args.run_dir,
        config=config,
        client=client,
        sample_size=args.sample_size,
        max_codes=args.max_codes,
    )
    codes = result.get("codes") or []
    print(f"Taxonomy induction: {result['status']}; {len(codes)} codes over {result.get('sampled_case_count', 0)} sampled cases")
    return 0 if result["status"] in {"ok", "dry_run"} else 2


def cmd_divergence_taxonomy_classify(args: argparse.Namespace) -> int:
    config = TaxonomyConfig(
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    client = None if args.dry_run else ChatCompletionClient(
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        timeout_seconds=args.timeout,
        retries=args.retries,
        backoff_seconds=args.backoff,
    )
    second_client = None
    second_config = None
    if args.second_model and not args.dry_run:
        second_config = TaxonomyConfig(
            provider=args.second_provider or args.provider,
            model=args.second_model,
            reasoning_effort=args.second_reasoning_effort,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            seed=args.seed,
            dry_run=False,
        )
        second_client = ChatCompletionClient(
            provider=args.second_provider or args.provider,
            model=args.second_model,
            reasoning_effort=args.second_reasoning_effort,
            timeout_seconds=args.timeout,
            retries=args.retries,
            backoff_seconds=args.backoff,
        )
    result = run_taxonomy_classification(
        report_path=args.report,
        codebook_path=args.codebook,
        out=args.out,
        run_dir=args.run_dir,
        config=config,
        client=client,
        batch_size=args.batch_size,
        max_codes_per_case=args.max_codes_per_case,
        second_client=second_client,
        second_label=args.second_model,
        second_config=second_config,
    )
    aggregate = result.get("aggregate", {})
    print(f"Taxonomy classify: {result['status']}; coded {aggregate.get('cases_coded', 0)} cases")
    return 0 if result["status"] in {"ok", "dry_run"} else 2


def cmd_human_agreement_report(args: argparse.Namespace) -> int:
    try:
        dataset = load_comparison_dataset(
            manifest=args.manifest,
            ai_scores=args.ai_scores,
            strong_ranking=args.strong_ranking,
            min_coverage=args.min_coverage,
        )
    except ValueError as exc:
        print(f"Cannot build human comparison: {exc}")
        return 2
    if not dataset.rows:
        print("No papers joined between manifest and AI scores.")
        return 2
    try:
        report = build_agreement_report(
            dataset.rows,
            k_values=args.k or [10, 20, 50, 100],
            include_award=args.honored_includes_award,
            coverage=dataset.coverage,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
    except ValueError as exc:
        print(f"Cannot build agreement report: {exc}")
        return 2
    write_json(args.out, report)
    full = report["full_coverage"]
    if full is not None:
        print(
            f"Agreement: n={full['n']} tau_b(tier)={full['kendall_tau_b']['vs_tier']} "
            f"auc(honored_vs_poster)={full['roc_auc']['honored_vs_poster']}"
        )
    else:
        decision = report["acceptance_outcome"]
        print(
            f"Agreement: n={decision['n']} "
            f"auc(accepted_vs_rejected)={decision['roc_auc_accepted_vs_rejected']}"
        )
    return 0


def cmd_ai_axis_decomposition(args: argparse.Namespace) -> int:
    try:
        dataset = load_comparison_dataset(
            manifest=args.manifest,
            ai_scores=args.ai_scores,
            min_coverage=args.min_coverage,
        )
    except ValueError as exc:
        print(f"Cannot build human comparison: {exc}")
        return 2
    if not dataset.rows:
        print("No papers joined between manifest and AI scores.")
        return 2
    try:
        report = build_axis_decomposition_report(
            dataset.rows,
            include_award=args.honored_includes_award,
            coverage=dataset.coverage,
        )
    except ValueError as exc:
        print(f"Cannot build axis decomposition: {exc}")
        return 2
    write_json(args.out, report)
    ivp = report["impact_vs_polish"]
    print(
        f"Axis decomposition: {len(report['axes'])} axes; "
        f"conventional-minus-executive tau_b(tier) delta={ivp['delta_conventional_minus_executive']}"
    )
    return 0


def cmd_query_ranked_papers(args: argparse.Namespace) -> int:
    try:
        rows = query_ranked_papers(
            metadata_path=args.metadata,
            cheap_ranking_path=args.cheap_ranking,
            final_ranking_path=args.final_ranking,
            query=RankingQuery(
                preset=args.preset,
                query=args.query,
                topic=args.topic,
                contribution_class=args.contribution_class,
                scope=args.scope,
                limit=args.limit,
                include_abstract=args.include_abstract,
            ),
        )
    except (OSError, ValueError) as exc:
        print(f"Cannot query ranked papers: {exc}", file=sys.stderr)
        return 2

    output_format = args.format
    if output_format is None:
        suffix = args.out.suffix.lower() if args.out else ""
        output_format = {".csv": "csv", ".jsonl": "jsonl"}.get(suffix, "tsv")
    write_query_results(rows, path=args.out, output_format=output_format)
    finalist_count = sum(row.get("final_rank") is not None for row in rows)
    playoff_count = sum(row.get("final_stage") == "playoff_all_pairs" for row in rows)
    destination = str(args.out) if args.out else "stdout"
    print(
        f"Matched {len(rows)} papers: {finalist_count} frontier finalists, "
        f"{playoff_count} all-pairs playoff papers; wrote {destination}.",
        file=sys.stderr,
    )
    return 0


def cmd_top_ranked_papers(args: argparse.Namespace) -> int:
    try:
        rows = top_ranked_papers(
            args.category,
            top_n=args.top,
            scope=args.scope,
            metadata_path=args.metadata,
            cheap_ranking_path=args.cheap_ranking,
            final_ranking_path=args.final_ranking,
            include_abstract=args.include_abstract,
        )
        write_query_results(rows, path=args.out, output_format=args.format)
    except (OSError, ValueError) as exc:
        print(f"Cannot retrieve top-ranked papers: {exc}", file=sys.stderr)
        return 2
    destination = str(args.out) if args.out else "stdout"
    print(
        f"Returned {len(rows)} papers for category {args.category!r} "
        f"from scope {args.scope!r}; wrote {destination}.",
        file=sys.stderr,
    )
    return 0


def cmd_list_ranking_categories(args: argparse.Namespace) -> int:
    del args
    print(json.dumps(available_ranking_categories(), indent=2, sort_keys=True))
    return 0


def openreview_public_counts(client: OpenReviewClient, group: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for status_name in ["accepted", "rejected", "all_submissions", "submitted", "desk_rejected", "withdrawn"]:
        for query_index, query in enumerate(queries_for_status(group, status_name)):
            row: dict[str, Any] = {
                "status_name": status_name,
                "query_index": query_index,
                "query": query,
            }
            try:
                row["count"] = client.count_notes(query=query)
                row["status"] = "ok"
            except urllib.error.HTTPError as exc:
                row["status"] = "http_error"
                row["http_status"] = exc.code
                row["error"] = str(exc)
            except Exception as exc:  # noqa: BLE001 - release audit should preserve all probe failures.
                row["status"] = "failed"
                row["error"] = repr(exc)
            rows.append(row)
    return {
        "queries": rows,
        "nonzero_queries": [row for row in rows if (row.get("count") or 0) > 0],
    }


def probe_url(http: HttpClient, url: str, *, accept: str) -> dict[str, Any]:
    try:
        result = http.fetch(url, accept=accept)
        return {
            "url": url,
            "status": result.status,
            "content_type": result.content_type,
            "final_url": result.final_url,
            "bytes": len(result.body),
            "ok": 200 <= result.status < 300,
        }
    except urllib.error.HTTPError as exc:
        return {
            "url": url,
            "status": exc.code,
            "content_type": exc.headers.get("content-type") if exc.headers else None,
            "ok": False,
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001 - probe result should be persisted rather than crash.
        return {
            "url": url,
            "status": None,
            "ok": False,
            "error": repr(exc),
        }


def is_success_probe(probe: dict[str, Any] | None) -> bool:
    return bool(probe and probe.get("ok"))


def ratio(numerator: Any, denominator: Any) -> float | None:
    try:
        denominator_float = float(denominator)
        if denominator_float == 0:
            return None
        return round(float(numerator) / denominator_float, 6)
    except (TypeError, ValueError):
        return None


def safe_model_name(model: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in model)


if __name__ == "__main__":
    raise SystemExit(main())
