# Production Preflight: 2026-07-25

## Scope

The preflight exercised the integrated pipeline on real ICML 2026 artifacts
before the full-corpus run:

- a six-paper end-to-end smoke test;
- a 48-paper medium run stratified across awards, orals, spotlights, regular
  review-score bands, and missing-score cases; and
- the full unit suite.

The production inventory contains 6,628 accepted records. OpenReview exposed
6,614 official PDFs; 6,611 parsed `ok`, three require review, and 14 records have
no official PDF. The frozen scoring manifest contains 6,617 papers: all 6,614
official PDFs plus three validated high-confidence arXiv fallbacks. The three
parser warnings were inspected and retained because their abstract and main
body text are intact. Eleven papers without any usable PDF are explicitly
reported as excluded. A full-corpus dry run selected all 6,617 papers into 414
classification batches.

## Medium-Run Results

| Stage | Result | Observed time | Reported usage cost |
| --- | ---: | ---: | ---: |
| Contribution classification | 48/48 | resumable 4-batch run | $0.02 |
| Four-model cheap ensemble | 384/384 judgments | 18.6 summed model-minutes | $1.53 |
| Terra semifinal | 64/64 judgments | 6.3 min | API did not report cost |
| Sonnet semifinal | 64/64 judgments | 13.0 min | $2.04 |
| Frontier PDF cards | 36/36 | about 14 min at slowest stream | $12.00 |
| Sol synthesis | 12/12 ranked | 1.8 min | $0.36 |
| Randomized Sol tournament | 66/66 pairs | 4.6 min | $0.89 |

The 36 frontier cards include one explicit Fable-to-Opus 4.8 fallback. Known
successful OpenRouter usage across scoring, ranking, tournament, and qualitative
coding was about $17.02, plus direct Terra usage. Failed development attempts
and superseded tournament probes make actual debugging spend higher; artifact
cost should therefore be treated as a production-path estimate, not an invoice.

The cheap top-32 retained all nine award papers. The conservative selector then
formed 12 finalists from consensus rank, class leaders, cheap/strong
disagreement, and Terra/Sonnet disagreement.

A separate live concurrency probe ran all four cheap judges over the same eight
papers with `--model-workers 4`. All 32 judgments completed in 51 seconds for
$0.12, versus 111 seconds of summed model time. This validates four independent
model streams without changing deterministic per-model batching or resume
state.

## Ranking Diagnostics

The final tournament randomized pair-to-batch assignment, A/B presentation, and
paper evidence order. It produced 32 paper-A wins and 34 paper-B wins.
Regularized Bradley-Terry and direct all-pairs win ordering agreed exactly.

Against the 12-paper tournament:

| Candidate ranking | Spearman | Top-5 recall | Top-10 recall |
| --- | ---: | ---: | ---: |
| Cheap ensemble | 0.546 | 0.60 | 1.00 |
| Terra/Sonnet semifinal | 0.664 | 0.80 | 0.90 |
| Frontier-card synthesis | 0.972 | 1.00 | 1.00 |

This is the intended stage behavior: cheap and semifinal passes preserve and
coarsely order candidates, while PDF-aware frontier review determines the final
ranking. The figures are operational diagnostics, not population estimates,
because the sample deliberately oversampled human-honored papers.

The corrected ICML 2025 gold tournament then completed 435/435 randomized
comparisons with 221 A wins and 214 B wins. Valid batches cost $8.52 and used
38 minutes of provider time; one malformed 25-pair response was rejected and
retried for about $0.41. PDF synthesis had Spearman 0.984 and 10/10 top-10
recall. The current four-model cheap ensemble retained all gold top-10 papers
by rank 20 and 18/20 gold top-20 papers by rank 30, while the old strong-model
top-35 cut retained only 8/10 and 15/20 by rank 30. A simulated Swiss-10 plus
top-20 playoff used 255/435 comparisons, had Spearman 0.996, and retained 10/10
top-10 and 19/20 top-20 papers. Bradley-Terry and direct all-pairs ranks had
Spearman 0.997, shared 9/10 top-10 papers, and differed by at most two
positions.

## Reliability Findings

The preflight found and fixed:

1. unsupported OpenAI Responses `seed` requests and crawler-sized timeouts;
2. non-resumable classification and strong-ranking calls;
3. infeasible corpus-wide semifinal prompts;
4. incomplete JSON/content-block provider responses and explicit card fallback;
5. incorrect ranking-file detection by filename extension;
6. finalist reason leakage and floating-point usage undercounting;
7. seed-correlated tournament A/B position and pair-batch order;
8. missing connectivity validation for repeated listwise semifinal batches;
   and
9. repeated batch requests after authentication, permission, billing-credit,
   or rate-limit failures; and
10. human outcome labels in legacy single-paper and strong-ranking prompt
    formatters.

All model batches now persist fingerprints, exact coverage, prompts, responses,
requested and served model IDs, and scalar usage. Reruns reuse only matching
valid batches. The pipeline stops short of aggregation when coverage or
comparison-graph invariants fail, and stops the remaining suite after the first
request-wide `400`, `401`, `402`, `403`, `404`, `422`, or exhausted `429`
response. Production prompt versions exclude reviews, ratings, area-chair
comments, decisions, tiers, and awards; human metadata is joined only after AI
ranking is complete.

## Launch Assessment

The smoke, medium run, corrected historical gold evaluation, concurrency probe,
and 156-test suite all pass. The integrated code is ready for the full accepted
paper run. At the observed rate, plan approximately 14-15 hours for the
four-stream cheap pass, about nine hours for concurrent Terra/Sonnet semifinal
judging of a 20% shortlist, about five hours for 250 concurrent frontier-card
streams, and a few hours for synthesis plus Swiss/dense tournament ranking.
End-to-end wall time should be budgeted at roughly one to two days, with every
stage resumable.
