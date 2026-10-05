# Scoring Pipeline

The first scoring pass evaluates identity-redacted `scoring_repr.txt`
derivatives with cheap models and writes fully auditable model-run artifacts.
`scoring_repr.txt` is the main paper body extracted from the PDF with references
removed and appendix or supplement stripped when detected. For ICML 2026
camera-ready PDFs, the default
uses the first 9 PDF pages as the main-paper prior before reference/appendix
cleanup, matching the camera-ready main-body limit. For originally submitted
versions, use `parse-pdfs --main-paper-max-pages 8`. `compact_repr.txt` remains
useful for cheap smoke tests, but it is not sufficient for substantive
scientific scoring.

## Model Strategy

Rationale: the first pass needs broad, inexpensive, structured triage over
thousands of papers, not final judgment. Independent absolute cheap-model scores
are too compressed, so the preferred first pass is listwise forced-bucket
ranking over small batches. The stronger semifinal independently uses GPT-5.6
Terra and Claude Sonnet 5 on shortlisted paper bodies, then aggregates their
normalized ranks.

Current production cheap-model preset, selected by a July 2026 accepted-50
bakeoff:

```text
production_2026_v2:
- nvidia/nemotron-3-ultra-550b-a55b
- google/gemini-3.5-flash-lite
- openai/gpt-5.6-luna
- x-ai/grok-4.3
```

This intentionally uses four model families. The signal is still recall
oriented; the cheap ensemble selects candidates and stores useful priors, but it
does not define the headline ranking.

There is also a broader probe preset for the accepted-50 evaluation:

```text
broad_2026_probe:
- qwen/qwen3.7-plus
- nvidia/nemotron-3-ultra-550b-a55b
- google/gemini-3.1-flash-lite
- google/gemini-3.5-flash-lite
- stepfun/step-3.7-flash
- inclusionai/ring-2.6-1t
```

Use the broader preset only after validating JSON reliability and gold-set
recall. The historical `qwen/qwen3-32b`, `qwen/qwen3.6-35b-a3b`, and
`qwen/qwen3-235b-a22b-2507` probe set is retained as `legacy_2025_gold` for
reproducibility.

Do not use web search or retrieval during scoring. The prompt explicitly asks
for paper-only judgment and tells the model to ignore author identity. Scoring
manifests route to validated derivatives with bylines, affiliations, emails,
acknowledgements, and PDF author metadata removed; canonical files remain
unchanged. The stage fails closed when a derivative does not validate. This
does not prevent identification from titles, self-citations, project names, or
model memory. Reviews, reviewer ratings, area-chair comments, decisions,
presentation tiers, and awards are never serialized into model-facing prompts.

OpenRouter structured calls default to `reasoning: {"effort": "none",
"exclude": true}`. This avoids paying for returned reasoning tokens and reduces
the risk that reasoning output consumes the JSON budget. To intentionally test a
reasoning cheap pass, set `OPENROUTER_REASONING_EFFORT` to `minimal`, `low`,
`medium`, `high`, or another OpenRouter-supported value.

Gemini 3.5 Flash Lite is the narrow exception: OpenRouter requires at least
`minimal` reasoning for that model. The client upgrades only this model from
`none` to `minimal`, keeps `exclude: true`, and stores the effective request in
the batch audit.

## Historical Accepted-50 Cheap-Model Probe

On 2026-05-15, a one-partition accepted-50 probe used the v2 batch prompt,
batch size 8, class-round-robin batching, and class-balanced top-30 shortlist
construction. This is a historical baseline, not the current production
default. It remains useful for explaining why the pipeline treats cheap models
as recall filters rather than final rankers.

| Model | Cost | Gold top-10 in candidate top-30 | Gold top-20 in candidate top-30 | Shared Spearman | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `qwen/qwen3-32b` | $0.0204 | 1.00 | 0.75 | -0.0816 | Best recall into the shortlist, poor ordering. |
| `qwen/qwen3-235b-a22b-2507` | $0.0233 | 0.80 | 0.75 | 0.4572 | Better relative ordering and class macro recall, but missed two gold top-10 papers. |
| `qwen/qwen3.6-35b-a3b` with reasoning disabled | $0.0388 | 1.00 | 0.75 | 0.3215 | Matches baseline top-k recall and is faster than the recovered baseline run, but costs more. |

Historical implication: single cheap models were useful for recall but not
trusted for ordering. The current production default is the diverse
`production_2026_v2` cheap ensemble above.

## July 2026 Accepted-50 Model Refresh

Fourteen current models were screened over the same accepted-50 representation
and listwise batches. Eleven completed usable full or near-full rankings. Step
3.7 Flash and Gemini 3.5 Flash require reasoning and rejected the production
reasoning-off request; DeepSeek V4 Pro scored 48/50 before a malformed final
batch; DeepSeek V4 Flash high was stopped after two exceptionally slow batches.
The follow-up explicitly tested Grok 4.5 low, Claude Sonnet 5 low, and Qwen3.7
Max in addition to the initial ten-model screen.

All 330 four-model subsets of the 11 usable rankings were aggregated and evaluated against the PDF-aware
tournament gold. The selected Nemotron + Gemini Flash Lite + GPT-5.6 Luna +
Grok panel captured 10/10 gold top-10 papers by shortlist rank 20, 18/20 gold
top-20 by rank 30, and 24/30 gold top-30 by rank 30. The previous three-model
preset captured 17/20 and 23/30 at rank 30; all 11 usable models captured
18/20 and 23/30 at about four times the selected panel's cost. No equally
strong four-model subset was cheaper or faster. More models did not
monotonically improve recall.

The selected panel cost $0.784 for 50 papers and completed in 8.2 minutes of
summed model runtime. The longer 48-paper, two-partition production preflight
cost $1.53. Together these imply roughly $100-$215 for 6,617 papers, depending
mainly on representation length and partition count. After the bounded
concurrency preflight, use four concurrent model streams; observed throughput
projects to about 14-15 hours at the slowest model's rate.

An additional matched test compared Flash Lite versions on the same 50 papers
and prompt. Gemini 3.5 completed 50/50 for $0.101 including one locally
recoverable malformed-escape response; Gemini 3.1 completed 50/50 for $0.069.
With Nemotron, Luna, and Grok held fixed, 3.5 increased ensemble Spearman from
0.646 to 0.662, top-10 recall at rank 10 from 6/10 to 7/10, and top-20 recall at
rank 20 from 15/20 to 16/20. Both variants retained 10/10 gold top-10 papers by
rank 20 and 18/20 gold top-20 papers by rank 30. The expected full-run cost
increase is under $10 for two partitions.

The same matched contribution-routing test favored Gemini 3.1: primary-class
accuracy was 0.80 versus 0.72, and macro-F1 was 0.764 versus 0.602. Production
therefore uses 3.1 for routing and 3.5 for ranking.

## Ensemble First Pass and Conservative Finalists

First classify provisional contribution routes in small resumable batches. The
production input is deterministically hash-randomized so prompt position is
independent of the source manifest order:

```bash
python3 -m icml_ai_ac.cli classify-contributions \
  --manifest data/model_runs/icml_2026_full_launch/stage01_randomized_routing_manifest.jsonl \
  --model google/gemini-3.1-flash-lite \
  --batch-size 16 \
  --out data/metadata/icml_2026_contribution_classes_randomized.jsonl \
  --run-dir data/model_runs/icml_2026_contribution_classes_randomized
```

These labels route ranking batches; they are not treated as final scientific
categories. The four-model cheap ranking aggregate stores each model's class
votes, agreement rate, the provisional routing class, and a resolved ensemble
class for downstream category analysis.

The easiest production entrypoint is `score-pass1-ensemble`, which runs a
configured cheap-model preset and can immediately write the aggregate signal:

```bash
python3 -m icml_ai_ac.cli score-pass1-ensemble \
  --manifest data/metadata/icml_2026_scoring_anonymized.jsonl \
  --out-dir data/model_runs/icml_2026_pass1_cheap_ensemble \
  --model-preset production_2026_v2 \
  --model-workers 4 \
  --class-path data/metadata/icml_2026_contribution_classes_randomized.jsonl \
  --strategy class_round_robin \
  --batch-size 8 \
  --partitions 2 \
  --aggregate-out data/scores/icml_2026_pass1_cheap_ensemble_signal.jsonl \
  --aggregate-report data/scores/icml_2026_pass1_cheap_ensemble_signal.report.json
```

The aggregate JSONL stores `shortlist_metrics.aggregate_priority`,
`source_model_priorities`, `source_model_count`, source paths, advance votes,
and per-source rows. Later ranking layers can use this as a prior or audit
signal without rerunning the cheap calls.

All production model runs use stable run directories and content/configuration
fingerprints. Exact reruns reuse validated batches or cards and retry only
failed or missing units. HTTP 402 insufficient-credit responses stop the
remaining model stream, as do authentication, permission, incompatible-request,
and exhausted rate-limit responses. Replenish credits and invoke the identical
command without `--overwrite` to continue. See `docs/final_run_plan.md` for the
full operational and artifact contract.

The lower-level shortlist builder still accepts multiple score files directly.
It normalizes priority by source model so a model with more partitions does not
automatically dominate the ensemble:

```bash
python3 -m icml_ai_ac.cli build-shortlist \
  --scores data/scores/icml_2025_modelprobe_v2_qwen3_32b_recovered.jsonl \
  --scores data/scores/icml_2025_modelprobe_v2_qwen36_35b_a3b_noreason.jsonl \
  --scores data/scores/icml_2025_modelprobe_v2_qwen3_235b_a22b_2507_recovered.jsonl \
  --limit 45 \
  --min-per-class 3 \
  --class-path data/metadata/icml_2025_contribution_classes_qwen_chunks.jsonl \
  --out data/scores/icml_2025_pass1_ensemble_v2_3models_shortlist45_classbalanced3.jsonl
```

Run two independent strong semifinal rankings over the same complete paper set.
The accepted-50 commands below use the single-response diagnostic because the
set is small. Production uses `rank-pass2-batches` with identical
class-balanced batches, two partitions, exact coverage, a connected comparison
graph, and resumable state for both judges. Avoid using either judge to make an
aggressive final cut; the
accepted-50 tournament gold shows that the old stage-2 top-35 cut dropped two
tournament top-10 papers.

```bash
python3 -m icml_ai_ac.cli rank-pass2 \
  --manifest data/metadata/icml_2025_accepted_50_parsed.jsonl \
  --pass1 data/scores/icml_2025_pass1_ensemble_v2_3models_shortlist45_classbalanced3.jsonl \
  --provider openrouter \
  --model openai/gpt-5.6-terra \
  --reasoning-effort high \
  --limit 45 \
  --text-source scoring \
  --per-paper-char-budget 20000 \
  --max-output-tokens 12000 \
  --out data/scores/icml_2025_pass2_semifinal_terra.json \
  --run-dir data/model_runs/icml_2025_pass2_semifinal_terra
```

```bash
python3 -m icml_ai_ac.cli rank-pass2 \
  --manifest data/metadata/icml_2025_accepted_50_parsed.jsonl \
  --pass1 data/scores/icml_2025_pass1_ensemble_v2_3models_shortlist45_classbalanced3.jsonl \
  --provider openrouter \
  --model anthropic/claude-sonnet-5 \
  --reasoning-effort high \
  --limit 45 \
  --text-source scoring \
  --per-paper-char-budget 20000 \
  --max-output-tokens 12000 \
  --out data/scores/icml_2025_pass2_semifinal_sonnet5.json \
  --run-dir data/model_runs/icml_2025_pass2_semifinal_sonnet5
```

The local aggregation step fails if either ranking is incomplete. It preserves
source rows and model metadata for later analysis:

```bash
python3 -m icml_ai_ac.cli ensemble-semifinal-rankings \
  --ranking data/scores/icml_2025_pass2_semifinal_terra.json \
  --ranking data/scores/icml_2025_pass2_semifinal_sonnet5.json \
  --label terra \
  --label sonnet \
  --out data/scores/icml_2025_pass2_semifinal_ensemble.json \
  --report data/scores/icml_2025_pass2_semifinal_ensemble.report.json
```

Then build a conservative finalist set from the cheap ensemble plus semifinal
consensus:

```bash
python3 -m icml_ai_ac.cli select-finalists \
  --cheap data/scores/icml_2025_pass1_ensemble_v2_3models_shortlist45_classbalanced3.jsonl \
  --semifinal data/scores/icml_2025_pass2_semifinal_ensemble.json \
  --limit 45 \
  --semifinal-top 25 \
  --cheap-top 5 \
  --min-per-class 1 \
  --disagreement-saves 3 \
  --judge-disagreement-saves 3 \
  --out data/scores/icml_2025_finalists_selector_limit45.jsonl \
  --report data/scores/icml_2025_finalists_selector_limit45.report.json
```

Under randomized v3, the current cheap ensemble top-45 captured every gold
top-10 paper by rank 20 and 18/20 gold top-20 papers by rank 30. The old GPT
stage-2 top-35 cut retained only 8/10 and 15/20 by rank 30, respectively.
Production should therefore use `select-finalists` to preserve a wider pool for
PDF-aware cards and tournament adjudication.

The union of configured top, class, and disagreement preservation sets must fit
within `--limit`. The selector fails rather than silently dropping a requested
save; reduce the quotas or raise the finalist limit when that invariant is not
met.

For the full 5k-6k paper run, the tunable stage sizes should be set with these
arguments:

- Cheap semifinal pool: `build-shortlist --limit`, initially 15%-20% of parsed
  papers.
- Strong semifinal pool: `rank-pass2-batches --limit`, usually the full cheap
  shortlist, run once per judge with matching batch and partition settings;
  `ensemble-semifinal-rankings` requires exact matching coverage.
- Conservative finalist pool: `select-finalists --limit`, initially 150-300
  papers, plus `--semifinal-top`, `--cheap-top`, `--min-per-class`, and
  both disagreement-save controls.
- PDF-aware card pool: `rank-frontier-pdf-cards --paper-list`, pointed at the
  finalist JSONL.
- Swiss pool and final playoff: `rank-frontier-card-tournament --strategy
  swiss_playoff --top-n ... --swiss-rounds ... --playoff-top-n ...`.

Recommended first production defaults:

```text
cheap shortlist: top 20%
strong semifinal: Terra and Sonnet over all cheap-shortlisted papers,
                    size-8 batches, 2 class-stratified partitions
PDF-aware finalists: 250
Swiss pool: 150
Swiss rounds: 10
final dense playoff: 60
```

## Frontier PDF-Aware Reference

The accepted-50 reference has been upgraded beyond the original single
text-only GPT-5.4 ranking. Its historical PDF-aware reference workflow is:

1. Create one card per paper per frontier judge with `rank-frontier-pdf-cards`.
   For ICML 2026, each card sees the first 9 PDF pages plus
   `scoring_repr.txt`; use `--pdf-excerpt-pages 8` for originally submitted
   versions.
2. Use three judges:
   - OpenRouter `openai/gpt-5.5` with `--reasoning-effort xhigh`
   - OpenRouter `openai/gpt-5.4` with `--reasoning-effort high`
   - OpenRouter `anthropic/claude-opus-4.7` with `--reasoning-effort high`
3. Compress oversized PDF excerpts before upload. The stable GPT runs used:
   `--pdf-optimize-threshold-bytes 1000000 --pdf-settings /screen`.
4. Aggregate the cards with `rank-frontier-card-ensemble`, then synthesize a
   forced 1..50 ranking with `rank-frontier-card-synthesis`.
5. Finalize the top-rank gold ordering with `rank-frontier-card-tournament`,
   which compares all pairs in the synthesis top 30 using the three PDF-aware
   cards and aggregates win points.

For ICML 2026 production, the updated PDF-card panel is:

- OpenRouter `openai/gpt-5.6-sol` with `--reasoning-effort xhigh`
- OpenRouter `anthropic/claude-fable-5` with `--reasoning-effort high`
- OpenRouter `google/gemini-3.1-pro-preview` with `--reasoning-effort high`

Use `openai/gpt-5.6-sol` xhigh for card synthesis and tournament adjudication.
Run Fable with `--fallback-model anthropic/claude-opus-4.8
--fallback-reasoning-effort high`. Store both requested and served model IDs,
the primary failure, fallback reason, and cumulative usage.
Use a 12,000-token output ceiling for frontier cards. The production run showed
that 6,000 tokens can be exhausted by hidden reasoning before the structured
card is complete.
The refreshed panel passed a 3/3 one-paper live PDF/JSON smoke test. Sol, Fable,
and Gemini reported their requested model IDs and cost $0.258, $0.608, and
$0.065 respectively. In the 12-paper preflight, one Fable card required the
explicit Opus fallback; all 36 cards completed.

The 250-paper production pass completed 750/750 valid cards. Four Sol cards
required same-model 12,000-token repairs, and 29 Fable requests used the
explicit Opus fallback. Gemini's overall-priority scores were materially more
compressed and lenient than Sol or Fable, so downstream aggregation must use
within-judge ranks or calibrated percentiles rather than raw score means.

The production `rank-frontier-card-ensemble` output uses equal-weight,
tie-aware within-judge rank percentiles. Overall gold priority is primary;
broad impact, ML impact, technical soundness, evidence confidence, and novelty
break exact score ties. It requires complete matching paper coverage in every
stream and retains raw scores, source ranks, served models, fallbacks, and
judge disagreement. The focused 250-paper audit found 0.927-0.936 Spearman
correlation between the full ensemble and each leave-one-judge-out result.

Artifacts:

- `data/reference/icml_2025_accepted_50_frontier_cards_gpt55_xhigh.jsonl`
- `data/reference/icml_2025_accepted_50_frontier_cards_gpt54_high.jsonl`
- `data/reference/icml_2025_accepted_50_frontier_cards_opus47_high.jsonl`
- `data/reference/icml_2025_accepted_50_frontier_pdf_card_ensemble.json`
- `data/reference/icml_2025_accepted_50_frontier_pdf_synthesis_gpt55.json`
- `data/reference/icml_2025_accepted_50_frontier_pdf_tournament_gold.json`

The three card passes completed 150/150 rows. Final-row usage cost reported by
OpenRouter was about $33.82 total: $12.64 for GPT-5.5, $6.18 for GPT-5.4, and
$15.00 for Opus 4.7. This excludes failed transient attempts that returned no
usage and small smoke tests.

The corrected accepted-50 tournament ran 435 pairwise comparisons in 18
resumable GPT-5.5 xhigh batches over the synthesis top 30. It randomizes pair
assignment, A/B presentation, and evidence order; outcomes were 221 A wins and
214 B wins. Valid batches cost $8.52 and used 38 minutes of provider time. One
malformed response was rejected and retried for about $0.41. Ranks 31-50 remain
from the synthesis seed and should not be overinterpreted as
pairwise-adjudicated. The earlier seed-ordered artifact is retained only as a
legacy diagnostic.

Against randomized v3, the PDF card synthesis has Spearman 0.984 and 10/10
same-k top-10 recall. The current four-model cheap ensemble captures 10/10 gold
top-10 papers by rank 20 and 18/20 gold top-20 papers by rank 30. The old
strong-model top-35 cut captures only 8/10 and 15/20 by rank 30, respectively.
This confirms that conservative advancement, rather than aggressive
intermediate reranking, is the important recall safeguard.

For larger production finalist pools, a hybrid sparse tournament is the current
cost-control candidate. Re-evaluate it from each current all-pairs artifact
rather than carrying forward legacy schedule metrics:

```bash
python3 -m icml_ai_ac.cli simulate-hybrid-tournament \
  --tournament data/reference/icml_2025_accepted_50_frontier_pdf_tournament_gold_randomized_v3.json \
  --out data/evals/icml_2025_randomized_v3_swiss10_playoff20.json \
  --swiss-rounds 10 \
  --playoff-top-n 20
```

The simulator replays the production Swiss pairing and playoff logic using
already-paid all-pairs outcomes. On randomized v3, 10 Swiss rounds plus a
top-20 dense playoff used 255/435 comparisons, achieved Spearman 0.996, retained
10/10 top-10 and 19/20 top-20 papers, and put every all-pairs top-10 paper in
its top 15. It estimates schedule loss without new model calls, but assumes a
pair judgment would not change under different batch context. Pure
seed-neighborhood round robins were much weaker and should not be used as the
only adjudication step.

Production hybrid tournament template:

```bash
python3 -m icml_ai_ac.cli rank-frontier-card-tournament \
  --cards data/reference/icml_2026_frontier_cards_gpt56_sol_xhigh.jsonl \
  --cards data/reference/icml_2026_frontier_cards_fable5_high.jsonl \
  --cards data/reference/icml_2026_frontier_cards_gemini31pro_high.jsonl \
  --seed-ranking data/reference/icml_2026_frontier_pdf_synthesis.json \
  --out data/reference/icml_2026_frontier_pdf_hybrid_tournament.json \
  --run-dir data/model_runs/icml_2026_frontier_pdf_hybrid_tournament \
  --provider openrouter \
  --model openai/gpt-5.6-sol \
  --reasoning-effort xhigh \
  --strategy swiss_playoff \
  --top-n 150 \
  --swiss-rounds 10 \
  --playoff-top-n 60 \
  --pairs-per-batch 25
```

Use `--strategy all_pairs --top-n 50` or `--strategy all_pairs --top-n 60` when
the final candidate set is small enough that the cleaner all-pairs design is
affordable.

Tournament v3 deterministically randomizes pair-to-batch assignment, A/B
presentation, and paper evidence-block order. Do not reuse a v1/v2 run
directory for a v3 tournament.

## Rubric Shape

The first-pass prompt separates:

- conventional reviewer strength
- technical soundness and methodological rigor
- empirical credibility and theoretical depth
- novelty, clarity, and reproducibility
- ML-field impact forecast
- broader scientific impact forecast
- impact type axes such as theory, systems, data/benchmark, safety, and science
- executive-AC priority
- false-negative likelihood if the paper had been rejected
- whether the paper should advance to the strong-model ranking pass
- observed weaknesses versus unverified risks from missing/truncated text

This is deliberate: later analysis can compare "AI likes this because it is
reviewer-solid" against "AI likes this because it could matter broadly."

## Dry Run

Generate prompts without calling a provider:

```bash
python3 -m icml_ai_ac.cli score-pass1 \
  --manifest data/metadata/icml_2025_accepted_50_parsed.jsonl \
  --limit 5 \
  --provider openrouter \
  --model qwen/qwen3.7-plus \
  --text-source scoring \
  --dry-run \
  --out data/scores/icml_2025_pass1_sample_dryrun.jsonl \
  --run-dir data/model_runs/icml_2025_pass1_sample_dryrun
```

## Live Run

Set `OPENROUTER_API_KEY`, then run:

```bash
python3 -m icml_ai_ac.cli score-pass1 \
  --manifest data/metadata/icml_2025_accepted_50_parsed.jsonl \
  --limit 5 \
  --provider openrouter \
  --model qwen/qwen3.7-plus \
  --text-source scoring \
  --out data/scores/icml_2025_pass1_sample.jsonl \
  --run-dir data/model_runs/icml_2025_pass1_sample \
  --delay 1 \
  --temperature 0.2
```

The run directory contains:

- `prompts/*.json`
- `responses/*.json`
- `parsed/*.json`
- `errors/*.json`
- `run.json`

The scores JSONL stores paths to those artifacts, validation status, token
estimates, provider usage when available, and optional cost estimates.

## Low-Level Batch Triage

Use `score-pass1-ensemble` for production. The lower-level
`score-pass1-batch` command is mainly useful for debugging one model on one
batch with forced ranks and buckets:

```bash
python3 -m icml_ai_ac.cli score-pass1-batch \
  --manifest data/metadata/icml_2025_accepted_50_parsed.jsonl \
  --provider openrouter \
  --model qwen/qwen3.7-plus \
  --limit 8 \
  --text-source scoring \
  --per-paper-char-budget 10000 \
  --out data/scores/icml_2025_pass1_batch8_qwen3_32b.jsonl \
  --run-dir data/model_runs/icml_2025_pass1_batch8_qwen3_32b
```

This prompt forces unique ranks and score buckets, and the parser validates that
every expected paper id appears exactly once. Small case-only or one-edit paper
id typos can be repaired against the known candidate set; duplicated or missing
coverage still fails.

For a full paper set with one explicit model, run multiple small batches and
preserve per-batch artifacts:

```bash
python3 -m icml_ai_ac.cli score-pass1-batches \
  --manifest data/metadata/icml_2025_accepted_50_parsed.jsonl \
  --provider openrouter \
  --model qwen/qwen3.7-plus \
  --text-source scoring \
  --per-paper-char-budget 10000 \
  --batch-size 8 \
  --partitions 3 \
  --strategy class_round_robin \
  --class-path data/metadata/icml_2025_contribution_classes_qwen_chunks.jsonl \
  --out data/scores/icml_2025_pass1_batches_rr8x3_qwen3_32b.jsonl \
  --run-dir data/model_runs/icml_2025_pass1_batches_rr8x3_qwen3_32b
```

Aggregate repeated local rankings into a unique shortlist:

```bash
python3 -m icml_ai_ac.cli build-shortlist \
  --scores data/scores/icml_2025_pass1_batches_rr8x3_qwen3_32b.jsonl \
  --limit 30 \
  --min-per-class 2 \
  --class-path data/metadata/icml_2025_contribution_classes_qwen_chunks.jsonl \
  --out data/scores/icml_2025_pass1_batches_rr8x3_qwen3_32b_shortlist30_classbalanced2.jsonl
```

## Retry Failed Rows

Provider and JSON failures are row-local. Re-attempt failed papers directly:

```bash
python3 -m icml_ai_ac.cli score-pass1 \
  --manifest data/metadata/icml_2025_accepted_50_parsed.jsonl \
  --provider openrouter \
  --model qwen/qwen3.7-plus \
  --text-source scoring \
  --paper-id DkRYImuQA9 \
  --paper-id NWKjVzkDzg \
  --out data/scores/icml_2025_pass1_retry.jsonl \
  --run-dir data/model_runs/icml_2025_pass1_retry
```

## Single-Judge Strong Ranking

`rank-pass2` runs one strong judge. Production runs it once for Terra and once
for Sonnet as shown above; this smaller command is useful for diagnostics:

```bash
python3 -m icml_ai_ac.cli rank-pass2 \
  --manifest data/metadata/icml_2025_accepted_50_parsed.jsonl \
  --pass1 data/scores/icml_2025_pass1_20_qwen3_32b_v3_scoring_with_retries.jsonl \
  --provider openrouter \
  --model openai/gpt-5.6-terra \
  --reasoning-effort high \
  --limit 10 \
  --text-source scoring \
  --per-paper-char-budget 36000 \
  --out data/scores/icml_2025_pass2_top10_gpt56_terra.json \
  --run-dir data/model_runs/icml_2025_pass2_top10_gpt56_terra
```

In the ICML 2025 pilot, cheap models were useful for pulling plausible
candidates into a shortlist, but their absolute scores were compressed. A
strong-model pass is the first comparative ordering step; production uses the
two-judge consensus rather than treating either ordering as final.

## Historical Text-Only Reference Ranking

The following command reproduces the obsolete text-only ICML 2025 reference.
Current evaluation should use the PDF-aware tournament gold described above:

```bash
python3 -m icml_ai_ac.cli rank-reference \
  --manifest data/metadata/icml_2025_accepted_50_parsed.jsonl \
  --provider openai \
  --model gpt-5.4 \
  --paper-set-name icml_2025_accepted_50 \
  --text-source scoring \
  --per-paper-char-budget 24000 \
  --out data/reference/icml_2025_accepted_50_gold.json \
  --run-dir data/model_runs/icml_2025_accepted_50_gold_gpt54
```

The command validates exact paper-id coverage and ranks 1-N. If a long JSON
response duplicates or omits a paper id, repair from the saved parsed response
with `repair-reference`; do not hand-edit the gold file.
