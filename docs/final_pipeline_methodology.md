# Final Pipeline Methodology

This document describes the current production pipeline for identifying
high-impact ICML papers. It is written as a concise methods section: the running
decision history remains in `docs/methodology_decisions.md`.

## Stage Numbering

The stage numbers below are the implementation's own, and they match the audit
filenames (`docs/stage3_production_audit.md`,
`docs/stage5_production_audit.md`, `docs/stage6_production_audit.md`) and the
`stage3_*`, `stage5_*`, and `stage7_*` data artifacts. The public results site
presents a simplified six-step version of the same pipeline, which merges the
preparation and selection stages and splits the tournament into its two phases.
The two numbering schemes correspond as follows:

| Results site | This document |
| --- | --- |
| Stage 01, Anonymize and label | Paper Representation (identity redaction) and Stage 1 |
| Stage 02, First-pass ranking | Stage 2 |
| Stage 03, Second-pass ranking | Stages 3 and 4 |
| Stage 04, Full-paper judgments | Stage 5 |
| Stage 05, Swiss-system rounds | Stage 6, Swiss phase |
| Stage 06, Round robin | Stage 6, all-pairs phase |
| (reported as results, not a stage) | Stage 7 |

Use this document's numbering when referring to code, commands, audits, or
stored artifacts.

## Objective

The pipeline ranks papers by expected broad scientific and machine-learning
impact, not by conventional reviewer score alone. The target output is an
auditable overall ranking plus category-specific rankings for distinct
contribution routes: algorithms, theory, benchmarks/datasets, infrastructure,
scientific tools, safety/evaluation, applications, and analysis papers.

All model stages are paper-grounded. Retrieval, web search, citation memory,
and external tools are excluded from scoring. Reviews, reviewer scores,
area-chair comments, decisions,
presentation tiers, and awards are structurally excluded. The only allowed
evidence is the paper identifier and title, extracted paper text, PDF-derived
visual/table evidence where explicitly supplied, and previous AI pipeline
judgments. Human outcomes are joined only after the AI ranking is frozen.

## Paper Representation

Each paper is stored with metadata, PDF path, and parse status. PDF text is
extracted into multiple representations. The main scoring representation uses
the ICML page-limit prior, then removes references and detected
appendix/supplement material. For ICML 2026 camera-ready papers this is the
first 9 PDF pages; for originally submitted versions it should be set to 8
pages. This reduces accidental inclusion of non-main-paper content while
preserving experiments, tables, baselines, ablations, and method details that
were missing from earlier compact representations.

Before scoring, a separate resumable stage creates identity-redacted derivatives
of the first nine PDF pages and all text representations. It removes a
layout-validated first-page identity band, author-name occurrences, affiliation
and correspondence lines, emails, acknowledgements, mail links,
source/reviewer footers, and PDF author/XML metadata. The detector handles
fragmented superscripts, camera-ready middle-name drift, and already-anonymous
bylines, while rejecting implausibly short uppercase metadata identities.
Canonical PDFs and parsed text remain immutable. Each derivative is
fingerprinted against its source, checked for residual known identities and
email addresses, and excluded from scoring unless validation passes. Source
repair diagnostics and metadata-to-byline name drift remain auditable.

For PDF-aware stages, the pipeline uses the identity-redacted PDF derivative.
Oversized excerpts can be compressed before upload so that image-heavy papers
do not cause provider failures. Text remains included alongside the PDF
excerpt. This removes direct identity cues, not all possible identification:
titles, self-citations, project names, writing style, and model memory may still
identify a paper.

The frozen ICML 2026 scoring population contains 6,617 of 6,628 indexed papers:
6,614 official OpenReview PDFs and three validated high-confidence arXiv
fallbacks. Three official-PDF parser warnings were reviewed and retained
because their abstract and main body text are intact. The remaining eleven
papers have no usable PDF and are excluded with identifiers and reasons stored
in `data/metadata/icml_2026_scoring_manifest.report.json`.

## Stage 1: Contribution Routing

Gemini 3.1 Flash Lite first classifies papers by contribution route using
compact title, abstract, introduction, and conclusion-style text. These labels
support stratified batching and class-balanced selection; they are not quality
judgments. Classification runs in small fingerprinted batches, and reruns reuse
valid batches while retrying only missing or invalid ones.

## Stage 2: Cheap Ensemble Triage

All parsed papers are evaluated by the `production_2026_v2` OpenRouter
ensemble: `nvidia/nemotron-3-ultra-550b-a55b`,
`google/gemini-3.5-flash-lite`, `openai/gpt-5.6-luna`, and
`x-ai/grok-4.3`. The design uses listwise forced-ranking batches rather than
independent absolute 1-10 scores, because early Qwen runs produced compressed
score distributions. The cheap ensemble's goal is high recall into the
strong-model stages; its ordering is not trusted for headline ranking. Each
paper is placed in two deterministic class-stratified partitions by default.
Prompts, responses, model IDs, usage, and fingerprints are persisted for
resume after interruption.

The shortlist builder aggregates multiple cheap score files, normalizes by
source model so one model cannot dominate by producing more rows, and selects a
class-balanced semifinal pool. A paired fixed-effect sensitivity analysis
estimates each model's input-slot effect from the two observations of every
paper. Production advances the union of the raw and adjusted top 20%, retaining
both signals rather than treating the adjustment as ground truth. It stores
the ensemble signal, including
aggregate priority, source-model priorities, advance votes, and source rows, so
later ranking layers can reuse the cheap evidence. The key production knobs are:

```text
score-pass1-ensemble --model-preset
score-pass1-ensemble --model-workers 4
score-pass1-ensemble --aggregate-out
build-shortlist --limit
build-position-robust-shortlist --cutoff
```

For the 2026 full run, the raw and adjusted top-1,323 sets overlap on 1,204
papers; their union sends 1,442 papers to the strong semifinal.

## Stage 3: Strong Semifinal Ranking

Two independent text-only judges evaluate the same complete cheap shortlist
over main-paper scoring text: GPT-5.6 Terra high and Claude Sonnet 5 high,
both through OpenRouter with exact served-model checks. Corpus-wide single responses are infeasible,
so each judge uses the same repeated, class-stratified listwise batches. We
average each paper's normalized local rank across partitions, require one valid
judgment per partition, and then average the judges' global normalized ranks.
Every batch is fingerprinted and resumable. Exact partition coverage and a
connected overlapping-batch comparison graph are required before aggregation.
Model priority scores are used only to break within-judge local-rank ties.
Complete 0-100 batches are rescaled to the intended 0-10 range before this
tie-break; raw values and detected scales remain stored.

The output retains source judgments, prompts, requested and served model IDs,
local and global ranks, within-judge batch variation, and between-judge
disagreement.

Equal weighting avoids an unsupported calibration claim. Normalized ranks avoid
mixing model-specific absolute score scales. Consensus ties prefer lower judge
disagreement; disagreement is retained separately so the selector can preserve
papers strongly favored by either judge. This stage improves comparative
ordering and rationales but is not allowed to make a narrow final cut.

The ICML 2026 production audit, including the scale correction, structural
checks, position sensitivity, and judge-agreement results, is recorded in
`docs/stage3_production_audit.md`.

The production commands are:

```text
rank-pass2-batches --provider openrouter --model openai/gpt-5.6-terra --reasoning-effort high
rank-pass2-batches --provider openrouter --model anthropic/claude-sonnet-5 --reasoning-effort high
ensemble-semifinal-rankings --ranking ... --ranking ... --label terra --label sonnet
```

The recommended default gives both judges the same complete shortlist, size-8
batches, two partition seeds, and contribution labels.
Claude Opus 4.8 is an optional adjudicator for high-disagreement cases or a
small robustness sample, not a default full-shortlist judge.

## Stage 4: Conservative Finalist Selection

Finalists are selected deterministically from the cheap ensemble and semifinal
consensus. The selector keeps top consensus papers, cheap-stage leaders,
per-class leaders, cheap/semifinal disagreement saves, and configurable
Terra/Sonnet disagreement saves. This replaces the previous GPT stage-2 top-N
cut.

The tunable knobs are:

```text
select-finalists --limit
select-finalists --semifinal-top
select-finalists --cheap-top
select-finalists --min-per-class
select-finalists --disagreement-saves
select-finalists --judge-disagreement-saves
```

For the full run, the initial target is 150-300 PDF-aware finalists.
The union of preservation sets must fit within the finalist limit; configuration
fails closed rather than silently discarding a requested class or disagreement
save.

## Stage 5: PDF-Aware Frontier Cards

Only the conservative finalist pool receives expensive PDF-aware review. Each
frontier judge sees the configured first-page PDF excerpt plus extracted main-paper text
and writes a structured judgment card with contribution class, impact route,
scores, best-case impact, main risk, and visual/table evidence notes.

The production panel uses GPT-5.6 Sol xhigh, Claude Fable 5 high, and Gemini
3.1 Pro high. This provides independent OpenAI, Anthropic, and Google judgments.
Each card stores both the requested and served model ID so provider routing is
auditable. The important design principle is to pay the PDF cost once per
finalist and reuse the resulting cards for ranking.

Fable has an explicit Opus 4.8 high fallback. A failed primary attempt and its
usage remain stored alongside the fallback reason and successful fallback card;
provider routing is never silently treated as a Fable judgment.

One-paper live smoke cards completed 3/3 with valid structured outputs and no
served-model substitutions. Reported costs were $0.258 for Sol, $0.608 for
Fable, and $0.065 for Gemini, or $0.93 total. This is an interface/cost check,
not a claim that the refreshed panel has independently replaced the historical
accepted-50 gold. A 250-finalist run projects to roughly $233 at this observed
paper size.

In the 12-paper production preflight, all 36 cards completed; one Fable paper
required the explicit Opus fallback. Successful card usage was about $12.00,
which projects to roughly $250 for 250 finalists before paper-size variation.

The finalist file flows directly into card generation:

```text
rank-frontier-pdf-cards --paper-list finalists.jsonl
```

The production pass completed 750/750 cards. Because Gemini's score scale was
substantially higher and narrower than the Sol and Fable-panel scales, the
three streams are combined by equal-weight, tie-aware within-judge rank
percentiles. Overall gold priority is the primary within-judge key; broad
impact, ML impact, technical soundness, evidence confidence, and novelty break
exact score ties. Raw scores and judge disagreement remain in the artifact but
do not determine the ensemble seed.

A focused production audit found complete 250-paper coverage. Each judge had
Spearman correlation 0.72-0.85 with the ensemble; omitting any one judge left
Spearman correlation 0.927-0.936 with the full result. The 29 explicit
Fable-to-Opus fallbacks were not elevated as a group. A deterministic sample
of consensus leaders, prior-stage leaders, and the largest disagreements found
paper-specific evidence and limitations rather than malformed or generic
judgments. This ensemble is a transparent tournament seed and robustness
baseline, not the final ranking.

## Stage 6: Hybrid Tournament Ranking

The final ranking uses card evidence rather than raw PDFs, with GPT-5.6 Sol
xhigh as the synthesis and pairwise adjudication model. For small finalist sets,
all-pairs comparison is preferred. For larger finalist sets, the current
cost-control design is a hybrid tournament:

1. Run Swiss pairwise comparisons over a broad card-ranked pool.
2. Select a provisional top subset from Swiss standings.
3. Run all-pairs comparisons only within that subset.
4. Use the dense playoff as the headline ranking and Swiss standings as
   supporting evidence outside the playoff.

The production Sol synthesis completed all 250 ranks in one validated request,
using 277,774 prompt tokens and 81,189 completion tokens, including 15,611
reasoning tokens. It cost $6.43 and took 23.4 minutes. The synthesis and
rank-calibrated ensemble had Spearman 0.848 and shared 128/150 papers at their
respective top-150 cuts. To avoid making this inexpensive disagreement a hard
recall loss, the production Swiss pool is their 172-paper union. This adds 110
Swiss comparisons relative to a 150-paper pool and does not change the
60-paper playoff size.

The synthesis seed chooses the tournament pool but is not shown as a rank.
Pair assignment to batches, A/B presentation, and evidence-block order are
separately and deterministically randomized. This prevents seed order and
position from becoming implicit adjudication signals while preserving exact
reproducibility.

Fit regularized Bradley-Terry models separately to the Swiss schedule and the
dense playoff. The Swiss fit adjusts provisional standings for opponent
difficulty under an uneven schedule. The playoff remains directly auditable by
win rate because every finalist faces every other finalist; its Bradley-Terry
ordering and conditional standard errors are reported as a sensitivity
analysis. These intervals are conditional on the observed judgments and do not
represent uncertainty over prompts, judge models, or model families.

The tunable knobs are:

```text
rank-frontier-card-tournament --strategy
rank-frontier-card-tournament --top-n
rank-frontier-card-tournament --swiss-rounds
rank-frontier-card-tournament --playoff-top-n
rank-frontier-card-tournament --pairs-per-batch
rank-bradley-terry --stage auto
```

Recommended first production defaults:

```text
PDF-aware finalists: 250
Swiss pool: union of synthesis top 150 and card-ensemble top 150 (172 observed)
Swiss rounds: 10
Dense playoff: 60
```

## Stage 7: Human-Outcome Comparison

Human-outcome metadata is stored separately from every scoring prompt and is
joined only in this stage. This preserves a paper-only counterfactual ranking
while still allowing complete post hoc agreement and divergence analysis.

The comparison layer consumes a unique, full-coverage ensemble ranking rather
than raw repeated model judgments or a truncated shortlist. It fails when the
manifest join falls below the declared coverage threshold and reports missing
and extra IDs. Analyses are separated by human signal:

1. presentation tier and awards among accepted papers;
2. published reviewer overall scores where available; and
3. acceptance versus the public opt-in rejected subset.

Agreement reports include tier-aware recall, ROC-AUC with deterministic
stratified-bootstrap intervals, Kendall tau-b, and per-tier distributions.
Strong-stage agreement is paired: the cheap and strong rankings are evaluated
on the same finalist subset, while advancement recall records human-honored
papers lost before that stage.

Divergence galleries use preregistered AI and human rank tails rather than
labeling every poster or honored paper as divergent. Qualitative coding receives
the paper title, abstract, contribution class, human outcome, and AI axes.
Candidate codes are induced on a balanced sample, reviewed and frozen, then
applied to the full divergence tail. Held-out summaries exclude codebook
examples. A second model reports per-code Cohen's kappa together with code
prevalence and raw agreement; failed or invalid classifications are excluded
rather than treated as negative labels.

## Gold-Set Evaluation

The current diagnostic reference is a 50-paper ICML 2025 accepted-paper set. The
old reference was a single GPT-5.4 text-only ranking; it is now a historical
baseline only. The upgraded reference uses:

1. Three independent PDF-aware frontier cards per paper.
2. A GPT-5.5 synthesis over the three cards.
3. A GPT-5.5 xhigh all-pairs tournament over the synthesis top 30.

The corrected tournament completed 435/435 comparisons in 18 resumable batches.
Pair batching, A/B presentation, and evidence order were deterministically
randomized; outcomes were balanced at 221 A wins and 214 B wins. Valid batches
cost $8.52 in reported OpenRouter usage and used 38 minutes of provider time.
One malformed response was rejected and retried, adding about $0.41 without
repeating valid batches. Ranks 1-30 are pairwise-adjudicated; ranks 31-50 remain
from the synthesis seed. The connected regularized Bradley-Terry fit had
Spearman 0.997 against direct all-pairs points, shared 9/10 top-10 papers, and
moved no paper by more than two positions.

The July 25 production preflight used 48 real ICML 2026 papers stratified across
awards, orals, spotlights, regular-paper review-score bands, and missing scores.
The cheap top-32 retained all nine award papers. Twelve finalists completed
36/36 frontier cards and 66/66 pairwise comparisons. A/B wins were balanced
32/34; direct all-pairs and Bradley-Terry orderings agreed exactly. Card
synthesis achieved Spearman 0.972 and top-5/top-10 recall 1.0 against the
tournament. Semifinal and cheap ordering achieved Spearman 0.664 and 0.546.
These figures test pipeline behavior, not human-alignment prevalence, because
the sample was deliberately stratified.

Against this tournament gold:

| Candidate pipeline artifact | Main result |
| --- | --- |
| Old text-only gold | Spearman 0.290 |
| PDF card ensemble | Spearman 0.962; 9/10 same-k top-10 |
| PDF card synthesis | Spearman 0.984; 10/10 same-k top-10 |
| Current four-model cheap top45 | 10/10 gold top-10 by rank 20; 18/20 gold top-20 by rank 30 |
| Old GPT stage-2 top35 cut | 8/10 gold top-10 and 15/20 gold top-20 by rank 30 |
| Swiss-10 plus top-20 playoff simulation | 255/435 comparisons; Spearman 0.996; 10/10 top-10 and 19/20 top-20 |

The principal empirical lesson is that the cheap ensemble is already effective
as a generous recall filter, while the aggressive strong-model narrowing step
was the main recall failure. The final pipeline therefore favors conservative
candidate preservation before PDF-aware carding and tournament adjudication.
The Terra/Sonnet ensemble was added after these accepted-50 artifacts. Its
incremental value should therefore be measured on the production finalists
rather than inferred from non-identical historical candidate sets.

## July 2026 Model Refresh

Before the ICML 2026 run, 14 current OpenRouter models were screened on the
accepted-50 set. Eleven produced usable complete or near-complete rankings.
Step 3.7 Flash and Gemini 3.5 Flash rejected the shared reasoning-off
configuration, DeepSeek V4 Pro returned one malformed final batch, and
DeepSeek V4 Flash high was stopped after two very slow batches. Across all 330
four-model subsets of the 11 usable rankings, the selected production panel
gave the best practical recall/reliability/runtime tradeoff:

| Cheap panel | Gold top-10 in top 20 | Gold top-20 in top 30 | Gold top-30 in top 30 | Accepted-50 cost |
| --- | ---: | ---: | ---: | ---: |
| Previous three-model preset | 10/10 | 17/20 | 23/30 | $0.31 |
| Initially selected four-model preset (Gemini 3.1) | 10/10 | 18/20 | 24/30 | $0.78 |
| All 11 usable models | 10/10 | 18/20 | 23/30 | $3.05 |

The production panel is therefore limited to four. No other four-model subset
that matched its recall and category coverage was cheaper or faster. Grok 4.5
low, Claude Sonnet 5 low, and Qwen3.7 Max were specifically tested and did not
improve first-pass recall enough to justify replacing a selected model. At the
observed context budget, the panel's accepted-50 cost projects to roughly $104
for 6,617 papers.

A later matched Flash Lite bakeoff held the other three production streams
fixed. Replacing Gemini 3.1 Flash Lite with Gemini 3.5 Flash Lite increased
ensemble Spearman from 0.646 to 0.662 and improved same-budget gold top-10 and
top-20 recall at rank 10 and rank 20, while both variants retained all gold
top-10 papers by rank 20 and 18/20 gold top-20 papers by rank 30. In this
matched aggregate-ordering comparison, gold top-30 same-k recall decreased from
23/30 to 22/30, which is secondary to the stage's top-tail recall objective.
Gemini 3.5 uses `minimal` reasoning because its
OpenRouter endpoint rejects reasoning-off requests; returned reasoning is
excluded. Gemini 3.1 remains the routing classifier after outperforming 3.5 on
primary contribution class accuracy (0.80 versus 0.72) and macro-F1 (0.764
versus 0.602).

## Tournament Cost Evaluation

The all-pairs top-30 tournament is the highest-confidence reference. To estimate
whether a cheaper production tournament is viable, `simulate-hybrid-tournament`
replays the production Swiss pairing and playoff logic from already-paid
all-pairs outcomes. This measures schedule loss without new model calls, though
it assumes pair judgments are stable across batch contexts. On randomized v3,
10 Swiss rounds followed by a top-20 dense playoff used 255/435 comparisons,
recovered 10/10 all-pairs top-10 and 19/20 top-20 papers, and had Spearman
0.996. Pure neighborhood round-robin schedules were much weaker.

This supports Swiss as a broad reranking step before the final dense playoff,
not as the final ranking by itself.

## Auditability

Every stage writes resumable artifacts: prompts, raw provider responses, parsed
JSON, validation errors, run metadata, costs when reported by the provider, and
output rankings. Later invocations reuse successful paper-level or batch-level
artifacts and retry only missing or failed units. Evaluation uses top-k recall,
gold-top-N-in-candidate-top-M recall, rank correlation over shared papers, and
gold-anchored contribution-class recall.

Provider-wide request failures stop a model stream rather than generating
thousands of repeated failures. This includes insufficient-credit HTTP 402
responses as well as authentication, permission, incompatible-request, and
exhausted rate-limit errors. Repeating the same command with the same run
directory resumes from matching fingerprints. Frontier-card fingerprints cover
the model, prompt version, decoding settings, paper text, and PDF provenance,
preventing an old successful card from being silently reused under a changed
configuration. The concise operational checklist is
`docs/final_run_plan.md`.
