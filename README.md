# AI as Area Chair: What Science Would AI Send to ICML?

## Motivation

Our scientific literature is increasingly co-planned, co-executed, and
co-written with AI [1]. Our reviews, rebuttals, and likely even final decisions
are increasingly shaped by AI assistance [2]. This influence, for better or
worse, is rising quickly and remains poorly understood.

As a natural experiment, we take this to the extreme: what if ICML peer review
and acceptance were decided *only* by AI? In other words, what if the
progress of science were shaped by AI?

To study this question, we crawl ICML 2026's public papers and ask AI systems to
judge which papers are most technically strong, most impactful for machine
learning, and most important for scientific progress. We compare AI's rankings
to those of human reviewers (the spotlights, orals, and paper awards).

## Research Questions

1. Which papers would AI rank as the best contributions to science and the
   future of machine learning?
2. Which rejected papers would AI identify as overlooked contributions?
3. How different are these rankings from the conference's human-led outcomes?
4. How sensitive are these rankings to model choice, prompt design, and
   contamination controls?

## Current Data Status

As of **2026-07-15**:

- ICML's public 2026 virtual-site index contains **6,628 accepted poster-paper
  rows**, including **574 spotlights** and **168 papers with oral
  presentations**. Oral event rows are merged into their canonical poster
  records rather than counted twice.
- **6,614/6,628** accepted papers have authenticated OpenReview forum PDFs.
  All 6,614 official files were downloaded and validated; parsing produced
  6,611 `ok`, three `needs_review`, and zero failures. The remaining 14 records
  have no OpenReview forum ID rather than failed PDF requests.
- The frozen scoring manifest contains **6,617/6,628 papers (99.83%)**:
  6,614 official PDFs plus three validated high-confidence arXiv fallbacks.
  The three parser-review cases were inspected and included because their
  abstracts and main bodies are intact. Eleven records have no usable PDF and
  are listed explicitly in the manifest report.
- The current metadata snapshot includes published reviewer ratings for
  **6,341** accepted papers: overall assessment, soundness, and confidence.
- Official awards are linked for two Outstanding Papers, five Honorable
  Mentions, one Outstanding Position Paper, and one position-paper Honorable
  Mention.
- Public OpenReview rejected-note queries currently return **214 rejected
  notes**. Rejected-paper analysis therefore applies only to the public opt-in
  rejected subset, not all rejected submissions.
- arXiv is useful as fallback and provenance enrichment, but not as the primary
  PDF source. In a 500-paper pilot, high-confidence arXiv coverage was 329/500.
  Among the top 50 accepted papers by public reviewer score, arXiv coverage was
  35/50 after deep-search retry.
- ICML reported **497 desk rejects, about 2% of submissions**, implying roughly
  **25,000 total submissions** overall.

## Methodology

All ranking stages are paper-only: reviews, reviewer ratings, area-chair
comments, decisions, presentation tiers, and awards are excluded from model
inputs and joined only after the AI ranking is frozen. Retrieval and web search
are disabled. Before scoring, validated derivatives remove the author byline,
affiliations, emails, acknowledgements, and PDF author metadata. The canonical
PDFs remain unchanged.

### 1. Corpus and PDF Processing

We use the ICML virtual-site JSON feed as the accepted-paper inventory, resolve
official PDFs through OpenReview forum IDs, and download accepted plus public
opt-in rejected PDFs. arXiv is fallback only, with separate provenance.

PDFs are parsed with Poppler. For ICML 2026 camera-ready papers, we use the
first **9 PDF pages** as the main-paper prior, then strip references and detected
appendix/supplement material; original submissions use 8 pages. The pipeline
writes compact, scoring, and full paper representations. Substantive scoring
uses the main-paper `scoring_repr`. Identity-redacted PDF and text derivatives
are fingerprinted, validated, and fail closed before entering scoring.

### 2. Cheap High-Recall Triage

Every parsed paper is scored by a low-cost OpenRouter ensemble:

- `nvidia/nemotron-3-ultra-550b-a55b`
- `google/gemini-3.5-flash-lite`
- `openai/gpt-5.6-luna`
- `x-ai/grok-4.3`

This stage uses listwise forced-ranking batches rather than independent 1-10
scores, which were too compressed in early tests. Models judge technical
soundness, novelty, clarity, empirical credibility, ML-field impact, broader
scientific importance, contribution type, and whether the paper should advance.

Gemini 3.1 Flash Lite first assigns provisional contribution routes in a fixed
hash-randomized order, preventing source order from being confounded with prompt
position. The four-model ranking pass also stores its class votes and agreement,
which define the downstream ensemble category signal.

We aggregate by source model and advance a top **15-20%** class-balanced
shortlist. A within-paper position sensitivity check expands the handoff when
raw and position-adjusted cutoffs disagree. This pass is a recall filter, not
the final ranking. Classification and ranking batches are fingerprinted and
resumable; reruns reuse valid responses and retry only unfinished batches.

After a bounded preflight, the four model streams can run concurrently with
`--model-workers 4`; batches within each stream remain sequential.

### 3. Strong Semifinal and Conservative Finalists

The complete shortlist is independently evaluated over `scoring_repr` by:

- `openai/gpt-5.6-terra` through OpenRouter with `high` reasoning
- `anthropic/claude-sonnet-5` with `high` reasoning

Each judge uses two deterministic, class-stratified listwise partitions instead
of one infeasible corpus-wide response. We average normalized local ranks
within judge, then combine judges with equal weight. We retain batch judgments,
requested/served model IDs, and disagreement signals. A run fails closed unless
every paper is judged in every partition and the overlapping batches form one
connected comparison graph. Because our ICML 2025
gold-set evaluation found that aggressive intermediate cuts lost true top
papers, this stage improves ordering but does not make a narrow cut.

Finalists combine the consensus rank, cheap-ensemble and per-class leaders, and
both cheap/strong and Terra/Sonnet disagreement saves. The initial full-run
target is **150-300 PDF-aware finalists**. Claude Opus 4.8 is reserved for
optional disagreement adjudication rather than the full semifinal pool.

### 4. PDF-Aware Frontier Judgment

For finalists only, three model families review a first-9-page PDF excerpt plus
extracted main-paper text and produce independent structured judgment cards:

- `openai/gpt-5.6-sol` with `xhigh` reasoning
- `anthropic/claude-fable-5` with `high` reasoning
- `google/gemini-3.1-pro-preview` with `high` reasoning

Requested and served model IDs are stored for every card. A failed or
safeguarded Fable request explicitly falls back to
`anthropic/claude-opus-4.8` high; both attempts, the fallback reason, model IDs,
and cumulative usage remain attached to the paper. The historical ICML 2025
gold-set protocol used:

- `openai/gpt-5.5` with `xhigh` reasoning
- `openai/gpt-5.4` with `high` reasoning
- `anthropic/claude-opus-4.7` with `high` reasoning

The historical panel is retained for reproducibility. Production pays the PDF
reading cost once per finalist, then reuses the cards for ranking and audit.
All three refreshed judges passed a live PDF/JSON smoke card with exact served
model IDs; the architecture, rather than the new panel's ranking, is what the
historical gold set validates.

### 5. Final Tournament Ranking

The final ranking uses frontier cards, not raw PDFs, with GPT-5.6 Sol xhigh as
the synthesis and pairwise adjudication model. Small finalist sets use
all-pairs comparison; larger pools use Swiss pairwise comparisons over a broad
pool, followed by dense all-pairs comparison within the provisional playoff
subset. Headline claims come from the dense playoff, while Swiss standings
provide broader finalist ordering. Regularized Bradley-Terry strengths adjust
the sparse Swiss standings for opponent difficulty; direct win rates remain the
transparent primary check for the balanced all-pairs playoff. Pair batching,
A/B presentation, and evidence-block order are deterministically randomized so
the synthesis seed cannot leak into the adjudicator through prompt position.

Recommended defaults: cheap shortlist top 20%, Terra/Sonnet semifinal over the
full shortlist in size-8 batches and two partitions, 250 PDF-aware finalists,
a recall-preserving union of the synthesis and card-ensemble top-150 sets
(172 papers in production), 10 Swiss rounds, and a 60-paper dense playoff.

### 6. Human Comparison

We compare the canonical full-coverage ensemble ranking with presentation tier,
published reviewer ratings, awards, and, where public, acceptance versus
rejection. Coverage must be explicit and complete; strong-stage comparisons
evaluate cheap and strong rankings on the same finalist subset and separately
report which human-honored papers survived selection. Divergence case studies
use fixed rank tails, and an independently coded, frozen taxonomy reports
recurring reasons with held-out summaries and inter-model agreement.

### Querying Ranked Papers

For a fresh shell or LLM thread, the shortest interface returns machine-readable
JSONL from the frozen 250-paper finalist ranking:

```bash
python3 -m icml_ai_ac.cli list-ranking-categories
python3 -m icml_ai_ac.cli top-ranked-papers \
  --category data-pretraining --top 20
```

`--category` accepts a preset (`data-pretraining`), contribution class
(`theory`, `benchmark_dataset`, and so on), official topic family
(`Deep Learning`), or explicit leaf topic prefixed with `topic:`. Use
`--scope tournament`, `--scope playoff`, or `--scope all` to change the default
finalist scope.

The same operation is available as a Python API with production paths already
configured:

```python
from icml_ai_ac.ranking_query import top_ranked_papers

papers = top_ranked_papers("data and pretraining", top_n=20)
```

For arbitrary combinations, `query-ranked-papers` joins the frozen ranking to
titles, abstracts, official ICML topics, and contribution classes without
rerunning any model:

```bash
python3 -m icml_ai_ac.cli query-ranked-papers \
  --preset data-pretraining \
  --out data/analysis/icml_2026_data_pretraining_ranked.csv
```

The transparent `data-pretraining` preset matches the official
`General Machine Learning->Data` topic, the `benchmark_dataset` contribution
class, specific data-practice language such as data curation or training data,
or explicit pretraining terms in the title or abstract. Every result includes
`match_reasons`, its full-corpus `cheap_rank`, and, when applicable, its frozen
`final_rank` and ranking stage. Papers outside the 250 frontier finalists have
no `final_rank`; this is distinct from receiving a low final rank.

Filters can also be composed directly:

```bash
# Only data/pretraining papers among the 250 frontier finalists.
python3 -m icml_ai_ac.cli query-ranked-papers \
  --preset data-pretraining --scope finalists

# Exact official-topic family, using a case-insensitive glob.
python3 -m icml_ai_ac.cli query-ranked-papers \
  --topic 'Deep Learning->*' --contribution-class benchmark_dataset

# Explicit pretraining papers that reached the 60-paper all-pairs playoff.
python3 -m icml_ai_ac.cli query-ranked-papers \
  --preset pretraining --scope playoff
```

Use `--query` for a case-insensitive title/abstract/topic search, `--scope` for
`all`, `finalists`, `tournament`, or `playoff`, and `--format` for CSV, JSONL,
or TSV output. The input paths are configurable for rebuilt or alternate runs.

## Gold-Set Validation

We validated the ranking design on a 50-paper ICML 2025 accepted-paper testbed.
The current reference is not a single text-only model ranking. It uses:

1. three independent PDF-aware frontier cards per paper;
2. GPT-5.5 synthesis over those cards;
3. GPT-5.5 xhigh all-pairs tournament over the synthesis top 30.

The corrected all-pairs tournament completed **435/435** comparisons in 18
resumable batches. It deterministically randomized pair batching, A/B
presentation, and evidence order, producing 221 A wins and 214 B wins. Valid
batches cost **$8.52** in reported OpenRouter usage and took 38 minutes of
provider time; one rejected malformed batch added about $0.41 and was retried
without repeating completed work.

Key lessons:

- the old text-only gold ranking had Spearman 0.290 against the tournament
  gold;
- the PDF card ensemble had Spearman 0.962, and the PDF card synthesis had
  Spearman 0.984 with 10/10 top-10 recall;
- the current four-model cheap ensemble captured 10/10 gold top-10 papers by
  shortlist rank 20 and 18/20 gold top-20 papers by rank 30;
- the old strong-model top-35 cut captured only 8/10 gold top-10 and 15/20 gold
  top-20 papers by rank 30;
- regularized Bradley-Terry and direct all-pairs ranks had Spearman 0.997,
  shared 9/10 top-10 papers, and differed by at most two rank positions;
- a simulated 10-round Swiss plus top-20 dense playoff used 255 rather than 435
  comparisons, retained 10/10 top-10 and 19/20 top-20 papers, and had Spearman
  0.996 against all-pairs.

The main empirical lesson is to preserve candidates generously before
PDF-aware carding and tournament adjudication.

The July 2026 Terra/Sonnet semifinal ensemble is an auditable diversity and
reliability improvement. Because it postdates the accepted-50 artifacts, its
incremental ranking benefit will be reported on the full production run rather
than inferred from a mismatched historical pass.

A July 25 end-to-end preflight ran 48 real ICML 2026 papers, including awards,
orals, spotlights, regular papers, and missing-score cases. All stages
completed. The cheap top-32 retained all nine award papers; the 12-paper
PDF-aware tournament completed 66/66 randomized comparisons with 32 A wins and
34 B wins. Against that tournament, card synthesis had Spearman 0.972 and
top-5/top-10 recall 1.0; semifinal and cheap rankings had Spearman 0.664 and
0.546. These are diagnostic rather than population estimates because the
sample was deliberately stratified by human outcomes.

A July 2026 refresh screened 14 current OpenRouter models on the same 50-paper
set and produced 11 usable rankings. All 330 four-model subsets were evaluated.
The initially selected Gemini 3.1 panel retained 10/10 gold top-10 papers by
shortlist rank 20, 18/20 gold top-20 papers by rank 30, and 24/30 gold top-30
papers. A subsequent matched Flash Lite comparison replaced
Gemini 3.1 with Gemini 3.5 for ranking: ensemble Spearman rose from 0.646 to
0.662, gold top-10 recall at rank 10 rose from 6/10 to 7/10, and gold top-20
recall at rank 20 rose from 15/20 to 16/20; both variants retained 10/10 by rank
20 and 18/20 by rank 30, while top-30 same-k recall moved from 23/30 to 22/30.
Gemini 3.1 remains the contribution classifier because it achieved 0.80
accuracy and 0.764 macro-F1 versus 0.72 and 0.602 for 3.5.

## Cost and Runtime

- **Metadata and ratings:** cheap and fast. Accepted OpenReview ratings were
  fetched in 7 batched requests in about 18 seconds.
- **Official PDF download:** complete for all 6,614 available OpenReview PDFs.
  The resumable authenticated crawl used a two-second delay and zero in-run
  retries; two isolated timeouts recovered on later bounded invocations.
- **PDF parsing:** local and fast. A 200-PDF parse took about 54 seconds;
  full-corpus parsing should be on the order of tens of minutes once PDFs are
  present.
- **Cheap first pass:** measured probes imply roughly **$100-$215** for 6,617
  papers. On the 48-paper preflight, the four streams cost $1.53 and 18.6
  summed model-minutes; four-stream concurrency projects to about 14-15 hours
  at the slowest observed rate.
- **Strong semifinal:** Terra and Sonnet run independently, so either result can
  be retried without repeating the other. At equal token usage, the two-judge
  panel is roughly 1.7-1.8 times the cost of Terra alone at current rates and
  remains substantially cheaper than PDF-aware frontier review. The 32-paper
  preflight took 6.3 minutes for Terra and 13.0 minutes for Sonnet; run them
  concurrently in production.
- **Frontier PDF cards:** the historical ICML 2025 gold-set cards cost about
  **$33.82** for 150 card rows. The refreshed panel cost about **$12.00** for
  36 successful cards in the 12-paper preflight, including one Opus fallback,
  or roughly **$250** for 250 finalists before paper-level variation.
- **Tournament:** the corrected 12-paper Sol tournament cost **$0.89** for 66
  comparisons. A 60-paper all-pairs playoff projects to roughly **$24** at the
  same prompt density; Swiss exploration adds cost linearly in scheduled pairs.

Working full-run estimate: **$500-$2,000**, depending mainly on context budgets,
reruns, final tournament size, and the public rejected-paper count. Runtime
should be planned as roughly **one to two days** after PDFs are downloaded,
with each stage resumable and provider throughput the main uncertainty.

## Limitations

- **Rejected-paper coverage is incomplete.** Only authors who opted in have
  public rejected papers, so the "best rejected papers" analysis applies to the
  public rejected subset, not all rejects.
- **Possible contamination.** Some papers may already be in model training data.
  This can be mitigated, but not eliminated, by disabling retrieval, using
  paper-grounded prompts, comparing model families, and testing pre-decision or
  open-weight snapshots when feasible.
- **Anonymization is not perfect blinding.** Direct author, affiliation, email,
  acknowledgement, and PDF-metadata cues are removed from validated derivatives,
  but titles, self-citations, project names, writing style, or model memory may
  still reveal a paper's identity.
- **AI is not reproducing the real review process.** This is intentional: the
  study measures a paper-only executive AI judgment, not an AI simulation of
  reviewer discussion.
- **Prompt dependence matters.** Rankings may shift depending on whether the
  rubric emphasizes technical correctness, likely citation impact, conceptual
  novelty, or broad scientific importance.
- **arXiv is incomplete.** It misses roughly 30% of the top-rated accepted
  papers in our pilot and may expose older or non-camera-ready versions.
- **PDF parsing is imperfect.** Text extraction can miss figure/table nuance.
  PDF-aware frontier cards partially address this, but still use constrained
  excerpts plus extracted text.
- **Pairwise tournaments are expensive.** They are more defensible than a single
  ranked list, but scale quadratically, so the production plan uses Swiss
  exploration plus a dense playoff.

## Why This Is Interesting

This project isolates a question that is already implicit in modern peer review:
if AI is increasingly mediating how science is read and judged, what science
would it elevate if given final executive authority?

The result would not replace peer review. It would provide a concrete,
measurable picture of what an AI-centered evaluation regime selects for, misses,
and potentially gets right.

## Internal Methodology Docs

- Static blog and mock awards preview: [`site/index.html`](site/index.html)
- One-command blog build, ZIP packaging, data rebuild, and preview:
  [`site/README.md`](site/README.md)
- Results bundles and one-command deterministic rebuild: [`docs/reproducibility.md`](docs/reproducibility.md)
- Detailed pipeline: [`docs/final_pipeline_methodology.md`](docs/final_pipeline_methodology.md)
- Final run plan: [`docs/final_run_plan.md`](docs/final_run_plan.md)
- Production preflight: [`docs/production_preflight_2026-07-25.md`](docs/production_preflight_2026-07-25.md)
- Scoring and model details: [`docs/scoring.md`](docs/scoring.md)
- Crawling and PDF acquisition: [`docs/crawling.md`](docs/crawling.md)
- Methodology decision log: [`docs/methodology_decisions.md`](docs/methodology_decisions.md)

## References

[1] Ziming Luo, Zonglin Yang, Zexin Xu, Wei Yang, and Xinya Du. 2025.
**LLM4SR: A Survey on Large Language Models for Scientific Research.**
Surveying LLM use across hypothesis discovery, experiment planning,
implementation, scientific writing, and peer reviewing.
https://arxiv.org/abs/2501.04306

[2] Nitya Thakkar, Mert Yuksekgonul, Jake Silberg, Animesh Garg, Nanyun Peng,
Fei Sha, Rose Yu, Carl Vondrick, and James Zou. 2025. **Can LLM feedback enhance
review quality? A randomized study of 20K reviews at ICLR 2025.** The study
deployed optional LLM feedback to more than 20,000 ICLR 2025 reviews; 27% of
reviewers who received feedback updated their reviews.
https://arxiv.org/abs/2504.09737

## Repository

Repository: [github.com/shayne-longpre/icml-ai-ac](https://github.com/shayne-longpre/icml-ai-ac)
