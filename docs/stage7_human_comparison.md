# ICML 2026 Stage 7: Human-vs-AI Comparison

Status: deterministic analysis complete on 2026-07-30. No model calls were
made in this stage.

## Analysis boundary

The AI ranking was frozen before human outcomes were joined. Models received
only anonymized paper content and opaque paper IDs. Human presentation tiers,
review scores, decisions, awards, and author identities were read afterward
from `data/metadata/icml_2026_scoring_manifest.jsonl`.

The 6,617 scored papers contain three non-equivalent publication routes:

| Population | Papers | Oral | Spotlight | Poster | Comparable ICML review mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Main conference track | 6,341 | 159 | 377 | 5,805 | 6,341 |
| Position paper track | 213 | 9 | 29 | 175 | 0 |
| Journal presentations | 63 | 0 | 0 | 63 | 0 |

The main-track population is the primary quantitative comparison. Position
papers are analyzed separately. Journal presentations remain in the AI corpus
but cannot identify agreement with ICML review scores or presentation tiers.
Tracks are identified from the canonical source URL, not inferred from titles.

## Primary agreement results

On the 6,341 main-track papers, the full-coverage cheap ensemble had:

- ROC-AUC 0.6023 for oral/spotlight versus poster (95% bootstrap CI
  0.578-0.627);
- ROC-AUC 0.6223 for oral versus all other papers (95% CI 0.576-0.666);
- Kendall tau-b 0.0804 with presentation tier; and
- Kendall tau-b 0.0793 with published reviewer overall score.

This is modest but clear positive agreement. It should not be described as an
attempt to reproduce conference decisions: the AI rubric emphasizes broad
scientific and field impact, whereas presentation tier and reviewer score
reflect the conference process.

Position-track agreement was weaker: AUC 0.5382 for oral/spotlight versus
poster (95% CI 0.439-0.633) and tau-b 0.0367 with tier. Position papers have no
comparable OpenReview score in the canonical manifest.

## Enrichment through the pipeline

The main-track baseline is 8.45% oral/spotlight, with mean reviewer score
4.1468. The progressively stronger AI sets are enriched for human-honored
papers despite never receiving human outcomes:

| AI set | Papers | Main / position / journal | Oral or spotlight | Enrichment | Mean reviewer score | Award papers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Finalists | 250 | 248 / 2 / 0 | 49 (19.6%) | 2.32x | 4.331 | 2 |
| Swiss pool | 172 | 172 / 0 / 0 | 42 (24.4%) | 2.89x | 4.372 | 2 |
| All-pairs playoff | 60 | 60 / 0 / 0 | 13 (21.7%) | 2.56x | 4.447 | 1 |
| Final top 20 | 20 | 20 / 0 / 0 | 6 (30.0%) | 3.55x | 4.521 | 0 |
| Final top 10 | 10 | 10 / 0 / 0 | 4 (40.0%) | 4.73x | 4.583 | 0 |

The non-monotonic honored share at 172 versus 60 is not a contradiction. Swiss
selects a broad playoff set; all-pairs then optimizes the AI rubric within that
set. Human-tier replication was not a tournament objective.

## Final top 10

| Rank | Paper | Human tier | Reviewer mean |
| ---: | --- | --- | ---: |
| 1 | Spurious Rewards: Rethinking Training Signals in RLVR | Poster | 4.00 |
| 2 | Learning to Discover at Test Time | Spotlight | 5.00 |
| 3 | Revisiting Efficiency-Accuracy Scaling in Mixture-of-Experts Architectures | Poster | 3.75 |
| 4 | Reinforcement Learning via Self-Distillation | Poster | 3.75 |
| 5 | Maximum Likelihood Reinforcement Learning | Oral | 5.00 |
| 6 | AutoNumerics-Zero: Automated Discovery of State-of-the-Art Mathematical Functions | Poster | 4.50 |
| 7 | Reinforcement Learning with Evolving Rubrics for Deep Research | Oral | 5.33 |
| 8 | Rethinking the Trust Region in LLM Reinforcement Learning | Poster | 5.00 |
| 9 | Gromov-Wasserstein at Scale, Beyond Squared Norms | Poster | 4.75 |
| 10 | Flowers: A Warp Drive for Neural PDE Solvers | Spotlight | 4.75 |

No conference award paper appears in the top 10. Two of seven main-track award
papers reached the 248-paper main-track finalist subset; they finished at ranks
28 and 90. One reached the all-pairs top 60. This is low award recall, but still
well above random retention for a 3.91% finalist selection rate. The sample is
too small for a stable award-specific estimate.

## Tournament robustness

Within the 60-paper all-pairs playoff, direct win ordering had AUC 0.5957 for
oral/spotlight versus poster and tau-b 0.1249 with tier. Unweighted
Bradley-Terry aggregation had AUC 0.6023 and tau-b 0.1324. The two methods
preserved the same top 10 and had Spearman correlation 0.9983. Confidence-
weighted Bradley-Terry gave the same substantive conclusion.

Direct wins therefore remain the auditable headline result; Bradley-Terry
strength and uncertainty are sensitivity analyses.

## AI preference profile

The overall finalist set is not composition-matched to the corpus. Relative to
all 6,617 papers, the top 250:

- overrepresent core ML algorithms (35.2% versus 30.0%), theory (18.4% versus
  12.9%), safety/governance/evaluation (14.8% versus 8.4%), and scientific
  modeling tools (12.0% versus 8.3%);
- underrepresent application methods (1.6% versus 18.4%) and analysis/position
  papers (4.0% versus 8.2%).

The top 10 contain seven core ML algorithms, two scientific modeling tools,
and one analysis paper. This is a substantive output of the broad-impact
rubric, not an artifact-coverage failure. Category-specific rankings remain
necessary to show strong contributions that are disadvantaged in the single
overall ranking.

## Divergence cases

Among main-track papers, 531 posters fall in the AI top decile, while 35
oral/spotlight papers fall in the AI bottom decile. With at least three reviews,
47 papers are in the AI top decile and reviewer-score bottom decile, while 63
show the reverse.

Representative AI-over-human cases include:

- `65009`, *Spurious Correlation Learning in Preference Optimization*:
  cheap rank 4, poster, reviewer mean 4.75, final layered rank 199;
- `62675`, *An Exponential Separation Between Quantum and Quantum-Inspired
  Classical Algorithms for Linear Systems*: cheap rank 7, poster, reviewer
  mean 3.50, final rank 20; and
- `62669`, *SWE-ABS*: cheap rank 113, poster, reviewer mean 3.25, final rank 12.

Representative human-over-AI cases include several highly rated specialized
application papers. Focused artifact checks found complete prompts and eight
valid judgments per paper across four cheap models and two randomized
contexts. Their low ranks were consistent across models, so the pattern is a
model/rubric preference rather than missing input or a failed model stream.

## AI-axis comparison

On main-track papers, technical soundness aligned most strongly with reviewer
scores (tau-b 0.1254) and presentation honors (AUC 0.6109). Broader-science
impact, executive priority, and ML-field impact had honor AUCs of 0.6024,
0.6023, and 0.5945. The production aggregate does not contain a separate
`conventional_acceptance_strength` axis, so the preplanned
conventional-versus-impact delta is unavailable for this run.

## Interpretation

The strongest defensible conclusion is not that AI agrees or disagrees with
ICML in the aggregate. It is that the independent AI pipeline selects a set
that is clearly enriched for human-recognized papers while imposing a
different ordering and contribution-type preference. Agreement rises near the
very top, but many award papers and specialized high-review papers are absent.
That combination is the central empirical result to explain, not an error to
calibrate away.

## Artifacts

- Main agreement: `data/evals/icml_2026_human_agreement_main_track_finalists250.json`
- Main Stage 6 subsets: `data/evals/icml_2026_human_agreement_main_track_tournament172.json`,
  `data/evals/icml_2026_human_agreement_main_track_playoff60.json`
- Position agreement: `data/evals/icml_2026_human_agreement_position_track_finalists250.json`
- Reviewer divergence: `data/evals/icml_2026_human_divergence_reviewer.json`
- Main-tier divergence: `data/evals/icml_2026_human_divergence_main_track_tier.json`
- Position-tier divergence: `data/evals/icml_2026_human_divergence_position_track_tier.json`
- AI-axis analysis: `data/evals/icml_2026_ai_axis_decomposition_main_track.json`
- Playoff Bradley-Terry sensitivity:
  `data/evals/icml_2026_human_agreement_playoff60_bt.json`
