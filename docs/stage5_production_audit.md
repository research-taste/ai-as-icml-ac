# Stage 5 Production Audit

Stage 5 produced one PDF-aware judgment from each of three frontier streams for
all 250 frozen finalists: 750/750 valid cards.

## Completion

| Stream | Valid | Served models | Latest successful-card cost snapshot |
|---|---:|---|---:|
| GPT-5.6 Sol xhigh | 250/250 | 250 Sol | $87.21 |
| Claude Fable 5 high | 250/250 | 221 Fable, 29 Opus 4.8 fallbacks | $151.34 |
| Gemini 3.1 Pro high | 250/250 | 250 Gemini 3.1 Pro | $16.44 |

Four Sol responses ended at the original 6,000-token limit. They were repeated
with the same model, reasoning effort, prompt, blinded text, and nine-page PDF
at a 12,000-token ceiling. All four completed for $1.69.

Six model-generated title strings differed from canonical titles only by
formatting or an added parenthetical. Two Gemini responses placed complete
`evidence_from_pdf`, `ranking_hooks`, and `one_sentence_summary` fields inside
`impact_assessment`. These eight rows were normalized losslessly. Original raw
and parsed responses remain preserved, and no score or substantive judgment
was edited.

## Integrity

- The three streams contain the same 250 unique paper IDs.
- All rows are schema-valid and point to anonymized first-nine-page PDFs and
  anonymized extracted text.
- Requested and served model identities match, except for the 29 explicit and
  recorded Fable-to-Opus fallbacks.
- The model-facing manifest contains no authors, reviews, ratings, decisions,
  presentation tiers, awards, or external-retrieval configuration.

## Calibration

The judges are not on a common numeric scale. Overall-priority means and
standard deviations were:

| Judge | Mean | SD | Range |
|---|---:|---:|---:|
| Sol | 6.76 | 1.23 | 2-9 |
| Fable panel | 6.62 | 1.10 | 3-9 |
| Gemini | 8.48 | 0.55 | 7-10 |

Tie-aware score correlations were 0.631 for Sol/Fable, 0.309 for Sol/Gemini,
and 0.288 for Fable/Gemini. Gemini was substantially more lenient in a
deterministic content sample. Downstream aggregation must therefore use
within-judge ranks or calibrated percentiles, not raw score means. Judge
disagreement should remain available to the tournament and robustness
analysis.

## Accounting Limitation

The cost table sums the latest successful cards and includes Fable fallback
attempt usage. Some superseded failed attempts from bounded Sol and Gemini
reruns were overwritten by the legacy retry runner, so exact cumulative Stage
5 billing must be taken from the provider account. This affects cost reporting,
not judgment coverage or ranking provenance.

The machine-readable completion report is
`data/scores/icml_2026_stage5_frontier_cards.audit.json`.

## Rank-Calibrated Ensemble Gate

The final streams were combined into
`data/scores/icml_2026_frontier_card_ensemble_rank_calibrated.json`. Each file
is one requested judge stream, so the 29 Opus fallbacks remain observations
from the Fable panel rather than becoming a fourth judge. The ensemble averages
tie-aware within-judge rank percentiles and does not average raw scores.

The focused audit passed with 250/250 papers and three judgments per paper:

- judge-to-ensemble Spearman: Sol 0.852, Fable panel 0.818, Gemini 0.718;
- leave-one-judge-out Spearman: 0.927-0.936;
- rank scores span 0.017-0.991 with 233 unique values;
- the fallback stratum has mean rank 136.9 versus 124.0 for nonfallbacks, with
  one paper in the top 10 and two in the top 50; and
- the calibrated and legacy raw-mean rankings have Spearman 0.919 but only
  4/10 top-10 overlap, confirming that calibration materially removes scale
  effects.

A deterministic qualitative sample covered the top five papers, major
prior-stage leaders, and the largest disagreements. Consensus leaders had
specific evidence and calibrated risks. The largest disagreements were
substantive, most visibly where Gemini accepted claims that Sol and Fable
challenged on theorem soundness, benchmark scope, or practical generality.
No malformed-card, fallback, or aggregation defect was found.

This passes the card-ensemble gate. It is suitable as a synthesis/tournament
seed and robustness result, not as the final headline ranking.
