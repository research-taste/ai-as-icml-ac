# ICML 2026 Stage 3 Production Audit

Date: 2026-07-30

Stage 3 independently ranked the same 1,442-paper, position-robust shortlist
with GPT-5.6 Terra high and Claude Sonnet 5 high. Both judges used anonymized
main-paper text and the earlier AI ensemble summary. They did not receive human
reviews, reviewer scores, AC comments, decisions, presentation tiers, awards,
authors, affiliations, emails, or live paper URLs.

## Integrity Gate

- Both streams completed 362/362 batches and 2,884/2,884 paper judgments.
- Every paper has one valid judgment in each of two deterministic partitions.
- Requested and served model IDs match exactly for every batch.
- All prompts, raw responses, parsed responses, and batch states are present.
- Prompt fingerprints match; no saved prompt contains a live URL or a
  structured human-outcome field.
- The 1,442 model-facing manifest rows have empty author fields and no human
  outcome or URL values.
- Terra cost $49.7797 and Sonnet cost $92.2469 in successful tracked calls.

The gate passed with two documented warnings. One Sonnet batch required the
compact-text safety-filter recovery described in
`docs/methodology_decisions.md`. Terra returned `overall_priority_score` on a
0-100 scale in 33 batches and on a 0-10 scale in 329 batches.

## Local Score Repair

No model calls were repeated. The two global judge rankings were regenerated
from their preserved listwise judgments:

1. Mean normalized local rank across two partitions remains the primary
   signal.
2. Complete 0-100 batches are rescaled to 0-10.
3. Ties are resolved by mean calibrated strong-model priority, then
   within-batch normalized priority, cheap-stage rank, and paper ID.
4. Raw scores, detected scales, prompts, and responses remain unchanged and
   available for analysis.

This removes the scale advantage without allowing large winner ties to fall
back immediately to the shared cheap-stage ordering.

## Statistical Checks

- Terra/Sonnet rank Spearman: 0.6964.
- Same-budget overlap: 11/25, 28/50, 56/100, 156/250, and 352/500.
- Mean absolute judge-rank difference: 239.7 of 1,442.
- Contribution-class agreement: 74.0%.
- Partition-level local-priority correlation: 0.654 for Terra and 0.655 for
  Sonnet.

The disagreement is useful diversity, not missing coverage or provider
substitution. Manual inspection of deterministic top and disagreement samples
found evidence-grounded rationales. Large disagreements commonly reflect
soundness concerns versus upside, or broad-ML impact versus domain-specific
scientific value.

The ensemble top 250 contains papers from seven of the eight routing classes.
The small `application_method` class has no paper in the raw top 250, so the
next selector must retain its configured per-class leaders rather than treating
the ensemble top 250 as the finalist set.

## Position Sensitivity

Paired slot fixed-effect sensitivity produced raw/adjusted Spearman values of
0.9960 for Terra and 0.9955 for Sonnet. At the 250-paper handoff, the adjusted
rankings retain 234/250 and 232/250 papers, respectively. Top-50 membership is
less stable because listwise aggregation creates large near-tied bands.

Stage 3 is therefore treated as a conservative recall and reranking stage, not
as the final paper ranking. The finalist selector preserves consensus leaders,
cheap-stage leaders, contribution-class leaders, and high-disagreement papers
before PDF-aware review and tournament adjudication.

## Frozen Outputs

- Audit: `data/model_runs/icml_2026_full_launch/stage03_final_audit.json`
- Terra: `data/scores/icml_2026_pass2_semifinal_terra_openrouter.json`
- Sonnet: `data/scores/icml_2026_pass2_semifinal_sonnet5.json`
- Ensemble: `data/scores/icml_2026_pass2_semifinal_ensemble.json`
- Ensemble report:
  `data/scores/icml_2026_pass2_semifinal_ensemble.report.json`

The ensemble uses an equal-weight mean of the two normalized global ranks,
retains both complete source rows and judge disagreement, and records source
file checksums and exact model metadata.
