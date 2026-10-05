# ICML 2026 Final Run Plan

## Fixed Inputs and Blindness Boundary

The production population is the frozen 6,617-paper scoring manifest: 6,614
official OpenReview PDFs and three validated arXiv fallbacks. Eleven of 6,628
indexed papers lack a usable PDF and remain explicitly excluded.

Model-facing inputs contain only opaque paper identifiers, titles, identity-
redacted extracted text, identity-redacted PDF excerpts where configured, and
judgments from earlier AI stages.
They do not contain reviews, reviewer ratings, area-chair comments, decisions,
presentation tiers, or awards. Retrieval, web search, and external tools are
disabled. Derived artifacts remove direct byline, affiliation, email,
acknowledgement, and PDF-metadata cues and fail closed on validation; canonical
PDFs remain unchanged. This reduces direct prestige cues but cannot prevent
identification from titles, self-citations, project names, or model memory.
Human outcomes remain separate until the AI ranking is frozen.
The model-facing manifest itself also removes authors, canonical URLs, source
labels, and all nonessential metadata. Human outcomes remain in the separate
canonical manifest for post hoc joins.

## Stage 0 Gate

Identity redaction is complete under the validated v25 base plus selective
v26-v33 corrections.
The model-facing manifest contains 6,617/6,617 successful rows, 6,617 validated
PDF derivatives, and 19,851 validated text artifacts. The final manifest has
4,236 v25, 153 v26, 1,265 v27, 492 v28, 362 v29, 27 v30, 14 v31, 60 v32,
and eight v33 records. The selective corrections remove wrapped names,
contact lines, affiliation blocks displaced by two-column extraction, accented
or acronym-only institutions, and wrapped employment/contribution notes. Every
corrected row reuses the checksum-identical passed PDF; unaffected rows remain
unchanged. The final audit reports zero errors. All 6,049 metadata titles
present in the source title region were retained, while 568 metadata/PDF title
differences are recorded as provenance drift. Canonical PDFs are unchanged.
The 48 Stage 2 batches issued before the affiliation diagnostic cost $1.1134;
their prompts and responses are preserved in a clearly labeled pre-v27 archive
and excluded from every production aggregate.

## Stage 1 Gate

Contribution routing is complete for 6,617/6,617 papers using Gemini 3.1 Flash
Lite and the frozen hash-randomized manifest. All 414 batches have exact
prompt, raw/composite response, parsed, and state artifacts; requested and
served model IDs match, and the final blindness scan found no human-outcome
fields or live URLs. Corrected-pass cost was $2.7262 including archived failed
attempts. A controlled reversal probe found a small position effect, so these
classes are provisional batching labels. The cheap ranking ensemble stores a
separate multi-model category consensus for analysis.

## Production Stages

1. **Route contributions (complete: 6,617 papers).** Gemini 3.1 Flash Lite assigns
   provisional routing classes in fingerprinted 16-paper batches after a fixed
   hash randomization makes prompt slot independent of corpus order.
2. **Cheap recall ensemble (6,617 papers).** Nemotron 3 Ultra, Gemini 3.5 Flash
   Lite, GPT-5.6 Luna, and Grok 4.3 each rank two deterministic
   class-stratified partitions. Preserve the full aggregate signal and advance
   a position-robust union of the raw and within-paper position-adjusted top
   20%. The production union contains 1,442 papers. Store every model's
   contribution class vote and a resolved ensemble class separately from the
   routing label.
3. **Strong semifinal (complete: 1,442 papers).** GPT-5.6 Terra high and
   Claude Sonnet 5 high independently rank the same shortlist through
   OpenRouter in size-8, two-partition batches. Preserve both source rankings,
   normalized consensus, and disagreement.
4. **Select finalists (250 papers).** Take a deterministic union of consensus
   leaders, cheap-stage leaders, contribution-class leaders, and large
   cheap/strong or Terra/Sonnet disagreements.
5. **Create PDF-aware cards (complete: 750 judgments).** GPT-5.6 Sol xhigh,
   Claude Fable 5 high, and Gemini 3.1 Pro high each reviewed the first nine
   PDF pages plus the extracted main paper. Fable produced 221 cards directly;
   29 explicit failures used Opus 4.8 high. Four truncated Sol responses were
   repeated with the same model and prompt using a 12,000-token output ceiling.
   Lossless normalization corrected six title-format variants and moved two
   complete Gemini section groups to their required top-level keys. No score or
   substantive judgment was edited.
6. **Synthesize and adjudicate.** First combine the three card streams by
   equal-weight, tie-aware within-judge rank percentiles; raw numeric scores
   are retained for analysis but never averaged into the seed. Sol xhigh then
   synthesizes the 250 reusable card bundles. The production Swiss pool is the
   172-paper union of the synthesis top 150 and rank-calibrated ensemble top
   150. Run ten Swiss rounds over this recall-preserving union and a dense
   all-pairs playoff over the top 60. Report direct playoff ordering and
   regularized Bradley-Terry strengths with conditional standard errors.
7. **Compare with humans post hoc.** Only after freezing AI outputs, join
   awards, presentation tier, and published reviewer scores. Report agreement,
   advancement recall, divergence tails, contribution-category rankings, and
   qualitative divergence codes.

## Artifact and Resume Contract

Use a distinct, stable output and run directory for every stage and model.
Never use `--overwrite` when resuming an interrupted production run. Reinvoke
the exact same command after correcting the interruption.

Every multi-call stage stores:

- exact prompts and request fingerprints;
- raw provider responses and parsed structured judgments;
- validation failures and error records;
- requested and served model identifiers;
- token and cost usage when reported;
- per-batch or per-paper coverage and resume state; and
- archived failed attempts before any retry; and
- source judgments, normalized ranks, selector reasons, disagreements, pair
  schedules, A/B randomization, pair decisions, and aggregation outputs.

The Stage 5 completion gate and normalization provenance are recorded in
`data/scores/icml_2026_stage5_frontier_cards.audit.json` and
`data/scores/icml_2026_stage5_card_normalization.report.json`. The three final
card streams end in `_final.jsonl`. The rank-calibrated card ensemble and its
focused audit are
`data/scores/icml_2026_frontier_card_ensemble_rank_calibrated.json` and
`data/scores/icml_2026_frontier_card_ensemble_rank_calibrated.audit.json`.
The Sol synthesis, synthesis audit, and conservative tournament seed are
`data/scores/icml_2026_frontier_card_synthesis_sol56.json`,
`data/scores/icml_2026_frontier_card_synthesis_sol56.audit.json`, and
`data/scores/icml_2026_tournament_seed_union172.json`.

Matching valid units are reused; only failed or missing units are called again.
HTTP 400, 401, 402, 403, 404, 422, and exhausted 429 responses stop remaining
requests in that model stream. In particular, insufficient OpenRouter credits
produce a persisted partial run rather than consuming requests across the rest
of the stage. After credits are replenished, the same command resumes at the
first failed unit.

The expensive PDF cards are independent reusable artifacts. Synthesis,
tournament schedule, playoff size, direct aggregation, Bradley-Terry fitting,
and human-comparison analyses can therefore be replaced or rerun without
re-reading PDFs or repeating earlier model stages.
