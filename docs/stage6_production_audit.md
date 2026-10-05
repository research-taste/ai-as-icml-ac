# Stage 6 Production Audit

## Sol Synthesis Gate

GPT-5.6 Sol xhigh synthesized the three PDF-aware card streams into a complete
250-paper ranking on the first production request.

- requested and served model: `openai/gpt-5.6-sol`;
- coverage: 250/250 unique paper IDs with ranks 1-250;
- validation errors: zero;
- usage: 277,774 prompt tokens and 81,189 completion tokens, including 15,611
  reasoning tokens;
- completion headroom: 46,811 tokens under the 128,000-token ceiling;
- cost: $6.431245; and
- elapsed time: 1,401 seconds.

Scores were not collapsed: overall priority used 41 distinct values over
2.5-10.0. The top 10 covered algorithms, analysis, infrastructure, scientific
modeling, theory, and safety. A focused sample of consensus leaders and the
largest promotions and demotions found specific evidence, scope, and
soundness arguments rather than malformed or generic rationales.

The synthesis and rank-calibrated ensemble had Spearman 0.848. Their top-150
sets overlapped on 128 papers, with 22 unique to each. The production Swiss
pool therefore preserves their 172-paper union.

## Tournament Launch Gate

The initial production-scale dry run found that the greedy Swiss matcher
stranded two papers in rounds 8-10. The scheduler now computes a complete
deterministic matching over unplayed edges and fails closed if none exists.
The corrected dry run produced:

- 172 papers and 86 pairs in every round;
- 860 unique Swiss comparisons over 10 rounds;
- a 60-paper dense playoff; and
- 2,482 scheduled comparisons in the deterministic dry-run projection.

The complete repository suite passed 225 tests after the correction.

The supervised production worker is
`com.shayne.icml-ai-ac.stage06-tournament`. Its first batch completed 25/25
valid comparisons with exact Sol identity, no errors, 9,204 completion tokens
under a 32,000-token ceiling, and $0.5785725 cost. A/B outcomes were 14/11,
all reasons were nonempty, and confidence ranged from 0.55 to 0.79.

## Tournament Completion Gate

The production tournament completed successfully:

- 104/104 valid production batches and zero failed batches;
- 860/860 Swiss comparisons over 172 papers;
- all 1,770 unordered pairs among the 60 playoff papers, including 183
  comparisons already observed during Swiss;
- 2,447/2,447 unique comparisons overall;
- exact `openai/gpt-5.6-sol` identity in every batch;
- 3,843,870 prompt tokens and 991,322 completion tokens, including 704,671
  reasoning tokens;
- $53.5554 artifact-tracked cost; and
- empty stderr.

A/B outcomes were 1,184/1,263, all 2,447 rationales were nonempty, and mean
confidence was 0.660. The unweighted and confidence-weighted playoff
Bradley-Terry fits both converged on the complete connected graph. Their
Spearman correlations with the transparent direct-win ordering were 0.9983
and 0.9993, respectively, and both preserved the direct top 10 exactly.

The completed launchd service was removed after it repeatedly reinvoked the
fully cached command. Those post-completion invocations issued no paid calls
and did not change any batch judgment.
