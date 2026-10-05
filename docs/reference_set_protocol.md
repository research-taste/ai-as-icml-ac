# Reference Ranking Protocol

This document describes how to create and use a reference ranking for evaluating
cheap-model-to-frontier ranking pipelines on the 50-paper ICML 2025 testbed.
The original text-only Qwen -> GPT protocol is retained as historical context;
the preferred reference is now PDF-aware and tournament-adjudicated.

## Purpose

The reference set is not "truth." It is a strong-judge reference ranking used to
measure whether cheaper pipeline variants preserve the papers that a stronger
review process would prioritize.

## Recommended Reference Process

1. Use the 50 ICML 2025 papers with `scoring_repr.txt` generated from the main
   paper pages, references removed, and appendix/supplement stripped when
   detected.
2. Generate independent PDF-aware frontier cards for every paper with a diverse
   set of strong judges. Each card should see the configured PDF excerpt plus
   extracted main-paper text, with retrieval off.
3. Synthesize the cards into a full 1..50 seed ranking.
4. Run pairwise tournament adjudication over the top seed subset with
   deterministic randomization of pair batches, A/B presentation, and evidence
   order. Do not expose seed ranks in the adjudication prompt.
5. Store the reference ranking as JSON with a `ranked_papers` array and evaluate
   candidate pipelines against it with `eval-ranking`.

## Reference Schema

```json
{
  "evaluation_mode": "reference_gold_ranking",
  "judge": "strong_model_or_human_panel_identifier",
  "paper_set": "icml_2025_accepted_50",
  "modality": "text_only_or_pdf_visual",
  "retrieval": "off",
  "ranked_papers": [
    {
      "rank": 1,
      "paper_id": "paper_id",
      "title": "title",
      "primary_contribution_class": "core_ml_algorithm",
      "secondary_contribution_classes": ["scientific_modeling_tool"],
      "overall_priority_score": 9,
      "broad_scientific_impact_score": 9,
      "ml_field_impact_score": 8,
      "technical_soundness_score": 8,
      "category_rank": 1,
      "advance_to_final_review": true,
      "why_ranked_here": "evidence-grounded comparative reason",
      "main_risk": "main uncertainty"
    }
  ],
  "category_rankings": {
    "core_ml_algorithm": ["paper_id"],
    "theory": ["paper_id"],
    "benchmark_dataset": ["paper_id"],
    "infrastructure_systems": ["paper_id"],
    "scientific_modeling_tool": ["paper_id"],
    "safety_governance_eval": ["paper_id"],
    "application_method": ["paper_id"],
    "analysis_position": ["paper_id"]
  }
}
```

## Evaluation Command

```bash
python3 -m icml_ai_ac.cli eval-ranking \
  --gold data/reference/icml_2025_accepted_50_frontier_pdf_tournament_gold_randomized_v3.json \
  --candidate data/scores/icml_2025_pass1_ensemble_v2_3models_shortlist45_classbalanced3.jsonl \
  --out data/evals/cheap_ensemble_vs_tournament_gold.json \
  --k 5 --k 10 --k 20
```

Metrics include:

- top-k recall and precision against the reference
- missed gold top-k papers
- rank correlation on shared papers
- category-specific top-k metrics

Rank correlation is computed over the relative order of papers shared by the
gold and candidate files. For subset candidate files, top-k recall is the
primary metric: did the cheaper stage preserve the papers the reference judge
ranked highly?

## Historical Text-Only Reference Run

This older GPT-5.4 text-only reference is useful for reproducibility, but it is
not the preferred gold ranking for current pipeline claims.

Dry-run the full reference prompt:

```bash
python3 -m icml_ai_ac.cli rank-reference \
  --manifest data/metadata/icml_2025_accepted_50_parsed.jsonl \
  --out data/reference/icml_2025_accepted_50_gold_dryrun.json \
  --run-dir data/model_runs/icml_2025_accepted_50_gold_dryrun_gpt54 \
  --provider openai \
  --model gpt-5.4 \
  --paper-set-name icml_2025_accepted_50 \
  --text-source scoring \
  --per-paper-char-budget 24000 \
  --max-output-tokens 20000 \
  --dry-run
```

Live run:

```bash
python3 -m icml_ai_ac.cli rank-reference \
  --manifest data/metadata/icml_2025_accepted_50_parsed.jsonl \
  --out data/reference/icml_2025_accepted_50_gold.json \
  --run-dir data/model_runs/icml_2025_accepted_50_gold_gpt54 \
  --provider openai \
  --model gpt-5.4 \
  --paper-set-name icml_2025_accepted_50 \
  --text-source scoring \
  --per-paper-char-budget 24000 \
  --temperature 0.1 \
  --max-output-tokens 20000 \
  --timeout 900 \
  --retries 2 \
  --backoff 10
```

If structural validation fails because a long JSON response omits or duplicates
paper ids, repair from the saved parsed response rather than manually editing:

```bash
python3 -m icml_ai_ac.cli repair-reference \
  --manifest data/metadata/icml_2025_accepted_50_parsed.jsonl \
  --reference data/model_runs/icml_2025_accepted_50_gold_gpt54/parsed.json \
  --out data/reference/icml_2025_accepted_50_gold.json \
  --run-dir data/model_runs/icml_2025_accepted_50_gold_repair_gpt54 \
  --provider openai \
  --model gpt-5.4 \
  --paper-set-name icml_2025_accepted_50 \
  --text-source scoring \
  --per-paper-char-budget 24000 \
  --temperature 0.1 \
  --max-output-tokens 20000 \
  --timeout 600 \
  --retries 2 \
  --backoff 10
```

The May 15, 2026 accepted-50 run used `reference_gold_rank_v1`, then a
`reference_gold_rank_repair_v1` coverage repair after the first response
duplicated `3go0lhfxd0` and omitted `zFR5aWGaUs`. The final artifact
`data/reference/icml_2025_accepted_50_gold.json` validates with 50 unique paper
ids and ranks 1-50.

## Batch-Ranking Concern

If all of the strongest papers fall into the same cheap-model batch, forced
local ranking can suppress some of them. The mitigation is not to advance only
the top 1 paper per batch. Instead:

- use small but not tiny batches, currently 6-10 papers
- advance at least the top quartile from each batch
- include multiple randomized batch partitions when budget allows
- stratify batches by contribution class after a cheap classification pass
- evaluate recall against the reference set before using the setup on ICML 2026

The key metric is reference top-k recall: does the cheap first-pass stage
preserve the papers the reference judge ranks highly?

## Contribution Classification Before Ranking

Before cheap batch ranking, run a cheap contribution classification pass on
`compact_repr.txt`:

```bash
python3 -m icml_ai_ac.cli classify-contributions \
  --manifest data/metadata/icml_2025_accepted_50_parsed.jsonl \
  --provider openrouter \
  --model qwen/qwen3.7-plus \
  --text-source compact \
  --out data/metadata/icml_2025_contribution_classes.jsonl \
  --run-dir data/model_runs/icml_2025_contribution_classes
```

This classification should be used for routing and stratification, not for
quality judgment. Its purpose is to avoid putting all papers into undifferentiated
batches where infrastructure, datasets, theory, safety/eval work, and algorithms
are compared without context.
