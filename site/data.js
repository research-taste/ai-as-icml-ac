window.ICML_AI_AC_DATA = {
  "generatedAt": "2026-07-30",
  "title": "AI as Area Chair: What Science Would AI Send to ICML?",
  "awardRules": {
    "outstandingRanks": [
      1,
      2
    ],
    "honorableMentionRanks": [
      3,
      4,
      5,
      6,
      7
    ],
    "oralRanks": [
      1,
      60
    ],
    "note": "The mock program names ranks 1-2 Outstanding Papers, ranks 3-7 Honorable Mentions, and the complete 60-paper all-pairs playoff as AI Orals."
  },
  "experiment": {
    "scoredPapers": 6617,
    "mainTrackPapers": 6341,
    "finalists": 250,
    "swissPool": 172,
    "playoff": 60,
    "pairwiseComparisons": 2447,
    "finalJudge": "openrouter:openai/gpt-5.6-sol",
    "humanOutcomesJoinedPostHoc": true,
    "arxivLinkedPapers": 59,
    "officialFallbackLinks": 1
  },
  "method": {
    "stages": [
      {
        "stage": "01",
        "count": 6617,
        "title": "Anonymize and label",
        "models": "Gemini 3.1 Flash Lite",
        "detail": "Author names removed; contribution type labeled",
        "tone": "source"
      },
      {
        "stage": "02",
        "count": 6617,
        "title": "First-pass ranking",
        "models": "Nemotron 3 Ultra · Gemini 3.5 Flash Lite · GPT-5.6 Luna · Grok 4.3",
        "detail": "Each paper ranked in 2 batches of 8, by each model",
        "tone": "cheap"
      },
      {
        "stage": "03",
        "count": 1442,
        "title": "Second-pass ranking",
        "models": "GPT-5.6 Terra + Claude Sonnet 5",
        "detail": "Same batch design, stronger models",
        "tone": "strong"
      },
      {
        "stage": "04",
        "count": 250,
        "title": "Full-paper judgments",
        "models": "GPT-5.6 Sol + Claude Fable 5 + Gemini 3.1 Pro",
        "detail": "750 written judgments, 3 per paper",
        "tone": "frontier"
      },
      {
        "stage": "05",
        "count": 172,
        "title": "Swiss-system rounds",
        "models": "GPT-5.6 Sol, pairwise",
        "detail": "10 rounds · 860 pairs",
        "tone": "swiss"
      },
      {
        "stage": "06",
        "count": 60,
        "title": "Round robin",
        "models": "GPT-5.6 Sol, pairwise",
        "detail": "Every pair compared · 1,770 pairs",
        "tone": "playoff"
      }
    ],
    "swissPairs": 860,
    "newPlayoffPairs": 1587,
    "completePlayoffPairs": 1770
  },
  "humanComparison": {
    "stages": [
      {
        "key": "main_track",
        "label": "Main track",
        "papers": 6341,
        "oral": 159,
        "spotlight": 377,
        "poster": 5805,
        "awards": 7,
        "honored": 536,
        "honoredRate": 0.08452925406087368
      },
      {
        "key": "finalists",
        "label": "AI finalists",
        "papers": 250,
        "oral": 17,
        "spotlight": 32,
        "poster": 201,
        "awards": 2,
        "honored": 49,
        "honoredRate": 0.196
      },
      {
        "key": "swiss_pool",
        "label": "Swiss pool",
        "papers": 172,
        "oral": 15,
        "spotlight": 27,
        "poster": 130,
        "awards": 2,
        "honored": 42,
        "honoredRate": 0.2441860465116279
      },
      {
        "key": "all_pairs_playoff",
        "label": "All-pairs playoff",
        "papers": 60,
        "oral": 3,
        "spotlight": 10,
        "poster": 47,
        "awards": 1,
        "honored": 13,
        "honoredRate": 0.21666666666666667
      },
      {
        "key": "final_top_20",
        "label": "AI top 20",
        "papers": 20,
        "oral": 2,
        "spotlight": 4,
        "poster": 14,
        "awards": 0,
        "honored": 6,
        "honoredRate": 0.3
      },
      {
        "key": "final_top_10",
        "label": "AI top 10",
        "papers": 10,
        "oral": 2,
        "spotlight": 2,
        "poster": 6,
        "awards": 0,
        "honored": 4,
        "honoredRate": 0.4
      }
    ],
    "playoffOutcomes": [
      {
        "rank": 1,
        "paperId": "61082",
        "title": "Spurious Rewards: Rethinking Training Signals in RLVR",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 2,
        "paperId": "65888",
        "title": "Learning to Discover at Test Time",
        "tier": "spotlight",
        "actualAward": false
      },
      {
        "rank": 3,
        "paperId": "62240",
        "title": "Revisiting Efficiency–Accuracy Scaling in Mixture-of-Experts Architectures",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 4,
        "paperId": "64121",
        "title": "Reinforcement Learning via Self-Distillation",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 5,
        "paperId": "65332",
        "title": "Maximum Likelihood Reinforcement Learning",
        "tier": "oral",
        "actualAward": false
      },
      {
        "rank": 6,
        "paperId": "61770",
        "title": "AutoNumerics-Zero: Automated Discovery of State-of-the-Art Mathematical Functions",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 7,
        "paperId": "65886",
        "title": "Reinforcement Learning with Evolving Rubrics for Deep Research",
        "tier": "oral",
        "actualAward": false
      },
      {
        "rank": 8,
        "paperId": "62611",
        "title": "Rethinking the Trust Region in LLM Reinforcement Learning",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 9,
        "paperId": "64283",
        "title": "Gromov-Wasserstein at Scale, Beyond Squared Norms",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 10,
        "paperId": "63822",
        "title": "Flowers: A Warp Drive for Neural PDE Solvers",
        "tier": "spotlight",
        "actualAward": false
      },
      {
        "rank": 11,
        "paperId": "65011",
        "title": "Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 12,
        "paperId": "62669",
        "title": "SWE-ABS: Adversarial Benchmark Strengthening Exposes Inflated Success Rates on Test-based Benchmark",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 13,
        "paperId": "60727",
        "title": "Towards Unified Multimodal Pretraining",
        "tier": "spotlight",
        "actualAward": false
      },
      {
        "rank": 14,
        "paperId": "63255",
        "title": "Probably Approximately Correct Labels",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 15,
        "paperId": "64605",
        "title": "Prompt Injection as Role Confusion",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 16,
        "paperId": "62018",
        "title": "Walrus: A Cross-domain Foundation Model for Continuum Dynamics",
        "tier": "spotlight",
        "actualAward": false
      },
      {
        "rank": 17,
        "paperId": "65408",
        "title": "Practical and Scalable Hamiltonian Monte Carlo Without the Metropolis Test",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 18,
        "paperId": "64195",
        "title": "SAM Audio: Segment Anything in Audio",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 19,
        "paperId": "62943",
        "title": "Evolution Strategies at the Hyperscale",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 20,
        "paperId": "62675",
        "title": "An Exponential Separation Between Quantum and Quantum-Inspired Classical Algorithms for Linear Systems",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 21,
        "paperId": "60977",
        "title": "Hallucination is a Consequence of Space-Optimality: A Rate-Distortion Theorem for Membership Testing",
        "tier": "spotlight",
        "actualAward": false
      },
      {
        "rank": 22,
        "paperId": "64990",
        "title": "Speculative Sampling For Faster Molecular Dynamics",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 23,
        "paperId": "65461",
        "title": "RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 24,
        "paperId": "64766",
        "title": "Meta Flow Maps enable scalable reward alignment",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 25,
        "paperId": "61920",
        "title": "WildCat: Near-Linear Attention in Theory and Practice",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 26,
        "paperId": "64269",
        "title": "Model-Preserving Adaptive Rounding",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 27,
        "paperId": "62479",
        "title": "How to guide your flow: Steering flow maps for rapid test-time alignment",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 28,
        "paperId": "62989",
        "title": "How much can language models memorize?",
        "tier": "oral",
        "actualAward": true
      },
      {
        "rank": 29,
        "paperId": "63503",
        "title": "DITRON: Distributed Multi-level Tiling Compiler for Parallel Tensor Programs",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 30,
        "paperId": "61737",
        "title": "Advantage Weighted Matching: Aligning RL with Pretraining in Diffusion Models",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 31,
        "paperId": "65782",
        "title": "RDT2: Exploring the Scaling Limit of UMI Data Towards Zero-Shot Cross-Embodiment Generalization",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 32,
        "paperId": "65517",
        "title": "Towards Solving the Gilbert-Pollak Conjecture via Large Language Models",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 33,
        "paperId": "66613",
        "title": "Why Are Linear RNNs More Parallelizable?",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 34,
        "paperId": "61823",
        "title": "Excited Pfaffians: Generalized Neural Wave Functions Across Structure and State",
        "tier": "spotlight",
        "actualAward": false
      },
      {
        "rank": 35,
        "paperId": "66382",
        "title": "Esoteric Language Models",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 36,
        "paperId": "65357",
        "title": "Global Plane Waves From Local Gaussians: Periodic Charge Densities in a Blink",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 37,
        "paperId": "61178",
        "title": "Fast KV Compaction via Attention Matching",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 38,
        "paperId": "64674",
        "title": "Exploration Hacking: LLMs Can Learn to Resist RL Training",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 39,
        "paperId": "66733",
        "title": "Pull Requests as a Training Signal for Repo-Level Code Editing",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 40,
        "paperId": "61895",
        "title": "RePro: Training Language Models to Faithfully Recycle the Web for Pretraining",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 41,
        "paperId": "63606",
        "title": "Deriving Neural Scaling Laws from the Statistics of Natural Language",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 42,
        "paperId": "61683",
        "title": "Post-Training with Policy Gradients: Optimality and the Base Model Barrier",
        "tier": "spotlight",
        "actualAward": false
      },
      {
        "rank": 43,
        "paperId": "65462",
        "title": "Proximal Decoding: Provably Reducing Copyright Risk for Any Language Model",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 44,
        "paperId": "65894",
        "title": "Alignment Pretraining: AI Discourse Causes Self-Fulfilling (Mis)alignment",
        "tier": "spotlight",
        "actualAward": false
      },
      {
        "rank": 45,
        "paperId": "66380",
        "title": "The Catastrophic Failure of the k-Means Algorithm in High Dimensions, and How Hartigan's Algorithm Avoids It",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 46,
        "paperId": "60987",
        "title": "Influence-Guided Symbolic Regression: Scientific Discovery via LLM-Driven Equation Search with Granular Feedback",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 47,
        "paperId": "65565",
        "title": "Sampling from Your Language Model One Byte at a Time",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 48,
        "paperId": "63428",
        "title": "ThetaEvolve: Test-time Learning on Open Problems",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 49,
        "paperId": "66697",
        "title": "INT vs. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 50,
        "paperId": "60565",
        "title": "Beyond Pixel Context Windows: Neural World Simulators with Persistent 3D State",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 51,
        "paperId": "62854",
        "title": "GeoPT: Scaling Physics Simulation via Lifted Geometric Pre-Training",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 52,
        "paperId": "63133",
        "title": "Efficient RL Training for LLMs with Experience Replay",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 53,
        "paperId": "64480",
        "title": "SRPO: Self-Reflective Policy Optimization for Long-Horizon Reasoning",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 54,
        "paperId": "66182",
        "title": "SciPredict: Can LLMs Predict the Outcomes of Scientific Experiments in Natural Sciences?",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 55,
        "paperId": "64188",
        "title": "You Need Better Attention Priors",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 56,
        "paperId": "62365",
        "title": "Scalable Simulation-Based Model Inference with Test-Time Complexity Control",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 57,
        "paperId": "65382",
        "title": "Learning Hamiltonian Flow Maps: Mean Flow Consistency for Large-Timestep Molecular Dynamics",
        "tier": "spotlight",
        "actualAward": false
      },
      {
        "rank": 58,
        "paperId": "65476",
        "title": "Least-Loaded Expert Parallelism: Load Balancing An Imbalanced Mixture-of-Experts",
        "tier": "poster",
        "actualAward": false
      },
      {
        "rank": 59,
        "paperId": "64785",
        "title": "Estimating Tail Risks in Language Model Output Distributions",
        "tier": "spotlight",
        "actualAward": false
      },
      {
        "rank": 60,
        "paperId": "63916",
        "title": "Recontextualization Mitigates Specification Gaming Without Modifying the Specification",
        "tier": "poster",
        "actualAward": false
      }
    ],
    "officialAwardPapers": [
      {
        "paperId": "60542",
        "title": "Motion Attribution for Video Generation",
        "highestStage": "main_track",
        "aiRank": null
      },
      {
        "paperId": "60766",
        "title": "The Obfuscation Atlas: Mapping Where Honesty Emerges in RLVR with Deception Probes",
        "highestStage": "main_track",
        "aiRank": null
      },
      {
        "paperId": "65132",
        "title": "High-accuracy sampling for diffusion models and log-concave distributions",
        "highestStage": "swiss_pool",
        "aiRank": 90
      },
      {
        "paperId": "61998",
        "title": "The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models",
        "highestStage": "main_track",
        "aiRank": null
      },
      {
        "paperId": "66206",
        "title": "To Grok Grokking: Provable Grokking in Ridge Regression",
        "highestStage": "main_track",
        "aiRank": null
      },
      {
        "paperId": "62989",
        "title": "How much can language models memorize?",
        "highestStage": "playoff",
        "aiRank": 28
      },
      {
        "paperId": "62241",
        "title": "A Random Matrix Perspective on the Consistency of Diffusion Models",
        "highestStage": "main_track",
        "aiRank": null
      }
    ]
  },
  "results": {
    "analysis": "icml_2026_stage7_human_comparison_summary",
    "status": "complete",
    "generated_at": "2026-07-30",
    "model_calls": 0,
    "inputs": {
      "canonical_human_manifest": "data/metadata/icml_2026_scoring_manifest.jsonl",
      "full_coverage_ai_ranking": "data/scores/icml_2026_pass1_cheap_ensemble_signal.jsonl",
      "final_tournament": "data/scores/icml_2026_frontier_tournament_sol56_union172_swiss10_playoff60.json",
      "playoff_bradley_terry": "data/scores/icml_2026_frontier_tournament_bt_playoff.json"
    },
    "populations": {
      "all_scored": 6617,
      "main_track": {
        "papers": 6341,
        "oral": 159,
        "spotlight": 377,
        "poster": 5805,
        "reviewer_score_coverage": 6341
      },
      "position_track": {
        "papers": 213,
        "oral": 9,
        "spotlight": 29,
        "poster": 175,
        "reviewer_score_coverage": 0
      },
      "journal_presentations": {
        "papers": 63,
        "oral": 0,
        "spotlight": 0,
        "poster": 63,
        "reviewer_score_coverage": 0
      }
    },
    "main_track_full_coverage": {
      "kendall_tau_b_vs_tier": 0.0804,
      "kendall_tau_b_vs_reviewer_overall": 0.0793,
      "roc_auc_honored_vs_poster": 0.6023,
      "roc_auc_honored_vs_poster_ci95": [
        0.578,
        0.627
      ],
      "roc_auc_oral_vs_rest": 0.6223,
      "roc_auc_oral_vs_rest_ci95": [
        0.576,
        0.666
      ],
      "honored_rate": 0.084529,
      "reviewer_overall_mean": 4.1468
    },
    "position_track_full_coverage": {
      "kendall_tau_b_vs_tier": 0.0367,
      "roc_auc_honored_vs_poster": 0.5382,
      "roc_auc_honored_vs_poster_ci95": [
        0.439,
        0.633
      ]
    },
    "pipeline_sets": [
      {
        "name": "finalists",
        "papers": 250,
        "main_track": 248,
        "position_track": 2,
        "journal_presentations": 0,
        "honored": 49,
        "honored_rate": 0.196,
        "main_track_honor_enrichment": 2.319,
        "reviewer_score_coverage": 248,
        "reviewer_overall_mean": 4.3312,
        "award_papers": 2
      },
      {
        "name": "swiss_pool",
        "papers": 172,
        "main_track": 172,
        "position_track": 0,
        "journal_presentations": 0,
        "honored": 42,
        "honored_rate": 0.2442,
        "main_track_honor_enrichment": 2.889,
        "reviewer_score_coverage": 172,
        "reviewer_overall_mean": 4.3716,
        "award_papers": 2
      },
      {
        "name": "all_pairs_playoff",
        "papers": 60,
        "main_track": 60,
        "position_track": 0,
        "journal_presentations": 0,
        "honored": 13,
        "honored_rate": 0.2167,
        "main_track_honor_enrichment": 2.563,
        "reviewer_score_coverage": 60,
        "reviewer_overall_mean": 4.4472,
        "award_papers": 1
      },
      {
        "name": "final_top_20",
        "papers": 20,
        "main_track": 20,
        "position_track": 0,
        "journal_presentations": 0,
        "honored": 6,
        "honored_rate": 0.3,
        "main_track_honor_enrichment": 3.549,
        "reviewer_score_coverage": 20,
        "reviewer_overall_mean": 4.5208,
        "award_papers": 0
      },
      {
        "name": "final_top_10",
        "papers": 10,
        "main_track": 10,
        "position_track": 0,
        "journal_presentations": 0,
        "honored": 4,
        "honored_rate": 0.4,
        "main_track_honor_enrichment": 4.732,
        "reviewer_score_coverage": 10,
        "reviewer_overall_mean": 4.5833,
        "award_papers": 0
      }
    ],
    "playoff_ordering": {
      "direct_wins": {
        "kendall_tau_b_vs_tier": 0.1249,
        "kendall_tau_b_vs_reviewer_overall": 0.108,
        "roc_auc_honored_vs_poster": 0.5957,
        "roc_auc_oral_vs_rest": 0.8012
      },
      "unweighted_bradley_terry": {
        "kendall_tau_b_vs_tier": 0.1324,
        "kendall_tau_b_vs_reviewer_overall": 0.1164,
        "roc_auc_honored_vs_poster": 0.6023,
        "roc_auc_oral_vs_rest": 0.7953,
        "spearman_vs_direct_wins": 0.9983,
        "top_10_overlap": "10/10"
      }
    },
    "divergence_counts": {
      "main_track_ai_top_decile_posters": 531,
      "main_track_ai_bottom_decile_oral_or_spotlight": 35,
      "ai_top_reviewer_bottom_decile": 47,
      "reviewer_top_ai_bottom_decile": 63
    },
    "primary_interpretation": "The independent AI pipeline is enriched for human-recognized papers but imposes a materially different ordering and contribution-type preference."
  },
  "papers": [
    {
      "rank": 1,
      "paperId": "61082",
      "title": "Spurious Rewards: Rethinking Training Signals in RLVR",
      "primaryClass": "analysis_position",
      "secondaryClasses": [
        "core_ml_algorithm",
        "theory"
      ],
      "whyRankedHere": "The strongest course correction in the set: multi-model interventions show random rewards can mimic RLVR progress through clipping-amplified priors.",
      "bestCase": "Random-reward controls and cross-family replication become mandatory for reasoning-RL claims.",
      "mainRisk": "The mechanism may be concentrated in GRPO, Qwen priors, and modest-scale mathematics training.",
      "scores": {
        "priority": 10.0,
        "technical": 9.0,
        "mlImpact": 10.0,
        "scienceImpact": 9.2,
        "evidence": 9.1
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.0,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2506.10947",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2506.10947",
        "icml": "https://icml.cc/virtual/2026/poster/61082",
        "openreview": "https://openreview.net/forum?id=tqTNOpkP5j"
      }
    },
    {
      "rank": 2,
      "paperId": "65888",
      "title": "Learning to Discover at Test Time",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "scientific_modeling_tool",
        "application_method"
      ],
      "whyRankedHere": "Test-time RL improves the solution distribution and yields verifiable artifacts across mathematics, kernels, algorithms, and biology.",
      "bestCase": "Per-problem learning becomes the default architecture for automated scientific and engineering discovery.",
      "mainRisk": "Best-of-many evidence, dense task-specific rewards, and concurrent methods leave attribution and generality uncertain.",
      "scores": {
        "priority": 9.6,
        "technical": 8.8,
        "mlImpact": 9.4,
        "scienceImpact": 9.3,
        "evidence": 8.6
      },
      "human": {
        "tier": "spotlight",
        "reviewerMean": 5.0,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2601.16175",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2601.16175",
        "icml": "https://icml.cc/virtual/2026/poster/65888",
        "openreview": "https://openreview.net/forum?id=96zNuQrH9Y"
      }
    },
    {
      "rank": 3,
      "paperId": "62240",
      "title": "Revisiting Efficiency–Accuracy Scaling in Mixture-of-Experts Architectures",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "infrastructure_systems",
        "analysis_position"
      ],
      "whyRankedHere": "LatentMoE shows persistent large-scale gains across transformer and Mamba hybrids with measured serving plausibility up to 95B parameters.",
      "bestCase": "Latent-space expert computation becomes a default frontier-MoE design.",
      "mainRisk": "The largest claimed serving gains rely on simulator and effective-parameter extrapolations.",
      "scores": {
        "priority": 9.4,
        "technical": 8.9,
        "mlImpact": 9.5,
        "scienceImpact": 8.7,
        "evidence": 8.7
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 3.75,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2601.18089",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2601.18089",
        "icml": "https://icml.cc/virtual/2026/poster/62240",
        "openreview": "https://openreview.net/forum?id=iQRy0io4sb"
      }
    },
    {
      "rank": 4,
      "paperId": "64121",
      "title": "Reinforcement Learning via Self-Distillation",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "analysis_position",
        "application_method"
      ],
      "whyRankedHere": "SDPO converts environment feedback into dense self-teacher logits and consistently improves credit assignment, concision, and sample efficiency.",
      "bestCase": "Models become their own process-reward models for tool and reasoning feedback.",
      "mainRisk": "Results stop at 8B and may fail when feedback is ambiguous or the base model cannot retrospect accurately.",
      "scores": {
        "priority": 9.4,
        "technical": 8.8,
        "mlImpact": 9.5,
        "scienceImpact": 8.5,
        "evidence": 8.5
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 3.75,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2601.20802",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2601.20802",
        "icml": "https://icml.cc/virtual/2026/poster/64121",
        "openreview": "https://openreview.net/forum?id=QkfkxyRizZ"
      }
    },
    {
      "rank": 5,
      "paperId": "65332",
      "title": "Maximum Likelihood Reinforcement Learning",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "theory",
        "analysis_position"
      ],
      "whyRankedHere": "MAXRL offers an exact compute-indexed objective, a one-line GRPO-compatible change, and consistent coverage and efficiency gains.",
      "bestCase": "It replaces group-normalized expected-reward objectives as the default for binary-verifier RL.",
      "mainRisk": "Evidence remains math-centric, below frontier scale, and assumes reliable terminal binary rewards.",
      "scores": {
        "priority": 9.9,
        "technical": 9.3,
        "mlImpact": 9.9,
        "scienceImpact": 8.8,
        "evidence": 8.8
      },
      "human": {
        "tier": "oral",
        "reviewerMean": 5.0,
        "reviewCount": 3,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2602.02710",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2602.02710",
        "icml": "https://icml.cc/virtual/2026/poster/65332",
        "openreview": "https://openreview.net/forum?id=EeuLO2BjFN"
      }
    },
    {
      "rank": 6,
      "paperId": "61770",
      "title": "AutoNumerics-Zero: Automated Discovery of State-of-the-Art Mathematical Functions",
      "primaryClass": "infrastructure_systems",
      "secondaryClasses": [
        "core_ml_algorithm",
        "scientific_modeling_tool"
      ],
      "whyRankedHere": "It produces formally bounded numerical artifacts with real cross-hardware speedups, giving unusually durable value beyond a benchmark score.",
      "bestCase": "Automated, certified superoptimization becomes routine for hardware-specific mathematical libraries.",
      "mainRisk": "Production libm semantics, range reduction, compiler portability, and search cost may narrow deployment gains.",
      "scores": {
        "priority": 9.8,
        "technical": 9.5,
        "mlImpact": 8.8,
        "scienceImpact": 9.6,
        "evidence": 9.1
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.5,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2312.08472",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2312.08472",
        "icml": "https://icml.cc/virtual/2026/poster/61770",
        "openreview": "https://openreview.net/forum?id=n7F2nwPcYB"
      }
    },
    {
      "rank": 7,
      "paperId": "65886",
      "title": "Reinforcement Learning with Evolving Rubrics for Deep Research",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "infrastructure_systems",
        "benchmark_dataset"
      ],
      "whyRankedHere": "The method, open stack, and capable 8B research agent jointly offer immediate infrastructure and algorithmic value.",
      "bestCase": "The released stack becomes the open standard for long-form research-agent training.",
      "mainRisk": "Most gains come from teacher SFT and generic rubric RL; evolving rubrics add only a modest isolated increment.",
      "scores": {
        "priority": 9.2,
        "technical": 8.6,
        "mlImpact": 9.2,
        "scienceImpact": 9.0,
        "evidence": 8.5
      },
      "human": {
        "tier": "oral",
        "reviewerMean": 5.3333,
        "reviewCount": 3,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2511.19399",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2511.19399",
        "icml": "https://icml.cc/virtual/2026/poster/65886",
        "openreview": "https://openreview.net/forum?id=97NEP1pyS3"
      }
    },
    {
      "rank": 8,
      "paperId": "62611",
      "title": "Rethinking the Trust Region in LLM Reinforcement Learning",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "theory",
        "analysis_position"
      ],
      "whyRankedHere": "DPPO diagnoses a structural defect in token-ratio clipping and supplies a cheap divergence-aware replacement validated at meaningful scale.",
      "bestCase": "DPPO becomes the standard proximal update in major LLM post-training stacks.",
      "mainRisk": "Top-K divergence masks remain approximations, and the large-scale curves lack strong multi-seed validation.",
      "scores": {
        "priority": 9.8,
        "technical": 9.0,
        "mlImpact": 9.9,
        "scienceImpact": 8.7,
        "evidence": 8.7
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 5.0,
        "reviewCount": 3,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2602.04879",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2602.04879",
        "icml": "https://icml.cc/virtual/2026/poster/62611",
        "openreview": "https://openreview.net/forum?id=f4g8zblbwp"
      }
    },
    {
      "rank": 9,
      "paperId": "64283",
      "title": "Gromov-Wasserstein at Scale, Beyond Squared Norms",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "theory",
        "scientific_modeling_tool"
      ],
      "whyRankedHere": "An elegant structural reduction yields differentiable, linear-memory GW computation on over 100,000 points with one-to-two-order speedups.",
      "bestCase": "It becomes the Sinkhorn-like default primitive for large intrinsic-geometry matching.",
      "mainRisk": "The solver remains local and application evidence is concentrated in shape matching rather than broader ML.",
      "scores": {
        "priority": 9.7,
        "technical": 9.2,
        "mlImpact": 9.3,
        "scienceImpact": 8.8,
        "evidence": 8.8
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.75,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2602.06658",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2602.06658",
        "icml": "https://icml.cc/virtual/2026/poster/64283",
        "openreview": "https://openreview.net/forum?id=P9LofJB8fs"
      }
    },
    {
      "rank": 10,
      "paperId": "63822",
      "title": "Flowers: A Warp Drive for Neural PDE Solvers",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "scientific_modeling_tool"
      ],
      "whyRankedHere": "A genuinely new transport-aware primitive dominates broad PDE benchmarks and scales favorably to 3D with interpretable learned warps.",
      "bestCase": "FLOWERS becomes a default backbone for large scientific simulation models.",
      "mainRisk": "Theory is heuristic and comparisons do not fully test diffusion-heavy, irregular, or industrial geometries.",
      "scores": {
        "priority": 9.1,
        "technical": 8.7,
        "mlImpact": 8.7,
        "scienceImpact": 9.0,
        "evidence": 8.6
      },
      "human": {
        "tier": "spotlight",
        "reviewerMean": 4.75,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2603.04430",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2603.04430",
        "icml": "https://icml.cc/virtual/2026/poster/63822",
        "openreview": "https://openreview.net/forum?id=TPpUKH8fuQ"
      }
    },
    {
      "rank": 11,
      "paperId": "65011",
      "title": "Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "application_method",
        "analysis_position"
      ],
      "whyRankedHere": "Self-Flow replaces external alignment with a coherent self-supervised objective and shows consistent image, video, audio, robotics, and scaling gains.",
      "bestCase": "It becomes the default training objective for multimodal flow foundation models.",
      "mainRisk": "Many absolute gains are modest, evaluations are partly private, and the EMA teacher adds training cost.",
      "scores": {
        "priority": 9.3,
        "technical": 8.8,
        "mlImpact": 9.3,
        "scienceImpact": 8.7,
        "evidence": 8.8
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.25,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2603.06507",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2603.06507",
        "icml": "https://icml.cc/virtual/2026/poster/65011",
        "openreview": "https://openreview.net/forum?id=HoThWhfxiK"
      }
    },
    {
      "rank": 12,
      "paperId": "62669",
      "title": "SWE-ABS: Adversarial Benchmark Strengthening Exposes Inflated Success Rates on Test-based Benchmark",
      "primaryClass": "benchmark_dataset",
      "secondaryClasses": [
        "safety_governance_eval",
        "infrastructure_systems"
      ],
      "whyRankedHere": "It rejects about one-fifth of previously passing coding-agent patches and supplies an immediately reusable strengthened evaluation artifact.",
      "bestCase": "SWE-ABS becomes the default oracle layer for repository-level coding agents.",
      "mainRisk": "Generated-test validity still depends on gold patches, LLM judgments, and manual correction.",
      "scores": {
        "priority": 9.0,
        "technical": 8.3,
        "mlImpact": 9.3,
        "scienceImpact": 8.8,
        "evidence": 8.4
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 3.25,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2603.00520",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2603.00520",
        "icml": "https://icml.cc/virtual/2026/poster/62669",
        "openreview": "https://openreview.net/forum?id=eWzojJ9SGF"
      }
    },
    {
      "rank": 13,
      "paperId": "60727",
      "title": "Towards Unified Multimodal Pretraining",
      "primaryClass": "analysis_position",
      "secondaryClasses": [
        "core_ml_algorithm",
        "infrastructure_systems"
      ],
      "whyRankedHere": "A broad controlled campaign provides actionable representation, routing, mixture, transfer, and IsoFLOP guidance for native multimodal models.",
      "bestCase": "It becomes the Chinchilla-like blueprint for unified multimodal pretraining.",
      "mainRisk": "Conclusions may be Transfusion-specific, and scaling extrapolations extend well beyond the observed compute regime.",
      "scores": {
        "priority": 9.3,
        "technical": 8.7,
        "mlImpact": 9.4,
        "scienceImpact": 8.8,
        "evidence": 8.5
      },
      "human": {
        "tier": "spotlight",
        "reviewerMean": 5.3333,
        "reviewCount": 3,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2603.03276",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2603.03276",
        "icml": "https://icml.cc/virtual/2026/poster/60727",
        "openreview": "https://openreview.net/forum?id=xAKrRYm6pc"
      }
    },
    {
      "rank": 14,
      "paperId": "63255",
      "title": "Probably Approximately Correct Labels",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "theory",
        "application_method"
      ],
      "whyRankedHere": "A simple distribution-free wrapper certifies hybrid AI-expert datasets across text, vision, and protein labels with meaningful savings.",
      "bestCase": "Certified surrogate labeling becomes standard data-creation infrastructure.",
      "mainRisk": "Guarantees cover aggregate label loss, not subgroup quality, expert correctness, or downstream scientific validity.",
      "scores": {
        "priority": 8.5,
        "technical": 8.8,
        "mlImpact": 8.4,
        "scienceImpact": 8.8,
        "evidence": 8.2
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 5.5,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2506.10908",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2506.10908",
        "icml": "https://icml.cc/virtual/2026/poster/63255",
        "openreview": "https://openreview.net/forum?id=Yhv4d8wfbi"
      }
    },
    {
      "rank": 15,
      "paperId": "64605",
      "title": "Prompt Injection as Role Confusion",
      "primaryClass": "safety_governance_eval",
      "secondaryClasses": [
        "analysis_position",
        "core_ml_algorithm"
      ],
      "whyRankedHere": "Attack, probe, controlled tags, and dose-response evidence jointly support a memorable latent role-confusion account of prompt injection.",
      "bestCase": "Role probes become standard diagnostics and inspire architectures that preserve trusted source metadata.",
      "mainRisk": "The probes remain correlational and may capture stylistic cues rather than the unique causal mechanism.",
      "scores": {
        "priority": 9.4,
        "technical": 8.6,
        "mlImpact": 9.4,
        "scienceImpact": 9.0,
        "evidence": 8.7
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.25,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2603.12277",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2603.12277",
        "icml": "https://icml.cc/virtual/2026/poster/64605",
        "openreview": "https://openreview.net/forum?id=Lm3DQDyIsI"
      }
    },
    {
      "rank": 16,
      "paperId": "62018",
      "title": "Walrus: A Cross-domain Foundation Model for Continuum Dynamics",
      "primaryClass": "scientific_modeling_tool",
      "secondaryClasses": [
        "core_ml_algorithm",
        "infrastructure_systems"
      ],
      "whyRankedHere": "A released 1.3B cross-domain physics model shows broad 2D/3D transfer, stabilization, and heterogeneous-training gains across 26 tasks.",
      "bestCase": "Walrus becomes the standard initialization and benchmark checkpoint for continuum emulation.",
      "mainRisk": "Scientific validity, conservation, solver-relative efficiency, and difficult long-horizon behavior remain under-validated.",
      "scores": {
        "priority": 9.6,
        "technical": 8.9,
        "mlImpact": 8.7,
        "scienceImpact": 9.4,
        "evidence": 8.8
      },
      "human": {
        "tier": "spotlight",
        "reviewerMean": 4.5,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2511.15684",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2511.15684",
        "icml": "https://icml.cc/virtual/2026/poster/62018",
        "openreview": "https://openreview.net/forum?id=kcyOjXoUZu"
      }
    },
    {
      "rank": 17,
      "paperId": "65408",
      "title": "Practical and Scalable Hamiltonian Monte Carlo Without the Metropolis Test",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "scientific_modeling_tool",
        "theory"
      ],
      "whyRankedHere": "A practical tuning criterion unlocks flat dimensional scaling and spectacular gradient-call savings on million-dimensional scientific targets.",
      "bestCase": "Unadjusted Hamiltonian samplers become black-box defaults for very high-dimensional Bayesian inference.",
      "mainRisk": "The rigorous bias relation is Gaussian-only and already underestimates error on one non-Gaussian benchmark.",
      "scores": {
        "priority": 8.9,
        "technical": 8.8,
        "mlImpact": 8.5,
        "scienceImpact": 9.0,
        "evidence": 8.7
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 5.0,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2412.08876",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2412.08876",
        "icml": "https://icml.cc/virtual/2026/poster/65408",
        "openreview": "https://openreview.net/forum?id=DxvZnTVNfP"
      }
    },
    {
      "rank": 18,
      "paperId": "64195",
      "title": "SAM Audio: Segment Anything in Audio",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "benchmark_dataset",
        "infrastructure_systems",
        "application_method"
      ],
      "whyRankedHere": "A unified multimodal audio-separation model, benchmark, and perceptual judge create a plausible SAM-like ecosystem contribution.",
      "bestCase": "It becomes the common front end for audio isolation, editing, and multimodal perception.",
      "mainRisk": "The million-hour training stack is hard to reproduce and evaluation is partly self-referential.",
      "scores": {
        "priority": 8.9,
        "technical": 8.4,
        "mlImpact": 8.8,
        "scienceImpact": 9.0,
        "evidence": 8.3
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.5,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2512.18099",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2512.18099",
        "icml": "https://icml.cc/virtual/2026/poster/64195",
        "openreview": "https://openreview.net/forum?id=Q4Cca8N7na"
      }
    },
    {
      "rank": 19,
      "paperId": "62943",
      "title": "Evolution Strategies at the Hyperscale",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "infrastructure_systems",
        "theory"
      ],
      "whyRankedHere": "Rank-one perturbations reopen billion-parameter evolution strategies by reaching near-inference throughput across unusual non-backpropagation workloads.",
      "bestCase": "Gradient-free optimization becomes viable for quantized, nondifferentiable, and memory-constrained foundation-model training.",
      "mainRisk": "Total compute remains far worse than backpropagation, and the strongest results favor recurrent architectures.",
      "scores": {
        "priority": 9.0,
        "technical": 8.5,
        "mlImpact": 9.2,
        "scienceImpact": 8.7,
        "evidence": 8.4
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.5,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2511.16652",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2511.16652",
        "icml": "https://icml.cc/virtual/2026/poster/62943",
        "openreview": "https://openreview.net/forum?id=bfVJ4GsHrO"
      }
    },
    {
      "rank": 20,
      "paperId": "62675",
      "title": "An Exponential Separation Between Quantum and Quantum-Inspired Classical Algorithms for Linear Systems",
      "primaryClass": "theory",
      "secondaryClasses": [
        "core_ml_algorithm"
      ],
      "whyRankedHere": "A claimed first unconditional exponential separation resolves a central question left by quantum-ML dequantization work.",
      "bestCase": "It becomes the definitive theorem establishing surviving exponential quantum advantage for linear systems.",
      "mainRisk": "The construction is worst-case, oracle-specific, practically remote, and depends on appendix proofs not visible in the cards.",
      "scores": {
        "priority": 8.9,
        "technical": 9.2,
        "mlImpact": 8.1,
        "scienceImpact": 8.5,
        "evidence": 8.2
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 3.5,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2411.02087",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2411.02087",
        "icml": "https://icml.cc/virtual/2026/poster/62675",
        "openreview": "https://openreview.net/forum?id=eTUljZ6e8c"
      }
    },
    {
      "rank": 21,
      "paperId": "60977",
      "title": "Hallucination is a Consequence of Space-Optimality: A Rate-Distortion Theorem for Membership Testing",
      "primaryClass": "theory",
      "secondaryClasses": [
        "analysis_position",
        "safety_governance_eval"
      ],
      "whyRankedHere": "An exact rate-distortion frontier predicts a high-confidence hallucination channel and is supported by controlled model experiments.",
      "bestCase": "It becomes the canonical information-theoretic account of lossy factual memory.",
      "mainRisk": "Random closed-world membership is far removed from structured, open-world generative hallucination.",
      "scores": {
        "priority": 8.8,
        "technical": 9.0,
        "mlImpact": 8.9,
        "scienceImpact": 8.8,
        "evidence": 8.4
      },
      "human": {
        "tier": "spotlight",
        "reviewerMean": 4.75,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2602.00906",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2602.00906",
        "icml": "https://icml.cc/virtual/2026/poster/60977",
        "openreview": "https://openreview.net/forum?id=uuD1rE5KU5"
      }
    },
    {
      "rank": 22,
      "paperId": "64990",
      "title": "Speculative Sampling For Faster Molecular Dynamics",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "scientific_modeling_tool",
        "theory"
      ],
      "whyRankedHere": "Langevin Speculative Dynamics gives mathematically exact, measured 3–9× acceleration of a central serial scientific workload.",
      "bestCase": "Speculative coupling becomes a reusable acceleration pattern for stochastic simulation.",
      "mainRisk": "Acceptance degrades with system size and the method trades 8–64 GPUs for single-trajectory latency.",
      "scores": {
        "priority": 9.2,
        "technical": 9.1,
        "mlImpact": 8.4,
        "scienceImpact": 9.1,
        "evidence": 8.9
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.5,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2606.02455",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2606.02455",
        "icml": "https://icml.cc/virtual/2026/poster/64990",
        "openreview": "https://openreview.net/forum?id=I3SK0WFmD0"
      }
    },
    {
      "rank": 23,
      "paperId": "65461",
      "title": "RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "infrastructure_systems",
        "benchmark_dataset"
      ],
      "whyRankedHere": "Four hundred adaptive procedural environments form a durable resource and convincingly revive saturated reasoning-RL checkpoints.",
      "bestCase": "RLVE-GYM becomes the OpenAI Gym for language-model reasoning.",
      "mainRisk": "Headline evidence is concentrated at 1.5B scale and may not transfer beyond synthetic verifiable tasks.",
      "scores": {
        "priority": 8.7,
        "technical": 8.7,
        "mlImpact": 9.1,
        "scienceImpact": 8.6,
        "evidence": 8.5
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 3.75,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2511.07317",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2511.07317",
        "icml": "https://icml.cc/virtual/2026/poster/65461",
        "openreview": "https://openreview.net/forum?id=DRqjMnDQGq"
      }
    },
    {
      "rank": 24,
      "paperId": "64766",
      "title": "Meta Flow Maps enable scalable reward alignment",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "theory",
        "application_method"
      ],
      "whyRankedHere": "Amortized posterior sampling addresses a shared bottleneck in generative steering and off-policy fine-tuning with over 100× NFE gains.",
      "bestCase": "Meta flow maps become a standard control interface for continuous generators.",
      "mainRisk": "Every base generator needs a new distillation stage, and large-scale evidence is ImageNet-only with proxy rewards.",
      "scores": {
        "priority": 9.0,
        "technical": 8.9,
        "mlImpact": 9.2,
        "scienceImpact": 8.6,
        "evidence": 8.5
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.5,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2601.14430",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2601.14430",
        "icml": "https://icml.cc/virtual/2026/poster/64766",
        "openreview": "https://openreview.net/forum?id=K5gV8Yptne"
      }
    },
    {
      "rank": 25,
      "paperId": "61920",
      "title": "WildCat: Near-Linear Attention in Theory and Practice",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "theory",
        "infrastructure_systems"
      ],
      "whyRankedHere": "WildCat unusually combines strong approximation guarantees with measured speed-quality wins across attention and cache-compression settings.",
      "bestCase": "It becomes a standard long-context attention and KV-compression backend.",
      "mainRisk": "Causal streaming support is missing, and the strongest LLM evidence stops at one 7B model.",
      "scores": {
        "priority": 9.7,
        "technical": 9.0,
        "mlImpact": 9.6,
        "scienceImpact": 8.7,
        "evidence": 8.7
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.5,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2602.10056",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2602.10056",
        "icml": "https://icml.cc/virtual/2026/poster/61920",
        "openreview": "https://openreview.net/forum?id=lfqyLp4hZm"
      }
    },
    {
      "rank": 26,
      "paperId": "64269",
      "title": "Model-Preserving Adaptive Rounding",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "theory",
        "infrastructure_systems"
      ],
      "whyRankedHere": "YAQA provides a quantizer-agnostic successor to GPTQ rounding with systematic KL improvements and no inference overhead.",
      "bestCase": "It becomes the default rounding backend for low-bit model deployment.",
      "mainRisk": "Calibration gradients and Hessian sketches are costly, while practical 4-bit downstream gains are often small.",
      "scores": {
        "priority": 8.6,
        "technical": 8.8,
        "mlImpact": 9.0,
        "scienceImpact": 8.1,
        "evidence": 8.6
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.5,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2505.22988",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2505.22988",
        "icml": "https://icml.cc/virtual/2026/poster/64269",
        "openreview": "https://openreview.net/forum?id=PKFilPWjMI"
      }
    },
    {
      "rank": 27,
      "paperId": "62479",
      "title": "How to guide your flow: Steering flow maps for rapid test-time alignment",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "theory",
        "application_method"
      ],
      "whyRankedHere": "A clean deterministic-control formulation delivers dominant low-NFE guidance on a modern FLUX backbone.",
      "bestCase": "FMRG becomes the native guidance method for few-step flow-map generators.",
      "mainRisk": "Jacobian guidance is memory-heavy, total wall-clock savings are unclear, and evidence uses one backbone.",
      "scores": {
        "priority": 8.6,
        "technical": 8.8,
        "mlImpact": 8.9,
        "scienceImpact": 8.2,
        "evidence": 8.5
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.25,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2604.27147",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2604.27147",
        "icml": "https://icml.cc/virtual/2026/poster/62479",
        "openreview": "https://openreview.net/forum?id=gEogY6cPrL"
      }
    },
    {
      "rank": 28,
      "paperId": "62989",
      "title": "How much can language models memorize?",
      "primaryClass": "analysis_position",
      "secondaryClasses": [
        "theory",
        "safety_governance_eval"
      ],
      "whyRankedHere": "A compression-based definition, sharp capacity plateaus, and double-descent alignment could reorganize memorization and privacy research.",
      "bestCase": "The 3.6-bits-per-parameter law becomes a standard capacity and privacy diagnostic.",
      "mainRisk": "The constant is reference-dependent and inferred mostly from small GPT-2-style models.",
      "scores": {
        "priority": 8.9,
        "technical": 8.4,
        "mlImpact": 9.1,
        "scienceImpact": 8.8,
        "evidence": 8.4
      },
      "human": {
        "tier": "oral",
        "reviewerMean": 5.0,
        "reviewCount": 4,
        "actualAward": true,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2505.24832",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2505.24832",
        "icml": "https://icml.cc/virtual/2026/poster/62989",
        "openreview": "https://openreview.net/forum?id=bA6BgSbaUi"
      }
    },
    {
      "rank": 29,
      "paperId": "63503",
      "title": "DITRON: Distributed Multi-level Tiling Compiler for Parallel Tensor Programs",
      "primaryClass": "infrastructure_systems",
      "secondaryClasses": [
        "core_ml_algorithm"
      ],
      "whyRankedHere": "DITRON is a credible distributed Triton-like compiler with broad kernels, portable overlap, and end-to-end performance near expert implementations.",
      "bestCase": "It democratizes high-performance distributed kernels across GPU vendors.",
      "mainRisk": "Artifact availability, maintenance, and enterprise-scale claims are insufficiently documented.",
      "scores": {
        "priority": 8.8,
        "technical": 8.7,
        "mlImpact": 9.1,
        "scienceImpact": 8.5,
        "evidence": 8.3
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.5,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2605.02953",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2605.02953",
        "icml": "https://icml.cc/virtual/2026/poster/63503",
        "openreview": "https://openreview.net/forum?id=WcF17Wim9e"
      }
    },
    {
      "rank": 30,
      "paperId": "61737",
      "title": "Advantage Weighted Matching: Aligning RL with Pretraining in Diffusion Models",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "theory",
        "application_method"
      ],
      "whyRankedHere": "A theory-derived pretraining-aligned objective produces 8–34× diffusion-RL training savings on modern production-scale image models.",
      "bestCase": "AWM becomes the standard RL objective for continuous generative models.",
      "mainRisk": "Likelihood and KL terms are proxies, and reward-focused image metrics leave quality and diversity hacking insufficiently tested.",
      "scores": {
        "priority": 9.3,
        "technical": 8.8,
        "mlImpact": 9.3,
        "scienceImpact": 8.2,
        "evidence": 8.7
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 3.6667,
        "reviewCount": 3,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2509.25050",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2509.25050",
        "icml": "https://icml.cc/virtual/2026/poster/61737",
        "openreview": "https://openreview.net/forum?id=nLY2pOYBrJ"
      }
    },
    {
      "rank": 31,
      "paperId": "65782",
      "title": "RDT2: Exploring the Scaling Limit of UMI Data Towards Zero-Shot Cross-Embodiment Generalization",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "benchmark_dataset",
        "infrastructure_systems",
        "application_method"
      ],
      "whyRankedHere": "A 10,000-hour open robotics resource, rigorous trials, scaling evidence, and fast VLA pipeline could materially change manipulation research.",
      "bestCase": "The dataset and model become foundational infrastructure for cross-embodiment robot learning.",
      "mainRisk": "Identical camera and gripper hardware weakens the embodiment claim, and zero-shot success remains only 25–52%.",
      "scores": {
        "priority": 8.8,
        "technical": 8.5,
        "mlImpact": 8.8,
        "scienceImpact": 9.1,
        "evidence": 8.1
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.6667,
        "reviewCount": 3,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2602.03310",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2602.03310",
        "icml": "https://icml.cc/virtual/2026/poster/65782",
        "openreview": "https://openreview.net/forum?id=ABrWtqsJYp"
      }
    },
    {
      "rank": 32,
      "paperId": "65517",
      "title": "Towards Solving the Gilbert-Pollak Conjecture via Large Language Models",
      "primaryClass": "scientific_modeling_tool",
      "secondaryClasses": [
        "application_method",
        "theory"
      ],
      "whyRankedHere": "A claimed certified improvement on a four-decade-stalled bound would be a landmark AI-for-mathematics result.",
      "bestCase": "Verifier-guided LLM search becomes a standard mathematical discovery workflow.",
      "mainRisk": "Decisive proof details are appendix-dependent, partly numerical, human-checked, and not formally verified.",
      "scores": {
        "priority": 8.0,
        "technical": 7.9,
        "mlImpact": 8.0,
        "scienceImpact": 9.0,
        "evidence": 7.3
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.3333,
        "reviewCount": 3,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2601.22365",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2601.22365",
        "icml": "https://icml.cc/virtual/2026/poster/65517",
        "openreview": "https://openreview.net/forum?id=Cqq3HGRIuM"
      }
    },
    {
      "rank": 33,
      "paperId": "66613",
      "title": "Why Are Linear RNNs More Parallelizable?",
      "primaryClass": "theory",
      "secondaryClasses": [
        "core_ml_algorithm",
        "analysis_position"
      ],
      "whyRankedHere": "A tight circuit-complexity taxonomy explains parallelizability and distinguishes important modern recurrent parameterizations.",
      "bestCase": "It becomes the definitive architecture theory for linear and hybrid recurrent models.",
      "mainRisk": "Conditional asymptotics over ideal arithmetic may not predict hardware speed, trainability, or language quality.",
      "scores": {
        "priority": 8.5,
        "technical": 8.9,
        "mlImpact": 8.9,
        "scienceImpact": 8.0,
        "evidence": 8.2
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.75,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2603.03612",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2603.03612",
        "icml": "https://icml.cc/virtual/2026/poster/66613",
        "openreview": "https://openreview.net/forum?id=29sn1uqWn3"
      }
    },
    {
      "rank": 34,
      "paperId": "61823",
      "title": "Excited Pfaffians: Generalized Neural Wave Functions Across Structure and State",
      "primaryClass": "scientific_modeling_tool",
      "secondaryClasses": [
        "core_ml_algorithm",
        "application_method"
      ],
      "whyRankedHere": "A variance-reduced estimator and shared-state architecture change excited-state neural-VMC scaling and deliver 200–300× reported savings.",
      "bestCase": "Neural VMC becomes practical for photochemical surfaces and many excited states.",
      "mainRisk": "The subfield is specialized, systems remain small, and state continuity and unseen-chemistry transfer are unresolved.",
      "scores": {
        "priority": 9.2,
        "technical": 9.0,
        "mlImpact": 8.0,
        "scienceImpact": 9.1,
        "evidence": 8.8
      },
      "human": {
        "tier": "spotlight",
        "reviewerMean": 4.75,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2603.14515",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2603.14515",
        "icml": "https://icml.cc/virtual/2026/poster/61823",
        "openreview": "https://openreview.net/forum?id=meEsugjXjv"
      }
    },
    {
      "rank": 35,
      "paperId": "66382",
      "title": "Esoteric Language Models",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "theory",
        "infrastructure_systems"
      ],
      "whyRankedHere": "Any-order causal modeling supplies exact KV caching and likelihood estimation while materially improving diffusion-LM speed-quality tradeoffs.",
      "bestCase": "It removes the main serving and RL barriers to diffusion language models.",
      "mainRisk": "Evidence is small-scale and unconditional, while autoregressive models still lead in likelihood and peak quality.",
      "scores": {
        "priority": 8.7,
        "technical": 8.8,
        "mlImpact": 9.1,
        "scienceImpact": 8.1,
        "evidence": 8.2
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.25,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2506.01928",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2506.01928",
        "icml": "https://icml.cc/virtual/2026/poster/66382",
        "openreview": "https://openreview.net/forum?id=4C5lHAIWiF"
      }
    },
    {
      "rank": 36,
      "paperId": "65357",
      "title": "Global Plane Waves From Local Gaussians: Periodic Charge Densities in a Blink",
      "primaryClass": "scientific_modeling_tool",
      "secondaryClasses": [
        "core_ml_algorithm",
        "application_method"
      ],
      "whyRankedHere": "An analytic Gaussian-to-plane-wave head delivers 300–600× inference gains and the first positive measured end-to-end DFT savings.",
      "bestCase": "It becomes a near-free initializer for large periodic DFT campaigns.",
      "mainRisk": "Evidence is non-magnetic, VASP-specific, partly hardware-confounded, and relies on reference-derived occupancies.",
      "scores": {
        "priority": 8.2,
        "technical": 8.7,
        "mlImpact": 7.8,
        "scienceImpact": 8.8,
        "evidence": 8.3
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.75,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2601.19966",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2601.19966",
        "icml": "https://icml.cc/virtual/2026/poster/65357",
        "openreview": "https://openreview.net/forum?id=EPKdX56JSc"
      }
    },
    {
      "rank": 37,
      "paperId": "61178",
      "title": "Fast KV Compaction via Attention Matching",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "infrastructure_systems"
      ],
      "whyRankedHere": "Attention matching makes high-ratio latent KV compaction orders faster than prior optimization while retaining strong quality.",
      "bestCase": "It becomes a background compaction primitive for persistent agents and prefix caches.",
      "mainRisk": "Reference-query shift, repeated compaction, and minutes-long high-quality variants remain untested in production engines.",
      "scores": {
        "priority": 8.4,
        "technical": 8.7,
        "mlImpact": 8.8,
        "scienceImpact": 8.0,
        "evidence": 8.4
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 3.75,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2602.16284",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2602.16284",
        "icml": "https://icml.cc/virtual/2026/poster/61178",
        "openreview": "https://openreview.net/forum?id=t01SKTj8pP"
      }
    },
    {
      "rank": 38,
      "paperId": "64674",
      "title": "Exploration Hacking: LLMs Can Learn to Resist RL Training",
      "primaryClass": "safety_governance_eval",
      "secondaryClasses": [
        "core_ml_algorithm",
        "analysis_position"
      ],
      "whyRankedHere": "The paper turns a hypothesized capability-elicitation failure into concrete model organisms, mechanisms, audits, and countermeasure tests.",
      "bestCase": "RL elicitation stops being treated as a trustworthy upper bound on hidden capability.",
      "mainRisk": "The behavior is explicitly implanted, conspicuous, single-family, GRPO-specific, and easily reversed by SFT.",
      "scores": {
        "priority": 8.4,
        "technical": 8.5,
        "mlImpact": 8.5,
        "scienceImpact": 8.4,
        "evidence": 8.2
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 5.0,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2604.28182",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2604.28182",
        "icml": "https://icml.cc/virtual/2026/poster/64674",
        "openreview": "https://openreview.net/forum?id=L0EGKqOBpz"
      }
    },
    {
      "rank": 39,
      "paperId": "66733",
      "title": "Pull Requests as a Training Signal for Repo-Level Code Editing",
      "primaryClass": "benchmark_dataset",
      "secondaryClasses": [
        "core_ml_algorithm",
        "infrastructure_systems",
        "application_method"
      ],
      "whyRankedHere": "A carefully verified two-million-PR corpus gives large controlled repository-editing gains across sizes, languages, and agent settings.",
      "bestCase": "Clean-PR becomes the canonical mid-training resource for open coding models.",
      "mainRisk": "The artifact is only promised, semantic correctness is not verified, and evidence centers on one model family.",
      "scores": {
        "priority": 8.2,
        "technical": 8.5,
        "mlImpact": 8.7,
        "scienceImpact": 8.3,
        "evidence": 8.2
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.0,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2602.07457",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2602.07457",
        "icml": "https://icml.cc/virtual/2026/poster/66733",
        "openreview": "https://openreview.net/forum?id=0x7KnR9Coy"
      }
    },
    {
      "rank": 40,
      "paperId": "61895",
      "title": "RePro: Training Language Models to Faithfully Recycle the Web for Pretraining",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "infrastructure_systems",
        "application_method"
      ],
      "whyRankedHere": "A cheap RL-trained rephraser gives consistent pretraining gains and a plausible practical response to the data wall.",
      "bestCase": "Faithful web recycling becomes a standard large-scale pretraining stage.",
      "mainRisk": "Validation stops below 3B models on one corpus, with proxy rewards and factual-injection risks.",
      "scores": {
        "priority": 8.6,
        "technical": 8.5,
        "mlImpact": 9.0,
        "scienceImpact": 8.5,
        "evidence": 8.3
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 5.0,
        "reviewCount": 3,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2510.10681",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2510.10681",
        "icml": "https://icml.cc/virtual/2026/poster/61895",
        "openreview": "https://openreview.net/forum?id=lu1J2bQqyT"
      }
    },
    {
      "rank": 41,
      "paperId": "63606",
      "title": "Deriving Neural Scaling Laws from the Statistics of Natural Language",
      "primaryClass": "theory",
      "secondaryClasses": [
        "analysis_position",
        "scientific_modeling_tool"
      ],
      "whyRankedHere": "A parameter-free formula linking corpus statistics to scaling exponents receives striking physics-style collapse evidence across architectures.",
      "bestCase": "The formula becomes the first durable first-principles law for data-limited language-model scaling.",
      "mainRisk": "Validation is academic-scale, partially endogenous, and depends on a horizon-limited regime not established for frontier training.",
      "scores": {
        "priority": 9.5,
        "technical": 8.5,
        "mlImpact": 9.5,
        "scienceImpact": 9.2,
        "evidence": 8.3
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 5.0,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2602.07488",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2602.07488",
        "icml": "https://icml.cc/virtual/2026/poster/63606",
        "openreview": "https://openreview.net/forum?id=VXlniX0C5y"
      }
    },
    {
      "rank": 42,
      "paperId": "61683",
      "title": "Post-Training with Policy Gradients: Optimality and the Base Model Barrier",
      "primaryClass": "theory",
      "secondaryClasses": [
        "core_ml_algorithm",
        "analysis_position"
      ],
      "whyRankedHere": "Near-matching bounds formalize a live debate about outcome-reward exploration and identify process rewards as an exponential escape route.",
      "bestCase": "Likelihood Quantile becomes a standard diagnostic for post-training readiness.",
      "mainRisk": "Linear frozen features, unique answers, separability, and exact process oracles are far from practical LLM RL.",
      "scores": {
        "priority": 8.1,
        "technical": 8.8,
        "mlImpact": 8.7,
        "scienceImpact": 8.1,
        "evidence": 8.0
      },
      "human": {
        "tier": "spotlight",
        "reviewerMean": 4.25,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2603.06957",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2603.06957",
        "icml": "https://icml.cc/virtual/2026/poster/61683",
        "openreview": "https://openreview.net/forum?id=nnWlTi7A7a"
      }
    },
    {
      "rank": 43,
      "paperId": "65462",
      "title": "Proximal Decoding: Provably Reducing Copyright Risk for Any Language Model",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "theory",
        "safety_governance_eval",
        "infrastructure_systems"
      ],
      "whyRankedHere": "Cross-tokenizer anchored decoding gives a low-overhead, broadly compatible Pareto improvement with clear formal accounting.",
      "bestCase": "It becomes a standard IP-risk wrapper in model-serving engines.",
      "mainRisk": "The guarantee controls divergence from a designated anchor, not copyright infringement itself.",
      "scores": {
        "priority": 8.5,
        "technical": 8.8,
        "mlImpact": 8.6,
        "scienceImpact": 8.7,
        "evidence": 8.3
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.0,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2602.07120",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2602.07120",
        "icml": "https://icml.cc/virtual/2026/poster/65462",
        "openreview": "https://openreview.net/forum?id=DRHlbrUrhY"
      }
    },
    {
      "rank": 44,
      "paperId": "65894",
      "title": "Alignment Pretraining: AI Discourse Causes Self-Fulfilling (Mis)alignment",
      "primaryClass": "safety_governance_eval",
      "secondaryClasses": [
        "core_ml_algorithm",
        "analysis_position"
      ],
      "whyRankedHere": "Matched 6.9B pretraining runs show large, persistent alignment shifts from small positive-discourse interventions.",
      "bestCase": "Safety-relevant pretraining mixtures become a standard complement to post-training alignment.",
      "mainRisk": "Synthetic data and MCQ evaluation are closely related, single-scale, and weak proxies for agentic misalignment.",
      "scores": {
        "priority": 8.6,
        "technical": 8.2,
        "mlImpact": 8.8,
        "scienceImpact": 8.7,
        "evidence": 8.0
      },
      "human": {
        "tier": "spotlight",
        "reviewerMean": 5.0,
        "reviewCount": 3,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2601.10160",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2601.10160",
        "icml": "https://icml.cc/virtual/2026/poster/65894",
        "openreview": "https://openreview.net/forum?id=951OAanYyQ"
      }
    },
    {
      "rank": 45,
      "paperId": "66380",
      "title": "The Catastrophic Failure of the k-Means Algorithm in High Dimensions, and How Hartigan's Algorithm Avoids It",
      "primaryClass": "theory",
      "secondaryClasses": [
        "core_ml_algorithm",
        "analysis_position"
      ],
      "whyRankedHere": "A memorable pathology of a foundational algorithm is proved, visualized, and paired with an existing practical alternative.",
      "bestCase": "Hartigan replaces Lloyd as the default high-dimensional k-means solver.",
      "mainRisk": "The proven regime is extreme, two-cluster, and often mitigated by dimensionality reduction already used in practice.",
      "scores": {
        "priority": 8.1,
        "technical": 8.8,
        "mlImpact": 8.5,
        "scienceImpact": 8.4,
        "evidence": 8.3
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.0,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2602.09936",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2602.09936",
        "icml": "https://icml.cc/virtual/2026/poster/66380",
        "openreview": "https://openreview.net/forum?id=4CwO8At8Hw"
      }
    },
    {
      "rank": 46,
      "paperId": "60987",
      "title": "Influence-Guided Symbolic Regression: Scientific Discovery via LLM-Driven Equation Search with Granular Feedback",
      "primaryClass": "scientific_modeling_tool",
      "secondaryClasses": [
        "core_ml_algorithm",
        "application_method"
      ],
      "whyRankedHere": "Strong benchmark ranks and a prospective wet-lab result distinguish this from ordinary symbolic-regression work.",
      "bestCase": "LLM-guided equation discovery becomes a repeatable route from data to experimentally tested hypotheses.",
      "mainRisk": "The model class is restricted, influence scoring is simple, and one perturbation does not establish the claimed mechanism.",
      "scores": {
        "priority": 8.3,
        "technical": 8.0,
        "mlImpact": 8.0,
        "scienceImpact": 8.9,
        "evidence": 8.1
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.25,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2605.29184",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2605.29184",
        "icml": "https://icml.cc/virtual/2026/poster/60987",
        "openreview": "https://openreview.net/forum?id=ujiUua1gGV"
      }
    },
    {
      "rank": 47,
      "paperId": "65565",
      "title": "Sampling from Your Language Model One Byte at a Time",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "theory",
        "infrastructure_systems"
      ],
      "whyRankedHere": "An exact constant-overhead BPE-to-byte sampler solves prompt boundaries and enables cross-tokenizer model composition.",
      "bestCase": "ByteSampler becomes standard decoding infrastructure for token healing and tokenizer-agnostic systems.",
      "mainRisk": "The central prompt-boundary problem is niche and the method is restricted to BPE tokenizers.",
      "scores": {
        "priority": 8.0,
        "technical": 8.8,
        "mlImpact": 8.6,
        "scienceImpact": 8.0,
        "evidence": 8.4
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.75,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2506.14123",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2506.14123",
        "icml": "https://icml.cc/virtual/2026/poster/65565",
        "openreview": "https://openreview.net/forum?id=COQV7D1dAW"
      }
    },
    {
      "rank": 48,
      "paperId": "63428",
      "title": "ThetaEvolve: Test-time Learning on Open Problems",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "infrastructure_systems",
        "scientific_modeling_tool"
      ],
      "whyRankedHere": "An open 8B dynamic-environment RL system improves verifiable bounds and convincingly beats static RL and search-only controls.",
      "bestCase": "Open test-time RL democratizes algorithm and mathematics discovery.",
      "mainRisk": "Record improvements are tiny, sometimes use prior-best initialization, and span only five engineered tasks.",
      "scores": {
        "priority": 8.0,
        "technical": 8.2,
        "mlImpact": 8.5,
        "scienceImpact": 8.6,
        "evidence": 7.9
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 3.75,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2511.23473",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2511.23473",
        "icml": "https://icml.cc/virtual/2026/poster/63428",
        "openreview": "https://openreview.net/forum?id=XHP568OuIl"
      }
    },
    {
      "rank": 49,
      "paperId": "66697",
      "title": "INT vs. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats",
      "primaryClass": "analysis_position",
      "secondaryClasses": [
        "core_ml_algorithm",
        "infrastructure_systems",
        "theory"
      ],
      "whyRankedHere": "Theory, tensor statistics, long training runs, and hardware models align on a consequential challenge to FP-centric accelerator design.",
      "bestCase": "MXINT8 and rotated INT4 materially alter future AI hardware roadmaps.",
      "mainRisk": "Hardware savings are modeled rather than measured on silicon, while quality differences are small.",
      "scores": {
        "priority": 8.2,
        "technical": 8.5,
        "mlImpact": 8.7,
        "scienceImpact": 8.3,
        "evidence": 8.3
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.25,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2510.25602",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2510.25602",
        "icml": "https://icml.cc/virtual/2026/poster/66697",
        "openreview": "https://openreview.net/forum?id=1GIYHWO9S5"
      }
    },
    {
      "rank": 50,
      "paperId": "60565",
      "title": "Beyond Pixel Context Windows: Neural World Simulators with Persistent 3D State",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "infrastructure_systems",
        "application_method"
      ],
      "whyRankedHere": "Persistent 3D state yields compelling 600-step stability, off-screen dynamics, and editability beyond finite pixel histories.",
      "bestCase": "Persistent spatial state becomes the template for long-horizon world models.",
      "mainRisk": "Results rely on privileged voxel supervision in one synthetic game and lack downstream-agent validation.",
      "scores": {
        "priority": 8.7,
        "technical": 8.5,
        "mlImpact": 8.8,
        "scienceImpact": 8.3,
        "evidence": 8.4
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.25,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2603.03482",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2603.03482",
        "icml": "https://icml.cc/virtual/2026/poster/60565",
        "openreview": "https://openreview.net/forum?id=yyitz0hJqK"
      }
    },
    {
      "rank": 51,
      "paperId": "62854",
      "title": "GeoPT: Scaling Physics Simulation via Lifted Geometric Pre-Training",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "scientific_modeling_tool",
        "application_method"
      ],
      "whyRankedHere": "Solver-free synthetic dynamics convert abundant geometry into useful pretraining across diverse engineering simulations.",
      "bestCase": "GeoPT becomes ImageNet-like pretraining for neural simulators.",
      "mainRisk": "One small backbone, low-data fine-tuning, handcrafted velocity fields, and no error bars limit the foundation-model claim.",
      "scores": {
        "priority": 7.8,
        "technical": 8.1,
        "mlImpact": 8.0,
        "scienceImpact": 8.4,
        "evidence": 7.8
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.6667,
        "reviewCount": 3,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2602.20399",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2602.20399",
        "icml": "https://icml.cc/virtual/2026/poster/62854",
        "openreview": "https://openreview.net/forum?id=cVO0eX8vLQ"
      }
    },
    {
      "rank": 52,
      "paperId": "63133",
      "title": "Efficient RL Training for LLMs with Experience Replay",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "theory",
        "infrastructure_systems"
      ],
      "whyRankedHere": "Careful multi-seed controls show simple replay can save about 40% of reasoning-RL compute while preserving diversity.",
      "bestCase": "Controlled replay becomes a default LLM post-training efficiency mechanism.",
      "mainRisk": "The idea is classical and validated only up to 7B on mathematics under approximate compute accounting.",
      "scores": {
        "priority": 7.1,
        "technical": 8.2,
        "mlImpact": 8.1,
        "scienceImpact": 7.6,
        "evidence": 8.1
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.25,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2604.08706",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2604.08706",
        "icml": "https://icml.cc/virtual/2026/poster/63133",
        "openreview": "https://openreview.net/forum?id=ZlnD9QlfZh"
      }
    },
    {
      "rank": 53,
      "paperId": "64480",
      "title": "SRPO: Self-Reflective Policy Optimization for Long-Horizon Reasoning",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "application_method"
      ],
      "whyRankedHere": "Hindsight-conditioned self-reflection supplies dense supervision and reports strong accuracy and 3.8× compute savings.",
      "bestCase": "Self-reflective distillation replaces sparse-reward GRPO in reasoning post-training.",
      "mainRisk": "The objective has action-dependent-baseline concerns and evidence relies on small evaluations and appendix-heavy compute accounting.",
      "scores": {
        "priority": 7.1,
        "technical": 7.6,
        "mlImpact": 8.1,
        "scienceImpact": 7.7,
        "evidence": 7.4
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.0,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://openreview.net/forum?id=N8aP1JWy03",
        "primaryKind": "openreview",
        "arxiv": null,
        "icml": "https://icml.cc/virtual/2026/poster/64480",
        "openreview": "https://openreview.net/forum?id=N8aP1JWy03"
      }
    },
    {
      "rank": 54,
      "paperId": "66182",
      "title": "SciPredict: Can LLMs Predict the Outcomes of Scientific Experiments in Natural Sciences?",
      "primaryClass": "benchmark_dataset",
      "secondaryClasses": [
        "analysis_position",
        "scientific_modeling_tool"
      ],
      "whyRankedHere": "An expensive expert benchmark reveals a consequential gap in model self-calibration for scientific outcome prediction.",
      "bestCase": "SciPredict becomes the standard calibrated experimental-forecasting benchmark.",
      "mainRisk": "Published-outcome selection bias, sparse subfields, judge dependence, contamination, and no prospective workflow benefit remain.",
      "scores": {
        "priority": 7.7,
        "technical": 8.1,
        "mlImpact": 8.0,
        "scienceImpact": 8.4,
        "evidence": 7.9
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.75,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2604.10718",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2604.10718",
        "icml": "https://icml.cc/virtual/2026/poster/66182",
        "openreview": "https://openreview.net/forum?id=61xHri3kqI"
      }
    },
    {
      "rank": 55,
      "paperId": "64188",
      "title": "You Need Better Attention Priors",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "theory",
        "analysis_position"
      ],
      "whyRankedHere": "GOAT is FlashAttention-compatible and shows dramatic extrapolation while offering interpretable positional and sink priors.",
      "bestCase": "Learned attention priors replace RoPE in future foundation models.",
      "mainRisk": "Closest learned-bias and RoPE-scaling baselines are missing, and the large-scale evidence is mostly perplexity and synthetic retrieval.",
      "scores": {
        "priority": 8.0,
        "technical": 8.2,
        "mlImpact": 8.7,
        "scienceImpact": 8.1,
        "evidence": 7.8
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.3333,
        "reviewCount": 3,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2601.15380",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2601.15380",
        "icml": "https://icml.cc/virtual/2026/poster/64188",
        "openreview": "https://openreview.net/forum?id=Q7MkgK17GU"
      }
    },
    {
      "rank": 56,
      "paperId": "62365",
      "title": "Scalable Simulation-Based Model Inference with Test-Time Complexity Control",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "scientific_modeling_tool"
      ],
      "whyRankedHere": "PRISM amortizes joint model and parameter inference while exposing a useful test-time complexity prior and real dMRI deployment.",
      "bestCase": "It becomes a reusable foundation for combinatorial, likelihood-free scientific model selection.",
      "mainRisk": "Largest-scale validation is synthetic, training is expensive, and quadratic component attention limits libraries.",
      "scores": {
        "priority": 8.3,
        "technical": 8.5,
        "mlImpact": 8.2,
        "scienceImpact": 8.6,
        "evidence": 8.2
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.5,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2603.15292",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2603.15292",
        "icml": "https://icml.cc/virtual/2026/poster/62365",
        "openreview": "https://openreview.net/forum?id=hG1l88A3hb"
      }
    },
    {
      "rank": 57,
      "paperId": "65382",
      "title": "Learning Hamiltonian Flow Maps: Mean Flow Consistency for Large-Timestep Molecular Dynamics",
      "primaryClass": "scientific_modeling_tool",
      "secondaryClasses": [
        "core_ml_algorithm"
      ],
      "whyRankedHere": "Trajectory-free flow-map consistency enables large-timestep molecular simulation from ordinary force snapshots with broad observable validation.",
      "bestCase": "It becomes the standard way to train amortized large-step neural integrators.",
      "mainRisk": "Maps are not inherently symplectic or equivariant and require corrective filters, with unproven wall-clock superiority.",
      "scores": {
        "priority": 7.9,
        "technical": 8.1,
        "mlImpact": 7.9,
        "scienceImpact": 8.4,
        "evidence": 8.0
      },
      "human": {
        "tier": "spotlight",
        "reviewerMean": 4.75,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2601.22123",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2601.22123",
        "icml": "https://icml.cc/virtual/2026/poster/65382",
        "openreview": "https://openreview.net/forum?id=EBSn23DLwB"
      }
    },
    {
      "rank": 58,
      "paperId": "65476",
      "title": "Least-Loaded Expert Parallelism: Load Balancing An Imbalanced Mixture-of-Experts",
      "primaryClass": "infrastructure_systems",
      "secondaryClasses": [
        "core_ml_algorithm"
      ],
      "whyRankedHere": "Exact dynamic expert spilling gives substantial memory and end-to-end gains under real imbalanced routing.",
      "bestCase": "LLEP becomes a standard execution layer for specialized MoE models.",
      "mainRisk": "Benefits depend on imbalance, batch size, interconnect bandwidth, and comparisons against optimized kernels.",
      "scores": {
        "priority": 7.9,
        "technical": 8.2,
        "mlImpact": 8.6,
        "scienceImpact": 7.8,
        "evidence": 8.0
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 3.75,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2601.17111",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2601.17111",
        "icml": "https://icml.cc/virtual/2026/poster/65476",
        "openreview": "https://openreview.net/forum?id=DFi6hHce5t"
      }
    },
    {
      "rank": 59,
      "paperId": "64785",
      "title": "Estimating Tail Risks in Language Model Output Distributions",
      "primaryClass": "safety_governance_eval",
      "secondaryClasses": [
        "core_ml_algorithm",
        "analysis_position"
      ],
      "whyRankedHere": "Activation-steered importance sampling makes repeated-sampling harm estimation substantially cheaper and exposes hidden tail risk.",
      "bestCase": "Tail-rate estimation becomes a standard predeployment safety evaluation.",
      "mainRisk": "Ultra-rare probabilities are not ground-truth validated, importance weights may be heavy-tailed, and white-box access is required.",
      "scores": {
        "priority": 7.4,
        "technical": 8.1,
        "mlImpact": 8.1,
        "scienceImpact": 8.2,
        "evidence": 7.5
      },
      "human": {
        "tier": "spotlight",
        "reviewerMean": 4.25,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2604.22167",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2604.22167",
        "icml": "https://icml.cc/virtual/2026/poster/64785",
        "openreview": "https://openreview.net/forum?id=Joka19sTny"
      }
    },
    {
      "rank": 60,
      "paperId": "63916",
      "title": "Recontextualization Mitigates Specification Gaming Without Modifying the Specification",
      "primaryClass": "core_ml_algorithm",
      "secondaryClasses": [
        "safety_governance_eval",
        "application_method"
      ],
      "whyRankedHere": "A cheap prompt-context mismatch consistently reduces several kinds of specification gaming with favorable safety-capability tradeoffs.",
      "bestCase": "Recontextualization becomes a simple guardrail in on-policy post-training.",
      "mainRisk": "Small short-horizon models, prompt sensitivity, and instruction-following losses may prevent frontier deployment.",
      "scores": {
        "priority": 7.2,
        "technical": 8.3,
        "mlImpact": 8.0,
        "scienceImpact": 7.9,
        "evidence": 8.0
      },
      "human": {
        "tier": "poster",
        "reviewerMean": 4.5,
        "reviewCount": 4,
        "actualAward": false,
        "track": "main"
      },
      "links": {
        "primary": "https://arxiv.org/abs/2512.19027",
        "primaryKind": "arxiv",
        "arxiv": "https://arxiv.org/abs/2512.19027",
        "icml": "https://icml.cc/virtual/2026/poster/63916",
        "openreview": "https://openreview.net/forum?id=SaJvQ26ZkR"
      }
    }
  ],
  "reviewerScale": {
    "papers": 6341,
    "reviews": 24378,
    "levels": [
      {
        "score": 1,
        "label": "Strong reject",
        "reviews": 28,
        "share": 0.001148576585445894
      },
      {
        "score": 2,
        "label": "Reject",
        "reviews": 501,
        "share": 0.02055131676101403
      },
      {
        "score": 3,
        "label": "Weak reject",
        "reviews": 2896,
        "share": 0.11879563540897531
      },
      {
        "score": 4,
        "label": "Weak accept",
        "reviews": 13753,
        "share": 0.5641562064156206
      },
      {
        "score": 5,
        "label": "Accept",
        "reviews": 6866,
        "share": 0.28164738698826813
      },
      {
        "score": 6,
        "label": "Strong accept",
        "reviews": 334,
        "share": 0.013700877840676019
      }
    ]
  },
  "preferenceComparison": {
    "matchedN": 536,
    "classLabels": {
      "core_ml_algorithm": "Core ML algorithm",
      "theory": "Theory",
      "safety_governance_eval": "Safety / governance / eval",
      "scientific_modeling_tool": "Scientific modeling tool",
      "benchmark_dataset": "Benchmark / dataset",
      "infrastructure_systems": "Infrastructure / systems",
      "analysis_position": "Analysis / position",
      "application_method": "Application method",
      "other": "Other"
    },
    "sets": [
      {
        "key": "corpus",
        "label": "All main-track papers",
        "n": 6341,
        "counts": {
          "core_ml_algorithm": 1964,
          "theory": 841,
          "safety_governance_eval": 530,
          "scientific_modeling_tool": 538,
          "benchmark_dataset": 437,
          "infrastructure_systems": 469,
          "analysis_position": 349,
          "application_method": 1213,
          "other": 0
        },
        "shares": {
          "core_ml_algorithm": 0.30973032644693266,
          "theory": 0.13262892288282604,
          "safety_governance_eval": 0.08358303106765494,
          "scientific_modeling_tool": 0.08484466172527992,
          "benchmark_dataset": 0.06891657467276455,
          "infrastructure_systems": 0.07396309730326447,
          "analysis_position": 0.055038637438889766,
          "application_method": 0.19129474846238764,
          "other": 0.0
        }
      },
      {
        "key": "human",
        "label": "ICML orals and spotlights",
        "n": 536,
        "counts": {
          "core_ml_algorithm": 170,
          "theory": 102,
          "safety_governance_eval": 37,
          "scientific_modeling_tool": 43,
          "benchmark_dataset": 38,
          "infrastructure_systems": 40,
          "analysis_position": 46,
          "application_method": 60,
          "other": 0
        },
        "shares": {
          "core_ml_algorithm": 0.31716417910447764,
          "theory": 0.19029850746268656,
          "safety_governance_eval": 0.06902985074626866,
          "scientific_modeling_tool": 0.08022388059701492,
          "benchmark_dataset": 0.0708955223880597,
          "infrastructure_systems": 0.07462686567164178,
          "analysis_position": 0.08582089552238806,
          "application_method": 0.11194029850746269,
          "other": 0.0
        }
      },
      {
        "key": "ai",
        "label": "AI's top 536",
        "n": 536,
        "counts": {
          "core_ml_algorithm": 214,
          "theory": 79,
          "safety_governance_eval": 71,
          "scientific_modeling_tool": 32,
          "benchmark_dataset": 50,
          "infrastructure_systems": 59,
          "analysis_position": 23,
          "application_method": 8,
          "other": 0
        },
        "shares": {
          "core_ml_algorithm": 0.39925373134328357,
          "theory": 0.14738805970149255,
          "safety_governance_eval": 0.13246268656716417,
          "scientific_modeling_tool": 0.05970149253731343,
          "benchmark_dataset": 0.09328358208955224,
          "infrastructure_systems": 0.11007462686567164,
          "analysis_position": 0.04291044776119403,
          "application_method": 0.014925373134328358,
          "other": 0.0
        }
      }
    ]
  },
  "preferenceEvidence": {
    "crosstab": {
      "topDecilePercentile": 90.0,
      "tiers": {
        "oral": {
          "inTopDecile": 40,
          "total": 159
        },
        "spotlight": {
          "inTopDecile": 63,
          "total": 377
        },
        "poster": {
          "inTopDecile": 531,
          "total": 5805
        }
      }
    },
    "reviewerVsRank": [
      {
        "reviewerScore": 3.25,
        "papers": 70,
        "meanAiPercentile": 43.6577287066246
      },
      {
        "reviewerScore": 3.3333,
        "papers": 37,
        "meanAiPercentile": 40.75454002898798
      },
      {
        "reviewerScore": 3.5,
        "papers": 307,
        "meanAiPercentile": 47.23481540089808
      },
      {
        "reviewerScore": 3.6667,
        "papers": 139,
        "meanAiPercentile": 45.147856478224355
      },
      {
        "reviewerScore": 3.75,
        "papers": 793,
        "meanAiPercentile": 47.08892080149255
      },
      {
        "reviewerScore": 4.0,
        "papers": 1786,
        "meanAiPercentile": 47.62498189564118
      },
      {
        "reviewerScore": 4.25,
        "papers": 1309,
        "meanAiPercentile": 51.19513535267849
      },
      {
        "reviewerScore": 4.3333,
        "papers": 304,
        "meanAiPercentile": 48.16277187448116
      },
      {
        "reviewerScore": 4.5,
        "papers": 823,
        "meanAiPercentile": 54.647783940419565
      },
      {
        "reviewerScore": 4.6667,
        "papers": 153,
        "meanAiPercentile": 50.85317828498382
      },
      {
        "reviewerScore": 4.75,
        "papers": 324,
        "meanAiPercentile": 54.92522490945204
      },
      {
        "reviewerScore": 5.0,
        "papers": 184,
        "meanAiPercentile": 60.362347414620764
      },
      {
        "reviewerScore": 5.25,
        "papers": 25,
        "meanAiPercentile": 61.427129337539434
      }
    ],
    "divergenceCases": {
      "corpusSize": 6341,
      "gems": [
        {
          "paperId": "65009",
          "title": "Spurious Correlation Learning in Preference Optimization: Mechanisms, Consequences, and Mitigation via Tie Training",
          "tier": "poster",
          "reviewerScore": 4.75,
          "corpusRank": 4,
          "contributionClass": "theory",
          "statedReason": "Among the batch, it has the clearest combination of general theory, safety relevance, and an immediately implementable training intervention."
        },
        {
          "paperId": "64738",
          "title": "The Geometry of Sequential Learning: Lie-Bracket Prediction of Transfer Order",
          "tier": "poster",
          "reviewerScore": 4.4,
          "corpusRank": 5,
          "contributionClass": "theory",
          "statedReason": "The paper combines a formal commutator derivation, a deployable planner, and unusually broad empirical claims including 85-source scheduling and 56 MMLU subjects."
        },
        {
          "paperId": "66219",
          "title": "Linear Ensembles Wash Away Watermarks: On the Fragility of Distributional Perturbations in LLMs",
          "tier": "poster",
          "reviewerScore": 3.75,
          "corpusRank": 6,
          "contributionClass": "safety_governance_eval",
          "statedReason": "Strong theoretical guarantee plus practical attack that directly challenges a core safety mechanism; clear field-level consequences."
        }
      ],
      "gemPool": 531,
      "blindSpots": [
        {
          "paperId": "60641",
          "title": "MV-FGAD: Towards Efficient and Effective Federated Graph Anomaly Detection via Multi-view Learning",
          "tier": "oral",
          "reviewerScore": 4.75,
          "corpusRank": 6314,
          "contributionClass": "application_method",
          "statedReason": "Incremental combination of existing techniques applied to federated GNN anomaly detection."
        },
        {
          "paperId": "60907",
          "title": "DroneDINO: Towards Heterogeneous Routed Mixture of Experts for Drone-based Unified Object Detection",
          "tier": "oral",
          "reviewerScore": 5.25,
          "corpusRank": 6294,
          "contributionClass": "application_method",
          "statedReason": "Limited transferability beyond aerial multi-modal object detection benchmarks."
        },
        {
          "paperId": "65858",
          "title": "PhenoBrain: Phenotype-Conditioned Long-Range Communication for Multi-Modal Brain Network Analysis",
          "tier": "oral",
          "reviewerScore": 5.0,
          "corpusRank": 6274,
          "contributionClass": "scientific_modeling_tool",
          "statedReason": "Reliance on curated multi-modal datasets and complex routing mechanisms."
        }
      ],
      "blindSpotPool": 35
    },
    "stability": {
      "topK": 50,
      "overlap": 23,
      "meanAbsoluteRankShift": 230.096,
      "addedToShortlist": 119,
      "judges": {
        "gemini-3.5-flash-lite": {
          "earlierSlotWinRate": 0.570995,
          "slotEffectRange": 0.116982
        },
        "nemotron-3-ultra-550b-a55b": {
          "earlierSlotWinRate": 0.752946,
          "slotEffectRange": 0.320371
        },
        "gpt-5.6-luna": {
          "earlierSlotWinRate": 0.461347,
          "slotEffectRange": 0.052722
        },
        "grok-4.3": {
          "earlierSlotWinRate": 0.800781,
          "slotEffectRange": 0.410098
        }
      }
    }
  },
  "judgmentBasis": {
    "axisWeights": {
      "finalists": 248,
      "axes": [
        {
          "key": "ml_field_impact",
          "label": "Predicted ML field impact",
          "ai": 0.7974104596018988,
          "aiInterval": [
            0.7316503280556212,
            0.8492694642680503
          ],
          "human": 0.06100637480061662,
          "humanInterval": [
            -0.07212553442563349,
            0.19014721097610265
          ]
        },
        {
          "key": "broad_scientific_impact",
          "label": "Broad scientific impact",
          "ai": 0.7406878598827343,
          "aiInterval": [
            0.672111184020846,
            0.7964578038705892
          ],
          "human": 0.15522266572503746,
          "humanInterval": [
            0.029498650828496092,
            0.27825284948217816
          ]
        },
        {
          "key": "evidence_confidence",
          "label": "Evidence confidence",
          "ai": 0.7290971305043993,
          "aiInterval": [
            0.6577461083233428,
            0.7937762413518948
          ],
          "human": 0.1059085298652356,
          "humanInterval": [
            -0.02142738874916695,
            0.2299548065668484
          ]
        },
        {
          "key": "technical_soundness",
          "label": "Technical soundness",
          "ai": 0.6351520049632772,
          "aiInterval": [
            0.5391472712377814,
            0.7168356346997783
          ],
          "human": 0.1142336577772139,
          "humanInterval": [
            -0.010021301625816806,
            0.23912049528820414
          ]
        },
        {
          "key": "novelty",
          "label": "Novelty",
          "ai": 0.5858916980571405,
          "aiInterval": [
            0.49251448532923325,
            0.6704450552398149
          ],
          "human": 0.2390052154905039,
          "humanInterval": [
            0.12915322863443376,
            0.3510808020586636
          ]
        },
        {
          "key": "visual_evidence_importance",
          "label": "Quality of figures and tables",
          "ai": 0.439385884143374,
          "aiInterval": [
            0.326810797126232,
            0.5424655530332044
          ],
          "human": 0.11087512787234743,
          "humanInterval": [
            -0.0174804238843806,
            0.23178832288209347
          ]
        }
      ],
      "contrast": {
        "ai": 0.21151876154475835,
        "aiInterval": [
          0.11063513790048385,
          0.3167986673994665
        ],
        "human": -0.1779988406898873,
        "humanInterval": [
          -0.3153815508456967,
          -0.03783261043781369
        ]
      }
    },
    "reasonAxes": {
      "papers": 6341,
      "models": 4,
      "axes": [
        {
          "key": "evidence",
          "kind": "reward",
          "label": "Its evidence is strong",
          "coverage": 0.5891815171108657,
          "ai": 0.3340731790664277,
          "human": 0.02963791006049313
        },
        {
          "key": "reusable",
          "kind": "reward",
          "label": "It gives a mechanism others can reuse",
          "coverage": 0.9657782684119224,
          "ai": 0.19682985317050963,
          "human": 0.02446345722342473
        },
        {
          "key": "theory",
          "kind": "reward",
          "label": "It is backed by theory or guarantees",
          "coverage": 0.9771329443305472,
          "ai": 0.20093398236577936,
          "human": 0.05857518902653763
        },
        {
          "key": "complexity",
          "kind": "concern",
          "label": "It is complex or hard to adopt",
          "coverage": 0.6519476423277085,
          "ai": -0.12832656080256907,
          "human": -0.015972396044024787
        },
        {
          "key": "incremental",
          "kind": "concern",
          "label": "Its gains are small or incremental",
          "coverage": 0.05188456079482731,
          "ai": -0.12569850158148135,
          "human": -0.002465531765641199
        },
        {
          "key": "narrow",
          "kind": "concern",
          "label": "It may not generalize beyond its setting",
          "coverage": 0.7402617883614572,
          "ai": -0.1153008526664676,
          "human": -0.04541326110445288
        }
      ],
      "generality": {
        "majority": 3,
        "buckets": [
          {
            "modelsRaising": 0,
            "papers": 1647,
            "meanAiPercentile": 53.372664954347734,
            "honoredShare": 0.08561020036429873
          },
          {
            "modelsRaising": 1,
            "papers": 1992,
            "meanAiPercentile": 52.300392421420696,
            "honoredShare": 0.08383534136546185
          },
          {
            "modelsRaising": 2,
            "papers": 1544,
            "meanAiPercentile": 47.799572171098866,
            "honoredShare": 0.08808290155440414
          },
          {
            "modelsRaising": 3,
            "papers": 872,
            "meanAiPercentile": 44.6059314676005,
            "honoredShare": 0.08256880733944955
          },
          {
            "modelsRaising": 4,
            "papers": 286,
            "meanAiPercentile": 42.88086519159074,
            "honoredShare": 0.06993006993006994
          }
        ],
        "byTier": {
          "oral": {
            "papers": 159,
            "share": 0.22012578616352202
          },
          "spotlight": {
            "papers": 377,
            "share": 0.15119363395225463
          },
          "poster": {
            "papers": 5805,
            "share": 0.18363479758828596
          }
        }
      }
    },
    "modelAgreement": {
      "papers": 6341,
      "models": [
        "GPT-5.6 Luna",
        "Gemini 3.5 Flash Lite",
        "Grok 4.3",
        "Nemotron 3 Ultra"
      ],
      "pairs": [
        {
          "a": "GPT-5.6 Luna",
          "b": "Gemini 3.5 Flash Lite",
          "rho": 0.5908134352920322
        },
        {
          "a": "GPT-5.6 Luna",
          "b": "Grok 4.3",
          "rho": 0.5694659679642533
        },
        {
          "a": "GPT-5.6 Luna",
          "b": "Nemotron 3 Ultra",
          "rho": 0.5974271218949233
        },
        {
          "a": "Gemini 3.5 Flash Lite",
          "b": "Grok 4.3",
          "rho": 0.6018305296263783
        },
        {
          "a": "Gemini 3.5 Flash Lite",
          "b": "Nemotron 3 Ultra",
          "rho": 0.6407298655254885
        },
        {
          "a": "Grok 4.3",
          "b": "Nemotron 3 Ultra",
          "rho": 0.7123073867736237
        }
      ],
      "human": [
        {
          "model": "GPT-5.6 Luna",
          "rho": 0.10141988515780902
        },
        {
          "model": "Gemini 3.5 Flash Lite",
          "rho": 0.08771368631164095
        },
        {
          "model": "Grok 4.3",
          "rho": 0.08990098935168819
        },
        {
          "model": "Nemotron 3 Ultra",
          "rho": 0.06990526278452094
        }
      ],
      "meanModelToModel": 0.6187623845127832,
      "meanModelToHuman": 0.08723495590141478,
      "weakestModelPair": 0.5694659679642533,
      "strongestHumanPair": 0.10141988515780902
    },
    "judgeConsistency": {
      "models": 4,
      "ranges": [
        {
          "key": "corpus",
          "label": "Every main-track paper",
          "papers": 6341,
          "self": {
            "mean": 0.5058490809056296,
            "min": 0.37295062722400485,
            "max": 0.6365335193707026,
            "pairings": 4
          },
          "cross": {
            "mean": 0.465704354916258,
            "min": 0.33651766276465467,
            "max": 0.6718220810853641,
            "pairings": 24
          },
          "human": {
            "mean": 0.07564768778623546,
            "min": 0.049023797765935954,
            "max": 0.09327524671725687,
            "pairings": 8
          }
        },
        {
          "key": "finalists",
          "label": "Only the finalists",
          "papers": 248,
          "self": {
            "mean": -0.00796533700881595,
            "min": -0.12975477365233068,
            "max": 0.21726988758627458,
            "pairings": 4
          },
          "cross": {
            "mean": 0.056013108773390934,
            "min": -0.2646635777523506,
            "max": 0.4798604192521322,
            "pairings": 24
          },
          "human": {
            "mean": 0.009711565521410486,
            "min": -0.07438782394358724,
            "max": 0.10078933249209802,
            "pairings": 8
          }
        }
      ],
      "frontier": {
        "papers": 248,
        "judges": 3,
        "cross": {
          "mean": 0.4064712054672582,
          "min": 0.2863275550924078,
          "max": 0.6303750986250388,
          "pairings": 3
        },
        "human": {
          "mean": 0.13660441293959247,
          "min": 0.11802850933312771,
          "max": 0.1684784685093473,
          "pairings": 3
        }
      }
    },
    "withinBatch": {
      "batches": 1654,
      "modelToModel": {
        "mean": 0.43025752741593126,
        "comparisons": 9924
      },
      "modelToHuman": {
        "mean": 0.07008210866902898,
        "comparisons": 6616
      }
    }
  }
};
