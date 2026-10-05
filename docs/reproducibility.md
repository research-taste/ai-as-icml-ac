# Reproducing the ICML 2026 Results

The production data are distributed separately from Git because the parsed
judgments and raw provider records are too large for a normal source checkout.
The repository defines two checksummed release artifacts:

1. **Analysis bundle.** Canonical paper and human-outcome metadata; parsed
   contribution routing, cheap-ensemble judgments, semifinal rankings,
   frontier cards, synthesis, tournament matches, Bradley-Terry estimates, and
   all deterministic Stage 7 outputs. It excludes PDFs, extracted paper text,
   credentials, and raw provider payloads.
2. **Audit archive.** The production prompt, response, parsed, usage, model-ID,
   and run-state records for every model stage. It preserves failed and fallback
   attempts when they are part of the production run. Historical probes and the
   archived pre-v27 contaminated run are not included.

Local absolute repository paths are replaced with `<REPO_ROOT>` during export.
The exporter rejects PDFs, environment files, and recognizable credential
patterns. Every payload has a SHA-256 checksum and source checksum in its
manifest.

## Build the Release

From the repository root:

```bash
python -m icml_ai_ac.reproducibility export \
  --include-audit \
  --output dist/icml_2026_reproducibility
```

This creates an unpacked analysis bundle plus deterministic ZIP archives. The
output directory is ignored by Git and is intended for later release through a
large-artifact host such as Hugging Face.

## Verify a Download

```bash
python -m icml_ai_ac.reproducibility verify \
  /path/to/analysis_bundle

python -m icml_ai_ac.reproducibility verify \
  /path/to/icml_2026_full_experiment_audit.zip
```

Verification checks the complete file inventory, byte sizes, and SHA-256
digests. It fails on missing, modified, or unexpected payload files.

## Query the Frozen Rankings

The analysis bundle contains the inputs needed to search papers and join them
to the frozen full-corpus and finalist rankings. From the bundle root, with this
package installed:

```bash
icml-ai-ac list-ranking-categories
icml-ai-ac top-ranked-papers \
  --category data-pretraining \
  --top 20 \
  --out data_pretraining_top20.jsonl
```

The output records why each paper matched, its full-corpus cheap rank, and its
final rank and stage. `top-ranked-papers` defaults to the 250-paper frontier
set; run `icml-ai-ac query-ranked-papers --help` for arbitrary combinations of
official-topic, contribution-class, free-text, scope, and output-format filters.

## Rebuild the Analysis

```bash
python -m icml_ai_ac.reproducibility rebuild \
  --bundle /path/to/analysis_bundle \
  --output /tmp/icml_2026_rebuilt
```

The rebuild makes no model or network calls. It regenerates:

- the three frozen Stage 7 strong-ranking views;
- nine agreement reports, including direct-win and Bradley-Terry sensitivity;
- four tier/reviewer divergence reports and their Markdown galleries;
- the AI-axis decomposition reports;
- the consolidated Stage 7 summary; and
- `site/data.js`, which drives all website results and visualizations.

By default the command checks every rebuilt file against the frozen production
reference using type-aware canonicalization: parsed JSON/JSONL values, the
website's embedded JSON payload, and normalized text. This ignores incidental
key order or final-newline differences while detecting any result change. A
collaborator can therefore alter a later analysis or ranking layer while
retaining the original upstream judgments, and can distinguish an intentional
change from accidental drift.

The artifact inventory is declarative in
[`reproducibility/icml_2026_release.json`](../reproducibility/icml_2026_release.json).
Update that file when a new published result depends on another production
artifact.
