# Static Report

The repository includes the complete frozen static blog: HTML, CSS, JavaScript,
generated result data, and preview assets. A fresh clone can build a portable,
ready-to-serve directory and ZIP without downloading experiment artifacts:

```bash
python3 site/build.py
```

This writes `build/icml-ai-ac-blog/` and `build/icml-ai-ac-blog.zip`. Serve the
built directory from the repository root:

```bash
python3 -m http.server 8080 --directory build/icml-ai-ac-blog
```

Then open `http://localhost:8080/site/`.

For development, the committed source can also be served directly:

```bash
python3 -m http.server 8080
```

Then open `http://localhost:8080/site/`.

## Rebuild the Result Data

The default build uses the committed, frozen `site/data.js`. To regenerate that
file from a separately downloaded analysis bundle while packaging the blog:

```bash
python3 site/build.py \
  --data-root /path/to/analysis_bundle/data \
  --output build/icml-ai-ac-blog-rebuilt
```

The lower-level `python3 site/build_data.py` command is also available when only
`site/data.js` should be regenerated. The higher-level reproducibility command
in [`docs/reproducibility.md`](../docs/reproducibility.md) rebuilds the human
comparison reports and website data together, then verifies them against the
frozen production references.

The mock program uses final tournament ranks 1-2 for Outstanding Papers,
ranks 3-7 for Honorable Mentions, and the complete 60-paper all-pairs playoff
for AI Orals. Paper panels link to 59 verified arXiv records; SRPO, the only
paper without a discoverable arXiv record, links to its official OpenReview
page. Six renamed or older arXiv versions are documented in
`arxiv_overrides.json`. First-page previews come from the original official
OpenReview PDFs. `build_data.py` fails if the published slate, expected link
coverage, canonical titles, tournament schedule, or post hoc human-outcome
counts change. The methodology diagram and human-comparison figures are also
generated from the frozen tournament, shortlist, and Stage 7 analysis rather
than maintained as hand-entered graphics.
