(() => {
  "use strict";

  const data = window.ICML_AI_AC_DATA;
  if (!data || !Array.isArray(data.papers)) {
    throw new Error("Missing generated site data. Run: python3 site/build_data.py");
  }

  const classLabels = {
    analysis_position: "Analysis / position",
    application_method: "Application method",
    benchmark_dataset: "Benchmark / dataset",
    core_ml_algorithm: "Core ML algorithm",
    infrastructure_systems: "Infrastructure / systems",
    safety_governance_eval: "Safety / governance / eval",
    scientific_modeling_tool: "Scientific modeling tool",
    theory: "Theory",
    other: "Other",
  };

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function externalLink(href, text, className = "") {
    const link = element("a", className, text);
    link.href = href;
    link.target = "_blank";
    link.rel = "noreferrer";
    return link;
  }

  function paperPanel(paper, className, context) {
    const destination = paper.links.primaryKind === "arxiv" ? "arXiv" : "OpenReview";
    const link = externalLink(paper.links.primary, undefined, className);
    link.setAttribute("aria-label", `${context}: ${paper.title}. View on ${destination}.`);
    return link;
  }

  function paperLinkCue(paper) {
    const text = paper.links.primaryKind === "arxiv" ? "View on arXiv" : "Official paper";
    return element("span", "paper-link-cue", text);
  }

  function humanTierLabel(tier) {
    if (tier === "oral") return "ICML oral";
    if (tier === "spotlight") return "ICML spotlight";
    return "ICML poster";
  }

  function classLabel(value) {
    return classLabels[value] || value.replaceAll("_", " ");
  }

  function ordinal(value) {
    const n = Math.round(value);
    const rest = n % 100;
    if (rest >= 11 && rest <= 13) return `${n}th`;
    return `${n}${["th", "st", "nd", "rd"][n % 10] || "th"}`;
  }

  function svgElement(tag, attributes = {}, text) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attributes).forEach(([name, value]) => node.setAttribute(name, String(value)));
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function renderMethodDiagram() {
    const container = document.querySelector("#method-diagram");
    const flow = element("div", "method-flow");
    data.method.stages.forEach((stage, index) => {
      const node = element("div", `method-node method-${stage.tone}`);
      node.append(
        element("span", "method-stage", `Stage ${stage.stage}`),
        element("strong", "method-count", stage.count.toLocaleString()),
        element("span", "method-title", stage.title),
        element("span", "method-models", stage.models),
        element("small", "method-detail", stage.detail),
      );
      flow.append(node);
      if (index < data.method.stages.length - 1) {
        const arrow = element("div", "method-arrow");
        arrow.setAttribute("aria-hidden", "true");
        flow.append(arrow);
      }
    });
    const phaseLabels = element("div", "method-phase-labels");
    phaseLabels.append(
      element("span", "", "Scoring and narrowing"),
      element("span", "", "Pairwise tournament"),
    );
    container.append(flow, phaseLabels);
  }

  function renderCorrelationScale() {
    const container = document.querySelector("#correlation-scale");
    const coverage = data.results && data.results.main_track_full_coverage;
    if (!container || !coverage) return;
    const rows = [
      ["How prominently the conference presented the paper", coverage.kendall_tau_b_vs_tier],
      ["The average score the paper's reviewers gave it", coverage.kendall_tau_b_vs_reviewer_overall],
    ];
    const width = 1120;
    const height = 210;
    const left = 470;
    const right = 1000;
    const x = (value) => left + value * (right - left);

    const svg = svgElement("svg", {
      class: "evidence-svg",
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": rows
        .map(([label, value]) => `Against ${label}, the rank correlation is ${value.toFixed(2)} on a scale from 0 to 1`)
        .join(". "),
    });
    svg.append(
      svgElement("text", { class: "evidence-axis-title", x: 160, y: 30 }, "Kendall's τ between the AI ranking and each of ICML's records"),
    );

    [0, 0.25, 0.5, 0.75, 1].forEach((value) => {
      svg.append(
        svgElement("line", { class: "evidence-grid", x1: x(value), x2: x(value), y1: 58, y2: 152 }),
        svgElement("text", { class: "evidence-tick", x: x(value), y: 174, "text-anchor": "middle" }, value.toFixed(2)),
      );
    });
    svg.append(
      svgElement("text", { class: "evidence-scale-end", x: x(0), y: 50, "text-anchor": "middle" }, "no relationship"),
      svgElement("text", { class: "evidence-scale-end", x: x(1), y: 50, "text-anchor": "middle" }, "identical orderings"),
    );

    rows.forEach(([label, value], index) => {
      const y = 84 + index * 44;
      svg.append(
        svgElement("text", { class: "evidence-row-label", x: left - 24, y: y + 5, "text-anchor": "end" }, label),
        svgElement("rect", { class: "evidence-track", x: x(0), y: y - 11, width: right - left, height: 22 }),
        svgElement("rect", { class: "evidence-fill evidence-oral", x: x(0), y: y - 11, width: Math.max(2, x(value) - x(0)), height: 22 }),
        svgElement("text", { class: "evidence-value", x: x(value) + 14, y: y + 6 }, value.toFixed(2)),
      );
    });
    container.append(svg);
  }

  function renderRecallByTier() {
    const evidence = data.preferenceEvidence;
    const container = document.querySelector("#recall-by-tier");
    if (!container || !evidence) return;
    const tiers = [
      ["oral", "Chosen as oral at ICML"],
      ["spotlight", "Chosen as spotlight at ICML"],
      ["poster", "Chosen as poster at ICML"],
    ];
    const width = 1120;
    const height = 268;
    const left = 300;
    const right = 1060;
    const rowHeight = 62;
    const firstRow = 74;
    const scale = (share) => left + share * (right - left);

    const svg = svgElement("svg", {
      class: "evidence-svg",
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": tiers
        .map(([key, label]) => {
          const row = evidence.crosstab.tiers[key];
          return `${label}: ${row.inTopDecile} of ${row.total}, ${((row.inTopDecile / row.total) * 100).toFixed(1)} percent, reached the AI top ten percent`;
        })
        .join(". "),
    });

    svg.append(
      svgElement("text", { class: "evidence-axis-title", x: left, y: 30 }, "Share of each presentation tier that reached the top 10% of the AI ranking"),
    );

    const baseline = scale(0.1);
    svg.append(
      svgElement("line", { class: "evidence-reference", x1: baseline, x2: baseline, y1: firstRow - 24, y2: firstRow + rowHeight * 3 - 22 }),
      svgElement("text", { class: "evidence-reference-label", x: baseline, y: firstRow + rowHeight * 3 - 2, "text-anchor": "middle" }, "10%, the share expected if the two orderings were unrelated"),
    );

    tiers.forEach(([key, label], index) => {
      const row = evidence.crosstab.tiers[key];
      const share = row.inTopDecile / row.total;
      const y = firstRow + index * rowHeight;
      svg.append(
        svgElement("text", { class: "evidence-row-label", x: left - 20, y: y + 5, "text-anchor": "end" }, label),
        svgElement("rect", { class: "evidence-track", x: left, y: y - 13, width: right - left, height: 26 }),
        svgElement("rect", { class: `evidence-fill evidence-${key}`, x: left, y: y - 13, width: scale(share) - left, height: 26 }),
        svgElement("text", { class: "evidence-value", x: scale(share) + 12, y: y + 5 }, `${(share * 100).toFixed(1)}%`),
        svgElement(
          "text",
          { class: "evidence-count", x: right, y: y + 5, "text-anchor": "end" },
          `${row.inTopDecile.toLocaleString()} of ${row.total.toLocaleString()}`,
        ),
      );
    });
    container.append(svg);
  }

  function renderReviewerVsRank() {
    const evidence = data.preferenceEvidence;
    const container = document.querySelector("#reviewer-vs-rank");
    if (!container || !evidence || !evidence.reviewerVsRank) return;
    const points = evidence.reviewerVsRank;
    const width = 1120;
    const height = 340;
    const left = 300;
    const right = 1040;
    const top = 70;
    const bottom = 268;
    const minScore = Math.min(...points.map((p) => p.reviewerScore));
    const maxScore = Math.max(...points.map((p) => p.reviewerScore));
    const x = (score) => left + ((score - minScore) / (maxScore - minScore)) * (right - left);
    const y = (percentile) => bottom - ((percentile - 30) / 45) * (bottom - top);
    const biggest = Math.max(...points.map((p) => p.papers));

    const svg = svgElement("svg", {
      class: "evidence-svg",
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": points
        .map((p) => `Papers scored ${p.reviewerScore} by reviewers sit on average at the ${p.meanAiPercentile.toFixed(0)}th percentile of the AI ranking`)
        .join(". "),
    });
    svg.append(
      svgElement("text", { class: "evidence-axis-title", x: left, y: 30 }, "Average position in the AI ranking, by the score the paper's human reviewers gave it"),
    );

    [40, 50, 60, 70].forEach((value) => {
      svg.append(
        svgElement("line", { class: value === 50 ? "evidence-reference" : "evidence-grid", x1: left, x2: right, y1: y(value), y2: y(value) }),
        svgElement("text", { class: "evidence-tick", x: left - 16, y: y(value) + 4, "text-anchor": "end" }, ordinal(value)),
      );
    });
    svg.append(
      svgElement("text", { class: "evidence-row-sub", x: left - 16, y: top - 6, "text-anchor": "end" }, "Rated higher by the AI"),
      svgElement("text", { class: "evidence-row-sub", x: left - 16, y: bottom + 6, "text-anchor": "end" }, "Rated lower by the AI"),
    );

    const path = points
      .map((p, index) => `${index === 0 ? "M" : "L"} ${x(p.reviewerScore)} ${y(p.meanAiPercentile)}`)
      .join(" ");
    svg.append(svgElement("path", { class: "evidence-trend", d: path }));

    points.forEach((p) => {
      const radius = 5 + Math.sqrt(p.papers / biggest) * 13;
      const dot = svgElement("g", { class: "evidence-dot" });
      dot.append(svgElement("title", {}, `${p.papers.toLocaleString()} papers scored ${p.reviewerScore}; average AI position ${p.meanAiPercentile.toFixed(1)}th percentile`));
      dot.append(svgElement("circle", { cx: x(p.reviewerScore), cy: y(p.meanAiPercentile), r: radius }));
      svg.append(dot);
    });

    points.forEach((p, index) => {
      const isFirst = index === 0;
      const isLast = index === points.length - 1;
      if (!isFirst && !isLast) return;
      svg.append(
        svgElement(
          "text",
          {
            class: "evidence-value",
            x: x(p.reviewerScore) + (isFirst ? 16 : -8),
            y: y(p.meanAiPercentile) + (isFirst ? 34 : -24),
            "text-anchor": isFirst ? "start" : "end",
          },
          ordinal(p.meanAiPercentile),
        ),
      );
    });

    [minScore, 4.0, 4.5, maxScore].forEach((score) => {
      svg.append(
        svgElement("text", { class: "evidence-tick", x: x(score), y: bottom + 26, "text-anchor": "middle" }, score.toFixed(2)),
      );
    });
    svg.append(
      svgElement("text", { class: "evidence-axis-caption", x: (left + right) / 2, y: height - 14, "text-anchor": "middle" }, "Mean score from the paper's reviewers (4 = weak accept, 5 = accept). Circle size shows how many papers had that mean."),
    );
    container.append(svg);
  }

  function renderPreferenceComparison() {
    const container = document.querySelector("#preference-comparison");
    const comparison = data.preferenceComparison;
    if (!container || !comparison) return;

    const sets = comparison.sets;
    const human = sets.find((set) => set.key === "human");
    const ai = sets.find((set) => set.key === "ai");
    // Order rows by how far AI's mix departs from the human mix.
    const rows = Object.keys(comparison.classLabels)
      .map((key) => ({
        key,
        label: comparison.classLabels[key],
        shares: sets.map((set) => set.shares[key] || 0),
        delta: (ai.shares[key] || 0) - (human.shares[key] || 0),
      }))
      .filter((row) => row.shares.some((share) => share > 0))
      .sort((a, b) => b.delta - a.delta);

    const widest = Math.max(...rows.map((row) => Math.abs(row.delta)), 0.01);
    const table = element("table", "preference-table");
    const headRow = element("tr");
    headRow.append(element("th", "preference-corner", "Contribution type"));
    sets.forEach((set) => {
      const cell = element("th");
      cell.append(element("span", "", set.label), element("small", "", `n = ${set.n.toLocaleString()}`));
      headRow.append(cell);
    });
    headRow.append(element("th", "preference-delta-head", "AI minus human"));
    const head = element("thead");
    head.append(headRow);

    const body = element("tbody");
    rows.forEach((row) => {
      const tr = element("tr");
      tr.append(element("th", "preference-row-label", row.label));
      row.shares.forEach((share) => {
        tr.append(element("td", "", `${(share * 100).toFixed(1)}%`));
      });
      const deltaCell = element("td", "preference-delta");
      const points = row.delta * 100;
      const bar = element("span", `preference-bar ${points >= 0 ? "is-up" : "is-down"}`);
      bar.style.width = `${Math.max(2, (Math.abs(row.delta) / widest) * 104)}px`;
      const value = element("span", "preference-delta-value", `${points >= 0 ? "+" : ""}${points.toFixed(1)}`);
      deltaCell.append(bar, value);
      tr.append(deltaCell);
      body.append(tr);
    });
    table.append(head, body);
    container.append(table);
  }

  // Bars carry direction through colour, so the label shows size without a signed zero.
  function magnitude(value) {
    return Math.abs(value).toFixed(2);
  }

  // Push labels apart just enough to stay legible when two values nearly coincide.
  function spreadLabels(positions, minimumGap) {
    const order = positions.map((y, index) => ({ y, index })).sort((a, b) => a.y - b.y);
    for (let i = 1; i < order.length; i += 1) {
      const gap = order[i].y - order[i - 1].y;
      if (gap < minimumGap) {
        const shift = (minimumGap - gap) / 2;
        order[i - 1].y -= shift;
        order[i].y += shift;
      }
    }
    const result = [];
    order.forEach((entry) => {
      result[entry.index] = entry.y;
    });
    return result;
  }

  function renderAxisWeights() {
    const host = document.querySelector("#axis-weights");
    const weights = data.judgmentBasis && data.judgmentBasis.axisWeights;
    if (!host || !weights) return;

    const width = 840;
    const height = 470;
    const labelEdge = 300;
    const leftColumn = 400;
    const rightColumn = 700;
    const top = 96;
    const bottom = 410;
    const ceiling = 0.85;
    const y = (value) => bottom - (Math.max(0, value) / ceiling) * (bottom - top);
    // The two axes the section is about; the rest stay grey so the crossing reads.
    const highlighted = { ml_field_impact: "is-ai", novelty: "is-human" };

    const svg = svgElement("svg", {
      class: "axis-slope",
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label":
        "Each panel axis plotted twice: how much it drove the AI's ordering, and how much it drove the human reviewer score.",
    });

    svg.append(
      svgElement(
        "text",
        { class: "evidence-axis-title", x: 24, y: 30 },
        "How the AI's six axis scores relate to the AI's overall score and to the reviewer score",
      ),
      svgElement("text", { class: "axis-slope-head", x: leftColumn, y: 64, "text-anchor": "middle" }, "AI's overall score"),
      svgElement("text", { class: "axis-slope-head", x: rightColumn, y: 64, "text-anchor": "middle" }, "Mean reviewer score"),
    );

    [0, 0.2, 0.4, 0.6, 0.8].forEach((value) => {
      svg.append(
        svgElement("line", { class: value === 0 ? "evidence-reference" : "evidence-grid", x1: leftColumn - 34, x2: rightColumn + 34, y1: y(value), y2: y(value) }),
        svgElement("text", { class: "evidence-tick", x: rightColumn + 92, y: y(value) + 4 }, value.toFixed(1)),
      );
    });
    svg.append(
      svgElement("text", { class: "evidence-row-sub", x: width - 8, y: y(0.85) - 6, "text-anchor": "end" }, "rank correlation"),
    );

    const labelY = spreadLabels(weights.axes.map((axis) => y(axis.ai) + 5), 19);
    weights.axes.forEach((axis, index) => {
      const tone = highlighted[axis.key] || "is-other";
      const group = svgElement("g", { class: `axis-slope-row ${tone}` });
      group.append(svgElement("title", {}, `${axis.label}: ${axis.ai.toFixed(3)} against the AI's ordering, ${axis.human.toFixed(3)} against the reviewer score`));
      group.append(
        svgElement("line", { class: "axis-slope-line", x1: leftColumn, x2: rightColumn, y1: y(axis.ai), y2: y(axis.human) }),
        // Leader from the de-collided label back to the dot it belongs to.
        svgElement("path", { class: "axis-slope-leader", d: `M ${labelEdge + 8} ${labelY[index] - 4} H ${leftColumn - 26} L ${leftColumn - 10} ${y(axis.ai)}` }),
        svgElement("circle", { class: "axis-slope-dot", cx: leftColumn, cy: y(axis.ai), r: tone === "is-other" ? 4.5 : 6.5 }),
        svgElement("circle", { class: "axis-slope-dot", cx: rightColumn, cy: y(axis.human), r: tone === "is-other" ? 4.5 : 6.5 }),
        svgElement("text", { class: "axis-slope-name", x: labelEdge, y: labelY[index], "text-anchor": "end" }, axis.label),
      );
      if (tone !== "is-other") {
        group.append(
          svgElement("line", { class: "axis-slope-interval", x1: rightColumn, x2: rightColumn, y1: y(axis.humanInterval[0]), y2: y(axis.humanInterval[1]) }),
          svgElement("text", { class: "axis-slope-endvalue", x: rightColumn + 18, y: y(axis.human) + 5 }, axis.human.toFixed(2)),
        );
      }
      svg.append(group);
    });

    const impact = weights.axes.find((axis) => axis.key === "ml_field_impact");
    const novelty = weights.axes.find((axis) => axis.key === "novelty");
    svg.append(
      // Above the dot, so the descending line does not strike through the text.
      svgElement("text", { class: "axis-slope-callout is-ai", x: leftColumn + 14, y: y(impact.ai) - 12 }, `${impact.ai.toFixed(2)}, the AI's top criterion`),
      svgElement("text", { class: "axis-slope-callout is-human", x: leftColumn + 14, y: y(novelty.ai) - 12 }, `${novelty.ai.toFixed(2)}, its fifth`),
      svgElement(
        "text",
        { class: "evidence-axis-caption", x: 24, y: height - 34 },
        `Computed over the ${weights.finalists} main-track finalists scored by the three frontier models.`,
      ),
      svgElement(
        "text",
        { class: "evidence-axis-caption", x: 24, y: height - 14 },
        "Predicted field impact is first on the models' side and last on the reviewers' side. Novelty is fifth and first.",
      ),
    );
    host.append(svg);
  }

  function renderReasonAxes() {
    const host = document.querySelector("#reason-axes");
    const reasons = data.judgmentBasis && data.judgmentBasis.reasonAxes;
    if (!host || !reasons) return;

    const rows = [...reasons.axes].sort((a, b) => Math.abs(b.ai) - Math.abs(a.ai));
    const width = 840;
    const rowHeight = 62;
    const top = 106;
    const height = top + rows.length * rowHeight + 76;
    const left = 336;
    const right = 744;
    const ceiling = 0.36;
    const length = (value) => (Math.abs(value) / ceiling) * (right - left);

    const svg = svgElement("svg", {
      class: "reason-axes",
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": "Each recurring line of argument, plotted against how much it moves the AI ranking and the reviewer score.",
    });
    svg.append(
      svgElement("text", { class: "evidence-axis-title", x: 24, y: 30 }, "How each recurring argument relates to the AI ranking and to the reviewer score"),
      svgElement("rect", { class: "reason-key-swatch is-ai", x: 336, y: 50, width: 26, height: 11 }),
      svgElement("text", { class: "reason-key-label", x: 370, y: 60 }, "the AI's ranking of all 6,341 papers"),
      svgElement("rect", { class: "reason-key-swatch is-human", x: 336, y: 72, width: 26, height: 11 }),
      svgElement("text", { class: "reason-key-label", x: 370, y: 82 }, "the human reviewer score"),
    );

    [0, 0.1, 0.2, 0.3].forEach((value) => {
      const x = left + length(value);
      svg.append(
        svgElement("line", { class: value === 0 ? "evidence-reference" : "evidence-grid", x1: x, x2: x, y1: top - 12, y2: top + rows.length * rowHeight - 20 }),
        svgElement("text", { class: "evidence-tick", x, y: top + rows.length * rowHeight + 2, "text-anchor": "middle" }, value.toFixed(1)),
      );
    });

    rows.forEach((row, index) => {
      const y = top + index * rowHeight;
      const group = svgElement("g", { class: `reason-row ${row.kind === "reward" ? "is-reward" : "is-concern"}` });
      group.append(svgElement("title", {}, `"${row.label}": ${row.ai.toFixed(3)} against the AI ranking, ${row.human.toFixed(3)} against the reviewer score`));
      group.append(
        svgElement("text", { class: "reason-row-label", x: left - 22, y: y + 4, "text-anchor": "end" }, row.label),
        svgElement("text", { class: "reason-row-kind", x: left - 22, y: y + 22, "text-anchor": "end" }, row.kind === "reward" ? "raised in a paper's favor" : "raised against a paper"),
        svgElement("rect", { class: "reason-bar is-ai", x: left, y: y - 9, width: Math.max(2, length(row.ai)), height: 14 }),
        svgElement("text", { class: "reason-value is-ai", x: left + length(row.ai) + 12, y: y + 3 }, magnitude(row.ai)),
        svgElement("rect", { class: "reason-bar is-human", x: left, y: y + 9, width: Math.max(2, length(row.human)), height: 14 }),
        svgElement("text", { class: "reason-value is-human", x: left + length(row.human) + 12, y: y + 21 }, magnitude(row.human)),
      );
      svg.append(group);
    });

    svg.append(
      svgElement("text", { class: "evidence-axis-caption", x: 24, y: height - 38 }, "Rank correlation between the number of models (0 to 4) that made the argument about a paper and each outcome."),
      svgElement("text", { class: "evidence-axis-caption", x: 24, y: height - 18 }, "Colored bars: teal for arguments in a paper's favor, red for arguments against it. Grey bars: the reviewer score."),
    );
    host.append(svg);
  }

  function renderWithinBatch() {
    const host = document.querySelector("#within-batch");
    const block = data.judgmentBasis && data.judgmentBasis.withinBatch;
    if (!host || !block) return;

    const rows = [
      { key: "models", label: "Another company's model", stat: block.modelToModel, tone: "is-model" },
      { key: "humans", label: "The human reviewers", stat: block.modelToHuman, tone: "is-human" },
    ];
    const width = 840;
    const height = 300;
    const left = 330;
    const right = 720;
    const ceiling = 0.5;
    const x = (value) => left + (Math.max(0, value) / ceiling) * (right - left);

    const svg = svgElement("svg", {
      class: "within-batch",
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": "Agreement measured inside a single batch of eight papers that every model read together.",
    });
    svg.append(
      svgElement("text", { class: "evidence-axis-title", x: 24, y: 30 }, "Agreement within a single batch of eight papers"),
      svgElement("text", { class: "evidence-axis-caption", x: 24, y: 54 }, `Averaged over ${block.batches.toLocaleString()} batches. Every model saw the same batches, so the context is identical on both sides.`),
    );

    [0, 0.1, 0.2, 0.3, 0.4, 0.5].forEach((value) => {
      svg.append(
        svgElement("line", { class: value === 0 ? "evidence-reference" : "evidence-grid", x1: x(value), x2: x(value), y1: 96, y2: 224 }),
        svgElement("text", { class: "evidence-tick", x: x(value), y: 246, "text-anchor": "middle" }, value.toFixed(1)),
      );
    });

    rows.forEach((row, index) => {
      const y = 130 + index * 62;
      const group = svgElement("g", { class: `within-row ${row.tone}` });
      group.append(svgElement("title", {}, `${row.label}: rank agreement ${row.stat.mean.toFixed(3)} over ${row.stat.comparisons.toLocaleString()} comparisons`));
      group.append(
        svgElement("text", { class: "within-row-label", x: left - 22, y: y + 1, "text-anchor": "end" }, row.label),
        svgElement("text", { class: "within-row-sub", x: left - 22, y: y + 19, "text-anchor": "end" }, `${row.stat.comparisons.toLocaleString()} comparisons`),
        svgElement("rect", { class: "evidence-track", x: left, y: y - 15, width: right - left, height: 28 }),
        svgElement("rect", { class: "within-bar", x: left, y: y - 15, width: Math.max(3, x(row.stat.mean) - left), height: 28 }),
        svgElement("text", { class: "within-value", x: x(row.stat.mean) + 14, y: y + 6 }, row.stat.mean.toFixed(2)),
      );
      svg.append(group);
    });

    svg.append(
      svgElement("text", { class: "evidence-axis-caption", x: 24, y: height - 14 }, "Rank correlation computed within each batch, then averaged across batches."),
    );
    host.append(svg);
  }

  function renderModelAgreement() {
    const host = document.querySelector("#model-agreement");
    const agreement = data.judgmentBasis && data.judgmentBasis.modelAgreement;
    if (!host || !agreement) return;

    const lookup = new Map();
    agreement.pairs.forEach((pair) => {
      lookup.set(`${pair.a}|${pair.b}`, pair.rho);
      lookup.set(`${pair.b}|${pair.a}`, pair.rho);
    });
    agreement.human.forEach((row) => {
      lookup.set(`${row.model}|Human reviewers`, row.rho);
      lookup.set(`Human reviewers|${row.model}`, row.rho);
    });
    const names = [...agreement.models, "Human reviewers"];

    // The first judge never labels a row, because the lower triangle leaves it empty.
    const rows = names.slice(1);
    const columns = names.slice(0, -1);
    const cell = 104;
    const left = 300;
    const top = 92;
    const width = 840;
    const height = top + rows.length * cell + 92;

    const svg = svgElement("svg", {
      class: "agreement-matrix",
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": "Rank correlation between every pair of judges, including the human reviewers.",
    });
    svg.append(
      svgElement("text", { class: "evidence-axis-title", x: 24, y: 30 }, "Rank correlation between each pair of judges"),
      svgElement("text", { class: "evidence-axis-caption", x: 24, y: 54 }, `Rank correlation over the same ${agreement.papers.toLocaleString()} main-track papers.`),
    );

    rows.forEach((rowName, rowIndex) => {
      const y = top + rowIndex * cell;
      const isHumanRow = rowName === "Human reviewers";
      svg.append(
        svgElement(
          "text",
          { class: `agreement-name ${isHumanRow ? "is-human" : ""}`, x: left - 18, y: y + cell / 2 + 5, "text-anchor": "end" },
          rowName,
        ),
      );
      columns.forEach((columnName, columnIndex) => {
        if (columnIndex > rowIndex) return;
        const value = lookup.get(`${rowName}|${columnName}`);
        if (value === undefined) return;
        const x = left + columnIndex * cell;
        const tone = isHumanRow || columnName === "Human reviewers" ? "is-human" : "is-model";
        const group = svgElement("g", { class: `agreement-cell ${tone}` });
        group.append(svgElement("title", {}, `${rowName} and ${columnName}: rank correlation ${value.toFixed(3)}`));
        group.append(
          svgElement("rect", { x: x + 4, y: y + 4, width: cell - 8, height: cell - 8, opacity: (0.16 + 0.84 * Math.min(1, value / 0.72)).toFixed(3) }),
          svgElement("text", { class: "agreement-value", x: x + cell / 2, y: y + cell / 2 + 7, "text-anchor": "middle" }, value.toFixed(2)),
        );
        svg.append(group);
      });
    });

    columns.forEach((name, index) => {
      const x = left + index * cell + cell / 2;
      svg.append(
        svgElement("text", { class: "agreement-column", x, y: top + rows.length * cell + 26, "text-anchor": "middle" }, name.split(" ")[0]),
      );
    });

    svg.append(
      // A value, not an inequality: the floor is 0.5695, so "0.57 or above" would be false.
      svgElement("text", { class: "evidence-axis-caption", x: 24, y: height - 34 }, `The weakest pair of models reaches ${agreement.weakestModelPair.toFixed(2)}.`),
      svgElement("text", { class: "evidence-axis-caption", x: 24, y: height - 14 }, `The closest any model comes to the reviewers is ${agreement.strongestHumanPair.toFixed(2)}.`),
    );
    host.append(svg);
  }

  function renderDivergenceTable() {
    const container = document.querySelector("#divergence-table");
    const cases = data.preferenceEvidence && data.preferenceEvidence.divergenceCases;
    if (!container || !cases) return;
    const total = cases.corpusSize.toLocaleString();

    const table = element("table", "divergence-table");
    const colgroup = element("colgroup");
    ["auto", "88px", "96px"].forEach((width) => {
      const col = document.createElement("col");
      col.style.width = width;
      colgroup.append(col);
    });
    const head = element("thead");
    const headRow = element("tr");
    headRow.append(
      element("th", "", "Paper, and what the AI said about it"),
      element("th", "", "ICML"),
      element("th", "", `AI rank of ${total}`),
    );
    head.append(headRow);

    const body = element("tbody");
    [
      ["gems", `ICML posters the AI ranked near the top (${cases.gemPool} in total)`, "why the AI did not rank it lower"],
      ["blindSpots", `ICML orals and spotlights the AI ranked near the bottom (${cases.blindSpotPool} in total)`, "why the AI did not rank it higher"],
    ].forEach(([key, caption, prompt]) => {
      const groupRow = element("tr", "divergence-group");
      const groupCell = element("th", "", caption);
      groupCell.colSpan = 3;
      groupRow.append(groupCell);
      body.append(groupRow);

      cases[key].forEach((row) => {
        const tr = element("tr");
        const paper = element("th", "divergence-paper");
        paper.append(element("span", "divergence-title", row.title));
        if (row.statedReason) {
          paper.append(element("span", "divergence-reason-cue", prompt));
          paper.append(element("q", "divergence-reason", row.statedReason));
        }
        tr.append(paper);
        const outcome = element("td", `divergence-tier tier-${row.tier}`);
        outcome.append(element("span", "", humanTierLabel(row.tier).replace("ICML ", "")));
        if (row.reviewerScore !== null) {
          outcome.append(element("small", "divergence-score", row.reviewerScore.toFixed(2)));
        }
        tr.append(outcome);
        const rank = element("td", `divergence-rank ${key === "gems" ? "is-top" : "is-bottom"}`);
        rank.append(element("strong", "", row.corpusRank.toLocaleString()));
        tr.append(rank);
        body.append(tr);
      });
    });
    table.append(colgroup, head, body);
    container.append(table);
  }

  function renderMosaic() {
    const container = document.querySelector("#paper-mosaic");
    data.papers.slice(0, 5).forEach((paper) => {
      const figure = element("figure", "paper-preview");
      const link = paperPanel(paper, "paper-preview-link", `AI rank ${paper.rank}`);
      const image = document.createElement("img");
      image.src = `assets/paper-${paper.paperId}.jpg`;
      image.alt = `Official first page of ${paper.title}`;
      image.loading = paper.rank <= 3 ? "eager" : "lazy";
      const caption = element("figcaption");
      caption.append(
        element("span", "preview-rank", `AI #${paper.rank}`),
        element("span", "preview-title", paper.title),
      );
      link.append(image, caption);
      figure.append(link);
      container.append(figure);
    });
  }

  function renderTopTen() {
    const container = document.querySelector("#top-ten");
    data.papers.slice(0, 10).forEach((paper) => {
      const row = paperPanel(paper, "rank-row paper-panel", `AI rank ${paper.rank}`);
      const rank = element("div", "rank-number", String(paper.rank).padStart(2, "0"));
      const content = element("div", "rank-content");
      const title = element("div", "rank-title", paper.title);
      const meta = element("div", "rank-meta");
      meta.append(
        element("span", `class-tag class-${paper.primaryClass}`, classLabel(paper.primaryClass)),
        element("span", `human-tag human-${paper.human.tier}`, humanTierLabel(paper.human.tier)),
      );
      if (paper.human.reviewerMean !== null) {
        meta.append(element("span", "review-score", `Human reviewer score (mean) ${paper.human.reviewerMean.toFixed(2)} / 6`));
      }
      meta.append(paperLinkCue(paper));
      content.append(title, element("p", "rank-summary", paper.whyRankedHere), meta);
      row.append(rank, content);
      container.append(row);
    });
  }

  function renderReviewerScale() {
    const host = document.querySelector("#reviewer-scale");
    const caption = document.querySelector("#reviewer-scale-caption");
    const scale = data.reviewerScale;
    if (!host || !caption) return;
    if (!scale) {
      host.closest(".score-key").hidden = true;
      return;
    }
    const coverage = data.results && data.results.main_track_full_coverage;
    const mean = coverage && coverage.reviewer_overall_mean;
    caption.append(
      element("strong", "", "Reading the reviewer scores. "),
      document.createTextNode(
        `ICML 2026 reviewers gave each paper an overall score from 1 to 6. The bars show how often ` +
          `each score was used in the ${scale.reviews.toLocaleString()} reviews of the ` +
          `${scale.papers.toLocaleString()} accepted main-track papers` +
          (mean ? `, where the average paper's mean score was ${mean.toFixed(2)}.` : "."),
      ),
    );

    const widest = Math.max(...scale.levels.map((level) => level.share));
    const table = element("table");
    const head = element("thead");
    const headRow = element("tr");
    headRow.append(
      element("th", "", "Score"),
      element("th", "", "Official label"),
      element("th", "", "Share of reviews"),
    );
    head.append(headRow);
    const body = element("tbody");
    // Highest score first, as on the official review form.
    [...scale.levels].reverse().forEach((level) => {
      const row = element("tr");
      row.title = `${level.reviews.toLocaleString()} reviews scored ${level.score} (${level.label})`;
      const bar = element("span", "score-key-bar");
      bar.style.width = `${(level.share / widest) * 100}%`;
      const track = element("span", "score-key-track");
      track.append(bar);
      const shareCell = element("td", "score-key-share");
      shareCell.append(track, element("span", "score-key-value", `${(level.share * 100).toFixed(1)}%`));
      row.append(
        element("th", "", String(level.score)),
        element("td", "score-key-label", level.label),
        shareCell,
      );
      body.append(row);
    });
    table.append(head, body);
    host.append(table);
  }

  function awardCard(paper) {
    const card = paperPanel(paper, "outstanding-card paper-panel", `Outstanding Paper, AI rank ${paper.rank}`);
    const imageWrap = element("div", "award-paper-image");
    const image = document.createElement("img");
    image.src = `assets/paper-${paper.paperId}.jpg`;
    image.alt = `Official first page of ${paper.title}`;
    imageWrap.append(image);

    const copy = element("div", "award-paper-copy");
    copy.append(
      element("div", "award-rank", `AI rank ${paper.rank}`),
      element("div", "award-title", paper.title),
      element("p", "award-why", paper.whyRankedHere),
    );
    const scores = element("div", "score-strip");
    [
      ["Technical", paper.scores.technical],
      ["ML impact", paper.scores.mlImpact],
      ["Science", paper.scores.scienceImpact],
      ["Evidence", paper.scores.evidence],
    ].forEach(([label, value]) => {
      const item = element("div", "score-item");
      item.append(element("strong", "", value.toFixed(1)), element("span", "", label));
      scores.append(item);
    });
    const posthoc = element("div", "posthoc-line");
    posthoc.append(
      element("span", "", "ICML outcome, shown post hoc"),
      element("strong", "", humanTierLabel(paper.human.tier).replace("ICML ", "")),
    );
    if (paper.human.reviewerMean !== null) {
      posthoc.append(element("span", "", `Human reviewer score (mean) ${paper.human.reviewerMean.toFixed(2)} / 6`));
    }
    posthoc.append(paperLinkCue(paper));
    copy.append(scores, posthoc);
    card.append(imageWrap, copy);
    return card;
  }

  function renderAwards() {
    const outstanding = document.querySelector("#outstanding-papers");
    data.papers.slice(0, 2).forEach((paper) => outstanding.append(awardCard(paper)));

    const mentions = document.querySelector("#honorable-mentions");
    data.papers.slice(2, 7).forEach((paper) => {
      const row = paperPanel(paper, "mention-row paper-panel", `Honorable Mention, AI rank ${paper.rank}`);
      const contribution = element("div", "mention-class", classLabel(paper.primaryClass));
      contribution.append(paperLinkCue(paper));
      row.append(
        element("div", "mention-rank", `#${paper.rank}`),
        element("div", "mention-title", paper.title),
        contribution,
        element("p", "mention-why", paper.whyRankedHere),
      );
      mentions.append(row);
    });
  }

  function oralRow(paper) {
    const row = paperPanel(paper, "oral-row paper-panel", `AI oral, rank ${paper.rank}`);
    const rank = element("div", "oral-rank", String(paper.rank).padStart(2, "0"));
    const main = element("div", "oral-main");
    const title = element("div", "oral-title", paper.title);
    const meta = element("div", "oral-meta");
    meta.append(
      element("span", `class-tag class-${paper.primaryClass}`, classLabel(paper.primaryClass)),
      element("span", `human-tag human-${paper.human.tier}`, humanTierLabel(paper.human.tier)),
      paperLinkCue(paper),
    );
    main.append(title, element("p", "oral-summary", paper.whyRankedHere), meta);
    const impact = element("div", "oral-impact");
    impact.append(
      element("strong", "", paper.scores.mlImpact.toFixed(1)),
      element("span", "", "ML impact"),
    );
    row.append(rank, main, impact);
    return row;
  }

  function renderOrals() {
    const container = document.querySelector("#oral-list");
    const search = document.querySelector("#oral-search").value.trim().toLowerCase();
    const selectedClass = document.querySelector("#oral-class-filter").value;
    const filtered = data.papers.filter((paper) => {
      const matchesText =
        !search ||
        paper.title.toLowerCase().includes(search) ||
        paper.paperId.toLowerCase().includes(search);
      return matchesText && (!selectedClass || paper.primaryClass === selectedClass);
    });
    container.replaceChildren(...filtered.map(oralRow));
    document.querySelector("#oral-result-count").textContent =
      `${filtered.length} of ${data.papers.length} papers`;
  }

  function populateFilters() {
    const select = document.querySelector("#oral-class-filter");
    const classes = [...new Set(data.papers.map((paper) => paper.primaryClass))].sort((a, b) =>
      classLabel(a).localeCompare(classLabel(b)),
    );
    classes.forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = classLabel(value);
      select.append(option);
    });
    select.addEventListener("change", renderOrals);
    document.querySelector("#oral-search").addEventListener("input", renderOrals);
  }

  function showView(view, updateHash = true) {
    const target = view === "awards" ? "awards" : "story";
    document.querySelectorAll("[data-view-panel]").forEach((panel) => {
      const active = panel.dataset.viewPanel === target;
      panel.hidden = !active;
      panel.classList.toggle("is-active", active);
    });
    document.querySelectorAll("[data-view]").forEach((tab) => {
      const active = tab.dataset.view === target;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
    });
    if (updateHash) history.replaceState(null, "", `#${target}`);
    window.scrollTo({ top: 0, behavior: "instant" });
  }

  document.querySelectorAll("[data-view]").forEach((tab) => {
    tab.addEventListener("click", () => showView(tab.dataset.view));
  });
  document.querySelectorAll("[data-go-awards]").forEach((button) => {
    button.addEventListener("click", () => showView("awards"));
  });
  document.querySelectorAll("[data-go-story]").forEach((button) => {
    button.addEventListener("click", () => showView("story"));
  });
  window.addEventListener("hashchange", () => showView(location.hash.slice(1), false));

  renderMosaic();
  renderMethodDiagram();
  renderTopTen();
  renderReviewerScale();
  renderCorrelationScale();
  renderRecallByTier();
  renderReviewerVsRank();
  renderPreferenceComparison();
  renderAxisWeights();
  renderReasonAxes();
  renderModelAgreement();
  renderWithinBatch();
  renderDivergenceTable();
  renderAwards();
  populateFilters();
  renderOrals();
  showView(location.hash.slice(1), false);
})();
