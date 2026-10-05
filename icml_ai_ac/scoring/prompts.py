from __future__ import annotations

from dataclasses import dataclass

from icml_ai_ac.models import PaperRecord
from icml_ai_ac.scoring.schema import CONTRIBUTION_CLASSES, schema_for_prompt


PROMPT_VERSION = "pass1_executive_ac_v5"


@dataclass(slots=True)
class PromptBundle:
    prompt_version: str
    system: str
    user: str

    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]


def build_pass1_prompt(
    record: PaperRecord,
    paper_text: str,
    *,
    prompt_version: str = PROMPT_VERSION,
    text_source: str = "scoring",
) -> PromptBundle:
    if prompt_version != PROMPT_VERSION:
        raise ValueError(f"Unsupported prompt version: {prompt_version}")
    system = """You are conducting a counterfactual executive Area Chair evaluation for a machine learning conference.

Evaluate only the paper content provided by the user. Do not use web search, citation memory, author identity, institutional prestige, venue reputation, or outside knowledge. If author names, affiliations, or other identity cues appear in the text, ignore them.

Your goal is not to imitate a full peer-review process. Your goal is to make a paper-only executive judgment about technical quality, likely future importance for machine learning, and likely broader scientific importance. Separate conventional reviewer strength from sweeping long-run impact.

Return only valid JSON matching the requested schema. Use calibrated 1-10 scores:
Score relative to accepted ICML-style papers, not relative to all submissions:
1-2 = among the weakest accepted papers or seriously flawed; 3-4 = below-average accepted paper; 5 = median accepted paper; 6 = above-average accepted paper; 7 = top-quartile accepted paper; 8 = roughly top 10% accepted paper; 9 = roughly top 1-2% accepted paper; 10 = once-in-years transformative paper. Use 9-10 rarely. If every accepted-looking paper receives 7-8, the ranking is not useful.
"""
    user = f"""Evaluate this paper from the paper-only representation below.

Representation source: {text_source}
The preferred source is `scoring`: main paper body extracted from the PDF, with references removed and appendix/supplement removed when detected. If the text is truncated, treat missing later details as uncertainty rather than factual absence.

Scoring instructions:
- Be evidence-grounded. Tie major judgments to claims, methods, experiments, theory, or limitations visible in the paper text.
- Distinguish technical soundness from importance. A technically solid paper may have low field impact; a risky paper may have high impact if its core idea could reshape an area.
- Distinguish conventional reviewer judgment from executive-AC priority. Conventional judgment should reflect likely ICML-style reviewer criteria; executive priority should emphasize long-run contributions to ML and science.
- Penalize observed unclear claims, observed missing evidence, weak baselines, irreproducibility, or overclaiming. Do not invent missing ablations, missing baselines, or missing proofs.
- Do not reward popularity of topics by itself.
- Classify the paper's contribution route before judging impact. Benchmarks, datasets, infrastructure, safety/evals, scientific tools, theory, and algorithms can all be high-impact, but they should be compared partly within their route.
- Separate observed weaknesses from unverified risks. If the representation omits needed details, lower confidence and state what is not visible. Do not assert that the paper lacks ablations, baselines, proofs, limitations, or implementation details unless the provided text supports that absence.
- Make executive_ac_priority the primary first-pass ranking score for deciding which papers go to the strong OpenAI model. It should prioritize broad scientific impact and sweeping ML-field importance over incremental reviewer polish.
- A paper should advance to the strong-model ranking pass only if it has a plausible top-slice case on importance, not merely because it is competent.
- Use the full 1-10 scale. A score of 8 should mean unusually strong among accepted ICML papers, not merely competent. If you are uncertain, prefer 5-6 rather than defaulting to 7-8.
- Include a calibration judgment that estimates this paper's percentile among accepted ICML-style papers. 50 means median accepted paper; 75 means top quartile; 90 means top 10%; 98 means top 2%.
- Provide an adoption-path judgment: who would plausibly use this in two years, and what concrete practice or scientific workflow would change?
- Apply a benchmark-locality penalty: benchmark wins alone do not imply broad impact unless the benchmark/resource is likely to become a durable community standard.

Required JSON schema:
{schema_for_prompt()}

Additional field guidance:
- summary.main_claims: array of 2-5 concise claims.
- contribution_profile.contribution_types: array using any of: theory, algorithm, empirical, benchmark, dataset, systems, application, analysis, position, safety, scientific_modeling, other.
- contribution_profile.primary_contribution_class: one of {", ".join(CONTRIBUTION_CLASSES)}.
- contribution_profile.impact_route: short explanation of how this contribution would create value if it mattered.
- scores: numeric 1-10 values for all required fields.
- impact_axes: numeric 1-10 values estimating what kind of importance this paper has, even if overall score is moderate.
- reviewer_lens.human_review_alignment: short note on whether AI’s judgment would likely agree or disagree with conventional reviewers and why.
- executive_lens.sweeping_impact_scenario: best plausible path by which this work could become highly important.
- evidence.key_positive_evidence and evidence.key_negative_evidence: arrays of short evidence-grounded bullets. Paraphrase; do not quote long passages.
- calibration.triage_bucket: one of bottom_half_accepted, above_average, top_quartile, top_10_percent, top_2_percent, transformative_candidate.
- calibration.why_not_higher and calibration.why_not_lower: force score calibration.
- visibility_limits.not_visible_in_provided_repr: list important missing details that require full-paper review; do not convert these into factual paper flaws.
- ranking_signals.should_advance_to_strong_model: boolean; true only if this paper is worth a stronger-model ranking pass.
- evidence_audit.observed_weaknesses: weaknesses visible in the provided text.
- evidence_audit.not_visible_or_unverified_risks: risks caused by missing or truncated information.
- evidence_audit.unsupported_inferences_to_avoid: claims a reader should not make from this representation alone.
- uncertainty.confidence: numeric 1-10 confidence in this evaluation from the provided text alone.

Paper metadata:
paper_id: {record.paper_id}
title: {record.title or ""}

Paper text:
<<<PAPER_TEXT
{sanitize_for_prompt(paper_text)}
PAPER_TEXT
"""
    return PromptBundle(prompt_version=prompt_version, system=system, user=user)


def sanitize_for_prompt(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.strip().lower().startswith("authors:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()
